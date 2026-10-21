"""Post-hoc recalibration of Jev probabilities: fit on val, score on test.

Lives in report/ rather than eval/ so it does not change the harness version:
it reads cached predictions and never touches how runs are scored.

Methods:
  platt     binary: logistic regression on logit(p); multiclass: temperature
            scaling on log(p), the one-parameter multiclass form of Platt
  isotonic  isotonic regression on p (one-vs-rest + renormalize for multiclass)

Outputs (results/report/test/): recalibration.json, recalibration.md
"""

import json
import re
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from jev_bench.config import PREDS_DIR, REPORT_DIR
from jev_bench.eval.metrics import score

METHODS = ("platt", "isotonic")
REPORT_METRICS = ("accuracy", "ece", "brier", "nll")
# Jev emits exact 0.0 and 1.0; clip before taking logs.
PROB_EPS = 1e-6
# Search range for log(temperature).
LOG_TEMPERATURE_BOUNDS = (-5.0, 5.0)

Calibrator = Callable[[np.ndarray], np.ndarray]


def _log_probs(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs, PROB_EPS, 1.0))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_platt_binary(probs: np.ndarray, labels: np.ndarray) -> Calibrator:
    def logit(p):
        q = np.clip(p[:, 1], PROB_EPS, 1 - PROB_EPS)
        return np.log(q / (1 - q)).reshape(-1, 1)

    model = LogisticRegression(C=1e6).fit(logit(probs), labels)
    return lambda p: model.predict_proba(logit(p))


def _fit_temperature(probs: np.ndarray, labels: np.ndarray) -> Calibrator:
    logp = _log_probs(probs)
    rows = np.arange(len(labels))

    def nll(log_t):
        return -np.log(_softmax(logp / np.exp(log_t))[rows, labels] + 1e-12).mean()

    temperature = float(np.exp(minimize_scalar(nll, bounds=LOG_TEMPERATURE_BOUNDS, method="bounded").x))
    return lambda p: _softmax(_log_probs(p) / temperature)


def _fit_isotonic(probs: np.ndarray, labels: np.ndarray) -> Calibrator:
    k = probs.shape[1]
    classes = [1] if k == 2 else range(k)
    models = {c: IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(probs[:, c], labels == c) for c in classes}

    def apply(p):
        if k == 2:
            q = models[1].predict(p[:, 1])
            return np.column_stack([1 - q, q])
        out = np.column_stack([models[c].predict(p[:, c]) for c in classes])
        total = out.sum(axis=1, keepdims=True)
        return np.where(total > 0, out / np.where(total > 0, total, 1), 1.0 / k)

    return apply


def fit_calibrator(method: str, probs: np.ndarray, labels: np.ndarray) -> Calibrator:
    """Fit on (probs, labels); return a function mapping probs to recalibrated probs."""
    labels = np.asarray(labels, dtype=np.int64)
    if method == "platt":
        return _fit_platt_binary(probs, labels) if probs.shape[1] == 2 else _fit_temperature(probs, labels)
    if method == "isotonic":
        return _fit_isotonic(probs, labels)
    raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")


def _load(path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path).sort_values("id")
    return df[[c for c in df.columns if c.startswith("p_")]].to_numpy(), df["label"].to_numpy()


def _largest_val_file(task: str, prompt: str):
    """The biggest cached val subsample for this prompt (seed 0)."""
    pattern = re.compile(rf"^jev-{re.escape(prompt)}__seed0__n(\d+)\.parquet$")
    found = [(int(m.group(1)), p) for p in (PREDS_DIR / "val" / task).glob("*.parquet") if (m := pattern.match(p.name))]
    if not found:
        raise FileNotFoundError(f"No cached val predictions for {task}/{prompt}; run it on val first")
    return max(found)


def recalibrate(task: str, prompt: str) -> dict:
    n_val, val_path = _largest_val_file(task, prompt)
    val_probs, val_labels = _load(val_path)
    calibrators = {m: fit_calibrator(m, val_probs, val_labels) for m in METHODS}
    test_paths = sorted((PREDS_DIR / "test" / task).glob(f"jev-{prompt}__seed*.parquet"))
    if not test_paths:
        raise FileNotFoundError(f"No test predictions for {task}/{prompt}")
    per_method = {m: [] for m in ("raw", *METHODS)}
    for path in test_paths:
        probs, labels = _load(path)
        k = probs.shape[1]
        per_method["raw"].append(score(probs, labels, k))
        for m, calibrate in calibrators.items():
            per_method[m].append(score(calibrate(probs), labels, k))
    return {
        "task": task,
        "prompt": prompt,
        "n_val": n_val,
        "n_test": len(labels),
        "test_repeats": len(test_paths),
        **{m: {k: float(np.mean([s[k] for s in scores])) for k in REPORT_METRICS} for m, scores in per_method.items()},
    }


def _markdown(rows: list[dict]) -> str:
    lines = [
        "# Jev recalibration (fit on val, scored on test)\n",
        "Mean over test repeats. Platt = temperature scaling for multiclass.\n",
        "| task | prompt | n val | method | accuracy | ECE | Brier | NLL |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for m in ("raw", *METHODS):
            s = r[m]
            lines.append(
                f"| {r['task']} | {r['prompt']} | {r['n_val']} | {m} | {s['accuracy']:.4f} | {s['ece']:.4f} | {s['brier']:.4f} | {s['nll']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def build_recalibration(prompts: dict[str, list[str]]):
    rows = [recalibrate(task, p) for task in sorted(prompts) for p in prompts[task]]
    out_dir = REPORT_DIR / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recalibration.json").write_text(json.dumps(rows, indent=2) + "\n")
    path = out_dir / "recalibration.md"
    path.write_text(_markdown(rows))
    return path
