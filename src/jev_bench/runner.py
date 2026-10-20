"""Run one (model, task, split, seed) evaluation and write its result JSON.

Integrity rules enforced here:
  * Jev prompts must be frozen before they touch the test split.
  * Splits are hash-checked on load (see data.splits.load_split).
  * Every result records the harness version; the report refuses to mix them.
"""

import json
import time
from datetime import datetime, timezone

import pandas as pd

from jev_bench.config import (
    CPU_USD_PER_HOUR,
    GPU_USD_PER_HOUR,
    LATENCY_SAMPLE_SIZE,
    PREDS_DIR,
    RUNS_DIR,
    get_task,
)
from jev_bench.data.serialize import serialize_frame
from jev_bench.data.splits import load_split, sample
from jev_bench.eval.metrics import latency_stats, score
from jev_bench.models.registry import TABULAR_MODELS, get_model_class
from jev_bench.version import harness_version

LATENCY_WARMUP_CALLS = 5
USD_PER_HOUR = {"gpu": GPU_USD_PER_HOUR, "cpu": CPU_USD_PER_HOUR}


def run_name(model: str, prompt_version: str | None, serialization: str | None) -> str:
    if model == "jev":
        return f"jev-{prompt_version}"
    return f"{model}-{serialization}" if serialization else model


def _with_text(df: pd.DataFrame, track: str, serialization: str | None) -> pd.DataFrame:
    return df.assign(text=serialize_frame(df, serialization)) if track == "B" else df


def _single_item_latency(model, df: pd.DataFrame) -> list[float]:
    """Time predict_proba one item at a time, as a live service would call it."""
    rows = df.sample(n=min(LATENCY_SAMPLE_SIZE, len(df)), random_state=0)
    for i in range(min(LATENCY_WARMUP_CALLS, len(rows))):
        model.predict_proba(rows.iloc[[i]])
    timings = []
    for i in range(len(rows)):
        start = time.perf_counter()
        model.predict_proba(rows.iloc[[i]])
        timings.append(time.perf_counter() - start)
    return timings


def _jev_cost_latency(info: dict, calls: list[dict], n: int) -> tuple[dict, dict]:
    per_item = [c["latency_s"] for c in calls for _ in range(c["items"])]
    latency = {"single_item": latency_stats([c["latency_s"] for c in calls]), "per_item_in_batch": latency_stats(per_item)}
    cost = {
        "train_usd": 0.0,
        "inference_usd": info["cost_usd"],
        "usd_per_1k_predictions": info["cost_usd"] / n * 1000,
        "spent_this_run_usd": sum(c["cost_usd"] for c in calls if not c["cached"]),
        "basis": "OpenRouter usage.cost",
    }
    return cost, latency


def run(
    model_name: str,
    task_name: str,
    split: str = "test",
    seed: int = 0,
    prompt_version: str | None = None,
    serialization: str | None = None,
    limit: int | None = None,
    concurrency: int = 8,
) -> dict:
    task = get_task(task_name)
    cls = get_model_class(model_name)
    if model_name == "jev":
        if prompt_version is None:
            raise ValueError("jev runs need --prompt")
        model = cls(task, seed, prompt_version=prompt_version, concurrency=concurrency)
        if split == "test" and not model.prompt.frozen:
            raise PermissionError(
                f"Prompt {prompt_version} is not frozen. Iterate on val, set `frozen: true`, then run test."
            )
        serialization = model.prompt.serialization
    else:
        model = cls(task, seed)
        if model.input_kind == "text" and task.track == "B":
            serialization = serialization or "raw"
        elif model.input_kind == "tabular" and task.track != "B":
            raise ValueError(f"{model_name} needs tabular features; task {task_name} is text")
        else:
            serialization = None

    eval_df = sample(load_split(task_name, split), limit)
    needs_text = model.input_kind == "text"
    if model.trainable:
        train, val = load_split(task_name, "train"), load_split(task_name, "val")
        if needs_text:
            train, val = _with_text(train, task.track, serialization), _with_text(val, task.track, serialization)
    else:
        train = val = None
    if needs_text:
        eval_df = _with_text(eval_df, task.track, serialization)

    start = time.perf_counter()
    model.fit(train, val)
    fit_s = time.perf_counter() - start

    start = time.perf_counter()
    probs = model.predict_proba(eval_df)
    predict_s = time.perf_counter() - start
    n = len(eval_df)

    labels = eval_df["label"].to_numpy()
    metrics = score(probs, labels, len(task.labels))
    info = model.run_info()

    if model.device == "api":
        cost, latency = _jev_cost_latency(info, model.calls, n)
    else:
        rate = USD_PER_HOUR[model.device]
        latency = {"single_item": latency_stats(_single_item_latency(model, eval_df))}
        cost = {
            "train_usd": fit_s * rate / 3600,
            "inference_usd": predict_s * rate / 3600,
            "usd_per_1k_predictions": predict_s / n * 1000 * rate / 3600,
            "basis": f"{model.device} at ${rate:.2f}/h, batched inference",
        }
    latency["batch_throughput_ms_per_item"] = predict_s / n * 1000

    name = run_name(model_name, prompt_version, serialization)
    result = {
        "run": name,
        "model": model_name,
        "task": task_name,
        "track": task.track,
        "split": split,
        "limit": limit,
        "n": n,
        "seed": seed,
        "harness_version": harness_version(),
        "prompt_version": prompt_version,
        "serialization": serialization,
        "metrics": metrics,
        "latency": latency,
        "cost": cost,
        "timing": {"fit_s": fit_s, "predict_s": predict_s},
        "info": info,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    stem = _stem(split, task_name, name, seed, limit)
    out = RUNS_DIR / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    preds = pd.DataFrame(probs, columns=[f"p_{label}" for label in task.labels])
    preds.insert(0, "label", labels)
    preds.insert(0, "id", eval_df["id"].to_numpy())
    pred_path = PREDS_DIR / f"{stem}.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(pred_path, index=False)
    return result


def _stem(split, task_name, name, seed, limit) -> str:
    return f"{split}/{task_name}/{name}__seed{seed}" + (f"__n{limit}" if limit else "")


def result_exists(model_name, task_name, split, seed, prompt_version=None, serialization=None, limit=None) -> bool:
    """True if a result from the current harness version is already on disk."""
    if model_name != "jev" and model_name not in TABULAR_MODELS and get_task(task_name).track == "B":
        serialization = serialization or "raw"
    name = run_name(model_name, prompt_version, serialization)
    path = RUNS_DIR / f"{_stem(split, task_name, name, seed, limit)}.json"
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("harness_version") == harness_version()

