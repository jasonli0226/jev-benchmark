"""Aggregate run JSONs + predictions into summary tables and figures.

Outputs (results/report/<split>/):
  summary.md     tables for the write-up
  summary.json   the same numbers, machine-readable
  figures/       reliability diagrams, accuracy-vs-latency, accuracy-vs-cost
"""

import json
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jev_bench.config import PREDS_DIR, REPORT_DIR, RUNS_DIR, TASKS  # noqa: E402
from jev_bench.eval.metrics import reliability_bins  # noqa: E402

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0
METRIC_COLS = ("accuracy", "macro_f1", "ece", "brier", "nll", "roc_auc")


def load_runs(split: str) -> pd.DataFrame:
    rows = []
    for path in sorted((RUNS_DIR / split).rglob("*.json")):
        r = json.loads(path.read_text())
        if r.get("limit"):  # val subsamples are iteration runs, not results
            continue
        rows.append(
            {
                "task": r["task"],
                "track": r["track"],
                "run": r["run"],
                "model": r["model"],
                "seed": r["seed"],
                "n": r["n"],
                "harness_version": r["harness_version"],
                "prompt_version": r["prompt_version"],
                "serialization": r["serialization"],
                **{k: r["metrics"].get(k) for k in METRIC_COLS},
                "p50_ms": r["latency"]["single_item"].get("p50_ms"),
                "p95_ms": r["latency"]["single_item"].get("p95_ms"),
                "throughput_ms": r["latency"]["batch_throughput_ms_per_item"],
                "usd_per_1k": r["cost"]["usd_per_1k_predictions"],
                "train_usd": r["cost"]["train_usd"],
                "fit_s": r["timing"]["fit_s"],
                "parse_failure_rate": r["info"].get("parse_failure_rate"),
                "pred_path": str(path.relative_to(RUNS_DIR).with_suffix(".parquet")),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No results under {RUNS_DIR / split}")
    df = pd.DataFrame(rows)
    versions = sorted(df["harness_version"].unique())
    if len(versions) > 1:
        counts = df.groupby("harness_version").size().to_dict()
        raise ValueError(
            f"Results span harness versions {counts}. Results from different harness versions are "
            "never compared: re-run the stale ones (`suite --rerun`) or delete them."
        )
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    stats = {
        **{f"{m}_mean": (m, "mean") for m in METRIC_COLS},
        **{f"{m}_std": (m, "std") for m in ("accuracy", "macro_f1", "ece")},
        "seeds": ("seed", "nunique"),
        "n": ("n", "first"),
        "p50_ms": ("p50_ms", "median"),
        "p95_ms": ("p95_ms", "median"),
        "throughput_ms": ("throughput_ms", "median"),
        "usd_per_1k": ("usd_per_1k", "mean"),
        "train_usd": ("train_usd", "mean"),
        "fit_s": ("fit_s", "mean"),
        "parse_failure_rate": ("parse_failure_rate", "max"),
    }
    agg = df.groupby(["task", "run"]).agg(**stats).reset_index()
    return agg.sort_values(["task", "accuracy_mean"], ascending=[True, False])


def _preds(rel: str) -> pd.DataFrame:
    return pd.read_parquet(PREDS_DIR / rel).sort_values("id").reset_index(drop=True)


def _probs(preds: pd.DataFrame) -> np.ndarray:
    return preds[[c for c in preds.columns if c.startswith("p_")]].to_numpy()


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """95% CI of accuracy(a) - accuracy(b) on the same test items."""
    if not np.array_equal(a["id"].to_numpy(), b["id"].to_numpy()):
        raise ValueError("Paired bootstrap needs predictions on identical items")
    labels = a["label"].to_numpy()
    diff = (_probs(a).argmax(1) == labels).astype(float) - (_probs(b).argmax(1) == labels).astype(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP_SAMPLES, len(diff)))
    boots = diff[idx].mean(axis=1)
    return {"delta": float(diff.mean()), "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5))}


def jev_vs_best_local(df: pd.DataFrame, agg: pd.DataFrame) -> list[dict]:
    out = []
    for task, group in agg.groupby("task"):
        local = group[~group["run"].str.startswith("jev-")]
        jev = group[group["run"].str.startswith("jev-")]
        if local.empty or jev.empty:
            continue
        best_local = local.iloc[0]["run"]
        for jev_run in jev["run"]:
            a = df[(df.task == task) & (df.run == jev_run) & (df.seed == 0)]
            b = df[(df.task == task) & (df.run == best_local) & (df.seed == 0)]
            if a.empty or b.empty:
                continue
            out.append({"task": task, "jev_run": jev_run, "vs": best_local, **paired_bootstrap(_preds(a.iloc[0].pred_path), _preds(b.iloc[0].pred_path))})
    return out


def jev_repeat_stability(df: pd.DataFrame) -> list[dict]:
    """Jev is non-deterministic; quantify how much repeats disagree."""
    out = []
    for (task, run), group in df[df.model == "jev"].groupby(["task", "run"]):
        preds = [_preds(p) for p in group.sort_values("seed").pred_path]
        if len(preds) < 2:
            continue
        agree, mad = [], []
        for x, y in combinations(preds, 2):
            px, py = _probs(x), _probs(y)
            agree.append(float((px.argmax(1) == py.argmax(1)).mean()))
            mad.append(float(np.abs(px - py).max(axis=1).mean()))
        out.append({"task": task, "run": run, "repeats": len(preds), "label_agreement": float(np.mean(agree)), "mean_abs_prob_diff": float(np.mean(mad))})
    return out


def serialization_table(agg: pd.DataFrame) -> list[dict]:
    adult = agg[agg.task == "adult"]
    if adult.empty:
        return []
    ref = adult[adult.run == "tabular_lgbm"]
    ref_acc = float(ref.accuracy_mean.iloc[0]) if not ref.empty else None
    rows = []
    for _, r in adult.iterrows():
        rows.append(
            {
                "run": r.run,
                "accuracy": r.accuracy_mean,
                "roc_auc": r.roc_auc_mean,
                "ece": r.ece_mean,
                "delta_vs_tabular_lgbm": (r.accuracy_mean - ref_acc) if ref_acc is not None else None,
            }
        )
    return rows


def plot_reliability(df: pd.DataFrame, task: str, out_dir) -> str:
    runs = df[(df.task == task) & (df.seed == 0)]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="perfect calibration")
    for _, r in runs.sort_values("run").iterrows():
        preds = _preds(r.pred_path)
        bins = reliability_bins(_probs(preds), preds["label"].to_numpy())
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o", markersize=3, label=r.run)
    lo = 1.0 / len(TASKS[task].labels)
    ax.set(xlim=(lo, 1), ylim=(0, 1), xlabel="confidence (top-label)", ylabel="accuracy", title=f"Reliability: {task}")
    ax.legend(fontsize=7)
    path = out_dir / f"reliability_{task}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def plot_tradeoff(agg: pd.DataFrame, task: str, x: str, xlabel: str, out_dir) -> str:
    sub = agg[agg.task == task]
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, r in sub.iterrows():
        ax.errorbar(r[x], r.accuracy_mean, yerr=r.accuracy_std if pd.notna(r.accuracy_std) else 0, fmt="o", capsize=3)
        ax.annotate(r.run, (r[x], r.accuracy_mean), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylabel="accuracy", title=f"{task}: accuracy vs {xlabel}")
    path = out_dir / f"{x}_{task}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _usd(value) -> str:
    return "-" if value is None or pd.isna(value) else f"{value:.2e}"


def _fmt(mean, std=None, digits=4) -> str:
    if mean is None or pd.isna(mean):
        return "-"
    return f"{mean:.{digits}f}" + (f" ± {std:.{digits}f}" if std is not None and pd.notna(std) else "")


def _markdown(agg, versus, stability, serial, figures, version) -> str:
    lines = [f"# Jev benchmark summary\n\nHarness version `{version}`. Mean ± std over seeds (Jev: repeats).\n"]
    for task, group in agg.groupby("task"):
        lines += [
            f"## {task} (track {TASKS[task].track}, n={int(group.n.iloc[0])})\n",
            "| run | seeds | accuracy | macro F1 | ECE | Brier | AUC | p50 ms | p95 ms | $/1k | train $ |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in group.iterrows():
            lines.append(
                f"| {r.run} | {r.seeds} | {_fmt(r.accuracy_mean, r.accuracy_std)} | {_fmt(r.macro_f1_mean, r.macro_f1_std)} "
                f"| {_fmt(r.ece_mean, r.ece_std)} | {_fmt(r.brier_mean)} | {_fmt(r.roc_auc_mean)} | {_fmt(r.p50_ms, digits=1)} "
                f"| {_fmt(r.p95_ms, digits=1)} | {_usd(r.usd_per_1k)} | {_usd(r.train_usd)} |"
            )
        lines += [""] + [f"![{f}](figures/{f})" for f in figures.get(task, [])] + [""]
    if versus:
        lines += ["## Jev vs best local model (paired bootstrap, seed 0)\n", "| task | Jev run | vs | Δ accuracy | 95% CI |", "|---|---|---|---|---|"]
        lines += [f"| {v['task']} | {v['jev_run']} | {v['vs']} | {v['delta']:+.4f} | [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] |" for v in versus]
        lines.append("")
    if serial:
        lines += ["## Track B: cost of serialization\n", "| run | accuracy | AUC | ECE | Δ acc vs tabular_lgbm |", "|---|---|---|---|---|"]
        lines += [f"| {s['run']} | {_fmt(s['accuracy'])} | {_fmt(s['roc_auc'])} | {_fmt(s['ece'])} | {_fmt(s['delta_vs_tabular_lgbm'])} |" for s in serial]
        lines.append("")
    if stability:
        lines += ["## Jev repeat stability\n", "| task | run | repeats | label agreement | mean max |Δp| |", "|---|---|---|---|---|"]
        lines += [f"| {s['task']} | {s['run']} | {s['repeats']} | {s['label_agreement']:.4f} | {s['mean_abs_prob_diff']:.4f} |" for s in stability]
        lines.append("")
    return "\n".join(lines)


def build_report(split: str = "test"):
    df = load_runs(split)
    agg = aggregate(df)
    out_dir = REPORT_DIR / split
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        task: [
            plot_reliability(df, task, fig_dir),
            plot_tradeoff(agg, task, "p50_ms", "p50 latency (ms, single item)", fig_dir),
            plot_tradeoff(agg, task, "usd_per_1k", "USD per 1k predictions", fig_dir),
        ]
        for task in sorted(df.task.unique())
    }
    versus = jev_vs_best_local(df, agg)
    stability = jev_repeat_stability(df)
    serial = serialization_table(agg)
    version = df.harness_version.iloc[0]
    summary = {
        "harness_version": version,
        "split": split,
        "runs": df.drop(columns=["pred_path"]).to_dict(orient="records"),
        "aggregate": agg.to_dict(orient="records"),
        "jev_vs_best_local": versus,
        "jev_repeat_stability": stability,
        "serialization": serial,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    path = out_dir / "summary.md"
    path.write_text(_markdown(agg, versus, stability, serial, figures, version))
    return path
