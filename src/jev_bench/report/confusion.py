"""AG News confusion matrices: where Jev's extra errors over DistilBERT come from.

Reads seed-0 test predictions. Outputs:
  reports/figures/agnews-confusion.png    side-by-side matrices for the report
  results/report/test/confusion_agnews.json  the matrices and the per-cell error gap
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from jev_bench.config import REPORT_DIR, ROOT  # noqa: E402
from jev_bench.report.build import _preds, _probs  # noqa: E402

TASK = "agnews"
RUNS = {"Jev (agnews_v2)": "jev-agnews_v2", "DistilBERT": "distilbert"}
CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")
FIGURE_PATH = ROOT / "reports" / "figures" / "agnews-confusion.png"

# One-hue sequential ramp (light -> dark blue) for error counts; diagonal is neutral.
ERROR_RAMP = ("#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
DIAGONAL_FILL = "#ebeae6"
INK, MUTED_INK, GRID = "#1f1f1d", "#6b6a66", "#ffffff"
# Above this share of the colour scale, cell text switches to white.
DARK_CELL_THRESHOLD = 0.55


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, n_labels: int) -> np.ndarray:
    """Counts with rows = true label, columns = predicted label."""
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)
    np.add.at(cm, (np.asarray(labels), np.asarray(preds)), 1)
    return cm


def error_gap_by_cell(worse: np.ndarray, better: np.ndarray) -> dict:
    """Split the difference in total errors across off-diagonal cells.

    Cells where ``worse`` makes fewer errors than ``better`` get negative shares,
    so shares always sum to 1.
    """
    off = ~np.eye(worse.shape[0], dtype=bool)
    diff = np.where(off, worse - better, 0)
    total = int(diff.sum())
    cells = {(int(i), int(j)): {"extra_errors": int(diff[i, j]), "share": diff[i, j] / total} for i, j in zip(*np.nonzero(diff))}
    return {"total": total, "cells": cells}


def _matrix(run: str) -> np.ndarray:
    preds = _preds(f"test/{TASK}/{run}__seed0.parquet")
    return confusion_matrix(preds["label"].to_numpy(), _probs(preds).argmax(1), len(CLASS_NAMES))


def _draw(ax, cm: np.ndarray, title: str, vmax: int, cmap) -> None:
    k = len(CLASS_NAMES)
    errors = np.where(np.eye(k, dtype=bool), np.nan, cm)
    ax.imshow(np.where(np.eye(k, dtype=bool), 0, 1), cmap=LinearSegmentedColormap.from_list("d", [DIAGONAL_FILL, DIAGONAL_FILL]))
    ax.imshow(errors, cmap=cmap, vmin=0, vmax=vmax)
    for i in range(k):
        for j in range(k):
            dark = i != j and cm[i, j] / vmax > DARK_CELL_THRESHOLD
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=10,
                    color="white" if dark else (MUTED_INK if i == j else INK), fontweight="normal" if i == j else "bold")
    ax.set_xticks(range(k), CLASS_NAMES, fontsize=9, color=INK)
    ax.set_yticks(range(k), CLASS_NAMES, fontsize=9, color=INK)
    ax.set_xticks(np.arange(-0.5, k), minor=True)
    ax.set_yticks(np.arange(-0.5, k), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("predicted", fontsize=9, color=MUTED_INK)
    ax.set_ylabel("true label", fontsize=9, color=MUTED_INK)
    ax.set_title(f"{title}: {cm.trace() / cm.sum():.1%} correct", fontsize=11, color=INK, loc="left", pad=28)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot(matrices: dict[str, np.ndarray], path: Path = FIGURE_PATH) -> None:
    vmax = int(max((np.where(np.eye(len(CLASS_NAMES), dtype=bool), 0, cm)).max() for cm in matrices.values()))
    cmap = LinearSegmentedColormap.from_list("errors", ERROR_RAMP)
    fig, axes = plt.subplots(1, len(matrices), figsize=(10, 4.4))
    for ax, (title, cm) in zip(axes, matrices.items()):
        _draw(ax, cm, title, vmax, cmap)
    fig.suptitle("Jev's largest AG News error: Business stories filed as Sci/Tech", x=0.02, ha="left", fontsize=13, color=INK)
    fig.text(0.02, 0.89, "Test split (n = 7,600), seed 0. Off-diagonal cells are errors, shaded on one shared scale; grey diagonal = correct.",
             fontsize=9, color=MUTED_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def build_confusion() -> tuple[Path, Path]:
    matrices = {title: _matrix(run) for title, run in RUNS.items()}
    jev, local = matrices.values()
    gap = error_gap_by_cell(jev, local)
    plot(matrices)
    out = {
        "task": TASK,
        "split": "test",
        "seed": 0,
        "classes": list(CLASS_NAMES),
        "matrices": {run: matrices[title].tolist() for title, run in RUNS.items()},
        "extra_jev_errors": gap["total"],
        "gap_by_cell": [
            {"true": CLASS_NAMES[i], "predicted": CLASS_NAMES[j], **c}
            for (i, j), c in sorted(gap["cells"].items(), key=lambda kv: -kv[1]["extra_errors"])
        ],
    }
    json_path = REPORT_DIR / "test" / "confusion_agnews.json"
    json_path.write_text(json.dumps(out, indent=2) + "\n")
    return FIGURE_PATH, json_path
