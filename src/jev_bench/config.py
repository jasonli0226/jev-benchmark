"""Paths, task definitions and cost constants shared across the harness."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
PREDS_DIR = RESULTS_DIR / "preds"
CACHE_DIR = RESULTS_DIR / "cache"
REPORT_DIR = RESULTS_DIR / "report"
PROMPTS_DIR = ROOT / "configs" / "prompts"

SPLIT_SEED = 20261021
SPLITS = ("train", "val", "test")

# Jev pricing on OpenRouter (USD per input token). Responses also report
# usage.cost, which is what the harness actually bills against.
JEV_MODEL = "typesafe/jev-1.14"
JEV_USD_PER_INPUT_TOKEN = 0.042 / 1_000_000

# Local compute amortization: rented-hardware equivalents, USD per hour.
GPU_USD_PER_HOUR = 0.40  # RTX 4090-class cloud instance
CPU_USD_PER_HOUR = 0.10  # 8-16 vCPU instance

# Per-item latency is measured on this many single-item calls.
LATENCY_SAMPLE_SIZE = 200

# Calibration binning.
ECE_BINS = 15


@dataclass(frozen=True)
class Task:
    name: str
    track: str  # "A" text, "B" structured
    labels: tuple[str, ...]


TASKS = {
    "sst2": Task("sst2", "A", ("negative", "positive")),
    "agnews": Task("agnews", "A", ("world", "sports", "business", "sci_tech")),
    "adult": Task("adult", "B", ("<=50K", ">50K")),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise ValueError(f"Unknown task {name!r}; expected one of {sorted(TASKS)}")
    return TASKS[name]
