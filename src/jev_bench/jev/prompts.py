"""Versioned Jev prompt configs: building requests and parsing answers.

A prompt config (configs/prompts/*.yaml) defines one question about one
item. An optional ``examples`` block adds labeled few-shot rows, drawn from
the train split only, to every request's state. With ``batch_size > 1`` several items share a request: they go under
``state.items`` and each gets its own question. Every config must be
``frozen: true`` before it may run on the test split.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from jev_bench.config import JEV_MODEL, PROMPTS_DIR, Task
from jev_bench.data.serialize import VARIANTS

QUESTION_TYPES = ("choice", "noul")
EXAMPLE_BALANCE = ("natural", "balanced")


@dataclass(frozen=True)
class ExampleSpec:
    n: int
    seed: int
    balance: str  # "natural" keeps the train label rate; "balanced" equal per label


@dataclass(frozen=True)
class PromptConfig:
    version: str
    task: str
    frozen: bool
    question_type: str
    instructions: str  # contains "{target}"
    criteria: dict
    noun: str
    batch_size: int
    model: str
    serialization: str | None  # Track B only
    background: str | None  # extra framing put in state, or None
    examples: ExampleSpec | None
    sha256: str

    @property
    def target_single(self) -> str:
        return f"the {self.noun}"


def load_prompt(version: str) -> PromptConfig:
    path = PROMPTS_DIR / f"{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No prompt config {path}")
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw)
    if cfg.get("version") != version:
        raise ValueError(f"{path.name}: version field {cfg.get('version')!r} must match file name")
    q = cfg["question"]
    if q["type"] not in QUESTION_TYPES:
        raise ValueError(f"{path.name}: question type must be one of {QUESTION_TYPES}")
    if "{target}" not in q["instructions"]:
        raise ValueError(f"{path.name}: instructions must contain '{{target}}'")
    serialization = cfg.get("serialization")
    if serialization is not None and serialization not in VARIANTS:
        raise ValueError(f"{path.name}: serialization must be one of {VARIANTS}")
    examples = None
    if "examples" in cfg:
        ex = cfg["examples"]
        examples = ExampleSpec(n=int(ex["n"]), seed=int(ex.get("seed", 0)), balance=ex.get("balance", "natural"))
        if examples.n < 1 or examples.balance not in EXAMPLE_BALANCE:
            raise ValueError(f"{path.name}: examples need n >= 1 and balance in {EXAMPLE_BALANCE}")
    return PromptConfig(
        version=version,
        task=cfg["task"],
        frozen=bool(cfg.get("frozen", False)),
        question_type=q["type"],
        instructions=q["instructions"],
        criteria=q["criteria"],
        noun=cfg.get("noun", "item"),
        batch_size=int(cfg.get("batch_size", 1)),
        model=cfg.get("model", JEV_MODEL),
        serialization=serialization,
        background=cfg.get("background"),
        examples=examples,
        sha256=hashlib.sha256(raw).hexdigest()[:12],
    )


def check_against_task(cfg: PromptConfig, task: Task) -> None:
    if cfg.task != task.name:
        raise ValueError(f"Prompt {cfg.version} is for task {cfg.task!r}, not {task.name!r}")
    if cfg.question_type == "choice":
        if list(cfg.criteria) != list(task.labels):
            raise ValueError(f"Choice criteria keys {list(cfg.criteria)} must equal task labels {list(task.labels)}")
    elif len(task.labels) != 2 or set(cfg.criteria) != {"true", "false"}:
        raise ValueError("Noul prompts need a binary task and criteria keys 'true'/'false'")
    if task.track == "B" and cfg.serialization is None:
        raise ValueError(f"Track B prompt {cfg.version} must set serialization")


def _question(cfg: PromptConfig, target: str) -> dict:
    return {
        "type": cfg.question_type,
        "instructions": cfg.instructions.format(target=target),
        "criteria": dict(cfg.criteria),
    }


def build_requests(
    cfg: PromptConfig, texts: list[str], examples: list[dict] | None = None
) -> list[tuple[dict, list[str]]]:
    """Return (request body, question keys) per request; keys map answers back to items.

    ``examples`` (from ``select_examples``) is shared by every request.
    """
    out = []
    for start in range(0, len(texts), cfg.batch_size):
        chunk = texts[start : start + cfg.batch_size]
        state: dict = {}
        if cfg.background:
            state["background"] = cfg.background.strip()
        if examples:
            state["labeled_examples"] = examples
        if cfg.batch_size == 1:
            state[cfg.noun] = chunk[0]
            keys = ["answer"]
            questions = {"answer": _question(cfg, cfg.target_single)}
        else:
            keys = [f"item_{i:02d}" for i in range(len(chunk))]
            state["items"] = dict(zip(keys, chunk))
            questions = {k: _question(cfg, f"the {cfg.noun} under items.{k}") for k in keys}
        out.append(({"model": cfg.model, "state": state, "questions": questions}, keys))
    return out


def parse_answer(cfg: PromptConfig, task: Task, answer: dict | None) -> np.ndarray | None:
    """Map one typed answer to probabilities over task.labels; None if unusable."""
    if not isinstance(answer, dict) or answer.get("type") != cfg.question_type:
        return None
    if cfg.question_type == "noul":
        p = answer.get("noul")
        if not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
            return None
        return np.array([1.0 - p, float(p)])
    probs = answer.get("probabilities")
    if not isinstance(probs, dict):
        return None
    vec = np.array([float(probs.get(label, 0.0)) for label in task.labels])
    total = vec.sum()
    if not np.isfinite(total) or total <= 0 or np.any(vec < 0):
        return None
    return vec / total


def select_examples(cfg: PromptConfig, task: Task, train_texts: list[str], train_labels: list[int]) -> list[dict]:
    """Pick the few-shot rows deterministically. Callers must pass train-split
    data only: examples drawn from val or test would leak labels."""
    spec = cfg.examples
    if spec is None:
        return []
    rng = np.random.default_rng(spec.seed)
    labels = np.asarray(train_labels)
    if spec.balance == "balanced":
        per_label = -(-spec.n // len(task.labels))
        picks = np.concatenate(
            [rng.choice(np.flatnonzero(labels == k), size=per_label, replace=False) for k in range(len(task.labels))]
        )[: spec.n]
    else:
        picks = rng.choice(len(labels), size=spec.n, replace=False)
    picks = rng.permutation(picks)
    return [{cfg.noun: train_texts[i], "label": task.labels[labels[i]]} for i in picks]
