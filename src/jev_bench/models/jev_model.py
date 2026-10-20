"""Jev wrapped in the common Model interface."""

import numpy as np
import pandas as pd

from jev_bench.config import JEV_USD_PER_INPUT_TOKEN, Task
from jev_bench.data.serialize import serialize_frame
from jev_bench.data.splits import load_split
from jev_bench.jev.client import JevClient
from jev_bench.jev.prompts import build_requests, check_against_task, load_prompt, parse_answer, select_examples
from jev_bench.models.base import Model


class Jev(Model):
    name = "jev"
    input_kind = "api"
    device = "api"
    trainable = False

    def __init__(self, task: Task, seed: int, prompt_version: str, concurrency: int):
        super().__init__(task, seed)
        self.prompt = load_prompt(prompt_version)
        check_against_task(self.prompt, task)
        self.client = JevClient(concurrency=concurrency)
        self.calls: list[dict] = []
        self.failures = 0
        self.examples: list[dict] = []
        if self.prompt.examples:
            train = load_split(task.name, "train")
            self.examples = select_examples(self.prompt, task, self._texts(train), train["label"].tolist())

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Jev is not trained; prompts are tuned on val by hand, outside the harness."""

    def _texts(self, df: pd.DataFrame) -> list[str]:
        if self.task.track == "B":
            return serialize_frame(df, self.prompt.serialization).tolist()
        return df["text"].tolist()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        requests = build_requests(self.prompt, self._texts(df), self.examples)
        # seed doubles as the repeat index: same seed reuses cached answers.
        responses = self.client.decide_many([body for body, _ in requests], repeat=self.seed)
        uniform = np.full(len(self.task.labels), 1.0 / len(self.task.labels))
        rows = []
        for (_, keys), res in zip(requests, responses):
            answers = res.body.get("answers") or {}
            usage = res.body.get("usage") or {}
            tokens = int(usage.get("input_tokens") or 0)
            cost = usage.get("cost")
            self.calls.append(
                {
                    "items": len(keys),
                    "latency_s": res.latency_s,
                    "attempts": res.attempts,
                    "cached": res.cached,
                    "input_tokens": tokens,
                    "cost_usd": float(cost) if cost is not None else tokens * JEV_USD_PER_INPUT_TOKEN,
                    "cost_estimated": cost is None,
                    "served_model": res.body.get("model"),
                }
            )
            for key in keys:
                probs = parse_answer(self.prompt, self.task, answers.get(key))
                if probs is None:
                    self.failures += 1
                    probs = uniform
                rows.append(probs)
        return np.vstack(rows)

    def run_info(self) -> dict:
        n_items = sum(c["items"] for c in self.calls)
        return {
            "prompt_version": self.prompt.version,
            "prompt_sha256": self.prompt.sha256,
            "serialization": self.prompt.serialization,
            "batch_size": self.prompt.batch_size,
            "n_examples": len(self.examples),
            "requests": len(self.calls),
            "cached_requests": sum(c["cached"] for c in self.calls),
            "retried_requests": sum(c["attempts"] > 1 for c in self.calls),
            "input_tokens": sum(c["input_tokens"] for c in self.calls),
            "cost_usd": sum(c["cost_usd"] for c in self.calls),
            "cost_estimated_requests": sum(c["cost_estimated"] for c in self.calls),
            "parse_failures": self.failures,
            "parse_failure_rate": self.failures / n_items if n_items else 0.0,
            "served_models": sorted({c["served_model"] for c in self.calls if c["served_model"]}),
        }
