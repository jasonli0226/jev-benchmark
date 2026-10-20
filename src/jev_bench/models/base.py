"""Common model interface. Every model returns probabilities over the task's
label tuple, in that order, so the harness scores all models the same way."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from jev_bench.config import Task


class Model(ABC):
    name: str
    # "text" models read df["text"]; "tabular" models read the Adult feature
    # columns; "api" models build their own requests from the frame.
    input_kind: str = "text"
    # Hardware used for cost amortization: "gpu", "cpu" or "api".
    device: str = "cpu"
    trainable: bool = True

    def __init__(self, task: Task, seed: int):
        self.task = task
        self.seed = seed

    @abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None: ...

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray: ...

    def run_info(self) -> dict:
        """Extra per-run facts (e.g. chosen hyperparameters, API usage)."""
        return {}
