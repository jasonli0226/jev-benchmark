"""Sentence-transformer embeddings + LightGBM."""

import numpy as np
import pandas as pd
import torch

from jev_bench.models.base import Model
from jev_bench.models.classical import fit_lgbm

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENCODE_BATCH_SIZE = 256


class EmbedLGBM(Model):
    name = "embed_lgbm"
    device = "gpu"

    def _encode(self, texts: pd.Series) -> np.ndarray:
        batch = ENCODE_BATCH_SIZE if len(texts) > 1 else 1
        return self.encoder.encode(
            texts.tolist(), batch_size=batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(EMBEDDING_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
        self.clf = fit_lgbm(
            self._encode(train["text"]),
            train["label"],
            self._encode(val["text"]),
            val["label"],
            self.seed,
            len(self.task.labels),
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._encode(df["text"]))

    def run_info(self) -> dict:
        return {"encoder": EMBEDDING_MODEL, "best_iteration": int(self.clf.best_iteration_ or 0)}
