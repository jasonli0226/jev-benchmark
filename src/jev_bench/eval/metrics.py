"""Scoring on a shared interface: ``probs`` is (n, k) over the task's labels."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from jev_bench.config import ECE_BINS


def validate_probs(probs: np.ndarray, n_labels: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != n_labels:
        raise ValueError(f"probs must have shape (n, {n_labels}); got {probs.shape}")
    if np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("probs rows must be non-negative and sum to 1")
    return probs


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Top-label ECE with equal-width confidence bins."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS) -> list[dict]:
    """Per-bin accuracy vs confidence, for reliability diagrams."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            out.append(
                {
                    "lo": float(edges[b]),
                    "hi": float(edges[b + 1]),
                    "count": int(mask.sum()),
                    "confidence": float(confidence[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    return out


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def score(probs: np.ndarray, labels: np.ndarray, n_labels: int) -> dict:
    probs = validate_probs(probs, n_labels)
    labels = np.asarray(labels, dtype=np.int64)
    preds = probs.argmax(axis=1)
    clipped = np.clip(probs, 1e-7, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "ece": expected_calibration_error(probs, labels),
        "brier": brier(probs, labels),
        "nll": float(log_loss(labels, clipped, labels=list(range(n_labels)))),
    }
    if n_labels == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs[:, 1]))
    return metrics


def latency_stats(seconds: list[float]) -> dict:
    ms = np.asarray(seconds, dtype=np.float64) * 1000.0
    if ms.size == 0:
        return {"n": 0}
    return {
        "n": int(ms.size),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
    }
