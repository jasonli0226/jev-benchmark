import numpy as np
import pytest

from jev_bench.eval.metrics import expected_calibration_error, latency_stats, score, validate_probs


def test_perfect_confident_predictions_score_perfectly():
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    m = score(probs, np.array([0, 1, 0]), 2)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["ece"] == pytest.approx(0.0)
    assert m["roc_auc"] == 1.0


def test_ece_measures_overconfidence():
    # 0.9 confidence but only half right -> ECE 0.4
    probs = np.array([[0.9, 0.1]] * 4)
    assert expected_calibration_error(probs, np.array([0, 0, 1, 1])) == pytest.approx(0.4)


def test_multiclass_has_no_auc():
    probs = np.full((4, 4), 0.25)
    assert "roc_auc" not in score(probs, np.array([0, 1, 2, 3]), 4)


@pytest.mark.parametrize("bad", [np.array([[0.5, 0.6]]), np.array([[1.0]]), np.array([[-0.1, 1.1]])])
def test_validate_probs_rejects_bad_rows(bad):
    with pytest.raises(ValueError):
        validate_probs(bad, 2)


def test_latency_stats_in_ms():
    s = latency_stats([0.1] * 10)
    assert s["p50_ms"] == pytest.approx(100.0)
    assert s["n"] == 10
