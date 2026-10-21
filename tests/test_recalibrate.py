import numpy as np
import pytest

from jev_bench.eval.metrics import expected_calibration_error, validate_probs
from jev_bench.report.recalibrate import fit_calibrator


def _overconfident_binary(n=400, seed=0):
    """Always 0.95 confident in class 1, but right only 75% of the time."""
    rng = np.random.default_rng(seed)
    probs = np.tile([0.05, 0.95], (n, 1))
    labels = (rng.random(n) < 0.75).astype(np.int64)
    return probs, labels


def _onehot_multiclass(n=400, k=4, seed=0):
    """Hard 0/1 outputs (like Jev on AG News), right 80% of the time."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    preds = np.where(rng.random(n) < 0.8, labels, (labels + 1) % k)
    return np.eye(k)[preds], labels


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_binary_calibration_removes_overconfidence(method):
    probs, labels = _overconfident_binary()
    calibrate = fit_calibrator(method, probs, labels)
    assert expected_calibration_error(probs, labels) > 0.15
    assert expected_calibration_error(calibrate(probs), labels) < 0.03


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_multiclass_calibration_softens_hard_one_hot_outputs(method):
    probs, labels = _onehot_multiclass()
    out = fit_calibrator(method, probs, labels)(probs)
    validate_probs(out, 4)
    assert out.max(axis=1).mean() == pytest.approx(0.8, abs=0.03)


def test_multiclass_platt_keeps_the_predicted_label():
    probs, labels = _onehot_multiclass()
    soft = probs * 0.7 + 0.075  # rows still sum to 1, argmax unchanged
    out = fit_calibrator("platt", soft, labels)(soft)
    assert np.array_equal(out.argmax(axis=1), soft.argmax(axis=1))


def test_unknown_method_is_rejected():
    probs, labels = _overconfident_binary()
    with pytest.raises(ValueError, match="method"):
        fit_calibrator("beta", probs, labels)
