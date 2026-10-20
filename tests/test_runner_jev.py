import json

import pytest

from jev_bench import runner
from jev_bench.jev import client as client_mod
from jev_bench.jev.client import JevResponse


@pytest.fixture
def fake_jev(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(client_mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runner, "PREDS_DIR", tmp_path / "preds")

    def decide_many(self, bodies, repeat=0):
        out = []
        for body in bodies:
            answers = {
                key: {"type": "choice", "choice": "positive", "probabilities": {"negative": 0.3, "positive": 0.7}}
                for key in body["questions"]
            }
            answers.pop("item_01", None)  # one malformed batch answer, to exercise failure counting
            out.append(JevResponse({"answers": answers, "usage": {"input_tokens": 100, "cost": 4.2e-6}, "model": "typesafe/jev-1.14-x"}, 0.2, 1, False))
        return out

    monkeypatch.setattr(client_mod.JevClient, "decide_many", decide_many)
    return tmp_path


def test_jev_val_run_writes_result(fake_jev):
    result = runner.run("jev", "sst2", split="val", prompt_version="sst2_v1", limit=20)
    assert result["n"] == 20
    assert result["info"]["requests"] == 20 and result["info"]["parse_failures"] == 0
    assert result["cost"]["usd_per_1k_predictions"] == pytest.approx(4.2e-3)
    assert result["latency"]["single_item"]["p50_ms"] == pytest.approx(200.0)
    saved = json.loads((fake_jev / "runs/val/sst2/jev-sst2_v1__seed0__n20.json").read_text())
    assert saved["prompt_version"] == "sst2_v1"
    assert (fake_jev / "preds/val/sst2/jev-sst2_v1__seed0__n20.parquet").exists()


def test_unfrozen_prompt_cannot_touch_test(fake_jev):
    with pytest.raises(PermissionError, match="not frozen"):
        runner.run("jev", "sst2", split="test", prompt_version="sst2_v1")


def test_missing_answers_are_counted_not_dropped(fake_jev, monkeypatch):
    from dataclasses import replace

    from jev_bench.jev import prompts
    from jev_bench.models import jev_model

    original = prompts.load_prompt
    monkeypatch.setattr(jev_model, "load_prompt", lambda v: replace(original(v), batch_size=2))
    result = runner.run("jev", "sst2", split="val", prompt_version="sst2_v1", limit=20)
    assert result["n"] == 20
    assert result["info"]["parse_failures"] == 10
    assert result["info"]["parse_failure_rate"] == pytest.approx(0.5)
