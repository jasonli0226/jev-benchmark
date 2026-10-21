import numpy as np
import pytest

from jev_bench.config import get_task
from jev_bench.jev.prompts import build_requests, check_against_task, load_prompt, parse_answer, select_examples


@pytest.mark.parametrize(
    "version,task",
    [("sst2_v1", "sst2"), ("agnews_v1", "agnews"), ("adult_raw_v1", "adult"), ("adult_natural_v1", "adult"), ("adult_contextual_v1", "adult"), ("adult_natural_fewshot_v1", "adult"), ("adult_contextual_fewshot_v1", "adult"), ("adult_natural_baserate_v1", "adult"), ("adult_natural_correlates_v1", "adult")],
)
def test_shipped_prompts_are_valid(version, task):
    check_against_task(load_prompt(version), get_task(task))


def test_single_item_request_shape():
    cfg = load_prompt("sst2_v1")
    [(body, keys)] = build_requests(cfg, ["great film"])
    assert keys == ["answer"]
    assert body["state"] == {"review": "great film"}
    q = body["questions"]["answer"]
    assert q["type"] == "choice" and "the review" in q["instructions"]
    assert list(q["criteria"]) == ["negative", "positive"]


def test_contextual_prompt_puts_background_in_state():
    cfg = load_prompt("adult_contextual_v1")
    [(body, _)] = build_requests(cfg, ["A 39-year-old ..."])
    assert "24%" in body["state"]["background"]
    assert body["state"]["person"] == "A 39-year-old ..."


def test_decomposition_prompts_split_the_contextual_background():
    full, baserate, correlates = (load_prompt(v) for v in ("adult_contextual_v1", "adult_natural_baserate_v1", "adult_natural_correlates_v1"))
    assert "24%" in baserate.background and "education" not in baserate.background
    assert "education" in correlates.background and "24%" not in correlates.background
    # Same question and record text as the contextual prompt ("natural" renders the
    # same text as "contextual"), so only the background differs.
    for half in (baserate, correlates):
        assert half.serialization == "natural"
        assert (half.instructions, half.criteria) == (full.instructions, full.criteria)


def test_batched_requests_key_each_item():
    from dataclasses import replace

    cfg = replace(load_prompt("sst2_v1"), batch_size=2)
    reqs = build_requests(cfg, ["a", "b", "c"])
    assert [keys for _, keys in reqs] == [["item_00", "item_01"], ["item_00"]]
    assert reqs[0][0]["state"]["items"] == {"item_00": "a", "item_01": "b"}
    assert "items.item_01" in reqs[0][0]["questions"]["item_01"]["instructions"]


def test_parse_choice_orders_by_task_labels_and_normalizes():
    cfg, task = load_prompt("agnews_v1"), get_task("agnews")
    answer = {"type": "choice", "choice": "sports", "probabilities": {"sports": 0.6, "world": 0.2, "business": 0.2}}
    np.testing.assert_allclose(parse_answer(cfg, task, answer), [0.2, 0.6, 0.2, 0.0])


def test_parse_noul_maps_to_binary_probs():
    cfg, task = load_prompt("adult_raw_v1"), get_task("adult")
    np.testing.assert_allclose(parse_answer(cfg, task, {"type": "noul", "noul": 0.7}), [0.3, 0.7])


@pytest.mark.parametrize("answer", [None, {}, {"type": "noul", "noul": 1.5}, {"type": "choice"}])
def test_parse_rejects_bad_answers(answer):
    assert parse_answer(load_prompt("adult_raw_v1"), get_task("adult"), answer) is None


def test_fewshot_examples_come_from_given_rows_and_are_deterministic():
    cfg, task = load_prompt("adult_natural_fewshot_v1"), get_task("adult")
    texts = [f"row {i}" for i in range(100)]
    labels = [1 if i < 25 else 0 for i in range(100)]
    a = select_examples(cfg, task, texts, labels)
    assert len(a) == 20 and a == select_examples(cfg, task, texts, labels)
    assert all(ex["person"] in texts and ex["label"] in task.labels for ex in a)
    [(body, _)] = build_requests(cfg, ["target row"], a)
    assert body["state"]["labeled_examples"] == a and body["state"]["person"] == "target row"


def test_balanced_examples_split_evenly():
    from dataclasses import replace

    from jev_bench.jev.prompts import ExampleSpec

    cfg = replace(load_prompt("adult_natural_fewshot_v1"), examples=ExampleSpec(n=10, seed=1, balance="balanced"))
    labels = [1] * 10 + [0] * 90
    ex = select_examples(cfg, get_task("adult"), [str(i) for i in range(100)], labels)
    assert sum(e["label"] == ">50K" for e in ex) == 5
