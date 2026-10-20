import pandas as pd
import pytest

from jev_bench.data.serialize import serialize

ROW = pd.Series(
    {
        "age": 39,
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "native_country": "unknown",
    }
)


def test_raw_lists_every_feature():
    text = serialize(ROW, "raw")
    assert text.startswith("age=39, capital_gain=2174")
    assert "native_country=unknown" in text


def test_natural_humanizes_values():
    text = serialize(ROW, "natural")
    assert "39-year-old male" in text
    assert "Adm clerical" in text
    assert "native country: not reported" in text
    assert "$2,174" in text


def test_contextual_text_matches_natural():
    assert serialize(ROW, "contextual") == serialize(ROW, "natural")


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        serialize(ROW, "json")
