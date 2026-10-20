"""Number-to-text serializations of Adult rows (Track B).

Each variant is a separate eval condition. ``raw`` is also the text input
for the text-native local models on Track B.
"""

import pandas as pd

from jev_bench.data.sources import ADULT_FEATURES, ADULT_MISSING

VARIANTS = ("raw", "natural", "contextual")


def _human(value: str) -> str:
    return "not reported" if value == ADULT_MISSING else value.replace("-", " ").replace("_", " ")


def raw(row: pd.Series) -> str:
    return ", ".join(f"{col}={row[col]}" for col in ADULT_FEATURES)


def natural(row: pd.Series) -> str:
    return (
        f"A {row.age}-year-old {_human(row.sex).lower()} ({_human(row.race)}), "
        f"native country: {_human(row.native_country)}. "
        f"Marital status: {_human(row.marital_status).lower()}; "
        f"household relationship: {_human(row.relationship).lower()}. "
        f"Highest education: {_human(row.education)}. "
        f"Works in the {_human(row.workclass)} sector as {_human(row.occupation)}, "
        f"{row.hours_per_week} hours per week. "
        f"Annual capital gains: ${row.capital_gain:,}; capital losses: ${row.capital_loss:,}."
    )


_RENDERERS = {"raw": raw, "natural": natural, "contextual": natural}


def serialize(row: pd.Series, variant: str) -> str:
    """Render one row. ``contextual`` shares the ``natural`` text; its extra
    framing lives in the versioned prompt config, not here."""
    if variant not in _RENDERERS:
        raise ValueError(f"Unknown serialization {variant!r}; expected one of {VARIANTS}")
    return _RENDERERS[variant](row)


def serialize_frame(df: pd.DataFrame, variant: str) -> pd.Series:
    return df.apply(lambda row: serialize(row, variant), axis=1)
