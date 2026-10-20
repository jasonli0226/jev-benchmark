"""Download raw datasets into normalized DataFrames.

Text tasks return columns ``text`` and ``label`` (int index into the task's
label tuple). Adult returns its feature columns plus ``label``.
"""

import pandas as pd

ADULT_NUMERIC = ["age", "capital_gain", "capital_loss", "hours_per_week"]
ADULT_CATEGORICAL = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
ADULT_FEATURES = ADULT_NUMERIC + ADULT_CATEGORICAL
ADULT_MISSING = "unknown"


def load_sst2() -> dict[str, pd.DataFrame]:
    """GLUE SST-2. Its public test split has no labels, so the GLUE
    validation split (872 rows) becomes our test set and val is carved
    out of train."""
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/sst2")
    frames = {}
    for src, dst in (("train", "pool"), ("validation", "test")):
        df = ds[src].to_pandas().rename(columns={"sentence": "text"})
        frames[dst] = df[["text", "label"]].assign(text=lambda d: d.text.str.strip())
    return frames


def load_agnews() -> dict[str, pd.DataFrame]:
    """AG News. Labels 0..3 = World, Sports, Business, Sci/Tech."""
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news")
    return {
        "pool": ds["train"].to_pandas()[["text", "label"]],
        "test": ds["test"].to_pandas()[["text", "label"]],
    }


def load_adult() -> dict[str, pd.DataFrame]:
    """UCI Adult (OpenML v2, train+test combined, 48,842 rows).

    ``fnlwgt`` (a census sampling weight, not a person attribute) and
    ``education-num`` (a recoding of ``education``) are dropped so every
    model, including Jev, sees the same features.
    """
    from sklearn.datasets import fetch_openml

    raw = fetch_openml("adult", version=2, as_frame=True).frame
    df = raw.rename(columns=lambda c: c.replace("-", "_"))
    df = df.rename(columns={"class": "income"})
    for col in ADULT_CATEGORICAL:
        df[col] = df[col].astype("string").fillna(ADULT_MISSING).str.strip()
    for col in ADULT_NUMERIC:
        df[col] = df[col].astype("int64")
    df["label"] = (df["income"].astype("string").str.strip() == ">50K").astype("int64")
    return {"pool": df[ADULT_FEATURES + ["label"]].reset_index(drop=True)}


LOADERS = {"sst2": load_sst2, "agnews": load_agnews, "adult": load_adult}
