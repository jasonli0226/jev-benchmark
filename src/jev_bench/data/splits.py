"""Create, freeze and load train/val/test splits with a hash manifest."""

import hashlib
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from jev_bench.config import DATA_DIR, SPLIT_SEED, SPLITS, get_task
from jev_bench.data.sources import LOADERS

MANIFEST_PATH = DATA_DIR / "manifest.json"

# Rows held out of the training pool. Adult also has no official test split
# in the combined OpenML frame, so its test set is carved out here.
VAL_SIZE = {"sst2": 2000, "agnews": 5000, "adult": 2000}
ADULT_TEST_SIZE = 5000


def frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def _stratified(df: pd.DataFrame, size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rest, held = train_test_split(df, test_size=size, stratify=df["label"], random_state=SPLIT_SEED)
    return rest, held


def build_splits(task_name: str) -> dict[str, pd.DataFrame]:
    raw = LOADERS[task_name]()
    pool = raw["pool"]
    if task_name == "adult":
        pool, test = _stratified(pool, ADULT_TEST_SIZE)
    else:
        test = raw["test"]
    train, val = _stratified(pool, VAL_SIZE[task_name])
    frames = {"train": train, "val": val, "test": test}
    return {
        name: df.reset_index(drop=True).rename_axis("row").reset_index().rename(columns={"row": "id"})
        for name, df in frames.items()
    }


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}


def freeze(task_name: str, overwrite: bool = False) -> dict:
    """Write splits to data/<task>/<split>.parquet and record them in the manifest."""
    get_task(task_name)
    manifest = _load_manifest()
    if task_name in manifest and not overwrite:
        raise FileExistsError(
            f"Splits for {task_name!r} are already frozen. Re-freezing invalidates every "
            "existing result; pass overwrite=True only if you mean that."
        )
    out_dir = DATA_DIR / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = {}
    for name, df in build_splits(task_name).items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        entry[name] = {
            "rows": len(df),
            "sha256": frame_hash(df),
            "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        }
    manifest[task_name] = entry
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return entry


def load_split(task_name: str, split: str) -> pd.DataFrame:
    """Load a frozen split, verifying its hash against the manifest."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    entry = _load_manifest().get(task_name)
    if entry is None:
        raise FileNotFoundError(f"No frozen splits for {task_name!r}. Run `jev-bench prepare --task {task_name}`.")
    df = pd.read_parquet(DATA_DIR / task_name / f"{split}.parquet")
    actual = frame_hash(df)
    if actual != entry[split]["sha256"]:
        raise ValueError(
            f"{task_name}/{split} does not match manifest hash "
            f"({actual[:12]} != {entry[split]['sha256'][:12]}). The frozen split was modified."
        )
    return df


def sample(df: pd.DataFrame, limit: int | None, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Deterministic stratified subsample, used for cheap Jev iteration on val."""
    if limit is None or limit >= len(df):
        return df
    _, held = train_test_split(df, test_size=limit, stratify=df["label"], random_state=seed)
    return held.sort_values("id").reset_index(drop=True)
