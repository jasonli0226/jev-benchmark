"""Harness version: a content hash of every file that affects scoring.

Model implementations are deliberately excluded so adding a model does not
invalidate earlier results; data loading, serialization, the Jev client,
metrics and the runner are included.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_HARNESS_PATHS = ("data", "eval", "jev", "runner.py", "config.py")


@lru_cache(maxsize=1)
def harness_version() -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for rel in _HARNESS_PATHS:
        path = _PKG / rel
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    for file in files:
        digest.update(str(file.relative_to(_PKG)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()[:12]
