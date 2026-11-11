"""Where `spel proptest` keeps its files.

- `work_dir`: `<case>/proptest/`, scratch that `spel create` may delete with the case (the
  synthetic domains, worker logs, the units cache);
- `data_dir`: `unit-tests/proptest-data/<case>/`, next to `input-data/`, for what must outlive
  a regenerated case: Hypothesis' example database (the failing examples it replays first),
  saved failures, the `stats` rule files and the `subspace` results.
"""

from __future__ import annotations

from pathlib import Path

PROPTEST_DIR = "proptest"
DATA_DIR = "proptest-data"


def work_dir(case_dir: Path) -> Path:
    return Path(case_dir) / PROPTEST_DIR


def data_dir(case_dir: Path) -> Path:
    case_dir = Path(case_dir)
    return case_dir.parent / DATA_DIR / case_dir.name


def slug(text: str) -> str:
    """A file name for a scenario, check or expression."""
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
