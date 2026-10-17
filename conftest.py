"""Shared pytest fixtures."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

# Frozen copy of an elm_drv analysis so tests never re-analyze E3SM.
# Override with SPEL_TEST_ANALYSIS=/path/to/analysis.pkl.
ANALYSIS_FIXTURE = Path(__file__).parent / ".pytest-fixtures/elm_drv_analysis.pkl"


@pytest.fixture(scope="session")
def elm_drv_analysis():
    """
    The cached elm_drv analysis (a FunctionalUnitTest). Seeded once from the
    E3SM_SRCROOT cache (`spel analyze`/`spel create`); skipped if neither exists.
    Shared by the whole session: tests must not modify it.
    """
    from spel.scripts.analysis_cache import ANALYSIS_PICKLE, cache_dir
    from spel.scripts.export_objects import _load_pickle

    path = Path(os.environ.get("SPEL_TEST_ANALYSIS", ANALYSIS_FIXTURE))
    if not path.is_file() and path == ANALYSIS_FIXTURE:
        cached = cache_dir() / ANALYSIS_PICKLE
        if not cached.is_file():
            pytest.skip(f"no elm_drv analysis: {path} or {cached}")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, path)
    if not path.is_file():
        pytest.skip(f"no elm_drv analysis at {path}")
    return _load_pickle(path)


@pytest.fixture(autouse=True)
def restore_edit_files_globals():
    """
    edit_files keeps per-run state in module globals (modules found unavailable,
    names they no longer export); restore it in place after each test so one
    test's analysis doesn't change another's edits.
    """
    import copy

    import spel.scripts.edit_files as ef

    names = ("bad_modules", "bad_symbols", "unavailable_exports", "bad_module_imports", "opaque_exports")
    saved = {n: copy.deepcopy(getattr(ef, n)) for n in names}
    yield
    for n, value in saved.items():
        current = getattr(ef, n)
        current.clear()
        current.update(value)
