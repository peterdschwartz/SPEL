"""
Round-trip test for pickling/unpickling a FunctionalUnitTest.

Runs `create_unit_test` on an example ELM subroutine, unpickles the result and
checks that the unpickled object graph matches the returned FunctionalUnitTest
exactly (attributes, container ordering and shared references).
"""

import math
from pathlib import Path

import pytest

from spel.scripts.config import ELM_SRC

EXAMPLE_SUBS = ["canopyfluxes"]
CASE_NAME = "test_pickle_canopyfluxes"

_ATOMS = (type(None), bool, int, float, complex, str, bytes, Path)


def assert_deep_equal(orig, loaded) -> None:
    """
    Structurally compare two object graphs. Custom ``__eq__`` methods are
    deliberately bypassed for objects with a ``__dict__`` so that every
    attribute is compared. Also verifies that aliasing is preserved: an object
    referenced in multiple places in ``orig`` must map to a single object in
    ``loaded``.
    """
    memo: dict[int, object] = {}
    stack = [(orig, loaded, "fut")]

    while stack:
        a, b, path = stack.pop()

        if a is b:
            continue

        assert type(a) is type(b), f"{path}: type {type(a)} != {type(b)}"

        if isinstance(a, float) and math.isnan(a):
            assert math.isnan(b), f"{path}: nan != {b!r}"
            continue
        if isinstance(a, _ATOMS):
            assert a == b, f"{path}: {a!r} != {b!r}"
            continue

        if id(a) in memo:
            assert memo[id(a)] is b, f"{path}: shared reference not preserved"
            continue
        memo[id(a)] = b

        if isinstance(a, dict):
            a_keys, b_keys = list(a.keys()), list(b.keys())
            assert a_keys == b_keys, f"{path}: dict keys differ"
            for ka, kb in zip(a_keys, b_keys):
                stack.append((ka, kb, f"{path}.<key {ka!r}>"))
                stack.append((a[ka], b[kb], f"{path}[{ka!r}]"))
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), f"{path}: len {len(a)} != {len(b)}"
            for i, (ea, eb) in enumerate(zip(a, b)):
                stack.append((ea, eb, f"{path}[{i}]"))
            if hasattr(a, "__dict__"):
                stack.append((vars(a), vars(b), f"{path}.__dict__"))
        elif isinstance(a, (set, frozenset)):
            assert a == b, f"{path}: sets differ"
            b_lookup = {eb: eb for eb in b}
            for ea in a:
                stack.append((ea, b_lookup[ea], f"{path}{{{ea!r}}}"))
        elif hasattr(a, "__dict__"):
            stack.append((vars(a), vars(b), f"{path}.__dict__"))
        else:
            assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.fixture
def created_fut(tmp_path, monkeypatch):
    """
    Run create_unit_test with the case dir and pickle output redirected to
    tmp_path so the real unit-tests dir and pickles in scripts_dir are untouched.
    """
    if not Path(ELM_SRC).is_dir():
        pytest.skip(f"ELM source not found at {ELM_SRC}")

    import spel.scripts.export_objects as eo
    import spel.scripts.UnitTestforELM as ut

    monkeypatch.setattr(ut, "unittests_dir", tmp_path)
    monkeypatch.setattr(eo, "scripts_dir", tmp_path)

    fut = ut.create_unit_test(
        sub_names=EXAMPLE_SUBS,
        casename=CASE_NAME,
        keep=False,
        db_mode=False,
    )
    assert (tmp_path / f"fut_{CASE_NAME}.pkl").is_file()
    return fut


def test_pickle_roundtrip_matches_fut(created_fut):
    from spel.scripts.export_objects import unpickle_unit_test

    fut = created_fut
    assert fut.subroutine_dict, "expected subroutines to be parsed"
    assert any(key.endswith("::canopyfluxes") for key in fut.subroutine_dict)

    loaded = unpickle_unit_test(CASE_NAME)

    assert loaded is not fut
    assert_deep_equal(fut, loaded)
