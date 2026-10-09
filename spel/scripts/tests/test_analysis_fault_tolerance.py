"""
The elm_drv analysis keeps going when a routine fails: the failure is
recorded, and routines whose call trees contain it are flagged incomplete.
"""

import logging
from types import SimpleNamespace

import pytest

from spel.scripts.helper_functions import construct_call_tree
from spel.scripts.record_access import AccessMapper
from spel.scripts.UnitTestforELM import incomplete_analyses


class FakeSub:
    def __init__(self, id_, children=(), fail_collect=False, fail_walk=False):
        self.id = id_
        self.name = id_.split("::")[-1]
        self.library = False
        self.preprocessed = False
        self.child_subroutines = {c.id: c for c in children}
        self.abstract_call_tree = None
        self.record_access = None
        self.logger = logging.getLogger(id_)
        self.fail_collect = fail_collect
        self.fail_walk = fail_walk

    def collect_var_and_call_info(self, *args, **kwargs):
        if self.fail_collect:
            raise SystemExit(f"cannot parse {self.id}")
        self.preprocessed = True

    def walk_syntax_tree(self, scopes):
        if self.fail_walk:
            raise ValueError(f"bad walk {self.id}")
        return SimpleNamespace()


@pytest.fixture
def tree():
    leaf = FakeSub("m::leaf")
    bad = FakeSub("m::bad", children=[leaf], fail_collect=True)
    mid = FakeSub("m::mid", children=[bad])
    ok = FakeSub("m::ok", children=[leaf])
    root = FakeSub("m::root", children=[mid, ok])
    return {s.id: s for s in (leaf, bad, mid, ok, root)}


def names(sub):
    return [t.node.subname for t in sub.abstract_call_tree.traverse_postorder()]


def test_call_tree_keeps_failed_child_as_leaf(tree):
    failures: dict[str, str] = {}
    construct_call_tree(tree["m::root"], tree, {}, {}, 0, failures)
    assert list(failures) == ["m::bad"]
    assert "cannot parse m::bad" in failures["m::bad"]
    assert names(tree["m::root"]) == ["m::bad", "m::mid", "m::leaf", "m::ok", "m::root"]


def test_call_tree_raises_without_failures(tree):
    with pytest.raises(SystemExit):
        construct_call_tree(tree["m::root"], tree, {}, {}, 0)


def test_mapper_records_failure_instead_of_raising():
    sub = FakeSub("m::bad", fail_walk=True)
    failures: dict[str, str] = {}
    mapper = AccessMapper({sub.id: sub}, failures=failures)
    assert mapper.maps(sub) is None
    assert failures == {"m::bad": "ValueError: bad walk m::bad"}
    sub.fail_walk = False  # a failed routine isn't retried
    assert mapper.maps(sub) is None
    with pytest.raises(ValueError):
        AccessMapper({sub.id: FakeSub("m::bad", fail_walk=True)}).maps(
            FakeSub("m::bad", fail_walk=True)
        )


def test_incomplete_analyses(tree):
    failures: dict[str, str] = {}
    construct_call_tree(tree["m::root"], tree, {}, {}, 0, failures)
    for sub in tree.values():
        if sub.id not in failures:
            sub.record_access = object()
    assert incomplete_analyses(tree, failures) == {
        "m::mid": ["m::bad"],
        "m::root": ["m::bad"],
    }
