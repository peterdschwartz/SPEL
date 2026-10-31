import pytest

from spel.transforms.core.edits import EditConflict, EditMismatch, EditSet, delete, insert, replace

SRC = ["a", "b", "c", "d"]


def test_edits_apply_against_original_lines():
    es = EditSet().add(insert(1, ["x"]), replace(2, ["C1", "C2"]), delete(3), insert(4, "end"))
    assert es.apply(SRC) == ["a", "x", "b", "C1", "C2", "end"]
    assert SRC == ["a", "b", "c", "d"]


def test_order_at_one_line():
    es = EditSet().add(insert(1, "late", order=2), replace(1, "B", order=1), insert(1, "early"),
                       insert(1, "early2"))
    assert es.apply(SRC) == ["a", "early", "early2", "B", "late", "c", "d"]


def test_conflicts():
    es = EditSet().add(replace(1, ["x"], n=2))
    with pytest.raises(EditConflict):
        es.add(replace(2, "y"))
    with pytest.raises(EditConflict):
        es.add(insert(2, "z"))
    es.add(insert(3, "after"), replace(1, ["x"], n=2))  # boundary insertion, identical replacement
    assert es.apply(SRC) == ["a", "x", "after", "d"]


def test_expect():
    with pytest.raises(EditMismatch):
        EditSet().add(replace(0, "A", expect=["z"])).apply(SRC)
    assert EditSet().add(replace(0, "A", expect=["a"])).apply(SRC)[0] == "A"


def test_diff():
    d = EditSet().add(replace(0, "A")).diff(SRC)
    assert "-a" in d and "+A" in d
