import pytest

from spel.transforms.core.edits import EditConflict, EditMismatch, EditSet, delete, diff_edits, insert, replace

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


def test_diff_edits_rebuild_the_new_text():
    old = ["module m", "  use a", "  type t", "  end type", "contains", "end module m"]
    new = ["module m", "  use a", "  use b, only : t", "contains", "  ! new", "end module m"]
    es = EditSet().extend(diff_edits(old, new))
    assert es.apply(old) == new
    with pytest.raises(EditMismatch):  # planned on other text
        es.apply(["module m", "  use a", "  type u", "  end type", "contains", "end module m"])


def test_text_edits():
    from spel.transforms.core.edits import TextEdit, replace_cols, replace_span, span_text

    lines = ["x(c) = y(c) + 1", "z = 2", "w = 3"]
    es = EditSet().add(
        replace_cols(0, 2, 3, "kc", expect="c"),
        replace_span((0, 9, 0, 10), "kc", lines),
        replace_cols(0, 9, 9, "k0+"),  # insertion before a replacement at the same column
        replace_cols(0, 15, 15, " + 2"),
        replace_cols(0, 15, 15, "!"),  # insertions at one column: in the order added
        insert(1, "new"),
        replace(2, "W = 3"),
    )
    es.add(replace_cols(0, 2, 3, "kc", expect="c"))  # identical: merged
    assert es.apply(lines) == ["x(kc) = y(k0+kc) + 1 + 2!", "new", "z = 2", "W = 3"]
    assert span_text(lines, (0, 0, 1, 1)) == "x(c) = y(c) + 1\nz"
    with pytest.raises(EditConflict):
        es.add(replace_cols(0, 1, 3, "q"))
    with pytest.raises(EditConflict):
        es.add(replace_cols(2, 0, 1, "q"))
    with pytest.raises(EditConflict):
        EditSet().add(replace_cols(1, 0, 1, "q"), delete(1))
    with pytest.raises(EditMismatch):
        EditSet().add(TextEdit(0, 0, 1, "q", expect="y")).apply(lines)
