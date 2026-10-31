from spel.transforms.core import fortran
from spel.transforms.core.source import (
    code, comment, continues, matching_paren, split_top_level, statement_end, statement_start,
    statement_text,
)

LINES = [
    "    x = a + &   ! first",
    "",
    "        ! a comment between continuations",
    "        & b  ! 'x'",
    "    s = 'a ! not a comment' // &",
    "        \"it's\"",
    "    !$acc parallel loop gang &",
    "    !$acc& private(i)",
    "    do i = 1, n",
]


def test_comments_respect_strings():
    assert code(LINES[4]) == LINES[4]
    assert comment(LINES[3]) == "! 'x'"
    assert code("a = '!' ! c") == "a = '!' "


def test_statement_bounds_skip_comment_lines():
    assert statement_end(LINES, 0) == 3
    assert statement_start(LINES, 3) == 0
    assert statement_text(LINES, 0) == "x = a + b"
    assert statement_end(LINES, 4) == 5


def test_directive_continuation():
    assert continues(LINES[6])
    assert statement_end(LINES, 6) == 7
    assert statement_end(LINES, 8) == 8


def test_paren_helpers():
    t = "f(a(1, 2), 'x)', b)"
    assert matching_paren(t, 1) == len(t) - 1
    assert split_top_level(t[2:-1]) == ["a(1, 2)", "'x)'", "b"]


def test_directive_list_wraps_with_sentinel_continuation():
    out = fortran.directive_list("  ", "update self", ["alpha", "beta", "gamma"], width=34,
                                 tail=") if_present")
    assert out == ["  !$acc update self(alpha, beta, &", "  !$acc&   gamma) if_present"]


def test_directive_clauses_and_items():
    assert fortran.directive_clauses("", "parallel do", ["private(i)", "collapse(2)"], width=30) == [
        "!$omp parallel do private(i) &", "!$omp   collapse(2)"]
    assert fortran.use_stmt("  ", "m", ["a", "b"]) == ["  use m, only : a, b"]
    assert fortran.wrap_items("  public ::", ["aaaa", "bbbb", "cccc"], width=22) == [
        "  public :: aaaa, &", "      bbbb, cccc"]


def test_if_then_breaks_at_logical_operators():
    out = fortran.if_then("  ", "a > 1 .and. b > 2 .or. c", width=24)
    assert out[0].endswith("&") and out[-1].endswith(") then")
    assert " ".join(l.strip(" &") for l in out).replace("  ", " ") .count(".and.") == 1


def test_fresh_name():
    assert fortran.fresh_name("i", ["I", "i_2"]) == "i_3"
    assert fortran.fresh_name("k", ["i"]) == "k"
