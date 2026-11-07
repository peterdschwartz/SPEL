"""Local work arrays stored on their filter (filter_locals)."""
from spel.scripts.types import LineTuple
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.fortran_parser.tests.test_scope_walk import walk
from spel.transforms.core.edits import EditSet
from spel.transforms.filter_locals import plan_routine

SRC = """\
subroutine s(bounds, num_nolakec, filter_nolakec, t)
  type(bounds_type), intent(in) :: bounds
  integer, parameter :: nlevgrnd = 15
  integer, intent(in) :: num_nolakec
  integer, intent(in) :: filter_nolakec(:)
  real(r8), intent(inout) :: t(bounds%begc:,:)
  real(r8) :: tk(bounds%begc:bounds%endc, 1:nlevgrnd)
  real(r8) :: fn(bounds%begc:bounds%endc)
  real(r8) :: whole(bounds%begc:bounds%endc)
  real(r8) :: after(bounds%begc:bounds%endc)
  real(r8) :: passed(bounds%begc:bounds%endc)
  integer :: c, fc, j
  whole(:) = 0._r8
  do fc = 1, num_nolakec
     c = filter_nolakec(fc)
     fn(c) = 0._r8
     whole(c) = 1._r8
     after(c) = 2._r8
     passed(c) = 3._r8
     do j = 1, nlevgrnd
        tk(c,j) = t(c,j)
     end do
  end do
  t(c,1) = after(c)
  call g(passed)
  do j = 1, nlevgrnd
     do fc = 1, num_nolakec
        c = filter_nolakec(fc)
        if (fn(c) > 1._r8) then
           t(c,j) = 0._r8
        else if (fn(c) < 0._r8) then
           t(c,j) = 1._r8
        end if
        t(c,j) = tk(c,j) + fn(c) * &
                 tk(c, j)
     end do
  end do
end subroutine s
"""


class Sub:
    id = "m::s"
    startline = 0


def _apply(src):
    lines = src.split("\n")
    Sub.record = walk(src)
    Sub.endline = len(lines) - 1
    tree = Parser(lines=[_lt(l, i) for i, l in enumerate(lines)], keep_spans=True).parse_program().statements[0]
    edits, done, why = plan_routine(Sub, lines, tree)
    es = EditSet("x")
    for e in edits:
        es.add(e)
    return es.apply(lines), done, why


def _lt(line, i):
    return LineTuple(line=line, ln=i)


def test_filter_locals():
    out, done, why = _apply(SRC)
    text = "\n".join(out)
    assert sorted(done) == ["fn", "tk"], why
    assert "real(r8) :: tk(num_nolakec, 1:nlevgrnd)" in text
    assert "real(r8) :: fn(num_nolakec)" in text
    assert "tk(fc,j) = t(c,j)" in text and "fn(fc) = 0._r8" in text
    assert "else if (fn(fc) < 0._r8) then" in text  # else-if conditions are rewritten too
    assert "t(c,j) = tk(fc,j) + fn(fc) * &" in text and "tk(fc, j)" in text  # a continued statement
    # left alone: used whole, used after its loop, passed as an argument
    assert "whole(c) = 1._r8" in text and "after(c) = 2._r8" in text and "passed(c) = 3._r8" in text
    reasons = "\n".join(why)
    assert "whole:" in reasons and "after:" in reasons and "passed:" in reasons


def test_result_parses_and_second_run_changes_nothing():
    out, _, _ = _apply(SRC)
    again, done, _ = _apply("\n".join(out))
    assert done == [] and again == out


def test_io_blanking_spans_comment_lines_in_a_continuation():
    from spel.scripts.registrations import _blank_io_statements
    src = ["write(iulog,*)'a ',&", "     ' b= ',b, &", "     !' c= ',c, &", "     ' d= ',d", "x = 1"]
    out = _blank_io_statements([LineTuple(line=l, ln=i) for i, l in enumerate(src)])
    assert [lt.line for lt in out] == ["", "", "", "", "x = 1"]


def test_io_blanking_ignores_bang_inside_strings():
    from spel.scripts.registrations import _blank_io_statements
    src = ["write(iulog,*)'Lake! c, rest', &", "     'x = ',c", "y = 2"]
    out = _blank_io_statements([LineTuple(line=l, ln=i) for i, l in enumerate(src)])
    assert [lt.line for lt in out] == ["", "", "y = 2"]


def test_uses_the_record_does_not_know_block_the_rewrite():
    # the analysis' record misses uses in branches it commented out (`!#py`): simulate with a
    # record of the source without the extra line, plan on the file that has it
    extra = SRC.replace("     fn(c) = 0._r8\n", "     fn(c) = 0._r8\n     t(c,1) = fn(c) + 1._r8\n")
    lines = extra.split("\n")
    Sub.record = walk(SRC)
    Sub.endline = len(lines) - 1
    tree = Parser(lines=[_lt(l, i) for i, l in enumerate(lines)], keep_spans=True).parse_program().statements[0]
    edits, done, why = plan_routine(Sub, lines, tree)
    assert "fn" not in done and any(w.startswith("fn: the file has uses") or w.startswith("fn: no subscript") for w in why)


def test_use_in_io_blocks_the_rewrite():
    src = SRC.replace("  call g(passed)\n", "  call g(passed)\n  write(iulog,*) 'fn ', fn(c)\n").replace(
        "  integer, parameter :: nlevgrnd = 15\n", "  integer, parameter :: nlevgrnd = 15\n  integer, parameter :: iulog = 6\n")
    _, done, why = _apply(src)
    assert "fn" not in done and any(w.startswith("fn:") for w in why)
