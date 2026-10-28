"""defer-checks: host-only checks moved out of loops (last failing element, reports, aborts)."""

from types import SimpleNamespace

from spel.scripts.fortran_parser.tests.test_scope_walk import parse_statements, walk
from spel.transforms import defer_checks as dc

SEARCH = """subroutine check(n, filter, err, found, idx)
  implicit none
  integer, intent(in) :: n
  integer, intent(in) :: filter(:)
  real(r8), intent(inout) :: err(:)
  logical, intent(out) :: found
  integer, intent(out) :: idx
  integer :: f, c
  found = .false.
  do f = 1, n
     c = filter(f)
     err(c) = err(c) * 2.0_r8
     if (abs(err(c)) > 1.0_r8) then
        found = .true.
        idx = c
     end if
  end do
  if (found) call endrun('bad')
end subroutine check"""

ABORT = """subroutine report(n, x)
  implicit none
  integer, intent(in) :: n
  real(r8), intent(inout) :: x(:)
  integer :: i, iulog
  do i = 1, n
     x(i) = x(i) + 1.0_r8
     if (x(i) < 0.0_r8) then
        write(iulog,*) 'negative x', i, x(i)
        call endrun('negative x')
     end if
  end do
end subroutine report"""

RESET = """subroutine reset(n, x)
  implicit none
  integer, intent(in) :: n
  real(r8), intent(inout) :: x(:)
  integer :: i, iulog
  do i = 1, n
     if (x(i) < 0.0_r8) then
        write(iulog,*) 'resetting', i
        x(i) = 0.0_r8
     end if
  end do
end subroutine reset"""


def _plan(src):
    lines = src.split("\n")
    rec = walk(src)
    tree = parse_statements(src).statements[0]
    sub = SimpleNamespace(id="m::s", startline=0, endline=len(lines) - 1, filepath="m.F90")
    return dc.plan_routine(sub, lines, tree, rec), lines


def _apply(src):
    p, lines = _plan(src)
    first = next(i for i, l in enumerate(lines) if l.strip().startswith(("found = .false.", "do i")))
    for ln, _, n, new in sorted(dc.edits_for(lines, p.deferrals, {"m::s": first}), key=lambda e: (-e[0], -e[1])):
        lines[ln:ln + n] = new
    return p, "\n".join(lines)


def test_search_keeps_the_last_failing_element():
    p, out = _apply(SEARCH)
    (d,) = p.deferrals
    assert d.cond == "abs(err(c)) > 1.0_r8" and d.replay == ["     c = filter(f)"]
    body = [l.strip() for l in out.split("\n")]
    assert "if (abs(err(c)) > 1.0_r8) spel_dc_1 = max(spel_dc_1, f - (1) + 1)  ! spel defer-checks" in body
    after = body[body.index("if (spel_dc_1 > 0) then  ! spel defer-checks: the last element where the check held"):]
    assert after[1:6] == ["f = 1 + spel_dc_1 - 1", "c = filter(f)", "found = .true.", "idx = c", "end if"]
    assert body.index("spel_dc_1 = 0  ! spel defer-checks") < body.index("do f = 1, n")
    # the rewritten loop is a max reduction for the planner
    from spel.transforms.acc_to_omp import _integer_max

    rec = walk(out)
    lines = out.split("\n")
    lo = next(i for i, l in enumerate(lines) if l.strip() == "do f = 1, n")
    hi = next(i for i in range(lo, len(lines)) if lines[i].strip() == "end do")
    assert _integer_max(rec, "spel_dc_1", lo, hi)


def test_aborting_check_moves_after_the_loop():
    p, out = _apply(ABORT)
    (d,) = p.deferrals
    assert [l.strip() for l in d.body] == ["write(iulog,*) 'negative x', i, x(i)", "call endrun('negative x')"]
    assert d.replay == []


def test_check_that_changes_state_is_left_alone():
    p, _ = _plan(RESET)
    assert p.deferrals == []
    assert p.refused and "array element" in p.refused[0][2]


NESTED = """subroutine nested(n, active, err, found, idx)
  implicit none
  integer, intent(in) :: n
  logical, intent(in) :: active(:)
  real(r8), intent(in) :: err(:)
  logical, intent(out) :: found
  integer, intent(out) :: idx
  integer :: c
  found = .false.
  do c = 1, n
     if (active(c)) then
        if (abs(err(c)) > 1.0_r8) then
           found = .true.
           idx = c
        end if
     end if
  end do
end subroutine nested"""


def test_the_search_if_is_deferred_not_its_guard():
    p, _ = _plan(NESTED)
    (d,) = p.deferrals
    assert d.cond == "abs(err(c)) > 1.0_r8"   # recorded only where the search test holds


FIRST = """subroutine first(n, cwd, srep)
  implicit none
  integer, intent(in) :: n
  logical, intent(in) :: cwd(:)
  integer, intent(out) :: srep
  integer :: s
  srep = 0
  do s = 1, n
     if (.not. cwd(s)) then
        srep = s
        exit
     end if
  end do
end subroutine first"""


def test_first_match_search_with_exit_is_left_alone():
    p, _ = _plan(FIRST)
    assert p.deferrals == []
    assert p.refused and "ExitStatement" in p.refused[0][2]


def test_a_second_run_leaves_deferred_checks_alone():
    p, out = _apply(SEARCH)
    lines = out.split("\n")
    rec = walk(out)
    tree = parse_statements(out).statements[0]
    sub = SimpleNamespace(id="m::s", startline=0, endline=len(lines) - 1, filepath="m.F90")
    again = dc.plan_routine(sub, lines, tree, rec)
    assert again.deferrals == []
