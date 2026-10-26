"""
acc_directives / acc_to_omp: OpenACC directives parsed from source, and their
OpenMP translation with data-sharing clauses derived from the analysis.

Line numbers are 0-based indices: the leading newline in each snippet makes
the header line 1, as in the scope_walk tests.
"""

from pathlib import Path

import pytest

from spel.scripts.fortran_parser.symbols import Origin, Symbol, SymbolKind
from spel.scripts.fortran_parser.tests.test_scope_walk import gvar, walk
from spel.scripts.record_access import AccessMapper
from spel.transforms.acc_directives import find_directives, next_code_line
from spel.transforms.acc_to_omp import (
    TranslationError,
    format_directive,
    plan_compute_loop,
    translate_file,
)

MOD = "m"


def test_find_directives():
    src = """
  !$acc parallel loop gang vector collapse(2) default(present) &
  !$acc    private(c, k_vals(:)) reduction(+:s1, s2)  ! trailing comment
  do j = 1, n
  !$acc loop seq
  !$acc routine(foo) seq
  !$acc exit data delete(a(:,:), b)  !!! ipiv(:)
  !$ACC Enter Data create(x)
"""
    lines = src.split("\n")[1:]
    ds = find_directives(lines)
    assert [(d.start_ln, d.end_ln, d.construct) for d in ds] == [
        (0, 1, "parallel loop"),
        (3, 3, "loop"),
        (4, 4, "routine"),
        (5, 5, "exit data"),
        (6, 6, "enter data"),
    ]
    par = ds[0]
    assert par.indent == "  "
    assert par.collapse() == 2
    assert par.var_list("private") == ["c", "k_vals"]
    assert par.reductions() == [("+", ["s1", "s2"])]

    assert ds[2].construct_args == "foo" and [c.name for c in ds[2].clauses] == ["seq"]
    assert ds[3].var_list("delete") == ["a", "b"]
    assert next_code_line(lines, par.end_ln) == 2


def test_find_directives_comment_between_continuations():
    # a commented-out entry of a clause list (LakeTemperatureMod) and a blank line
    src = """
      !$acc enter data create(&
      !! ! $acc   beta(1:num_lakec), &
      !$acc   a(1:n,1:m) ,&

      !$acc   b(1:n))
      x = 1
"""
    lines = src.split("\n")[1:]
    ds = find_directives(lines)
    assert [(d.start_ln, d.end_ln, d.construct) for d in ds] == [(0, 4, "enter data")]
    assert ds[0].var_list("create") == ["a", "b"]


def test_format_directive_wraps():
    clauses = [f"private({', '.join(f'var{i}' for i in range(12))})", "collapse(2)"]
    lines = format_directive("   ", "parallel do", clauses)
    assert lines[0].endswith(" &") and lines[1].startswith("   !$omp   ")
    assert all(len(l) <= 100 for l in lines)


GLOBALS = [
    gvar("r8"),
    gvar("col_pp"),
    gvar("nlev"),
    Symbol("child", SymbolKind.PROCEDURE, Origin.GLOBAL, target=f"{MOD}::child"),
]


class Sub:
    """The parts of Subroutine that the translation uses."""

    def __init__(self, txt: str):
        self.record = walk(txt, globals=GLOBALS)
        self.name = self.record.name
        self.module = MOD
        self.id = f"{MOD}::{self.name}"
        self.host_id = None
        self.library = False
        self.sub_call_desc = {}
        self.sub_lines = []
        self.record_access = None
        lns = [ln for ln, line in enumerate(txt.split("\n")) if line.strip()]
        self.startline, self.end_stmt_ln = lns[0], lns[-1]

    def walk_syntax_tree(self, scopes):
        return self.record


CHILD = """
subroutine child(x, y)
  implicit none
  real(r8), intent(in) :: x
  real(r8), intent(out) :: y
  integer :: ncalls = 0
  ncalls = ncalls + 1
  y = 2*x
end subroutine child
"""

ROUTINE = """
subroutine s(n, f, a, total, flag)
  implicit none
  integer, intent(in) :: n, f(:)
  real(r8), intent(inout) :: a(:,:)
  real(r8), intent(inout) :: total
  real(r8), intent(inout) :: flag
  integer :: i, j, k, c, l
  real(r8) :: t, u, s1, work(n), out
  t = 1._r8
  !$acc parallel loop gang vector collapse(2) default(present) &
  !$acc    private(c, u) reduction(+:s1)
  do j = 1, nlev
     do i = 1, n
        c = f(i)
        l = col_pp%landunit(c)
        t = t + a(c,j)
        !$acc loop seq
        do k = 1, 3
           work(i) = a(c,k)
        end do
        call child(a(c,j), out)
        a(c,j) = out + t
        s1 = s1 + a(c,j)
        flag = 1._r8
     end do
  end do
  total = s1 + l
  !$acc parallel loop collapse(2)
  do i = 1, n
     do j = 1, f(i)
        a(i,j) = 0._r8
     end do
  end do
end subroutine s
"""


@pytest.fixture
def subs():
    sub, child = Sub(ROUTINE), Sub(CHILD)
    mapper = AccessMapper({sub.id: sub, child.id: child})
    return sub, mapper


def test_plan_compute_loop(subs):
    sub, mapper = subs
    lines = ROUTINE.split("\n")
    d = find_directives(lines)[0]
    plan, notes = plan_compute_loop(sub, d, sub.record.loop_at(12), mapper)
    assert plan.collapse == 2 and [l.index for l in plan.nest] == ["j", "i"]
    # c: OpenACC private; l: written but missing from the OpenACC list;
    # k: inner sequential loop index; out: written by the call (intent out)
    assert plan.private == ["c", "l", "k", "out"]
    # t is read before it is written in the loop
    assert plan.firstprivate == ["t"]
    assert plan.reductions == [("+", ["s1"])]
    assert plan.added == ["l", "t", "out"]
    assert plan.unneeded == ["u"]
    assert plan.clauses() == [
        "collapse(2)",
        "private(c, l, k, out)",
        "firstprivate(t)",
        "reduction(+:s1)",
    ]
    msgs = [n.msg for n in notes]
    assert "'t' is read before it is written in the loop: firstprivate" in msgs
    assert "'l' is privatized but read after the loop (line 28)" in msgs
    assert "dummy scalar 'flag' is written in the parallel loop (race)" in msgs
    assert "call child: m::child writes SAVEd local 'ncalls'" in msgs
    # work(i) is a local array written per iteration: shared
    assert "work" not in plan.private


def test_non_rectangular_collapse_is_not_translated(subs):
    sub, mapper = subs
    d = [d for d in find_directives(ROUTINE.split("\n")) if d.is_compute_loop][1]
    plan, notes = plan_compute_loop(sub, d, sub.record.loop_at(29), mapper)
    assert plan is None
    assert "depend on ['i']" in notes[0].msg


def test_translate_file_and_apply_to_shifted_copy(subs, tmp_path: Path):
    sub, mapper = subs
    src = tmp_path / "m.F90"
    src.write_text(ROUTINE)
    tr = translate_file(src, [sub], mapper)
    new = "".join(tr.apply())
    assert "!$acc" not in new.replace("!$acc parallel loop collapse(2)", "")
    assert "  !$omp parallel do collapse(2) private(c, l, k, out) firstprivate(t) reduction(+:s1)\n" in new
    # the non-rectangular loop is left as OpenACC (serial in an OpenMP build)
    assert "!$acc parallel loop collapse(2)" in new
    assert any(n.level == "error" for n in tr.notes)

    # a unit-test copy: a line inserted at the top, another one commented out
    target = ["#include \"unittest_defs.h\"\n", *tr.lines]
    target[4] = "!#py " + target[4]
    shifted = tr.apply(target)
    assert shifted[0] == target[0] and shifted[4] == target[4]
    assert "".join(shifted[1:]).replace("!#py ", "", 1) == new

    # an edited directive line can't be matched
    target[11] = "  !$acc parallel loop gang\n"
    with pytest.raises(TranslationError):
        tr.apply(target)


FILL = """
subroutine fill(n, ok, out, this)
  implicit none
  integer, intent(in) :: n
  logical, intent(in) :: ok(:)
  integer, intent(inout) :: out(:)
  type(foo_type), intent(inout) :: this
  integer :: i, f, cnt, shared_w
  cnt = 0
  !$acc parallel loop gang vector default(present) private(f) copy(cnt, shared_w)
  do i = 1, n
     if (ok(i)) then
        !$acc atomic capture
        cnt = cnt + 1
        f = cnt
        !$acc end atomic
        out(f) = i
        this%vals(i) = 1
        shared_w = i
     end if
  end do
  !$acc parallel loop seq
  do i = 2, n
     out(i) = out(i-1)
  end do
end subroutine fill
"""


def test_atomic_counter_stays_shared(tmp_path: Path):
    sub = Sub(FILL)
    mapper = AccessMapper({sub.id: sub})
    lines = FILL.split("\n")
    ds = find_directives(lines)
    from spel.transforms.acc_to_omp import atomic_regions

    atomics = atomic_regions(ds, lines)
    assert [(sorted(a.lines), a.kind) for a in atomics] == [([13, 14], "capture")]
    plan, notes = plan_compute_loop(sub, ds[0], sub.record.loop_at(10), mapper, atomics)
    # cnt is the atomic target (shared); f is the captured value (private)
    assert plan.private == ["f"] and plan.firstprivate == []
    msgs = [n.msg for n in notes]
    assert msgs == ["scalar 'shared_w' is written but shared (OpenACC data clause): race unless atomic"]
    # writes of this%vals (a component of a dummy) are not scalar races

    src = tmp_path / "fill.F90"
    src.write_text(FILL)
    new = "".join(translate_file(src, [sub], mapper).apply())
    assert "!$omp atomic capture\n" in new and "!$omp end atomic\n" in new
    # `parallel loop seq` runs sequentially: no OpenMP directive
    assert new.count("!$omp parallel do") == 1 and "!$acc" not in new


UNUSED_DUMMY = """
subroutine untouched(p, obj, x)
  implicit none
  integer, intent(in) :: p
  type(foo_type), intent(inout) :: obj
  real(r8), intent(inout) :: x(:)
  x(p) = 0._r8
end subroutine untouched
"""

PASSES_OBJECT = """
subroutine caller(n, f, obj, x)
  implicit none
  integer, intent(in) :: n, f(:)
  type(foo_type), intent(inout) :: obj
  real(r8), intent(inout) :: x(:)
  integer :: fp, p
  !$acc parallel loop gang vector default(present) private(p)
  do fp = 1, n
     p = f(fp)
     call untouched(p, obj, x)
  end do
end subroutine caller
"""


def test_derived_type_passed_whole_is_not_a_scalar_race():
    # the callee never touches obj: the mapper reports its intent (rw) for the whole object on
    # the call line, which is not a write of a shared scalar
    callee = Sub(UNUSED_DUMMY)
    GLOBALS.append(Symbol("untouched", SymbolKind.PROCEDURE, Origin.GLOBAL, target=f"{MOD}::untouched"))
    try:
        sub = Sub(PASSES_OBJECT)
    finally:
        GLOBALS.pop()
    mapper = AccessMapper({sub.id: sub, callee.id: callee})
    d = find_directives(PASSES_OBJECT.split("\n"))[0]
    plan, notes = plan_compute_loop(sub, d, sub.record.loop_at(8), mapper)
    assert not [n.msg for n in notes if "race" in n.msg]
    assert plan.private == ["p"]
