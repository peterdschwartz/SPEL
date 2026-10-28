"""acc-locals: device storage of local work arrays."""

from types import SimpleNamespace

from spel.scripts.fortran_parser.tests.test_scope_walk import walk
from spel.transforms.acc_locals import check_routine

SRC = """subroutine host(n, x)
  implicit none
  integer, intent(in) :: n
  real(r8), intent(inout) :: x(:)
  integer :: i, nc
  integer :: tab(4)
  real(r8) :: work(10)
  nc = 0
  call add(2)
  !$acc parallel loop gang vector default(present)
  do i = 1, n
     work(i) = x(i) * tab(1)
     x(i) = work(i)
  end do
contains
  subroutine add(v)
    integer, intent(in) :: v
    nc = nc + 1
    tab(nc) = v
  end subroutine add
end subroutine host"""


def _sub(tmp_path):
    path = tmp_path / "m.F90"
    path.write_text(SRC)
    rec = walk(SRC)
    lns = [ln for ln, line in enumerate(SRC.split("\n")) if line.strip()]
    sub = SimpleNamespace(id="m::host", name="host", record=rec, filepath=path,
                          startline=lns[0], endline=lns[-1])
    inner = walk(SRC, host=rec, tree_index=0)
    return sub, inner


def test_internal_subprogram_writes_are_host_accesses(tmp_path):
    from spel.scripts.fortran_parser.symbols import Origin

    sub, inner = _sub(tmp_path)
    uses = {"add": {a.base.lower() for a in inner.accesses() if a.origin is Origin.HOST}}
    assert uses["add"] == {"nc", "tab"}
    lines = SRC.split("\n")
    r = check_routine(sub, set(), lines)
    assert set(r.locals_needed) == {"work", "tab"} and r.host_accessed == []   # tab would be device-only
    r = check_routine(sub, set(), lines, uses)
    assert r.host_accessed == ["tab"]            # set by add on the host: needs a copy to the device


def test_insertion_goes_above_the_directive_of_the_first_statement():
    from spel.transforms.acc_directives import find_directives
    from spel.transforms.acc_locals import _above_directives

    lines = [
        "subroutine s(n)\n",
        "  integer :: n, i\n",
        "  real :: a(10)\n",
        "  !$acc parallel loop gang vector &\n",
        "  !$acc& default(present)\n",
        "  do i = 1, n\n",
        "     a(i) = 0.\n",
        "  end do\n",
        "end subroutine s\n",
    ]
    directives = find_directives(lines, 0, len(lines) - 1)
    assert _above_directives(5, directives) == 3   # before the parallel loop, not between it and the do
    assert _above_directives(6, directives) == 6   # no directive right above


def test_generic_call_resolving_to_a_kernel_routine_is_a_device_use(tmp_path):
    src = """subroutine s(n, g)
  implicit none
  integer, intent(in) :: n
  real(r8), intent(out) :: g(:)
  real(r8) :: work(10)
  call avg(n, work, g)
end subroutine s"""
    path = tmp_path / "s.F90"
    path.write_text(src)
    rec = walk(src)
    lines = src.split("\n")
    call = next(c for c in rec.calls if str(c.name) == "avg")
    sub = SimpleNamespace(id="m::s", name="s", record=rec, filepath=path, startline=0, endline=len(lines) - 1,
                          sub_call_desc={})
    assert check_routine(sub, {"avg_1d"}, lines).locals_needed == []        # `avg` itself isn't known
    sub.sub_call_desc = {call.ln: SimpleNamespace(fn="m::avg_1d")}           # resolved generic
    assert check_routine(sub, {"avg_1d"}, lines).locals_needed == ["work"]
