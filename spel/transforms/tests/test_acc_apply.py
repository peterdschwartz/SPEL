"""acc-apply edits on a snippet: the interchange of a (level, column) nest."""
from spel.scripts.record_access import AccessMapper
from spel.scripts.tests.test_record_access import StubSub
from spel.transforms.acc_apply import Insertion, Port, _loop_edits, _swap, _swappable
from spel.transforms.acc_plan import propose
from spel.transforms.tests.test_acc_plan import SRC

TRIANGULAR = """
subroutine tri(num, filter, nlev, x, tot)
  implicit none
  integer, intent(in) :: num, filter(:), nlev(:)
  real(r8), intent(in) :: x(:,:)
  real(r8), intent(inout) :: tot(:)
  integer :: fc, c, j
  do j = 1, 10
     do fc = 1, nlev(j)
        c = filter(fc)
        tot(c) = tot(c) + x(c,j)
     end do
  end do
end subroutine tri
"""


def _props(src):
    sub = StubSub(src)
    sub.filepath = None
    return propose(sub, AccessMapper({sub.id: sub}), source=src.splitlines())


def test_interchange_swaps_the_nest():
    lines = SRC.splitlines()
    inter = [p for p in _props(SRC) if p.kind == "interchange"][0]
    assert _swappable(inter.loop)
    port = Port("demo", inter, interchange=True)
    _swap(port, lines)
    ins = _loop_edits(port, lines)
    for x in sorted(ins, key=lambda x: (-x.ln, -x.order)):
        lines[x.ln:x.ln] = x.lines
    body = [l.strip() for l in lines[14:23]]
    assert body[0].startswith("!$acc parallel loop gang vector default(present)")
    assert body[1:] == ["do fc = 1, num", "!$acc loop seq", "do j = 1, 10", "c = filter(fc)",
                        "tot(c) = tot(c) + x(c,j)", "end do", "end do", "do fp = 1, numv"]
    assert lines[15] == "  do fc = 1, num" and lines[17] == "     do j = 1, 10"


def test_triangular_nest_is_not_swapped():
    inter = [p for p in _props(TRIANGULAR) if p.kind == "interchange"]
    assert inter and not _swappable(inter[0].loop)


def test_abort_calls_from_the_parse_tree():
    from spel.scripts.fortran_parser.tests.test_scope_walk import parse_statements
    from spel.transforms.acc_apply import abort_calls

    src = """subroutine s(x)
  real :: x
  if (x < 0.) call endrun(msg='neg')
  if (x > 1.) then
     call shr_sys_abort('big')
  end if
  call foo(x)
end subroutine s"""
    assert abort_calls(parse_statements(src).statements[0].body) == [2, 4]


def test_cold_branches():
    from spel.scripts.fortran_parser.tests.test_scope_walk import parse_statements
    from spel.transforms.acc_apply import cold_branches

    src = """subroutine s(x, n)
  real :: x(:)
  integer :: n, i
  if (n > 0) then
     do i = 1, n
        if (x(i) < 0.) then
           write(*,*) x(i)
           call endrun('neg')
        end if
     end do
  else
     x(1) = 0.
  end if
  if (n > 3) x(2) = 1.
end subroutine s"""
    # the outermost if that can abort: both of its branches, not the nested if
    assert cold_branches(parse_statements(src).statements[0].body) == [(4, 9, 4), (11, 11, 11)]
