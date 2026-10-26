"""Host segments of a routine that mixes kernels and host code (acc_sync)."""
from types import SimpleNamespace

from spel.scripts.fortran_parser.tests.test_scope_walk import parse_statements
from spel.scripts.record_access import AccessMapper
from spel.scripts.tests.test_record_access import StubSub
from spel.transforms import acc_sync
from spel.transforms.acc_directives import find_directives

SRC = """subroutine mixed(n, x, y)
  implicit none
  integer, intent(in) :: n
  real(r8), intent(inout) :: x(:), y(:)
  integer :: i
  !$acc parallel loop default(present)
  do i = 1, n
     x(i) = 1.0_r8
  end do
  do i = 1, n
     y(i) = x(i) + 1.0_r8
  end do
  if (n > 0) then
     !$acc parallel loop default(present)
     do i = 1, n
        if (y(i) > 0.0_r8) x(i) = y(i)
     end do
     y(1) = 0.0_r8
     if (n > 5) return
     y(2) = 0.0_r8
  end if
end subroutine mixed"""


def test_segments_and_edits():
    lines = SRC.split("\n")
    sub = StubSub(SRC)
    sub.active_global_vars = {}
    fut = SimpleNamespace(type_dict={}, module_dict={}, subroutine_dict={sub.id: sub})
    types = acc_sync._Types(fut)
    spans = acc_sync._spans(find_directives(lines), sub.record)
    entries = acc_sync._entries(AccessMapper({sub.id: sub}).maps(sub))
    out = acc_sync.SyncPlan(device_written={("#dummy", sub.id, "x")})  # x: written by the kernels
    rs = acc_sync.RoutineSync(sub.id, None)
    node = parse_statements(SRC).statements[0]
    acc_sync._segments(rs, sub, node, spans, entries, lines, types, out)

    host_loop, tail = rs.segments  # the `if (y(i) > 0)` inside the second kernel is device code: no segment
    assert (host_loop.first, host_loop.after) == (9, 12)
    assert host_loop.stale == {"x"} and host_loop.writes == {"y"}
    # y(1) = ...; if (n > 5) return; y(2) = ...: one segment, the update also before the return
    assert (tail.first, tail.after) == (17, 20) and tail.writes == {"y"} and not tail.stale
    assert tail.returns == [18]

    by_file = acc_sync.edits(acc_sync.SyncPlan(routines=[rs]), {None: lines})
    got = {(ln, tuple(new)) for ln, _, new in by_file[None]}
    assert (9, ("  " + acc_sync.TAG + ": device data this host code uses",
                "  !$acc update self(x) if_present")) in got
    assert (12, ("  !$acc update device(y) if_present",)) in got
    assert (18, ("     !$acc update device(y) if_present",)) in got


COLD = """subroutine check(n, x, nbad)
  implicit none
  integer, intent(in) :: n
  real(r8), intent(inout) :: x(:)
  integer, intent(inout) :: nbad
  integer :: i
  !$acc parallel loop default(present) reduction(+:nbad)
  do i = 1, n
     if (x(i) < 0.0_r8) nbad = nbad + 1
  end do
  if (nbad > 0) then
     do i = 1, n
        if (x(i) < 0.0_r8) call endrun('negative x')
     end do
  end if
end subroutine check"""


def test_error_check_syncs_inside_the_branch():
    lines = COLD.split("\n")
    sub = StubSub(COLD)
    sub.active_global_vars = {}
    fut = SimpleNamespace(type_dict={}, module_dict={}, subroutine_dict={sub.id: sub})
    out = acc_sync.SyncPlan(device_written={("#dummy", sub.id, "x")})
    rs = acc_sync.RoutineSync(sub.id, None)
    acc_sync._segments(rs, sub, parse_statements(COLD).statements[0],
                       acc_sync._spans(find_directives(lines), sub.record),
                       acc_sync._entries(AccessMapper({sub.id: sub}).maps(sub)), lines, acc_sync._Types(fut), out)
    (seg,) = rs.segments  # the loop inside `if (nbad > 0)`, not the whole if construct
    assert (seg.first, seg.after) == (11, 14) and seg.stale == {"x"}
