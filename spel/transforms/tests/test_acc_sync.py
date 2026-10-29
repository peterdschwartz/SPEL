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


EARLY_RETURN = """subroutine early(n, lake, x, y)
  implicit none
  integer, intent(in) :: n
  logical, intent(in) :: lake
  real(r8), intent(inout) :: x(:), y(:)
  integer :: i
  !$acc parallel loop default(present)
  do i = 1, n
     x(i) = 0.0_r8
  end do
  if (lake) return
  do i = 1, n
     x(i) = x(i) + y(i)
  end do
end subroutine early"""


def test_return_update_follows_the_update_self():
    # ch4_aere_columns: a kernel zeroes x, then `if (lake) return` starts the host segment;
    # the return's update device must not run before the segment's update self (it would
    # push stale host values over the kernel's zeros)
    lines = EARLY_RETURN.split("\n")
    sub = StubSub(EARLY_RETURN)
    sub.active_global_vars = {}
    fut = SimpleNamespace(type_dict={}, module_dict={}, subroutine_dict={sub.id: sub})
    spans = acc_sync._spans(find_directives(lines), sub.record)
    entries = acc_sync._entries(AccessMapper({sub.id: sub}).maps(sub))
    out = acc_sync.SyncPlan(device_written={("#dummy", sub.id, "x")})
    rs = acc_sync.RoutineSync(sub.id, None)
    node = parse_statements(EARLY_RETURN).statements[0]
    acc_sync._segments(rs, sub, node, spans, entries, lines, acc_sync._Types(fut), out)

    (seg,) = rs.segments
    assert seg.first == 10 and seg.returns == [10] and seg.stale == {"x"} and seg.writes == {"x"}
    ins = sorted(acc_sync.edits(acc_sync.SyncPlan(routines=[rs]), {None: lines})[None], key=lambda e: e[:2])
    at_return = [new[-1].strip() for ln, _, new in ins if ln == 10]
    assert at_return == ["!$acc update self(x) if_present", "!$acc update device(x) if_present"]


FATES_BRANCH = """subroutine branchy(n, use_fates, x, y)
  implicit none
  integer, intent(in) :: n
  logical, intent(in) :: use_fates
  real(r8), intent(inout) :: x(:), y(:)
  integer :: i
  !$acc parallel loop default(present)
  do i = 1, n
     x(i) = 1.0_r8
  end do
  if (use_fates) then
     do i = 1, n
        y(i) = x(i)
     end do
  end if
end subroutine branchy"""


def test_off_by_default_branch_syncs_inside():
    # ch4_aere_columns: the host FATES branch copied every input to the host on each call
    lines = FATES_BRANCH.split("\n")
    sub = StubSub(FATES_BRANCH)
    sub.active_global_vars = {}
    fut = SimpleNamespace(type_dict={}, module_dict={}, subroutine_dict={sub.id: sub})
    spans = acc_sync._spans(find_directives(lines), sub.record)
    entries = acc_sync._entries(AccessMapper({sub.id: sub}).maps(sub))
    out = acc_sync.SyncPlan(device_written={("#dummy", sub.id, "x")})
    rs = acc_sync.RoutineSync(sub.id, None)
    node = parse_statements(FATES_BRANCH).statements[0]
    acc_sync._segments(rs, sub, node, spans, entries, lines, acc_sync._Types(fut), out)
    (seg,) = rs.segments
    assert seg.first == 11 and seg.stale == {"x"} and seg.writes == {"y"}   # the do loop inside the branch


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


HOST_ONLY = """subroutine balance(n, x, y)
  implicit none
  integer, intent(in) :: n
  real(r8), intent(in) :: x(:)
  real(r8), intent(inout) :: y(:)
  integer :: i
  do i = 1, n
     y(i) = x(i) - y(i)
  end do
end subroutine balance"""


def test_host_only_root_is_one_host_segment():
    lines = HOST_ONLY.split("\n")
    sub = StubSub(HOST_ONLY)
    sub.active_global_vars = {}
    fut = SimpleNamespace(type_dict={}, module_dict={}, subroutine_dict={sub.id: sub})
    entries = acc_sync._entries(AccessMapper({sub.id: sub}).maps(sub))
    # x and y: written on the device by the routines before it (another root's kernels)
    out = acc_sync.SyncPlan(device_written={("#dummy", sub.id, "x"), ("#dummy", sub.id, "y")})
    rs = acc_sync.RoutineSync(sub.id, None)
    node = parse_statements(HOST_ONLY).statements[0]
    acc_sync._segments(rs, sub, node, [], entries, lines, acc_sync._Types(fut), out)
    (seg,) = rs.segments
    assert (seg.first, seg.after) == (6, 9)
    assert seg.stale == {"x", "y"} and seg.writes == {"y"}


def test_apply_twice_inserts_once(tmp_path):
    path = tmp_path / "m.F90"
    path.write_text(SRC)
    lines = SRC.split("\n")
    sub = StubSub(SRC)
    sub.active_global_vars = {}
    fut = SimpleNamespace(type_dict={}, module_dict={}, subroutine_dict={sub.id: sub})
    entries = acc_sync._entries(AccessMapper({sub.id: sub}).maps(sub))
    out = acc_sync.SyncPlan(device_written={("#dummy", sub.id, "x")})
    rs = acc_sync.RoutineSync(sub.id, path)
    acc_sync._segments(rs, sub, parse_statements(SRC).statements[0], acc_sync._spans(find_directives(lines), sub.record),
                       entries, lines, acc_sync._Types(fut), out)
    plan_ = acc_sync.SyncPlan(routines=[rs])
    acc_sync.apply(fut, plan_)
    once = path.read_text()
    assert once != SRC
    acc_sync.apply(fut, plan_)   # the same plan again (line numbers of the unedited source)
    assert path.read_text().count("update self(x)") == once.count("update self(x)")


def test_dedupe_sync_blocks():
    tag = f"  {acc_sync.TAG}: device data this host code uses"
    upd = ["  !$acc update self(a, &", "  !$acc&   b) if_present"]
    lines = ["  x = 1", tag, *upd, tag, tag, *upd, "  !$acc update device(c) if_present",
             "  !$acc update device(c) if_present", "  !$acc update self(d)", "  !$acc update self(d)", "  y = 2"]
    assert acc_sync.dedupe_sync_blocks(lines) == [
        "  x = 1", tag, *upd, "  !$acc update device(c) if_present",
        "  !$acc update self(d)", "  !$acc update self(d)", "  y = 2"]   # hand-written updates are left alone


def test_boundary_segments_and_cond():
    """Accesses outside the routine (elm_drv's coupler): push what is written before at the first
    statement, pull what is read after (and device-written) before `end subroutine`."""
    lines = SRC.split("\n")
    sub = StubSub(SRC)
    sub.active_global_vars = {}
    fut = SimpleNamespace(type_dict={}, module_dict={}, subroutine_dict={sub.id: sub})
    types = acc_sync._Types(fut)
    spans = acc_sync._spans(find_directives(lines), sub.record)
    entries = acc_sync._entries(AccessMapper({sub.id: sub}).maps(sub))
    out = acc_sync.SyncPlan(device_written={("#dummy", sub.id, "x")})
    rs = acc_sync.RoutineSync(sub.id, None)
    node = parse_statements(SRC).statements[0]
    acc_sync._segments(rs, sub, node, spans, entries, lines, types, out,
                       boundary=([("y", "w"), ("x", "r")], [("x", "r"), ("y", "r")]))
    start, end = rs.segments[0], rs.segments[-1]
    # the first statement is the kernel loop (line 6): the push goes above its directive
    assert (start.first, start.after, start.writes, start.stale) == (6, 6, {"y"}, set())
    # y is not device-written: only x is pulled at the end
    assert (end.first, end.header_only, end.stale) == (len(lines) - 1, True, {"x"})

    by_file = acc_sync.edits(acc_sync.SyncPlan(routines=[rs]), {None: lines}, cond=".not. full")
    got = {(ln, tuple(new)) for ln, _, new in by_file[None]}
    assert (5, ("  !$acc update device(y) if_present if(.not. full)",)) in got
    assert (len(lines) - 1, (acc_sync.TAG + ": device data this host code uses",
                             "!$acc update self(x) if_present if(.not. full)")) in got
    # the conditional updates are recognised as acc-sync blocks
    block = ["  !$acc update self(x) if_present if(.not. full)"] * 2
    assert acc_sync.dedupe_sync_blocks(block) == block[:1]


def test_targeted_strip_removes_an_application():
    from spel.transforms.device import targeted

    src = ["  call a()", f"  !$acc update self(p, &", f"  !$acc&   q) if_present if({targeted.COND})",
           f"  call elm_device_pull_all()  {targeted.TAG}", "  !$acc update self(r) if_present", "  call b()"]
    assert targeted.strip(src) == ["  call a()", "  !$acc update self(r) if_present", "  call b()"]


FLAG_BRANCH = """subroutine weights(n, x, y, updated)
  implicit none
  integer, intent(in) :: n
  real(r8), intent(inout) :: x(:), y(:)
  logical, intent(in) :: updated
  integer :: i
  !$acc parallel loop default(present)
  do i = 1, n
     x(i) = 1.0_r8
  end do
  if (updated) then
     do i = 1, n
        y(i) = x(i)
     end do
  end if
  if (x(1) > 0.0_r8) then
     y(1) = x(2)
  end if
end subroutine weights"""


def test_flag_guarded_branch_syncs_inside():
    """dynSubgrid_driver's `if (weights_updated)`: the condition reads a host flag, so the
    host loop inside syncs inside the branch (nothing moves when the weights did not change).
    A condition that reads device-written data keeps the whole construct as one segment."""
    lines = FLAG_BRANCH.split("\n")
    sub = StubSub(FLAG_BRANCH)
    sub.active_global_vars = {}
    fut = SimpleNamespace(type_dict={}, module_dict={}, subroutine_dict={sub.id: sub})
    out = acc_sync.SyncPlan(device_written={("#dummy", sub.id, "x")})
    rs = acc_sync.RoutineSync(sub.id, None)
    acc_sync._segments(rs, sub, parse_statements(FLAG_BRANCH).statements[0],
                       acc_sync._spans(find_directives(lines), sub.record),
                       acc_sync._entries(AccessMapper({sub.id: sub}).maps(sub)), lines, acc_sync._Types(fut), out)
    firsts = sorted((s.first, s.after, s.header_only) for s in rs.segments)
    assert (11, 14, False) in firsts          # the loop inside `if (updated)`, not the if at line 10
    assert all(f != 10 for f, _, _ in firsts)
    assert (15, 18, False) in firsts          # `if (x(1) > 0)`: x is device-written, one segment


LOCAL_FILL = """subroutine divide(n, x, y)
  implicit none
  integer, intent(in) :: n
  real(r8), intent(in) :: x(:)
  real(r8), intent(inout) :: y(:)
  real(r8) :: work(n)
  integer :: i
  !$acc enter data create(work)
  do i = 1, n
     work(i) = x(i) * 2.0_r8
  end do
  !$acc parallel loop default(present)
  do i = 1, n
     y(i) = work(i)
  end do
  !$acc exit data delete(work)
end subroutine divide"""


def test_local_array_filled_on_the_host_is_pushed():
    """DivideSnowLayers: dzsno, swice, ... filled by a host loop, read by the next kernel
    (created-but-unfilled device copies). The host segment pushes the local arrays it writes."""
    lines = LOCAL_FILL.split("\n")
    sub = StubSub(LOCAL_FILL)
    sub.active_global_vars = {}
    fut = SimpleNamespace(type_dict={}, module_dict={}, subroutine_dict={sub.id: sub})
    out = acc_sync.SyncPlan(device_written={("#dummy", sub.id, "y")})
    rs = acc_sync.RoutineSync(sub.id, None)
    acc_sync._segments(rs, sub, parse_statements(LOCAL_FILL).statements[0],
                       acc_sync._spans(find_directives(lines), sub.record),
                       acc_sync._entries(AccessMapper({sub.id: sub}).maps(sub)), lines, acc_sync._Types(fut), out)
    (seg,) = rs.segments
    assert seg.first == 8 and seg.writes == {"work"} and "i" not in seg.writes  # the scalar index is not synced
