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


REORDER = """
subroutine reorder(num, filter, nlev, ntrans, x, tot, hr)
  implicit none
  integer, intent(in) :: num, filter(:), nlev, ntrans
  real(r8), intent(in) :: x(:,:)
  real(r8), intent(inout) :: tot(:,:), hr(:,:)
  integer :: fc, c, j, pi, p, k
  do j = 1, nlev
     do pi = 1, 4
        do fc = 1, num
           c = filter(fc)
           if (pi <= col_pp%npfts(c)) then
              p = col_pp%pfti(c) + pi - 1
              tot(c,j) = tot(c,j) + x(p,j)
           end if
        end do
     end do
  end do
  do k = 1, ntrans
     do j = 1, nlev
        do fc = 1, num
           c = filter(fc)
           hr(c,k) = hr(c,k) + x(c,j)
        end do
     end do
  end do
end subroutine reorder
"""


def test_parallel_loops_first():
    # GapPftToColumn and col_cf_summary_for_ch4: parallelizing only the outer (level or
    # transition) loop runs every column in one thread; the carried loop goes inside instead
    lines = REORDER.splitlines()
    gap, summ = _props(REORDER)
    assert (gap.kind, gap.order, gap.npar) == ("interchange", ["j", "fc", "pi"], 2), gap
    assert (summ.kind, summ.order, summ.npar) == ("interchange", ["k", "fc", "j"], 2), summ
    for prop in (summ, gap):  # bottom-up, as apply() inserts
        port = Port("reorder", prop, interchange=True)
        _swap(port, lines)
        for x in sorted(_loop_edits(port, lines), key=lambda x: (-x.ln, -x.order)):
            lines[x.ln:x.ln] = x.lines
    code = [l.strip() for l in lines if l.strip().startswith(("do ", "!$acc", "end do"))]
    assert code[1] == "!$acc   if(elm_device_ready)"  # the device data layer's flag, on every generated directive
    assert code[2:5] == ["do j = 1, nlev", "do fc = 1, num", "!$acc loop seq"], code
    assert "collapse(2)" in code[0]
    assert code[5] == "do pi = 1, 4"
    k = code.index("do k = 1, ntrans")
    assert code[k + 1:k + 4] == ["do fc = 1, num", "!$acc loop seq", "do j = 1, nlev"]
    assert "collapse(2)" in code[k - 2]  # the directive; its if(elm_device_ready) continuation follows


MEMBERS_INSIDE = """
subroutine litter(num, filter, nlev, x, tot)
  implicit none
  integer, intent(in) :: num, filter(:), nlev
  real(r8), intent(in) :: x(:,:)
  real(r8), intent(inout) :: tot(:,:)
  integer :: fc, c, j, p
  do j = 1, nlev
     do fc = 1, num
        c = filter(fc)
        do p = col_pp%pfti(c), col_pp%pftf(c)
           if (.not. veg_pp%active(p)) cycle
           tot(c,j) = tot(c,j) + x(p,j)
        end do
     end do
  end do
end subroutine litter
"""


def test_nest_collapsed_above_a_carried_member_loop():
    # CNLitterToColumn as a gather: (j, fc) collapsed, each column's patches in order inside
    (prop,) = _props(MEMBERS_INSIDE)
    assert prop.kind == "elementwise" and [lp.index for lp in prop.nest] == ["j", "fc"], prop
    assert "collapse(2)" in prop.directive_lines()[0]


OUTER_CARRIED = """
subroutine hrvr(num, filter, nlev, ntrans, x, hr)
  implicit none
  integer, intent(in) :: num, filter(:), nlev, ntrans
  real(r8), intent(in) :: x(:,:,:)
  real(r8), intent(inout) :: hr(:,:)
  integer :: fc, c, j, k
  do k = 1, ntrans
     do j = 1, nlev
        do fc = 1, num
           c = filter(fc)
           hr(c,j) = hr(c,j) + x(c,j,k)
        end do
     end do
  end do
end subroutine hrvr
"""


def test_outer_carried_loop_goes_inside():
    # col_cf_summary_for_ch4's hr_vr: the two-loop swap would leave only j parallel
    (prop,) = _props(OUTER_CARRIED)
    assert (prop.kind, prop.order, prop.npar) == ("interchange", ["j", "fc", "k"], 2), prop


def test_existing_device_routines_are_recognized():
    from types import SimpleNamespace

    from spel.transforms.acc_apply import _Facts

    lines = [
        "pure subroutine r(g0, g1, a)\n",
        "  integer, intent(in) :: g0, g1\n",
        "  real, intent(out) :: a(*)\n",
        "  integer :: g\n",
        "  !$acc routine vector\n",
        "  !$acc loop vector\n",
        "  do g = g0, g1\n",
        "     a(g) = 0.\n",
        "  end do\n",
        "end subroutine r\n",
    ]
    facts = object.__new__(_Facts)
    facts._device = {}
    facts.lines = lambda sub: lines
    sub = SimpleNamespace(id="m::r", startline=0, endline=len(lines) - 1)
    assert facts.is_device_routine(sub)
    lines[4] = "  ! no directive\n"
    facts._device = {}
    assert not facts.is_device_routine(sub)


def test_character_intrinsics_in_loops_are_found():
    from types import SimpleNamespace

    from spel.transforms.acc_apply import _Facts
    from spel.scripts.fortran_parser.tests.test_scope_walk import parse_statements

    src = """
subroutine s(n, isotope, a)
  integer, intent(in) :: n
  character(len=*), intent(in) :: isotope
  real, intent(inout) :: a(n)
  integer :: p
  do p = 1, n
     if (trim(isotope) == 'bulk') then
        a(p) = 0.
     end if
  end do
end subroutine s
"""
    tree = parse_statements(src).statements[0]
    facts = object.__new__(_Facts)
    facts.tree = lambda sub: tree
    lines = facts.string_intrinsic_lines(SimpleNamespace(id="m::s", startline=0, endline=11))
    if_ln = next(i for i, l in enumerate(src.splitlines()) if "trim(isotope)" in l)
    assert lines == [if_ln]
