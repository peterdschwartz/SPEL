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


def test_call_tree_stops_at_excluded_routines():
    """A routine reachable only through an excluded (host) caller stays out of the tree;
    one also reached from a kept routine stays in."""
    from types import SimpleNamespace

    from spel.transforms.acc_apply import call_tree

    def sub(sid, *children):
        return SimpleNamespace(id=sid, child_subroutines={c: SimpleNamespace(id=c) for c in children})

    fut = SimpleNamespace(subroutine_dict={
        "d::driver": sub("d::driver", "u::set_old_weights", "d::hwcontent"),
        "u::set_old_weights": sub("u::set_old_weights", "t::template"),
        "t::template": sub("t::template", "t::template_one"),
        "t::template_one": sub("t::template_one"),
        "d::hwcontent": sub("d::hwcontent", "t::template_one"),
    })
    full = call_tree(fut, ["d::driver"])
    assert set(full) == set(fut.subroutine_dict)
    cut = call_tree(fut, ["d::driver"], exclude=lambda r: r.startswith("u::"))
    assert set(cut) == {"d::driver", "d::hwcontent", "t::template_one"}
    assert "t::template" not in cut  # only reachable through the host routine


GUARDED = """
subroutine upd(g0, g1, dt, use_fates, use_c13, use_cn)
  implicit none
  integer, intent(in) :: g0, g1
  real(r8), intent(in) :: dt
  logical, intent(in) :: use_fates, use_c13, use_cn
  integer :: g
  if (.not. use_fates) then
     do g = g0, g1
        col_pp%seed(g) = col_pp%seed(g) - dt
        if (use_c13) then
           veg_pp%seed(g) = veg_pp%seed(g) - dt
        end if
     end do
  end if
  if (use_c13 .and. use_cn) then
     do g = g0, g1
        veg_pp%seed(g) = 0.0_r8
     end do
  end if
end subroutine upd
"""


def test_namelist_guarded_instances_get_no_create():
    """An instance the device layer copies only under use_c13 .and. use_cn, named in a loop that
    runs under use_cn .and. .not. use_fates, is listed; the loop known to run under use_c13 is
    not; an instance always copied never is."""
    from types import SimpleNamespace

    from spel.scripts.fortran_parser.boolen_expression import AllOf, Expectation
    from spel.transforms.acc_apply import guarded_instances, implies, kernel_condition, reach_conditions

    sub = StubSub(GUARDED)
    first, second = sub.record.loop_nest
    c13, cn, fates = Expectation("use_c13", "True"), Expectation("use_cn", "True"), Expectation("use_fates", "False")
    conds = {"veg_pp": AllOf((c13, cn))}  # col_pp: always on the device
    assert guarded_instances(sub, first.start_ln, first.end_ln, conds, AllOf((cn, fates))) == ["veg_pp"]
    assert guarded_instances(sub, second.start_ln, second.end_ln, conds, AllOf((cn, fates, c13))) == []
    assert implies(AllOf((c13, cn)), AllOf((cn,))) and not implies(AllOf((cn,)), AllOf((c13, cn)))

    # the condition at a kernel: how elm_drv reaches the routine, and the if around the loop
    def routine(sid, calls=(), if_at=None):
        return SimpleNamespace(id=sid, sub_call_desc={ln: SimpleNamespace(fn=fn) for ln, fn in calls},
                               child_subroutines={fn: SimpleNamespace(id=f"m::{fn}") for _, fn in calls},
                               namelist_if_at=lambda ln, if_at=if_at: if_at(ln) if if_at else None)

    gate = SimpleNamespace(condtional_expectation=AllOf((cn,)))
    fut = SimpleNamespace(subroutine_dict={
        "elm_driver::elm_drv": routine("elm_driver::elm_drv", [(10, "eco"), (20, "alb")],
                                       if_at=lambda ln: gate if ln == 10 else None),
        "m::eco": routine("m::eco", [(5, "upd")]),
        "m::alb": routine("m::alb"),
        "m::upd": routine("m::upd"),
    })
    reach = reach_conditions(fut)
    assert reach["m::upd"] == AllOf((cn,)) or implies(reach["m::upd"], AllOf((cn,)))
    assert reach["m::alb"] == AllOf(())
    assert implies(kernel_condition(fut.subroutine_dict["m::upd"], 0, reach), AllOf((cn,)))


def test_no_create_clause_in_the_directive():
    from spel.transforms.acc_plan import LoopProposal

    sub = StubSub(GUARDED)
    props = propose(sub, AccessMapper({sub.id: sub}), source=GUARDED.splitlines(), min_lines=1)
    prop = next(p for p in props if p.plan is not None)
    prop.no_create = ["c14_grc_cs"]
    text = " ".join(prop.directive_lines())
    assert "no_create(c14_grc_cs)" in text and "default(present)" in text


def test_excluded_lines_come_from_the_record():
    """A loop with a stand-in statement (`x = x`, the analysis' replacement of a statement that
    used an excluded module) is device-unsafe even without the marker comment in sub_lines."""
    from types import SimpleNamespace

    from spel.transforms.acc_apply import _Facts

    src = """subroutine interp(g0, g1, out)
  implicit none
  integer, intent(in) :: g0, g1
  real(r8), intent(inout) :: out(:)
  integer :: g, ig
  ig = 0
  do g = g0, g1
     ig = ig + 1
     out(g) = out(g)
  end do
end subroutine interp"""
    sub = StubSub(src)
    sub.sub_lines = [SimpleNamespace(ln=i, line=l) for i, l in enumerate(src.splitlines())]
    sub.startline = 0
    facts = object.__new__(_Facts)
    facts.tree = lambda s: (_ for _ in ()).throw(RuntimeError("no tree"))
    assert facts.excluded_lines(sub) == [8]
