"""Loop shape proposals on a snippet: elementwise, interchange, scatter, column-parallel."""
from spel.scripts.record_access import AccessMapper
from spel.scripts.tests.test_record_access import StubSub
from spel.transforms.acc_plan import propose

SRC = """
subroutine demo(num, filter, numv, filterp, colof, x, y, tot)
  implicit none
  integer, intent(in) :: num, filter(:), numv, filterp(:), colof(:)
  real(r8), intent(inout) :: x(:,:), y(:), tot(:)
  integer :: fc, c, j, fp, p
  real(r8) :: tmp
  do j = 1, 10
     do fc = 1, num
        c = filter(fc)
        tmp = x(c,j) * 2.0_r8
        x(c,j) = tmp + y(c)
     end do
  end do
  do j = 1, 10
     do fc = 1, num
        c = filter(fc)
        tot(c) = tot(c) + x(c,j)
     end do
  end do
  do fp = 1, numv
     p = filterp(fp)
     c = colof(p)
     y(c) = y(c) + x(c,1)
  end do
  do fc = 1, num
     c = filter(fc)
     tot(c) = 0.0_r8
     do j = 1, 10
        tot(c) = tot(c) + x(c,j)
        if (tot(c) < 0.0_r8) call endrun('neg')
     end do
  end do
end subroutine demo
"""


def test_shapes():
    sub = StubSub(SRC)
    sub.filepath = None
    mapper = AccessMapper({sub.id: sub})
    props = propose(sub, mapper, source=SRC.splitlines())
    kinds = [p.kind for p in props]
    assert kinds == ["elementwise", "interchange", "scatter", "column-parallel"], kinds
    e, i, s, cp = props
    assert len(e.nest) == 2 and "collapse(2)" in e.directive_lines()[0]
    assert "private(c, tmp)" in e.directive_lines()[0] or "private(c,tmp)" in e.directive_lines()[0].replace(" ", "")
    assert i.carried == {"j": ["tot"]}
    assert s.scatter == {"fp": ["y"]}
    assert cp.directive_lines()[-1].strip().startswith("!$acc loop seq")
    assert cp.unsafe and "endrun" in cp.unsafe[0][1]


SRC_LOOKUP = """
subroutine cascade(num, filter, ntrans, receiver, flux, pools)
  implicit none
  integer, intent(in) :: num, filter(:), ntrans, receiver(:)
  real(r8), intent(in) :: flux(:,:,:)
  real(r8), intent(inout) :: pools(:,:,:)
  integer :: fc, c, j, k
  do k = 1, ntrans
     do j = 1, 10
        do fc = 1, num
           c = filter(fc)
           pools(c,j,receiver(k)) = pools(c,j,receiver(k)) + flux(c,j,k)
        end do
     end do
  end do
end subroutine cascade
"""


def test_index_through_a_lookup_is_not_injective():
    # several k may share receiver(k): parallel over k would race (and reorder the sums)
    sub = StubSub(SRC_LOOKUP)
    sub.filepath = None
    mapper = AccessMapper({sub.id: sub})
    (p,) = propose(sub, mapper, source=SRC_LOOKUP.splitlines())
    assert p.kind not in ("elementwise", "column-parallel"), p.kind
    assert "pools" in p.scatter.get("k", []) + p.carried.get("k", []) + p.same_value.get("k", [])


SRC_INNER_SCATTER = """
subroutine litter(nump, filterp, colof, wt, prof, x, lit)
  implicit none
  integer, intent(in) :: nump, filterp(:), colof(:)
  real(r8), intent(in) :: wt(:), prof(:,:), x(:)
  real(r8), intent(inout) :: lit(:,:)
  integer :: fp, p, c, j
  real(r8) :: wt_col
  do j = 1, 10
     do fp = 1, nump
        p = filterp(fp)
        c = colof(p)
        wt_col = wt(p)
        lit(c,j) = lit(c,j) + x(p) * wt_col * prof(p,j)
     end do
  end do
end subroutine litter
"""


def test_inner_scatter_is_not_collapsed():
    # patches of one column add into lit(c,j): collapse(2) over (j, fp) would race (CNLitterToColumn)
    sub = StubSub(SRC_INNER_SCATTER)
    sub.filepath = None
    (p,) = propose(sub, AccessMapper({sub.id: sub}), source=SRC_INNER_SCATTER.splitlines())
    assert p.scatter == {"fp": ["lit"]}
    assert p.kind == "column-parallel", p.kind
    assert p.directive_lines()[-1].strip().startswith("!$acc loop seq")


SRC_SWEEP = """
subroutine sweep(num, filter, a, r, conc, scaled)
  implicit none
  integer, intent(in) :: num, filter(:)
  real(r8), intent(in) :: a(:,:), r(:,:)
  real(r8), intent(inout) :: conc(:,0:), scaled(:,:)
  integer :: fc, c, j
  do j = 1, 10
     do fc = 1, num
        conc(fc,j) = r(fc,j) - a(fc,j) * conc(fc,j-1)
     end do
  end do
  do j = 1, 10
     do fc = 1, num
        c = filter(fc)
        scaled(c,j) = scaled(c,j) * 2.0_r8
     end do
  end do
end subroutine sweep
"""


def test_neighbour_element_read_is_carried():
    # forward sweep of a tridiagonal solve (SoilLittVertTransp): conc(fc,j) needs conc(fc,j-1)
    sub = StubSub(SRC_SWEEP)
    sub.filepath = None
    sweep, scale = propose(sub, AccessMapper({sub.id: sub}), source=SRC_SWEEP.splitlines())
    assert sweep.carried == {"j": ["conc"]} and sweep.kind == "interchange", (sweep.kind, sweep.carried)
    assert scale.kind == "elementwise" and not scale.carried  # the same element read and written


SRC_MEMBERS = """
subroutine members(num, filter, npfts, pfti, x, tot)
  implicit none
  integer, intent(in) :: num, filter(:)
  integer, intent(in) :: npfts(:), pfti(:)
  real(r8), intent(inout) :: x(:), tot(:)
  integer :: fc, c, pi, p
  do pi = 1, 4
     do fc = 1, num
        c = filter(fc)
        if (pi <= col_pp%npfts(c)) then
           p = col_pp%pfti(c) + pi - 1
           x(p) = 2.0_r8 * x(p)
           tot(c) = tot(c) + x(p)
        end if
     end do
  end do
  do pi = 1, 4
     do fc = 1, num
        c = filter(fc)
        p = col_pp%pfti(c) + pi - 1
        x(p) = 2.0_r8 * x(p)
     end do
  end do
end subroutine members
"""


def test_patch_of_column_by_member_offset():
    # p = pfti(c) + pi - 1 under `pi <= npfts(c)`: one patch per (column, pi), so the column
    # sums make it an interchange (columns parallel, pi in order); without the guard p may run
    # into the next column's patches
    sub = StubSub(SRC_MEMBERS)
    sub.filepath = None
    props = propose(sub, AccessMapper({sub.id: sub}), source=SRC_MEMBERS.splitlines())
    guarded, unguarded = props
    assert guarded.kind == "interchange", guarded
    assert guarded.carried == {"pi": ["tot"]}
    assert unguarded.kind not in ("elementwise", "interchange", "column-parallel"), unguarded


BOUNDS = """
subroutine zero(bounds, x)
  implicit none
  type(bounds_type), intent(in) :: bounds
  real(r8), intent(inout) :: x(:)
  integer :: p
  do p = bounds%begp, bounds%endp
     x(p) = 0._r8
  end do
end subroutine zero
"""


def test_bounds_dummy_read_in_the_loop_is_copied_in():
    """A kernel that reads the `bounds` dummy needs copyin(bounds): a caller whose bounds variable is
    not on the device otherwise fails with "data in PRESENT clause was not found: bounds"."""
    sub = StubSub(BOUNDS)
    sub.filepath = None
    (p,) = propose(sub, AccessMapper({sub.id: sub}), source=BOUNDS.splitlines())
    assert p.kind == "elementwise" and p.copyin == ["bounds"]
    assert any("copyin(bounds)" in l for l in p.directive_lines())
