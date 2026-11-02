"""filter-map: loops over a soil column's patches use the soilc -> soilp map (transforms/filter_map.py)."""
from spel.transforms.core.testing import check_transform
from spel.transforms.filter_map import FilterMapLoops, Options

BEFORE = """\
module snippet_mod
contains
subroutine drv()
  integer :: nc
  do nc = 1, nclumps
     call mid(filter_inactive_and_active(nc)%num_soilc, filter_inactive_and_active(nc)%soilc, &
          filter_inactive_and_active(nc)%num_soilp, filter_inactive_and_active(nc)%soilp)
  end do
end subroutine drv
subroutine mid(num_soilc, filter_soilc, num_soilp, filter_soilp)
  integer, intent(in) :: num_soilc, filter_soilc(:), num_soilp, filter_soilp(:)
  call leaf(num_soilc, filter_soilc, num_soilp, filter_soilp)
end subroutine mid
subroutine leaf(num_soilc, filter_soilc, num_soilp, filter_soilp)
  integer, intent(in) :: num_soilc
  integer, intent(in) :: filter_soilc(:)
  integer, intent(in) :: num_soilp, filter_soilp(:)
  integer :: fc, c, p, pi
  !$acc parallel loop gang vector default(present) private(c)
  do fc = 1, num_soilc
     c = filter_soilc(fc)
     !$acc loop seq
     do p = col_pp%pfti(c), col_pp%pftf(c)
        col_cf%gpp(c) = col_cf%gpp(c) + veg_cf%gpp(p) * veg_pp%wtcol(p)
     end do
  end do
  do fc = 1, num_soilc
     do pi = 1, maxpatch_pft
        c = filter_soilc(fc)
        if (pi <= col_pp%npfts(c)) then
           p = col_pp%pfti(c) + pi - 1
           col_cf%er(c) = col_cf%er(c) + veg_cf%er(p)
        end if
     end do
  end do
end subroutine leaf
end module snippet_mod
"""

AFTER = """\
module snippet_mod
contains
subroutine drv()
  integer :: nc
  do nc = 1, nclumps
     call mid(filter_inactive_and_active(nc)%num_soilc, filter_inactive_and_active(nc)%soilc, &
          filter_inactive_and_active(nc)%num_soilp, filter_inactive_and_active(nc)%soilp, &
               soilc_soilp_beg=filter_inactive_and_active(nc)%soilc_soilp_beg, soilc_soilp_end=filter_inactive_and_active(nc)%soilc_soilp_end)
  end do
end subroutine drv
subroutine mid(num_soilc, filter_soilc, num_soilp, filter_soilp, soilc_soilp_beg, soilc_soilp_end)
  integer, intent(in) :: num_soilc, filter_soilc(:), num_soilp, filter_soilp(:)
  integer, intent(in) :: soilc_soilp_beg(:)  ! the soil patches of soil column fc are
  integer, intent(in) :: soilc_soilp_end(:)  !   filter_soilp(soilc_soilp_beg(fc):soilc_soilp_end(fc))
  call leaf(num_soilc, filter_soilc, num_soilp, filter_soilp, &
       soilc_soilp_beg=soilc_soilp_beg, soilc_soilp_end=soilc_soilp_end)
end subroutine mid
subroutine leaf(num_soilc, filter_soilc, num_soilp, filter_soilp, soilc_soilp_beg, soilc_soilp_end)
  integer, intent(in) :: num_soilc
  integer, intent(in) :: filter_soilc(:)
  integer, intent(in) :: soilc_soilp_beg(:)  ! the soil patches of soil column fc are
  integer, intent(in) :: soilc_soilp_end(:)  !   filter_soilp(soilc_soilp_beg(fc):soilc_soilp_end(fc))
  integer :: fp_map  ! spel filter-map: position in the soil patch filter
  integer, intent(in) :: num_soilp, filter_soilp(:)
  integer :: fc, c, p, pi
  !$acc parallel loop gang vector default(present) private(c) &
  !$acc&   private(fp_map, p)
  do fc = 1, num_soilc
     c = filter_soilc(fc)
     !$acc loop seq
     do fp_map = soilc_soilp_beg(fc), soilc_soilp_end(fc)  ! spel filter-map
        p = filter_soilp(fp_map)
        col_cf%gpp(c) = col_cf%gpp(c) + veg_cf%gpp(p) * veg_pp%wtcol(p)
     end do
  end do
  do fc = 1, num_soilc
     do fp_map = soilc_soilp_beg(fc), soilc_soilp_end(fc)  ! spel filter-map
        c = filter_soilc(fc)
        ! (all patches of the range belong to the column)
           p = filter_soilp(fp_map)
           col_cf%er(c) = col_cf%er(c) + veg_cf%er(p)
     end do
  end do
end subroutine leaf
end module snippet_mod
"""


def test_loops_and_the_call_chain(tmp_path):
    res = check_transform(FilterMapLoops, BEFORE, AFTER, tmp_path, opts=Options(root="snippet_mod::drv"))
    assert not res.plan.refusals, res.plan.report()


def test_active_filter_without_guard_is_refused(tmp_path):
    src = BEFORE.replace("filter_inactive_and_active", "filter")
    res = check_transform(FilterMapLoops, src, src, tmp_path, opts=Options(root="snippet_mod::drv"))
    assert sum("not exact" in r.reason for r in res.plan.refusals) == 2, res.plan.report()


def test_a_routine_without_the_patch_filter_gets_it(tmp_path):
    before = """\
module snippet_mod
contains
subroutine drv()
  integer :: nc
  call mid(filter_inactive_and_active(nc)%num_soilc, filter_inactive_and_active(nc)%soilc)
end subroutine drv
subroutine mid(num_soilc, filter_soilc)
  integer, intent(in) :: num_soilc, filter_soilc(:)
  integer :: fc, c, p
  do fc = 1, num_soilc
     c = filter_soilc(fc)
     do p = col_pp%pfti(c), col_pp%pftf(c)
        col_cf%gpp(c) = col_cf%gpp(c) + veg_cf%gpp(p)
     end do
  end do
end subroutine mid
end module snippet_mod
"""
    after = """\
module snippet_mod
contains
subroutine drv()
  integer :: nc
  call mid(filter_inactive_and_active(nc)%num_soilc, filter_inactive_and_active(nc)%soilc, &
       soilc_soilp_beg=filter_inactive_and_active(nc)%soilc_soilp_beg, soilc_soilp_end=filter_inactive_and_active(nc)%soilc_soilp_end, filter_soilp=filter_inactive_and_active(nc)%soilp)
end subroutine drv
subroutine mid(num_soilc, filter_soilc, soilc_soilp_beg, soilc_soilp_end, filter_soilp)
  integer, intent(in) :: num_soilc, filter_soilc(:)
  integer, intent(in) :: soilc_soilp_beg(:)  ! the soil patches of soil column fc are
  integer, intent(in) :: soilc_soilp_end(:)  !   filter_soilp(soilc_soilp_beg(fc):soilc_soilp_end(fc))
  integer, intent(in) :: filter_soilp(:)  ! filter for soil patches
  integer :: fp_map  ! spel filter-map: position in the soil patch filter
  integer :: fc, c, p
  do fc = 1, num_soilc
     c = filter_soilc(fc)
     do fp_map = soilc_soilp_beg(fc), soilc_soilp_end(fc)  ! spel filter-map
        p = filter_soilp(fp_map)
        col_cf%gpp(c) = col_cf%gpp(c) + veg_cf%gpp(p)
     end do
  end do
end subroutine mid
end module snippet_mod
"""
    check_transform(FilterMapLoops, before, after, tmp_path, opts=Options(root="snippet_mod::drv"))
