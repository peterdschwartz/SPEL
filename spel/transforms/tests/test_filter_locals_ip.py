"""filter-locals across calls (filter_locals_ip): a work array passed down to a solver."""
from spel.transforms.core.testing import check_transform
from spel.transforms.filter_locals import FilterLocals

BEFORE = """\
module snippet_mod
  implicit none
  integer, parameter :: r8 = 8
  type bounds_type
     integer :: begc, endc
  end type bounds_type
contains
  subroutine outer(bounds, num_nolakec, filter_nolakec, t)
    type(bounds_type), intent(in) :: bounds
    integer, intent(in) :: num_nolakec
    integer, intent(in) :: filter_nolakec(:)
    real(r8), intent(inout) :: t(bounds%begc:)
    real(r8) :: tk(bounds%begc:bounds%endc)
    real(r8) :: other(bounds%begc:bounds%endc)
    integer :: c, fc, begc, endc
    begc = bounds%begc
    endc = bounds%endc
    do fc = 1, num_nolakec
       c = filter_nolakec(fc)
       tk(c) = t(c)
       other(c) = t(c)
    end do
    call solve(bounds, num_nolakec, filter_nolakec, tk(begc:endc), t)
    call keep(bounds, other)
  end subroutine outer

  subroutine solve(bounds, num_filter, filter, tk, t)
    type(bounds_type), intent(in) :: bounds
    integer, intent(in) :: num_filter
    integer, intent(in) :: filter(:)
    real(r8), intent(in) :: tk(bounds%begc:)
    real(r8), intent(inout) :: t(bounds%begc:)
    integer :: c, fc
    do fc = 1, num_filter
       c = filter(fc)
       t(c) = t(c) + tk(c)
    end do
  end subroutine solve

  subroutine keep(bounds, x)
    type(bounds_type), intent(in) :: bounds
    real(r8), intent(inout) :: x(bounds%begc:)
    integer :: c
    do c = bounds%begc, bounds%endc
       x(c) = 0._r8
    end do
  end subroutine keep
end module snippet_mod
"""

AFTER = (BEFORE
         .replace("    real(r8) :: tk(bounds%begc:bounds%endc)", "    real(r8) :: tk(num_nolakec)")
         .replace("       tk(c) = t(c)", "       tk(fc) = t(c)")
         .replace("filter_nolakec, tk(begc:endc), t)", "filter_nolakec, tk, t)")
         .replace("    real(r8), intent(in) :: tk(bounds%begc:)", "    real(r8), intent(in) :: tk(:)")
         .replace("       t(c) = t(c) + tk(c)", "       t(c) = t(c) + tk(fc)"))


def test_group_through_a_call(tmp_path):
    # tk: outer's local and solve's dummy convert together; other: keep() loops over the clump
    check_transform(FilterLocals, BEFORE, AFTER, tmp_path)


def test_a_second_caller_with_a_clump_array_blocks_the_group(tmp_path):
    src = BEFORE.replace("    call keep(bounds, other)\n",
                         "    call keep(bounds, other)\n    call solve(bounds, num_nolakec, filter_nolakec, t, t)\n")
    check_transform(FilterLocals, src, src, tmp_path)
