"""Unused dummies removed with their actuals (drop_dummies)."""
from spel.transforms.core.testing import check_transform
from spel.transforms.drop_dummies import DropDummies

BEFORE = """\
module snippet_mod
  implicit none
  integer, parameter :: r8 = 8
  type bounds_type
     integer :: begc, endc
  end type bounds_type
contains
  subroutine outer(bounds, num_f, filter_f, t)
    type(bounds_type), intent(in) :: bounds
    integer, intent(in) :: num_f
    integer, intent(in) :: filter_f(:)
    real(r8), intent(inout) :: t(bounds%begc:)
    call solve(bounds, num_f, filter_f, t)
    call solve(num_filter=num_f, filter=filter_f, t=t, &
               bounds=bounds)
    call uses(bounds, t)
  end subroutine outer

  subroutine solve(bounds, num_filter, filter, t)
    type(bounds_type), intent(in) :: bounds   ! no longer read
    integer, intent(in) :: num_filter
    integer, intent(in) :: filter(:)
    real(r8), intent(inout) :: t(:)
    integer :: fc
    do fc = 1, num_filter
       t(filter(fc)) = 0._r8
    end do
  end subroutine solve

  subroutine uses(bounds, x)
    type(bounds_type), intent(in) :: bounds
    real(r8), intent(inout) :: x(bounds%begc:)
    x(bounds%begc) = 1._r8
  end subroutine uses
end module snippet_mod
"""

AFTER = (BEFORE
         .replace("    call solve(bounds, num_f, filter_f, t)", "    call solve(num_f, filter_f, t)")
         .replace("    call solve(num_filter=num_f, filter=filter_f, t=t, &\n               bounds=bounds)",
                  "    call solve(num_filter=num_f, filter=filter_f, t=t &\n               )")
         .replace("  subroutine solve(bounds, num_filter, filter, t)", "  subroutine solve(num_filter, filter, t)")
         .replace("    type(bounds_type), intent(in) :: bounds   ! no longer read\n", ""))


def test_drop_unused_bounds(tmp_path):
    # solve's bounds goes (with both actuals); outer's and uses' stay (used in declarations/statements)
    check_transform(DropDummies, BEFORE, AFTER, tmp_path)
