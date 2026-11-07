"""Stale names in OpenACC data clauses (acc_stale)."""
from spel.transforms.acc_stale import AccStale
from spel.transforms.core.testing import check_transform

BEFORE = """\
module snippet_mod
  implicit none
  integer, parameter :: r8 = 8
  real(r8) :: kept_var = 1._r8
  !$acc declare copyin(kept_var, gone_var)
  type t_type
     real(r8), pointer :: comp(:) => null()
  end type t_type
contains
  subroutine s(n, this, y)
    integer, intent(in) :: n
    type(t_type), intent(inout) :: this
    real(r8), intent(inout) :: y(:)
    real(r8) :: w(n)
    integer :: i
    !$acc enter data create(w, sum)
    !$acc update device(old_scalar)
    associate(c => this%comp)
    !$acc parallel loop gang vector present(c, old_arr(:,:)) private(i)
    do i = 1, n
       y(i) = c(i) + w(i) + kept_var
    end do
    end associate
  end subroutine s
end module snippet_mod
"""

AFTER = (BEFORE
         .replace("  !$acc declare copyin(kept_var, gone_var)", "  !$acc declare copyin(kept_var)")
         .replace("    !$acc enter data create(w, sum)", "    !$acc enter data create(w)")
         .replace("    !$acc update device(old_scalar)\n", "")
         .replace("present(c, old_arr(:,:)) private(i)", "present(c) private(i)"))


def test_stale_names(tmp_path):
    check_transform(AccStale, BEFORE, AFTER, tmp_path)
