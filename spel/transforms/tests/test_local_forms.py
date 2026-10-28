"""local_forms on a snippet: per-iteration arrays, constants, dead stores, and what disqualifies them."""
from spel.scripts.tests.test_record_access import StubSub
from spel.transforms.local_forms import check_routine

SRC = """
subroutine forms(num, filter, x, y)
  implicit none
  integer, intent(in) :: num, filter(:)
  real(r8), intent(inout) :: x(:,:), y(:,:)
  integer :: fc, c, j, k
  real(r8) :: tmp(num, 10)
  real(r8) :: carried(num, 10)
  real(r8) :: cond(num, 10)
  real(r8) :: kept(num)
  real(r8) :: ph(num)
  real(r8) :: half
  real(r8) :: unused(2)
  real(r8) :: rho = 1.e3_r8
  real(r8) :: lg
  ph(:) = 6.5_r8
  half = 0.5_r8
  lg = log(2.0_r8)
  unused(1) = 1.0_r8
  do j = 1, 10
     do fc = 1, num
        c = filter(fc)
        tmp(fc,j) = x(c,j) * rho
        y(c,j) = tmp(fc,j) + ph(fc) * half + lg
        if (x(c,j) > 0._r8) cond(fc,j) = 1._r8
        y(c,j) = y(c,j) + cond(fc,j)
        carried(fc,j) = carried(fc,j) + 1._r8
        kept(fc) = x(c,j)
     end do
  end do
  y(1,1) = kept(1)
end subroutine forms
"""


def test_findings():
    sub = StubSub(SRC)
    got = {(f.name, f.kind) for f in check_routine(sub)}
    assert got == {("tmp", "scalar"), ("ph", "parameter"), ("half", "parameter"), ("lg", "parameter"),
                   ("unused", "dead"), ("rho", "parameter")}, got
    # not: cond (conditional write before the read), carried (read before written), kept (used after the loop)
    lg = [f for f in check_routine(sub) if f.name == "lg"][0]
    assert "last bit" in lg.detail
