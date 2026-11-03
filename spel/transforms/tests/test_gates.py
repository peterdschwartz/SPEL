"""spel gates: transform gates pass when the transform has nothing left to do."""
from spel.transforms import gates as G
from spel.transforms.core.testing import snippet_context

SRC = """\
module snippet_mod
  implicit none
  integer, parameter :: r8 = 8
contains
  subroutine owner(n, x)
    integer, intent(in) :: n
    real(r8), intent(inout) :: x(:)
    integer :: i
    !$acc parallel loop gang vector default(present)
    do i = 1, n
       call helper(x(i))
    end do
  end subroutine owner

  subroutine helper(v)
    real(r8), intent(inout) :: v
    v = v + 1._r8
  end subroutine helper
end module snippet_mod
"""


def test_acc_routines_gate(tmp_path):
    ctx = snippet_context(SRC, tmp_path, "snippet_mod.F90")
    (res,) = G.run(ctx, only=["acc-routines"])
    assert not res.ok and "helper" in res.findings[0]          # helper lacks `routine seq`
    fixed = SRC.replace("    real(r8), intent(inout) :: v\n    v = v", "    real(r8), intent(inout) :: v\n    !$acc routine seq\n    v = v")
    ctx = snippet_context(fixed, tmp_path / "b", "snippet_mod.F90")
    (res,) = G.run(ctx, only=["acc-routines"])
    assert res.ok, res
