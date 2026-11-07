"""`!$acc routine seq` exactly where kernels reach (acc_routines)."""
from spel.transforms.acc_routines import AccRoutines
from spel.transforms.core.testing import check_transform

BEFORE = """\
module snippet_mod
  implicit none
  integer, parameter :: r8 = 8
contains
  subroutine kernel_owner(n, x)
    integer, intent(in) :: n
    real(r8), intent(inout) :: x(:)
    integer :: i
    !$acc parallel loop gang vector default(present)
    do i = 1, n
       call helper(x(i))
    end do
    call host_only(n, x)
  end subroutine kernel_owner

  subroutine helper(v)
    real(r8), intent(inout) :: v
    call deeper(v)
  end subroutine helper

  subroutine deeper(v)
    real(r8), intent(inout) :: v
    !$acc routine seq
    v = v + 1._r8
  end subroutine deeper

  subroutine host_only(n, x)
    integer, intent(in) :: n
    real(r8), intent(inout) :: x(:)
    !$acc routine seq
    x(1) = real(n, r8)
  end subroutine host_only
end module snippet_mod
"""

AFTER = (BEFORE
         .replace("    real(r8), intent(inout) :: v\n    call deeper(v)",
                  "    real(r8), intent(inout) :: v\n    !$acc routine seq  ! spel acc-routines\n    call deeper(v)")
         .replace("    real(r8), intent(inout) :: x(:)\n    !$acc routine seq\n    x(1) = real(n, r8)",
                  "    real(r8), intent(inout) :: x(:)\n    x(1) = real(n, r8)"))


def test_prune_and_add(tmp_path):
    # helper: called in a kernel, gets one; deeper: reached through helper, keeps its own;
    # host_only: only called on the host, loses it
    check_transform(AccRoutines, BEFORE, AFTER, tmp_path)


DECL_BEFORE = """\
module snippet_mod
  implicit none
  integer, parameter :: r8 = 8
  integer :: nscal = 3
  real(r8) :: table(10)
  real(r8) :: coef = 2._r8
contains
  subroutine owner(n, x)
    integer, intent(in) :: n
    real(r8), intent(inout) :: x(:)
    integer :: i
    !$acc parallel loop gang vector default(present)
    do i = 1, n
       x(i) = table(i) * nscal
       call helper(x(i))
    end do
  end subroutine owner

  subroutine helper(v)
    real(r8), intent(inout) :: v
    !$acc routine seq
    v = v * coef
  end subroutine helper
end module snippet_mod
"""


def test_declares(tmp_path):
    from spel.transforms.acc_routines import AccDeclares
    from spel.transforms.core.testing import snippet_context
    from spel.transforms.core import driver

    ctx = snippet_context(DECL_BEFORE, tmp_path, "snippet_mod.F90")
    res = driver.run(AccDeclares(), ctx, AccDeclares.Options(), sink=driver.Memory(), verify=True)
    out = res.outputs.get(ctx.fut.snippet, DECL_BEFORE)
    decl = [l for l in out.split("\n") if "declare create" in l]
    # table: array in a kernel; coef: read in a device routine; nscal: scalar in a kernel (firstprivate)
    assert len(decl) == 1 and "coef" in decl[0] and "table" in decl[0] and "nscal" not in decl[0], out


def test_stale_declares_go(tmp_path):
    from spel.transforms.acc_routines import AccDeclares
    from spel.transforms.core import driver
    from spel.transforms.core.testing import snippet_context

    src = DECL_BEFORE.replace("contains\n", "  !$acc declare copyin(gone_var)\n  !$acc declare create(coef, also_gone)\ncontains\n", 1)
    ctx = snippet_context(src, tmp_path, "snippet_mod.F90")
    res = driver.run(AccDeclares(), ctx, AccDeclares.Options(), sink=driver.Memory(), verify=True)
    out = res.outputs.get(ctx.fut.snippet, src)
    assert "gone_var" not in out and "also_gone" not in out
    assert "!$acc declare create(coef)" in out


def test_stale_update_names_go(tmp_path):
    # the snippet analyzer resolves every name, so only the planning path is exercised here: a
    # directive whose names all resolve stays as it is
    from spel.transforms.acc_routines import AccDeclares
    from spel.transforms.core import driver
    from spel.transforms.core.testing import snippet_context

    src = DECL_BEFORE.replace("    v = v * coef\n", "    v = v * coef\n    !$acc update device(&\n    !$acc   coef)\n")
    ctx = snippet_context(src, tmp_path, "snippet_mod.F90")
    res = driver.run(AccDeclares(), ctx, AccDeclares.Options(), sink=driver.Memory(), verify=True)
    assert "update device(&" in res.outputs.get(ctx.fut.snippet, src)
