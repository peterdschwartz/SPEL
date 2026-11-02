"""acc-locals: device-only work arrays held in `data copy` regions get one create/delete instead."""
from spel.transforms.acc_locals import AccLocals
from spel.transforms.core.testing import check_transform

BEFORE = """\
module snippet_mod
contains
subroutine upd(begp, endp, nstep)
  integer, intent(in) :: begp, endp, nstep
  real(r8), pointer :: rbufslp(:)
  integer :: p
  !$acc update self(rbufslp) if_present
  allocate(rbufslp(begp:endp))
  !$acc data copy(rbufslp)
  !$acc parallel loop gang vector default(present)
  do p = begp, endp
     rbufslp(p) = 1._r8
  end do
  !$acc end data
  call update_accum_field('T10', rbufslp, nstep)
  call extract_accum_field('T10', x_patch, nstep)
  !$acc data copy(rbufslp)
  !$acc parallel loop gang vector default(present)
  do p = begp, endp
     rbufslp(p) = 2._r8
  end do
  !$acc end data
  call update_accum_field('T20', rbufslp, nstep)
  deallocate(rbufslp)
end subroutine upd
end module snippet_mod
"""

AFTER = """\
module snippet_mod
contains
subroutine upd(begp, endp, nstep)
  integer, intent(in) :: begp, endp, nstep
  real(r8), pointer :: rbufslp(:)
  integer :: p
  allocate(rbufslp(begp:endp))
  !$acc enter data create(rbufslp)
  !$acc parallel loop gang vector default(present)
  do p = begp, endp
     rbufslp(p) = 1._r8
  end do
  call update_accum_field('T10', rbufslp, nstep)
  call extract_accum_field('T10', x_patch, nstep)
  !$acc parallel loop gang vector default(present)
  do p = begp, endp
     rbufslp(p) = 2._r8
  end do
  call update_accum_field('T20', rbufslp, nstep)
  !$acc exit data delete(rbufslp)
  deallocate(rbufslp)
end subroutine upd
end module snippet_mod
"""


def test_device_only_buffer_is_created_once(tmp_path):
    check_transform(AccLocals, BEFORE, AFTER, tmp_path, procedures=["update_accum_field", "extract_accum_field"])


def test_a_host_read_keeps_the_copies(tmp_path):
    src = BEFORE.replace("  deallocate(rbufslp)\n", "  x = sum(rbufslp)\n  deallocate(rbufslp)\n")
    res = check_transform(AccLocals, src, src, tmp_path, procedures=["update_accum_field", "extract_accum_field"])
    assert not res.plan.changes
