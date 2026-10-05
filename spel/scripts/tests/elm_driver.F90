module elm_driver
  ! Mock land-model driver: elm_drv is the top-most parent of unit-test roots
  use shr_kind_mod   , only : r8 => shr_kind_r8
  use decompMod      , only : bounds_type
  use constants_mod  , only : unused_inst
  use elm_instMod    , only : patch_state_updater
  use test_sub_parse , only : call_sub
  implicit none

contains

  subroutine elm_drv(nc)
    integer, intent(in) :: nc
    type(bounds_type) :: bounds_clump

    call call_sub(nc, bounds_clump, unused_inst, patch_state_updater)
  end subroutine elm_drv

end module elm_driver
