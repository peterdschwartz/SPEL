module test_types

  type, public :: column_nitrogen_flux 
    ! harvest fluxes
    real(r8), pointer :: hrv_deadstemn_to_prod10n              (:)     => null() ! dead stem N harvest mortality to 10-year product pool (gN/m2/s)
    real(r8), pointer :: hrv_deadstemn_to_prod100n             (:)     => null() ! dead stem N harvest mortality to 100-year product pool (gN/m2/s)
    real(r8), pointer :: m_n_to_litr_met_fire                  (:,:)   => null() ! N from leaf, froot, xfer and storage N to litter labile N by fire (gN/m3/s)
    contains
      procedure, public :: Init       => col_nf_init
      procedure, public :: SetValues  => col_nf_setvalues
      procedure, public :: type_stuff
  end type

contains

  subroutine col_nf_init(this, begc, endc)
    !
    ! !ARGUMENTS:
    class(column_nitrogen_flux) :: this
    integer, intent(in) :: begc,endc

    allocate(this%hrv_deadstemn_to_prod10n        (begc:endc))                   ; this%hrv_deadstemn_to_prod10n       (:)   = spval
    allocate(this%hrv_deadstemn_to_prod100n       (begc:endc))                   ; this%hrv_deadstemn_to_prod100n      (:)   = spval
    allocate(this%m_n_to_litr_met_fire            (begc:endc,1:nlevdecomp_full)) ; this%m_n_to_litr_met_fire           (:,:) = spval
  end subroutine col_nf_init

end module test_types
