module constants_mod

  use spmdMod, only : bad1, &
        bad2
   use shr_const_mod, only : test_type

  integer, parameter :: i_const = 5
  integer, parameter :: nlevdecomp_full = 10
  real :: param1(1:nlevdecomp_full)
  real, allocatable :: param2(:)
  real, allocatable :: notused(:)

  logical :: use_fates = .true.
  type(test_type), public :: unused_inst
  contains

   subroutine get_elmlevel_gsmap (elmlevel, gsmap)
     !
     ! !DESCRIPTION:
     ! Compute arguments for gatherv, scatterv for vectors
     !
     ! !ARGUMENTS:
     character(len=*), intent(in) :: elmlevel     ! type of input data
     type(mct_gsmap) , pointer    :: gsmap
     !----------------------------------------------------------------------

    select case (elmlevel)
    case(grlnd)
       gsmap => gsMap_lnd_gdc2glo
    case(nameg)
       gsmap => gsMap_gce_gdc2glo
    case(namet)
       gsmap => gsMap_top_gdc2glo
    case(namel)
       gsmap => gsMap_lun_gdc2glo
    case(namec)
       gsmap => gsMap_col_gdc2glo
    case(namep)
       gsmap => gsMap_patch_gdc2glo
    case(nameCohort)
       gsmap => gsMap_cohort_gdc2glo
    case default
       write(iulog,*) 'get_elmlevel_gsmap: Invalid expansion character: ',trim(elmlevel)
       call shr_sys_abort()
    end select

  end subroutine get_elmlevel_gsmap

  subroutine init_params()
    allocate(     notused    (1:i_const), param2      (1:i_const))
    if (bad1) write (*,*) "blah blah ! " &
       // "blah blah too"
  end subroutine init_params

end module constants_mod
