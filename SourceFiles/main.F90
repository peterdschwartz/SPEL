program main

use shr_kind_mod, only: r8 => shr_kind_r8
use UpdateParamsAccMod, only: update_params_acc
use elm_varctl
use filterMod
!!use decompMod, only: get_clump_bounds_gpu, gpu_clumps, gpu_procinfo, init_proc_clump_info
use decompMod, only: get_proc_bounds, get_clump_bounds, procinfo, clumps
use ReadWriteMod, only : write_elmtypes, read_elmtypes
use decompMod, only: bounds_type
#ifdef _CUDA
use cudafor
#endif
use timeInfoMod
use elm_initializeMod
use nc_io, only: nc_read_timeslices, io_constants, io_inputs, io_outputs
!#USE_START

!=======================================!
implicit none
type(bounds_type)  ::  bounds_clump, bounds_proc
integer :: beg = 1, fin = 10, p, nclumps, nc
integer :: err
#if _CUDA
integer(kind=cuda_count_kind) :: heapsize, free1, free2, total
integer  :: istat, val
#endif
character(len=50) :: clump_input_char, pproc_input_char
integer :: nsets, pproc_input, fc, c, l, fp, g, j
integer :: begg, endg
integer :: time_len
real(r8) :: declin, declinp1
real :: startt, stopt
real(r8), allocatable :: icemask_dummy_arr(:)
!#VAR_DECL

!========================== Initialize/Allocate variables =======================!
!First, make sure the right number of inputs have been provided

IF (COMMAND_ARGUMENT_COUNT() == 0) THEN
   WRITE (*, *) 'ONE COMMAND-LINE ARGUMENT DETECTED, Defaulting to 1 site per clump'
   nsets = 1
   pproc_input = 1 !1 site per clump

elseIF (COMMAND_ARGUMENT_COUNT() == 1) THEN
   WRITE (*, *) 'ONE COMMAND-LINE ARGUMENT DETECTED, Defaulting to 1 site per clump'
   call get_command_argument(1, clump_input_char)
   READ (clump_input_char, *) nsets
   pproc_input = 1 !1 site per clump

ELSEIF (COMMAND_ARGUMENT_COUNT() == 2) THEN
   call get_command_argument(1, clump_input_char)
   call get_command_argument(2, pproc_input_char)
   READ (clump_input_char, *) nsets
   READ (pproc_input_char, *) pproc_input
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
END IF

block 
   character(len=256) :: input_path = "/home/mrgex/SPEL_Openacc/unit-tests/input-data/"
   call io_constants%init(base_fn=trim(input_path)//'spel-constants',max_tpf=720,read_io=.true.)
   call io_inputs%init(base_fn=trim(input_path)//'spel-inputs',max_tpf=720,read_io=.true.)
   call io_outputs%init(base_fn=trim(input_path)//'fut-outputs',max_tpf=720,read_io=.false.)
end block

call elm_init(nsets, pproc_input, dtime_mod, year_curr, bounds_proc)

declin = -0.4030289369547867

#ifdef _OPENACC
   call init_proc_clump_info()
   call update_params_acc()

   !Note: copy/paste enter data directives here for FUT.
   !      Will make this automatic in the future
   !#ACC_COPYIN

   call get_proc_bounds(bounds_proc)
   ! Calculate filters on device
#endif
!$acc enter data copyin( doalb, declinp1, declin )
!$acc update device(dtime_mod, dayspyr_mod, &
!$acc    year_curr, mon_curr, day_curr, secs_curr, nstep_mod, thiscalday_mod &
!$acc  , nextsw_cday_mod, end_cd_mod, doalb )


#ifdef _OPENACC
#define gpuflag 1
#else
#define gpuflag 0
#endif

nclumps = procinfo%nclumps

block
    integer :: step = 0
    integer :: max_step = 1000
    do while (.true.)

       if (step .ne. 0) then
          call read_elmtypes(io_inputs, bounds_proc)
       end if
       if (io_inputs%end_run) exit

       !$acc parallel loop independent gang vector default(present) private(bounds_clump)
       do nc = 1, nclumps
          call get_clump_bounds(nc, bounds_clump)
          !#CALL_SUB

       end do

       call write_elmtypes(io_outputs, bounds_proc)
       step = step + 1
    end do
end block

print *, "Finished: Read ", io_inputs%filenum-1, " files"

end Program main
