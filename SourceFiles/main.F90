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
use nc_io, only: nc_read_timeslices
!#USE_START

!=======================================!
implicit none
type(bounds_type)  ::  bounds_clump, bounds_proc
integer :: beg = 1, fin = 10, p, nclumps, nc, step
integer :: err
#if _CUDA
integer(kind=cuda_count_kind) :: heapsize, free1, free2, total
integer  :: istat, val
#endif
character(len=50) :: clump_input_char, pproc_input_char
integer :: clump_input, pproc_input, fc, c, l, fp, g, j
logical :: found_thawlayer
integer :: k_frz
integer :: begg, endg
character(len=256) :: const_fn, var_fn
integer :: time_len
real(r8) :: declin, declinp1
real :: startt, stopt
real(r8), allocatable :: icemask_dummy_arr(:)
!#VAR_DECL

!========================== Initialize/Allocate variables =======================!
!First, make sure the right number of inputs have been provided

IF (COMMAND_ARGUMENT_COUNT() == 0) THEN
   WRITE (*, *) 'ONE COMMAND-LINE ARGUMENT DETECTED, Defaulting to 1 site per clump'
   clump_input = 1
   pproc_input = 1 !1 site per clump

elseIF (COMMAND_ARGUMENT_COUNT() == 1) THEN
   WRITE (*, *) 'ONE COMMAND-LINE ARGUMENT DETECTED, Defaulting to 1 site per clump'
   call get_command_argument(1, clump_input_char)
   READ (clump_input_char, *) clump_input
   pproc_input = 1 !1 site per clump

ELSEIF (COMMAND_ARGUMENT_COUNT() == 2) THEN
   call get_command_argument(1, clump_input_char)
   call get_command_argument(2, pproc_input_char)
   READ (clump_input_char, *) clump_input
   READ (pproc_input_char, *) pproc_input
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
END IF

const_fn = "spel-constants.nc"
var_fn = "spel-elmtypes.nc"
call elm_init(const_fn, var_fn, clump_input, pproc_input, dtime_mod, year_curr, bounds_proc)

time_len = nc_read_timeslices(var_fn)
print *, "Number of time slices found: ", time_len

declin = -0.4030289369547867
step = 0

#ifdef _OPENACC
if (step == 0) then
   print *, "transferring data to GPU"
   call init_proc_clump_info()
#ifdef _CUDA
   istat = cudaMemGetInfo(free1, total)
   print *, "Free1:", free1
#endif
   call update_params_acc()

   !Note: copy/paste enter data directives here for FUT.
   !      Will make this automatic in the future
   !#ACC_COPYIN

   call get_proc_bounds(bounds_proc)
   ! Calculate filters on device
   !call setProcFilters(bounds_proc, proc_filter, .false., icemask_dummy_arr)

#if _CUDA
   ! Heap Limit may need to be increased for certain routines
   ! if using routine directives with many automatic arrays
   ! should be adjusted based on problem size
   istat = cudaDeviceGetLimit(heapsize, cudaLimitMallocHeapSize)
   print *, "SETTING Heap Limit from", heapsize
   heapsize = 10_8*1024_8*1024_8
   print *, "TO:", heapsize
   istat = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapsize)
   istat = cudaMemGetInfo(free1, total)
   print *, "Free1:", free1/1.E+9
#endif
end if
#endif
!$acc enter data copyin( doalb, declinp1, declin )
!$acc update device(dtime_mod, dayspyr_mod, &
!$acc    year_curr, mon_curr, day_curr, secs_curr, nstep_mod, thiscalday_mod &
!$acc  , nextsw_cday_mod, end_cd_mod, doalb )

! Note: should add these to writeConstants in the future (as arguments?)
!$acc serial
declin = -0.4023686267583503
declinp1 = -0.4023686267583503
!$acc end serial

#ifdef _OPENACC
#define gpuflag 1
#else
#define gpuflag 0
#endif

nclumps = procinfo%nclumps

do step = 1, time_len

   call get_clump_bounds(1, bounds_clump)
   call read_elmtypes(1, step, trim(var_fn), bounds_clump)

!$acc parallel loop independent gang vector default(present) private(bounds_clump)
do nc = 1, nclumps
   call get_clump_bounds(nc, bounds_clump)
!#CALL_SUB

end do

call write_elmtypes(1,step,"fut-results.nc", bounds_clump)
end do

#if _CUDA
istat = cudaMemGetInfo(free1, total)
print *, "free after kernel:", free1/1.E+9
#endif
print *, "done with unit-test execution"
deallocate (clumps, procinfo%cid)
deallocate (filter, filter_inactive_and_active)

end Program main
