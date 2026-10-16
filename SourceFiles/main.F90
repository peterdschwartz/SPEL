program main

use shr_kind_mod, only: r8 => shr_kind_r8
use UpdateParamsAccMod, only: update_params_acc
use elm_varctl
use filterMod
use decompMod, only: get_proc_bounds, get_clump_bounds, procinfo, clumps
use ReadWriteMod, only : write_elmtypes, read_elmtypes, spel_inbuf_to_device
use decompMod, only: bounds_type
#ifdef _CUDA
use cudafor
#endif
use timeInfoMod
use elm_initializeMod
use duplicateMod, only: duplicate_sets
use nc_io, only: nc_read_timeslices, spel_io_init, io_constants, io_inputs, io_outputs
use spel_profMod, only: spel_prof_push, spel_prof_pop, spel_env_flag, spel_env_int
!#USE_START

!=======================================!
implicit none
type(bounds_type)  ::  bounds_clump, bounds_proc, bounds_io
integer :: beg = 1, fin = 10, p, nclumps, nc
integer :: err
#if _CUDA
integer(kind=cuda_count_kind) :: heapsize, free1, free2, total
integer  :: istat, val
#endif
character(len=50) :: nsets_char
integer :: nsets, fc, c, l, fp, g, j
integer :: begg, endg
integer :: time_len
real(r8) :: declin, declinp1
real :: startt, stopt
real(r8), allocatable :: icemask_dummy_arr(:)
! SPEL_PROFILE=1 skips the output (device->host copy and write_elmtypes);
! SPEL_MAX_STEPS=N runs at most N time steps
logical :: spel_profile
integer :: max_step
!#VAR_DECL

!========================== Initialize/Allocate variables =======================!
!First, make sure the right number of inputs have been provided

! Optional argument: number of copies (sets) of the input sites.
! SPEL_CLUMPS=N splits the sets into N clumps of whole sets that run in an
! OpenMP loop over clumps, as in elm_drv (default: one clump spanning every set).
nsets = 1
IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
   call get_command_argument(1, nsets_char)
   READ (nsets_char, *) nsets
END IF
IF (COMMAND_ARGUMENT_COUNT() > 1) THEN
   WRITE (*, *) 'WARNING: extra command-line arguments ignored (only nsets is used)'
END IF
WRITE (*, *) 'nsets = ', nsets
spel_profile = spel_env_flag('SPEL_PROFILE')
max_step = spel_env_int('SPEL_MAX_STEPS', 1001) - 1
if (spel_profile) WRITE (*, *) 'SPEL_PROFILE: outputs disabled'
if (max_step /= 1000) WRITE (*, *) 'SPEL_MAX_STEPS: running ', max_step + 1, ' steps'
nclumps = spel_env_int('SPEL_CLUMPS', 1)
WRITE (*, *) 'nclumps = ', nclumps
#ifdef _OPENACC
if (nclumps /= 1) stop "SPEL_CLUMPS > 1 needs a build without OpenACC (e.g. ACC=OPENMP)"
#endif

call spel_prof_push('spel_init')

block 
   !#INPUT_PATH
   call spel_io_init(input_path, read_io=.true.)
end block

call elm_init(nsets, dtime_mod, year_curr, bounds_proc, nclumps)

!#INIT

declin = -0.4030289369547867

#ifdef _OPENACC
   call update_params_acc()

   ! Manual deep copy of the unit-test data (SpelAccDataMod)
   !#ACC_COPYIN

   call get_proc_bounds(bounds_proc)
   call get_clump_bounds(1, bounds_clump)
   !$acc enter data copyin(bounds_clump)

   ! Set the filters now that the data they read is on the device (filterMod
   ! may build them in compute regions), then sync the host and device copies
   allocate (icemask_dummy_arr(bounds_clump%begg:bounds_clump%endg))
   icemask_dummy_arr(:) = 0._r8
   call setFilters(bounds_clump, icemask_dummy_arr)
   call spel_acc_update_filters()

   ! Keep the time-step input buffers on the device; read_elmtypes then
   ! copies each step's inputs on the device
   call spel_inbuf_to_device()
#endif
!$acc enter data copyin( doalb, declinp1, declin )
!$acc update device(dtime_mod, dayspyr_mod, &
!$acc    year_curr, mon_curr, day_curr, secs_curr, nstep_mod, thiscalday_mod &
!$acc  , nextsw_cday_mod, end_cd_mod, doalb )
call spel_prof_pop() ! spel_init


#ifdef _OPENACC
#define gpuflag 1
#else
#define gpuflag 0
#endif

nclumps = procinfo%nclumps

block
    integer :: step = 0
    call spel_prof_push('spel_timesteps')
    do while (.true.)

       call spel_prof_push('step')
       if (step .ne. 0) then
          call spel_prof_push('read_inputs')
          bounds_io = bounds_proc
          !#IO_READ
          call spel_prof_pop()

          ! read_elmtypes leaves bounds_io at one set's bounds. The inputs
          ! are copied from the (device) input buffers, so no host->device
          ! update is needed; duplicate_sets also offsets on the device.
          if (.not. io_inputs%end_run) then
             call spel_prof_push('duplicate_sets')
             call duplicate_sets(nsets, bounds_io)
             call spel_prof_pop()
          end if
       end if
       if (io_inputs%end_run .or. step > max_step) then
          call spel_prof_pop() ! step
          exit
       end if

      ! The 'compute' range is the wall time of all clumps. The clump loop is
      ! elm_drv's; it is only active with several clumps, so that with one
      ! clump the routines' own OpenMP loops (spel acc2omp) are not nested.
      call spel_prof_push('compute')
      !$OMP PARALLEL DO PRIVATE (nc, bounds_clump) IF (nclumps > 1)
      do nc = 1, nclumps
         call get_clump_bounds(nc, bounds_clump)
         !$acc update device(bounds_clump)
      !#CALL_SUB
      end do
      !$OMP END PARALLEL DO
      call spel_prof_pop()

       if (.not. spel_profile) then
          call spel_prof_push('update_host')
          !#ACC_UPDATE_HOST
          call spel_prof_pop()

          call spel_prof_push('write_outputs')
          bounds_io = bounds_proc
          !#IO_WRITE
          call spel_prof_pop()
       end if

       call spel_prof_pop() ! step
       step = step + 1
    end do
    !$acc wait
    call spel_prof_pop() ! spel_timesteps
end block

print *, "Finished: Read ", io_inputs%filenum-1, " files"

end Program main
