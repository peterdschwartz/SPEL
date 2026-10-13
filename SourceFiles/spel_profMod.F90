module spel_profMod
   !! Profiling hooks for SPEL unit tests.
   !! spel_prof_push/pop mark NVTX ranges for Nsight Systems / Nsight Compute
   !! when built with SPEL_NVTX (nvfortran, -cudalib=nvtx); otherwise no-ops.
#ifdef SPEL_NVTX
   use nvtx, only: nvtxStartRange, nvtxEndRange
#endif
   implicit none
   private
   public :: spel_prof_push, spel_prof_pop, spel_env_flag, spel_env_int

contains

   subroutine spel_prof_push(name)
      character(len=*), intent(in) :: name
#ifdef SPEL_NVTX
      call nvtxStartRange(name)
#endif
   end subroutine spel_prof_push

   subroutine spel_prof_pop()
#ifdef SPEL_NVTX
      call nvtxEndRange()
#endif
   end subroutine spel_prof_pop

   logical function spel_env_flag(name) result(flag)
      ! True when environment variable `name` is set to anything but "", "0" or "false"
      character(len=*), intent(in) :: name
      character(len=32) :: val
      integer :: stat
      flag = .false.
      call get_environment_variable(name, val, status=stat)
      if (stat /= 0) return
      select case (trim(adjustl(val)))
      case ('', '0', 'false', 'FALSE', 'False', 'no', 'NO', 'off', 'OFF')
         flag = .false.
      case default
         flag = .true.
      end select
   end function spel_env_flag

   integer function spel_env_int(name, default) result(n)
      ! Positive integer from environment variable `name`, or `default`
      character(len=*), intent(in) :: name
      integer, intent(in) :: default
      character(len=32) :: val
      integer :: stat
      n = default
      call get_environment_variable(name, val, status=stat)
      if (stat /= 0) return
      read (val, *, iostat=stat) n
      if (stat /= 0 .or. n <= 0) n = default
   end function spel_env_int

end module spel_profMod
