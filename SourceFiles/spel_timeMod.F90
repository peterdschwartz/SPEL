module spel_timeMod
  !-----------------------------------------------------------------------
  ! Unit-test stand-ins for the elm_time_manager functions routines call to
  ! test the time step or get its length (elm_time_manager needs ESMF's
  ! clock). SPEL points `use elm_time_manager, only : ...` of these names
  ! here (spel.scripts.io.time_scalars.TIME_SHIMS).
  !
  ! The ELM capture module records get_nstep(), is_first_restart_step(),
  ! get_step_size() and get_rad_step_size() at every capture; the unit test
  ! sets the spel_* variables from them after reading each step's inputs.
  ! is_end_curr_day()/get_curr_date() return timeinfoMod's per-step scalars.
  !-----------------------------------------------------------------------
  use shr_kind_mod, only : r8 => shr_kind_r8
  implicit none
  private
  integer, public :: spel_nstep = 0
  logical, public :: spel_first_restart_step = .false.
  integer, public :: spel_step_size = 0
  integer, public :: spel_rad_step_size = 0
  !$acc declare copyin(spel_nstep, spel_first_restart_step, spel_step_size, spel_rad_step_size)
  public :: is_first_step, is_first_restart_step
  public :: get_nstep, get_step_size, get_step_size_real, get_rad_step_size
  public :: is_end_curr_day, get_curr_date

contains

  logical function is_first_step()
    !$acc routine seq
    is_first_step = (spel_nstep == 0)
  end function is_first_step

  logical function is_first_restart_step()
    !$acc routine seq
    is_first_restart_step = spel_first_restart_step
  end function is_first_restart_step

  integer function get_nstep()
    !$acc routine seq
    get_nstep = spel_nstep
  end function get_nstep

  integer function get_step_size()
    !$acc routine seq
    get_step_size = spel_step_size
  end function get_step_size

  real(r8) function get_step_size_real()
    !$acc routine seq
    get_step_size_real = real(spel_step_size, r8)
  end function get_step_size_real

  integer function get_rad_step_size()
    !$acc routine seq
    get_rad_step_size = spel_rad_step_size
  end function get_rad_step_size

  logical function is_end_curr_day()
    !$acc routine seq
    ! the per-step capture of timeinfoMod's end_cd_mod (= is_end_curr_day() in ELM)
    use timeinfoMod, only : end_cd_mod
    is_end_curr_day = end_cd_mod
  end function is_end_curr_day

  subroutine get_curr_date(yr, mon, day, tod, offset)
    !$acc routine seq
    ! the per-step capture of timeinfoMod's current date (= get_curr_date() in ELM); an
    ! offset is not supported here (the unit-test calls don't use one)
    use timeinfoMod, only : year_curr, mon_curr, day_curr, secs_curr
    integer, intent(out) :: yr, mon, day, tod
    integer, optional, intent(in) :: offset
    yr = year_curr
    mon = mon_curr
    day = day_curr
    tod = secs_curr
  end subroutine get_curr_date

end module spel_timeMod
