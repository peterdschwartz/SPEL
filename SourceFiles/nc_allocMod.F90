module nc_allocMod

   use netcdf
   use nc_io, only: check
   use iso_fortran_env
   use iso_c_binding
   implicit none

   public

   interface nc_alloc
      module procedure nc_alloc_double_1, nc_alloc_double_2, nc_alloc_double_3, &
         nc_alloc_integer_1, nc_alloc_integer_2, nc_alloc_integer_3, &
         nc_alloc_logical_1, nc_alloc_logical_2, nc_alloc_logical_3
      module procedure nc_ptr_double_1, nc_ptr_double_2, nc_ptr_double_3, &
         nc_ptr_integer_1, nc_ptr_integer_2, nc_ptr_integer_3, &
         nc_ptr_logical_1, nc_ptr_logical_2, nc_ptr_logical_3
   end interface

contains


   subroutine nc_alloc_double_1(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      real(8), allocatable, intent(inout) :: var(:)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (allocated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))
      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1)))

   end subroutine nc_alloc_double_1

   subroutine nc_alloc_double_2(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      real(8), allocatable, intent(inout) :: var(:, :)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (allocated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1), lbs(2):ubs(2)))

   end subroutine nc_alloc_double_2

   subroutine nc_alloc_double_3(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      real(8), allocatable, intent(inout) :: var(:, :, :)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (allocated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1), lbs(2):ubs(2), lbs(3):ubs(3)))

   end subroutine nc_alloc_double_3

   subroutine nc_alloc_integer_1(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      integer, allocatable, intent(inout) :: var(:)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (allocated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1)))

   end subroutine nc_alloc_integer_1

   subroutine nc_alloc_integer_2(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      integer, allocatable, intent(inout) :: var(:, :)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (allocated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1), lbs(2):ubs(2)))

   end subroutine nc_alloc_integer_2

   subroutine nc_alloc_integer_3(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      integer, allocatable, intent(inout) :: var(:, :, :)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (allocated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1), lbs(2):ubs(2), lbs(3):ubs(3)))

   end subroutine nc_alloc_integer_3

   subroutine nc_alloc_logical_1(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      logical, allocatable, intent(inout) :: var(:)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (allocated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1)))

   end subroutine nc_alloc_logical_1

   subroutine nc_alloc_logical_2(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      logical, allocatable, intent(inout) :: var(:, :)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (allocated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1), lbs(2):ubs(2)))

   end subroutine nc_alloc_logical_2

   subroutine nc_alloc_logical_3(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      logical, allocatable, intent(inout) :: var(:, :, :)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (allocated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1), lbs(2):ubs(2), lbs(3):ubs(3)))

   end subroutine nc_alloc_logical_3

   subroutine nc_ptr_double_1(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      real(8), pointer, intent(inout) :: var(:)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (associated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1)))

   end subroutine nc_ptr_double_1

   subroutine nc_ptr_double_2(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      real(8), pointer, intent(inout) :: var(:, :)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (associated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1), lbs(2):ubs(2)))

   end subroutine nc_ptr_double_2

   subroutine nc_ptr_double_3(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      real(8), pointer, intent(inout) :: var(:, :, :)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (associated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1), lbs(2):ubs(2), lbs(3):ubs(3)))

   end subroutine nc_ptr_double_3

   subroutine nc_ptr_integer_1(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      integer, pointer, intent(inout) :: var(:)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (associated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1)))

   end subroutine nc_ptr_integer_1

   subroutine nc_ptr_integer_2(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      integer, pointer, intent(inout) :: var(:, :)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (associated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1), lbs(2):ubs(2)))

   end subroutine nc_ptr_integer_2

   subroutine nc_ptr_integer_3(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      integer, pointer, intent(inout) :: var(:, :, :)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (associated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1), lbs(2):ubs(2), lbs(3):ubs(3)))

   end subroutine nc_ptr_integer_3

   subroutine nc_ptr_logical_1(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      logical, pointer, intent(inout) :: var(:)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid
      if (associated(var)) return

      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1)))

   end subroutine nc_ptr_logical_1

   subroutine nc_ptr_logical_2(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      logical, pointer, intent(inout) :: var(:, :)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (associated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1), lbs(2):ubs(2)))

   end subroutine nc_ptr_logical_2

   subroutine nc_ptr_logical_3(ncid, varname, ndim, var)
      integer, intent(in):: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      logical, pointer, intent(inout) :: var(:, :, :)
      integer :: lbs(ndim), ubs(ndim), status
      logical :: has_bounds
      integer :: varid

      if (associated(var)) return
      call check(nf90_inq_varid(ncid, trim(varname), varid))

      ! Check if 'lbounds' attribute exists
      status = nf90_get_att(ncid, varid, "lbounds", lbs)
      has_bounds = (status == nf90_noerr)
      status = nf90_get_att(ncid, varid, "ubounds", ubs)
      has_bounds = (status == nf90_noerr) .and. has_bounds
      ! Allocate accordingly
      if (.not. has_bounds) then
         print *, "Error - couldn't find l/ubounds"
         stop
      end if
      allocate (var(lbs(1):ubs(1), lbs(2):ubs(2), lbs(3):ubs(3)))

   end subroutine nc_ptr_logical_3

end module nc_allocMod
