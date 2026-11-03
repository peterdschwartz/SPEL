
module nc_io
  use netcdf
  use iso_fortran_env
  use iso_c_binding
#ifdef _OPENACC
  use openacc, only: acc_is_present
#endif
  implicit none

  public
  integer, parameter :: read_file = 0, append_file = 1
  integer, parameter :: create_file = 2
  real(8), parameter :: fill_double = 1.d+36
  integer, parameter :: fill_int = -9999
  type, public :: spel_io_type
    character(len=256) :: fn
    integer :: timestep
    integer :: max_timesteps_per_file
    integer :: filenum
    logical :: new_file
    logical :: created = .false.
    logical :: read_mode = .false.
    logical :: end_run = .False.
    integer :: dt_in_file=99999 !used for reading files only
  contains
    procedure, public :: get_fn
    procedure, public :: init
    procedure, private :: check_file_exists
    procedure, private :: need_new_file
  end type spel_io_type
  type(spel_io_type), public :: io_constants, io_inputs, io_outputs
  ! Number of copies of the input sites (unit tests only). Arrays with a
  ! subgrid dimension are allocated nsets times larger along it, reads
  ! replicate the file data into every set and writes emit set 1 only.
  integer, public :: spel_nsets = 1
  ! Set once the input buffers are on the device (spel_inbuf_to_device):
  ! buffer loads then update the device and copies run on the device.
  logical, public :: spel_buf_on_device = .false.
  ! Gridcell subset (instrumented ELM only, see spel/scripts/subset.py).
  ! While a subset is active (nc_subset_activate), writes keep only the
  ! selected gridcells and their topounits, landunits, columns and
  ! patches, and renumber the subgrid index fields, so the files look
  ! like a capture of a smaller domain.
  integer, parameter :: spel_nlev = 5  ! gridcell, topo, landunit, column, patch
  type, public :: spel_subset_level
    integer :: beg = 1    ! lower bound of the full (process) arrays
    integer :: nfull = 0  ! their length
    integer, allocatable :: pos(:)     ! kept positions in the full arrays
    integer, allocatable :: newidx(:)  ! new index of each position, 0 if dropped
  end type spel_subset_level
  type, public :: spel_subset_type
    type(spel_subset_level) :: lev(spel_nlev)
    character(len=512) :: mask = ''
    ! provenance of the kept gridcells, written as global attributes
    integer, allocatable :: gindex(:)
    real(8), allocatable :: latdeg(:), londeg(:)
  end type spel_subset_type
  type(spel_subset_type), pointer, public :: spel_subset => null()
  interface nc_write_var_array
    module procedure nc_write_double
    module procedure nc_write_integer
    module procedure nc_write_logical
    module procedure nc_write_string
  end interface
  interface nc_write_var_scalar
    module procedure nc_write_double_scalar
    module procedure nc_write_integer_scalar
    module procedure nc_write_logical_scalar
  end interface
  interface nc_read_var
    module procedure nc_read_double_0
    module procedure nc_read_double_1
    module procedure nc_read_double_2
    module procedure nc_read_double_3
    module procedure nc_read_integer_0
    module procedure nc_read_integer_1
    module procedure nc_read_integer_2
    module procedure nc_read_integer_3
    module procedure nc_read_logical_0
    module procedure nc_read_logical_1
    module procedure nc_read_logical_2
    module procedure nc_read_logical_3
    module procedure nc_read_string
  end interface
  interface nc_select_set1
    module procedure nc_select_set1_r8, nc_select_set1_int
  end interface
  interface nc_subset_select
    module procedure nc_subset_select_r8, nc_subset_select_int
  end interface
  interface nc_buf_load
    module procedure nc_buf_load_double_0
    module procedure nc_buf_load_double_1
    module procedure nc_buf_load_double_2
    module procedure nc_buf_load_double_3
    module procedure nc_buf_load_integer_0
    module procedure nc_buf_load_integer_1
    module procedure nc_buf_load_integer_2
    module procedure nc_buf_load_integer_3
  end interface
  interface nc_buf_copy
    module procedure nc_buf_copy_double_0
    module procedure nc_buf_copy_double_1
    module procedure nc_buf_copy_double_2
    module procedure nc_buf_copy_double_3
    module procedure nc_buf_copy_integer_0
    module procedure nc_buf_copy_integer_1
    module procedure nc_buf_copy_integer_2
    module procedure nc_buf_copy_integer_3
    module procedure nc_buf_copy_logical_0
    module procedure nc_buf_copy_logical_1
    module procedure nc_buf_copy_logical_2
    module procedure nc_buf_copy_logical_3
  end interface
  public :: nc_read_timeslices
contains

subroutine check(status, name)
   ! name: the variable of a failed lookup/read, so 'Variable not found' says which
   integer, intent(in) :: status
   character(len=*), intent(in), optional :: name
   if (status /= nf90_noerr) then
      if (present(name)) then
         print *, trim(nf90_strerror(status)), ' (', trim(name), ')'
      else
         print *, trim(nf90_strerror(status))
      end if
      stop 2
   end if
end subroutine check

logical function spel_streq(a, b)
   ! case-insensitive string equality, used by namelist guards in ReadWriteMod
   character(len=*), intent(in) :: a, b
   integer :: i, ca, cb
   spel_streq = .false.
   if (len_trim(a) /= len_trim(b)) return
   do i = 1, len_trim(a)
      ca = iachar(a(i:i))
      cb = iachar(b(i:i))
      if (ca >= iachar('A') .and. ca <= iachar('Z')) ca = ca + 32
      if (cb >= iachar('A') .and. cb <= iachar('Z')) cb = cb + 32
      if (ca /= cb) return
   end do
   spel_streq = .true.
end function spel_streq

logical function nc_has_var(ncid, varname)
   ! false for globals that were unallocated when the file was written
   integer, intent(in) :: ncid
   character(len=*), intent(in) :: varname
   integer :: varid
   nc_has_var = nf90_inq_varid(ncid, trim(varname), varid) == nf90_noerr
end function nc_has_var

integer function nc_index_level(varname) result(ilev)
   ! Subgrid level (1-5: g, t, l, c, p) a subgrid index field (<inst>_pp__<field>)
   ! refers to, 0 for other variables
   character(len=*), intent(in) :: varname
   integer :: i
   ilev = 0
   i = index(varname, '_pp__', back=.true.)
   if (i == 0) return
   select case (trim(varname(i+5:)))
       case ('gindex', 'gridcell')
          ilev = 1
       case ('downhill_ti', 'topf', 'topi', 'topounit')
          ilev = 2
       case ('landunit', 'landunit_indices', 'lndf', 'lndi')
          ilev = 3
       case ('colf', 'coli', 'column')
          ilev = 4
       case ('pftf', 'pfti')
          ilev = 5
   end select
end function nc_index_level

logical function nc_is_index_field(varname)
   ! Subgrid index fields are offset per set by duplicateMod
   character(len=*), intent(in) :: varname
   nc_is_index_field = nc_index_level(varname) > 0
end function nc_is_index_field

integer function nc_subgrid_level(name) result(ilev)
   ! Subgrid level (1-5: g, t, l, c, p) of a dimension name, 0 if not subgrid
   character(len=*), intent(in) :: name
   select case (trim(name))
   case ('gridcell')
      ilev = 1
   case ('topo')
      ilev = 2
   case ('landunit')
      ilev = 3
   case ('column')
      ilev = 4
   case ('patch')
      ilev = 5
   case default
      ilev = 0
   end select
end function nc_subgrid_level

logical function nc_is_subgrid(name)
   character(len=*), intent(in) :: name
   nc_is_subgrid = nc_subgrid_level(name) > 0
end function nc_is_subgrid

integer function nc_first_subgrid_dim(ndim, dim_names) result(sd)
   ! First subgrid dimension of a variable, or 0 if it has none.
   integer, intent(in) :: ndim
   character(len=*), dimension(:), intent(in) :: dim_names
   integer :: i
   sd = 0
   do i = 1, min(ndim, size(dim_names))
      if (nc_is_subgrid(dim_names(i))) then
         sd = i
         return
      end if
   end do
end function nc_first_subgrid_dim

integer function nc_subgrid_dim(ndim, dim_names) result(sd)
   ! First subgrid dimension of a variable, or 0 when not duplicating.
   integer, intent(in) :: ndim
   character(len=*), dimension(:), intent(in) :: dim_names
   sd = 0
   if (spel_nsets <= 1) return
   sd = nc_first_subgrid_dim(ndim, dim_names)
end function nc_subgrid_dim

subroutine nc_file_subgrid(ncid, var_id, ndim, sd, nfile)
   ! First subgrid dimension (sd) of a file variable and its length in
   ! the file (nfile); sd = 0 when not duplicating.
   integer, intent(in) :: ncid, var_id, ndim
   integer, intent(out) :: sd, nfile
   integer :: nd, i
   integer, allocatable :: dimids(:)
   character(len=nf90_max_name) :: dname
   sd = 0; nfile = 0
   if (spel_nsets <= 1) return
   call check(nf90_inquire_variable(ncid, var_id, ndims=nd))
   allocate (dimids(nd))
   call check(nf90_inquire_variable(ncid, var_id, dimids=dimids))
   do i = 1, min(ndim, nd)
      call check(nf90_inquire_dimension(ncid, dimids(i), name=dname, len=nfile))
      if (nc_is_subgrid(dname)) then
         sd = i
         return
      end if
   end do
   nfile = 0
end subroutine nc_file_subgrid

integer function nc_set_ub(lb, ub) result(ub1)
   ! Upper bound of set 1 for a range duplicated spel_nsets times.
   integer, intent(in) :: lb, ub
   ub1 = ub
   if (spel_nsets <= 1) return
   if (mod(ub - lb + 1, spel_nsets) /= 0) return
   ub1 = lb - 1 + (ub - lb + 1)/spel_nsets
end function nc_set_ub

subroutine nc_put_bounds_att(ncid, varid, ndim, dim_names, lbs, ubs)
   ! Write l/ubounds attributes, as set 1's bounds when duplicating and
   ! as 1:n along the subgrid dimension of a gridcell subset.
   integer, intent(in) :: ncid, varid, ndim
   character(len=*), dimension(:), intent(in) :: dim_names
   integer, intent(in) :: lbs(ndim), ubs(ndim)
   integer :: lb1(ndim), ub1(ndim), sd
   lb1 = lbs
   ub1 = ubs
   sd = nc_subgrid_dim(ndim, dim_names)
   if (sd > 0) ub1(sd) = nc_set_ub(lbs(sd), ubs(sd))
   if (associated(spel_subset)) then
      sd = nc_first_subgrid_dim(ndim, dim_names)
      if (sd > 0) then
         lb1(sd) = 1
         ub1(sd) = nc_subset_n(dim_names(sd), ubs(sd) - lbs(sd) + 1, 'bounds attribute')
      end if
   end if
   call check(nf90_put_att(ncid, varid, 'lbounds', lb1))
   call check(nf90_put_att(ncid, varid, 'ubounds', ub1))
end subroutine nc_put_bounds_att

subroutine nc_select_set1_r8(ndim, dims, dim_names, buffer, varname)
   ! Reduce a flattened write buffer to set 1 along its subgrid dimension
   ! and report values in the other sets that differ from set 1.
   use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
   integer, intent(in) :: ndim
   integer, intent(inout) :: dims(ndim)
   character(len=*), dimension(:), intent(in) :: dim_names
   real(8), allocatable, intent(inout) :: buffer(:)
   character(len=*), intent(in) :: varname
   real(8), allocatable :: b3(:,:,:)
   integer :: sd, n1, pre, post, k, nbad
   sd = nc_subgrid_dim(ndim, dim_names)
   if (sd == 0) return
   if (mod(dims(sd), spel_nsets) /= 0) return
   n1 = dims(sd)/spel_nsets
   pre = product(dims(1:sd-1))
   post = product(dims(sd+1:ndim))
   b3 = reshape(buffer, [pre, dims(sd), post])
   nbad = 0
   do k = 2, spel_nsets
      nbad = nbad + count(differs(b3(:, (k-1)*n1+1:k*n1, :), b3(:, 1:n1, :)))
   end do
   if (nbad > 0 .and. .not. nc_is_index_field(varname)) print *, 'WARNING (spel_nsets): ', trim(varname), nbad, &
      ' values in sets 2..nsets differ from set 1'
   buffer = reshape(b3(:, 1:n1, :), [pre*n1*post])
   dims(sd) = n1
contains
   elemental logical function differs(a, b)
      real(8), intent(in) :: a, b
      differs = a /= b .and. .not. (ieee_is_nan(a) .and. ieee_is_nan(b))
   end function differs
end subroutine nc_select_set1_r8

subroutine nc_select_set1_int(ndim, dims, dim_names, buffer, varname)
   ! Reduce a flattened write buffer to set 1 along its subgrid dimension
   ! and report values in the other sets that differ from set 1.
   use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
   integer, intent(in) :: ndim
   integer, intent(inout) :: dims(ndim)
   character(len=*), dimension(:), intent(in) :: dim_names
   integer, allocatable, intent(inout) :: buffer(:)
   character(len=*), intent(in) :: varname
   integer, allocatable :: b3(:,:,:)
   integer :: sd, n1, pre, post, k, nbad
   sd = nc_subgrid_dim(ndim, dim_names)
   if (sd == 0) return
   if (mod(dims(sd), spel_nsets) /= 0) return
   n1 = dims(sd)/spel_nsets
   pre = product(dims(1:sd-1))
   post = product(dims(sd+1:ndim))
   b3 = reshape(buffer, [pre, dims(sd), post])
   nbad = 0
   do k = 2, spel_nsets
      nbad = nbad + count(differs(b3(:, (k-1)*n1+1:k*n1, :), b3(:, 1:n1, :)))
   end do
   if (nbad > 0 .and. .not. nc_is_index_field(varname)) print *, 'WARNING (spel_nsets): ', trim(varname), nbad, &
      ' values in sets 2..nsets differ from set 1'
   buffer = reshape(b3(:, 1:n1, :), [pre*n1*post])
   dims(sd) = n1
contains
   elemental logical function differs(a, b)
      integer, intent(in) :: a, b
      differs = a /= b
   end function differs
end subroutine nc_select_set1_int

subroutine nc_subset_set_level(sub, ilev, beg, keep)
   ! Level ilev (1-5: g, t, l, c, p) keeps the positions where keep is true;
   ! keep spans the full (process) arrays, which start at index beg.
   type(spel_subset_type), intent(inout) :: sub
   integer, intent(in) :: ilev, beg
   logical, intent(in) :: keep(:)
   integer :: i
   associate (lev => sub%lev(ilev))
      lev%beg = beg
      lev%nfull = size(keep)
      lev%pos = pack([(i, i = 1, size(keep))], keep)
      if (allocated(lev%newidx)) deallocate (lev%newidx)
      allocate (lev%newidx(size(keep)))
      lev%newidx = 0
      lev%newidx(lev%pos) = [(i, i = 1, size(lev%pos))]
   end associate
end subroutine nc_subset_set_level

subroutine nc_subset_activate(sub)
   ! Writes keep only sub's gridcells until nc_subset_deactivate
   type(spel_subset_type), target, intent(in) :: sub
   spel_subset => sub
end subroutine nc_subset_activate

subroutine nc_subset_deactivate()
   nullify (spel_subset)
end subroutine nc_subset_deactivate

integer function nc_subset_count(sub, ilev) result(n)
   type(spel_subset_type), intent(in) :: sub
   integer, intent(in) :: ilev
   n = 0
   if (allocated(sub%lev(ilev)%pos)) n = size(sub%lev(ilev)%pos)
end function nc_subset_count

integer function nc_subset_n(dim_name, nfull, varname) result(n)
   ! Length of a subgrid dimension in the active subset. A variable whose
   ! subgrid dimension doesn't span the process arrays can't be subset.
   character(len=*), intent(in) :: dim_name, varname
   integer, intent(in) :: nfull
   integer :: ilev
   ilev = nc_subgrid_level(dim_name)
   if (nfull /= spel_subset%lev(ilev)%nfull) then
      print *, 'ERROR (spel subset): ', trim(varname), ' has ', nfull, ' ', trim(dim_name), &
         ' entries, the process has ', spel_subset%lev(ilev)%nfull
      error stop 'spel subset: subgrid dimension does not match the process bounds'
   end if
   n = nc_subset_count(spel_subset, ilev)
end function nc_subset_n

subroutine nc_subset_put_atts(ncid)
   ! Record the subset in a new file's global attributes
   integer, intent(in) :: ncid
   call check(nf90_put_att(ncid, nf90_global, 'spel_subset_mask', trim(spel_subset%mask)))
   if (allocated(spel_subset%gindex)) &
      call check(nf90_put_att(ncid, nf90_global, 'spel_subset_gindex', spel_subset%gindex))
   if (allocated(spel_subset%latdeg)) &
      call check(nf90_put_att(ncid, nf90_global, 'spel_subset_latdeg', spel_subset%latdeg))
   if (allocated(spel_subset%londeg)) &
      call check(nf90_put_att(ncid, nf90_global, 'spel_subset_londeg', spel_subset%londeg))
end subroutine nc_subset_put_atts

subroutine nc_subset_select_r8(ndim, dims, dim_names, buffer, varname)
   ! Keep the subset's positions along the subgrid dimension of a flattened
   ! write buffer and renumber subgrid index fields.
   integer, intent(in) :: ndim
   integer, intent(inout) :: dims(ndim)
   character(len=*), dimension(:), intent(in) :: dim_names
   real(8), allocatable, intent(inout) :: buffer(:)
   character(len=*), intent(in) :: varname
   real(8), allocatable :: b3(:,:,:)
   integer :: sd, ilev, pre, post, n, i, k
   sd = nc_first_subgrid_dim(ndim, dim_names)
   if (sd > 0) then
      ilev = nc_subgrid_level(dim_names(sd))
      n = nc_subset_n(dim_names(sd), dims(sd), varname)
      pre = product(dims(1:sd-1))
      post = product(dims(sd+1:ndim))
      b3 = reshape(buffer, [pre, dims(sd), post])
      buffer = reshape(b3(:, spel_subset%lev(ilev)%pos, :), [pre*n*post])
      dims(sd) = n
   end if
   ilev = nc_index_level(varname)
   if (ilev == 0) return
   if (varname(index(varname, '_pp__', back=.true.)+5:) == 'gindex') then
      ! global index of the kept gridcells: the subset is its own domain
      buffer = [(i, i = 1, size(buffer))]
      return
   end if
   associate (lev => spel_subset%lev(ilev))
      do i = 1, size(buffer)
         k = nint(buffer(i)) - lev%beg + 1
         if (k < 1 .or. k > lev%nfull) cycle  ! e.g. ispval: not an index
         if (lev%newidx(k) > 0) then
            buffer(i) = lev%newidx(k)
         else
            buffer(i) = fill_int  ! refers to a dropped gridcell
         end if
      end do
   end associate
end subroutine nc_subset_select_r8

subroutine nc_subset_select_int(ndim, dims, dim_names, buffer, varname)
   ! Keep the subset's positions along the subgrid dimension of a flattened
   ! write buffer and renumber subgrid index fields.
   integer, intent(in) :: ndim
   integer, intent(inout) :: dims(ndim)
   character(len=*), dimension(:), intent(in) :: dim_names
   integer, allocatable, intent(inout) :: buffer(:)
   character(len=*), intent(in) :: varname
   integer, allocatable :: b3(:,:,:)
   integer :: sd, ilev, pre, post, n, i, k
   sd = nc_first_subgrid_dim(ndim, dim_names)
   if (sd > 0) then
      ilev = nc_subgrid_level(dim_names(sd))
      n = nc_subset_n(dim_names(sd), dims(sd), varname)
      pre = product(dims(1:sd-1))
      post = product(dims(sd+1:ndim))
      b3 = reshape(buffer, [pre, dims(sd), post])
      buffer = reshape(b3(:, spel_subset%lev(ilev)%pos, :), [pre*n*post])
      dims(sd) = n
   end if
   ilev = nc_index_level(varname)
   if (ilev == 0) return
   if (varname(index(varname, '_pp__', back=.true.)+5:) == 'gindex') then
      ! global index of the kept gridcells: the subset is its own domain
      buffer = [(i, i = 1, size(buffer))]
      return
   end if
   associate (lev => spel_subset%lev(ilev))
      do i = 1, size(buffer)
         k = buffer(i) - lev%beg + 1
         if (k < 1 .or. k > lev%nfull) cycle  ! e.g. ispval: not an index
         if (lev%newidx(k) > 0) then
            buffer(i) = lev%newidx(k)
         else
            buffer(i) = fill_int  ! refers to a dropped gridcell
         end if
      end do
   end associate
end subroutine nc_subset_select_int

integer function nc_create_or_open_file(fn, mode) result(ncid)
   character(len=*), intent(in) :: fn
   integer, intent(in) :: mode
   if (mode == read_file) then
          block 
             logical :: exists
             inquire(file=fn,exist=exists)
             if (.not. exists) stop "Error: " // trim(fn) // " Not Found"
          end block
      call check(nf90_open(trim(fn), nf90_nowrite + nf90_netcdf4, ncid))
   else if(mode == append_file) then
      call check(nf90_open(trim(fn), nf90_write + nf90_netcdf4, ncid))
   else
      call check(nf90_create(trim(fn), nf90_clobber + nf90_netcdf4, ncid))
      if (associated(spel_subset)) call nc_subset_put_atts(ncid)
   end if
end function nc_create_or_open_file


subroutine spel_io_init(path,read_io,max_tpf)
  ! Initialize io_constants/io_inputs/io_outputs (no-op if already done).
  ! Reading (offline unit test): outputs go to 'fut-outputs'.
  ! Writing (instrumented ELM):  outputs go to 'spel-outputs'.
  character(len=*), intent(in) :: path
  logical, intent(in) :: read_io
  integer, intent(in), optional :: max_tpf
  integer :: tpf
  if(io_constants%created) return
  tpf = 720
  if(present(max_tpf)) tpf = max_tpf
  call io_constants%init(base_fn=trim(path)//'spel-constants',max_tpf=tpf,read_io=read_io)
  call io_inputs%init(base_fn=trim(path)//'spel-inputs',max_tpf=tpf,read_io=read_io)
  if(read_io) then
    call io_outputs%init(base_fn=trim(path)//'fut-outputs',max_tpf=tpf,read_io=.false.)
  else
    call io_outputs%init(base_fn=trim(path)//'spel-outputs',max_tpf=tpf,read_io=.false.)
  end if
end subroutine spel_io_init

subroutine init(this,base_fn,max_tpf,read_io)
  class(spel_io_type), intent(inout) :: this
  character(len=*), intent(in) :: base_fn
  integer, intent(in) :: max_tpf
  logical, intent(in) :: read_io
  if(this%created) error stop "(spel_io_type::init) Error -- io_inst already initialized"
  this%read_mode = read_io
  this%max_timesteps_per_file = max_tpf
  this%new_file = .true.
  this%fn = trim(base_fn)
  this%timestep = 0
  this%new_file = .true.
  this%filenum = 0
  this%created = .true.
end subroutine init

logical function need_new_file(this) result(flag)
  class(spel_io_type), intent(inout) :: this
  if(this%read_mode) then 
      flag = (this%timestep > this%dt_in_file)
  else
      flag = (this%timestep > this%max_timesteps_per_file)
  end if
end function need_new_file

logical function check_file_exists(this,fn) result(exists)
    class(spel_io_type), intent(inout) :: this
    character(len=*), intent(in) :: fn
    inquire(file=trim(fn), exist=exists)
end function check_file_exists

character(len=256) function get_fn(this) result(new_fn)
  class(spel_io_type), intent(inout) :: this
  character(len=10) :: ch_num
  integer :: max_time_steps
  this%timestep = this%timestep + 1
  this%new_file = .False.
  if(this%need_new_file() .or. this%timestep == 1) then
    this%filenum = this%filenum + 1
    this%timestep = 1
    this%new_file = .true.
    if(this%read_mode) this%dt_in_file = 99999
  end if
  write(unit=ch_num,fmt='(I0.4)') this%filenum
  new_fn = trim(this%fn)//trim(ch_num)//'.nc'
  if(this%read_mode) this%end_run = .not. (this%check_file_exists(new_fn))
end function get_fn

subroutine nc_define_var(ncid, ndim, dims, dim_list, varname, xtype, var_id, time)
   integer, intent(in) :: ncid, ndim
   integer, intent(in) :: dims(ndim)
   character(len=32), dimension(ndim), intent(in) :: dim_list
   character(len=*), intent(in) :: varname
   integer, intent(in) :: xtype  ! e.g. NF90_DOUBLE, NF90_INT, NF90_CHAR, NF90_STRING
   integer, intent(out) :: var_id
   logical, intent(in) :: time
   ! Locals
   integer :: i, status, total_dims, sd
   integer :: ldims(ndim+1)
   integer, allocatable :: dim_ids(:)
   character(len=32), allocatable :: dim_names(:)

   allocate(dim_names(ndim))
   dim_names(:) = dim_list(:)
   ldims(1:ndim) = dims
   ldims(ndim+1) = 0
   sd = nc_subgrid_dim(ndim, dim_names)
   if (sd > 0) then
      if (mod(ldims(sd), spel_nsets) == 0) ldims(sd) = ldims(sd)/spel_nsets
   end if
   if (associated(spel_subset)) then
      sd = nc_first_subgrid_dim(ndim, dim_names)
      if (sd > 0) ldims(sd) = nc_subset_n(dim_names(sd), ldims(sd), varname)
   end if

   total_dims = ndim
   if (time) then 
      total_dims = ndim + 1
      dim_names = [dim_names, 'time']
   end if

   allocate (dim_ids(total_dims))
   do i = 1, total_dims
      status = nf90_inq_dimid(ncid, trim(dim_names(i)), dim_ids(i))
      if(status .ne. nf90_noerr) then
         call check(nf90_def_dim(ncid, trim(dim_names(i)), ldims(i), dim_ids(i)))
      end if
   end do

   if (ndim == 0) then
      ! scalar variable: pass zero-length dim_ids
      call check(nf90_def_var(ncid, trim(varname), xtype, dim_ids(1:0), var_id))
   else
      call check(nf90_def_var(ncid, trim(varname), xtype, dim_ids, var_id))
   end if

   select case (xtype)
   case (nf90_double)
      call check(nf90_put_att(ncid,var_id,"_FillValue", fill_double))
   case (nf90_int)
      call check(nf90_put_att(ncid,var_id,"_FillValue", fill_int))
   end select

   deallocate (dim_ids)
end subroutine

subroutine nc_read_double_0(ncid, varname, var, timestep)
   integer, intent(in) :: ncid
   character(len=*), intent(in) :: varname
   real(8), intent(out) :: var
   integer, intent(in) :: timestep
   real(8) :: tmp(1)
   integer :: var_id, start(1), count(1)

   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   if (timestep > 0) then 
       start = [timestep]
       call check(nf90_get_var(ncid, var_id, tmp, start=[timestep],count=[1]))
       var = tmp(1)
    else 
       call check(nf90_get_var(ncid, var_id, var))
    endif 
end subroutine nc_read_double_0


  subroutine nc_read_double_1(ncid, varname, ndim, var, timestep)
     integer, intent(in) :: ncid
     character(len=*), intent(in) :: varname
     integer, intent(in) :: ndim 
     real(8), intent(out) :: var(:)
     integer, intent(in) :: timestep
     integer :: var_id
     integer, allocatable :: start(:), count(:)
     real(8), allocatable :: tmp(:)
     integer :: sd, nfile, k, lo, hi

     if (timestep > 0) then
       allocate(start(ndim+1), count(ndim+1))
       start(1:ndim) = 1
       start(ndim+1) = timestep
       count(1:ndim) = shape(var)
       count(ndim+1) = 1
    else
       allocate(start(ndim), count(ndim))
       start(1:ndim) = 1
       count(1:ndim) = shape(var)
    end if

     call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
     call nc_file_subgrid(ncid, var_id, ndim, sd, nfile)
     if (sd > 0) then
        if (size(var, sd) /= spel_nsets*nfile) sd = 0
     end if
     if (sd == 0) then
        call check(nf90_get_var(ncid, var_id, var,start=start, count=count))
     else
        ! read set 1 from the file and replicate it into every set
        count(sd) = nfile
        allocate(tmp(count(1)))
        call check(nf90_get_var(ncid, var_id, tmp,start=start, count=count))
        do k = 1, spel_nsets
           lo = (k-1)*nfile + 1; hi = k*nfile
           select case (sd)
           case (1)
              var(lo:hi) = tmp
           end select
        end do
     end if
  end subroutine nc_read_double_1


  subroutine nc_read_double_2(ncid, varname, ndim, var, timestep)
     integer, intent(in) :: ncid
     character(len=*), intent(in) :: varname
     integer, intent(in) :: ndim 
     real(8), intent(out) :: var(:,:)
     integer, intent(in) :: timestep
     integer :: var_id
     integer, allocatable :: start(:), count(:)
     real(8), allocatable :: tmp(:,:)
     integer :: sd, nfile, k, lo, hi

     if (timestep > 0) then
       allocate(start(ndim+1), count(ndim+1))
       start(1:ndim) = 1
       start(ndim+1) = timestep
       count(1:ndim) = shape(var)
       count(ndim+1) = 1
    else
       allocate(start(ndim), count(ndim))
       start(1:ndim) = 1
       count(1:ndim) = shape(var)
    end if

     call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
     call nc_file_subgrid(ncid, var_id, ndim, sd, nfile)
     if (sd > 0) then
        if (size(var, sd) /= spel_nsets*nfile) sd = 0
     end if
     if (sd == 0) then
        call check(nf90_get_var(ncid, var_id, var,start=start, count=count))
     else
        ! read set 1 from the file and replicate it into every set
        count(sd) = nfile
        allocate(tmp(count(1),count(2)))
        call check(nf90_get_var(ncid, var_id, tmp,start=start, count=count))
        do k = 1, spel_nsets
           lo = (k-1)*nfile + 1; hi = k*nfile
           select case (sd)
           case (1)
              var(lo:hi,:) = tmp
           case (2)
              var(:,lo:hi) = tmp
           end select
        end do
     end if
  end subroutine nc_read_double_2


  subroutine nc_read_double_3(ncid, varname, ndim, var, timestep)
     integer, intent(in) :: ncid
     character(len=*), intent(in) :: varname
     integer, intent(in) :: ndim 
     real(8), intent(out) :: var(:,:,:)
     integer, intent(in) :: timestep
     integer :: var_id
     integer, allocatable :: start(:), count(:)
     real(8), allocatable :: tmp(:,:,:)
     integer :: sd, nfile, k, lo, hi

     if (timestep > 0) then
       allocate(start(ndim+1), count(ndim+1))
       start(1:ndim) = 1
       start(ndim+1) = timestep
       count(1:ndim) = shape(var)
       count(ndim+1) = 1
    else
       allocate(start(ndim), count(ndim))
       start(1:ndim) = 1
       count(1:ndim) = shape(var)
    end if

     call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
     call nc_file_subgrid(ncid, var_id, ndim, sd, nfile)
     if (sd > 0) then
        if (size(var, sd) /= spel_nsets*nfile) sd = 0
     end if
     if (sd == 0) then
        call check(nf90_get_var(ncid, var_id, var,start=start, count=count))
     else
        ! read set 1 from the file and replicate it into every set
        count(sd) = nfile
        allocate(tmp(count(1),count(2),count(3)))
        call check(nf90_get_var(ncid, var_id, tmp,start=start, count=count))
        do k = 1, spel_nsets
           lo = (k-1)*nfile + 1; hi = k*nfile
           select case (sd)
           case (1)
              var(lo:hi,:,:) = tmp
           case (2)
              var(:,lo:hi,:) = tmp
           case (3)
              var(:,:,lo:hi) = tmp
           end select
        end do
     end if
  end subroutine nc_read_double_3


subroutine nc_read_integer_0(ncid, varname, var, timestep)
   integer, intent(in) :: ncid
   character(len=*), intent(in) :: varname
   integer, intent(out) :: var
   integer, intent(in) :: timestep
   integer :: tmp(1)
   integer :: var_id, start(1), count(1)

   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   if (timestep > 0) then 
       start = [timestep]
       call check(nf90_get_var(ncid, var_id, tmp, start=[timestep],count=[1]))
       var = tmp(1)
    else 
       call check(nf90_get_var(ncid, var_id, var))
    endif 
end subroutine nc_read_integer_0


  subroutine nc_read_integer_1(ncid, varname, ndim, var, timestep)
     integer, intent(in) :: ncid
     character(len=*), intent(in) :: varname
     integer, intent(in) :: ndim 
     integer, intent(out) :: var(:)
     integer, intent(in) :: timestep
     integer :: var_id
     integer, allocatable :: start(:), count(:)
     integer, allocatable :: tmp(:)
     integer :: sd, nfile, k, lo, hi

     if (timestep > 0) then
       allocate(start(ndim+1), count(ndim+1))
       start(1:ndim) = 1
       start(ndim+1) = timestep
       count(1:ndim) = shape(var)
       count(ndim+1) = 1
    else
       allocate(start(ndim), count(ndim))
       start(1:ndim) = 1
       count(1:ndim) = shape(var)
    end if

     call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
     call nc_file_subgrid(ncid, var_id, ndim, sd, nfile)
     if (sd > 0) then
        if (size(var, sd) /= spel_nsets*nfile) sd = 0
     end if
     if (sd == 0) then
        call check(nf90_get_var(ncid, var_id, var,start=start, count=count))
     else
        ! read set 1 from the file and replicate it into every set
        count(sd) = nfile
        allocate(tmp(count(1)))
        call check(nf90_get_var(ncid, var_id, tmp,start=start, count=count))
        do k = 1, spel_nsets
           lo = (k-1)*nfile + 1; hi = k*nfile
           select case (sd)
           case (1)
              var(lo:hi) = tmp
           end select
        end do
     end if
  end subroutine nc_read_integer_1


  subroutine nc_read_integer_2(ncid, varname, ndim, var, timestep)
     integer, intent(in) :: ncid
     character(len=*), intent(in) :: varname
     integer, intent(in) :: ndim 
     integer, intent(out) :: var(:,:)
     integer, intent(in) :: timestep
     integer :: var_id
     integer, allocatable :: start(:), count(:)
     integer, allocatable :: tmp(:,:)
     integer :: sd, nfile, k, lo, hi

     if (timestep > 0) then
       allocate(start(ndim+1), count(ndim+1))
       start(1:ndim) = 1
       start(ndim+1) = timestep
       count(1:ndim) = shape(var)
       count(ndim+1) = 1
    else
       allocate(start(ndim), count(ndim))
       start(1:ndim) = 1
       count(1:ndim) = shape(var)
    end if

     call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
     call nc_file_subgrid(ncid, var_id, ndim, sd, nfile)
     if (sd > 0) then
        if (size(var, sd) /= spel_nsets*nfile) sd = 0
     end if
     if (sd == 0) then
        call check(nf90_get_var(ncid, var_id, var,start=start, count=count))
     else
        ! read set 1 from the file and replicate it into every set
        count(sd) = nfile
        allocate(tmp(count(1),count(2)))
        call check(nf90_get_var(ncid, var_id, tmp,start=start, count=count))
        do k = 1, spel_nsets
           lo = (k-1)*nfile + 1; hi = k*nfile
           select case (sd)
           case (1)
              var(lo:hi,:) = tmp
           case (2)
              var(:,lo:hi) = tmp
           end select
        end do
     end if
  end subroutine nc_read_integer_2


  subroutine nc_read_integer_3(ncid, varname, ndim, var, timestep)
     integer, intent(in) :: ncid
     character(len=*), intent(in) :: varname
     integer, intent(in) :: ndim 
     integer, intent(out) :: var(:,:,:)
     integer, intent(in) :: timestep
     integer :: var_id
     integer, allocatable :: start(:), count(:)
     integer, allocatable :: tmp(:,:,:)
     integer :: sd, nfile, k, lo, hi

     if (timestep > 0) then
       allocate(start(ndim+1), count(ndim+1))
       start(1:ndim) = 1
       start(ndim+1) = timestep
       count(1:ndim) = shape(var)
       count(ndim+1) = 1
    else
       allocate(start(ndim), count(ndim))
       start(1:ndim) = 1
       count(1:ndim) = shape(var)
    end if

     call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
     call nc_file_subgrid(ncid, var_id, ndim, sd, nfile)
     if (sd > 0) then
        if (size(var, sd) /= spel_nsets*nfile) sd = 0
     end if
     if (sd == 0) then
        call check(nf90_get_var(ncid, var_id, var,start=start, count=count))
     else
        ! read set 1 from the file and replicate it into every set
        count(sd) = nfile
        allocate(tmp(count(1),count(2),count(3)))
        call check(nf90_get_var(ncid, var_id, tmp,start=start, count=count))
        do k = 1, spel_nsets
           lo = (k-1)*nfile + 1; hi = k*nfile
           select case (sd)
           case (1)
              var(lo:hi,:,:) = tmp
           case (2)
              var(:,lo:hi,:) = tmp
           case (3)
              var(:,:,lo:hi) = tmp
           end select
        end do
     end if
  end subroutine nc_read_integer_3


subroutine nc_read_logical_0(ncid, varname, var, timestep)
  integer, intent(in) :: ncid
  character(len=*), intent(in) :: varname
  logical, intent(out) :: var
  integer, intent(in) :: timestep
  integer :: temp

  call nc_read_var(ncid,varname, temp, timestep)
  if (temp == 1) then
     var = .true.
  else
     var = .false.
  end if

end subroutine nc_read_logical_0


subroutine nc_read_logical_1(ncid, varname, ndim, var, timestep)
   integer, intent(in) :: ncid
   character(len=*), intent(in) :: varname
   integer, intent(in) :: ndim
   logical, intent(out) :: var(:)
   integer, intent(in) :: timestep
   integer, allocatable :: temp(:)
   ! Locals:
   integer :: lbs(ndim), ubs(ndim)
   lbs = lbound(var); ubs = ubound(var)
   allocate(temp(lbs(1):ubs(1)))

   call nc_read_var(ncid,varname,ndim,temp, timestep)
   var(:) = .false.
   where(temp == 1) var = .true.

end subroutine nc_read_logical_1


subroutine nc_read_logical_2(ncid, varname, ndim, var, timestep)
   integer, intent(in) :: ncid
   character(len=*), intent(in) :: varname
   integer, intent(in) :: ndim
   logical, intent(out) :: var(:,:)
   integer, intent(in) :: timestep
   integer, allocatable :: temp(:,:)
   ! Locals:
   integer :: lbs(ndim), ubs(ndim)
   lbs = lbound(var); ubs = ubound(var)
   allocate(temp(lbs(1):ubs(1),lbs(2):ubs(2)))

   call nc_read_var(ncid,varname,ndim,temp, timestep)
   var(:,:) = .false.
   where(temp == 1) var = .true.

end subroutine nc_read_logical_2


subroutine nc_read_logical_3(ncid, varname, ndim, var, timestep)
   integer, intent(in) :: ncid
   character(len=*), intent(in) :: varname
   integer, intent(in) :: ndim
   logical, intent(out) :: var(:,:,:)
   integer, intent(in) :: timestep
   integer, allocatable :: temp(:,:,:)
   ! Locals:
   integer :: lbs(ndim), ubs(ndim)
   lbs = lbound(var); ubs = ubound(var)
   allocate(temp(lbs(1):ubs(1),lbs(2):ubs(2),lbs(3):ubs(3)))

   call nc_read_var(ncid,varname,ndim,temp, timestep)
   var(:,:,:) = .false.
   where(temp == 1) var = .true.

end subroutine nc_read_logical_3


subroutine nc_read_string(ncid, varname, dim_name, var)
   integer, intent(in) :: ncid
   character(len=*), intent(in) :: varname
   character(len=*), intent(in) :: dim_name
   character(len=*), intent(out) :: var
   integer :: var_id, dimid, strlen,i
   character(:), allocatable :: buf

   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   call check(nf90_inq_dimid(ncid, trim(dim_name), dimid))
   call check(nf90_inquire_dimension(ncid, dimid, len=strlen))
   allocate(character(strlen) :: buf); 
   call check(nf90_get_var(ncid, var_id, buf))
   var = ""
   ! netCDF pads with NULs, which trim() keeps; stop at the first one
   do i=1, min(strlen, len(var))
      if (buf(i:i) == achar(0)) exit
      var(i:i) = buf(i:i)
   end do
end subroutine


subroutine nc_write_integer(ncid, ndim, dims, dim_names, var, varname, timestep)
   integer, intent(in) :: ncid
   integer, intent(in) :: ndim
   integer, intent(in) :: dims(ndim)
   character(len=*), dimension(:), intent(in) :: dim_names
   integer, intent(in) :: var(product(dims))
   character(len=*), intent(in) :: varname
   integer, intent(in) :: timestep
   !! locals:
   integer :: var_id, i
   logical :: unlim
   real(8), allocatable :: data_1d(:), data_2d(:, :), data_3d(:, :, :)
   real(8), allocatable :: buffer(:)
   integer, allocatable :: start(:), count(:)
   integer :: ldims(ndim)

   allocate (buffer(product(dims)))
   buffer = var
   ldims = dims
   if (spel_nsets > 1) call nc_select_set1(ndim, ldims, dim_names, buffer, varname)
   if (associated(spel_subset)) call nc_subset_select(ndim, ldims, dim_names, buffer, varname)
   if(timestep > 0) then 
      unlim = .true.
      allocate(start(ndim+1), count(ndim+1))
   else
      unlim = .false.
      allocate(start(ndim), count(ndim))
   endif

   do i = 1, ndim
      start(i) = 1
      count(i) = ldims(i)
   end do

   if (unlim) then 
      start(ndim+1) = timestep
      count(ndim+1) = 1
   endif

   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   select case (ndim)
   case (1)
      allocate (data_1d(ldims(1))); data_1d = reshape(buffer, [ldims(1)])
      call check(nf90_put_var(ncid, var_id, data_1d,start=start,count=count))
   case (2)
      allocate (data_2d(ldims(1), ldims(2))); data_2d = reshape(buffer, [ldims(1), ldims(2)])
      call check(nf90_put_var(ncid, var_id, data_2d,start=start,count=count))
   case (3)
      allocate (data_3d(ldims(1), ldims(2), ldims(3))); data_3d = reshape(buffer, [ldims(1), ldims(2), ldims(3)])
      call check(nf90_put_var(ncid, var_id, data_3d,start=start,count=count))
   case default
      stop "(nc_write_integer) doesn't support >3D doubles"
   end select
end subroutine nc_write_integer


subroutine nc_write_double(ncid, ndim, dims, dim_names, var, varname, timestep)
   integer, intent(in) :: ncid
   integer, intent(in) :: ndim
   integer, intent(in) :: dims(ndim)
   character(len=*), dimension(:), intent(in) :: dim_names
   real(8), intent(in) :: var(product(dims))
   character(len=*), intent(in) :: varname
   integer, intent(in) :: timestep
   !! locals:
   integer :: var_id, i
   logical :: unlim
   real(8), allocatable :: data_1d(:), data_2d(:, :), data_3d(:, :, :)
   real(8), allocatable :: buffer(:)
   integer, allocatable :: start(:), count(:)
   integer :: ldims(ndim)

   allocate (buffer(product(dims)))
   buffer = var
   ldims = dims
   if (spel_nsets > 1) call nc_select_set1(ndim, ldims, dim_names, buffer, varname)
   if (associated(spel_subset)) call nc_subset_select(ndim, ldims, dim_names, buffer, varname)
   if(timestep > 0) then 
      unlim = .true.
      allocate(start(ndim+1), count(ndim+1))
   else
      unlim = .false.
      allocate(start(ndim), count(ndim))
   endif

   do i = 1, ndim
      start(i) = 1
      count(i) = ldims(i)
   end do

   if (unlim) then 
      start(ndim+1) = timestep
      count(ndim+1) = 1
   endif

   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   select case (ndim)
   case (1)
      allocate (data_1d(ldims(1))); data_1d = reshape(buffer, [ldims(1)])
      call check(nf90_put_var(ncid, var_id, data_1d,start=start,count=count))
   case (2)
      allocate (data_2d(ldims(1), ldims(2))); data_2d = reshape(buffer, [ldims(1), ldims(2)])
      call check(nf90_put_var(ncid, var_id, data_2d,start=start,count=count))
   case (3)
      allocate (data_3d(ldims(1), ldims(2), ldims(3))); data_3d = reshape(buffer, [ldims(1), ldims(2), ldims(3)])
      call check(nf90_put_var(ncid, var_id, data_3d,start=start,count=count))
   case default
      stop "(nc_write_double) doesn't support >3D doubles"
   end select
end subroutine nc_write_double


subroutine nc_write_integer_scalar(ncid, var, varname)
   integer, intent(in) :: ncid
   integer, intent(in) :: var
   character(len=*), intent(in) :: varname
   !! locals:
   integer :: var_id

   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   call check(nf90_put_var(ncid, var_id, var))

end subroutine nc_write_integer_scalar

subroutine nc_write_double_scalar(ncid, var, varname)
   integer, intent(in) :: ncid
   real(8), intent(in) :: var
   character(len=*), intent(in) :: varname
   !! locals:
   integer :: var_id

   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   call check(nf90_put_var(ncid, var_id, var))

end subroutine nc_write_double_scalar

subroutine nc_write_string(ncid, var, varname)
  integer, intent(in) :: ncid
   character(len=*), intent(in) :: varname
   character(len=*), intent(in) :: var
   integer :: var_id, i
   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   call check(nf90_put_var(ncid, var_id, trim(var)))
end subroutine nc_write_string

subroutine nc_write_logical(ncid, ndim, dims, dim_names, var, varname, timestep)
   integer, intent(in) :: ncid, ndim
   integer, intent(in) :: dims(ndim)
   character(len=*), dimension(:), intent(in) :: dim_names
   logical, intent(in) :: var(product(dims))
   character(len=*), intent(in) :: varname
   integer, intent(in) :: timestep
   integer :: var_id, i
   integer, allocatable :: int_buf(:)
   integer, allocatable :: data_1d(:), data_2d(:, :), data_3d(:, :, :)
   logical :: unlim
   integer, allocatable :: start(:), count(:)

   integer :: ldims(ndim)

   allocate (int_buf(product(dims)))
   int_buf = merge(1, 0, var)  ! logical → int (1=true, 0=false)
   ldims = dims
   if (spel_nsets > 1) call nc_select_set1(ndim, ldims, dim_names, int_buf, varname)
   if (associated(spel_subset)) call nc_subset_select(ndim, ldims, dim_names, int_buf, varname)
   if(timestep > 0) then 
      unlim = .true.
      allocate(start(ndim+1), count(ndim+1))
   else
      unlim = .false.
      allocate(start(ndim), count(ndim))
   endif

   do i = 1, ndim
      start(i) = 1
      count(i) = ldims(i)
   end do

   if (unlim) then 
      start(ndim+1) = timestep
      count(ndim+1) = 1
   endif

   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   select case (ndim)
   case (1)
      allocate (data_1d(ldims(1))); data_1d = reshape(int_buf, [ldims(1)])
      call check(nf90_put_var(ncid, var_id, data_1d,start=start,count=count))
   case (2)
      allocate (data_2d(ldims(1), ldims(2))); data_2d = reshape(int_buf, [ldims(1), ldims(2)])
      call check(nf90_put_var(ncid, var_id, data_2d,start=start,count=count))
   case (3)
      allocate (data_3d(ldims(1), ldims(2), ldims(3))); data_3d = reshape(int_buf, [ldims(1), ldims(2), ldims(3)])
      call check(nf90_put_var(ncid, var_id, data_3d,start=start,count=count))
   case default
      stop "(nc_write_logical) doesn't support >3D vars"
   end select
end subroutine nc_write_logical

subroutine nc_write_logical_scalar(ncid, var, varname)
   integer, intent(in) :: ncid
   logical, intent(in) :: var
   character(len=*), intent(in) :: varname
   !! locals:
   integer :: var_id, buf
   buf = 0
   if (var) buf = 1
   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   call check(nf90_put_var(ncid, var_id, buf))
end subroutine nc_write_logical_scalar

integer function nc_read_timeslices(fn) result(time_len)
    character(len=*), intent(in) :: fn

    integer :: ncid         ! file ID
    integer :: time_dimid   ! dimension ID for 'time'
    integer :: ierr         ! error status

    ! Open file in read-only mode
    ierr = nf90_open(trim(fn), NF90_NOWRITE, ncid)
    if (ierr /= nf90_noerr) stop "Error opening file"

    ! Inquire about the 'time' dimension
    ierr = nf90_inq_dimid(ncid, "time", time_dimid)
    if (ierr /= nf90_noerr) stop "Error: 'time' dimension not found"

    ! Get the length of the 'time' dimension
    call check(nf90_inquire_dimension(ncid, time_dimid, len=time_len))
    if (ierr /= nf90_noerr) stop "Error inquiring length of 'time' dimension"


    ! Close file
    call check(nf90_close(ncid))

end function nc_read_timeslices

integer function nc_buf_chunk(dt_in_file) result(chunk)
   ! Timesteps per input buffer: the whole file unless SPEL_IO_CHUNK caps it
   integer, intent(in) :: dt_in_file
   character(len=32) :: env
   integer :: stat, n
   chunk = dt_in_file
   call get_environment_variable('SPEL_IO_CHUNK', env, status=stat)
   if (stat == 0) then
      read (env, *, iostat=stat) n
      if (stat == 0 .and. n > 0) chunk = min(n, dt_in_file)
   end if
   print *, 'SPEL input buffer: ', chunk, ' timesteps per read'
end function nc_buf_chunk

subroutine nc_buf_load_double_0(ncid, varname, vshape, buf, t0, nt, chunk, sd, nfile)
   integer, intent(in) :: ncid, vshape(:), t0, nt, chunk
   character(len=*), intent(in) :: varname
   real(8), allocatable, intent(inout) :: buf(:)
   integer, intent(inout) :: sd, nfile
   integer :: var_id, nd, t
   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   sd = 0; nfile = 0
   if (allocated(buf)) then
      if (size(buf) /= chunk) then
         print *, 'nc_buf_load: input shape changed for ', trim(varname)
         error stop
      end if
   else
      allocate (buf(chunk))
      if (spel_buf_on_device) then
         !$acc enter data create(buf)
      end if
   end if
   call check(nf90_inquire_variable(ncid, var_id, ndims=nd))
   if (nd > 0) then
      call check(nf90_get_var(ncid, var_id, buf, start=[t0], count=[nt]))
   else
      ! no time dimension in the file (e.g. scalars): same value every step
      call check(nf90_get_var(ncid, var_id, buf(1)))
      do t = 2, nt
         buf(t) = buf(1)
      end do
   end if
   if (spel_buf_on_device) then
      !$acc update device(buf(1:nt))
   end if
end subroutine nc_buf_load_double_0

subroutine nc_buf_copy_double_0(var, buf, it, sd, nfile)
   real(8), intent(inout) :: var
   real(8), intent(in) :: buf(:)
   integer, intent(in) :: it, sd, nfile
   var = buf(it)
#ifdef _OPENACC
   if (spel_buf_on_device) then
      if (acc_is_present(var, storage_size(var)/8)) then
         !$acc update device(var)
      end if
   end if
#endif
end subroutine nc_buf_copy_double_0

subroutine nc_buf_load_double_1(ncid, varname, vshape, buf, t0, nt, chunk, sd, nfile)
   integer, intent(in) :: ncid, vshape(:), t0, nt, chunk
   character(len=*), intent(in) :: varname
   real(8), allocatable, intent(inout) :: buf(:,:)
   integer, intent(inout) :: sd, nfile
   integer :: var_id, nd, t
   integer :: fshape(1)
   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   call nc_file_subgrid(ncid, var_id, 1, sd, nfile)
   fshape = vshape
   if (sd > 0) then
      if (vshape(sd) /= spel_nsets*nfile) then
         sd = 0
      else
         fshape(sd) = nfile
      end if
   end if
   if (allocated(buf)) then
      if (any(shape(buf) /= [fshape, chunk])) then
         print *, 'nc_buf_load: input shape changed for ', trim(varname)
         error stop
      end if
   else
      allocate (buf(fshape(1), chunk))
      if (spel_buf_on_device) then
         !$acc enter data create(buf)
      end if
   end if
   call check(nf90_inquire_variable(ncid, var_id, ndims=nd))
   if (nd > 1) then
      call check(nf90_get_var(ncid, var_id, buf, start=[1, t0], count=[fshape, nt]))
   else
      ! no time dimension in the file (e.g. scalars): same value every step
      call check(nf90_get_var(ncid, var_id, buf(:, 1)))
      do t = 2, nt
         buf(:, t) = buf(:, 1)
      end do
   end if
   if (spel_buf_on_device) then
      !$acc update device(buf(:, 1:nt))
   end if
end subroutine nc_buf_load_double_1

subroutine nc_buf_copy_double_1(var, buf, it, sd, nfile)
   real(8), intent(inout) :: var(:)
   real(8), intent(in) :: buf(:,:)
   integer, intent(in) :: it, sd, nfile
   integer :: k, nk, o, i1
   logical :: dev
   nk = 1
   if (sd > 0) nk = spel_nsets
   dev = .false.
#ifdef _OPENACC
   if (spel_buf_on_device) dev = acc_is_present(var)
#endif
   if (dev) then
      !$acc parallel loop gang vector collapse(2) private(o) present(var, buf)
      do k = 1, nk
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1)) = buf(i1, it)
      end do
      end do
   else
      do k = 1, nk
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1)) = buf(i1, it)
      end do
      end do
   end if
end subroutine nc_buf_copy_double_1

subroutine nc_buf_load_double_2(ncid, varname, vshape, buf, t0, nt, chunk, sd, nfile)
   integer, intent(in) :: ncid, vshape(:), t0, nt, chunk
   character(len=*), intent(in) :: varname
   real(8), allocatable, intent(inout) :: buf(:,:,:)
   integer, intent(inout) :: sd, nfile
   integer :: var_id, nd, t
   integer :: fshape(2)
   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   call nc_file_subgrid(ncid, var_id, 2, sd, nfile)
   fshape = vshape
   if (sd > 0) then
      if (vshape(sd) /= spel_nsets*nfile) then
         sd = 0
      else
         fshape(sd) = nfile
      end if
   end if
   if (allocated(buf)) then
      if (any(shape(buf) /= [fshape, chunk])) then
         print *, 'nc_buf_load: input shape changed for ', trim(varname)
         error stop
      end if
   else
      allocate (buf(fshape(1), fshape(2), chunk))
      if (spel_buf_on_device) then
         !$acc enter data create(buf)
      end if
   end if
   call check(nf90_inquire_variable(ncid, var_id, ndims=nd))
   if (nd > 2) then
      call check(nf90_get_var(ncid, var_id, buf, start=[1, 1, t0], count=[fshape, nt]))
   else
      ! no time dimension in the file (e.g. scalars): same value every step
      call check(nf90_get_var(ncid, var_id, buf(:, :, 1)))
      do t = 2, nt
         buf(:, :, t) = buf(:, :, 1)
      end do
   end if
   if (spel_buf_on_device) then
      !$acc update device(buf(:, :, 1:nt))
   end if
end subroutine nc_buf_load_double_2

subroutine nc_buf_copy_double_2(var, buf, it, sd, nfile)
   real(8), intent(inout) :: var(:,:)
   real(8), intent(in) :: buf(:,:,:)
   integer, intent(in) :: it, sd, nfile
   integer :: k, nk, o, i1, i2
   logical :: dev
   nk = 1
   if (sd > 0) nk = spel_nsets
   dev = .false.
#ifdef _OPENACC
   if (spel_buf_on_device) dev = acc_is_present(var)
#endif
   if (dev) then
      !$acc parallel loop gang vector collapse(3) private(o) present(var, buf)
      do k = 1, nk
      do i2 = 1, size(buf, 2)
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1), i2 + merge(o, 0, sd == 2)) = buf(i1, i2, it)
      end do
      end do
      end do
   else
      do k = 1, nk
      do i2 = 1, size(buf, 2)
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1), i2 + merge(o, 0, sd == 2)) = buf(i1, i2, it)
      end do
      end do
      end do
   end if
end subroutine nc_buf_copy_double_2

subroutine nc_buf_load_double_3(ncid, varname, vshape, buf, t0, nt, chunk, sd, nfile)
   integer, intent(in) :: ncid, vshape(:), t0, nt, chunk
   character(len=*), intent(in) :: varname
   real(8), allocatable, intent(inout) :: buf(:,:,:,:)
   integer, intent(inout) :: sd, nfile
   integer :: var_id, nd, t
   integer :: fshape(3)
   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   call nc_file_subgrid(ncid, var_id, 3, sd, nfile)
   fshape = vshape
   if (sd > 0) then
      if (vshape(sd) /= spel_nsets*nfile) then
         sd = 0
      else
         fshape(sd) = nfile
      end if
   end if
   if (allocated(buf)) then
      if (any(shape(buf) /= [fshape, chunk])) then
         print *, 'nc_buf_load: input shape changed for ', trim(varname)
         error stop
      end if
   else
      allocate (buf(fshape(1), fshape(2), fshape(3), chunk))
      if (spel_buf_on_device) then
         !$acc enter data create(buf)
      end if
   end if
   call check(nf90_inquire_variable(ncid, var_id, ndims=nd))
   if (nd > 3) then
      call check(nf90_get_var(ncid, var_id, buf, start=[1, 1, 1, t0], count=[fshape, nt]))
   else
      ! no time dimension in the file (e.g. scalars): same value every step
      call check(nf90_get_var(ncid, var_id, buf(:, :, :, 1)))
      do t = 2, nt
         buf(:, :, :, t) = buf(:, :, :, 1)
      end do
   end if
   if (spel_buf_on_device) then
      !$acc update device(buf(:, :, :, 1:nt))
   end if
end subroutine nc_buf_load_double_3

subroutine nc_buf_copy_double_3(var, buf, it, sd, nfile)
   real(8), intent(inout) :: var(:,:,:)
   real(8), intent(in) :: buf(:,:,:,:)
   integer, intent(in) :: it, sd, nfile
   integer :: k, nk, o, i1, i2, i3
   logical :: dev
   nk = 1
   if (sd > 0) nk = spel_nsets
   dev = .false.
#ifdef _OPENACC
   if (spel_buf_on_device) dev = acc_is_present(var)
#endif
   if (dev) then
      !$acc parallel loop gang vector collapse(4) private(o) present(var, buf)
      do k = 1, nk
      do i3 = 1, size(buf, 3)
      do i2 = 1, size(buf, 2)
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1), i2 + merge(o, 0, sd == 2), i3 + merge(o, 0, sd == 3)) = buf(i1, i2, i3, it)
      end do
      end do
      end do
      end do
   else
      do k = 1, nk
      do i3 = 1, size(buf, 3)
      do i2 = 1, size(buf, 2)
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1), i2 + merge(o, 0, sd == 2), i3 + merge(o, 0, sd == 3)) = buf(i1, i2, i3, it)
      end do
      end do
      end do
      end do
   end if
end subroutine nc_buf_copy_double_3

subroutine nc_buf_load_integer_0(ncid, varname, vshape, buf, t0, nt, chunk, sd, nfile)
   integer, intent(in) :: ncid, vshape(:), t0, nt, chunk
   character(len=*), intent(in) :: varname
   integer, allocatable, intent(inout) :: buf(:)
   integer, intent(inout) :: sd, nfile
   integer :: var_id, nd, t
   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   sd = 0; nfile = 0
   if (allocated(buf)) then
      if (size(buf) /= chunk) then
         print *, 'nc_buf_load: input shape changed for ', trim(varname)
         error stop
      end if
   else
      allocate (buf(chunk))
      if (spel_buf_on_device) then
         !$acc enter data create(buf)
      end if
   end if
   call check(nf90_inquire_variable(ncid, var_id, ndims=nd))
   if (nd > 0) then
      call check(nf90_get_var(ncid, var_id, buf, start=[t0], count=[nt]))
   else
      ! no time dimension in the file (e.g. scalars): same value every step
      call check(nf90_get_var(ncid, var_id, buf(1)))
      do t = 2, nt
         buf(t) = buf(1)
      end do
   end if
   if (spel_buf_on_device) then
      !$acc update device(buf(1:nt))
   end if
end subroutine nc_buf_load_integer_0

subroutine nc_buf_copy_integer_0(var, buf, it, sd, nfile)
   integer, intent(inout) :: var
   integer, intent(in) :: buf(:)
   integer, intent(in) :: it, sd, nfile
   var = buf(it)
#ifdef _OPENACC
   if (spel_buf_on_device) then
      if (acc_is_present(var, storage_size(var)/8)) then
         !$acc update device(var)
      end if
   end if
#endif
end subroutine nc_buf_copy_integer_0

subroutine nc_buf_load_integer_1(ncid, varname, vshape, buf, t0, nt, chunk, sd, nfile)
   integer, intent(in) :: ncid, vshape(:), t0, nt, chunk
   character(len=*), intent(in) :: varname
   integer, allocatable, intent(inout) :: buf(:,:)
   integer, intent(inout) :: sd, nfile
   integer :: var_id, nd, t
   integer :: fshape(1)
   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   call nc_file_subgrid(ncid, var_id, 1, sd, nfile)
   fshape = vshape
   if (sd > 0) then
      if (vshape(sd) /= spel_nsets*nfile) then
         sd = 0
      else
         fshape(sd) = nfile
      end if
   end if
   if (allocated(buf)) then
      if (any(shape(buf) /= [fshape, chunk])) then
         print *, 'nc_buf_load: input shape changed for ', trim(varname)
         error stop
      end if
   else
      allocate (buf(fshape(1), chunk))
      if (spel_buf_on_device) then
         !$acc enter data create(buf)
      end if
   end if
   call check(nf90_inquire_variable(ncid, var_id, ndims=nd))
   if (nd > 1) then
      call check(nf90_get_var(ncid, var_id, buf, start=[1, t0], count=[fshape, nt]))
   else
      ! no time dimension in the file (e.g. scalars): same value every step
      call check(nf90_get_var(ncid, var_id, buf(:, 1)))
      do t = 2, nt
         buf(:, t) = buf(:, 1)
      end do
   end if
   if (spel_buf_on_device) then
      !$acc update device(buf(:, 1:nt))
   end if
end subroutine nc_buf_load_integer_1

subroutine nc_buf_copy_integer_1(var, buf, it, sd, nfile)
   integer, intent(inout) :: var(:)
   integer, intent(in) :: buf(:,:)
   integer, intent(in) :: it, sd, nfile
   integer :: k, nk, o, i1
   logical :: dev
   nk = 1
   if (sd > 0) nk = spel_nsets
   dev = .false.
#ifdef _OPENACC
   if (spel_buf_on_device) dev = acc_is_present(var)
#endif
   if (dev) then
      !$acc parallel loop gang vector collapse(2) private(o) present(var, buf)
      do k = 1, nk
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1)) = buf(i1, it)
      end do
      end do
   else
      do k = 1, nk
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1)) = buf(i1, it)
      end do
      end do
   end if
end subroutine nc_buf_copy_integer_1

subroutine nc_buf_load_integer_2(ncid, varname, vshape, buf, t0, nt, chunk, sd, nfile)
   integer, intent(in) :: ncid, vshape(:), t0, nt, chunk
   character(len=*), intent(in) :: varname
   integer, allocatable, intent(inout) :: buf(:,:,:)
   integer, intent(inout) :: sd, nfile
   integer :: var_id, nd, t
   integer :: fshape(2)
   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   call nc_file_subgrid(ncid, var_id, 2, sd, nfile)
   fshape = vshape
   if (sd > 0) then
      if (vshape(sd) /= spel_nsets*nfile) then
         sd = 0
      else
         fshape(sd) = nfile
      end if
   end if
   if (allocated(buf)) then
      if (any(shape(buf) /= [fshape, chunk])) then
         print *, 'nc_buf_load: input shape changed for ', trim(varname)
         error stop
      end if
   else
      allocate (buf(fshape(1), fshape(2), chunk))
      if (spel_buf_on_device) then
         !$acc enter data create(buf)
      end if
   end if
   call check(nf90_inquire_variable(ncid, var_id, ndims=nd))
   if (nd > 2) then
      call check(nf90_get_var(ncid, var_id, buf, start=[1, 1, t0], count=[fshape, nt]))
   else
      ! no time dimension in the file (e.g. scalars): same value every step
      call check(nf90_get_var(ncid, var_id, buf(:, :, 1)))
      do t = 2, nt
         buf(:, :, t) = buf(:, :, 1)
      end do
   end if
   if (spel_buf_on_device) then
      !$acc update device(buf(:, :, 1:nt))
   end if
end subroutine nc_buf_load_integer_2

subroutine nc_buf_copy_integer_2(var, buf, it, sd, nfile)
   integer, intent(inout) :: var(:,:)
   integer, intent(in) :: buf(:,:,:)
   integer, intent(in) :: it, sd, nfile
   integer :: k, nk, o, i1, i2
   logical :: dev
   nk = 1
   if (sd > 0) nk = spel_nsets
   dev = .false.
#ifdef _OPENACC
   if (spel_buf_on_device) dev = acc_is_present(var)
#endif
   if (dev) then
      !$acc parallel loop gang vector collapse(3) private(o) present(var, buf)
      do k = 1, nk
      do i2 = 1, size(buf, 2)
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1), i2 + merge(o, 0, sd == 2)) = buf(i1, i2, it)
      end do
      end do
      end do
   else
      do k = 1, nk
      do i2 = 1, size(buf, 2)
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1), i2 + merge(o, 0, sd == 2)) = buf(i1, i2, it)
      end do
      end do
      end do
   end if
end subroutine nc_buf_copy_integer_2

subroutine nc_buf_load_integer_3(ncid, varname, vshape, buf, t0, nt, chunk, sd, nfile)
   integer, intent(in) :: ncid, vshape(:), t0, nt, chunk
   character(len=*), intent(in) :: varname
   integer, allocatable, intent(inout) :: buf(:,:,:,:)
   integer, intent(inout) :: sd, nfile
   integer :: var_id, nd, t
   integer :: fshape(3)
   call check(nf90_inq_varid(ncid, trim(varname), var_id), varname)
   call nc_file_subgrid(ncid, var_id, 3, sd, nfile)
   fshape = vshape
   if (sd > 0) then
      if (vshape(sd) /= spel_nsets*nfile) then
         sd = 0
      else
         fshape(sd) = nfile
      end if
   end if
   if (allocated(buf)) then
      if (any(shape(buf) /= [fshape, chunk])) then
         print *, 'nc_buf_load: input shape changed for ', trim(varname)
         error stop
      end if
   else
      allocate (buf(fshape(1), fshape(2), fshape(3), chunk))
      if (spel_buf_on_device) then
         !$acc enter data create(buf)
      end if
   end if
   call check(nf90_inquire_variable(ncid, var_id, ndims=nd))
   if (nd > 3) then
      call check(nf90_get_var(ncid, var_id, buf, start=[1, 1, 1, t0], count=[fshape, nt]))
   else
      ! no time dimension in the file (e.g. scalars): same value every step
      call check(nf90_get_var(ncid, var_id, buf(:, :, :, 1)))
      do t = 2, nt
         buf(:, :, :, t) = buf(:, :, :, 1)
      end do
   end if
   if (spel_buf_on_device) then
      !$acc update device(buf(:, :, :, 1:nt))
   end if
end subroutine nc_buf_load_integer_3

subroutine nc_buf_copy_integer_3(var, buf, it, sd, nfile)
   integer, intent(inout) :: var(:,:,:)
   integer, intent(in) :: buf(:,:,:,:)
   integer, intent(in) :: it, sd, nfile
   integer :: k, nk, o, i1, i2, i3
   logical :: dev
   nk = 1
   if (sd > 0) nk = spel_nsets
   dev = .false.
#ifdef _OPENACC
   if (spel_buf_on_device) dev = acc_is_present(var)
#endif
   if (dev) then
      !$acc parallel loop gang vector collapse(4) private(o) present(var, buf)
      do k = 1, nk
      do i3 = 1, size(buf, 3)
      do i2 = 1, size(buf, 2)
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1), i2 + merge(o, 0, sd == 2), i3 + merge(o, 0, sd == 3)) = buf(i1, i2, i3, it)
      end do
      end do
      end do
      end do
   else
      do k = 1, nk
      do i3 = 1, size(buf, 3)
      do i2 = 1, size(buf, 2)
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1), i2 + merge(o, 0, sd == 2), i3 + merge(o, 0, sd == 3)) = buf(i1, i2, i3, it)
      end do
      end do
      end do
      end do
   end if
end subroutine nc_buf_copy_integer_3

subroutine nc_buf_copy_logical_0(var, buf, it, sd, nfile)
   logical, intent(inout) :: var
   integer, intent(in) :: buf(:)
   integer, intent(in) :: it, sd, nfile
   var = buf(it) == 1
#ifdef _OPENACC
   if (spel_buf_on_device) then
      if (acc_is_present(var, storage_size(var)/8)) then
         !$acc update device(var)
      end if
   end if
#endif
end subroutine nc_buf_copy_logical_0

subroutine nc_buf_copy_logical_1(var, buf, it, sd, nfile)
   logical, intent(inout) :: var(:)
   integer, intent(in) :: buf(:,:)
   integer, intent(in) :: it, sd, nfile
   integer :: k, nk, o, i1
   logical :: dev
   nk = 1
   if (sd > 0) nk = spel_nsets
   dev = .false.
#ifdef _OPENACC
   if (spel_buf_on_device) dev = acc_is_present(var)
#endif
   if (dev) then
      !$acc parallel loop gang vector collapse(2) private(o) present(var, buf)
      do k = 1, nk
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1)) = buf(i1, it) == 1
      end do
      end do
   else
      do k = 1, nk
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1)) = buf(i1, it) == 1
      end do
      end do
   end if
end subroutine nc_buf_copy_logical_1

subroutine nc_buf_copy_logical_2(var, buf, it, sd, nfile)
   logical, intent(inout) :: var(:,:)
   integer, intent(in) :: buf(:,:,:)
   integer, intent(in) :: it, sd, nfile
   integer :: k, nk, o, i1, i2
   logical :: dev
   nk = 1
   if (sd > 0) nk = spel_nsets
   dev = .false.
#ifdef _OPENACC
   if (spel_buf_on_device) dev = acc_is_present(var)
#endif
   if (dev) then
      !$acc parallel loop gang vector collapse(3) private(o) present(var, buf)
      do k = 1, nk
      do i2 = 1, size(buf, 2)
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1), i2 + merge(o, 0, sd == 2)) = buf(i1, i2, it) == 1
      end do
      end do
      end do
   else
      do k = 1, nk
      do i2 = 1, size(buf, 2)
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1), i2 + merge(o, 0, sd == 2)) = buf(i1, i2, it) == 1
      end do
      end do
      end do
   end if
end subroutine nc_buf_copy_logical_2

subroutine nc_buf_copy_logical_3(var, buf, it, sd, nfile)
   logical, intent(inout) :: var(:,:,:)
   integer, intent(in) :: buf(:,:,:,:)
   integer, intent(in) :: it, sd, nfile
   integer :: k, nk, o, i1, i2, i3
   logical :: dev
   nk = 1
   if (sd > 0) nk = spel_nsets
   dev = .false.
#ifdef _OPENACC
   if (spel_buf_on_device) dev = acc_is_present(var)
#endif
   if (dev) then
      !$acc parallel loop gang vector collapse(4) private(o) present(var, buf)
      do k = 1, nk
      do i3 = 1, size(buf, 3)
      do i2 = 1, size(buf, 2)
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1), i2 + merge(o, 0, sd == 2), i3 + merge(o, 0, sd == 3)) = buf(i1, i2, i3, it) == 1
      end do
      end do
      end do
      end do
   else
      do k = 1, nk
      do i3 = 1, size(buf, 3)
      do i2 = 1, size(buf, 2)
      do i1 = 1, size(buf, 1)
         o = (k-1)*nfile
         var(i1 + merge(o, 0, sd == 1), i2 + merge(o, 0, sd == 2), i3 + merge(o, 0, sd == 3)) = buf(i1, i2, i3, it) == 1
      end do
      end do
      end do
      end do
   end if
end subroutine nc_buf_copy_logical_3
end module nc_io
