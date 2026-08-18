module field_mod
   use kinds, only: rkind, ikind
   use iso_fortran_env, only: output_unit
   implicit none

   private
   type, public:: field_desc_t
      !! Describes how a field maps to an FTorch Tensor
      !! - a1d, a2d, a3d: pointer to underlying model variable
      !! - rank: dimension of field
      !! - n_mask: number of active elements for this field
      !! - mask: index to active elements
      !! - sample_map: maps active values to samples
      !! - per_sample_size: flattened size this field contributes per masked index (e.g. nlev)
      !! - offset: offset of this field in the flat feature/target vector
      real(rkind), pointer, contiguous :: a1d(:) => null()
      real(rkind), pointer, contiguous :: a2d(:, :) => null()
      real(rkind), pointer, contiguous :: a3d(:, :, :) => null()
      integer(ikind) :: rank = 0
      integer(ikind) :: n_mask = 0
      integer(ikind), pointer :: mask(:) => null()

      integer(ikind), pointer,contiguous :: sample_map(:) =>null()

      integer :: per_sample_size = 0
      ! so that buf(base+offset) = field(1)
      integer :: offset = 0
   end type field_desc_t

   type :: field_list_t
      logical :: initialized = .false.
      integer(ikind) :: n_fields = 0
      integer(ikind) :: n_samples = 0
      character(len=256) :: name = ""
      type(field_desc_t), allocatable :: fields(:)
   contains
      generic  :: add => add_1d, add_2d, add_3d

      procedure, private :: add_1d
      procedure, private :: add_2d
      procedure, private :: add_3d

      procedure, private :: append_field
      procedure :: compute_layout
      procedure :: print => print_field_list
      procedure :: write_formatted
      generic :: write (formatted) => write_formatted
      procedure :: map_to_buffer => map_fields_to_buffer
      procedure :: map_from_buffer => unmap_buffer_to_fields
   end type field_list_t

   interface field_list_t
      module procedure initialize_field_list
   end interface field_list_t

   public :: field_list_t

contains

   subroutine add_1d(this, arr, numf, filter)
      class(field_list_t), intent(inout) :: this
      real(rkind), pointer, intent(inout) :: arr(:)
      integer(ikind), intent(in) :: numf
      integer(ikind), pointer, intent(in) :: filter(:)
      type(field_desc_t) :: f

      f%a1d => arr
      f%rank = 1
      f%n_mask = numf
      f%mask => filter

      call this%append_field(f)
   end subroutine add_1d

   subroutine add_2d(this, arr, numf, filter)
      class(field_list_t), intent(inout) :: this
      real(rkind), pointer, intent(inout) :: arr(:, :)
      integer(ikind), intent(in) :: numf
      integer(ikind), pointer, intent(in) :: filter(:)
      type(field_desc_t) :: f

      f%a2d => arr
      f%rank = 2
      f%n_mask = numf
      f%mask => filter

      call this%append_field(f)
   end subroutine add_2d

   subroutine add_3d(this, arr, numf, filter)
      class(field_list_t), intent(inout) :: this
      real(rkind), pointer, intent(inout) :: arr(:, :, :)
      integer(ikind), intent(in) :: numf
      integer(ikind), pointer, intent(in) :: filter(:)
      type(field_desc_t) :: f

      f%a3d => arr
      f%rank = 3
      f%n_mask = numf
      f%mask => filter

      call this%append_field(f)
   end subroutine add_3d

   subroutine append_field(this, f)
      class(field_list_t), intent(inout) :: this
      type(field_desc_t), intent(in)    :: f
      if (.not. this%initialized) error stop "Tried to append to un-initialized field list"
      this%n_fields = this%n_fields + 1
      this%fields(this%n_fields) = f
   end subroutine append_field

   function initialize_field_list(nfields, name, n_samples) result(field_list)
      integer(ikind), intent(in) :: nfields
      character(len=*), intent(in) :: name
      integer(ikind), intent(in) :: n_samples

      type(field_list_t):: field_list

      field_list%name = name

      allocate (field_list%fields(nfields))
      field_list%n_fields = 0_ikind
      field_list%n_samples = n_samples
      field_list%initialized = .true.
   end function initialize_field_list

   function compute_layout(this) result(total_dim)
      !! Function returns the total size for the needed buffer
      class(field_list_t), intent(inout) :: this
      integer(ikind) :: total_dim
      integer :: i
      integer :: per_sz, offset

      total_dim = 0
      offset = 0

      do i = 1, this%n_fields
         select case (this%fields(i)%rank)
         case (1)
            per_sz = 1
         case (2)
            per_sz = size(this%fields(i)%a2d, 2)
         case (3)
            per_sz = size(this%fields(i)%a3d, 2)*size(this%fields(i)%a3d, 3)
         case default
            stop "Unsupported rank in compute_layout"
         end select

         this%fields(i)%per_sample_size = per_sz
         this%fields(i)%offset = offset
         offset = offset + per_sz
         total_dim = total_dim + per_sz
      end do
   end function compute_layout

   subroutine map_fields_to_buffer(this, buf)
      class(field_list_t), intent(in)    :: this
      real(rkind), intent(inout) :: buf(:, :)

      integer(ikind)    :: numf
      integer :: i_field, i_idx, col, lo2, hi2
      integer :: n3, lo3, hi3, k, base, per_sz
      integer :: j, j2, j3

      do i_field = 1, this%n_fields
         numf = this%fields(i_field)%n_mask
         per_sz = this%fields(i_field)%per_sample_size
         base = this%fields(i_field)%offset

         select case (this%fields(i_field)%rank)
         case (3)
            lo2 = lbound(this%fields(i_field)%a3d, 2)
            hi2 = ubound(this%fields(i_field)%a3d, 2)
            lo3 = lbound(this%fields(i_field)%a3d, 3)
            hi3 = ubound(this%fields(i_field)%a3d, 3)
            n3 = hi3 - lo3 + 1
            do i_idx = 1, numf
               col = this%fields(i_field)%mask(i_idx)
               do concurrent(j2=lo2:hi2, j3=lo3:hi3)
                  k = (j2 - lo2)*n3 + (j3 - lo3) + 1
                  buf(i_idx, base + k) = this%fields(i_field)%a3d(col, j2, j3)
               end do
            end do
         case (2)
            lo2 = lbound(this%fields(i_field)%a2d, 2)
            hi2 = ubound(this%fields(i_field)%a2d, 2)

            do i_idx = 1, numf
               col = this%fields(i_field)%mask(i_idx)
               do concurrent(j=lo2:hi2)
                  k = j - lo2 + 1
                  buf(i_idx, base + k) = this%fields(i_field)%a2d(col, j)
               end do
            end do

         case (1)
            ! Example: 1D field already indexed by mask: a1d(mask(i_idx))
            do i_idx = 1, numf
               col = this%fields(i_field)%mask(i_idx)
               buf(i_idx, base + 1) = this%fields(i_field)%a1d(col)
            end do

         case default
            stop "Unsupported rank in map_fields_to_buffer"
         end select
      end do
   end subroutine map_fields_to_buffer

   subroutine unmap_buffer_to_fields(this, n_samples, buf)
      class(field_list_t), intent(inout) :: this
      integer(ikind), intent(in)    :: n_samples
      real(rkind), intent(in)    :: buf(:, :)

      integer :: i_field, i_idx, col
      integer :: j, j2, j3, lo2, hi2, lo3, hi3
      integer :: per_sz, base, n3, k

      do i_field = 1, this%n_fields
         per_sz = this%fields(i_field)%per_sample_size
         base = this%fields(i_field)%offset

         select case (this%fields(i_field)%rank)
         case (3)
            lo2 = lbound(this%fields(i_field)%a3d, 2)
            hi2 = ubound(this%fields(i_field)%a3d, 2)
            lo3 = lbound(this%fields(i_field)%a3d, 3)
            hi3 = ubound(this%fields(i_field)%a3d, 3)
            n3 = hi3 - lo3 + 1

            do i_idx = 1, n_samples
               col = this%fields(i_field)%mask(i_idx)
               do concurrent(j2=lo2:hi2, j3=lo3:hi3)
                  k = (j2 - lo2)*n3 + (j3 - lo3) + 1
                  this%fields(i_field)%a3d(col, j2, j3) = buf(i_idx, base + k)
               end do
            end do
         case (2)
            lo2 = lbound(this%fields(i_field)%a2d, 2)
            hi2 = ubound(this%fields(i_field)%a2d, 2)

            do i_idx = 1, n_samples
               col = this%fields(i_field)%mask(i_idx)
               do concurrent(j=lo2:hi2)
                  k = j - lo2 + 1
                  this%fields(i_field)%a2d(col, j) = buf(i_idx, base + k)
               end do
            end do

         case (1)
            do i_idx = 1, n_samples
               col = this%fields(i_field)%mask(i_idx)
               this%fields(i_field)%a1d(col) = buf(i_idx, base + 1)
            end do

         case default
            stop "Unsupported rank in unmap_buffer_to_fields"
         end select
      end do
   end subroutine unmap_buffer_to_fields

   subroutine print_field_list(self, ounit)
      class(field_list_t), intent(in) :: self
      integer, intent(in), optional :: ounit

      integer :: i_field, unit
      character(len=1) :: endl = new_line('a')

      unit = output_unit
      if (present(ounit)) unit = ounit
      write (unit, *) "Field list "//trim(self%name)//endl
      do i_field = 1, self%n_fields
         call print_field(self%fields(i_field))
      end do
   contains

      subroutine print_field(this)
         type(field_desc_t), intent(in) :: this

         integer :: u
         integer :: i, j, k, l

         write (unit, '(a)') "field_desc_t"
         write (unit, '(a,i0)') "  rank            : ", this%rank
         write (unit, '(a,i0)') "  n_mask          : ", this%n_mask
         write (unit, '(a,i0)') "  per_sample_size : ", this%per_sample_size
         write (unit, '(a,i0)') "  offset          : ", this%offset

         if (associated(this%mask)) then
            write (unit, '(a)', advance='no') "  mask            : ["
            do i = 1, size(this%mask)
               if (i > 1) write (unit, '(a)', advance='no') ", "
               write (unit, '(i0)', advance='no') this%mask(i)
            end do
            write (unit, '(a)') "]"
         else
            write (unit, '(a)') "  mask            : <not associated>"
         end if

         select case (this%rank)

         case (1)
            if (.not. associated(this%a1d)) then
               write (unit, '(a)') "  data            : <not associated>"
               return
            end if

            write (unit, '(a,i0,a)') "  shape           : [", size(this%a1d), "]"
            write (unit, '(a)') "  data:"
            do i = lbound(this%a1d, 1), ubound(this%a1d, 1)
               write (unit, '(a,i0,a,es16.8)') &
                  "    (", i, ") = ", this%a1d(i)
            end do

         case (2)
            if (.not. associated(this%a2d)) then
               write (unit, '(a)') "  data            : <not associated>"
               return
            end if

            write (unit, '(a,i0,a,i0,a)') &
               "  shape           : [", size(this%a2d, 1), ", ", size(this%a2d, 2), "]"

            write (unit, '(a)') "  data:"
            do i = lbound(this%a2d, 1), ubound(this%a2d, 1)
               write (unit, '(a,i0,a)', advance='no') "    row ", i, ": "
               do j = lbound(this%a2d, 2), ubound(this%a2d, 2)
                  write (unit, '(es16.8,1x)', advance='no') this%a2d(i, j)
               end do
               write (unit, *)
            end do

         case (3)
            if (.not. associated(this%a3d)) then
               write (unit, '(a)') "  data            : <not associated>"
               return
            end if

            write (unit, '(a,i0,a,i0,a,i0,a)') &
               "  shape           : [", size(this%a3d, 1), ", ", &
               size(this%a3d, 2), ", ", &
               size(this%a3d, 3), "]"

            write (unit, '(a)') "  data:"
            do i = lbound(this%a3d, 1), ubound(this%a3d, 1)
               write (unit, '(a,i0)') "    sample/index ", i
               do j = lbound(this%a3d, 2), ubound(this%a3d, 2)
                  write (unit, '(a,i0,a)', advance='no') "      dim2=", j, ": "
                  do k = lbound(this%a3d, 3), ubound(this%a3d, 3)
                     write (unit, '(es16.8,1x)', advance='no') this%a3d(i, j, k)
                  end do
                  write (unit, *)
               end do
            end do

         case default
            write (unit, '(a,i0)') "  data            : <unsupported rank ", this%rank
         end select

      end subroutine print_field

   end subroutine print_field_list

   subroutine write_formatted(self, unit, iotype, v_list, iostat, iomsg)
      class(field_list_t), intent(in) :: self
      integer, intent(in) :: unit
      character(len=*), intent(in) :: iotype
      integer, intent(in) :: v_list(:)
      integer, intent(out) :: iostat
      character(len=*), intent(inout) :: iomsg

      iostat = 0
      iomsg = ""
      call self%print(unit)
      if (iostat /= 0) return
   end subroutine write_formatted
end module field_mod
