module field_mod
   use iso_fortran_env, only: real64
   implicit none

   integer, parameter :: rkind = real64

   type, public:: field_desc_t
      real(rkind), pointer :: a1d(:) => null()
      real(rkind), pointer :: a2d(:, :) => null()
      real(rkind), pointer :: a3d(:, :, :) => null()
      real(rkind), pointer :: a4d(:, :, :, :) => null()
      integer :: rank = 0
      integer :: n_mask = 0
      integer, pointer :: mask(:) => null()
      ! Flattened size this field contributes per masked index (e.g. nlev)
      integer :: per_index_size = 0
      ! Offset of this field in the flat feature/target vector
      integer :: offset = 0
   contains
      procedure :: map => mask_field
      procedure :: remap => unmask_field
   end type field_desc_t

   type, public :: field_list_t
      type(field_desc_t), allocatable :: fields(:)
      integer :: n = 0
      logical :: initialized = .false.
   contains
      generic  :: add => add_1d, add_2d, add_3d, add_4d

      procedure, private :: add_1d
      procedure, private :: add_2d
      procedure, private :: add_3d
      procedure, private :: add_4d

      procedure, private :: append_field
      procedure :: init => initialize_field_list
      procedure :: compute_layout

   end type field_list_t

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

   subroutine add_4d(this, arr, numf, filter)
      class(field_list_t), intent(inout) :: this
      real(rkind), pointer, intent(inout) :: arr(:, :, :, :)
      integer(ikind), intent(in) :: numf
      integer(ikind), pointer, intent(in) :: filter(:)
      type(field_desc_t) :: f

      f%a4d => arr
      f%rank = 4
      f%n_mask = numf
      f%mask => filter

      call this%append_field(f)
   end subroutine add_4d

   subroutine append_field(this, f)
      class(field_list_t), intent(inout) :: this
      type(field_desc_t), intent(in)    :: f
      integer :: n_old
    !!
      if (.not. this%initialized) error stop "Tried to append to un-initialized field list"
      n_old = this%n
      this%fields(n_old) = f
      this%n = this%n + 1
   end subroutine append_field

   subroutine initialize_field_list(this, nfields)
      class(field_list_t), intent(inout) :: this
      integer, intent(in) :: nfields
      allocate (this%fields(nfields))
      this%n = 1
      this%initialized = .true.
   end subroutine initialize_field_list

   function compute_layout(this) result(total_dim)
    !! Function returns the total size for the needed buffer
      type(field_list_t), intent(inout) :: this
      integer(ikind) :: total_dim
      integer :: i
      integer :: per_sz, offset

      total_dim = 0
      offset = 0

      do i = 1, this%n
         select case (this%fields(i)%rank)
         case (1)
            per_sz = 1
         case (2)
            per_sz = size(this%fields(i)%a2d, 2)
         case (3)
            per_sz = size(this%fields(i)%a3d, 2)*size(this%fields(i)%a3d, 3)
         case (4)
            per_sz = size(this%fields(i)%a4d, 2)*size(this%fields(i)%a4d, 3)* &
                     size(this%fields(i)%a4d, 4)
         case default
            stop "Unsupported rank in compute_layout"
         end select

         this%fields(i)%per_index_size = per_sz
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

      do i_field = 1, this%n
         numf = this%fields(i_field)%n_mask
         per_sz = this%fields(i_field)%per_index_size
         base = this%fields(i_field)%offset

         select case (this%fields(i_field)%rank)
         case (3)
            lo2 = lbound(list%fields(i_field)%a3d, 2)
            hi2 = ubound(list%fields(i_field)%a3d, 2)
            lo3 = lbound(list%fields(i_field)%a3d, 3)
            hi3 = ubound(list%fields(i_field)%a3d, 3)
            n3 = hi3 - lo3 + 1
            do i_idx = 1, numf
               col = list%fields(i_field)%mask(i_idx)
               do concurrent(j2=lo2:hi2, j3=lo3:hi3)
                  k = (j2 - lo2)*n3 + (j3 - lo3) + 1
                  buf(i_idx, base + k) = list%fields(i_field)%a3d(col, j2, j3)
               end do
            end do
         case (2)
            lo2 = lbound(this%fields(i_field)%a2d, 2)
            hi2 = ubound(this%fields(i_field)%a2d, 2)

            do i_idx = 1, numf
               col = this%fields(i_field)%mask(i_idx)
               do concurrent(j=lo2:hi2)
                  k = j - lo2 + 1
                  buf(i_idx, base + k) = list%fields(i_field)%a2d(col, j)
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

   subroutine unmap_buffer_to_fields(this, n_index, buf)
      class(field_list_t), intent(inout) :: this
      integer, intent(in)    :: n_index
      real(rkind), intent(in)    :: buf(:, :)

      integer :: i_field, i_idx, col
      integer :: j2, j3, lo2, hi2, lo3, hi3
      integer :: per_sz, base, n3, k

      do i_field = 1, this%n
         per_sz = this%fields(i_field)%per_index_size
         base = this%fields(i_field)%offset

         select case (this%fields(i_field)%rank)
         case (3)
            lo2 = lbound(list%fields(i_field)%a3d, 2)
            hi2 = ubound(list%fields(i_field)%a3d, 2)
            lo3 = lbound(list%fields(i_field)%a3d, 3)
            hi3 = ubound(list%fields(i_field)%a3d, 3)
            n3 = hi3 - lo3 + 1

            do i_idx = 1, n_index
               col = list%fields(i_field)%mask(i_idx)
               do concurrent(j2=lo2:hi2, j3=lo3:hi3)
                  k = (j2 - lo2)*n3 + (j3 - lo3) + 1
                  list%fields(i_field)%a3d(col, j2, j3) = buf(i_idx, base + k)
               end do
            end do
         case (2)
            lo2 = lbound(this%fields(i_field)%a2d, 2)
            hi2 = ubound(this%fields(i_field)%a2d, 2)

            do i_idx = 1, n_index
               col = this%fields(i_field)%mask(i_idx)
               do concurrent(j=lo2:hi2)
                  k = j - lo2 + 1
                  this%fields(i_field)%a2d(col, j) = buf(i_idx, base + k)
               end do
            end do

         case (1)
            do i_idx = 1, n_index
               col = this%fields(i_field)%mask(i_idx)
               this%fields(i_field)%a1d(col) = buf(i_idx, base + 1)
            end do

         case default
            stop "Unsupported rank in unmap_buffer_to_fields"
         end select
      end do
   end subroutine unmap_buffer_to_fields
end module field_mod
