program test_tiny_emulator

   use kinds, only: rkind, ikind
   use emulator_mod, only: emulator_t
   use field_mod, only: field_list_t

   implicit none

   interface populate_by_mask
      procedure :: populate_by_mask_1d
      procedure :: populate_by_mask_2d
   end interface

   type :: field_storage_t
      real(rkind), pointer, contiguous :: a1d(:) => null()
      real(rkind), pointer, contiguous :: a2d(:, :) => null()
   end type field_storage_t

   type(emulator_t) :: emulator
   type(field_list_t) :: inputs
   type(field_list_t) :: outputs
   type(field_storage_t), allocatable :: storage(:)

   integer(ikind), parameter :: n_features = 4
   integer(ikind), parameter :: beg = 1, end = 5
   integer(ikind), parameter :: nlevs = 2

   real(rkind), pointer, contiguous :: y(:)

   integer(ikind), parameter :: n_mask = 3
   integer(ikind), target :: mask_storage(n_mask)
   integer(ikind), pointer :: mask(:)

   character(len=*), parameter :: test_path = "../../unit-tests/input-data/tiny_case/"
   character(len=*), parameter :: model_path = trim(test_path)//"spel_emulator_torchscript.pt"

   real(rkind), parameter :: canonical_input(3, 4) = reshape( &
                             [1.0_rkind, 2.0_rkind, -1.0_rkind, &
                              2.0_rkind, 4.0_rkind, 0.0_rkind, &
                              3.0_rkind, 7.0_rkind, 1.0_rkind, &
                              4.0_rkind, 8.0_rkind, 2.0_rkind], &
                             [n_mask, n_features] &
                             )

   ! ------------------------------------------------------------
   ! Set up pointers expected by field_list_t
   ! ------------------------------------------------------------

   allocate (y(beg:end))
   mask => mask_storage

   ! Select physical indices 2, 4, and 5.
   mask = [2_ikind, 4_ikind, 5_ikind]

   ! output
   y = -999.0_rkind

   ! ------------------------------------------------------------
   ! These three masked positions reproduce the Python input:
   !   x1  x2 x3  x4
   ! [ 1,  2, 3, 4 ]
   ! [ 2,  4, 7, 8 ]
   ! [-1,  0, 1, 2 ]
   ! ------------------------------------------------------------
   ! ------------------------------------------------------------
   ! Construct input/output field lists.
   ! ------------------------------------------------------------

   call decompose_to_input_list(decomp_dims=[1_ikind, 2_ikind, 1_ikind], &
                                inputs=inputs, storage=storage)
   call inputs%print()

   outputs = field_list_t(1_ikind, "outputs")
   call outputs%add(y, n_mask, mask)

   ! ------------------------------------------------------------
   ! Load TorchScript model and run inference.
   ! ------------------------------------------------------------

   call emulator%init(trim(model_path), inputs, outputs)
   call emulator%infer(inputs, outputs)

   print *, "FTorch results:"
   print *, "y(2) = ", y(2)
   print *, "y(4) = ", y(4)
   print *, "y(5) = ", y(5)

contains

   subroutine populate_by_mask_1d(arr, numf, mask, vals)
      real(rkind), intent(inout) :: arr(:)
      integer(ikind), intent(in) :: numf, mask(1:numf)
      real(rkind), intent(in) :: vals(:)
      integer :: i, n
      if (.not. size(arr) .ge. numf) error stop "(1d) numf > total array size"
      do concurrent(i=1:numf)
         n = mask(i)
         arr(n) = vals(i)
      end do
   end subroutine populate_by_mask_1d

   subroutine populate_by_mask_2d(arr, numf, mask, vals)
      real(rkind), intent(inout) :: arr(:, :)
      integer(ikind), intent(in) :: numf, mask(1:numf)
      real(rkind), intent(in) :: vals(:, :)
      integer :: i, n, j
      integer :: j_ub, j_lb
      j_ub = ubound(arr, 2)
      j_lb = lbound(arr, 2)
      if (size(arr, 2) .ne. size(vals, 2)) error stop "'J' Dimension needs to be identical"
      if (.not. size(arr, 1) .ge. numf) error stop "(2d) numf > total array size"
      do concurrent(i=1:numf, j=j_lb:j_ub)
         n = mask(i)
         arr(n, j) = vals(i, j)
      end do
   end subroutine populate_by_mask_2d

   subroutine decompose_to_input_list(decomp_dims, inputs, storage)
      integer(ikind), intent(in) :: decomp_dims(:)
      type(field_list_t), intent(out) :: inputs
      type(field_storage_t), allocatable, intent(out) :: storage(:)

      integer(ikind) :: n_fields
      integer(ikind) :: i_field
      integer(ikind) :: width
      integer(ikind) :: first_feature
      integer(ikind) :: last_feature

      n_fields = size(decomp_dims)

      if (sum(decomp_dims) /= n_features) then
         error stop "Invalid partition of canonical input"
      end if

      inputs = field_list_t(n_fields, "inputs")
      allocate (storage(n_fields))

      first_feature = 1

      do i_field = 1, n_fields

         width = decomp_dims(i_field)
         last_feature = first_feature + width - 1

         if (width == 1) then

            allocate (storage(i_field)%a1d(beg:end))
            storage(i_field)%a1d(:) = -999.0_rkind

            call populate_by_mask( &
               storage(i_field)%a1d, &
               n_mask, &
               mask, &
               canonical_input(:, first_feature) &
               )

            call inputs%add(storage(i_field)%a1d, n_mask, mask)

         else

            allocate (storage(i_field)%a2d(beg:end, width))

            storage(i_field)%a2d(:, :) = -999.0_rkind

            call populate_by_mask( &
               arr=storage(i_field)%a2d, &
               numf=n_mask, &
               mask=mask, &
               vals=canonical_input(:, first_feature:last_feature))

            call inputs%add(storage(i_field)%a2d, n_mask, mask)

         end if
         first_feature = last_feature + 1
      end do

   end subroutine decompose_to_input_list
end program test_tiny_emulator
