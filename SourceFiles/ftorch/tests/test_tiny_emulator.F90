program test_tiny_emulator

   use kinds, only: rkind, ikind
   use emulator_mod, only: emulator_t
   use field_mod, only: field_list_t

   implicit none

   interface populate_by_mask
      procedure :: populate_by_mask_1d
   end interface

   type(emulator_t) :: emulator
   type(field_list_t) :: inputs
   type(field_list_t) :: outputs

   integer(ikind), parameter :: n_features = 4
   integer(ikind), parameter :: beg = 1, end = 5
   real(rkind), target :: x1_storage(beg:end)
   real(rkind), target :: x2_storage(beg:end)
   real(rkind), target :: x3_storage(beg:end)
   real(rkind), target :: x4_storage(beg:end)
   real(rkind), target :: y_storage(beg:end)

   real(rkind), pointer :: x1(:)
   real(rkind), pointer :: x2(:)
   real(rkind), pointer :: x3(:)
   real(rkind), pointer :: x4(:)
   real(rkind), pointer :: y(:)

   integer(ikind), parameter :: n_mask = 3
   integer(ikind), target :: mask_storage(n_mask)
   integer(ikind), pointer :: mask(:)

   character(len=*), parameter :: test_path = "../../unit-tests/input-data/tiny_case/"
   character(len=*), parameter :: model_path = trim(test_path)//"spel_emulator_torchscript.pt"

   ! ------------------------------------------------------------
   ! Set up pointers expected by field_list_t
   ! ------------------------------------------------------------

   x1 => x1_storage
   x2 => x2_storage
   x3 => x3_storage
   x4 => x4_storage
   y => y_storage

   mask => mask_storage

   ! Select physical indices 2, 4, and 5.
   mask = [2_ikind, 4_ikind, 5_ikind]

   ! ------------------------------------------------------------
   ! Fill everything with sentinels first.
   ! ------------------------------------------------------------

   x1 = -999.0_rkind
   x2 = -999.0_rkind
   x3 = -999.0_rkind
   x4 = -999.0_rkind

   ! output
   y = -999.0_rkind

   ! ------------------------------------------------------------
   ! These three masked positions reproduce the Python input:
   !   x1  x2 x3  x4
   ! [ 1,  2, 4, 4 ]
   ! [ 2,  4, 7, 8 ]
   ! [-1,  0, 1, 2 ]
   ! ------------------------------------------------------------

   call populate_by_mask(x1, n_mask, mask, [1.0_rkind, 2.0_rkind, -1.0_rkind])
   call populate_by_mask(x2, n_mask, mask, [2.0_rkind, 4.0_rkind, 0.0_rkind])
   call populate_by_mask(x3, n_mask, mask, [4.0_rkind, 7.0_rkind, 1.0_rkind])
   call populate_by_mask(x4, n_mask, mask, [4.0_rkind, 8.0_rkind, 2.0_rkind])

   ! ------------------------------------------------------------
   ! Construct input/output field lists.
   ! ------------------------------------------------------------

   inputs = field_list_t(n_features, "inputs")

   call inputs%add(x1, n_mask, mask)
   call inputs%add(x2, n_mask, mask)
   call inputs%add(x3, n_mask, mask)
   call inputs%add(x4, n_mask, mask)

   ! call inputs%print()

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

   ! Make sure unmasked values stayed untouched.
   if (y(1) /= -999.0_rkind) then
      error stop "y(1) was unexpectedly modified"
   end if

   if (y(3) /= -999.0_rkind) then
      error stop "y(3) was unexpectedly modified"
   end if

contains

   subroutine populate_by_mask_1d(arr, numf, mask, vals)
      real(rkind), intent(inout) :: arr(:)
      integer(ikind), intent(in) :: numf, mask(1:numf)
      real(rkind), intent(in) :: vals(:)
      integer :: i, n
      do concurrent(i=1:numf)
         n = mask(i)
         arr(n) = vals(i)
      end do
   end subroutine populate_by_mask_1d

end program test_tiny_emulator
