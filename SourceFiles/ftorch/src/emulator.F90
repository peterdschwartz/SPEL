module emulator_mod
   use iso_c_binding, only: c_int
   use field_mod, only: field_list_t
   use kinds, only: rkind, ikind
   use ftorch, only: torch_model, torch_tensor, torch_model_load, &
                     torch_kCPU, torch_model_print_parameters, torch_model_forward, &
                     torch_tensor_from_array, torch_tensor_empty, torch_kFloat64
   implicit none
   logical, parameter :: verbose = .true.
   integer(c_int), parameter :: device = torch_kCPU

   type :: emulator_t
      logical             :: initialized = .false.
      character(len=64) :: name
      integer(ikind) :: in_dim
      integer(ikind) :: out_dim
      integer(ikind) :: n_samples
      type(torch_model)   :: model
      type(torch_tensor)  :: input_tensor
      type(torch_tensor)  :: output_tensor

   contains
      procedure :: init => init_from_field_lists
      procedure :: infer
      procedure :: write_formatted
      generic :: write (formatted) => write_formatted

   end type emulator_t

contains

   subroutine init_from_field_lists(this, model_path, inputs, outputs)
      class(emulator_t), intent(inout) :: this
      character(len=*), intent(in)    :: model_path
      type(field_list_t), intent(inout) :: inputs
      type(field_list_t), intent(inout) :: outputs

      integer :: in_total, out_total
      integer(c_int), parameter :: ndim = int(2, c_int)

      if (.not. associated(inputs%fields(1)%mask)) error stop "Mask not set on inputs"

      this%n_samples = inputs%fields(1)%n_mask

      in_total = inputs%compute_layout()
      out_total = outputs%compute_layout()

      this%in_dim = in_total
      this%out_dim = out_total

      call torch_model_load(this%model, trim(model_path), device)

      call torch_tensor_empty(this%input_tensor, ndim, [this%n_samples, this%in_dim], torch_kFloat64, device)
      call torch_tensor_empty(this%output_tensor, ndim, [this%n_samples, this%out_dim], torch_kFloat64, device)

      this%initialized = .true.
   end subroutine init_from_field_lists

   subroutine infer(this, inputs, outputs)
      class(emulator_t), intent(inout) :: this
      type(field_list_t), intent(in)    :: inputs
      type(field_list_t), intent(inout) :: outputs

      real(rkind), pointer, contiguous :: buf_in(:, :), buf_out(:, :)

      if (.not. this%initialized) then
         error stop "emulator::infer - emulator not initialized"
      end if

      if (inputs%n_fields .ne. size(inputs%fields)) then
         error stop "emulator::infer - Didn't add all expected fields to input list"
      end if 
      if (outputs%n_fields .ne. size(outputs%fields)) then
         error stop "emulator::infer - Didn't add all expected fields to output list"
      end if 

      ! Allocate temporary host buffers
      allocate (buf_in(this%n_samples, this%in_dim))
      allocate (buf_out(this%n_samples, this%out_dim))

      ! Pack Fortran fields into flat input buffer
      call inputs%map_to_buffer(buf_in)

      ! Copy buffer into FTorch input tensor
      ! (tensor, data_in, device_type, OPTIONAL: device_index, requires_grad)
      call torch_tensor_from_array(this%input_tensor, buf_in, device)
      call torch_tensor_from_array(this%output_tensor, buf_out, device)

      ! Forward pass:  input tensor -> output tensor
      call torch_model_forward(this%model, [this%input_tensor], [this%output_tensor])
      call outputs%map_from_buffer(this%n_samples, buf_out)

      deallocate (buf_in)
      deallocate (buf_out)
   end subroutine infer

   subroutine write_formatted(self, unit, iotype, v_list, iostat, iomsg)
      class(emulator_t), intent(in) :: self
      integer, intent(in) :: unit
      character(len=*), intent(in) :: iotype
      integer, intent(in) :: v_list(:)
      integer, intent(out) :: iostat
      character(len=*), intent(inout) :: iomsg

      iostat = 0
      iomsg = ""

      write (unit, '(a)', iostat=iostat, iomsg=iomsg) &
         "SPEL EMULATOR: "//trim(self%name)//NEW_LINE('a')
      if (iostat /= 0) return

      write (unit, '(a,i0,a)', iostat=iostat, iomsg=iomsg) &
         "Inferring on batches of ", self%n_samples, " samples"//NEW_LINE('a')
      if (iostat /= 0) return

      write (unit, '(a,i0,a)', iostat=iostat, iomsg=iomsg) &
         "Input dim: ", self%in_dim, NEW_LINE('a')
      if (iostat /= 0) return

      write (unit, '(a,i0,a)', iostat=iostat, iomsg=iomsg) &
         "Output dim: ", self%out_dim, NEW_LINE('a')
      if (iostat /= 0) return

      if (verbose) then
         write (unit, *)
         write (unit, '(a,/)') "Model parameters"
         flush (unit)

         call self%model%print_parameters()

         write (unit, *)
         write (unit, '(a,/)') "Current input tensor:"
         flush (unit)

         call self%input_tensor%print()

         write (unit, *)
         write (unit, '(a,/)') "Current output tensor:"
         flush (unit)

         call self%output_tensor%print()
      end if
   end subroutine write_formatted

end module emulator_mod
