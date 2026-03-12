module emulator_mod
   use field_mod, only: field_list_t
   use kinds ,only : rkind, ikind
   implicit none


   type :: emulator_t
      type(torch_model)   :: model
      type(torch_tensor)  :: input_tensor
      type(torch_tensor)  :: output_tensor
      logical             :: initialized = .false.
      character(len=16)   :: device = "cpu"
      integer :: in_dim, out_dim
   contains
      procedure :: init => init_from_field_lists
      procedure :: infer
   end type emulator_t

contains
   subroutine init_from_field_lists(this, model_path, inputs, outputs)
      class(emulator_t), intent(inout) :: this
      character(len=*), intent(in)    :: model_path
      type(field_list_t), intent(inout) :: inputs
      type(field_list_t), intent(inout) :: outputs

      integer :: in_total, out_total

      if (.not. associated(inputs%fields(1)%mask)) error stop "Mask not set on inputs"
      ! NOTE: Assumes all n_mask are the same
      this%n_index = inputs%fields(1)%n_mask

      in_total = inputs%compute_layout()
      out_total = outputs%compute_layout()

      this%in_dim = in_total
      this%out_dim = out_total

      call torch_model_load(this%model, trim(model_path))

      ! Allocate tensors: (n_index, dim)
      call torch_tensor_empty(this%input_tensor, [this%n_index, this%in_dim])
      call torch_tensor_empty(this%output_tensor, [this%n_index, this%out_dim])

      this%initialized = .true.
   end subroutine init_from_field_lists

  subroutine infer(this, inputs, outputs)
    class(spel_emulator_t), intent(inout) :: this
    type(field_list_t),     intent(in)    :: inputs
    type(field_list_t),     intent(inout) :: outputs

    real(rkind), allocatable :: buf_in(:,:), buf_out(:,:)

    if (.not. this%initialized) then
       stop "infer_from_field_lists: emulator not initialized"
    end if

    ! Allocate temporary host buffers
    allocate(buf_in(this%n_index, this%in_dim))
    allocate(buf_out(this%n_index, this%out_dim))

    ! Pack Fortran fields into flat input buffer
    call map_fields_to_buffer(inputs,  this%n_index, buf_in)

    ! Copy buffer into FTorch input tensor
    call torch_tensor_from_array(this%input_tensor, buf_in)

    ! Forward pass:  input tensor -> output tensor
    call torch_model_forward(this%model, [this%input_tensor], [this%output_tensor])

    ! Copy FTorch output tensor back to buffer
    call torch_tensor_to_array(this%output_tensor, buf_out)

    ! Unpack flat buffer back into Fortran fields
    call unmap_buffer_to_fields(outputs, this%n_index, buf_out)

    deallocate(buf_in)
    deallocate(buf_out)
  end subroutine infer

end module emulator_mod
