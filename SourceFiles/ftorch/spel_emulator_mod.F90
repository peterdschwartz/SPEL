module spel_emulator_mod
  use ftorch
  implicit none

  type :: spel_emulator_t
     type(torch_model)    :: model
     type(torch_tensor)   :: input_tensor
     type(torch_tensor)   :: output_tensor
     logical              :: initialized = .false.
  contains
     procedure :: init
     procedure :: infer
  end type spel_emulator_t

contains

  subroutine init(this, model_path, n_samples)
    class(spel_emulator_t), intent(inout) :: this
    character(len=*),       intent(in)    :: model_path
    integer,                intent(in)    :: n_samples

    ! load TorchScript model
    call torch_model_load(this%model, model_path)

    ! allocate tensors with shapes known from static analysis/codegen
    call torch_tensor_from_array(this%input_tensor,  reshape(...),  device=TORCH_CPU)
    call torch_tensor_empty(this%output_tensor, [n_samples, out_dim], device=TORCH_CPU)

    this%initialized = .true.
  end subroutine init

  subroutine infer(this, fortran_inputs, fortran_outputs)
    class(spel_emulator_t), intent(inout) :: this
    real(real64),           intent(in)    :: fortran_inputs(:,:,:,:)  ! example
    real(real64),           intent(out)   :: fortran_outputs(:,:,:,:) ! example

    ! 1. Pack Fortran inputs into this%input_tensor (view or copy)
    ! 2. call torch_model_forward(this%model, [this%input_tensor], [this%output_tensor])
    ! 3. Unpack this%output_tensor into fortran_outputs
  end subroutine infer

end module spel_emulator_mod
