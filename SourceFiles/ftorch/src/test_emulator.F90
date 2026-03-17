program test_emulator
  use field_mod, only : field_list_t
  use kinds, only : rkind, ikind

  implicit none


  type(spel_emulator_t) :: emulator
  real(real64), allocatable :: x(:), y(:)
  integer :: n_in, n_out

  ! For the demo, these must match the TorchScript model's input/output dims.
  n_in  = 15120  
  n_out = 960

  allocate(x(n_in))
  allocate(y(n_out))

  !
  ! call emulator%init("spel_emulator_traced.pt", 1, 1)  ! 1 in, 1 out tensor
  !
  ! call emulator%infer(x, y)

  print *, "First few outputs:"
  print *, y(1:min(10, n_out))

end program test_emulator
