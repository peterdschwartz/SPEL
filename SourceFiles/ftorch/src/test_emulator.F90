program test_emulator
  use iso_fortran_env, only: real64
  use field_mod
  implicit none

  type(spel_emulator_t) :: emulator
  real(real64), allocatable :: x(:), y(:)
  integer :: n_in, n_out

  ! For the demo, these must match the TorchScript model's input/output dims.
  n_in  = 15120   ! example; replace with actual in_dim
  n_out = 960     ! example; replace with actual out_dim

  allocate(x(n_in))
  allocate(y(n_out))

  x = 0.0_real64
  x(1) = 1.0_real64  ! some dummy pattern

  call emulator%init("spel_emulator_traced.pt", 1, 1)  ! 1 in, 1 out tensor

  call emulator%infer(x, y)

  print *, "First few outputs:"
  print *, y(1:min(10, n_out))

end program test_emulator
