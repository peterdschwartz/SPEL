"""OpenACC directives removed by acc-strip."""
from spel.transforms.acc_strip import strip_ranges, stripped

SRC = """\
subroutine s(x, n)
  integer, intent(in) :: n
  real(r8), intent(inout) :: x(n)
  integer :: i
  !$acc routine seq
  !$acc declare create(x)
  !$acc enter data create(x)
  !$acc parallel loop gang vector default(present) &
  !$acc   private(i) &
  !!$acc   copyin(n) &
  !$acc   if(elm_device_ready)
  do i = 1, n
     x(i) = 0._r8
  end do
  ! spel acc-sync: device data this host code uses
  !$acc update self(x) if_present
  print *, x(1)
  !$acc update device(x)
  !$acc data copy(x)
  !$acc end data
end subroutine s
""".split("\n")


def text(lines):
    return "\n".join(lines)


def test_default_keeps_routine_and_declare():
    out = text(stripped(SRC))
    assert "routine seq" in out and "declare create" in out
    assert "parallel loop" not in out and "if(elm_device_ready)" not in out and "copyin(n)" not in out
    assert "enter data" not in out and "update" not in out and "data copy" not in out and "end data" not in out
    assert "spel acc-sync" not in out          # the tag goes with its directive
    assert "do i = 1, n" in out and "print *, x(1)" in out


def test_sync_only_strips_if_present_updates_and_tags():
    out = text(stripped(SRC, sync_only=True))
    assert "update self(x) if_present" not in out and "spel acc-sync" not in out
    assert "update device(x)" in out and "parallel loop" in out


def test_only_one_construct():
    out = text(stripped(SRC, only=["data"]))
    assert "data copy" not in out and "end data" not in out
    assert "enter data" in out and "parallel loop" in out


def test_range_limits():
    # directives outside lo..hi stay
    assert strip_ranges(SRC, 0, 5) == []
    first = strip_ranges(SRC, 0, 7)
    assert first == [(6, 1), (7, 4)]
