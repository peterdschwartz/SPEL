from spel.scripts.fortran_parser.tests.test_scope_walk import walk
from spel.transforms.blocked_loops import find_blocked_loops, unblock

SRC = """\
subroutine setvalues(this, num_column, filter_column, value_column)
  implicit none
  class(foo_type) :: this
  integer, intent(in) :: num_column
  integer, intent(in) :: filter_column(:)
  real, intent(in) :: value_column
  integer :: fi, j, nlev, blk
  do j = 1, nlev
     do fi = 1,num_column,blk
        associate (idx => filter_column(fi:min(fi+blk-1,num_column)))
        this%hr_vr(idx,j) = value_column
        end associate
     end do
  end do
  do fi = 1,num_column,blk
     associate (idx => &
          filter_column(fi:min(fi+blk-1,num_column)))
     this%er(idx) = value_column   ! idx in a comment stays
     this%core%idx(idx) = value_column
     end associate
  end do
end subroutine setvalues
"""


def test_find_and_unblock():
    lines = SRC.split("\n")
    loops = find_blocked_loops(walk(SRC), lines)
    assert [(b.do_ln, b.name, b.filter, b.index, b.start, b.end) for b in loops] == [
        (8, "idx", "filter_column", "fi", "1", "num_column"),
        (14, "idx", "filter_column", "fi", "1", "num_column"),
    ]
    assert loops[1].assoc_lines == [15, 16, 19]
    assert loops[1].uses == [17, 18]
    assert unblock(lines, loops) == 2
    out = [l for l in lines if l is not None]
    assert out[8].strip() == "do fi = 1, num_column  ! spel unblock"
    assert out[9].strip() == "this%hr_vr(filter_column(fi),j) = value_column"
    assert out[12].strip().startswith("do fi = 1, num_column")
    assert out[13].strip() == "this%er(filter_column(fi)) = value_column   ! idx in a comment stays"
    assert out[14].strip() == "this%core%idx(filter_column(fi)) = value_column"
    assert out[15].strip() == "end do"
