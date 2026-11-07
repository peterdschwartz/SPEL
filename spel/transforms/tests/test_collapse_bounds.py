"""Inquiry bounds of collapsed nests hoisted into host scalars (collapse_bounds)."""
from spel.scripts.fortran_parser.tests.test_scope_walk import walk
from spel.transforms.collapse_bounds import has_inquiry, rewrite_routine

SRC = """\
subroutine s(top, x, n)
  implicit none
  type(top_type), intent(inout) :: top
  real(r8), intent(inout) :: x(:,:)
  integer, intent(in) :: n
  integer :: i, j, k
  associate(fa => top%solad_pp)
  !$acc parallel loop gang vector collapse(2) default(present) if(elm_device_ready)
  do j = lbound(fa,2), ubound(fa,2)  ! spel section-loop
     do i = lbound(fa,1), ubound(fa,1)
        fa(i, j) = top%solad(i, j)
     end do
  end do
  !$acc parallel loop gang vector collapse(2) default(present)
  do j = 1, n
     do i = 1, size(x, 1)
        x(i, j) = 0._r8
     end do
  end do
  !$acc parallel loop gang vector default(present)
  do i = lbound(x,1), ubound(x,1)
     x(i, 1) = 1._r8
  end do
  !$acc parallel loop gang vector collapse(2) default(present)
  do j = 1, n
     do i = 1, n
        x(i, j) = 2._r8
     end do
  end do
  end associate
end subroutine s
"""


def test_hoist_inquiry_bounds_of_collapsed_nests():
    lines = SRC.split("\n")
    rec = walk(SRC)

    class Sub:
        id = "m::s"
        record = rec
        startline = 0
        endline = len(lines) - 1
    notes: list[str] = []
    assert rewrite_routine(Sub, lines, notes) == 2, notes
    out = "\n".join(l for l in lines if l is not None).split("\n")
    text = "\n".join(out)
    # declared after the last declaration
    assert out[5].strip() == "integer :: i, j, k"
    assert out[6].strip().startswith("integer :: spel_lb1, spel_lb2, spel_ub1, spel_ub2  ! spel collapse-bounds")
    # the first nest: both headers hoisted (k counts the nest outer-first), the directive right
    # after the assignments, the section-loop tag kept; expressions print as the AST renders them
    i = out.index("  spel_lb1 = lbound(fa,2)  ! spel collapse-bounds")
    assert out[i + 1].strip() == "spel_ub1 = ubound(fa,2)  ! spel collapse-bounds"
    assert out[i + 2].strip() == "spel_lb2 = lbound(fa,1)  ! spel collapse-bounds"
    assert out[i + 3].strip() == "spel_ub2 = ubound(fa,1)  ! spel collapse-bounds"
    assert out[i + 4].strip().startswith("!$acc parallel loop gang vector collapse(2)")
    assert out[i + 5] == "  do j = spel_lb1, spel_ub1  ! spel section-loop"
    assert out[i + 6] == "     do i = spel_lb2, spel_ub2"
    # the second nest: only the inner upper bound is an inquiry
    assert "spel_ub2 = size(x,1)  ! spel collapse-bounds" in text
    assert "     do i = 1, spel_ub2" in text and "  do j = 1, n\n" in text
    # a 1-D loop and a nest without inquiries are untouched
    assert "do i = lbound(x,1), ubound(x,1)" in text
    assert text.count("! spel collapse-bounds") == 6
    assert not notes


def test_has_inquiry():
    rec = walk("subroutine t(x, n)\n real(r8) :: x(:)\n integer :: i, n\n do i = lbound(x,1), 3*size(x)+1\n x(i)=0\n end do\n do i = 1, n\n x(i)=0\n end do\nend subroutine t\n")
    a, b = rec.loop_nest
    assert has_inquiry(a.node.start) and has_inquiry(a.node.end)
    assert not has_inquiry(b.node.start) and not has_inquiry(b.node.end) and not has_inquiry(None)
