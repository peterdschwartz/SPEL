"""Array-section assignments rewritten as explicit loops (section_loops)."""
from spel.scripts.fortran_parser.tests.test_scope_walk import gvar, parse_statements, proc, walk
from spel.transforms.section_loops import apply_to_lines, find_section_statements

SRC = """\
subroutine s(this, bounds, x, y, n, num, filt, fc)
  implicit none
  class(foo_type) :: this
  type(bounds_type), intent(in) :: bounds
  real(r8), intent(inout) :: x(:,:), y(:,:)
  integer, intent(in) :: n, num, filt(:)
  integer :: fc, c, j
  real(r8) :: d(num, n), v(1:n)
  integer, allocatable :: w(:)
  x(bounds%begc:bounds%endc, 1:n) = 1._r8
  this%core%a(:,:,:) = 0._r8
  y(:,:) = x(:,:) * 2.0_r8 + &
       max(x(:,:), 1.e-20_r8)
  do fc = 1, num
     c = filt(fc)
     d(fc,:) = 0._r8
     v(1:n) = x(c, 2:n+1)
  end do
  !$acc parallel loop
  do fc = 1, num
     d(fc,:) = 1._r8
  end do
  j = sum(x(:,1))
  y(:,1) = sum(x(:,:))
  allocate(w(n)); w(:) = -1
end subroutine s
"""


def test_find_and_rewrite():
    lines = SRC.split("\n")
    rec = walk(SRC)

    class Sub:
        id = "m::s"
        record = rec
    stmts, notes = find_section_statements(Sub, lines, device=[(19, 22)])
    print("\n".join(notes))
    assert [s.ln + 1 for s in stmts] == [10, 11, 12, 16, 17, 25]
    assert any("line 21: inside a compute construct" in n for n in notes)
    assert any("line 24" in n and "reduction" in n for n in notes)
    assert apply_to_lines(Sub, lines, stmts, parse_statements(SRC).statements[0]) == 6
    out = "\n".join(l for l in lines if l is not None).split("\n")
    text = "\n".join(out)
    # indices declared before the first executable statement
    assert out[9].strip() == "integer :: spel_i1, spel_i2, spel_i3  ! spel section-loop indices"
    assert "do spel_i2 = 1, n  ! spel section-loop" in text
    assert "do spel_i1 = bounds%begc, bounds%endc" in text
    assert "x(spel_i1, spel_i2) = 1._r8" in text
    assert "do spel_i3 = lbound(this%core%a,3), ubound(this%core%a,3)" in text
    assert "this%core%a(spel_i1, spel_i2, spel_i3) = 0._r8" in text
    # the right-hand side keeps its text (kind suffixes) with the sections indexed
    assert "y(spel_i1, spel_i2) = x(spel_i1, spel_i2) * 2.0_r8 + max(x(spel_i1, spel_i2), 1.e-20_r8)" in text
    assert "d(fc, spel_i1) = 0._r8" in text
    # shifted section on the right-hand side: lo_rhs + (index - lo_lhs)
    assert "v(spel_i1) = x(c, 2 + (spel_i1 - (1)))" in text
    assert "j = sum(x(:,1))" in text and "y(:,1) = sum(x(:,:))" in text
    i = out.index("  allocate(w(n))")
    assert out[i + 1].strip().startswith("do spel_i1 = lbound(w,1), ubound(w,1)") and out[i + 2].strip() == "w(spel_i1) = -1"


CALLS = """\
subroutine t(bounds, rootfr, n, nlev2bed, x)
  implicit none
  type(bounds_type), intent(in) :: bounds
  real(r8), intent(inout) :: rootfr(:,:), x(:)
  integer, intent(in) :: n, nlev2bed(:)
  real(r8) :: hdef(n), x0
  rootfr(bounds%begp:bounds%endp, 1:n) = zeng2001_rootfr(bounds, n, nlev2bed)
  rootfr(bounds%begp:bounds%endp, n+1:nlevgrnd) = 0._r8
  x(1:n) = max(x(1:n), 0._r8)
  x(1:n) = hdef
  x(1:n) = x0
  x(1:2) = (/ 1._r8, 2._r8 /)
end subroutine t
"""


def test_array_valued_calls_are_left_alone_and_expression_bounds_are_sections():
    lines = CALLS.split("\n")
    rec = walk(CALLS, globals=[proc("zeng2001_rootfr"), gvar("nlevgrnd")])

    class Sub:
        id = "m::t"
        record = rec
    stmts, notes = find_section_statements(Sub, lines)
    assert [s.ln + 1 for s in stmts] == [8, 9, 11]
    assert any("line 7" in n and "zeng2001_rootfr" in n for n in notes)
    assert any("line 10" in n and "whole array" in n for n in notes)
    assert any("line 12" in n and "array constructor" in n for n in notes)
    apply_to_lines(Sub, lines, stmts, parse_statements(CALLS).statements[0])
    text = "\n".join(l for l in lines if l is not None)
    assert "do spel_i2 = n+1, nlevgrnd" in text.replace("(n+1)", "n+1")
    assert "rootfr(spel_i1, spel_i2) = 0._r8" in text
    assert "x(spel_i1) = max(x(spel_i1), 0._r8)" in text
