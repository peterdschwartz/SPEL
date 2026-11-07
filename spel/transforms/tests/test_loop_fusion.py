"""Adjacent loop nests over the same iteration space fused into one (loop_fusion)."""
import pytest

from spel.transforms.core.testing import check_pipeline, check_transform
from spel.transforms.loop_fusion import Fuser, LoopFusion
from spel.transforms.section_loops import SectionLoops


def mod(body: str, decls: str = "") -> str:
    return f"""\
module m
contains
subroutine s(x, y, z, n, c0, c1)
  real(r8), intent(inout) :: x(:), y(:), z(:,:)
  integer, intent(in) :: n, c0, c1
  integer :: i, j
  real(r8) :: t
{decls}{body}end subroutine s
end module m
"""


TWO = mod("""\
  do i = c0, c1
     x(i) = 0._r8
  end do
  ! the second field
  do i = c0, c1
     y(i) = 1._r8
  end do
""")

TWO_FUSED = mod("""\
  do i = c0, c1
     x(i) = 0._r8
  ! the second field
     y(i) = 1._r8
  end do
""")


def test_two_adjacent_loops_are_fused(tmp_path):
    res = check_transform(LoopFusion, TWO, TWO_FUSED, tmp_path)
    assert "2 nests fused at depth 1" in res.plan.report()


def test_a_run_of_three_and_a_two_level_nest(tmp_path):
    before = mod("""\
  do j = 1, n
     do i = c0, c1
        z(i, j) = 0._r8
     end do
  end do
  do j = 1, n
     do i = c0, c1
        z(i, j) = z(i, j) + 1._r8
     end do
  end do
  do j = 1, n
     do i = c0, c1
        z(i, j) = 2._r8 * z(i, j)
     end do
  end do
""")
    after = mod("""\
  do j = 1, n
     do i = c0, c1
        z(i, j) = 0._r8
        z(i, j) = z(i, j) + 1._r8
        z(i, j) = 2._r8 * z(i, j)
     end do
  end do
""")
    res = check_transform(LoopFusion, before, after, tmp_path)
    assert "3 nests fused at depth 2" in res.plan.report()


def test_fused_to_the_depth_the_headers_agree(tmp_path):
    before = mod("""\
  do j = 1, n
     do i = c0, c1
        z(i, j) = 0._r8
     end do
  end do
  do j = 1, n
     x(j) = 1._r8
  end do
""")
    after = mod("""\
  do j = 1, n
     do i = c0, c1
        z(i, j) = 0._r8
     end do
     x(j) = 1._r8
  end do
""")
    check_transform(LoopFusion, before, after, tmp_path)


def test_a_per_iteration_temporary_in_every_body(tmp_path):
    before = mod("""\
  do i = c0, c1
     t = x(i)
     y(i) = t
  end do
  do i = c0, c1
     t = 2._r8 * y(i)
     x(i) = t
  end do
""")
    after = mod("""\
  do i = c0, c1
     t = x(i)
     y(i) = t
     t = 2._r8 * y(i)
     x(i) = t
  end do
""")
    check_transform(LoopFusion, before, after, tmp_path)


@pytest.mark.parametrize("body, reason", [
    # the second loop reads an element the first writes in a later iteration
    ("""\
  do i = c0, c1
     x(i) = 0._r8
  end do
  do i = c0, c1
     y(i) = x(i + 1)
  end do
""", "different elements"),
    # a scalar the first loop leaves for the second
    ("""\
  do i = c0, c1
     t = x(i)
  end do
  do i = c0, c1
     y(i) = t
  end do
""", "written in one nest and used in another"),
    # a temporary written only under a condition
    ("""\
  do i = c0, c1
     if (x(i) > 0._r8) t = x(i)
     y(i) = t
  end do
  do i = c0, c1
     t = y(i)
     x(i) = t
  end do
""", "written in one nest and used in another"),
    # unknown effects
    ("""\
  do i = c0, c1
     call work(x(i))
  end do
  do i = c0, c1
     y(i) = 0._r8
  end do
""", "calls work"),
    ("""\
  do i = c0, c1
     if (x(i) < 0._r8) exit
  end do
  do i = c0, c1
     y(i) = 0._r8
  end do
""", "ExitStatement"),
    # a reduction over the array the first loop writes (all of it, not one element per iteration)
    ("""\
  do i = c0, c1
     x(i) = 1._r8
  end do
  do i = c0, c1
     z(i, 1) = sum(x)
  end do
""", "is written in one nest and used in another"),
])
def test_refused_when_the_result_could_change(tmp_path, body, reason):
    src = mod(body)
    res = check_transform(LoopFusion, src, src, tmp_path, procedures=["work"])
    assert any(reason in r.reason for r in res.plan.refusals), res.plan.report()


def test_left_alone_with_a_directive_or_other_bounds(tmp_path):
    src = mod("""\
  !$acc parallel loop default(present)
  do i = c0, c1
     x(i) = 0._r8
  end do
  do i = c0, c1
     y(i) = 0._r8
  end do
  do i = 1, n
     x(i) = 1._r8
  end do
""")
    res = check_transform(LoopFusion, src, src, tmp_path)
    assert any("directive" in r.reason for r in res.plan.refusals)
    assert not res.plan.changes


def test_bounds_written_by_an_earlier_body_are_refused(tmp_path):
    src = mod("""\
  do i = 1, n
     x(i) = 0._r8
  end do
  do i = 1, n
     y(i) = 0._r8
  end do
""").replace("integer, intent(in) :: n, c0, c1", "integer, intent(inout) :: n\n  integer, intent(in) :: c0, c1")
    src2 = src.replace("     x(i) = 0._r8\n", "     x(i) = 0._r8\n     n = n\n", 1)
    res = check_transform(LoopFusion, src2, src2, tmp_path)
    assert res.plan.refusals and not res.plan.changes


SECTIONS = mod("""\
  x(c0:c1) = 0._r8
  y(c0:c1) = 1._r8
  z(c0:c1, 1) = 2._r8
""")


def test_section_loops_then_loop_fusion(tmp_path):
    after = mod("""\
  do spel_i1 = c0, c1  ! spel section-loop
     x(spel_i1) = 0._r8
     y(spel_i1) = 1._r8
     z(spel_i1, 1) = 2._r8
  end do
""", decls="  integer :: spel_i1  ! spel section-loop indices\n")
    check_pipeline([SectionLoops, LoopFusion], SECTIONS, after, tmp_path)


def test_inquiry_bounds_of_arrays_allocated_alike(tmp_path, monkeypatch):
    # `x(:) = 0` gives lbound(x,1)/ubound(x,1): fused when the allocations agree in that dimension
    before = """\
module m
contains
subroutine s()
  integer :: spel_i1
  do spel_i1 = lbound(gx,1), ubound(gx,1)
     gx(spel_i1) = 0._r8
  end do
  do spel_i1 = lbound(gy,1), ubound(gy,1)
     gy(spel_i1) = 0._r8
  end do
  do spel_i1 = lbound(gw,1), ubound(gw,1)
     gw(spel_i1) = 0._r8
  end do
end subroutine s
end module m
"""
    after = before.replace("""\
  end do
  do spel_i1 = lbound(gy,1), ubound(gy,1)
""", "")
    alloc = {"gx": ["begc:endc"], "gy": ["begc:endc"], "gw": ["begp:endp"]}
    monkeypatch.setattr(Fuser, "alloc_dims", lambda self, path: alloc.get(path))
    check_transform(LoopFusion, before, after, tmp_path)


def test_an_associated_section_has_unknown_bounds(tmp_path, monkeypatch):
    src = """\
module m
contains
subroutine s(c0, c1)
  integer, intent(in) :: c0, c1
  integer :: spel_i1
  associate(a => gx(c0:c1), b => gy)
  do spel_i1 = lbound(a,1), ubound(a,1)
     a(spel_i1) = 0._r8
  end do
  do spel_i1 = lbound(b,1), ubound(b,1)
     b(spel_i1) = 0._r8
  end do
  end associate
end subroutine s
end module m
"""
    monkeypatch.setattr(Fuser, "alloc_dims", lambda self, path: ["begc:endc"])
    res = check_transform(LoopFusion, src, src, tmp_path)
    assert not res.plan.changes
