"""genf90 templates generated for the analysis (spel.scripts.genf90)."""
import pytest

from spel.scripts.genf90 import LINE_MARKER, generate, genf90_script, strip_markers

TPL = """module tiny_mod
  implicit none
  interface get
     ! TYPE int,double
     module procedure get_{TYPE}
  end interface
contains
  ! TYPE int,double
  subroutine get_{TYPE}(x)
    {VTYPE}, intent(out) :: x
    x = 0
  end subroutine get_{TYPE}
end module tiny_mod
"""


def test_strip_markers():
    text = 'module m\n# 12 "x.F90.in"\n  use a\n#include "shr_assert.h"\nend module m\n'
    assert strip_markers(text) == 'module m\n  use a\n#include "shr_assert.h"\nend module m\n'
    assert LINE_MARKER.match('# 3 "f"') and not LINE_MARKER.match("#ifdef X")


def test_generate(tmp_path):
    if genf90_script() is None:
        pytest.skip("no genf90.pl in the checkout")
    tpl = tmp_path / "tiny_mod.F90.in"
    tpl.write_text(TPL)
    out = generate(tmp_path / "gen", files=[tpl])
    assert [p.name for p in out] == ["tiny_mod.F90"]
    text = out[0].read_text()
    assert "subroutine get_int(x)" in text and "subroutine get_double(x)" in text
    assert not any(LINE_MARKER.match(l) for l in text.split("\n"))
    # up to date: not regenerated
    before = out[0].stat().st_mtime_ns
    assert generate(tmp_path / "gen", files=[tpl]) == out and out[0].stat().st_mtime_ns == before
