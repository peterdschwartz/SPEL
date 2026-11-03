"""The capture call gets the bounds actual as source text, subscripts included."""
from pathlib import Path

from spel.scripts.driver_callsites import call_actual_text, routine_lines
from spel.scripts.types import LineTuple

SRC = """module drv
contains
  subroutine elm_drv(nclumps)
    integer, intent(in) :: nclumps
    integer :: nc
    do nc = 1, nclumps
       call setFilters(bounds_by_clump(nc), filter, &
            frac(bounds_by_clump(nc)%begp:bounds_by_clump(nc)%endp))
       call Other(num, bounds = bounds_by_clump( nc ), x=y)
       write(iulog,*) 'skipped io', (k, k=1,3)
       call Plain(bounds_clump, z)
    end do
  end subroutine elm_drv
end module drv
"""


def test_call_actual_text_keeps_subscripts(tmp_path: Path):
    p = tmp_path / "drv.F90"
    p.write_text(SRC)
    src = SRC.split("\n")
    lines = [LineTuple(line=l, ln=i) for i, l in enumerate(src)]
    text_of = call_actual_text(p, routine_lines(lines, "elm_drv"))
    ln = {kind: next(i for i, l in enumerate(src) if f"call {kind}(" in l) for kind in ("setFilters", "Other", "Plain")}
    assert text_of(ln["setFilters"], "bounds_by_clump") == "bounds_by_clump(nc)"
    assert text_of(ln["setFilters"], "filter") == "filter"
    assert text_of(ln["Other"], "bounds_by_clump") == "bounds_by_clump( nc )"  # keyword actual
    assert text_of(ln["Plain"], "bounds_clump") == "bounds_clump"
    assert text_of(ln["Plain"], "bounds_by_clump") is None
