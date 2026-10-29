"""elm_drv's host/device transitions are placed by the driver file's own statements."""
from types import SimpleNamespace

from spel.transforms.device import generate

DRIVER = """module elm_driver
  implicit none
contains
  subroutine elm_drv(doalb)
    logical, intent(in) :: doalb
    ! lines added after the analysis shift every call below
    call t_startf('x')
    call hostwork(1)
    if (doalb) then
       call devwork(2)
    end if
    if (doalb) call hostwork(3)
    call ndep_interp(5)
    call dynsubgrid_driver(4)
  end subroutine elm_drv
end module elm_driver
"""


def test_transitions_land_before_their_calls(tmp_path, monkeypatch):
    drv = tmp_path / "elm_driver.F90"
    drv.write_text(DRIVER)
    monkeypatch.setattr(generate, "device_routines", lambda fut: {"devwork", "dynsubgrid_driver"})
    # the analysis' kinds in source order; its call lines are deliberately wrong (they must not be
    # used), and ndep_interp (a module the analysis excludes) is not in it: kind by name
    monkeypatch.setattr(generate, "call_kinds", lambda fut: [
        (1, "hostwork", "host"), (1, "devwork", "device"), (1, "hostwork", "host"), (1, "dynsubgrid_driver", "host")])
    fut = SimpleNamespace(subroutine_dict={"elm_driver::elm_drv": SimpleNamespace()})
    edits, _ = generate._transition_edits(fut, drv)
    lines = DRIVER.split("\n")
    for e in sorted(edits, key=lambda e: -e.at_line):
        lines[e.at_line:e.at_line] = e.lines
    placed = [(ln, lines[i + 2].strip()) for i, ln in enumerate(lines) if "spel device-data" in ln]
    assert len(placed) == 5
    for ln, nxt in placed:
        name = ln.split("before ")[-1]
        assert name in nxt.lower(), (ln, nxt)
    assert "to_device=.true." in placed[1][0]          # devwork
    assert "to_device=.false." in placed[3][0]         # ndep_interp, by name
    assert "to_device=.false." in placed[4][0]         # dynSubgrid runs as a host segment
