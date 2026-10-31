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



def _build(tmp_path, monkeypatch, driver_text, module_text=None):
    from spel.transforms.core.transform import Context

    src = tmp_path / "components" / "elm" / "src"
    (src / "main").mkdir(parents=True)
    (src / "data_types").mkdir()
    drv = src / "main" / "elm_driver.F90"
    drv.write_text(driver_text)
    if module_text is not None:
        (src / "data_types" / "ElmDeviceDataMod.F90").write_text(module_text)
    texts = {"ElmDeviceDataMod.F90": "module x\nend module\n"}
    monkeypatch.setattr(generate, "generate", lambda fut, root: (texts, generate.Plan([], {})))
    monkeypatch.setattr(generate, "_driver_edits", lambda fut, s: (
        [generate.Edit(drv, None, ["first"], at_line=1), generate.Edit(drv, None, ["second"], at_line=1)], []))
    ctx = Context(None, tmp_path)
    return generate.DeviceData().build(ctx, generate.Options()), ctx, drv


def test_device_data_transform_keeps_the_legacy_order_at_one_line(tmp_path, monkeypatch):
    plan, ctx, drv = _build(tmp_path, monkeypatch, "a\nb\nc")
    assert not plan.refusals
    out = plan.outputs(ctx.sources)
    assert out[drv] == "a\nsecond\nfirst\nb\nc"  # what the bottom-up splicing gave
    assert out[drv.parent.parent / "data_types" / "ElmDeviceDataMod.F90"] == "module x\nend module\n"


def test_device_data_transform_refuses_an_applied_driver(tmp_path, monkeypatch):
    plan, _, _ = _build(tmp_path, monkeypatch, f"a\n  call x()  {generate.TAG}\nc")
    assert not plan.changes and "already carries" in plan.refusals[0].reason


def test_device_data_transform_refuses_to_overwrite_a_hand_written_module(tmp_path, monkeypatch):
    plan, _, _ = _build(tmp_path, monkeypatch, "a\nb", module_text="module ElmDeviceDataMod\nend module\n")
    assert not plan.changes and not plan.new_files and "not generated here" in plan.refusals[0].reason
    plan, _, _ = _build(tmp_path / "again", monkeypatch, "a\nb", module_text=f"! {generate.GENERATED}\n")
    assert not plan.refusals and plan.new_files
