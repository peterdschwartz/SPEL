"""elm_drv's host calls are found on the driver file's own statements; device-data places no transitions."""
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


def test_host_calls_are_reported_on_the_driver_lines(tmp_path, monkeypatch):
    drv = tmp_path / "elm_driver.F90"
    drv.write_text(DRIVER)
    monkeypatch.setattr(generate, "device_routines", lambda fut: {"devwork", "dynsubgrid_driver"})
    # the analysis' kinds in source order; its call lines are deliberately wrong (they must not be
    # used), and ndep_interp (a module the analysis excludes) is not in it: kind by name
    monkeypatch.setattr(generate, "call_kinds", lambda fut: [
        (1, "hostwork", "host"), (1, "devwork", "device"), (1, "hostwork", "host"), (1, "dynsubgrid_driver", "host")])
    fut = SimpleNamespace(subroutine_dict={"elm_driver::elm_drv": SimpleNamespace()})
    report = generate.host_calls(fut, drv)
    lines = DRIVER.split("\n")
    got = [(int(r.split(":")[0]), r.split("host routine ")[1].split()[0]) for r in report]
    assert [b for _, b in got] == ["hostwork", "hostwork", "ndep_interp", "dynsubgrid_driver"]
    for ln, base in got:
        assert base in lines[ln - 1].lower(), (ln, base, lines[ln - 1])


def test_driver_edits_insert_no_transitions(tmp_path, monkeypatch):
    src = tmp_path / "src"
    (src / "main").mkdir(parents=True)
    monkeypatch.setattr(generate, "host_calls", lambda fut, drv: [])
    edits, _ = generate._driver_edits(SimpleNamespace(), src)
    text = "\n".join(l for e in edits for l in e.lines)
    assert "elm_device_update" not in text
    # the state stays on the device after both copies
    assert text.count("call elm_device_copyin()") == 2
    assert text.count("elm_device_fullsync = .false.") == 2
    # the attach of the device pointers after the copy in initialize2, with their imports
    assert "call subgrid_ave_device_attach()" in text and "call hist_device_attach()" in text
    assert "only : hist_device_attach" in text and "only : subgrid_ave_device_attach" in text
    # reweight_wrapup sends only its clump's filters, not the whole subgrid group
    assert "subgrid_update" not in text
    assert text.count("call update_clumpfilter(") == 2 and "bounds%clump_index" in text



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
