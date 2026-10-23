import json
import textwrap
from pathlib import Path

import pytest

from types import SimpleNamespace

from spel.scripts import instrument_elm as ie
from spel.scripts.driver_callsites import DriverCall, DriverSites
from spel.scripts.module_resolver import scan_module_head
from spel.scripts.types import LineTuple

DRIVER = textwrap.dedent("""\
    module elm_driver
      ! !USES:
      use shr_kind_mod , only : r8 => shr_kind_r8
      use elm_instMod
      implicit none
    contains
      subroutine elm_drv(doalb)
        logical, intent(in) :: doalb
        do nc = 1,nclumps
           call get_clump_bounds(nc, bounds_clump)
           call t_startf('canflux')
           call CanopyFluxes(bounds_clump,                         &
                filter(nc)%num_nolakeurbanp, filter(nc)%nolakeurbanp, & ! comment
                canopystate_vars, photosyns_vars )
           call t_stopf('canflux')
        end do
      end subroutine elm_drv
    end module elm_driver
    """)

PRIV_MOD = textwrap.dedent("""\
    module SoilMoistStressMod
      implicit none
      save
      private
      public :: calc_root_moist_stress
      integer ::   root_moist_stress_method
      integer, parameter :: moist_stress_clm_default = 0
      logical,  private :: perchroot     = .false.  ! comment
      logical,  private :: perchroot_alt = .false.
      real :: other
    contains
      subroutine calc_root_moist_stress()
        integer :: perchroot_local
      end subroutine
    end module SoilMoistStressMod
    """)

PROT_MOD = textwrap.dedent("""\
    module PhotosynthesisMod
      private
      type, public :: photo_params_type
         real, pointer :: krmax(:)
      contains
         procedure, public :: readParams
      end type photo_params_type
      type(photo_params_type), public, protected :: params_inst  ! populated in readParamsMod
      type(photo_params_type), public :: other_inst, keep_inst
      protected :: other_inst, keep_inst
    end module PhotosynthesisMod
    """)

CF = "canopyfluxesmod::canopyfluxes"
CALL_LN = 11  # 0-based line of `call CanopyFluxes` in DRIVER


def lines_of(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def head_of(text: str):
    """ModuleHead as `spel create` would see it (fixtures have no continuations)."""
    return scan_module_head(LineTuple(line=ln.split("!")[0], ln=i) for i, ln in enumerate(text.splitlines()))


NAMES = ie.CaptureNames.for_case("canflux")


def one(sites_: DriverSites, names=NAMES):
    return [(sites_, names)]


def sites(calls=None, capture_lns=()) -> DriverSites:
    return DriverSites(
        module="elm_driver",
        routine="elm_drv",
        path="components/elm/src/main/elm_driver.F90",
        calls=calls if calls is not None else {CF: [DriverCall(CF, CALL_LN, "bounds_clump")]},
        capture_lns=list(capture_lns),
    )


def test_instrument_lines_inserts_and_is_idempotent():
    out, bounds = ie.instrument_lines(lines_of(DRIVER), one(sites()))
    assert bounds == {"canflux": "bounds_clump"}
    text = "".join(out)
    assert out[1].strip().startswith("use SpelCapture_canflux, only : spel_capture_inputs_canflux")
    # inputs right before the call, outputs right after its last continuation line
    i = next(i for i, ln in enumerate(out) if "call CanopyFluxes" in ln)
    assert out[i - 1].strip() == f"call spel_capture_inputs_canflux(bounds_clump) {ie.MARKER}"
    assert out[i + 3].strip() == f"call spel_capture_outputs_canflux(bounds_clump) {ie.MARKER}"
    assert out[i - 1].startswith("       call")  # same indentation as the call
    assert text.count(ie.MARKER) == 3

    # recorded lines refer to the marker-free file, so re-running is a no-op
    again, _ = ie.instrument_lines(out, one(sites()))
    assert again == out
    assert ie.strip_instrumentation(out) == lines_of(DRIVER)


def test_instrument_lines_spans_multiple_routines():
    driver = DRIVER.replace(
        "       call t_stopf('canflux')\n",
        "       call t_stopf('canflux')\n       call UrbanFluxes(bounds_clump)\n",
    )
    uf = "urbanfluxesmod::urbanfluxes"
    calls = {
        uf: [DriverCall(uf, CALL_LN + 4, "bounds_clump")],
        CF: [DriverCall(CF, CALL_LN, "bounds_clump")],
    }
    out, _ = ie.instrument_lines(lines_of(driver), one(sites(calls)))
    calls = [ln.strip() for ln in out if "spel_capture" in ln and "call" in ln]
    assert calls[0].startswith("call spel_capture_inputs")
    assert calls[1].startswith("call spel_capture_outputs")
    i = next(i for i, ln in enumerate(out) if "call UrbanFluxes" in ln)
    assert "spel_capture_outputs" in out[i + 1]


def test_instrument_lines_stale_call_site():
    with pytest.raises(ie.InstrumentError, match="re-run create"):
        ie.instrument_lines(lines_of(DRIVER), one(sites({CF: [DriverCall(CF, CALL_LN - 1, "b")]})))
    with pytest.raises(ie.InstrumentError, match="bounds_type"):
        ie.instrument_lines(lines_of(DRIVER), one(sites({CF: [DriverCall(CF, CALL_LN, None)]})))


def test_instrument_lines_rejects_manual_capture():
    with pytest.raises(ie.InstrumentError, match=r"hand-written capture calls at lines \[11\]"):
        ie.instrument_lines(lines_of(DRIVER), one(sites(capture_lns=[10])))


def test_make_accessible_private_module():
    decl = {"root_moist_stress_method": 5, "perchroot": 7, "perchroot_alt": 8}
    lines = lines_of(PRIV_MOD)
    head = head_of(PRIV_MOD)
    assert not any(head.is_public(n) for n in decl)
    out, changed = ie.make_accessible(lines, head, decl, "m.F90")
    assert changed == sorted(decl)
    new_head = head_of("".join(out))
    assert all(new_head.is_public(n) for n in decl)
    assert not new_head.is_public("other")
    assert "! comment" in "".join(out)
    # same (create-time) analysis on the edited file: no-op
    assert ie.make_accessible(out, head, decl, "m.F90") == (out, [])


def test_make_accessible_protected():
    head = head_of(PROT_MOD)
    out, changed = ie.make_accessible(
        lines_of(PROT_MOD), head, {"params_inst": 7, "other_inst": 8}, "m.F90"
    )
    assert changed == ["other_inst", "params_inst"]
    text = "".join(out)
    assert "type(photo_params_type), public :: params_inst  ! populated" in text
    assert "protected :: keep_inst\n" in text
    assert not head_of(text).is_protected("other_inst")


def test_make_accessible_stale_declaration():
    with pytest.raises(ie.InstrumentError, match="no longer declares perchroot"):
        ie.make_accessible(lines_of(PRIV_MOD), head_of(PRIV_MOD), {"perchroot": 6}, "m.F90")


TYPE_MOD = textwrap.dedent("""\
    module dynSubgridControlMod
      implicit none
      private
      type dyn_subgrid_control_type
         private
         logical :: do_transient_pfts = .false. ! comment
         logical, private :: do_harvest = .false.
         logical :: other = .false.
      contains
         procedure :: get
      end type dyn_subgrid_control_type
      type(dyn_subgrid_control_type), public :: dyn_subgrid_control_inst
    end module dynSubgridControlMod
    """)


def test_make_components_public():
    comps = {"do_transient_pfts": 5, "do_harvest": 6}
    out, changed = ie.components_public(
        lines_of(TYPE_MOD), {"dyn_subgrid_control_type": comps}, "m.F90"
    )
    assert changed == [
        "dyn_subgrid_control_type (default private)", "dyn_subgrid_control_type%do_harvest"
    ]
    assert len(out) == len(lines_of(TYPE_MOD))
    assert out[4] == "     ! private\n"
    assert out[5] == lines_of(TYPE_MOD)[5]
    assert out[6] == "     logical, public :: do_harvest = .false.\n"
    assert out[2] == "  private\n"  # module default untouched
    again = ie.components_public(out, {"dyn_subgrid_control_type": comps}, "m.F90")
    assert again == (out, [])
    with pytest.raises(ie.InstrumentError, match="no longer declares"):
        ie.components_public(lines_of(TYPE_MOD), {"dyn_subgrid_control_type": {"do_harvest": 7}}, "m.F90")


def test_capture_module():
    insts = ["canopystate_vars", "photosyns_vars"]
    src = ie.generate_capture_module(NAMES, insts, freq=4)
    flat = " ".join(src.replace("&\n", " ").split())
    assert src.startswith("module SpelCapture_canflux\n")
    assert "use elm_instMod, only : canopystate_vars, photosyns_vars" in flat
    assert "use ReadWriteMod_canflux, only : write_elmtypes" in src
    assert "use FUTConstantsMod_canflux, only : write_constants" in src
    assert "call io_inputs%init(base_fn='canflux.spel-inputs'" in src
    assert "type(spel_io_type) :: io_constants, io_inputs, io_outputs" in src
    assert "capture_freq = 4" in src
    assert (
        "call write_elmtypes(io_inputs, bounds, canopystate_vars=canopystate_vars, "
        "photosyns_vars=photosyns_vars)" in flat
    )
    assert max(len(ln) for ln in src.splitlines()) <= 132
    assert "use elm_instMod" not in ie.generate_capture_module(NAMES, [])


def test_capture_names():
    assert ie.CaptureNames.for_case("SoilLitt-VertTransp").tag == "soillitt_verttransp"
    assert ie.CaptureNames.for_case("2case").tag == "c2case"
    with pytest.raises(ie.InstrumentError):
        ie.CaptureNames.for_case("x" * 41)
    with pytest.raises(ie.InstrumentError):
        ie.CaptureNames.for_case("--")


def test_rename_io_modules():
    text = "module ReadWriteMod\n  use FUTConstantsMod, only: x\n  use readwritemod_other\nend module ReadWriteMod\n"
    out = ie.rename_io_modules(text, NAMES)
    assert out == (
        "module ReadWriteMod_canflux\n  use FUTConstantsMod_canflux, only: x\n"
        "  use readwritemod_other\nend module ReadWriteMod_canflux\n"
    )
    assert ie.rename_io_modules(out, NAMES) == out


def test_instrument_lines_multiple_cases():
    driver = DRIVER.replace(
        "       call t_stopf('canflux')\n",
        "       call t_stopf('canflux')\n       call UrbanFluxes(bounds_clump)\n",
    )
    uf = "urbanfluxesmod::urbanfluxes"
    both = {uf: [DriverCall(uf, CALL_LN + 4, "bounds_clump")], CF: [DriverCall(CF, CALL_LN, "bounds_clump")]}
    groups = [
        (sites({CF: both[CF]}), NAMES),
        (sites({uf: both[uf]}), ie.CaptureNames.for_case("urban")),
        (sites(both), ie.CaptureNames.for_case("both")),
    ]
    out, bounds = ie.instrument_lines(lines_of(driver), groups)
    assert set(bounds) == {"canflux", "urban", "both"}
    code = [ln.strip().split(" !")[0] for ln in out]
    assert code[1:4] == [
        "use SpelCapture_canflux, only : spel_capture_inputs_canflux, spel_capture_outputs_canflux",
        "use SpelCapture_urban, only : spel_capture_inputs_urban, spel_capture_outputs_urban",
        "use SpelCapture_both, only : spel_capture_inputs_both, spel_capture_outputs_both",
    ]
    i = code.index("call CanopyFluxes(bounds_clump,                         &")
    assert code[i - 2 : i] == [
        "call spel_capture_inputs_canflux(bounds_clump)",
        "call spel_capture_inputs_both(bounds_clump)",
    ]
    assert code[i + 3] == "call spel_capture_outputs_canflux(bounds_clump)"
    j = code.index("call UrbanFluxes(bounds_clump)")
    assert code[j - 1] == "call spel_capture_inputs_urban(bounds_clump)"
    # nested captures close in reverse order
    assert code[j + 1 : j + 3] == [
        "call spel_capture_outputs_urban(bounds_clump)",
        "call spel_capture_outputs_both(bounds_clump)",
    ]
    assert ie.strip_instrumentation(out) == lines_of(driver)
    assert ie.instrument_lines(out, groups)[0] == out


def fake_module(path: Path):
    lines = path.read_text().splitlines()
    return SimpleNamespace(
        filepath=path,
        module_lines=[LineTuple(line=ln.split("!")[0].lower(), ln=i) for i, ln in enumerate(lines)],
        end_of_head_ln=len(lines),
    )


@pytest.fixture
def fake_tree(tmp_path: Path, monkeypatch):
    src = tmp_path / "E3SM"
    main = src / "components/elm/src/main"
    bgp = src / "components/elm/src/biogeophys"
    main.mkdir(parents=True)
    bgp.mkdir(parents=True)
    (main / "elm_driver.F90").write_text(DRIVER)
    (bgp / "SoilMoistStressMod.F90").write_text(PRIV_MOD)
    (bgp / "PhotosynthesisMod.F90").write_text(PROT_MOD)
    case = tmp_path / "case"
    case.mkdir()
    for f in (*ie.SHARED_IO_MODULES, *ie.CASE_IO_MODULES):
        (case / f).write_text(f"module {f.removesuffix('.F90')}\nend module\n")

    fut = SimpleNamespace(
        case_dir=str(case),
        case_name="case",
        driver_sites=sites(),
        primary_subroutines={CF: None},
        module_dict={
            "soilmoiststressmod": fake_module(bgp / "SoilMoistStressMod.F90"),
            "photosynthesismod": fake_module(bgp / "PhotosynthesisMod.F90"),
        },
    )
    monkeypatch.setattr(
        ie,
        "io_symbols",
        lambda fut: {
            "soilmoiststressmod": {"perchroot": 7, "perchroot_alt": 8, "root_moist_stress_method": 5},
            "photosynthesismod": {"params_inst": 7, "photo_params_type": None},
        },
    )
    monkeypatch.setattr(ie, "capture_instances", lambda fut: ["canopystate_vars"])
    return src, case, fut


def test_instrument_elm_end_to_end(fake_tree):
    src, case, fut = fake_tree
    report = ie.instrument_elm(fut, src, freq=3)
    main = src / "components/elm/src/main"
    driver = (main / "elm_driver.F90").read_text()
    assert driver.count(ie.MARKER) == 3
    for f in report.copied:
        assert f.exists()
    assert {f.name for f in report.copied} == {
        "nc_io.F90", "nc_allocMod.F90", "ReadWriteMod_case.F90",
        "FUTConstantsMod_case.F90", "SpelCapture_case.F90",
    }
    assert (main / "ReadWriteMod_case.F90").read_text().startswith("module ReadWriteMod_case\n")
    assert not (case / "SpelCapture_case.F90").exists()
    capture = (main / "SpelCapture_case.F90").read_text()
    assert "capture_freq = 3" in capture and "canopystate_vars" in capture
    assert (case / ie.CAPTURE_DIR / "SpelCapture_case.F90").read_text() == capture
    assert report.made_public == {
        src / "components/elm/src/biogeophys/SoilMoistStressMod.F90": [
            "perchroot", "perchroot_alt", "root_moist_stress_method"
        ],
        src / "components/elm/src/biogeophys/PhotosynthesisMod.F90": ["params_inst"],
    }
    manifest = json.loads((case / ie.MANIFEST).read_text())
    assert manifest["bounds"] == "bounds_clump"
    assert manifest["tag"] == "case"

    # Re-running (same create-time analysis) changes nothing
    snapshot = {p: p.read_text() for p in src.rglob("*.F90")}
    assert ie.instrument_elm(fut, src, freq=3).made_public == {}
    assert {p: p.read_text() for p in src.rglob("*.F90")} == snapshot

    bgp = src / "components/elm/src/biogeophys"
    ledger = ie.load_access_ledger(src)
    assert set(ledger) == {
        "components/elm/src/biogeophys/SoilMoistStressMod.F90",
        "components/elm/src/biogeophys/PhotosynthesisMod.F90",
    }

    (main / "SpelCaptureMod.F90").write_text("stale")  # pre-tagging layout
    assert ie.uninstrument_elm(src) == sorted([
        main / "elm_driver.F90",
        bgp / "SoilMoistStressMod.F90",
        bgp / "PhotosynthesisMod.F90",
    ])
    assert (main / "elm_driver.F90").read_text() == DRIVER
    assert (bgp / "SoilMoistStressMod.F90").read_text() == PRIV_MOD
    assert (bgp / "PhotosynthesisMod.F90").read_text() == PROT_MOD
    assert sorted(p.name for p in main.iterdir()) == ["elm_driver.F90"]
    assert ie.uninstrument_elm(src) == []


def test_undo_survives_shifted_lines_and_keeps_drifted_edits(fake_tree):
    src, _, fut = fake_tree
    ie.instrument_elm(fut, src)
    bgp = src / "components/elm/src/biogeophys"
    soil, photo = bgp / "SoilMoistStressMod.F90", bgp / "PhotosynthesisMod.F90"
    # Developer adds a line at the top of one file and rewrites an edited
    # line of the other after instrumenting
    soil.write_text("! new comment\n" + soil.read_text())
    entry = ie.load_access_ledger(src)["components/elm/src/biogeophys/PhotosynthesisMod.F90"][0]
    lines = photo.read_text().splitlines(keepends=True)
    lines[entry["ln"]] = "  ! rewritten by hand\n"
    photo.write_text("".join(lines))

    ie.uninstrument_elm(src)
    assert soil.read_text() == "! new comment\n" + PRIV_MOD
    left = ie.load_access_ledger(src)
    assert list(left) == ["components/elm/src/biogeophys/PhotosynthesisMod.F90"]
    assert left[list(left)[0]][0]["original"] == entry["original"]


def test_instrument_elm_requires_elm_drv_call_site(fake_tree):
    src, case, fut = fake_tree
    fut.primary_subroutines = {CF: None, "lakemod::lake": None}
    with pytest.raises(ie.InstrumentError, match="not called from elm_drv"):
        ie.instrument_elm(fut, src)


def test_load_case_requires_pickle(tmp_path):
    with pytest.raises(ie.InstrumentError, match="run `spel create` first"):
        ie.load_case(str(tmp_path))


def test_collect_outputs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for name in ("spel-inputs0001.nc", "spel-outputs0001.nc", "elm.h0.nc", "a.spel-inputs0001.nc"):
        (run / name).write_text("x")
    copied = ie.collect_outputs(run, tmp_path / "data")
    assert [p.name for p in copied] == ["spel-inputs0001.nc", "spel-outputs0001.nc"]
    copied = ie.collect_outputs(run, tmp_path / "a", prefix="a.")
    assert copied == [tmp_path / "a/spel-inputs0001.nc"]
    assert ie.collect_outputs(run, tmp_path / "b", prefix="b.") == []
    assert not (tmp_path / "b").exists()


def test_instrument_elm_prefers_shared_mods_dir(fake_tree, tmp_path):
    src, case, fut = fake_tree
    mods = tmp_path / "SourceFiles"
    mods.mkdir()
    (mods / "nc_io.F90").write_text("module nc_io ! fresh\nend module\n")
    ie.instrument_elm(fut, src, mods_dir=mods)
    main = src / "components/elm/src/main"
    assert "fresh" in (main / "nc_io.F90").read_text()
    assert (main / "nc_allocMod.F90").read_text() == (case / "nc_allocMod.F90").read_text()


def test_run_case_protocol(tmp_path):
    casedir = tmp_path / "casedir"
    casedir.mkdir()
    xmlquery = casedir / "xmlquery"
    xmlquery.write_text(f"#!/bin/bash\necho {tmp_path}/run\n")
    xmlquery.chmod(0o755)
    script = tmp_path / "casegen.sh"
    script.write_text(f'#!/bin/bash\necho "args: $@"\necho CASEDIR={casedir}\n')
    rundir = ie.run_case(script, "mycase", tmp_path / "E3SM", ["--stop-n", "1"])
    assert rundir == tmp_path / "run"

    script.write_text("#!/bin/bash\nexit 3\n")
    with pytest.raises(ie.CaseRunError, match="exit code 3") as err:
        ie.run_case(script, "mycase", tmp_path)
    assert not err.value.build_failed

    log = tmp_path / "e3sm.bldlog"
    log.write_text(
        "noise\n/src/main/FUTConstantsMod_x.F90:48:4:\n\n   48 |  dim_names(1:0) = x\n"
        "      |    1\nError: Different shape for array assignment\nmore\n"
    )
    (casedir / "CaseStatus").write_text(
        f"t: case.build starting\n ---------\nt: case.build error\n"
        f"ERROR: BUILD FAIL: build e3sm failed, cat {log}\n ---------\nt: xmlchange success\n"
    )
    script.write_text(f"#!/bin/bash\necho CASEDIR={casedir}\nexit 1\n")
    with pytest.raises(ie.CaseRunError) as err:
        ie.run_case(script, "mycase", tmp_path)
    assert err.value.build_failed and err.value.casedir == casedir
    assert "FUTConstantsMod_x.F90:48:4:\nError: Different shape" in str(err.value)

    # a later successful build means the failure was in the run
    with (casedir / "CaseStatus").open("a") as f:
        f.write("t: case.build success\n ---------\n")
    assert ie.build_errors(casedir) is None


def test_instrument_cases_shares_one_build(fake_tree, tmp_path):
    src, case, fut = fake_tree
    case2 = tmp_path / "case2"
    case2.mkdir()
    for f in (*ie.SHARED_IO_MODULES, *ie.CASE_IO_MODULES):
        (case2 / f).write_text((case / f).read_text())
    fut2 = SimpleNamespace(**{**vars(fut), "case_dir": str(case2), "case_name": "case2"})
    bad = SimpleNamespace(**{**vars(fut), "case_name": "bad", "primary_subroutines": {"x::y": None}})
    result = ie.instrument_cases([fut, bad, fut2], src, freq=3)
    assert set(result.reports) == {"case", "case2"}
    assert "not called from elm_drv" in result.failures["bad"]
    main = src / "components/elm/src/main"
    driver = (main / "elm_driver.F90").read_text()
    assert driver.count(ie.MARKER) == 6
    assert "use SpelCapture_case2," in driver
    for tag in ("case", "case2"):
        assert (main / f"SpelCapture_{tag}.F90").exists()
        assert (main / f"FUTConstantsMod_{tag}.F90").exists()
    # each module is edited once with the union of the cases' symbols
    assert result.made_public[src / "components/elm/src/biogeophys/PhotosynthesisMod.F90"] == ["params_inst"]
    assert result.reports["case2"].made_public == result.reports["case"].made_public

    # re-instrumenting a subset drops the other case's modules
    ie.instrument_cases([fut2], src)
    assert not (main / "SpelCapture_case.F90").exists()
    assert "use SpelCapture_case," not in (main / "elm_driver.F90").read_text()


def test_instrument_cases_rejects_tag_clash(fake_tree):
    src, case, fut = fake_tree
    clash = SimpleNamespace(**{**vars(fut), "case_name": "CASE"})
    result = ie.instrument_cases([fut, clash], src)
    assert "clashes with case" in result.failures["CASE"]


def test_nc_read_string_stops_at_nul_padding():
    from spel.scripts.io.netcdf_io import gen_read_str

    src = " ".join(gen_read_str().split())
    assert "do i=1, min(strlen, len(var))" in src
    assert "if (buf(i:i) == achar(0)) exit" in src


def test_nc_def_character_scalar_has_one_dim():
    from spel.scripts.io.netcdf_io import create_nc_def
    from spel.scripts.utilityFunctions import Variable

    lines = create_nc_def({"urban_hac": Variable("character", "urban_hac", "", 0, 0)}, time=False)
    text = "".join(lines)
    assert "dim_names(1:1) = [character(len=32) :: 'urban_hac_str']" in text
    assert "call nc_define_var(ncid, 1, [len(urban_hac)], dim_names" in text


def test_instrument_lines_calls_in_different_branches():
    # elm_drv calls HydrologyDrainage in both branches of the pflotran if
    driver = DRIVER.replace(
        "       call t_stopf('canflux')\n",
        "       call t_stopf('canflux')\n"
        "       if (use_pflotran) then\n"
        "         call HydrologyDrainage(bounds_clump, a)\n"
        "       else\n"
        "         call HydrologyDrainage(bounds_clump, b)\n"
        "       end if\n",
    )
    hd = "hydrologydrainagemod::hydrologydrainage"
    calls = {hd: [DriverCall(hd, CALL_LN + 5, "bounds_clump"), DriverCall(hd, CALL_LN + 7, "bounds_clump")]}
    out, _ = ie.instrument_lines(lines_of(driver), one(sites(calls)))
    for i in (i for i, ln in enumerate(out) if "call HydrologyDrainage" in ln):
        assert "spel_capture_inputs" in out[i - 1]
        assert "spel_capture_outputs" in out[i + 1]
    assert ie.strip_instrumentation(out) == lines_of(driver)


def test_same_block():
    lines = lines_of("call a()\nif (x) then\n  call b()\nend if\ncall c()\nelse\n")
    assert ie._same_block(lines, 0, 4)
    assert not ie._same_block(lines, 2, 4)  # leaves the if
    assert not ie._same_block(lines, 4, 5)  # crosses an else
    assert not ie._same_block(lines, 0, 2)  # ends inside the if


def test_wait_for_case_jobs_polls_until_complete(tmp_path):
    import pytest
    from spel.scripts.instrument_elm import CaseRunError, case_job_ids, job_state, wait_for_case_jobs

    assert case_job_ids("case.run:60004107,case.st_archive:60004108") == ["60004107", "60004108"]
    assert case_job_ids("") == []
    assert job_state("   CANCELLED by 1234 \n") == "CANCELLED"
    assert job_state("") == ""

    states = iter(["PENDING", "RUNNING", "COMPLETED"])
    calls = []

    def run(cmd, cwd=None):
        calls.append(cmd[0])
        if cmd[0] == "./xmlquery":
            return "case.run:1\n"
        return next(states) + "\n"

    slept = []
    wait_for_case_jobs(tmp_path, run=run, sleep=slept.append, poll_s=1.0)
    assert calls.count("sacct") == 3 and slept == [1.0, 1.0]

    def failing(cmd, cwd=None):
        return "case.run:2\n" if cmd[0] == "./xmlquery" else "FAILED\n"

    with pytest.raises(CaseRunError):
        wait_for_case_jobs(tmp_path, run=failing, sleep=lambda s: None)

    with pytest.raises(CaseRunError):
        wait_for_case_jobs(tmp_path, run=lambda cmd, cwd=None: "case.run:3\n" if cmd[0] == "./xmlquery" else "RUNNING\n",
                           sleep=lambda s: None, timeout_s=2.0, poll_s=1.0)


def test_capture_module_compact_fields():
    """A compact field is gathered into a full-layout scratch array around every capture."""
    from spel.scripts.subset import SubsetSpec
    from spel.transforms.compact.unit_test import CompactField

    compact = [
        CompactField("veg_cf", "core", "leaf_to_litter", "vegetation_nutrient_flux", "NutrientFluxType", "VegetationDataType", "p"),
        CompactField("col_nf", "", "x", "column_nitrogen_flux", "ColumnFluxTypes", "ColumnDataType", "c"),
    ]
    src = ie.generate_capture_module(NAMES, ["cnstate_vars"], compact=compact)
    assert "use VegetationDataType, only : veg_cf" in src and "use ColumnDataType, only : col_nf" in src
    assert "use NutrientFluxType, only : vegetation_nutrient_flux_compact_bind" in src
    assert "real(r8), allocatable, target :: cap_veg_cf_core_leaf_to_litter(:)" in src
    assert "call gather_p(veg_cf%core%leaf_to_litter, cap_veg_cf_core_leaf_to_litter, bounds%begp, bounds%endp, proc%begp, proc%endp)" in src
    assert "call gather_c(col_nf%x, cap_col_nf_x, bounds%begc, bounds%endc, proc%begc, proc%endc)" in src
    assert "veg_cf%core%leaf_to_litter => cap_veg_cf_core_leaf_to_litter" in src
    assert "call vegetation_nutrient_flux_compact_bind(veg_cf%core)" in src and "call column_nitrogen_flux_compact_bind(col_nf)" in src
    assert src.count("call compact_capture_begin(bounds)") == 2 and src.count("call compact_capture_end()") == 2
    for io in ("io_inputs", "io_outputs"):
        at = src.index(f"write_elmtypes({io}")
        assert src.rfind("compact_capture_begin", 0, at) > src.rfind("compact_capture_end", 0, at)
        assert src.index("compact_capture_end", at) < src.index("end subroutine", at)
    # subset captures get the same treatment
    spec = SubsetSpec.parse("urban")
    sub = ie.generate_capture_module(NAMES, ["cnstate_vars"], subset=spec, compact=compact)
    assert sub.count("call compact_capture_begin(bounds)") == 2 and "subroutine gather_p" in sub
    # without compact fields the module is unchanged
    assert ie.generate_capture_module(NAMES, ["cnstate_vars"]) == ie.generate_capture_module(NAMES, ["cnstate_vars"], compact=())
    assert "compact_capture" not in ie.generate_capture_module(NAMES, ["cnstate_vars"])


def test_compact_fields_from_manifest_and_module():
    """Fields of the manifest's groups, for active instances and used components; the unit-test module."""
    from types import SimpleNamespace

    from spel.transforms.compact.unit_test import compact_fields, compact_module, deactivate_runtime_components, hook_lines

    def comp(name, type_, active=True):
        return SimpleNamespace(name=name, type=type_, active=active)

    base = SimpleNamespace(
        type_name="vegetation_nutrient_flux", declaration="NutrientFluxType",
        components={"x": comp("x", "real"), "y": comp("y", "real"), "cgrp_id": comp("cgrp_id", "integer")},
    )
    carbon = SimpleNamespace(
        type_name="vegetation_carbon_flux", declaration="VegetationFluxTypes",
        components={"core": comp("core", "vegetation_nutrient_flux"), "core%x": comp("core%x", "real"),
                    "core%y": comp("core%y", "real", active=False), "core%cgrp_id": comp("core%cgrp_id", "integer")},
    )
    type_dict = {t.type_name: t for t in (base, carbon)}
    active = {"veg_cf": SimpleNamespace(type="vegetation_carbon_flux", declaration="VegetationDataType")}
    manifest = {"types": {"vegetation_nutrient_flux": {
        "level": "p", "instances": ["veg_cf", "veg_nf"], "embedded_in": {"veg_cf": "core", "veg_nf": "core"},
        "groups": [{"index": 1, "guard": None, "fields": ["x", "y"]}]}}}
    fields = compact_fields(type_dict, active, manifest)
    assert [(f.instance, f.path, f.comp, f.subgrid) for f in fields] == [("veg_cf", "core", "x", "p")]
    assert fields[0].shadow == "sh_veg_cf_core_x" and fields[0].bind_call == "call vegetation_nutrient_flux_compact_bind(veg_cf%core)"
    mod = compact_module(fields)
    assert "sh_veg_cf_core_x => veg_cf%core%x" in mod
    assert "veg_cf%core%x(off+1:off+n) = sh_veg_cf_core_x(filter(nc)%soilp(1:n))" in mod
    assert "sh_veg_cf_core_x(filter(nc)%soilp(1:n)) = veg_cf%core%x(off+1:off+n)" in mod
    assert "veg_cf%core%x => sh_veg_cf_core_x" in mod
    assert "call compact_filters_set(compact_lev_col" in mod
    hooks = hook_lines(fields)
    assert hooks["!#COMPACT_BEFORE"] == ["      call spel_compact_pack()\n"]
    assert any("spel_compact_init(bounds_proc)" in l for l in hooks["!#COMPACT_INIT"])
    assert deactivate_runtime_components(type_dict) == ["vegetation_carbon_flux%core%cgrp_id", "vegetation_nutrient_flux%cgrp_id"]
    assert not carbon.components["core%cgrp_id"].active
    assert compact_fields(type_dict, active, None) == [] and hook_lines([]) == {}
