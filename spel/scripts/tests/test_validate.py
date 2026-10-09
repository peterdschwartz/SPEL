import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spel.scripts import analysis_cache, config
from spel.scripts import validate as v
from spel.scripts.instrument_elm import (
    CaseRunError,
    InstrumentError,
    InstrumentReport,
    MultiInstrumentReport,
)


def test_parse_request(tmp_path):
    assert v.parse_request(["CanopyFluxes", "soillittverttransp"]) == {
        "canopyfluxes": ["canopyfluxes"],
        "soillittverttransp": ["soillittverttransp"],
    }
    lst = tmp_path / "cases.txt"
    lst.write_text("# ecosystem pieces\nlitt: SoilLittVertTransp  # comment\n\ncanopyfluxes\nb: x y\n")
    assert v.parse_request(list_file=lst) == {
        "litt": ["soillittverttransp"],
        "canopyfluxes": ["canopyfluxes"],
        "b": ["x", "y"],
    }
    with pytest.raises(ValueError, match="listed twice"):
        v.parse_request(["canopyfluxes"], lst)
    lst.write_text("a b\n")
    with pytest.raises(ValueError, match="expected"):
        v.parse_request(list_file=lst)
    with pytest.raises(ValueError, match="no cases"):
        v.parse_request()


@pytest.fixture
def env(tmp_path, monkeypatch):
    ut = tmp_path / "unit-tests"
    ut.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / analysis_cache.ANALYSIS_PICKLE).write_text("")
    casegen = tmp_path / "casegen.sh"
    casegen.write_text("")
    rundir = tmp_path / "run"
    rundir.mkdir()
    (rundir / "lnd_in").write_text("&elm_inparm\n/\n")
    for f in ("a.spel-inputs0001.nc", "a.spel-outputs0001.nc", "a.spel-constants0001.nc"):
        (rundir / f).write_text("x")
    monkeypatch.setattr(config, "unittests_dir", ut)
    monkeypatch.setattr(config, "input_data_dir", ut / "input-data")
    monkeypatch.setattr(config, "CASEGEN_SCRIPT", casegen)
    monkeypatch.setattr(config, "E3SM_SRCROOT", tmp_path / "E3SM")
    monkeypatch.setattr(analysis_cache, "cache_dir", lambda: cache)

    calls = SimpleNamespace(runs=[], uninstrumented=0, ran=[], resubmits=0)

    def create_cases(report, jobs, log_dir):
        for case, r in report.cases.items():
            if case == "broken":
                r.status = v.CREATE_FAILED
            else:
                (ut / case).mkdir()
                (ut / case / "fut.pkl").write_text("")

    def instrument_cases(futs, srcroot, freq, dry_run, mods_dir):
        res = MultiInstrumentReport()
        for fut in futs:
            if fut.case_name == "nosite":
                res.failures["nosite"] = "not called from elm_drv"
            else:
                res.reports[fut.case_name] = InstrumentReport(
                    Path("elm_driver.F90"), "elm_driver", "bounds_clump", tag=fut.case_name
                )
        return res

    def run_case(script, name, srcroot, args):
        calls.runs.append(list(args or []))
        if len(calls.runs) == 1 and calls.fail_first:
            raise CaseRunError("exit code 1", tmp_path / "casedir", build_failed=calls.build_fails)
        return rundir

    def resubmit_case(casedir):
        calls.resubmits += 1
        return rundir

    def uninstrument(srcroot):
        calls.uninstrumented += 1
        return []

    def run_cases(report, cases, jobs):
        calls.ran = cases
        for c in cases:
            report.cases[c].status = v.PASS

    calls.fail_first = False
    calls.build_fails = False
    monkeypatch.setattr(v, "resubmit_case", resubmit_case)
    monkeypatch.setattr(v, "create_cases", create_cases)
    monkeypatch.setattr(v, "instrument_cases", instrument_cases)
    monkeypatch.setattr(v, "load_case", lambda case: SimpleNamespace(case_name=case))
    monkeypatch.setattr(v, "run_case", run_case)
    monkeypatch.setattr(v, "uninstrument_elm", uninstrument)
    monkeypatch.setattr(v, "run_cases", run_cases)
    return ut, calls


def test_validate_flow(env):
    ut, calls = env
    calls.fail_first = True
    request = {"a": ["canopyfluxes"], "b": ["x"], "broken": ["y"], "nosite": ["z"]}
    report = v.validate(request, case_args=["--stop-n", "1"])

    assert calls.runs == [["--stop-n", "1"]]
    assert calls.resubmits == 1
    assert calls.uninstrumented == 1
    assert calls.ran == ["a"]
    status = {c: r.status for c, r in report.cases.items()}
    assert status == {
        "a": v.PASS, "b": v.NOT_CALLED, "broken": v.CREATE_FAILED, "nosite": v.INSTRUMENT_FAILED
    }
    assert not report.ok
    data = ut / "input-data/a"
    assert sorted(p.name for p in data.iterdir()) == [
        "lnd_in", "spel-constants0001.nc", "spel-inputs0001.nc", "spel-outputs0001.nc"
    ]
    saved = json.loads((ut / v.REPORT_FILE).read_text())
    assert saved["cases"]["b"]["status"] == v.NOT_CALLED
    assert saved["lnd_in"].endswith("lnd_in")


def test_validate_keeps_instrumentation_and_reports_run_failure(env, monkeypatch):
    ut, calls = env

    calls.fail_first = calls.build_fails = True
    report = v.validate({"a": ["canopyfluxes"]}, keep_instrumentation=True)
    assert len(calls.runs) == 1 and calls.resubmits == 0  # builds aren't retried
    assert calls.uninstrumented == 0
    assert report.cases["a"].status == v.CASE_RUN_FAILED


def test_validate_rejects_bad_case_name(env):
    with pytest.raises(InstrumentError, match="can't be used in Fortran"):
        v.validate({"x" * 50: ["canopyfluxes"]})


def test_run_logged_has_no_stdin(tmp_path):
    log = tmp_path / "x.log"
    code = v.run_logged(["bash", "-c", "read -p 'ok? ' a; echo got=$a"], log)
    assert code == 0
    assert "got=" in log.read_text()
