"""
Validation of SPEL's static read/write classification against data:
for each timestep, the state before the call (spel-inputs) is compared with
the state after it (spel-outputs).
  * variables classified read-only must never change (hard failure)
  * variables SPEL never saw must never change either (hard failure)
  * outputs that never change on any step are only flagged
"""

import json

import numpy as np
import pytest
import xarray as xr

from spel.scripts import validate_access as va
from spel.scripts.types import ReadWrite

NT = 3


def dataset(**arrays) -> xr.Dataset:
    data = {}
    for name, arr in arrays.items():
        arr = np.asarray(arr, dtype=float)
        dims = ("time", "patch")[: arr.ndim] if arr.ndim and arr.shape[0] == NT else ("patch",)
        data[name] = (dims, arr)
    return xr.Dataset(data)


BASE = np.arange(NT * 4, dtype=float).reshape(NT, 4)


@pytest.fixture
def pre():
    return dataset(
        veg_es__thm=BASE,
        veg_es__t_veg=BASE,
        veg_ef__eflx=BASE,
        veg_ws__h2o=BASE,
        bounds__begp=np.ones(NT),
        top_as__tbot=BASE,
        hidden__x=BASE,
        static__p=np.arange(4.0),
    )


ACCESS = {
    "veg_es%thm": "r",
    "top_as%tbot": "r",
    "veg_es%t_veg": "w",
    "veg_ef%eflx": "rw",
    "veg_ws%h2o": "w",
    "static%p": "r",
    "gone%v": "r",
}


def post_from(pre: xr.Dataset, **changes) -> xr.Dataset:
    post = pre.copy(deep=True)
    for name, fn in changes.items():
        post[name].values[...] = fn(post[name].values)
    return post


def test_consistent_run_passes(pre):
    def bump_step1(v):
        v[1, 2] += 1.0
        return v

    post = post_from(
        pre, veg_es__t_veg=lambda v: v + 1.0, veg_ef__eflx=bump_step1, veg_ws__h2o=lambda v: v * 2
    )
    report = va.validate_access(pre, post, ACCESS)
    assert report.ok
    assert report.violations == []
    assert report.unchanged_outputs == []
    assert report.missing == ["gone__v"]
    assert report.nsteps == NT


def test_changed_input_is_a_violation(pre):
    def bump(v):
        v[2, 0] = v[2, 0] + 0.5
        v[0, 3] = -1.0
        return v

    post = post_from(pre, veg_es__thm=bump, veg_es__t_veg=lambda v: v + 1, veg_ef__eflx=lambda v: v + 1, veg_ws__h2o=lambda v: v + 1)
    report = va.validate_access(pre, post, ACCESS)
    assert not report.ok
    (v,) = report.violations
    # steps are 1-based, like the coordinates in the diff report
    assert (v.name, v.status, v.steps_changed, v.first_step) == ("veg_es__thm", "r", 2, 1)
    assert v.max_abs_diff == pytest.approx(4.0)

def test_unclassified_variable_that_changes_is_a_violation(pre):
    post = post_from(pre, hidden__x=lambda v: v + 1, veg_es__t_veg=lambda v: v + 1, veg_ef__eflx=lambda v: v + 1, veg_ws__h2o=lambda v: v + 1)
    report = va.validate_access(pre, post, ACCESS)
    assert [(v.name, v.status) for v in report.violations] == [("hidden__x", "-")]


def test_unchanged_outputs_are_flagged_not_failed(pre):
    post = post_from(pre, veg_es__t_veg=lambda v: v + 1)
    report = va.validate_access(pre, post, ACCESS)
    assert report.ok
    assert report.unchanged_outputs == ["veg_ef__eflx", "veg_ws__h2o"]


def test_nan_is_equal_to_nan(pre):
    pre["veg_es__thm"].values[1, 1] = np.nan
    post = post_from(pre, veg_es__t_veg=lambda v: v + 1, veg_ef__eflx=lambda v: v + 1, veg_ws__h2o=lambda v: v + 1)
    assert va.validate_access(pre, post, ACCESS).ok


def test_time_length_mismatch_uses_common_steps(pre):
    post = post_from(pre, veg_es__t_veg=lambda v: v + 1, veg_ef__eflx=lambda v: v + 1, veg_ws__h2o=lambda v: v + 1)
    report = va.validate_access(pre, post.isel(time=slice(0, 2)), ACCESS)
    assert report.ok and report.nsteps == 2


def test_merge_statuses_across_subroutines():
    class Sub:
        def __init__(self, d):
            self.elmtype_access_summary = {k: ReadWrite(s, -1, None) for k, s in d.items()}

    merged = va.access_manifest(
        [Sub({"a%x": "r", "a%y": "w", "a%z": "r"}), Sub({"a%x": "r", "a%y": "r", "a%z": "w", "a%w": "rw"})]
    )
    assert merged == {"a%x": "r", "a%y": "rw", "a%z": "rw", "a%w": "rw"}


def test_manifest_round_trip(tmp_path):
    va.write_access_manifest(tmp_path, ["mod::sub"], {"a%x": "r", "a%y": "w"})
    raw = json.loads((tmp_path / va.ACCESS_MANIFEST_FILENAME).read_text())
    assert raw["subroutines"] == ["mod::sub"]
    assert va.load_access_manifest(tmp_path) == {"a%x": "r", "a%y": "w"}


def test_run_validation_exit_code(tmp_path, pre):
    good = post_from(pre, veg_es__t_veg=lambda v: v + 1, veg_ef__eflx=lambda v: v + 1, veg_ws__h2o=lambda v: v + 1)
    bad = post_from(good, top_as__tbot=lambda v: v + 1)
    pre.to_netcdf(tmp_path / "in.nc")
    good.to_netcdf(tmp_path / "good.nc")
    bad.to_netcdf(tmp_path / "bad.nc")
    va.write_access_manifest(tmp_path, ["mod::sub"], ACCESS)
    assert va.run_validation(tmp_path / "in.nc", tmp_path / "good.nc", tmp_path) == 0
    assert va.run_validation(tmp_path / "in.nc", tmp_path / "bad.nc", tmp_path) == 1


# --- namelist guards: fields SPEL found to be accessed only under namelist ifs ---
from spel.scripts.fortran_parser.boolen_expression import AllOf, AnyOf, Expectation

PHS = Expectation("use_hydrstress", "True")
GUARDS = {"veg_ws%h2o": PHS, "gone%v": PHS}


@pytest.mark.parametrize(
    "cond, nml, expected",
    [
        (PHS, {"use_hydrstress": 1}, True),
        (PHS, {"use_hydrstress": 0}, False),
        (Expectation("use_hydrstress", "False"), {"use_hydrstress": 0}, True),
        (PHS, {}, None),  # unknown value -> can't decide
        (Expectation("method", "== 2"), {"method": 2}, True),
        (Expectation("method", ".ne. 2"), {"method": 2}, False),
        (Expectation("method", "> nlev"), {"method": 2, "nlev": 1}, True),
        (Expectation("flag", "== .true."), {"flag": 0}, False),
        (Expectation("x", "== 1.0_r8"), {"x": 1.0}, True),
        (AllOf((PHS, Expectation("crop", "True"))), {"use_hydrstress": 0}, False),
        (AllOf((PHS, Expectation("crop", "True"))), {"use_hydrstress": 1}, None),
        (AnyOf((PHS, Expectation("crop", "True"))), {"use_hydrstress": 1}, True),
        (AnyOf((PHS, Expectation("crop", "True"))), {"use_hydrstress": 0}, None),
    ],
)
def test_evaluate_guard(cond, nml, expected):
    assert va.evaluate_guard(cond, nml) is expected


def test_namelist_values_are_scalars_only():
    ds = xr.Dataset({"use_hydrstress": ((), np.int32(0)), "crop": (("n",), np.zeros(3))})
    assert va.namelist_values(ds) == {"use_hydrstress": 0}


def test_inactive_guard_explains_unchanged_and_missing_outputs(pre):
    post = post_from(pre, veg_es__t_veg=lambda v: v + 1, veg_ef__eflx=lambda v: v + 1)
    report = va.validate_access(pre, post, ACCESS, GUARDS, {"use_hydrstress": 0})
    assert report.ok
    assert report.unchanged_outputs == []
    assert report.missing == []
    assert report.inactive == ["gone__v", "veg_ws__h2o"]


def test_active_guard_keeps_warning_and_reports_guard(pre):
    post = post_from(pre, veg_es__t_veg=lambda v: v + 1, veg_ef__eflx=lambda v: v + 1)
    report = va.validate_access(pre, post, ACCESS, GUARDS, {"use_hydrstress": 1})
    assert report.unchanged_outputs == ["veg_ws__h2o"]
    assert report.missing == ["gone__v"]
    assert report.inactive == []
    assert "use_hydrstress" in va.format_report(report)


def test_change_under_inactive_guard_is_a_violation(pre):
    post = post_from(pre, veg_ws__h2o=lambda v: v + 1)
    report = va.validate_access(pre, post, ACCESS, GUARDS, {"use_hydrstress": 0})
    assert not report.ok
    assert [v.name for v in report.violations] == ["veg_ws__h2o"]
    assert report.violations[0].guard == PHS.to_fortran()


def test_guards_merge_across_subroutines():
    # guarded only if every subroutine accessing the field guards it
    merged = va.merge_guards(
        [
            ({"a%x": "w", "a%y": "w", "a%z": "w"}, {"a%x": PHS, "a%y": PHS}),
            ({"a%x": "r", "a%y": "r"}, {"a%x": Expectation("crop", "True")}),
        ]
    )
    assert set(merged) == {"a%x"}
    assert va.evaluate_guard(merged["a%x"], {"use_hydrstress": 0, "crop": 1}) is True
    assert va.evaluate_guard(merged["a%x"], {"use_hydrstress": 0, "crop": 0}) is False


def test_guard_manifest_round_trip(tmp_path):
    cond = AnyOf((PHS, AllOf((Expectation("m", "== 2"), Expectation("crop", "False")))))
    va.write_access_manifest(tmp_path, ["mod::sub"], {"a%x": "w"}, {"a%x": cond})
    assert va.load_access_guards(tmp_path) == {"a%x": cond}
    va.write_access_manifest(tmp_path, ["mod::sub"], {"a%x": "w"})
    assert va.load_access_guards(tmp_path) == {}


def test_run_validation_reads_namelist_from_constants(tmp_path, pre):
    post = post_from(pre, veg_es__t_veg=lambda v: v + 1, veg_ef__eflx=lambda v: v + 1)
    pre.to_netcdf(tmp_path / "spel-inputs0001.nc")
    post.to_netcdf(tmp_path / "spel-outputs0001.nc")
    xr.Dataset({"use_hydrstress": ((), np.int32(0))}).to_netcdf(tmp_path / "spel-constants0001.nc")
    va.write_access_manifest(tmp_path, ["mod::sub"], ACCESS, GUARDS)

    import io

    out = io.StringIO()
    code = va.run_validation(
        tmp_path / "spel-inputs0001.nc", tmp_path / "spel-outputs0001.nc", tmp_path, ostream=out
    )
    assert code == 0
    assert "WARNING" not in out.getvalue()
    assert "inactive" in out.getvalue()
