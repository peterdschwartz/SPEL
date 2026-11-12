"""Multi-step runs (SPEC.md "Multi-step runs"): time, the runner on the fake case, the spec, coupling."""

import io

import numpy as np
import pytest

from spel.proptest import runner
from spel.proptest import spec as sp
from spel.proptest import timestep
from spel.proptest.errors import SpecError
from spel.proptest.tests import fake

pytest.importorskip("netCDF4")


# -- time
def test_advance_rolls_over_day_month_and_year():
    t = {"nstep_mod": 7, "spel_nstep": 7, "dtime_mod": 1800.0, "year_curr": 2000, "mon_curr": 12, "day_curr": 31,
         "secs_curr": 84600, "year_prev": 0, "mon_prev": 0, "day_prev": 0, "secs_prev": 0, "jday_mod": 365,
         "thiscalday_mod": 365.979, "nextsw_cday_mod": 366.0, "end_cd_mod": False, "doalb": True}
    n = timestep.advance(t)
    assert (n["year_curr"], n["mon_curr"], n["day_curr"], n["secs_curr"]) == (2001, 1, 1, 0)
    assert (n["year_prev"], n["mon_prev"], n["day_prev"], n["secs_prev"]) == (2000, 12, 31, 84600)
    assert n["nstep_mod"] == 8 and n["spel_nstep"] == 8 and n["jday_mod"] == 1
    assert n["thiscalday_mod"] == 1.0 and n["end_cd_mod"] is True and n["doalb"] is True
    assert abs(n["nextsw_cday_mod"] - (1.0 + 1800 / 86400)) < 1e-12
    n2 = timestep.advance(n)
    assert n2["secs_curr"] == 1800 and n2["end_cd_mod"] is False
    # 28 February -> 1 March (no-leap calendar), calday 60
    m = timestep.advance({"dtime_mod": 3600.0, "year_curr": 2004, "mon_curr": 2, "day_curr": 28, "secs_curr": 82800,
                          "jday_mod": 59})
    assert (m["mon_curr"], m["day_curr"], m["secs_curr"]) == (3, 1, 0) and m["jday_mod"] == 60
    assert timestep.calday(7, 1, 43200) == 182.5
    assert timestep.inconsistent({"mon_curr": 7, "day_curr": 1, "secs_curr": 43200, "jday_mod": 182}) == []
    assert timestep.inconsistent({"mon_curr": 7, "day_curr": 2, "secs_curr": 0, "jday_mod": 182})
    with pytest.raises(ValueError):
        timestep.advance({"dtime_mod": 1800.5, "mon_curr": 1, "day_curr": 1, "secs_curr": 0})


# -- the runner on the fake case: state(p) = state(p) + 1 every call, y(p) = 2 x(p) + state(p) before it
def load(tmp_path, scenario=None, **tables):
    case = tmp_path / "case"
    inv = fake.inventory(case)
    data = fake.spec_data(case, **tables)
    if scenario is not None:
        data["scenario"]["s"].update(scenario)
    return sp.parse(data, tmp_path / "t.proptest.toml", case, inv), inv


def run(tmp_path, scenario=None, **tables):
    s, inv = load(tmp_path, scenario, **tables)
    out = io.StringIO()
    backend = fake.FakeBackend()
    with runner.Session(s, inv, backend=backend) as session:
        res = runner.run(s, inv, settings={"seed": 0, "max_examples": 10}, out=out, session=session)
    return res, backend


def test_state_is_carried_and_prev_tend_step_see_it(tmp_path):
    res, backend = run(tmp_path, {"steps": 3}, property=[
        {"name": "carried", "expr": "state(p) == input%state(p) + step"},
        {"name": "prev", "expr": "state(p) == prev%state(p) + 1.0"},
        {"name": "tend", "expr": "close(tend(state(p)), 1.0)"},
        {"name": "tend of an expression", "expr": "close(tend(2.0 * state(p) + x(p)), 2.0)"},
        {"name": "y uses the state before the step", "expr": "close(y(p), 2.0 * x(p) + prev%state(p))"},
    ])
    assert [(o.check, o.passed) for o in res] == [(o.check, True) for o in res], [o.message for o in res]


def test_one_call_tend_is_against_the_input(tmp_path):
    res, _ = run(tmp_path, property=[{"name": "tend", "expr": "close(tend(state(p)), 1.0) .and. step == 1"}])
    assert res[0].passed, res[0].message


def test_a_failure_names_the_step(tmp_path):
    res, _ = run(tmp_path, {"steps": 4}, property=[{"name": "bounded", "expr": "state(p) < input%state(p) + 2.5"}])
    assert not res[0].passed
    assert "after step 3 of 4" in res[0].message


def test_steps_selection(tmp_path):
    res, _ = run(tmp_path, {"steps": 4}, property=[
        {"name": "last", "expr": "state(p) == input%state(p) + 4.0", "steps": "last"},
        {"name": "range", "expr": "state(p) >= input%state(p) + 2.0", "steps": [2, 3]},
        {"name": "one", "expr": "state(p) == input%state(p) + 1.0", "steps": 1},
    ])
    assert all(o.passed for o in res), [o.message for o in res]


def test_forcing_rules_and_expressions_are_set_before_every_step(tmp_path):
    res, backend = run(tmp_path, {"steps": 3, "forcing": {"a%x": {"range": [5.0, 6.0]}}}, property=[
        {"name": "forced", "expr": "x(p) >= 5.0 .and. close(y(p), 2.0 * x(p) + prev%state(p))"}])
    assert res[0].passed, res[0].message
    res, _ = run(tmp_path, {"steps": 3, "forcing": {"a%x": "0.5 * step + prev%state(p)"}}, property=[
        {"name": "by expression", "expr": "close(x(p), 0.5 * step + prev%state(p))"}])
    assert res[0].passed, res[0].message


def test_forcing_draws_differ_between_steps(tmp_path):
    s, inv = load(tmp_path, {"steps": 5, "forcing": {"a%x": {"range": [0.0, 1.0]}}})
    with runner.Session(s, inv, backend=fake.FakeBackend()) as session:
        seen = set()
        for seed in range(12):
            ex = session.sample("s", seed=seed)
            f = ex.forcing["a%x"]
            assert f.shape[0] == 5 and f.shape[1:] == ex.inputs["a%x"].shape
            seen.add(bool(np.ptp(f, axis=0).max() > 0))
        assert seen == {True, False}  # some modes vary over the steps, the constant one does not


def test_between_runs_after_every_step_and_the_next_step_starts_from_it(tmp_path):
    res, _ = run(tmp_path, {"steps": 3, "between": {"a%state": "0.5 * state(p)"}}, property=[
        {"name": "halved after the step", "expr": "close(state(p), 0.5 * (prev%state(p) + 1.0))"},
        {"name": "the routine saw the halved state", "expr": "close(y(p), 2.0 * x(p) + prev%state(p))"},
    ])
    assert all(o.passed for o in res), [o.message for o in res]


def test_child_aggregates_reduce_over_descendants(tmp_path):
    # the fake domain has one patch per gridcell (weight 1): the aggregate is the patch's value
    res, _ = run(tmp_path, {"steps": 2}, property=[
        {"name": "sum", "expr": "close(child_sum(a%state, 'g'), state(p)) .and. close(child_wsum(b%y, 'g'), y(p))"},
        {"name": "tendency of a sum", "expr": "close(tend(child_sum(a%state, 'g')), 1.0)"},
    ])
    assert all(o.passed for o in res), [o.message for o in res]


def test_execute_keeps_every_step(tmp_path):
    s, inv = load(tmp_path, {"steps": 3})
    with runner.Session(s, inv, backend=fake.FakeBackend()) as session:
        ex = session.sample("s", seed=1)
        res = session.execute("s", ex)
        series = res.series("a%state")
        assert series.shape[0] == 3
        np.testing.assert_allclose(series - ex.inputs["a%state"], np.arange(1, 4)[:, None] * np.ones_like(series))
        assert res.times[1]["nstep_mod"] == res.times[0]["nstep_mod"] + 1 if "nstep_mod" in res.times[0] else True
        np.testing.assert_allclose(res.at(2).arrays["a%state"].data, series[1])


@pytest.mark.parametrize("scenario, message", [
    ({"steps": 0}, "steps"),
    ({"forcing": {"a%x": {"range": [0.0, 1.0]}}}, "need `steps` > 1"),
    ({"steps": 2, "between": {"b%y": "1.0"}}, "between sets input or inout"),
    ({"steps": 2, "keep": ["nothing%like_this"]}, "matches no variable"),
    ({"steps": 2, "time": {"no_such_time": 1}}, "per-step scalar"),
    ({"steps": 2, "carry": {"a%x": 0.0}}, "`carry` is for outputs"),
    ({"carry": {"b%partial": 0.0}}, "`carry` needs `steps` > 1"),
    ({"steps": 2, "carry": {"b%partial": {"scale": [0.5, 2.0]}}}, "relative to the captured value"),
])
def test_spec_errors(tmp_path, scenario, message):
    with pytest.raises(SpecError, match=message):
        load(tmp_path, scenario)


def test_check_steps_must_be_valid(tmp_path):
    with pytest.raises(SpecError, match="steps"):
        load(tmp_path, {"steps": 2}, property=[{"name": "x", "expr": "y(p) > 0.0", "steps": [3, 1]}])


def test_boundary_declared():
    s = sp.Scenario("s", "c", None, {}, [], [], steps=2, keep=["col_ws%*"], forcing={"top_af%rain": None})
    assert s.boundary_declared("col_ws%h2osoi_liq") and s.boundary_declared("top_af%rain")
    assert not s.boundary_declared("col_wf%qflx_surf")


# -- coupling on the frozen elm_drv analysis
def test_coupling_of_hydrology_drainage(elm_drv_analysis):
    from pathlib import Path

    from spel.proptest import coupling

    fut = elm_drv_analysis
    if "hydrologydrainagemod::hydrologydrainage" not in fut.subroutine_dict:
        pytest.skip("the fixture's analysis has no HydrologyDrainage")
    cpl = coupling.trace_coupling(fut, Path("/"), ["hydrologydrainagemod::hydrologydrainage"],
                                  {"col_wf%qflx_surf", "col_ws%h2osoi_liq", "top_ws%from_uphill"})
    surf = {w.sub for w in cpl.writers.get("col_wf%qflx_surf", [])}
    assert "soilhydrologymod::surfacerunoff" in surf
    assert all(w.when == "before" for w in cpl.writers["col_wf%qflx_surf"] if w.sub == "soilhydrologymod::surfacerunoff")
    if "top_ws%from_uphill" in cpl.writers:  # IM2 code in the fixture's E3SM
        drain = [w for w in cpl.writers["top_ws%from_uphill"] if w.sub == "balancecheckmod::colwaterbalancecheck"]
        assert drain and all(w.when == "after" for w in drain)
    assert "*" in {k for ks in cpl.case_filters.values() for k in ks} or cpl.case_filters


def test_acc_sums_over_the_steps(tmp_path):
    # state after step j is s0 + j: acc(state) after step k = k s0 + k (k + 1) / 2
    res, _ = run(tmp_path, {"steps": 4}, property=[
        {"name": "acc", "expr": "close(acc(state(p)), step * input%state(p) + step * (step + 1) / 2.0)"},
        {"name": "acc of a tendency", "expr": "close(acc(tend(state(p))), real(step))"},
    ])
    assert all(o.passed for o in res), [o.message for o in res]


def test_tend_and_acc_see_the_halves_of_each_step(tmp_path):
    res, _ = run(tmp_path, {"steps": 3}, relation=[
        {"name": "halves agree on tendencies", "expect": "close(tend(follow%state(p)), tend(base%state(p)))",
         "steps": [2, 3]},
        {"name": "halves agree on sums", "expect": "close(acc(follow%y(p)), acc(base%y(p)))"},
    ])
    assert all(o.passed for o in res), [o.message for o in res]
    res, _ = run(tmp_path, {"steps": 3}, relation=[
        {"name": "before step 1", "expect": "close(tend(follow%state(p)), 1.0)"}])
    assert not res[0].passed and "before the step is not available" in res[0].message


def test_topounit_functions_from_the_declaration(tmp_path):
    hills = [{"name": "hill", "topounits": [
        {"weight": 0.5, "elevation": 300.0, "landunits": [{"kind": "soil", "patches": [1]}]},
        {"weight": 0.3, "elevation": 50.0, "landunits": [{"kind": "soil", "patches": [13]}]},
        {"weight": 0.2, "elevation": 150.0, "landunits": [{"kind": "soil", "patches": [13]}]}]}]
    domain = {"gridcell": hills, "nsets": 2}
    res, _ = run(tmp_path, domain=domain, property=[
        {"name": "elevations", "expr": "elevation(t) == merge(300.0, merge(50.0, 150.0, mod(t - 1, 3) == 1), "
                                       "mod(t - 1, 3) == 0)"},
        {"name": "lowest", "expr": "lowest(t) .eqv. (mod(t - 1, 3) == 1)"},
        # ELM's downhill_ti: 300 -> 150, 50 -> 150 (the quirk: kept from the topounit before), 150 -> 50
        {"name": "uphill", "expr": "uphill_count(t) == merge(0, merge(1, 2, mod(t - 1, 3) == 1), mod(t - 1, 3) == 0)"},
    ])
    assert all(o.passed for o in res), [o.message for o in res]


# -- outputs written on some elements only (b%partial: where x > 0.5)
KEPT = "close(partial(p), merge(x(p), prev%partial(p), x(p) > 0.5))"


def test_a_carried_output_keeps_what_the_previous_step_left(tmp_path):
    res, _ = run(tmp_path, {"steps": 3, "carry": {"b%partial": 0.25}}, property=[
        {"name": "starts from the carry rule", "expr": "input%partial(p) == 0.25"},
        {"name": "written or kept", "expr": KEPT},
        {"name": "finite", "expr": "finite(partial(p))"},
    ])
    assert all(o.passed for o in res), [o.message for o in res]


def test_an_output_not_carried_is_nan_where_a_step_does_not_write_it(tmp_path):
    res, _ = run(tmp_path, {"steps": 3}, property=[
        {"name": "finite", "expr": "implies(x(p) <= 0.5, .not. finite(partial(p)))"},
        {"name": "written", "expr": "implies(x(p) > 0.5, partial(p) == x(p))"},
    ])
    assert all(o.passed for o in res), [o.message for o in res]


def test_carry_is_a_rule_of_the_state_before_step_1(tmp_path):
    s, _ = load(tmp_path, {"steps": 2, "carry": {"b%partial": 0.0}})
    sc = s.scenarios[0] if isinstance(s.scenarios, list) else s.scenarios["s"]
    assert sc.carry == ["b%partial"] and "b%partial" in sc.inputs
    assert sp.dormant_rules(sc, fake.inventory(tmp_path / "case")) == []


def test_child_wsum_skips_inactive_children():
    from spel.proptest.hierarchy import Gridcell, Layout, Soil, Topounit
    from spel.proptest.results import _aggregate

    g = Gridcell(name="dead outlet", topounits=(Topounit((Soil(13),), weight=0.0, elevation=50.0),
                                                Topounit((Soil(13),), weight=0.5, elevation=150.0),
                                                Topounit((Soil(13),), weight=0.5, elevation=300.0)))
    lay = Layout([g], nsets=1)
    store = np.array([np.inf, 2.0, 4.0])  # what ELM's division leaves in the weight-0 topounit
    with np.errstate(all="raise"):
        np.testing.assert_allclose(_aggregate(lay, store, "t", "g", "wsum"), [3.0])
    assert np.isinf(_aggregate(lay, store, "t", "g", "sum")[0])  # the plain sum still sees it


def test_downhill_inactive_names_a_weight_0_neighbour(tmp_path):
    def hill(name, weights):
        return {"name": name, "topounits": [
            {"weight": w, "elevation": e, "landunits": [{"kind": "soil", "patches": [13]}]}
            for w, e in zip(weights, (50.0, 150.0, 300.0))]}

    # lowest first: 300 -> 150 -> 50; the weight-0 topounit is the outlet, the middle, the top
    hills = [hill("dead outlet", (0.0, 0.5, 0.5)), hill("dead middle", (0.5, 0.0, 0.5)),
             hill("dead top", (0.5, 0.5, 0.0))]
    res, _ = run(tmp_path, domain={"gridcell": hills, "nsets": 2}, property=[
        {"name": "downhill_inactive", "over": "t",
         "expr": "downhill_inactive(t) .eqv. (mod(t - 1, 9) == 1 .or. mod(t - 1, 9) == 5)"}])
    assert res[0].passed, res[0].message
