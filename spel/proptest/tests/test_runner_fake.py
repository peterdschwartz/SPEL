"""The runner on a synthetic case (tests/fake.py): no Fortran, no build."""

import io
from pathlib import Path

import numpy as np
import pytest

from spel.proptest import runner
from spel.proptest import spec as sp
from spel.proptest import strategies as strat
from spel.proptest.tests import fake

pytest.importorskip("hypothesis")
pytest.importorskip("netCDF4")


def load(tmp_path, bug="", inv=None, **tables):
    case = tmp_path / "case"
    inv = inv or fake.inventory(case)
    s = sp.parse(fake.spec_data(case, **tables), tmp_path / "t.proptest.toml", case, inv)
    return s, inv, fake.FakeBackend(bug)


def run(tmp_path, bug="", **tables):
    s, inv, backend = load(tmp_path, bug, **tables)
    out = io.StringIO()
    with runner.Session(s, inv, backend=backend) as session:
        res = runner.run(s, inv, settings={"seed": 0}, out=out, session=session)
    assert backend.stopped
    return res, out.getvalue()


def test_property_passes_and_fails_with_a_report(tmp_path):
    res, _ = run(tmp_path, property=[{"name": "linear", "expr": "close(y(p), 2.0 * x(p) + input%state(p))"},
                                 {"name": "stepped", "expr": "state(p) == input%state(p) + 1.0"}])
    assert [o.passed for o in res] == [True, True] and res[0].kind == "property"
    res, text = run(tmp_path, property=[{"name": "small", "expr": "y(p) < 2.5"}])
    assert not res[0].passed
    assert "fails at" in res[0].message and "a%x = " in res[0].message and "y = " in res[0].message
    assert "saved:" in text and (tmp_path / "proptest-data" / "case" / "failures" / "s__small" / "example.json").is_file()


def test_inout_is_the_input_before_and_the_result_after(tmp_path):
    # `state` in a check is what the routine left; derive and assume see the drawn value
    res, _ = run(tmp_path, scenario={"s": {
        "inputs": {"a%x": {"range": [0.0, 1.0]}, "a%prof": 1.0},
        "derive": {"a%state": "3.0 * x(p)"},
        "assume": ["x(p) > 0.25"],
    }}, property=[{"name": "derived", "expr": "close(state(p), 3.0 * x(p) + 1.0) .and. x(p) > 0.25"},
                  {"name": "layers", "expr": "ysum(p) == nlev"}])
    assert [o.passed for o in res] == [True, True]


def test_relation_halves_and_follow(tmp_path):
    rel = {"name": "shift", "follow": {"a%x": "x(p) + 1.0"}, "expect": "close(follow%y(p), base%y(p) + 2.0, atol=1.0e-12)"}
    res, _ = run(tmp_path, relation=[rel])
    assert res[0].passed and res[0].kind == "relation"
    res, _ = run(tmp_path, relation=[dict(rel, expect="follow%y(p) == base%y(p)")])
    assert not res[0].passed and "follow%y" in res[0].message


def test_copies_relation_needs_no_follow_and_permute_finds_neighbour_dependence(tmp_path):
    copies = {"name": "copies", "expect": "same(follow%y(p), base%y(p)) .and. same(follow%never(p), base%never(p))"}
    assert run(tmp_path, relation=[copies])[0][0].passed
    # a value leaking from the neighbouring element: identical halves hide it, a reordering shows it
    assert run(tmp_path, "neighbour", relation=[copies])[0][0].passed
    res, _ = run(tmp_path, "neighbour", relation=[dict(copies, permute=True)])
    assert not res[0].passed
    assert run(tmp_path, relation=[dict(copies, permute=True)])[0][0].passed


def test_urban_sites_in_relations_and_reports(tmp_path):
    # urban sites have five columns: halves, permutations (whole sites) and reports per column
    urban = {"pfts": [1], "urban": ["md", "hd"], "per_pft": 2, "nsets": 2}
    copies = {"name": "copies", "expect": "same(follow%y(p), base%y(p))"}
    assert run(tmp_path, domain=urban, relation=[dict(copies, permute=True)])[0][0].passed
    res, _ = run(tmp_path, "neighbour", domain=urban, relation=[dict(copies, permute=True)])
    assert not res[0].passed
    res, _ = run(tmp_path, domain=urban, property=[{"name": "small", "expr": "y(p) < 2.0"}])
    assert not res[0].passed and "set " in res[0].message


def test_finer_element_variable_is_unbound_for_a_site_with_columns(tmp_path):
    s, inv, backend = load(tmp_path, domain={"pfts": [], "urban": ["md"], "nsets": 2})
    with runner.Session(s, inv, backend=backend) as session:
        loc = session.locals_for(np.array([1, 2]), "l")
        assert "c" not in loc and "p" not in loc and loc["g"].tolist() == [1, 2]
        loc = session.locals_for(np.array([3, 7]), "p")
        assert loc["c"].tolist() == [3, 7] and loc["l"].tolist() == [1, 2]
        assert session.rows_of("l", 2, "p").tolist() == [5, 6, 7, 8, 9]


def test_relation_needs_an_even_number_of_sets(tmp_path):
    with pytest.raises(sp.SpecError, match="nsets must be even"):
        load(tmp_path, domain={"pfts": [1], "nsets": 3}, relation=[{"name": "c", "expect": "same(follow%y(p), base%y(p))"}])


def test_abort_is_a_failure_that_is_saved_and_replayed(tmp_path):
    s, inv, backend = load(tmp_path, "abort", property=[{"name": "finite", "expr": "finite(y(p))"}])
    with runner.Session(s, inv, backend=backend) as session:
        res = runner.run(s, inv, settings={"seed": 0}, out=io.StringIO(), session=session)
        assert not res[0].passed and "the routine aborted" in res[0].message and "ERROR STOP" in res[0].message
        meta, ex = session.load_failure(res[0].failure)
        assert meta["check"] == "finite" and float(np.max(ex.inputs["a%x"])) > 0.9
        with pytest.raises(runner.PropertyFailure, match="aborted"):
            session.judge(s.scenarios["s"], s.checks[0], ex, hypothesis=False)


def test_differential_compares_two_backends(tmp_path):
    tables = dict(compare={"build": {"dbg": True}}, differential=[{"name": "same answers"}])
    s, inv, backend = load(tmp_path, **tables)
    assert s.checks[0].all_outputs  # "all": every output both have, under the configuration of the run
    with runner.Session(s, inv, backend=backend, other=runner.Session(s, inv, backend=fake.FakeBackend())) as session:
        text = s.checks[0].expr_for(session, s.scenarios["s"]).text
    assert "b%y" in text and "a%state" in text and "b%extra" not in text  # extra: dead while use_extra is off

    def outcome(bug, **check):
        s, inv, backend = load(tmp_path, compare={"build": {"dbg": True}},
                               differential=[{"name": "same answers", **check}])
        other = runner.Session(s, inv, backend=fake.FakeBackend(bug))
        with runner.Session(s, inv, backend=backend, other=other) as session:
            return runner.run(s, inv, settings={"seed": 0}, out=io.StringIO(), session=session)[0]

    assert outcome("").passed
    o = outcome("offset")
    assert not o.passed and o.kind == "differential" and "dev%b%y" in o.message
    assert outcome("offset", outputs=["b%y"], atol=1.0e-5).passed
    assert outcome("offset", expect="abs(dev%y(p) - ref%y(p)) < 1.0e-5").passed


def test_differential_needs_a_compare_table(tmp_path):
    with pytest.raises(sp.SpecError, match=r"no \[compare\] table"):
        load(tmp_path, differential=[{"name": "d"}])


def test_probe_lists_what_the_routine_did(tmp_path):
    s, inv, backend = load(tmp_path, property=[{"name": "p", "expr": "finite(y(p))"}])
    with runner.Session(s, inv, backend=backend) as session:
        res = session.probe(s.scenarios["s"], n=3)
    assert res.never_written == ["b%never"] and res.elements == 8 and res.assume_pass == 1.0
    assert res.unchanged == [] and "never written on the checked elements: 1" in res.text()


def test_levels_override_and_uniform_columns(tmp_path):
    s, inv, backend = load(tmp_path, scenario={"s": {"inputs": {
        "a%x": 0.5, "a%state": 0.0,
        "a%prof": {"range": [0.0, 1.0], "levels": {"1": 10.0, "3:nlev": {"range": [20.0, 30.0]}}},
    }}}, property=[{"name": "p", "expr": "finite(y(p))"}])
    with runner.Session(s, inv, backend=backend) as session:
        prof = session.inputs_at(s.scenarios["s"], np.random.default_rng(0).random).inputs["a%prof"]
    assert (prof[:, 0] == 10.0).all() and ((prof[:, 1] >= 0.0) & (prof[:, 1] <= 1.0)).all()
    assert ((prof[:, 2:] >= 20.0) & (prof[:, 2:] <= 30.0)).all()
    with pytest.raises(sp.SpecError, match="levels"):
        load(tmp_path, scenario={"s": {"inputs": {"a%x": {"range": [0.0, 1.0], "levels": {"1": 0.0}},
                                                  "a%prof": 1.0, "a%state": 0.0}}})

    class Draw:  # the mode asked for, then u = 0.25 for whatever else is drawn
        def __init__(self, mode):
            self.mode = mode

        def __call__(self, strategy, label=""):
            return self.mode if label.startswith("mode") else 0.25

    rule = s.scenarios["s"].inputs["a%prof"]
    u, mode = strat.draw_u(Draw(3), "a%prof", rule, (8, 4), np.random.default_rng(1), axis=0)
    assert mode == 3 and (u == u[:, :1]).all() and len(np.unique(u[:, 0])) == 8  # one value down each column


def test_derive_fills_layers_per_element_when_counts_coincide(tmp_path):
    # 4 elements and 4 levels: only the expression says which axis a vector is along
    s, inv, backend = load(tmp_path, domain={"pfts": [1], "per_pft": 2, "nsets": 2}, scenario={"s": {
        "inputs": {"a%x": {"range": [0.0, 1.0]}, "a%state": 0.0},
        "derive": {"a%prof": "cumsum(x(p) + 0.0 * prof(p, :))"},
    }}, property=[{"name": "p", "expr": "finite(y(p))"}])
    with runner.Session(s, inv, backend=backend) as session:
        ex = session.inputs_at(s.scenarios["s"], np.random.default_rng(0).random)
    x = ex.inputs["a%x"]
    np.testing.assert_allclose(ex.inputs["a%prof"], x[:, None] * np.arange(1, 5)[None, :])


def test_spec_functions_from_a_module(tmp_path):
    (tmp_path / "helpers.py").write_text(
        "from spel.proptest.functions import register\n\n@register\ndef twice(a):\n    return 2.0 * a\n")
    res, _ = run(tmp_path, functions={"modules": ["helpers.py"]},
                 property=[{"name": "f", "expr": "close(y(p), twice(x(p)) + input%state(p))"}])
    assert res[0].passed
    (tmp_path / "empty.py").write_text("x = 1\n")
    with pytest.raises(sp.SpecError, match="registers no function"):
        load(tmp_path, functions={"modules": ["empty.py"]})
    with pytest.raises(sp.SpecError, match="unknown name 'thrice'"):
        load(tmp_path, property=[{"name": "f", "expr": "thrice(x(p)) > 0.0"}])


def test_unknown_name_inside_a_namespace_stops_loading(tmp_path):
    with pytest.raises(sp.SpecError, match="not a variable"):
        load(tmp_path, relation=[{"name": "r", "expect": "follow%nosuch(p) == base%y(p)"}])


def test_filled_parameter_refuses_to_run(tmp_path):
    case = tmp_path / "case"
    inv = fake.inventory(case, [fake.var("pftcon%slatop", "0:78", "parameter")])
    s = sp.parse(fake.spec_data(case, property=[{"name": "p", "expr": "finite(y(p))"}]), tmp_path / "t.toml", case, inv)
    inv.parameters["x78"] = 78
    with runner.Session(s, inv, backend=fake.FakeBackend()) as session:
        assert "pftcon%slatop" in session.domain.filled
        with pytest.raises(sp.SpecError, match="pftcon%slatop .no reference capture"):
            session.preflight(s.scenarios["s"])


def test_init_without_a_reference_leaves_todo_lines(tmp_path):
    from spel.proptest import scaffold

    case = tmp_path / "case"
    inv = fake.inventory(case)
    path, todo = scaffold.write(case, inv, tmp_path / "new.proptest.toml")
    text = path.read_text()
    assert '# TODO "a%x" = { range = [?, ?] }' in text and "finite(b%y(p))" in text
    assert "same(follow%b%y(p), base%b%y(p))" in text
    assert any("3 inputs have no rule" in line for line in todo) and any("no reference capture" in line for line in todo)
    s, _ = sp.load(path, inv=inv)  # the starter loads; a run refuses until the TODO lines are rules
    scen = s.scenarios["default"]
    assert sp.missing_rules(scen, inv.configured(sp.constants(s, inv, scen)), {}) == ["a%prof", "a%state", "a%x"]
    assert "[namelist]" in text and "extra" not in text and "boost" not in text  # dead while use_extra is off
    with pytest.raises(sp.SpecError, match="exists"):
        scaffold.write(case, inv, path)


def _bindings(*actuals):
    from types import SimpleNamespace

    return SimpleNamespace(driver_bindings=[SimpleNamespace(callee="m::r", actual=a) for a in actuals])


def test_init_takes_the_site_kinds_from_the_routine_filters(tmp_path):
    from spel.proptest import scaffold
    from spel.proptest.domain import SiteKind

    inv = fake.inventory(tmp_path / "case")
    cases = [
        (("filter%num_urbanl", "filter%urbanl", "filter%nourbanl", "filter%urbanp", "bounds_clump"), {SiteKind.URBAN}),
        (("filter%lakec", "filter%num_lakep"), {SiteKind.LAKE}),
        (("filter%urbanc", "filter%nolakec"), {SiteKind.URBAN, SiteKind.SOIL, SiteKind.CROP}),
        (("filter%nourbanl",), {SiteKind.SOIL, SiteKind.CROP}),  # a complement alone: the other sites
        (("bounds_clump",), {SiteKind.SOIL, SiteKind.CROP}),
    ]
    for actuals, kinds in cases:
        inv.unit_test = _bindings(*actuals)
        assert scaffold.routine_sites(inv) == kinds, actuals
    inv.unit_test = _bindings("filter%urbanl", "filter%urbanp")
    path, _ = scaffold.write(tmp_path / "case", inv, tmp_path / "nofilter.proptest.toml")
    assert 'over = "p"' in path.read_text()  # the case has no urbanp filter: over every patch
    inv.filters.update({"urbanp": 1, "num_urbanp": 0})
    path, todo = scaffold.write(tmp_path / "case", inv, tmp_path / "u.proptest.toml")
    text = path.read_text()
    assert 'urban = ["hd", "md"]' in text and "pfts" not in text.split("[namelist]")[0]
    assert 'over = "p in urbanp"' in text and any("densities" in line for line in todo)
    s, _ = sp.load(path, inv=inv)
    assert [d.value for d in s.domain.urban] == ["hd", "md"] and s.domain.pfts == []
    inv.unit_test = _bindings("filter%lakep")
    path, _ = scaffold.write(tmp_path / "case", inv, tmp_path / "l.proptest.toml", sites={SiteKind.SOIL, SiteKind.LAKE})
    text = path.read_text()
    assert "lake = true" in text and "pfts = [1]" in text and 'over = "p"' in text and "# sites: lake, soil (--sites)" in text


def test_stats_sites_select_patches_by_their_landunit(tmp_path):
    import netCDF4

    from spel.proptest import stats
    from spel.proptest.domain import SiteKind

    path = tmp_path / "ref.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 2)
        ds.createDimension("patch", 4)
        ds.createDimension("landunit", 3)
        ds.createVariable("veg_pp__landunit", "i4", ("time", "patch"))[:] = [[1, 2, 2, 3]] * 2
        ds.createVariable("lun_pp__itype", "i4", ("time", "landunit"))[:] = [[1, 8, 5]] * 2  # soil, urban hd, lake
    ref = stats.Reference(path)
    try:
        steps = np.arange(2)
        assert stats.site_mask(ref, steps, {SiteKind.URBAN})[0].tolist() == [False, True, True, False]
        assert stats.site_mask(ref, steps, {SiteKind.SOIL, SiteKind.LAKE})[1].tolist() == [True, False, False, True]
    finally:
        ref.close()


def test_site_choices_of_the_cli_are_the_site_kinds():
    from spel.proptest.cli import SITE_CHOICES
    from spel.proptest.domain import SiteKind

    assert set(SITE_CHOICES) == {k.value for k in SiteKind}


def test_active_dir_reads_the_saved_subspace(tmp_path):
    import json

    s, inv, backend = load(tmp_path, property=[{"name": "along", "expr": "abs(active_dir(p, 'y(p)') - (2.0 * x(p) - 1.0)) < 1.0e-12"}])
    out = tmp_path / "proptest-data" / "case" / "subspace"
    out.mkdir(parents=True)
    (out / "s.y_p.json").write_text(json.dumps({"inputs": ["a%x"], "eigenvectors": [[1.0]]}))
    with runner.Session(s, inv, backend=backend) as session:
        res = runner.run(s, inv, settings={"seed": 0}, out=io.StringIO(), session=session)
    assert res[0].passed, res[0].message


# -- namelist options


def test_variables_dead_under_the_configuration_do_not_show(tmp_path):
    made = []
    s, inv, _ = load(tmp_path, property=[{"name": "p", "expr": "finite(y(p))"}])
    scen = s.scenarios["s"]
    with runner.Session(s, inv, backend=fake.backends(made=made)) as session:
        assert session.inv.vars["a%boost"].kind == "unused" and session.inv.vars["b%extra"].kind == "unused"
        assert session.inv.vars["b%extra"].dead_under == ("use_extra",)
        assert sp.missing_rules(scen, session.inv, session.domain.constants) == []  # boost needs no rule
        assert "b%extra" not in session.live_outputs()
        assert session.probe(scen, n=2).never_written == ["b%never"]
    assert inv.vars["a%boost"].kind == "input"  # without a configuration every branch counts


def test_a_rule_for_a_dead_input_is_dormant(tmp_path):
    made = []
    inputs = {"a%x": {"range": [0.0, 1.0]}, "a%prof": 1.0, "a%state": 0.0, "a%boost": {"range": [1.0, 2.0]}}
    s, inv, _ = load(tmp_path, scenario={"s": {"inputs": inputs}}, property=[{"name": "p", "expr": "finite(y(p))"}])
    with runner.Session(s, inv, backend=fake.backends(made=made)) as session:
        assert sp.dormant_rules(s.scenarios["s"], session.inv) == ["a%boost"]
        res = runner.run(s, inv, settings={"seed": 0}, out=io.StringIO(), session=session)
    assert res[0].passed and "a%boost" not in made[0].seen and "a%x" in made[0].seen


def test_naming_a_dead_variable_is_a_load_error_that_names_the_option(tmp_path):
    with pytest.raises(sp.SpecError, match=r"extra \(b%extra\) has no value in scenario s.*use_extra = \.false\."):
        load(tmp_path, property=[{"name": "p", "expr": "finite(extra(p))"}])
    with pytest.raises(sp.SpecError, match="boost .* has no value"):
        load(tmp_path, scenario={"s": {"inputs": {"a%x": 0.5, "a%prof": 1.0}, "derive": {"a%state": "boost(p)"}}},
             property=[{"name": "p", "expr": "finite(y(p))"}])
    # live in the scenario that turns the option on
    load(tmp_path, namelist={"use_extra": True}, property=[{"name": "p", "expr": "finite(extra(p))"}])


def test_namelist_keys_are_options_of_the_case(tmp_path):
    with pytest.raises(sp.SpecError, match="use_extar is not an option of the case .*use_extra"):
        load(tmp_path, namelist={"use_extar": True})
    with pytest.raises(sp.SpecError, match="use_other is not an option"):
        load(tmp_path, relation=[{"name": "r", "namelist": {"use_other": 1}, "expect": "same(follow%y(p), base%y(p))"}])
    with pytest.raises(sp.SpecError, match="cannot be combined with `namelist`"):
        load(tmp_path, relation=[{"name": "r", "namelist": {"use_extra": True}, "permute": True,
                                  "expect": "same(follow%y(p), base%y(p))"}])


def test_scenario_under_another_option_needs_its_inputs(tmp_path):
    on = {"namelist": {"use_extra": True}}
    base = {"a%x": {"range": [0.0, 1.0]}, "a%prof": 1.0, "a%state": {"range": [-1.0, 1.0]}}
    checks = [{"name": "boosted", "scenarios": ["on"],
               "expr": "close(y(p), 2.0 * x(p) + input%state(p) + boost(p)) .and. extra(p) == 2.0 * boost(p)"},
              {"name": "plain", "scenarios": ["off"], "expr": "close(y(p), 2.0 * x(p) + input%state(p))"}]
    scenarios = {"off": {"inputs": base}, "on": {"inputs": base, **on}}
    made = []
    s, inv, _ = load(tmp_path, scenario=scenarios, property=checks)
    with runner.Session(s, inv, backend=fake.backends(made=made)) as session:
        with pytest.raises(sp.SpecError, match="inputs without a rule.*a%boost"):
            session.for_scenario(s.scenarios["on"]).preflight(s.scenarios["on"])
    scenarios["on"]["inputs"] = {**base, "a%boost": {"range": [1.0, 2.0]}}
    s, inv, _ = load(tmp_path, scenario=scenarios, property=checks)
    made = []
    with runner.Session(s, inv, backend=fake.backends(made=made)) as session:
        res = runner.run(s, inv, settings={"seed": 0}, out=io.StringIO(), session=session)
    assert [(o.scenario, o.passed) for o in res] == [("off", True), ("on", True)]
    assert [b.extra for b in made] == [False, True] and all(b.stopped for b in made)  # one process per configuration


def test_relation_under_other_options(tmp_path):
    inputs = {"a%x": {"range": [0.0, 1.0]}, "a%prof": 1.0, "a%state": {"range": [-1.0, 1.0]},
              "a%boost": {"range": [1.0, 2.0]}}
    rel = {"name": "the option adds the boost", "namelist": {"use_extra": True},
           "expect": "close(follow%y(p), base%y(p) + follow%boost(p), atol=1.0e-12) .and. finite(follow%extra(p))"}
    tables = dict(domain={"pfts": [1, 13], "per_pft": 2, "nsets": 1}, scenario={"s": {"inputs": inputs}})  # one set is enough
    made = []
    s, inv, _ = load(tmp_path, relation=[rel, dict(rel, name="with a transform", follow={"a%x": "x(p) + 1.0"},
                                                 expect="close(follow%y(p), base%y(p) + 2.0 + follow%boost(p), atol=1.0e-12)")],
                     **tables)
    with runner.Session(s, inv, backend=fake.backends(made=made)) as session:
        res = runner.run(s, inv, settings={"seed": 0}, out=io.StringIO(), session=session)
    assert [o.passed for o in res] == [True, True], res[0].message + res[1].message
    assert [b.extra for b in made] == [False, True]
    assert "a%boost" not in made[0].seen and "a%boost" in made[1].seen  # each side gets the inputs it reads

    # `follow%` is checked in the follow-up's configuration, `base%` in the scenario's
    with pytest.raises(sp.SpecError, match="base%extra .* has no value"):
        load(tmp_path, relation=[dict(rel, expect="same(follow%extra(p), base%extra(p))")], **tables)

    # a failure keeps both runs and replays
    s, inv, _ = load(tmp_path, relation=[dict(rel, name="no effect", expect="same(follow%y(p), base%y(p))")], **tables)
    with runner.Session(s, inv, backend=fake.backends()) as session:
        res = runner.run(s, inv, settings={"seed": 0}, out=io.StringIO(), session=session)
        assert not res[0].passed and "follow%y" in res[0].message
        assert (Path(res[0].failure) / "follow" / "spel-inputs0001.nc").is_file()
        meta, ex = session.load_failure(res[0].failure)
        assert meta["extra"]["namelist"] == {"use_extra": True} and ex.follow is not None
        with pytest.raises(runner.PropertyFailure, match="no effect"):
            session.judge(s.scenarios["s"], s.checks[0], ex, hypothesis=False)


def test_status_follows_the_configuration():
    from spel.proptest.inventory import TracedAccess, status_of
    from spel.scripts.fortran_parser.boolen_expression import AllOf, Expectation

    init = TracedAccess("w", False, (AllOf((Expectation("use_cn", "True"), Expectation("use_c13", "True"))),))
    use = TracedAccess("r", False, ())
    assert status_of([init, use], None, {}) == ("w", ())  # undecided: every branch counts
    assert status_of([init, use], None, {"use_cn": 1, "use_c13": 1}) == ("w", ())
    assert status_of([init, use], None, {"use_cn": 1, "use_c13": 0}) == ("r", ())  # the write is off: an input
    assert status_of([init], None, {"use_cn": 0}) == (None, ("use_c13", "use_cn"))
    assert status_of(None, "rw", {"use_cn": 0}) == ("rw", ())  # not traced: the manifest's status
