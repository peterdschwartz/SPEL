"""The Python API (api.py), the TOML writer and the session's inspection tools, on the synthetic
case of fake.py (no Fortran)."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from spel.proptest import api as pt
from spel.proptest import checks, rules, runner, tomlw
from spel.proptest import spec as sp
from spel.proptest.api import Gridcell, Lake, Patch, Soil, Topounit, Urban
from spel.proptest.errors import SpecError
from spel.proptest.tests import fake

pytest.importorskip("netCDF4")
EXAMPLES = Path(sp.__file__).parent / "examples"


def _case(tmp_path):
    inv = fake.inventory(tmp_path / "case")
    return pt.Case.of(inv), inv


def _spec(tmp_path, **kw):
    case, inv = _case(tmp_path)
    dom = pt.Domain([Gridcell.of(Soil(1), copies=2, name="trees"), Gridcell.of(Soil(13), copies=2, name="grass")])
    return pt.Spec(case, dom, path=tmp_path / "t.proptest.toml", **kw), inv


def _base(spec):
    return spec.scenario("s", inputs={"x": pt.Range(0.0, 1.0), "prof": pt.Range(0.0, 2.0),
                                      "a%state": pt.Range(-1.0, 1.0)})


# -- the TOML writer


def test_tomlw_round_trips_the_example_and_starter_specs():
    for path in EXAMPLES.glob("*.toml"):
        data = tomllib.loads(path.read_text())
        assert tomllib.loads(tomlw.dumps(data)) == data, path.name


def test_tomlw_layout_and_comments():
    data = {"domain": {"nsets": 2, "gridcell": [{"name": "g", "landunits": [{"kind": "lake"}]}]},
            "scenario": {"s": {"inputs": {"a%x": {"range": [0.0, 1.0]}}}},
            "property": [{"name": "p", "expr": "x(p) > 0"}]}
    text = tomlw.dumps(data, {("scenario", "s", "inputs", "a%x"): "why this range", ("property", 0): "first"})
    assert "[[domain.gridcell]]" in text and '"a%x" = { range = [0.0, 1.0] }' in text
    assert "# why this range\n\"a%x\"" in text and "# first\n[[property]]" in text
    assert tomllib.loads(text) == data


# -- rules


def test_every_rule_kind_has_an_api_class_and_round_trips():
    assert set(pt.RULES) == set(rules.KINDS)
    examples = [pt.Range(0.0, 1.0, clip=(0.0, 0.5)), pt.LogRange(1e-3, 1.0), pt.Values([1, 2, 4]), pt.Value(3.0),
                pt.Scale(0.5, 2.0), pt.Dist("normal", mean=0.0, sigma=1.0),
                pt.Range(0.0, 1.0, pft={13: pt.Value(0.5)}, levels={"1:3": pt.Range(0.2, 0.3)})]
    for r in examples:
        table = r.to_table()
        rules.parse("x", table, "test")  # what the reader accepts
        assert pt.Rule.from_table(table) == r
    assert pt.Value(2.0).to_table() == 2.0  # a constant is a bare value, as in the TOML


def test_every_check_kind_has_a_builder_method():
    assert set(checks.KINDS) <= {"property", "relation", "differential"}
    assert all(hasattr(pt.Spec, k) for k in checks.KINDS)


# -- the builder


def test_builder_writes_what_the_reader_reads(tmp_path):
    spec, inv = _spec(tmp_path, namelist={"use_extra": False})
    base = _base(spec)
    spec.property("linear", "close(y(p), 2.0 * x(p) + input%state(p))", scenarios=[base], comment="the routine")
    spec.relation("copies", "same(follow%y(p), base%y(p))", permute=True)
    text = spec.to_toml(tmp_path / "t.proptest.toml")
    loaded, _ = sp.load(tmp_path / "t.proptest.toml", inv=inv)
    built = spec.validate()
    assert [c.name for c in loaded.checks] == [c.name for c in built.checks] == ["linear", "copies"]
    assert loaded.scenarios["s"].inputs == built.scenarios["s"].inputs
    assert loaded.domain.gridcells == built.domain.gridcells
    assert '"a%x" = { range = [0.0, 1.0] }' in text and "# the routine" in text  # canonical names
    again = pt.Spec.from_toml(tmp_path / "t.proptest.toml", inv=inv)
    assert tomllib.loads(again.to_toml()) == tomllib.loads(text)


def test_builder_refuses_at_once_with_the_readers_message(tmp_path):
    spec, _ = _spec(tmp_path)
    with pytest.raises(SpecError, match="not a variable of the case"):
        spec.scenario("bad", inputs={"nosuch": pt.Value(1.0)})
    base = _base(spec)
    with pytest.raises(SpecError, match="unknown name"):
        spec.property("typo", "y(p) > xx(p)")
    with pytest.raises(SpecError, match="exists"):
        spec.scenario("s")
    base.input("x", pt.Range(0.0, 1.0, pft={1: pt.Value(0.5)}))  # a patch variable takes pft overrides
    with pytest.raises(SpecError, match="exists"):
        spec.property("linear", "y(p) > 0.0") or spec.property("linear", "y(p) > 0.0")


def test_domain_view_queries(tmp_path):
    dom = pt.Domain([
        Gridcell.of(Soil(Patch(1, 0.6), 13, weight=0.7), Urban("md", weight=0.2), Lake(), name="mixed"),
        Gridcell(name="hill", topounits=(Topounit((Soil(1),), weight=1.0), Topounit((Lake(),), weight=0.0))),
    ], nsets=1)
    view = dom.view()
    assert view.sizes == {"g": 2, "t": 3, "l": 5, "c": 9, "p": 10}
    mixed = view.gridcells[0]
    assert [lu.kind.value for lu in mixed.landunits] == ["soil", "urban", "lake"]
    assert [p.pft for p in mixed.columns[0].patches] == [1, 13] and mixed.columns[0].patches[0].weight == 0.6
    assert [c.role for c in mixed.columns[1:6]] == ["roof", "sunwall", "shadewall", "road_imperv", "road_perv"]
    hill = view.gridcells[1]
    assert [(t.number, t.active) for t in hill.topounits] == [(1, True), (2, False)]
    assert view.select("p", name="mixed", kind="soil", pft=13).tolist() == [2]
    assert view.select("c", role="road_perv").tolist() == [6]  # soil column 1, urban columns 2-6
    f = view.expected_filters()
    assert f["urbanl"].tolist() == [2] and f["lakec"].tolist() == [7]  # the hill's lake (9) is inactive
    assert f["hydrologyc"].tolist() == [1, 6, 8]
    table = view.table("p")
    assert table.loc[2, "pft"] == 13 and np.isclose(table.loc[1, "wtgcell"], 0.7 * 0.6)


# -- sessions: sampling, results, verdicts (fake backend)


def test_sample_execute_frame_and_verdict(tmp_path):
    spec, inv = _spec(tmp_path)
    base = _base(spec)
    spec.property("linear", "close(y(p), 2.0 * x(p) + input%state(p))")
    spec.property("small", "y(p) < 0.5")
    with spec.session(backend=fake.FakeBackend()) as s:
        ex = s.sample(base.name, seed=7)
        again = s.sample(base.name, seed=7)  # a seed gives the same example
        assert again.seed == ex.seed and again.modes == ex.modes
        assert all(np.array_equal(again.inputs[k], v) for k, v in ex.inputs.items())
        assert set(ex.modes) == {"a%x", "a%prof", "a%state"} and ex.seed >= 0
        res = s.execute(base.name, ex)
        df = res.frame("p", names=["a%x", "b%y", "a%prof"])
        assert list(df.index) == list(range(1, 9)) and {"name", "pft", "a%x", "b%y", "a%prof[1]"} <= set(df.columns)
        assert np.allclose(df["b%y"], 2.0 * df["a%x"] + ex.inputs["a%state"])
        good = res.verdict("linear")
        assert good.passed and good.ok.all() and "b%y" in good.frame().columns
        bad = res.verdict("small")
        assert not bad.passed and len(bad.failing) == int((df["b%y"] >= 0.5).sum())
        g = res.frame("g", aggregate={"b%y": "sum"})
        assert np.allclose(g["sum(b%y)"], df["b%y"].to_numpy())  # one patch per gridcell here


def test_multi_step_scenario_round_trips_and_runs(tmp_path):
    spec, inv = _spec(tmp_path)
    spec.scenario("season", inputs={"x": pt.Range(0.0, 1.0), "prof": pt.Range(0.0, 2.0),
                                    "a%state": pt.Range(-1.0, 1.0)},
                  steps=3, forcing={"a%x": pt.Range(0.0, 1.0), "prof": "0.5 * step"},
                  between={"a%state": "0.5 * state(p)"}, keep=["b%*"])
    spec.property("halved", "close(state(p), 0.5 * (prev%state(p) + 1.0))", steps="all")
    spec.property("last only", "close(tend(state(p)), 0.5 * (1.0 - prev%state(p)))", steps="last")
    text = spec.to_toml(tmp_path / "t.proptest.toml")
    data = tomllib.loads(text)
    assert data["scenario"]["season"]["steps"] == 3 and data["scenario"]["season"]["keep"] == ["b%*"]
    assert data["scenario"]["season"]["forcing"]["a%prof"] == "0.5 * step"
    assert data["property"][1]["steps"] == "last"
    again = pt.Spec.from_toml(tmp_path / "t.proptest.toml", inv=inv)
    assert tomllib.loads(again.to_toml()) == data
    parsed = again.validate()
    assert parsed.scenarios["season"].steps == 3 and [c.steps for c in parsed.checks] == ["all", "last"]
    with runner.Session(parsed, inv, backend=fake.FakeBackend()) as s:
        res = runner.run(parsed, inv, scenarios=["season"], settings={"seed": 0, "max_examples": 5}, session=s,
                         out=__import__("io").StringIO())
    assert all(o.passed for o in res), [o.message for o in res]
