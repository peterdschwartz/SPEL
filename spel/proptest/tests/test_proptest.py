"""Snippet tests of the proptest pieces that need no case build (evaluator, spec, rules, analysis maths)."""

from pathlib import Path

import numpy as np
import pytest

from spel.proptest import functions as fn
from spel.proptest import spec as sp
from spel.proptest import strategies as strat
from spel.proptest.hierarchy import Gridcell, Layout, Soil
from spel.proptest.fexpr import Evaluator, ExprError, FArray, parse_expr
from spel.proptest.inventory import Inventory, VarInfo, parse_dims
from spel.proptest.runner import _selected
from spel.proptest.stats import VarStats, suggest
from spel.proptest.subspace import _eig, _gap_rank


def _var(name, bounds, kind, ftype="real"):
    inst, _, comp = name.partition("%")
    rank = len(bounds.split(",")) if bounds else 0
    return VarInfo(
        name=name,
        instance=inst,
        field=comp,
        type=ftype,
        rank=rank,
        dims=parse_dims(bounds, rank, name),
        status="r",
        kind=kind,
    )


@pytest.fixture
def inv(tmp_path):
    vs = [
        _var("top_as%tbot", "begt:endt", "input"),
        _var("top_as%pbot", "begt:endt", "input"),
        _var("top_as%rhobot", "begt:endt", "input"),
        _var("top_as%thbot", "begt:endt", "input"),
        _var("veg_es%t_veg", "begp:endp", "inout"),
        _var("veg_ef%eflx_sh_veg", "begp:endp", "output"),
        _var("veg_pp%itype", "begp:endp", "topology", "integer"),
    ]
    return Inventory(
        case_dir=tmp_path,
        routines=["m::r"],
        vars={v.name: v for v in vs},
        constants={},
        parameters={"rair": 287.0},
        aliases={"t_veg": "veg_es%t_veg"},
        filters={"nolakeurbanp": 1},
        instances={},
    )


def _spec(inv, **scenario):
    data = {
        "case": {"dir": str(inv.case_dir)},
        "domain": {"gridcell": [{"landunits": [{"kind": "soil", "patches": [1]}]}]},
        "scenario": {
            "s": {"inputs": {"top_as%tbot": {"range": [280.0, 300.0]}}, **scenario}
        },
        "property": [{"name": "warm", "expr": "t_veg(p) > 0.0"}],
    }
    return data


# -- expressions


def test_eval_elementwise_with_bounds_and_intrinsics():
    x = FArray(np.array([1.0, -2.0, 3.0]), (5,), "x")
    ev = Evaluator(lambda path: {"x": x}[path[0]], fn.FUNCTIONS)
    out = ev(
        parse_expr("abs(x(p)) * 2.0 + max(x(p), 0.5)"),
        local={"p": np.array([5, 6, 7])},
        n_elem=3,
    )
    np.testing.assert_allclose(out, [3.0, 4.5, 9.0])
    ok = ev(
        parse_expr("between(x(p), -3.0, 2.0) .and. .not. finite(x(p) * 1.0e36)"),
        local={"p": np.array([5, 6, 7])},
    )
    assert ok.tolist() == [True, True, False]


def test_eval_multiline_and_out_of_bounds():
    x = FArray(np.array([1.0, 2.0]), (1,), "x")
    ev = Evaluator(lambda path: {"x": x}[path[0]], fn.FUNCTIONS)
    assert ev(
        parse_expr("x(p) > 0.0\n  .and. x(p) < 3.0"), local={"p": np.array([1, 2])}
    ).all()
    with pytest.raises(ExprError, match="out of bounds"):
        ev(parse_expr("x(p)"), local={"p": np.array([3])})


def test_parse_rejects_assignment():
    with pytest.raises(ExprError):
        parse_expr("a = b")


# -- spec validation


def test_spec_unknown_name(inv):
    data = _spec(inv)
    data["property"][0]["expr"] = "t_leaf(p) > 0.0"
    with pytest.raises(sp.SpecError, match="unknown name 't_leaf'"):
        sp.parse(data, inv.case_dir / "x.toml", inv.case_dir, inv)


def test_spec_rule_on_output_rejected(inv):
    data = _spec(inv)
    data["scenario"]["s"]["inputs"]["veg_ef%eflx_sh_veg"] = 1.0
    with pytest.raises(sp.SpecError, match="output, not an input"):
        sp.parse(data, inv.case_dir / "x.toml", inv.case_dir, inv)


def test_spec_missing_rules_and_derive_order(inv):
    data = _spec(
        inv,
        derive={
            "top_as%rhobot": "top_as%pbot(t) / (rair * top_as%thbot(t))",  # reads thbot, derived below
            "top_as%thbot": "top_as%tbot(t)",
        },
    )
    s = sp.parse(data, inv.case_dir / "x.toml", inv.case_dir, inv)
    scen = s.scenarios["s"]
    assert [v for v, _ in scen.derive] == ["top_as%thbot", "top_as%rhobot"]
    assert sp.missing_rules(scen, inv, {}) == ["top_as%pbot", "veg_es%t_veg"]


def test_spec_derive_cycle(inv):
    data = _spec(
        inv,
        derive={"top_as%pbot": "top_as%rhobot(t)", "top_as%rhobot": "top_as%pbot(t)"},
    )
    with pytest.raises(sp.SpecError, match="circular"):
        sp.parse(data, inv.case_dir / "x.toml", inv.case_dir, inv)


# -- rules


def test_rule_quantiles_and_pft_override():
    rule = sp._rule("a%x", {"range": [0.0, 10.0], "pft": {"4": {"value": -1.0}}}, "t")
    u = np.array([0.0, 0.5, 1.0])
    pft = np.array([1, 4, 1])
    np.testing.assert_allclose(
        strat.values_of(rule, u, pft, 0, "real"), [0.0, -1.0, 10.0]
    )
    lr = sp._rule("a%y", {"log_range": [1.0, 100.0]}, "t")
    np.testing.assert_allclose(strat.quantile(lr)(np.array([0.5])), [10.0])
    vals = sp._rule("a%z", {"values": [1, 2, 3]}, "t")
    assert strat.values_of(
        vals, np.array([0.0, 0.5, 1.0]), None, None, "integer"
    ).tolist() == [1, 2, 3]


def test_rule_validation():
    with pytest.raises(sp.SpecError):
        sp._rule("a%x", {"range": [2.0, 1.0]}, "t")
    with pytest.raises(sp.SpecError):
        sp._rule("a%x", {"dist": "lognormal", "mean": 0.0}, "t")


def test_stats_suggest():
    v = _var("a%x", "begp:endp", "input")
    s = VarStats(
        "a%x", "input", "", count=10, q={0.0: 0.001, 0.01: 0.001, 0.99: 50.0, 1.0: 60.0}
    )
    assert "log_range" in suggest(v, s)
    s = VarStats(
        "a%x", "input", "", count=10, q={0.0: 5.0, 0.01: 5.0, 0.99: 5.0, 1.0: 5.0}
    )
    assert suggest(v, s) == {"value": 5.0}


# -- domain, selection, subspace


def test_layout_sets_and_pfts():
    lay = Layout([Gridcell.of(Soil(1), copies=2), Gridcell.of(Soil(13), copies=2)], nsets=2)
    assert lay.sizes() == {lev: 8 for lev in "gtlcp"}
    assert lay.set_of("p").tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert lay.pft_of("p").tolist() == [1, 1, 13, 13, 1, 1, 13, 13]


def test_selected_names_with_brackets():
    name = "soil water stress in [0, 1]"
    assert _selected(name, [name])
    assert _selected(name, ["soil water*"])
    assert not _selected(name, ["other"])


def test_active_subspace_of_ridge_function():
    rng = np.random.default_rng(0)
    w = np.array([3.0, 4.0, 0.0]) / 5.0
    X = rng.uniform(-1, 1, (200, 3))
    G = np.cos(X @ w)[:, None] * w[None, :]  # gradient of sin(w.x)
    lam, W = _eig(G)
    assert _gap_rank(lam) == 1
    np.testing.assert_allclose(np.abs(W[:, 0]), w, atol=1e-10)
    assert lam[1] < 1e-12 * lam[0]


# -- nothing is defaulted or guessed without a word


def test_pft_clip_stays_on_its_pft():
    rule = sp._rule(
        "a%x",
        {
            "range": [0.0, 10.0],
            "pft": {"4": {"range": [0.0, 10.0], "clip": [2.0, 3.0]}},
        },
        "t",
    )
    out = strat.values_of(
        rule, np.array([0.0, 0.5, 1.0]), np.array([1, 4, 1]), 0, "real"
    )
    np.testing.assert_allclose(out, [0.0, 3.0, 10.0])
    base = sp._rule(
        "a%x",
        {
            "range": [0.0, 10.0],
            "clip": [1.0, 9.0],
            "pft": {"4": {"range": [0.0, 20.0]}},
        },
        "t",
    )
    out = strat.values_of(
        base, np.array([0.0, 1.0, 1.0]), np.array([1, 4, 1]), 0, "real"
    )
    np.testing.assert_allclose(
        out, [1.0, 9.0, 9.0]
    )  # a pft rule without a clip keeps the base rule's


def test_domain_keys_must_name_something(inv):
    from types import SimpleNamespace

    from spel.proptest.domain import DomainError

    inv.vars["col_es%t_soisno"] = _var(
        "col_es%t_soisno", "begc:endc,-nlevsno+1:nlevgrnd", "input"
    )
    inv.unit_test = SimpleNamespace(time_scalars=[SimpleNamespace(name="jday_mod")])
    data = _spec(inv)
    data["scenario"]["s"]["inputs"]["col_es%t_soisno"] = 280.0
    data["domain"] = {"gridcell": [{"landunits": [{"kind": "soil", "patches": [1]}]}], "time": {"jday_mod": 10}, "dims": {"nlevsno": 5}}
    sp.parse(data, inv.case_dir / "x.toml", inv.case_dir, inv)
    data["domain"] = {"gridcell": [{"landunits": [{"kind": "soil", "patches": [1]}]}], "time": {"jdya_mod": 10}}
    with pytest.raises(DomainError, match="jdya_mod"):
        sp.parse(data, inv.case_dir / "x.toml", inv.case_dir, inv)
    data["domain"] = {"gridcell": [{"landunits": [{"kind": "soil", "patches": [1]}]}], "dims": {"nlevsnoo": 5}}
    with pytest.raises(DomainError, match="nlevsnoo"):
        sp.parse(data, inv.case_dir / "x.toml", inv.case_dir, inv)


def test_outputs_are_not_available_before_compute(inv):
    data = _spec(inv)
    data["scenario"]["s"]["assume"] = {"exprs": ["eflx_sh_veg(p) > 0.0"]}
    inv.aliases["eflx_sh_veg"] = "veg_ef%eflx_sh_veg"
    with pytest.raises(sp.SpecError, match="is an output"):
        sp.parse(data, inv.case_dir / "x.toml", inv.case_dir, inv)
    data = _spec(inv, derive={"top_as%thbot": "veg_ef%eflx_sh_veg(p)"})
    with pytest.raises(sp.SpecError, match="is an output"):
        sp.parse(data, inv.case_dir / "x.toml", inv.case_dir, inv)


def test_driver_argument_is_an_input_that_needs_a_rule(inv):
    inv.vars["nextsw_cday"] = VarInfo(
        name="nextsw_cday",
        instance="",
        field="nextsw_cday",
        type="real",
        rank=0,
        dims=[],
        status="r",
        kind="input",
        driver=True,
    )
    assert inv.vars["nextsw_cday"].nc_name == "spel_drv_nextsw_cday"
    s = sp.parse(_spec(inv), inv.case_dir / "x.toml", inv.case_dir, inv)
    assert "nextsw_cday" in sp.missing_rules(s.scenarios["s"], inv, {})
    data = _spec(inv)
    data["scenario"]["s"]["inputs"]["nextsw_cday"] = {"range": [1.0, 365.0]}
    s = sp.parse(data, inv.case_dir / "x.toml", inv.case_dir, inv)
    assert "nextsw_cday" not in sp.missing_rules(s.scenarios["s"], inv, {})


def test_element_axis_comes_from_the_expression_not_the_shape():
    # 3 elements and 3 levels: a shape cannot tell the two axes apart
    x = FArray(np.arange(9.0).reshape(3, 3), (1, 1), "x")  # x(element, level)
    z = FArray(np.arange(6.0).reshape(2, 3), (1, 1), "z")  # z(level, element)
    ev = Evaluator(lambda path: {"x": x, "z": z}[path[0]], fn.FUNCTIONS)
    p = {"p": np.array([1, 2, 3])}
    np.testing.assert_allclose(ev(parse_expr("sum(x(p, :))"), p, 3), [3.0, 12.0, 21.0])
    np.testing.assert_allclose(
        ev(parse_expr("sum(x(p, :), dim=1)"), p, 3), [3.0, 12.0, 21.0]
    )
    assert (
        ev(parse_expr("sum(x(:, :))"), p, 3) == 36.0
    )  # no element variable: one value for all elements
    assert ev(parse_expr("z(:, p)"), p, 3).shape == (3, 2)  # the element axis first
    np.testing.assert_allclose(ev(parse_expr("maxval(z(:, p))"), p, 3), [3.0, 4.0, 5.0])
    assert ev.elemental(parse_expr("x(p, 1) + 1.0")) and not ev.elemental(
        parse_expr("x(1, 1) + 1.0")
    )


def test_check_must_give_one_logical_per_element():
    from spel.proptest.runner import _per_element

    assert _per_element(np.array([True, False]), (2,), "t").tolist() == [True, False]
    assert _per_element(True, (2,), "t").tolist() == [True, True]
    with pytest.raises(sp.SpecError, match="one logical per element"):
        _per_element(np.ones((2, 5), dtype=bool), (2,), "t")


# -- end to end (opt-in: needs the CanopyFluxes case, built with `spel proptest build canflux`)

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "canopyfluxes.proptest.toml"
)


def _example_case() -> Path:
    import tomllib

    from spel.scripts.config import unittests_dir

    case = Path(unittests_dir) / tomllib.loads(EXAMPLE.read_text())["case"]["name"]
    if (
        not (case / "fut.pkl").is_file()
        or not (case / "build-py" / "libspelcase.so").exists()
    ):
        pytest.skip(
            f"{case.name} is not there or not built as a shared library (spel create, spel proptest build)"
        )
    return case


def test_canopyfluxes_example_end_to_end():
    import io

    from spel.proptest import runner

    _example_case()
    s, inventory = sp.load(EXAMPLE)
    out = io.StringIO()
    res = runner.run(
        s,
        inventory,
        checks=["copies*", "soil water*", "methane off*"],
        settings={"max_examples": 3, "seed": 1},
        out=out,
    )
    assert [o.passed for o in res] == [
        True,
        True,
        True,
    ], out.getvalue()  # the last one under another namelist option


def test_canopyfluxes_inventory_follows_the_namelist():
    _example_case()
    s, inv = sp.load(EXAMPLE)
    scen = s.scenarios["snowfree_day"]
    c13 = "photosyns_vars%c13_psnsun_patch"
    assert inv.vars[c13].kind == "output"  # every branch counted
    off = inv.configured(sp.constants(s, inv, scen))
    assert off.vars[c13].kind == "unused" and "use_c13" in off.vars[c13].dead_under
    on = inv.configured(sp.constants(s, inv, scen, {"use_c13": True}))
    assert on.vars[c13].kind == "output" and on.vars["top_as%pc13o2bot"].kind == "input"
    assert "photosyns_vars%alphapsnsun_patch" in sp.dormant_rules(scen, off)


def test_coverage_tells_branches_the_namelist_turns_off():
    from spel.proptest.coverage import decider

    decide = decider({"use_c13", "use_cn"}, {"use_c13": 0, "use_cn": 1})
    assert decide(parse_expr("use_c13")) is False
    assert decide(parse_expr("use_cn .and. use_c13")) is False
    assert (
        decide(parse_expr("(.not. lakpoi(l)) .and. use_c13")) is False
    )  # false whatever the other operand is
    assert decide(parse_expr(".not. use_c13")) is True
    assert decide(parse_expr("use_cn .and. .not. use_c13")) is True
    assert (
        decide(parse_expr("use_cn .and. frac(p) > 0.0")) is None
    )  # the namelist part holds: still open
    assert decide(parse_expr("frac(p) > 0.0")) is None  # not a namelist branch
