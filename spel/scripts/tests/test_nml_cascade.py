from spel.scripts.fortran_parser.boolen_expression import (
    NO_EXPECTATION,
    AllOf,
    Expectation,
    infer_condition_expectations,
    simplify,
)
from spel.scripts.fortran_parser.spel_ast import FuncExpression, Identifier, InfixExpression, PrefixExpression
from spel.scripts.fortran_parser.tokens import Token, TokenTypes
from spel.scripts.nml.analyze_namelist import cascade_to_triggers
from spel.scripts.nml.namelist_cascade import NML_CASCADES, cascade_trigger_variables

tok = Token(token=TokenTypes.EOF, literal="temp")

# h2osfc(c) > min_h2osfc .and. lun_pp%ispolygon(l)  (CanopyHydrologyMod::FracH2OSfc)
ISPOLYGON = FuncExpression(
    tok=tok, fn=Identifier(tok=tok, value="lun_pp%ispolygon"), args=[Identifier(tok=tok, value="l")]
)
COND = InfixExpression(
    tok=tok,
    op=".and.",
    left=InfixExpression(
        tok=tok,
        op=">",
        left=Identifier(tok=tok, value="h2osfc"),
        right=Identifier(tok=tok, value="min_h2osfc"),
    ),
    right=ISPOLYGON,
)


def test_cascade_array_element_is_matched():
    assert infer_condition_expectations(COND, {"ispolygon"}) == Expectation("ispolygon", "True")


def test_ispolygon_implies_use_polygonal_tundra():
    expectation = cascade_to_triggers(infer_condition_expectations(COND, {"ispolygon"}))
    assert expectation == Expectation("use_polygonal_tundra", "True")
    # non-polygon landunits exist with either setting
    negated = PrefixExpression(tok=tok, op=".not.", right=ISPOLYGON)
    assert simplify(
        cascade_to_triggers(infer_condition_expectations(negated, {"ispolygon"}))
    ) == NO_EXPECTATION


def test_integer_cascades_become_trigger_guards():
    # a filter count isn't a global the IO modules could test, so the guard
    # moves to the trigger (num_pcropp > 0 only happens with use_crop)
    guard = AllOf((Expectation("num_pcropp", "> 0"),))
    assert cascade_to_triggers(guard) == AllOf((Expectation("use_crop", "True"),))


def test_trigger_constraint_both_pairs():
    use_crop = NML_CASCADES["num_pcropp"]
    assert use_crop.trigger_name == "use_crop"
    assert use_crop.trigger_constraint("True") is None


def test_cascade_trigger_variables():
    found = cascade_trigger_variables(["use_polygonal_tundra", "irrigate"])
    assert list(found) == ["use_polygonal_tundra"]
    var = found["use_polygonal_tundra"]
    assert (var.type, var.dim, var.declaration) == ("logical", 0, "elm_varctl")


def test_callee_guard_propagates_through_call_site():
    # CanopyHydrology calls FracH2OSfc (line 10), which reads iwp only under ispolygon
    from types import SimpleNamespace

    from spel.scripts.nml.analyze_namelist import nml_guards

    polygon = Expectation("use_polygonal_tundra", "True")
    irrigate = Expectation("irrigate", "True")
    if_at = lambda guards: lambda ln: (  # noqa: E731
        SimpleNamespace(condtional_expectation=guards[ln]) if ln in guards else None
    )
    rw = lambda ln: SimpleNamespace(ln=ln)  # noqa: E731
    child = SimpleNamespace(
        id="frach2osfc",
        sub_call_desc={},
        child_subroutines={},
        elmtype_access_by_ln={"col_ws%iwp_microrel": [rw(3)], "col_ws%h2osfc": [rw(1)]},
        namelist_if_at=if_at({3: polygon}),
    )
    root = SimpleNamespace(
        id="canopyhydrology",
        sub_call_desc={10: SimpleNamespace(fn="frach2osfc")},
        child_subroutines={"frach2osfc": child},
        elmtype_access_by_ln={
            "col_ws%iwp_microrel": [rw(10)],
            "col_ws%h2osfc": [rw(10)],
            "veg_wf%qflx_supply_patch": [rw(5)],
        },
        namelist_if_at=if_at({5: irrigate}),
    )
    assert nml_guards(root) == {
        "col_ws%iwp_microrel": polygon,
        "veg_wf%qflx_supply_patch": irrigate,
    }


def test_if_condition_line_is_not_guarded_by_the_if():
    from types import SimpleNamespace

    from spel.scripts.analyze_subroutines import Subroutine

    polygon_if = SimpleNamespace(
        start_ln=10, end_ln=20, nml_vars={}, nml_cascades={"ispolygon": None}, copy=lambda: "if"
    )
    sub = SimpleNamespace(flat_ifs=[polygon_if])
    assert Subroutine.namelist_if_at(sub, 10) is None  # the condition reads lun_pp%ispolygon
    assert Subroutine.namelist_if_at(sub, 11) == "if"


def test_ifs_inside_loops_are_flattened():
    # CH4Mod: if (.not. lake) > do fp > if (.not. col_pp%is_fates(c)) ... else
    from spel.scripts.fortran_parser.spel_ast import BlockStatement, DoLoop, IfConstruct
    from spel.scripts.nml.analyze_ifs import nested_ifs

    inner = IfConstruct.__new__(IfConstruct)
    body = BlockStatement(tok)
    body.statements = [inner]
    loop = DoLoop.__new__(DoLoop)
    loop.body = body
    assert nested_ifs([loop]) == [inner]


def test_guarded_vars_are_the_types_own_components():
    # guards are looked up by identity, so they must hold dtype.components[...] itself
    from types import SimpleNamespace

    from spel.scripts.nml.analyze_namelist import check_subs_for_nml_guarded_vars
    from spel.scripts.utilityFunctions import Variable

    fates = Expectation("use_fates", "True")
    comp = Variable("logical", "is_bareground", "patch", 0, 1, bounds="begp:endp")
    dtype = SimpleNamespace(components={"is_bareground": comp}, procedures={})
    root = SimpleNamespace(
        id="ch4mod::ch4",
        sub_call_desc={},
        child_subroutines={},
        elmtype_access_by_ln={"veg_pp%is_bareground": [SimpleNamespace(ln=5)]},
        namelist_if_at=lambda ln: SimpleNamespace(condtional_expectation=fates),
    )
    guards = check_subs_for_nml_guarded_vars([root], {"veg_pp": dtype})
    assert guards[fates][0] is comp


def test_integer_cascade_to_trigger():
    from spel.scripts.nml.analyze_namelist import cascade_to_triggers
    from spel.scripts.fortran_parser.boolen_expression import Expectation, AnyOf, NO_EXPECTATION, simplify

    # num_pcropp > 0 only with use_crop; == 0 happens either way (a rank without crop patches)
    assert cascade_to_triggers(Expectation("num_pcropp", "> 0")) == Expectation("use_crop", "True")
    assert cascade_to_triggers(Expectation("num_ppercropp", "/= 0")) == Expectation("use_crop", "True")
    assert cascade_to_triggers(Expectation("num_pcropp", "== 0")) == NO_EXPECTATION
    assert cascade_to_triggers(Expectation("num_pcropp", "<= 0")) == NO_EXPECTATION
    both = cascade_to_triggers(AnyOf((Expectation("num_pcropp", "> 0"), Expectation("num_ppercropp", "> 0"))))
    assert "num_pcropp" not in both.variable_names() and "num_ppercropp" not in both.variable_names()
    assert simplify(both) == Expectation("use_crop", "True")


def test_multi_trigger_cascades():
    # carbon_only and friends follow suplnitro/suplphos (AllocationMod::EvaluateSupplStatus),
    # known only outside accelerated spinup
    dep = NML_CASCADES["carbon_only"]
    assert dep.trigger_names == ("suplnitro", "suplphos", "spinup_state")
    on = [dep.pair_values(p) for p in dep.pairs if p.cascade_val == ".true."]
    assert on == [("all", "all", "0")]
    assert dep.trigger_constraint("True") is None  # no single trigger value follows
    assert NML_CASCADES["carbonnitrogen_only"].pairs[1] == (("none", "all", "0"), ".true.")
    found = cascade_trigger_variables(["suplnitro", "spinup_state", "use_crop"])
    assert {n: v.type for n, v in found.items()} == {"suplnitro": "character", "spinup_state": "integer", "use_crop": "logical"}
    assert found["suplnitro"].declaration == "allocationmod"


def test_multi_trigger_guard_has_no_trigger_expectation():
    guard = AllOf((Expectation("carbon_only", "True"),))
    assert simplify(cascade_to_triggers(guard)) == NO_EXPECTATION


# if (get_do_transient_pfts()) then   (dynSubgridDriverMod: dynpft_interp)
TRANSIENT_PFTS = FuncExpression(tok=tok, fn=Identifier(tok=tok, value="get_do_transient_pfts"), args=[])


def test_function_of_no_arguments_is_a_cascade_variable():
    assert infer_condition_expectations(TRANSIENT_PFTS, {"get_do_transient_pfts"}) == Expectation(
        "get_do_transient_pfts", "True")
    # an array element of the same name is not (only functions of no arguments)
    elem = FuncExpression(tok=tok, fn=Identifier(tok=tok, value="get_do_transient_pfts"),
                          args=[Identifier(tok=tok, value="i")])
    assert infer_condition_expectations(elem, {"get_do_transient_pfts"}) == NO_EXPECTATION


def test_transient_data_accessors_follow_the_dynamic_subgrid_namelist():
    for what in ("pfts", "crops"):
        dep = NML_CASCADES[f"get_do_transient_{what}"]
        assert dep.trigger_name == f"do_transient_{what}"
        assert dep.trigger_defaults == (".false.",)
        cond = infer_condition_expectations(
            FuncExpression(tok=tok, fn=Identifier(tok=tok, value=f"get_do_transient_{what}"), args=[]),
            {f"get_do_transient_{what}"})
        assert cascade_to_triggers(cond) == Expectation(f"do_transient_{what}", "True")


def test_configuration_takes_the_trigger_defaults():
    from types import SimpleNamespace as NS

    from spel.transforms.core.config import Configuration

    f = NS(start_ln=10, end_ln=14, nml_vars={}, nml_cascades={"get_do_transient_pfts": NML_CASCADES["get_do_transient_pfts"]},
           condtional_expectation=Expectation("get_do_transient_pfts", "True"))
    fut = NS(subroutine_dict={"m::s": NS(id="m::s", flat_ifs=[f])})
    cfg = Configuration({})
    cfg.add_defaults(fut)
    assert cfg.values["do_transient_pfts"] is False
    assert cfg.dead_branches(fut.subroutine_dict["m::s"]) == [(11, 13)]
    on = Configuration({"do_transient_pfts": True})
    on.add_defaults(fut)
    assert on.dead_branches(fut.subroutine_dict["m::s"]) == []
