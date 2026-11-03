"""access_guards: an instance whose history sources are registered outside its namelist guard is copied unguarded."""
from types import SimpleNamespace

from spel.scripts.fortran_parser.boolen_expression import Expectation
from spel.scripts.nml import analyze_namelist
from spel.scripts.registrations import Guard
from spel.scripts import validate_access
from spel.transforms.device import generate


def _type(name, instances, comps=()):
    return SimpleNamespace(type_name=name, instances={i: SimpleNamespace(name=i) for i in instances},
                           components={c: SimpleNamespace(name=c) for c in comps})


def test_instance_with_history_sources_is_not_gated(monkeypatch):
    dyn = Expectation("active_dynamic_subgrid", "True")
    c13 = Expectation("use_c13", "True")
    # both instances are accessed only under a namelist flag in the analyzed tree
    hill = Expectation("use_im2_hillslope_hydrology", "True")
    raw = {"subgrid_weights_diagnostics%pct_cft": dyn, "c13_col_cf%x": c13, "top_ws%from_uphill": hill}
    monkeypatch.setattr(analyze_namelist, "check_subs_for_nml_guarded_vars", lambda roots, d: {})
    monkeypatch.setattr(analyze_namelist, "nml_guards", lambda r: raw)
    monkeypatch.setattr(validate_access, "merge_guards", lambda pairs: (list(pairs), raw)[1])
    swd = _type("subgrid_weights_diagnostics_type", ["subgrid_weights_diagnostics"], ["pct_cft"])
    cf = _type("column_carbon_flux", ["c13_col_cf"], ["x"])
    tws = _type("topounit_water_state", ["top_ws"], ["from_uphill"])
    types = {"swd": swd, "cf": cf, "tws": tws}
    # PCT_CFT is registered on the module instance (hist_addfld2d ptr_topo=subgrid_weights_diagnostics%pct_cft)
    reg = SimpleNamespace(kind="hist", field="pct_cft", base="subgrid_weights_diagnostics", guards=())
    # WATER_FROM_UPHILL is registered only `if (use_IM2_hillslope_hydrology)`, the copy's own condition
    reg_top = SimpleNamespace(kind="hist", field="from_uphill", base="top_ws",
                              guards=(Guard("use_im2_hillslope_hydrology", None),))
    root = SimpleNamespace(elmtype_access_by_ln=dict.fromkeys(raw, {}))
    fut = SimpleNamespace(
        subroutine_dict={"elm_driver::elm_drv": root,
                         "subgridweightsmod::init_subgrid_weights_mod": SimpleNamespace(registrations=[reg], arguments={}),
                         "topounitdatatype::init_top_ws": SimpleNamespace(registrations=[reg_top], arguments={})},
        type_dict=types,
        module_dict={"elm_varctl": SimpleNamespace(global_vars={"active_dynamic_subgrid": 1, "use_c13": 1,
                                                          "use_im2_hillslope_hydrology": 1})})
    g = generate.access_guards(fut, types)
    assert "subgrid_weights_diagnostics" not in g.instances
    assert g.instances.get("c13_col_cf") == "use_c13"
    assert g.instances.get("top_ws") == "use_im2_hillslope_hydrology"  # its only field needs the same flag
    assert any("subgrid_weights_diagnostics owns history sources" in n for n in g.notes)
