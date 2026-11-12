"""The IM2 season spec on hs_season (development version of the notebook's code)."""
import os
import sys

os.environ.setdefault("SPEL_E3SM_SRCROOT", "/lcrc/globalscratch/ac.schwartzpd/next")
from pathlib import Path

import netCDF4
import numpy as np

from spel.proptest import api as pt, coupling
from spel.proptest.api import Gridcell, Soil, Topounit

HILLS = {
    "lowest first":   [(50, .3), (150, .3), (300, .4)],
    "lowest last":    [(300, .4), (150, .3), (50, .3)],
    "cycle":          [(300, .25), (200, .25), (50, .25), (150, .25)],
    "caspian 184059": [(1.8, .70), (-20.7, .15), (-64.9, .10), (-110.7, .05)],
    "flat":           [(100, 1.0)],
}
IM2 = {"use_IM2_hillslope_hydrology": True}
OFF = {"use_IM2_hillslope_hydrology": False}
STEPS = 48
DRAIN = ("merge(0.0, max(0.0, top_ws%from_uphill(t) - dtime_mod * child_sum(col_wf%qflx_from_uphill, 't')), "
         "max(0.0, top_ws%from_uphill(t) - dtime_mod * child_sum(col_wf%qflx_from_uphill, 't')) < 1.0e-20)")


def hill(name, tops):
    return Gridcell(name=name, topounits=tuple(Topounit((Soil(13),), weight=w, elevation=e) for e, w in tops))


def reference_geometry(case):
    out = {}
    with netCDF4.Dataset(case.reference / "spel-inputs0001.nc") as d:
        d.set_auto_mask(False)
        active = np.asarray(d["col_pp__active"][0]).astype(bool)
        col = int(np.nonzero(active & (np.asarray(d["col_pp__itype"][0]) == 1))[0][0])
        for name in ("col_pp%dz", "col_pp%z", "col_pp%zi"):
            v = d[name.replace("%", "__")]
            vals, lb = np.asarray(v[0])[:, col], int(np.atleast_1d(v.lbounds)[-1])
            out[name] = pt.Value(0.0, levels={str(lb + j): pt.Value(float(x)) for j, x in enumerate(vals)})
    return out


def process(case):
    """The isolated process both specs run: the inputs every scenario shares, and `keep` (the
    boundary variables left to the process, less those declared otherwise)."""
    live = case.inv.configured({k.lower(): v for k, v in IM2.items()})
    boundary = sorted(n for n, w in coupling.boundary(case.inv, {"use_im2_hillslope_hydrology": True}).items()
                      if live.vars[n].kind in ("input", "inout"))
    boundary = sorted(set(boundary) | {"col_wf%qflx_snwcp_ice", "col_wf%qflx_snwcp_liq"})  # live under the
    # reference's other options too (snow capping): the run checks with all of them
    # the process alone: no snow, no ice, no evaporation or transpiration, warm soil; rain on the ground
    zero = [n for n in boundary if n.startswith(("col_wf%", "veg_wf%")) and n != "col_wf%qflx_rain_grnd"]
    zero += ["col_ws%h2osno", "col_ws%frac_sno", "col_ws%frac_sno_eff", "col_ws%int_snow", "col_ws%swe_old",
             "col_ws%snw_rds", "col_ws%h2osoi_ice", "col_ef%imelt", "col_ws%frac_iceold"]
    zero += [n for n in boundary if n.startswith("aerosol_vars%")]
    common = {n: 0.0 for n in dict.fromkeys(zero) if n in live.vars and live.vars[n].kind in ("input", "inout")}
    common.update({
        "col_es%t_soisno": pt.Range(278.0, 300.0), "col_es%t_grnd": pt.Range(280.0, 300.0),
        "col_es%t_h2osfc": pt.Range(280.0, 300.0), "col_ws%do_capsnow": 0,
        "atm2lnd_vars%frac_h2orof_grc": 0.0, "atm2lnd_vars%h2orof_grc": 0.0,  # no river water on the land
        **reference_geometry(case),
    })

    def keep(forcing=()):
        return sorted(set(boundary) - {"top_ws%from_uphill"} - set(forcing))

    return common, keep


def build(path: Path):
    case = pt.Case("hs_season", reference="hs_season")
    dom = pt.Domain([hill(n, t) for n, t in HILLS.items()], nsets=2, reference="hs_season")
    spec = pt.Spec(case, dom, namelist=IM2, build={"fc": "gfortran"}, max_examples=8, path=path)
    rules = case.stats_rules(namelist=IM2)
    common, keep = process(case)
    between = {"top_ws%from_uphill": DRAIN}

    scen = dict(over="c in hydrologyc", extends=rules, steps=STEPS, between=between)
    spec.scenario("storm", inputs={**common, "col_wf%qflx_rain_grnd": pt.LogRange(2e-4, 2e-3),
                                   "top_ws%from_uphill": 0.0},
                  forcing={"col_wf%qflx_rain_grnd": "merge(col_wf%qflx_rain_grnd(c), 0.0, step <= 12)"},
                  keep=keep(["col_wf%qflx_rain_grnd"]), **scen,
                  comment="6 hours of 0.7-7 mm/h rain on the ground, then 18 dry hours; stores start empty")
    spec.scenario("steady", inputs={**common, "col_wf%qflx_rain_grnd": pt.LogRange(1e-4, 1e-3),
                                    "top_ws%from_uphill": 0.0},
                  keep=keep(), **scen, comment="one day of constant rain (held: keep); stores start empty")
    spec.scenario("recession", inputs={**common, "col_wf%qflx_rain_grnd": 0.0,
                                       "top_ws%from_uphill": pt.Range(0.5, 5.0)},
                  keep=keep(), **scen, comment="no rain; the stores start full")

    # -- properties
    spec.property("the store stays finite and non-negative",
                  "finite(top_ws%from_uphill(t)) .and. top_ws%from_uphill(t) >= 0.0", over="t")
    spec.property("no topounit sends water to itself",
                  "implies(col_wf%qflx_to_downhill(c) > 0.0, top_pp%downhill_ti(t) /= t)")
    spec.property("the stores gain what is sent and lose what is released",
                  "close(tend(child_wsum(top_ws%from_uphill, 'g')), dtime_mod * (child_wsum(col_wf%qflx_to_downhill, 'g')"
                  " - child_wsum(col_wf%qflx_from_uphill, 'g')), rtol=1.0e-9, atol=1.0e-12)", over="g")
    spec.property("a topounit releases half its store per step",
                  "close(dtime_mod * child_sum(col_wf%qflx_from_uphill, 't'), 0.5 * prev%top_ws%from_uphill(t),"
                  " rtol=1.0e-12, atol=1.0e-15)", over="t")
    spec.property("the topmost topounit's store halves every step in a recession",
                  "implies(uphill_count(t) == 0, close(top_ws%from_uphill(t), 0.5 * prev%top_ws%from_uphill(t),"
                  " rtol=1.0e-12, atol=1.0e-20))", over="t", scenarios=["recession"])
    # -- relations: the same example with IM2 off
    same_soil = ("all(same(follow%col_ws%h2osoi_liq(c, 1:nlevgrnd), base%col_ws%h2osoi_liq(c, 1:nlevgrnd)))"
                 " .and. same(follow%col_ws%h2osfc(c), base%col_ws%h2osfc(c))")
    spec.relation("R1 a gridcell of one topounit is the same with IM2 off",
                  f"implies(top_pp%downhill_ti(t) == -1 .and. uphill_count(t) == 0, {same_soil}"
                  " .and. same(follow%col_wf%qflx_runoff(c), base%col_wf%qflx_runoff(c)))",
                  namelist=OFF, scenarios=["storm", "steady"])
    spec.relation("R2 the soil of the topmost topounit is the same with IM2 off",
                  f"implies(uphill_count(t) == 0, {same_soil})", namelist=OFF, scenarios=["storm", "steady"])
    spec.relation("R4 water is conserved: soil + stores + runoff so far are the same with IM2 off",
                  "close(child_wsum(base%col_ws%endwb, 'g') + child_wsum(base%top_ws%from_uphill, 'g')"
                  " + dtime_mod * acc(child_wsum(base%col_wf%qflx_runoff, 'g')),"
                  " child_wsum(follow%col_ws%endwb, 'g') + dtime_mod * acc(child_wsum(follow%col_wf%qflx_runoff, 'g')),"
                  " rtol=1.0e-9, atol=1.0e-9)", namelist=OFF, over="g", scenarios=["storm", "steady"])
    spec.relation("D1 IM2 never gets water to the river sooner",
                  "acc(child_wsum(base%col_wf%qflx_runoff, 'g')) <= acc(child_wsum(follow%col_wf%qflx_runoff, 'g'))"
                  " * (1.0 + 1.0e-12) + 1.0e-15", namelist=OFF, over="g", scenarios=["storm", "steady"])
    spec.relation("D2 a topounit that receives water holds at least as much",
                  "implies(uphill_count(t) > 0, base%col_ws%endwb(c) >= follow%col_ws%endwb(c) - 1.0e-9)",
                  namelist=OFF, scenarios=["storm", "steady"])
    spec.relation("D3 the outlet passes on at least what it does without IM2",
                  "implies(lowest(t), acc(base%col_wf%qflx_runoff(c)) >= acc(follow%col_wf%qflx_runoff(c))"
                  " * (1.0 - 1.0e-12) - 1.0e-15)", namelist=OFF, scenarios=["storm", "steady"])
    return spec


# H2: a weight-0 (inactive) topounit as a downhill neighbour. Listed lowest first, so H1 never
# triggers: every failure is H2's. "lowest first" is the intact control.
HILLS_H2 = {
    "lowest first": HILLS["lowest first"],
    "dead outlet":  [(50, 0.0), (150, .5), (300, .5)],   # the lowest has weight 0
    "dead middle":  [(50, .5), (150, 0.0), (300, .5)],   # the chain breaks in the middle
    "dead top":     [(50, .5), (150, .5), (300, 0.0)],   # sends nothing: behaves as 2 topounits
}
# ELM's drain reads qflx_from_uphill on every column; SurfaceRunoff writes it on hydrologyc only, so
# the inactive columns of a weight-0 topounit keep the cold start's 0 (carried, not NaN)
CARRY = {"col_wf%qflx_from_uphill": 0.0}


def build_h2(path: Path, steps: int = 24):
    case = pt.Case("hs_season", reference="hs_season")
    dom = pt.Domain([hill(n, t) for n, t in HILLS_H2.items()], nsets=2, reference="hs_season")
    spec = pt.Spec(case, dom, namelist=IM2, build={"fc": "gfortran"}, max_examples=6, path=path)
    rules = case.stats_rules(namelist=IM2)
    common, keep = process(case)
    scen = dict(over="c in hydrologyc", extends=rules, steps=steps, between={"top_ws%from_uphill": DRAIN},
                carry=CARRY)
    storm = dict(inputs={**common, "col_wf%qflx_rain_grnd": pt.LogRange(2e-4, 2e-3), "top_ws%from_uphill": 0.0},
                 forcing={"col_wf%qflx_rain_grnd": f"merge(col_wf%qflx_rain_grnd(c), 0.0, step <= {steps // 4})"},
                 keep=keep(["col_wf%qflx_rain_grnd"]), **scen)
    spec.scenario("storm", **storm, comment="rain on the ground for the first quarter of the run; stores start empty")
    spec.scenario("storm trapped", **storm, trap_fpe=True,
                  comment="the same storm with floating-point exceptions trapped: ELM's division by the weight-0 "
                          "topounit's weight stops step 1")

    spec.property("H2 no active topounit sends water to an inactive one",
                  "implies(col_wf%qflx_to_downhill(c) > 0.0, .not. downhill_inactive(t))", scenarios=["storm"])
    spec.property("the store stays finite and non-negative",
                  "finite(top_ws%from_uphill(t)) .and. top_ws%from_uphill(t) >= 0.0", over="t", scenarios=["storm"])
    spec.property("the stores gain what is sent and lose what is released",
                  "close(tend(child_wsum(top_ws%from_uphill, 'g')), dtime_mod * (child_wsum(col_wf%qflx_to_downhill, 'g')"
                  " - child_wsum(col_wf%qflx_from_uphill, 'g')), rtol=1.0e-9, atol=1.0e-12)", over="g",
                  scenarios=["storm"])
    spec.property("the routines finish every step", ".true.", over="g", scenarios=["storm trapped"])
    spec.relation("R4 water is conserved: soil + stores + runoff so far are the same with IM2 off",
                  "close(child_wsum(base%col_ws%endwb, 'g') + child_wsum(base%top_ws%from_uphill, 'g')"
                  " + dtime_mod * acc(child_wsum(base%col_wf%qflx_runoff, 'g')),"
                  " child_wsum(follow%col_ws%endwb, 'g') + dtime_mod * acc(child_wsum(follow%col_wf%qflx_runoff, 'g')),"
                  " rtol=1.0e-9, atol=1.0e-9)", namelist=OFF, over="g", scenarios=["storm"])
    return spec


if __name__ == "__main__":
    which = sys.argv[2] if len(sys.argv) > 2 else "season"
    spec = (build_h2 if which == "h2" else build)(Path(sys.argv[1]))
    spec.to_toml(spec.path)
    print("written", spec.path)
