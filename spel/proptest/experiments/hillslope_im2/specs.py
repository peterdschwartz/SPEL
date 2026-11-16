"""The specs of the IM2 hillslope experiment (hillslope_im2.ipynb, README.md), all on the case
`hs_season` (HydrologyNoDrainage + HydrologyDrainage, multi-step):

- `random_hillslopes`: IM2's invariants on drawn domains, written without knowing any bug; the
  property test that finds them;
- one spec per hypothesis its shrunk failures suggest, on declared hillslopes:
  `downhill_follows_listing_order`, `water_sent_to_inactive_topounit`, `no_weighted_hydrology_column`.

`python -m spel.proptest.experiments.hillslope_im2.specs [name ...]` writes their TOML next to this
file (`<name>.proptest.toml`), for `spel proptest run`."""

import sys
from pathlib import Path

import numpy as np

from spel.proptest import api as pt, coupling
from spel.proptest.api import Gridcell, Lake, Soil, Topounit

HERE = Path(__file__).resolve().parent
CASE = "hs_season"
IM2_ON = {"use_IM2_hillslope_hydrology": True}
IM2_OFF = {"use_IM2_hillslope_hydrology": False}

# -- the isolated process every spec runs

# ColWaterBalanceCheck drains the store after the case's routines: what the rest of the step does to it
STORE_DRAIN = ("merge(0.0, max(0.0, top_ws%from_uphill(t) - dtime_mod * child_sum(col_wf%qflx_from_uphill, 't')), "
               "max(0.0, top_ws%from_uphill(t) - dtime_mod * child_sum(col_wf%qflx_from_uphill, 't')) < 1.0e-20)")
# ELM's drain reads qflx_from_uphill on every column; SurfaceRunoff writes it on hydrologyc only, so
# the inactive columns of a weight-0 topounit keep the cold start's 0 (carried, not NaN)
CARRY = {"col_wf%qflx_from_uphill": 0.0}
# Water conservation sums the columns the routines compute (hydrologyc): a lake's qflx_runoff, which
# lake hydrology would set, stays NaN here. The stores are per topounit.
HYDROLOGY_COLUMNS = "filter%hydrologyc"
WATER_CONSERVED = (
    "water is conserved: soil + stores + runoff so far are the same with IM2 off",
    f"close(child_wsum(base%col_ws%endwb, 'g', {HYDROLOGY_COLUMNS}) + child_wsum(base%top_ws%from_uphill, 'g')"
    f" + dtime_mod * acc(child_wsum(base%col_wf%qflx_runoff, 'g', {HYDROLOGY_COLUMNS})),"
    f" child_wsum(follow%col_ws%endwb, 'g', {HYDROLOGY_COLUMNS})"
    f" + dtime_mod * acc(child_wsum(follow%col_wf%qflx_runoff, 'g', {HYDROLOGY_COLUMNS})),"
    " rtol=1.0e-9, atol=1.0e-9)")
STORE_FINITE = ("the store stays finite and non-negative",
                "finite(top_ws%from_uphill(t)) .and. top_ws%from_uphill(t) >= 0.0")
NO_SELF_SEND = ("no topounit sends water to itself", "implies(col_wf%qflx_to_downhill(c) > 0.0, top_pp%downhill_ti(t) /= t)")
FLUX_FINITE = ("the flux from uphill is finite on every hydrology column", "finite(col_wf%qflx_from_uphill(c))")
STORE_BALANCE = ("the stores gain what is sent and lose what is released",
                 "close(tend(child_wsum(top_ws%from_uphill, 'g')), dtime_mod * (child_wsum(col_wf%qflx_to_downhill, 'g')"
                 " - child_wsum(col_wf%qflx_from_uphill, 'g')), rtol=1.0e-9, atol=1.0e-12)")


def reference_geometry(case: pt.Case) -> dict:
    """ELM's soil vertical grid (dz, z, zi), one value per level, from the capture's first active soil
    column: drawn per level, the depths and interfaces would not fit together."""
    out = {}
    with case.reference_inputs() as d:
        active = np.asarray(d["col_pp__active"][0]).astype(bool)
        col = int(np.nonzero(active & (np.asarray(d["col_pp__itype"][0]) == 1))[0][0])
        for name in ("col_pp%dz", "col_pp%z", "col_pp%zi"):
            v = d[name.replace("%", "__")]
            vals, lb = np.asarray(v[0])[:, col], int(np.atleast_1d(v.lbounds)[-1])
            out[name] = pt.Value(0.0, levels={str(lb + j): pt.Value(float(x)) for j, x in enumerate(vals)})
    return out


def isolated_process(case: pt.Case):
    """IM2 routing alone: the inputs every scenario shares (no snow, ice, evaporation or
    transpiration, warm soil, ELM's soil grid), and `keep(forcing)`: the boundary variables
    (written elsewhere in ELM's step) left to the process, less those declared otherwise."""
    live = case.inv.configured({k.lower(): v for k, v in IM2_ON.items()})
    boundary = sorted(n for n, w in coupling.boundary(case.inv, {"use_im2_hillslope_hydrology": True}).items()
                      if live.vars[n].kind in ("input", "inout"))
    boundary = sorted(set(boundary) | {"col_wf%qflx_snwcp_ice", "col_wf%qflx_snwcp_liq"})  # live under the
    # reference's other options too (snow capping): the run checks with all of them
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


def _spec(path: Path, domain: pt.Domain, max_examples: int) -> tuple[pt.Spec, pt.Case, Path, dict, object]:
    case = pt.Case(CASE, reference=CASE)
    spec = pt.Spec(case, domain, namelist=IM2_ON, build={"fc": "gfortran"}, max_examples=max_examples, path=path)
    common, keep = isolated_process(case)
    return spec, case, case.stats_rules(namelist=IM2_ON), common, keep


def _storm(spec: pt.Spec, rules: Path, common: dict, keep, steps: int, rain_steps: int, name: str = "storm",
           comment: str = "", **extra) -> None:
    """Rain on the ground for the first `rain_steps` steps; the stores start empty."""
    spec.scenario(name, over="c in hydrologyc", extends=rules, steps=steps,
                  between={"top_ws%from_uphill": STORE_DRAIN}, carry=CARRY,
                  inputs={**common, "col_wf%qflx_rain_grnd": pt.LogRange(2e-4, 2e-3), "top_ws%from_uphill": 0.0},
                  forcing={"col_wf%qflx_rain_grnd": f"merge(col_wf%qflx_rain_grnd(c), 0.0, step <= {rain_steps})"},
                  keep=keep(["col_wf%qflx_rain_grnd"]),
                  comment=comment or f"0.7-7 mm/h of rain on the ground for {rain_steps} of {steps} steps; stores start empty",
                  **extra)


# -- the property test: IM2's invariants on random hillslopes

RANDOM_HILLSLOPES = pt.DomainDraw(count=4, gridcells=(2, 3), topounits=(1, 3),
                                  kinds={"soil": 0.7, "wetland": 0.4, "lake": 0.3, "glacier": 0.3},
                                  zero_topounit=0.15, zero_landunit=0.3, pfts=(13,), elevation=(0.0, 500.0))


def random_hillslopes(path: Path, steps: int = 12, downhill: str = "elm") -> pt.Spec:
    """What IM2 routing must do on any hillslope, on drawn domains: gridcells of 1-3 topounits of
    soil, wetland, lake and plain glacier at random elevations and weights (some of weight 0), over a
    storm. Nothing here knows of a bug."""
    dom = pt.Domain(draw=RANDOM_HILLSLOPES, nsets=2, reference=CASE, downhill=downhill)
    spec, case, rules, common, keep = _spec(path, dom, max_examples=4)
    _storm(spec, rules, common, keep, steps, steps // 2)
    spec.property(*FLUX_FINITE)
    spec.property(*STORE_FINITE, over="t")
    spec.property(*NO_SELF_SEND)
    spec.relation(*WATER_CONSERVED, namelist=IM2_OFF, over="g")
    return spec


# -- hypothesis: the downhill neighbour follows the order the topounits are listed in

HILLSLOPES = {  # (elevation m, weight) per topounit, in the order the surface data lists them
    "lowest first":   [(50, .3), (150, .3), (300, .4)],
    "lowest last":    [(300, .4), (150, .3), (50, .3)],   # the same hillslope, listed the other way
    "cycle":          [(300, .25), (200, .25), (50, .25), (150, .25)],
    "caspian 184059": [(1.8, .70), (-20.7, .15), (-64.9, .10), (-110.7, .05)],   # a real gridcell
    "flat":           [(100, 1.0)],   # one topounit: IM2 must change nothing
}


def downhill_follows_listing_order(path: Path, steps: int = 48, downhill: str = "elm") -> pt.Spec:
    """The same hillslope listed lowest first and lowest last, a loop, a real gridcell listed highest
    first and a one-topounit gridcell, over a day: storm, steady rain and a recession. If the
    neighbour follows the listing order, only the hillslopes not listed lowest first break the
    routing checks, and the river gets less, later."""
    dom = pt.Domain([Gridcell.hillslope(t, Soil(13), name=n) for n, t in HILLSLOPES.items()], nsets=2, reference=CASE,
                    downhill=downhill)
    spec, case, rules, common, keep = _spec(path, dom, max_examples=8)
    scen = dict(over="c in hydrologyc", extends=rules, steps=steps, between={"top_ws%from_uphill": STORE_DRAIN})
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

    spec.property(*STORE_FINITE, over="t")
    spec.property(*NO_SELF_SEND)
    spec.property(*STORE_BALANCE, over="g")
    spec.property("a topounit releases half its store per step",
                  "close(dtime_mod * child_sum(col_wf%qflx_from_uphill, 't'), 0.5 * prev%top_ws%from_uphill(t),"
                  " rtol=1.0e-12, atol=1.0e-15)", over="t")
    spec.property("the topmost topounit's store halves every step in a recession",
                  "implies(uphill_count(t) == 0, close(top_ws%from_uphill(t), 0.5 * prev%top_ws%from_uphill(t),"
                  " rtol=1.0e-12, atol=1.0e-20))", over="t", scenarios=["recession"])
    # the same example with IM2 off
    same_soil = ("all(same(follow%col_ws%h2osoi_liq(c, 1:nlevgrnd), base%col_ws%h2osoi_liq(c, 1:nlevgrnd)))"
                 " .and. same(follow%col_ws%h2osfc(c), base%col_ws%h2osfc(c))")
    wet = ["storm", "steady"]
    spec.relation("a gridcell of one topounit is the same with IM2 off",
                  f"implies(top_pp%downhill_ti(t) == -1 .and. uphill_count(t) == 0, {same_soil}"
                  " .and. same(follow%col_wf%qflx_runoff(c), base%col_wf%qflx_runoff(c)))",
                  namelist=IM2_OFF, scenarios=wet)
    spec.relation("the soil of the topmost topounit is the same with IM2 off",
                  f"implies(uphill_count(t) == 0, {same_soil})", namelist=IM2_OFF, scenarios=wet)
    spec.relation(*WATER_CONSERVED, namelist=IM2_OFF, over="g", scenarios=wet)
    spec.relation("IM2 never gets water to the river sooner",
                  "acc(child_wsum(base%col_wf%qflx_runoff, 'g')) <= acc(child_wsum(follow%col_wf%qflx_runoff, 'g'))"
                  " * (1.0 + 1.0e-12) + 1.0e-15", namelist=IM2_OFF, over="g", scenarios=wet)
    spec.relation("a topounit that receives water holds at least as much",
                  "implies(uphill_count(t) > 0, base%col_ws%endwb(c) >= follow%col_ws%endwb(c) - 1.0e-9)",
                  namelist=IM2_OFF, scenarios=wet)
    spec.relation("the outlet passes on at least what it does without IM2",
                  "implies(lowest(t), acc(base%col_wf%qflx_runoff(c)) >= acc(follow%col_wf%qflx_runoff(c))"
                  " * (1.0 - 1.0e-12) - 1.0e-15)", namelist=IM2_OFF, scenarios=wet)
    return spec


# -- hypothesis: water routed to a weight-0 (inactive) topounit leaves the gridcell's water budget

INACTIVE_TOPOUNIT_HILLSLOPES = {  # all listed lowest first, so the listing order plays no part
    "lowest first": HILLSLOPES["lowest first"],   # the intact control
    "dead outlet":  [(50, 0.0), (150, .5), (300, .5)],   # the lowest has weight 0
    "dead middle":  [(50, .5), (150, 0.0), (300, .5)],   # the chain breaks in the middle
    "dead top":     [(50, .5), (150, .5), (300, 0.0)],   # sends nothing: behaves as 2 topounits
}


def water_sent_to_inactive_topounit(path: Path, steps: int = 24, downhill: str = "elm") -> pt.Spec:
    """Hillslopes with one topounit of weight 0 at the bottom, middle or top. If water routed to an
    inactive topounit leaves the budget, the checks fail where an active topounit's downhill
    neighbour has weight 0 (dead outlet, dead middle) and nowhere else; with floating-point traps,
    the routines stop."""
    dom = pt.Domain([Gridcell.hillslope(t, Soil(13), name=n) for n, t in INACTIVE_TOPOUNIT_HILLSLOPES.items()],
                    nsets=2, reference=CASE, downhill=downhill)
    spec, case, rules, common, keep = _spec(path, dom, max_examples=6)
    _storm(spec, rules, common, keep, steps, steps // 4)
    _storm(spec, rules, common, keep, steps, steps // 4, name="storm trapped", trap_fpe=True,
           comment="the same storm with floating-point exceptions trapped: ELM's division by the weight-0 "
                   "topounit's weight stops step 1")
    storm = ["storm"]
    spec.property("no active topounit sends water to an inactive one",
                  "implies(col_wf%qflx_to_downhill(c) > 0.0, .not. downhill_inactive(t))", scenarios=storm)
    spec.property(*STORE_FINITE, over="t", scenarios=storm)
    spec.property(*STORE_BALANCE, over="g", scenarios=storm)
    spec.property("the routines finish every step", ".true.", over="g", scenarios=["storm trapped"])
    spec.relation(*WATER_CONSERVED, namelist=IM2_OFF, over="g", scenarios=storm)
    return spec


# -- hypothesis: a topounit whose hydrology columns all weigh 0 releases its store as 0/0

def lake_topounit(elevation: float, weight: float) -> Topounit:
    """A lake topounit as ELM builds it: the lake, and the virtual natural-vegetation landunit of
    weight 0 that ELM keeps active (subgridWeightsMod is_active_l), whose column is the topounit's
    only hydrology column."""
    return Topounit((Lake(weight=1.0), Soil(13, weight=0.0)), weight=weight, elevation=elevation)


def no_weighted_hydrology_column(path: Path, steps: int = 12, downhill: str = "elm") -> pt.Spec:
    """A lake topounit at the bottom of a hillslope and one alone, next to an all-soil control. If a
    topounit whose hydrology columns all weigh 0 releases its store as 0/0 (SurfaceRunoff divides by
    their summed weight), the flux and the store go NaN at the lake topounits only."""
    cells = [Gridcell.hillslope(HILLSLOPES["lowest first"], Soil(13), name="lowest first"),
             Gridcell(name="lake low", topounits=(lake_topounit(50.0, .4),
                                                  *Gridcell.hillslope([(150, .3), (300, .3)], Soil(13)).topounits)),
             Gridcell(name="lake alone", topounits=(lake_topounit(100.0, 1.0),))]
    dom = pt.Domain(cells, nsets=2, reference=CASE, downhill=downhill)
    spec, case, rules, common, keep = _spec(path, dom, max_examples=6)
    _storm(spec, rules, common, keep, steps, steps // 2)
    spec.property(*FLUX_FINITE)
    spec.property(*STORE_FINITE, over="t")
    spec.property("a topounit with water in store releases some of it",
                  "implies(prev%top_ws%from_uphill(t) > 0.0, child_sum(col_wf%qflx_from_uphill, 't') > 0.0)",
                  over="t", steps=[2, steps])
    spec.relation(*WATER_CONSERVED, namelist=IM2_OFF, over="g")
    return spec


SPECS = {f.__name__: f for f in (random_hillslopes, downhill_follows_listing_order,
                                 water_sent_to_inactive_topounit, no_weighted_hydrology_column)}


if __name__ == "__main__":
    for name in sys.argv[1:] or SPECS:
        spec = SPECS[name](HERE / f"{name}.proptest.toml")
        spec.to_toml(spec.path)
        print("written", spec.path)
