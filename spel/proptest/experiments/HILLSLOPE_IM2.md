# Experiment: ELM's IM2 hillslope hydrology (topounits) — confirm the `downhill_ti` bug, then test the feature with SPEL

Written 2026-11-11 on the laptop (E3SM master b39003b) for an agent on a machine with topounit
surface data. Read `AGENTS.md` first (machine notes, the parser rule, the Don'ts), then this.
The proptest hierarchy/API this relies on is described in `spel/proptest/SPEC.md` (`[domain]`,
"Python API") and in AGENTS.md ("Property-based tests").

## Background

`use_IM2_hillslope_hydrology` (elm_varctl, default `.false.`) routes part of each column's surface
runoff to the next lower topounit of the same gridcell:

| Step | Where | What |
|---|---|---|
| neighbours | `main/initGridCellsMod.F90::set_topounit` (init, not in elm_drv) | `top_pp%downhill_ti(t)` = the highest topounit below t (−1: none), from `top_pp%elevation` (surface data `TopounitAveElv`) |
| send | `biogeophys/HydrologyDrainageMod.F90::HydrologyDrainage` (~l.294–322) | if `downhill_ti(t) /= -1`: `frac_to_downhill` of `qflx_surf`, `qflx_drain_perched`, `qflx_h2osfc_surf` -> `qflx_to_downhill(c)`, and `top_ws%from_uphill(downhill_t) += qflx_to_downhill(c)*dtime * col_pp%wtgcell(c)/top_pp%wtgcell(downhill_t)` |
| receive | `biogeophys/SoilHydrologyMod.F90::SurfaceRunoff` (~l.262–285, called from `HydrologyNoDrainage`) | hydrology columns (soil, crop, pervious road) of t get `qflx_from_uphill(c) = wttopounit(c)/uphill_wt(t) * frac_from_uphill * from_uphill(t)/dtime` (`frac_from_uphill` = 0.5) |
| drain the store | `biogeophys/BalanceCheckMod.F90::ColWaterBalanceCheck` (~l.321) | `from_uphill(t) -= qflx_from_uphill(c)*dtime`, floored at 0; the column balance counts `qflx_from_uphill` in and `qflx_to_downhill` out |

Surface data: dimension `topounit`, `topoPerGrid`, `TopounitFracArea`, `TopounitAveElv`
(`main/surfrdMod.F90` ~l.1360–1420, `main/topounit_varcon.F90`). `max_topounits` = the
`topounit` dimension; a topounit is active when its weight > 0 (`set_topounit`).

## Hypotheses

**H1 (found by reading the code; not yet seen in a run).** In `set_topounit`, `dn_index = -1`
is set once before the `t1` loop and never reset. A topounit with no lower neighbour (the lowest)
keeps the previous topounit's result:

| Elevations (topounit order) | ELM `downhill_ti` | intended |
|---|---|---|
| 50, 150, 300 (lowest first) | −1, 1, 2 | same: no bug |
| 300, 150, 50 | 2, 3, **3** (self) | 2, 3, −1 |
| 100, 50, 300 | 2, **2** (self), 1 | 2, −1, 1 |
| 300, 200, 50, 150 | 2, 4, **4** (uphill; 3 and 4 form a cycle) | 2, 4, −1, 3 |
| 100, 50, 50 (tie) | 3, **3**, **3** | 3, −1, −1 |

Indices are 1-based within the gridcell here (ELM's are global topounit indices). Whether real
datasets trigger it depends on the order of topounits in the surface data. Consequence: the lowest
topounit (the outlet) sends `frac_to_downhill` of its runoff back into itself, or uphill; the
column water balance still closes, so nothing aborts. Fix: move `dn_index = -1` inside the `t1`
loop (`min_index` is also unused).

**H2.** The neighbour search ignores activity. If the downhill topounit has weight 0, the send
divides by `top_pp%wtgcell(downhill_t) = 0`: Inf, or NaN (`0*Inf`) on every step even without
runoff.

**H3.** A downhill topounit with no hydrology column (only lake / impervious urban) receives
`from_uphill` and never releases it: a growing store (a sink for the runoff budget).

`spel/proptest/hierarchy.py::Layout.downhill` reproduces `set_topounit` including the quirk
(test `test_topounits_numbers_activity_and_downhill`).

## Ground rules

- Machine notes: find which machine this is (AGENTS.md "Environment") and edit only its section.
  `spel machine check` before building. Slurm jobs and batch submissions need the user's OK.
- Do not modify uncommitted work in an E3SM checkout. To instrument a checkout with local
  changes, use a scratch `git worktree` (AGENTS.md, "Gridcell subsets", last bullet).
- Don't change ELM to fix H1 in the user's branches; prototype a fix only in a case or a worktree,
  and report.
- Record commands and results as you go in `spel/proptest/experiments/HILLSLOPE_IM2_RESULTS.md`
  (create it): what ran, on which commit, the numbers. Quote real outputs only.

## Phase 0 — sync and sanity (≈15 min)

1. `git pull` on branch `property-based-testing`; `pip install -e '.[proptest]'` if needed.
2. `DJANGO_DEBUG=true python -m pytest -q spel/proptest` (the canflux end-to-end tests are skipped
   or fail without a fresh `canflux` case; everything else must pass).
3. `spel config` / `.spel.env`: the srcroot is the E3SM checkout to use. Check that it has the
   code above (`grep -n "dn_index" components/elm/src/main/initGridCellsMod.F90`), and note its
   commit. If upstream already fixed H1, say so and continue with H2/H3.

## Phase 1 — H1 on real data, without running ELM (≈30 min)

1. Find surface datasets with topounits: `ncdump -h <surfdata> | grep -E "topounit|topoPerGrid|TopounitAveElv"`.
   Candidates: NGEE Arctic site data (Kougarok, Council, Teller), any `*topounit*` surfdata under
   the machine's inputdata, or ask the user for the path.
2. With netCDF4 in Python, for each gridcell with `topoPerGrid > 1`: take `TopounitAveElv[:n]` and
   `TopounitFracArea[:n]`, compute ELM's `downhill_ti` (copy the Fortran loop exactly, or build a
   `hierarchy.Layout` of `Gridcell(topounits=...)` and call `downhill()`) and the intended one.
   Report: number of gridcells, how many have the lowest topounit not first, how many get a
   self-loop / uphill pointer / cycle, how many have an inactive (weight 0) topounit that is
   someone's downhill target (H2).
3. If no gridcell triggers H1 (lowest always first), the bug is latent for that dataset: record
   the ordering convention and move on to H2/H3.

## Phase 2 — confirm in ELM with a capture (needs a CIME run: ask the user first)

1. A short case (2–5 days) on a topounit surface dataset with `use_IM2_hillslope_hydrology = .true.`
   in `user_nl_elm`, through `e3sm_casegen.sh` (the machine's preset/user mods) so `spel validate`
   can drive it. Ask the user which compset/resolution matches the dataset.
2. A validate list, e.g.:
   ```
   hs_drain: hydrologydrainage
   hs_runoff: hydrologynodrainage
   ```
   (`HydrologyDrainage` sends; `SurfaceRunoff` is inside `HydrologyNoDrainage`.) Check first with
   `spel analyze` + the `.spel-cache` that these routines are analyzed and IM2's branches are in
   the record (the namelist-gated `if (use_IM2_hillslope_hydrology)` must show up as a gate:
   `spel proptest show <case>` lists `use_im2_hillslope_hydrology` among the options once a case
   exists).
3. Run `spel validate --list ... --case-args "..."` (see README); both cases must PASS bit-for-bit.
4. From `unit-tests/input-data/hs_drain/spel-inputs0001.nc`: `top_pp__downhill_ti`,
   `top_pp__elevation`, `top_pp__wtgcell`, `top_pp__active`, `top_ws__from_uphill`,
   `col_wf__qflx_to_downhill` (whichever the capture holds). Compare `downhill_ti` with the
   intended neighbours; check `from_uphill` for NaN/Inf (H2) and growth on topounits without
   hydrology columns (H3) over the captured steps.
5. Optional: prototype the one-line fix in the case's copy of `initGridCellsMod.F90` inside a
   worktree run and compare outlet runoff (`QRUNOFF`/`QOVER` history of the lowest topounit)
   between buggy and fixed runs. This is the measurement of impact the user will want.

## Phase 3 — property tests of the feature (no CIME run needed beyond Phase 2's capture)

Use one of Phase 2's cases (or any capture of these routines from master) as `reference` for
constants and parameters; the topounits come from the declared domain.

```python
# a script with `if __name__ == "__main__":` (the worker is a spawned process)
from spel.proptest import api as pt
from spel.proptest.api import Gridcell, Lake, Soil, Topounit, Urban

def hill(name, *tops):          # tops: (elevation, weight, landunits)
    return Gridcell(name=name, topounits=tuple(Topounit(lus, weight=w, elevation=e) for e, w, lus in tops))

S = (Soil(1, 13),)
case = pt.Case("hs_drain", reference="hs_drain")
dom = pt.Domain([
    hill("lowest first", (50, .3, S), (150, .3, S), (300, .4, S)),
    hill("lowest last",  (300, .4, S), (150, .3, S), (50, .3, S)),
    hill("cycle",        (300, .25, S), (200, .25, S), (50, .25, S), (150, .25, S)),
    hill("inactive low", (300, .5, S), (150, .5, S), (50, 0.0, S)),          # H2
    hill("lake low",     (300, .6, S), (50, .4, (Lake(),))),                 # H3
], nsets=2, reference="hs_drain")
spec = pt.Spec(case, dom, namelist={"use_im2_hillslope_hydrology": True})
```

Checks to write (expressions use the case's canonical names; `spel proptest show` lists them):

- **downhill only**: where `qflx_to_downhill(c) > 0`, the target is lower:
  `top_pp%elevation(top_pp%downhill_ti(t)) < top_pp%elevation(t)`. Expected to FAIL on
  "lowest last" and "cycle" (H1).
- **routing only moves water**: a relation with `namelist = { use_im2_hillslope_hydrology = false }`
  as follow-up: `qflx_runoff(off) = qflx_runoff(on) + qflx_to_downhill(on)` per column (close,
  rtol 1e-12); on the lowest topounit `qflx_to_downhill = 0` (fails under H1).
- **mass**: per gridcell, Σ_c `qflx_to_downhill·dtime·col_pp%wtgcell` = Δ `from_uphill·top_pp%wtgcell`
  summed over topounits (use `Result.frame("t", aggregate=...)` and `input%` for the before value).
- **finite**: `finite(top_ws%from_uphill(t))` with `trap_fpe = true` on "inactive low" (H2).
- **released**: on a topounit with `from_uphill > 0`, Σ `qflx_from_uphill·dtime·wttopounit` over its
  hydrology columns = `frac_from_uphill·from_uphill` (fails on "lake low": H3; needs
  `hydrologynodrainage` as the routine).

Inspect before trusting: `spec.view().table("t")` (elevations, weights, active), `s.built().check()`
(must be empty; extend `domain.EXPECTED_FILTERS` / `topology_value` if the case reads a topology
field the domain has no rule for: the build stops and names it), `s.execute(...).frame("c")`.

Likely SPEL work this phase needs (do it on this branch, with tests, keeping AGENTS.md/SPEC.md in
step):

1. **Declared neighbours.** `Topounit(downhill=...)` (a topounit number within the gridcell, or
   `None` for "no neighbour") overriding `Layout.downhill()`, so the same routines run on ELM's
   topology and on the intended one side by side (relation or two scenarios).
2. **New topology fields** the hydrology routines read (e.g. `col_pp%wttopounit`,
   `top_pp%slope`/`aspect`, `uphill_wt` if the inventory calls it topology): add rules in
   `domain.topology_value` (from the layout or declared values), never fill values.
3. **Multi-step runs** (for H3 and the outlet delay: water sent in step n returns in n+1): an
   option to call compute N times carrying inout state, with time scalars advanced. This is the
   main missing proptest feature (AGENTS.md "Not done"); design it before coding and ask the user.

## Deliverables

- `HILLSLOPE_IM2_RESULTS.md`: Phase 1 table per dataset; Phase 2 captured `downhill_ti` vs intended
  and any NaN/growth; Phase 3 checks with their outcomes and the minimal failing examples
  (`unit-tests/proptest-data/<case>/failures/`).
- The spec (`<case>.proptest.toml`, written with `spec.to_toml`) and the script that builds it.
- If H1 is confirmed on real data: a short note for an E3SM issue (the loop, the table above, the
  dataset and gridcells affected, the one-line fix, and the measured runoff difference if Phase 2.5
  ran).
