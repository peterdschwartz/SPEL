# IM2 hillslope routing: three bugs found and diagnosed with `spel proptest`

ELM's IM2 hillslope hydrology (`use_IM2_hillslope_hydrology`, default off) routes part of each
column's surface runoff to the next lower topounit of its gridcell. A property test of that routing,
on randomly drawn multi-step domains, fails in three different ways. Each corresponds to a bug in ELM
(E3SM master `b70b675a7b` and later; still in `b39003b`):

1. **The listing order decides the downhill neighbour** (`set_topounit`). On a hillslope whose
   lowest topounit is not listed first, the outlet sends its runoff to itself or uphill.
2. **Water sent to an inactive (weight-0) topounit leaves the water budget.** The neighbour search
   ignores activity, and the send divides by the target's weight.
3. **A topounit whose hydrology columns all weigh 0 releases its store as 0/0.** That's a lake or
   wetland topounit with only ELM's virtual weight-0 soil column. The store is NaN in 1527
   topounits of a real run.

`hillslope_im2.ipynb` walks through how the property test finds them, how its shrunk failures
suggest these hypotheses, how a targeted spec confirms each one, and how they trace to the code.
This README is the report. (Earlier notes and commits call the bugs H1, H2 and H3.)

## How IM2 routing works

| Step | Where | What |
|---|---|---|
| neighbours | `main/initGridCellsMod.F90::set_topounit` (initialisation) | `top_pp%downhill_ti(t)` = the highest topounit below t (−1: none), from `top_pp%elevation` (surface data `TopounitAveElv`) |
| send | `biogeophys/HydrologyDrainageMod.F90::HydrologyDrainage` (~l.294–322) | if `downhill_ti(t) /= -1`: `frac_to_downhill` (= 1) of `qflx_surf`, `qflx_drain_perched`, `qflx_h2osfc_surf` → `qflx_to_downhill(c)`; `top_ws%from_uphill(downhill_t) += qflx_to_downhill(c)*dtime * col_pp%wtgcell(c)/top_pp%wtgcell(downhill_t)` |
| receive | `biogeophys/SoilHydrologyMod.F90::SurfaceRunoff` (~l.262–285, in `HydrologyNoDrainage`) | t's hydrology columns get `qflx_from_uphill(c) = wttopounit(c)/uphill_wt(t) * frac_from_uphill * from_uphill(t)/dtime` (`frac_from_uphill` = 0.5) |
| drain the store | `biogeophys/BalanceCheckMod.F90::ColWaterBalanceCheck` (~l.321) | `from_uphill(t) -= qflx_from_uphill(c)*dtime`, floored at 0 |

`frac_to_downhill` and `frac_from_uphill` are parameters (`main/elm_varcon.F90:240,242`). The feature
is switched on by the testmod `elm/topounit_im2` (`ELM_BLDNML_OPTS = "-bgc sp -topounit"`, the
0.5° `half_degree_merge_surfdata_0.5x0.5_simyr2000_c190418.with_aveDTB.20201222.nc`), used by
`SMS.r05_r05.IELM.elm-topounit_im2` in `e3sm_land_developer`.

## The listing order decides the downhill neighbour

**Cause.** In `set_topounit` (`initGridCellsMod.F90:316`), `dn_index = -1` is set once per gridcell,
*before* the loop over its topounits (l.317–328), and never reset. A topounit with no lower
neighbour, the lowest, keeps the previous topounit's answer. `min_index` (l.308–312) is computed and
never used.

| Elevations (listing order) | ELM's `downhill_ti` | intended |
|---|---|---|
| 50, 150, 300 (lowest first) | −1, 1, 2 | the same |
| 300, 150, 50 | 2, 3, **3** (itself) | 2, 3, −1 |
| 100, 50, 300 | 2, **2** (itself), 1 | 2, −1, 1 |
| 300, 200, 50, 150 | 2, 4, **4** (uphill: 3 ↔ 4 loop) | 2, 4, −1, 3 |
| 100, 50, 50 (tie) | 3, **3**, **3** | 3, −1, −1 |

(numbers within the gridcell; ELM's are global topounit indices)

**In real data.**
- **The 0.5° testmod dataset** (720×360, 12 topounits): 49415 gridcells have several topounits.
  - 49398 are listed lowest first, 7 highest first, and 10 in another order but still with the
    lowest first.
  - ELM's neighbours differ from the intended ones in the 7 listed highest first, each with a
    self-loop at the outlet.
  - All 7 are below sea level (Salton Sea, Caspian coast):

| lat | lon | TopounitAveElv | TopounitFracArea |
|---|---|---|---|
| 33.25 | -115.75 | 14.1, -50.7, -163.0 | 0.9, 0.05, 0.05 |
| 37.75 | 49.25 | 1.8, -20.7, -64.9, -110.7 | 0.701, 0.15, 0.1, 0.05 |
| 39.75 | 48.25 | 47.8, -613.5 | 0.951, 0.049 |
| 40.25 | 52.75 | 117.9, -2431.4 | 0.95, 0.05 |
| 40.75 | 49.75 | -8.6, -141.2 | 0.95, 0.05 |
| 40.75 | 53.75 | 212.9, -1173.1 | 0.851, 0.149 |
| 41.25 | 49.25 | 1.3, -14.6, -32.6, -50.0 | 0.7, 0.2, 0.05, 0.05 |

- **The NGEE Arctic CAVM dataset** (`topounit_surfdata_0.5x0.5_simyr1850.c20220204_cavm1d.nc`,
  10266 gridcells): its 5529 multi-topounit gridcells are all listed lowest first, so the bug is
  latent there. topoPerGrid histogram: 1: 4410, 2: 1361, 3: 942, 4: 773, 5: 680, 6: 607, 7: 473,
  8: 339, 9: 182, 10: 120, 11: 45, 12: 7.
- `h1_scan.py SURFDATA` does this scan.

**In ELM (`hs-im2`).**
- **The run:** IELM r05_r05, `elm/topounit_im2`, cold start 0001-01-01, 5 days, Intel, 256 tasks,
  with a SPEL subset capture of the self-looping topounits (`hillslope_validate.txt`). The 7
  gridcells landed on 7 ranks, and the capture merges them: 19 topounits, 323 columns.
- **ELM's own `top_pp%downhill_ti`** is `[2,3,3, 5,6,7,7, 9,9, 11,11, 13,13, 15,15, 17,18,19,19]`:
  self-loops at the 7 outlets.
- **Over the 5 days**, those outlets send their positive surface runoff to their own store and keep
  0.000 mm of it. Their store fills (up to 0.821 kg/m² at one outlet) and drains back. Control
  outlets (`downhill_ti = -1`) keep their runoff. The amounts are small: a January start in dry
  basins.

**The fix** (branch `peterdschwartz/lnd/fix-im2-downhill-neighbor`, 6ee4269ec5, not pushed): move
`dn_index = -1` into the `t1` loop. I compared the same 5-day run with the fix using
`tools/compare_restarts.py`:
- **Where it differs:** 42 restart variables differ, only in 6 of the 7 gridcells; the Salton Sea
  outlet had no surface runoff in those days.
- **`FROM_UPHILL`** differs exactly at each outlet (e.g. 0.0014 → 0.0011, 0.0102 → 0.0097 kg/m²).
  Soil water and temperatures follow, and so does the MOSART restart.

**With proptest** (`downhill_follows_listing_order`):
- **The setup:** the same hillslope listed lowest first and lowest last, a loop, the Caspian
  gridcell and a one-topounit gridcell, over 48 half-hour steps of storm, steady rain and recession.
- **What holds:** every store invariant and every IM2-on/off identity, in every scenario (the store
  stays finite, gains what is sent, releases half per step; a one-topounit gridcell and the topmost
  soil are identical with IM2 off; water is conserved; IM2 never gets water to the river sooner).
- **What fails:** only two checks, at the outlets of the hillslopes not listed lowest first:
  - "no topounit sends water to itself";
  - "the outlet passes on at least what it does without IM2", which also fails at the loop's lowest
    topounit.
- **The effect over 20 mirrored storms** (the same inputs on both listings, 40 twin pairs; ratio
  of the day's river runoff to IM2 off):

| | mean | min | step by which half had arrived |
|---|---|---|---|
| IM2 off | 1 | 1 | 8.6 |
| listed lowest first | 0.776 | 0.175 | 9.7 |
| listed lowest last | 0.627 | 3e-6 | 17.5 |

  The totals include subsurface drainage, which IM2 does not route, so these ratios understate the
  effect on surface runoff.

## Water sent to an inactive topounit leaves the water budget

**Cause.**
- `set_topounit`'s search runs over every topounit, weight-0 ones included.
- `HydrologyDrainage` divides the send by `top_pp%wtgcell(downhill_t)`, which is 0 for an inactive
  target. Its store becomes `inf`, and gridcell averages skip weight-0 topounits, so the water is
  gone from the budget.
- **On a dry step** the same division gives `0 * inf = NaN`. ColWaterBalanceCheck's
  `max(0, from_uphill - ...)` then depends on the compiler: `MAX(0, NaN)` is 0 with gfortran 11.2
  and NaN with ifx 2025.2 (-O0 and -O2). Intel builds keep a NaN store for good; gfortran builds
  flush it to 0.
- **Neither real dataset has a weight-0 topounit**, so the bug is latent there.

**With proptest** (`water_sent_to_inactive_topounit`):
- **The setup:** hillslopes listed lowest first, so the listing order plays no part, with the
  outlet, middle or top topounit of weight 0, plus an intact control. 24-step storm.
- **What fails:** only in "dead outlet" and "dead middle", at step 1:
  - "no active topounit sends water to an inactive one";
  - the store stays finite (`inf`);
  - the stores gain what is sent;
  - water conservation with IM2 off.
- **What passes:** "dead top" and the control.
- **With `trap_fpe`,** step 1 stops with SIGFPE "floating-point divide by zero".

**Fix direction:** skip inactive topounits in set_topounit's neighbour search, or guard the send
with `top_pp%active(downhill_t)`.

## A topounit whose hydrology columns weigh 0 releases 0/0

**Cause.**
- ELM keeps a virtual natural-vegetation landunit of weight 0 active on every active topounit that
  is not all plain glacier (`subgridWeightsMod.F90` is_active_l, l.380). Its column is in
  `hydrologyc`.
- On a topounit whose other landunits are wetland, lake or glacier, that column is the only
  hydrology column. SurfaceRunoff's `top_pp%uphill_wt(t)`, the sum of the hydrology columns'
  `wttopounit`, is then 0, so `qflx_from_uphill(c) = (0/0) * ...` (`SoilHydrologyMod.F90:283`) is
  NaN every step, even with an empty store and no uphill neighbour.
- The store's drain then gives `max(0, store - NaN)`. That's NaN with ifx. With gfortran it's 0,
  and the store and any water routed into it are silently deleted every step.
- The NaN also reaches the virtual column's `qflx_top_soil`, which gridcell averages hide because
  its weight is 0.

**In ELM.** The restart of `hs-im2` (`hs-im2.elm.r.0001-01-06-00000.nc`) has `FROM_UPHILL` NaN in
1527 of 277539 topounits (1314 gridcells, 85S–84N), at the same places with and without the
listing-order fix. They are exactly the topounits whose hydrology columns' `wttopounit` sum to 0,
both ways, with no exception:
- 1304 wetland;
- 158 glacier + lake;
- 55 lake;
- 10 lake + wetland.

The 37596 topounits without any hydrology column (all plain glacier: no virtual landunit) are finite.

**With proptest** (`no_weighted_hydrology_column`):
- **The setup:** a lake topounit (lake + the virtual weight-0 soil landunit) at the bottom of a
  hillslope, a lake topounit alone, and an all-soil control. 12-step storm.
- **What fails:** at step 1, only at the lake topounits:
  - the flux from uphill is finite on every hydrology column;
  - the store stays finite;
  - water conservation with IM2 off.
- **What passes:** the control.

**Fix direction** (a science choice for the IM2 developers):
- guard the release with `uphill_wt(t) > 0`;
- then decide where water routed into a topounit with no positive-weight hydrology column goes
  (into the lake or wetland, or straight to runoff), instead of keeping it in a store nothing
  releases.

## How the property test finds them

`random_hillslopes` states what IM2 routing must do on any hillslope, without knowing a bug:
- the flux from uphill is finite on every hydrology column;
- the store is finite and non-negative;
- no topounit sends water to itself;
- water is conserved, comparing the same example with IM2 off.

It runs these over a 12-step storm on 4 drawn domains: gridcells of 1–3 topounits of soil, wetland,
lake and glacier, at random elevations and weights, some weight 0.

With seed 0 (about 5 min on a laptop, most of it shrinking), 3 of the 4 domains fail. Each failure
is shrunk to a smaller domain that fails the same way (the same failure signature):
- the NaN checks → `soil(0)+lake` / `soil(0)+wetland`: the 0/0 release;
- the self-send → `soil | soil`, lowest listed last: the listing order;
- conservation with finite values (domain 4): the domain stays at two gridcells, because its four
  checks fail in three different ways and a candidate must keep all of them. Next to the other two
  patterns, it shows an active topounit sending to a weight-0 neighbour: the inactive target.

Run with `[domain] downhill = "intended"` (set_topounit fixed), exactly the self-send failures
disappear, and every other failure stays with the same shrunk domain and signature: the other two are
independent bugs. The whole notebook runs in about 14 minutes on a laptop.

The notebook follows this path. Building it needed several proptest features:
- multi-step runs, with the step boundary declared: `forcing`, `between`, `keep`, `carry`;
- ELM's `is_active` rules, including the virtual landunit;
- drawn domains, with signature-matched shrinking;
- `child_*` reductions over a filter;
- the `downhill` rule.

## Reproducing

- **The case** `hs_season`: HydrologyNoDrainage + HydrologyDrainage in elm_drv's order, made with
  `spel create -s hydrologynodrainage hydrologydrainage -c hs_season` from the `next` worktree of
  E3SM master. Its reference, `unit-tests/input-data/hs_season`, is the union of the
  `hs_bug_nodrain` and `hs_bug` captures of `hs-im2` (`hillslope_validate.txt`).
- **Elsewhere:** a case extracted on another machine is used with the srcroot recorded by
  `spel archive extract --set-srcroot`; the step coupling needs that checkout's `spel analyze`.
- **The specs:** `python -m spel.proptest.experiments.hillslope_im2.specs` writes the four specs
  next to `specs.py`, and `spel proptest run <name>.proptest.toml` runs one. The notebook builds
  them through the same functions.
- **Timing on a laptop:**
  - `random_hillslopes`: ~5 min;
  - `downhill_follows_listing_order`'s checks: ~2 min;
  - Chrysalis login nodes: about 2–3× longer.

## Files

| File | |
|---|---|
| `hillslope_im2.ipynb` | the walkthrough: property test → shrunk cases → hypotheses → targeted specs → code → proof |
| `specs.py` | every spec: `random_hillslopes`, `downhill_follows_listing_order`, `water_sent_to_inactive_topounit`, `no_weighted_hydrology_column` |
| `analysis.py` | the notebook's analyses as named functions |
| `*.proptest.toml` | the specs as written by `specs.py` |
| `h1_scan.py` | scans a surface dataset for gridcells whose listing order changes the downhill neighbour |
| `hillslope_validate.txt` | the `spel validate` list of the `hs-im2` captures |
| `../tools/compare_restarts.py` | where two ELM restart files differ, by gridcell |
