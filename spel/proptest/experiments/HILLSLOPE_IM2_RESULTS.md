# Results: IM2 hillslope hydrology (see HILLSLOPE_IM2.md)

## Phase 0 (Chrysalis, 2026-11-11)

- Checkout: worktree `/lcrc/globalscratch/ac.schwartzpd/next`, E3SM master `b70b675a7b`.
- H1 is present: `components/elm/src/main/initGridCellsMod.F90:316` sets `dn_index = -1` once, before
  the `t1` loop (l.317-328); `min_index` (l.308-312) is computed and never used.
- The neighbour search runs over `begt..endt` with `ntopos = topoPerGrid`, including weight-0 topounits.
- `frac_to_downhill = 1.0` and `frac_from_uphill = 0.5` (`main/elm_varcon.F90:240,242`, parameters):
  a topounit with a downhill neighbour sends *all* positive `qflx_surf`, `qflx_drain_perched` and
  `qflx_h2osfc_surf`.
- The feature is switched on by the testmod `elm/topounit_im2`
  (`components/elm/cime_config/testdefs/testmods_dirs/elm/topounit_im2`): `ELM_BLDNML_OPTS = "-bgc sp
  -topounit"` and `user_nl_elm` with `use_IM2_hillslope_hydrology = .true.` and
  `fsurdat = half_degree_merge_surfdata_0.5x0.5_simyr2000_c190418.with_aveDTB.20201222.nc`.
  It is used by `SMS.r05_r05.IELM.elm-topounit_im2` in the `e3sm_land_developer` suite
  (`cime_config/tests.py:100`).

## Phase 1: H1/H2 on that surface dataset (no ELM run)

Dataset: `/lcrc/group/e3sm/data/inputdata/lnd/clm2/surfdata/half_degree_merge_surfdata_0.5x0.5_simyr2000_c190418.with_aveDTB.20201222.nc`
(720x360, `topounit` = 12). Script: ELM's loop copied exactly vs. "highest topounit strictly below"
over `TopounitAveElv[:topoPerGrid]`.

| | count |
|---|---|
| gridcells with `topoPerGrid > 1` | 49415 |
| elevations strictly ascending (lowest first) | 49398 |
| strictly descending | 7 |
| neither (lowest still first) | 10 |
| ELM `downhill_ti` differs from intended (H1) | 7, all a self-loop on the lowest topounit |
| uphill pointer / cycle | 0 |
| inactive (weight 0) topounit at all / as a downhill target (H2) | 0 / 0 |

The 7 H1 gridcells are all below sea level (Salton Sea and Caspian coast), where the topounits are
ordered from highest to lowest:

| lat | lon | TopounitAveElv | TopounitFracArea |
|---|---|---|---|
| 33.25 | -115.75 | 14.1, -50.7, -163.0 | 0.9, 0.05, 0.05 |
| 37.75 | 49.25 | 1.8, -20.7, -64.9, -110.7 | 0.701, 0.15, 0.1, 0.05 |
| 39.75 | 48.25 | 47.8, -613.5 | 0.951, 0.049 |
| 40.25 | 52.75 | 117.9, -2431.4 | 0.95, 0.05 |
| 40.75 | 49.75 | -8.6, -141.2 | 0.95, 0.05 |
| 40.75 | 53.75 | 212.9, -1173.1 | 0.851, 0.149 |
| 41.25 | 49.25 | 1.3, -14.6, -32.6, -50.0 | 0.7, 0.2, 0.05, 0.05 |

Consequence (from the code, not yet seen in a run): in these 7 gridcells the outlet topounit sends
all its positive surface/perched/h2osfc runoff into its own `from_uphill` store instead of to the
river, and gets half the store back each step through `SurfaceRunoff`. Its `QOVER` to MOSART is
zero; the water recirculates until it infiltrates or evaporates. H2 is latent for this dataset
(no weight-0 topounits). Whether the ELM run uses these gridcells depends on the r05 land mask.

### NGEE Arctic dataset (2026-11-11)

From the user's `topos.tar.gz` (a plain tar), extracted to `/lcrc/globalscratch/ac.schwartzpd/ngee_topos/`:
`topounit_surfdata_0.5x0.5_simyr1850.c20220204_cavm1d.nc` (10266 gridcells on a 1-D `gridcell`
dimension, `topounit` = 12) and `domain.lnd.r05_RRSwISC6to18E3r5.240328_cavm1d_toposafe.nc` (same
gridcell order, 9939 land). Script: `h1_scan2.py` (ELM's loop copied exactly, any grid layout).

| | count |
|---|---|
| gridcells with `topoPerGrid > 1` | 5529 |
| elevations strictly ascending (lowest first) | 5529 |
| equal elevations | 0 |
| ELM `downhill_ti` differs from intended (H1) | 0 |
| weight-0 topounits / weight-0 downhill targets (H2) | 0 / 0 |
| `topoPerGrid = 0` | 327, all `mask = 0` in the domain (not run) |

H1 is latent for this dataset: every multi-topounit gridcell lists its lowest topounit first, so the
missing reset never matters. topoPerGrid histogram: 1: 4410, 2: 1361, 3: 942, 4: 773, 5: 680, 6: 607,
7: 473, 8: 339, 9: 182, 10: 120, 11: 45, 12: 7.

## Phase 2: preparation (2026-11-11)

- Subset captures of gridcells with several topounits are tested (`test_subset.py`,
  `TOPO_ROUNDTRIP`): `downhill_ti` (self-loop, -1), `landunit_indices` (topounit = 2nd dim, ispval),
  `topi/topf`, `lndi/lndf`, `*%topounit` renumber correctly, bit-for-bit with the offline subsetter.
- Subset captures are now written per MPI rank and merged (`spel merge-subset`, run by
  `spel validate`), so one run captures all 7 H1 gridcells wherever they land. Planned validate list:
  `hs_bug[topounit: top_pp%downhill_ti(t) == t]: hydrologydrainage` and
  `hs_ok[gridcell: grc_pp%ntopounits(g) > 2; max=20]: hydrologydrainage`.

## Phase 2: ELM run and capture (2026-11-11)

Case `hs-im2` (`spel validate --run-name hs-im2 --freq 1`, list `hillslope_validate.txt`): IELM r05_r05,
testmod `elm/topounit_im2`, cold start 0001-01-01, 5 days, DEBUG=FALSE, Intel, 256 tasks x 1 thread
(Slurm 1326032; 1326031 ran too but its merge failed, see below). Commit: `next` b70b675a7b.

- Per-rank subset capture: the 7 H1 gridcells are on 7 different ranks (122, 195, 212, 215, 234, 235, 254);
  merged: 7 gridcells, 19 topounits, 95 landunits, 323 columns, 627 patches. gindex 177249 184059 186937
  187666 188380 188388 189099 = exactly the Phase 1 table (same lat/lon/weights).
- ELM's captured `top_pp%downhill_ti` = `[2,3,3, 5,6,7,7, 9,9, 11,11, 13,13, 15,15, 17,18,19,19]`:
  self-loops on the 7 outlets (merged topounits 3, 7, 9, 11, 13, 15, 19). **H1 confirmed in ELM.**
- Over the 5 days (241 captured steps), the buggy outlets send their positive surface runoff to
  themselves (`qflx_to_downhill` > 0 at 6 of 7) and keep **0.000 mm** surface runoff; their
  `from_uphill` store fills (max 0.821 kg/m2 at t9) and drains back. Control outlets (`hs_ok`,
  `downhill_ti = -1`) send nothing and keep their surface runoff. Amounts are small (fractions of a mm):
  a January cold start in dry basins; an impact measurement needs a longer, wetter period.
- Unit tests (gfortran, ACC=OFF; ELM built with Intel, so not bit-for-bit):
  - `hs_bug` (hydrologydrainage): PASS within 1e-10.
  - `hs_ok` (hydrologydrainage): FAIL on `qflx_glcice_diag`, `qflx_glcice_frz_diag` only: ELM keeps
    spval at inactive (weight-0) columns, the unit test writes 0 (the routine zeroes every column;
    something outside it resets the history diagnostics). `hs_bug` has the same data but was reported
    PASS: the comparator is inconsistent here. Not related to IM2; open.
  - `hs_bug_nodrain` (hydrologynodrainage): FAIL on 12 aerosol deposition/mass arrays, RMS ~1e-15
    (relative differences on near-zero values; likely Intel vs gfortran). Not related to IM2; open.
  - All hillslope variables (`qflx_to_downhill`, `from_uphill`, `qflx_surf`, `qflx_runoff`) agree.
- SPEL fixes this run needed: per-level parents from partial index fields (merge), cross-module
  module-array allocations (`f_surf`), `nb:ne` = gridcell (`ldomain`), `--allow-differ` for
  `nlvic` (uninitialised without VIC) and `col_wf%mflx_neg_snow_1d` (flattened, VSFM only), and
  `fmoddir` in the generated CMake (spack netcdf-fortran on Chrysalis).

## Phase 2.5: the fix in ELM (2026-11-11)

- Branch `peterdschwartz/lnd/fix-im2-downhill-neighbor` (worktree
  `/lcrc/globalscratch/ac.schwartzpd/im2-downhill-fix`, commit 6ee4269ec5 on master b70b675a7b):
  `dn_index = -1` moved into the `t1` loop of `set_topounit` (one line). Not pushed.
- Case `hs-im2-fix`: the same casegen arguments as `hs-im2` (IELM r05_r05, testmod `elm/topounit_im2`,
  cold start, 5 days, Intel, 256x1, no SPEL instrumentation); Slurm 1326038. No `h0` history files
  in either run (monthly averages, 5 January days): the restarts were compared.
- `compare_restarts.py hs-im2.elm.r.0001-01-06-00000.nc hs-im2-fix.elm.r.0001-01-06-00000.nc --expect <the 7>`:
  42 variables differ, **only in 6 of the 7 expected gridcells** (none elsewhere); the Salton Sea
  gridcell is identical (its outlet had no surface runoff in these 5 days, as the capture showed).
  `FROM_UPHILL` differs in exactly the outlet of each of the 6 (bug -> fix: 0.0014 -> 0.0011,
  0.0010 -> 0.0003, 0.0102 -> 0.0097, 0.0004 -> 0.0002 kg/m2, two below 1e-4); the others follow
  (H2OSOI_LIQ, T_SOISNO, TH2OSFC, ZWT, WA, T_GRND, T_REF2M, ...). Besides these only the global
  budget arrays (`budg_*G`) differ. MOSART restart: different (cprnc), as the runoff of those cells changed.
- **New finding, independent of the fix:** `FROM_UPHILL` is NaN in 1527 of 277539 topounits (1314
  gridcells, 85S-84N), at the same places in both runs. All are active topounits whose landunits
  are wetland, lake, or lake+glacier: no soil/crop (hydrology) column, i.e. H3's case. Not Inf; the
  cold start sets `qflx_from_uphill`/`qflx_to_downhill` to 0 everywhere (ColumnDataType.F90:6050), so
  the NaN comes from the send from or to such topounits or from the drain loop in
  ColWaterBalanceCheck, which runs over every column (BalanceCheckMod.F90:322). To investigate with a
  proptest scenario ("lake low"/"wetland low" hillslopes, `hydrologynodrainage`).

## Phase 3: a day of IM2 routing with proptest multi-step runs (2026-11-12)

Notebook `hillslope_im2_season.ipynb`, spec `hillslope_im2_season.proptest.toml` (built by
`hillslope_im2_season_spec.py`). Case `hs_season` (hydrologynodrainage + hydrologydrainage, `next`),
reference merged from the hs_bug_nodrain and hs_bug captures. 48 half-hour steps; hillslopes lowest
first/last, cycle, Caspian 184059, flat (one topounit), two sets; scenarios storm (rain on the ground for
12 steps), steady (constant rain), recession (full stores, no rain). Boundary (coupling.py): 61 variables;
1 forcing (rain), 1 between (the store drain of ColWaterBalanceCheck), the rest keep (an isolated process:
no snow, ice, evaporation or transpiration).

- `runner.run`, 6 examples per check, seed 0, 354 s: every store invariant and every IM2-on/off identity
  passes (finite, non-negative, conserved, released at frac_from_uphill per step, topmost store halves in a
  recession; one-topounit gridcells and the topmost topounit's soil identical with IM2 off, total water
  conserved, never sooner to the river, receivers never hold less). Only two checks fail, in storm and
  steady: no self-sends (lowest last t3, Caspian t4) and D3 "the outlet passes on at least what it does
  without IM2" (those outlets and cycle t3, which points uphill).
- 20 mirrored storms (40 twin pairs): cumulative river runoff after the day relative to IM2 off, mean (min):
  intended 0.776 (0.175), bug 0.627 (3e-6); step by which half the day's runoff arrived: IM2 off 8.6,
  intended 9.7, bug 17.5. Totals include subsurface drainage (not routed), so the ratios understate the
  surface-runoff effect.
- A shrunk D3 failure had negative runoff: a drawn water table at the surface (`zwt` = 0) with unrelated
  soil water makes Drainage return negative drainage in step 1 (an inconsistent start, not IM2).
- Not covered: H2 and H3 (the between drain reads outputs proptest sets to NaN on columns the routines do
  not process).

## Phase 4: H2, a weight-0 downhill neighbour (2026-11-12)

Spec `hillslope_im2_h2.proptest.toml` (`hillslope_im2_season_spec.py ... h2`), case `hs_season`, 24 steps, two
sets. Hillslopes listed lowest first (H1 cannot trigger): "lowest first" (control), "dead outlet" (the lowest
has weight 0), "dead middle", "dead top". New in proptest for this: `carry` (outputs that keep their last
value where a step does not write them: `qflx_from_uphill` = 0 on the inactive columns, as ELM's cold start
leaves them; otherwise the drain read proptest's NaN), `child_wsum` skipping inactive/weight-0 children (as
subgridAveMod), the domain function `downhill_inactive(t)`.

- Storm (6 examples per check, seed 0): all four checks fail at step 1, only in dead outlet and dead middle
  (both sets): "no active topounit sends water to an inactive one" (4 of 18 columns), "the store stays
  finite" (the weight-0 topounit's store is `inf`: HydrologyDrainage divides by `top_pp%wtgcell` = 0),
  "the stores gain what is sent" (the sent water reaches no active store), R4 conservation with IM2 off
  (the gridcell loses what it sends). Dead top and the control pass.
- Storm with `trap_fpe`: step 1 stops with SIGFPE "floating-point divide by zero".
- On a dry step the same division gives `0 * Inf = NaN`. ColWaterBalanceCheck's
  `max(0, from_uphill - qflx_from_uphill*dtime)` then differs by compiler (tested with a small program):
  `MAX(0, NaN)` is 0 with gfortran 11.2 and NaN with ifx 2025.2 (-O0 and -O2). Intel builds keep a NaN
  store for good; gfortran builds flush it to 0. The full runs were Intel builds, consistent with the
  persistent NaN `FROM_UPHILL` of Phase 2 (its source is still open: H3). proptest's `max` propagates NaN
  (ifx-like).
- Fix direction (not part of the H1 branch): skip inactive topounits in set_topounit's neighbour
  search, or guard the send with `top_pp%active(downhill_t)`.

## Phase 5: H3, the NaN stores of the full runs (2026-11-12)

**Cause.** ELM keeps a virtual natural-vegetation (soil) landunit of weight 0 active on every active
topounit that is not all plain glacier (`subgridWeightsMod.F90` is_active_l, l.380). Its column is in
`hydrologyc`. On a topounit whose other landunits are wetland, lake or glacier it is the only hydrology
column, so SurfaceRunoff's `top_pp%uphill_wt(t)` (the sum of the hydrologyc columns' `wttopounit`) is 0 and
`qflx_from_uphill(c) = (0/0) * ...` (SoilHydrologyMod.F90:283) is NaN every step, even with an empty store and
without any uphill neighbour. ColWaterBalanceCheck's drain then gives `max(0, store - NaN)`: NaN with ifx (the
full runs), 0 with gfortran (where the store, and any water routed into it, is silently deleted every step).
The NaN also enters the virtual column's `qflx_top_soil`; its weight is 0, so gridcell averages hide it.

**Check against the full run** (`hs-im2.elm.r.0001-01-06-00000.nc`): the topounits with hydrologyc columns whose
`wttopounit` sum to 0 are exactly the 1527 NaN `FROM_UPHILL` topounits (1527 both ways, no exception): 1304
wetland, 158 glacier + lake, 55 lake, 10 lake + wetland. The 37596 topounits without a hydrology column (all
plain glacier: no virtual landunit) are finite.

**proptest** (`hillslope_im2_h3.proptest.toml`, `... h3`; `Layout.active` now follows is_active_l/c/p, so
`Soil(13, weight=0.0)` is that virtual landunit): hillslopes "lowest first" (control), "lake low" (a lake
topounit with the virtual soil landunit at the bottom), "lake alone" (one such topounit); 12-step storm, 6
examples. Fail at step 1, only at the lake topounits (both sets): the flux from uphill is finite on every
hydrology column (the virtual columns), the store stays finite, R4 conservation with IM2 off. The control passes.

**Fix direction** (a science choice for the IM2 developers): guard the release with `uphill_wt(t) > 0`; then
decide where water routed into a topounit with no positive-weight hydrology column goes (into the lake or
wetland, or straight to runoff), instead of keeping it in a store nothing releases.
