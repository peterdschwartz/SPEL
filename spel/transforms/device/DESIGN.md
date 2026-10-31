# Device data layer for full-model OpenACC runs (`spel device-data`)

Status: first version built 2026-10-25; brute-force synchronization.

## Why

The branch's OpenACC kernels were only ever executed inside SPEL unit tests, whose harness
moves the data. In the full model every type instance carries `!$acc declare create`
(the object exists on the device) but nothing copies the contents of its components, and
the `*_vars` instances passed as arguments are never created on the device at all. The
first kernel of a full run therefore reads unset device memory. `spel device-data`
generates the missing layer from the analysis.

## What is generated

Modules in `components/elm/src/data_types/` (from `fut.type_dict`: every module-level
instance of every analyzed derived type, its components, nested types, pointer targets),
plus calls inserted into `elm_initializeMod` and `elm_driver`. Everything compiles to
no-ops without `_OPENACC`, so CPU builds are untouched (bit-for-bit).

The types are split into group modules (`GROUPS`/`type_group` in `generate.py`), in `use`
order: a group may only nest types of earlier groups (checked at generation):

| Module | Types |
|---|---|
| `DeepCopyParamsMod` | parameter/constant types (`*paramstype`, `veg_vp`, `decomp_cascade_con`, `soilordercon`, urban params) |
| `DeepCopySubgridMod` | `*_physical_properties`, filters, clumps/decomposition, domain, subgrid averaging view |
| `DeepCopyCouplingMod` | `atm2lnd`, `lnd2atm`, glc/iac/ocn coupling, topounit atmospheric state/flux, dust, drydep |
| `DeepCopyBiogeophysMod` | energy/water state/flux types and the physics `*_vars` (default for unmatched types) |
| `DeepCopyVegNutrientMod` | `vegetation_*` carbon/nitrogen/phosphorus types, `cnstate`, `crop` |
| `DeepCopySoilNutrientMod` | column/gridcell C/N/P types, nutrient core types, CH4, the BGC interface |

A type the rules don't name takes the group of the type that nests it.

`DeepCopyGlobalsMod` covers the module variables named in `!$acc declare` directives
(215 on the branch): they exist on the device from program start, but their values are
set at run time (namelists, parameter files, and the per-step `timeInfoMod` scalars), so
`globals_copyin` uploads them once and `globals_update` moves them at every transition.
Parameters are skipped; private variables can't be named outside their module, so their
module must update them where it sets them (`PhenologyMod`, `DUSTMod`, `subgridAveMod`,
`CH4Mod`, `C14DecayMod`, ...: the generator lists them as skipped). `histComputeMod` and
`CompactLayoutMod` move their own data and are excluded. Module pointers get an
`associated` guard and `if_present` (no device copy when `=>`-assigned).

Namelist guards come from the access analysis (`access_guards`): `nml_guards` of `elm_drv`
gives every `inst%field` that its call tree touches only under namelist ifs, with the
condition (enclosing namelist ifs, through call sites and cascades), and
`check_subs_for_nml_guarded_vars` turns that into per-type-component guards that hold for
every instance accessing the component. In the per-type routines each such component is
copied/updated/checked only under its condition (475 components on the branch, e.g.
`use_lch4`, `use_hydrstress .and. .not. use_fates`, `do_budgets`). An instance whose
accessed fields are all gated is handled under the or of their conditions (39 instances:
`use_cn .or. use_fates` for the C/N/P types, `use_cn .and. use_c13` for `c13_grc_cf`,
`use_im2_hillslope_hydrology` for `top_ws`). Conditions are minimized exactly
(`boolen_expression.minimize`, Quine-McCluskey over the atoms) and common literals factored
out. Not guarded: history-registered components (`hist_update_hbuf`, outside the analysis,
reads them on the device) and conditions with string comparisons. The component-level
`allocated`/`associated` tests stay as the inner safety net.

Inline components (scalars, fixed-size arrays) are sent with `update device(this%x)` in
`deepcopy_<type>`: an object made by `declare create` never received its values, and
`enter data copyin` of an inline member of a present object is a no-op (a pattern test
showed the device reading 0 for such a scalar and the next `update self` writing the 0
back to the host). A shallow `update device(instance)` would do it in one call but
overwrites device descriptors that init code already attached (`elm_subgrid_view`,
compact groups), so it is not used. Each group module
has, per type, `deepcopy_<type>(this)`, `attach_<type>(this)` (only types that hold alias
pointers), `update_<type>(this, to_device)` and `check_<type>(this, label, nmiss)`, and
for the group's instances `<group>_copyin/_attach/_check/_update`. `ElmDeviceDataMod`
holds the entry points below and calls the groups in order.

- `elm_device_init()`: first call of `initialize1`. `acc_init` on the rank's device.
  Module data in `declare` directives (`declare copyin` constants, `declare create`
  instances) is placed on the *default* device when the program starts, before any
  Fortran runs, so a rank that later switches device loses it. The run must give each
  rank exactly one visible GPU (`srun --gpus-per-task=1`; CIME's pm-gpu affinity script
  only applies at 64 tasks per node). With several visible devices the routine selects one
  by `SLURM_LOCALID` and warns that declared data may be stranded.
- `elm_device_copyin()`: called at the top of `elm_drv` on the first time step only
  (`elm_device_resident` guard), i.e. after initialization and restart reads, as one
  explicit, hand-inspectable deep copy per instance rather than implicit OpenACC deep
  copy. Per type `deepcopy_<type>(this)` does `enter data copyin` of every
  array component with storage (`associated`/`allocated` guards), recursing into nested
  types and element-wise into arrays of types; instances without a `declare` get `enter
  data copyin(instance)` first. Pointer components the analysis records as aliases (`=>`
  to another component, e.g. `transport_ptr_list`, compact fields) are not copied but
  attached in a second pass (`attach_<type>`, after every group's copy) once their targets are present. Then
  `check_<type>` tests every array with `acc_is_present` and the misses are reported
  per rank (the first 50 by name), so a failed copy is diagnosed at the first step instead
  of at the first kernel.
- `elm_device_update(to_device)`: `update device` or `update self` of every component of
  every instance (`update_<type>`), the whole state.

Driver insertion (`_transition_edits`): the analysis's record of `elm_drv`'s calls, in
line order, each classified device (its call tree contains an OpenACC compute construct,
or is in `DEVICE_EXTRA` for modules the analysis excludes, e.g. `hist_update_hbuf`) or
host; neutral calls (timers, bounds getters) are ignored. At every transition a whole-
state update is inserted before the call: `update self` before the first host routine
after device work, `update device` before the first device routine after host work. The
step starts in the host state (coupler import, IO). On the branch this gives 18 syncs per
step. One clump per rank is required (`clump_pproc = 1`): the transitions sit inside the
clump loop and the syncs are process-wide.

## Known limits (first version)

- **Filters are built on the host.** `setFilters` lost its OpenACC directives (E3SM
  b2f830ef5c): the filters change only with the subgrid weights (never in a non-transient
  run) and their fill pass is serial, slow on one GPU thread. They used to run during
  `initialize2`, before the copy, and crashed the first GPU run (`filterMod:327`). When the
  weights change after the copy (first step, transient runs), `reweight_wrapup` sends the
  rebuilt filters and subgrid data with `subgrid_update(to_device=.true.)` (inserted by the
  generator; `acc_is_present(filter)` tells whether the copy has happened).
  `setExposedvegpFilters` (every step, inputs made on the device) keeps its kernels.
- Other kernels that run during initialization would read device data before the copy;
  none is known after the filter change, the next GPU run will tell.
- gfortran: GCC 14.3 on Perlmutter is configured for nvptx offload but the offload
  compiler (`accel/nvptx-none/mkoffload`) is not installed, so gfortran builds run the
  OpenACC code on the host (`ACC_DEVICE_TYPE=host`). The generated patterns pass a
  standalone test with gfortran (host) and nvfortran (A100): `$SCRATCH/devdata_patterns`.

- Pointer components aliased by a `=>` the module scan never sees get a separate device
  copy; host and device then disagree on them silently. The compact layout's fields are
  aliases the analysis knows (`DerivedType.components[x].pointer`).
- Character components stay on the host.
- The whole-state syncs are the cost of correctness without analysis; refinement is per
  transition the set of components the next host (device) section reads or writes, from
  `record_access` of the routines in its call tree.
- The device/host classification follows OpenACC compute constructs in the analyzed
  sources; a routine in an excluded module that runs kernels must be named in
  `DEVICE_EXTRA`.

## Host code inside a device section (`spel/transforms/acc_sync.py`)

The layer syncs only at elm_drv's transitions. A routine that runs kernels but keeps some
host statements (loops not ported yet, setup and first-step code) reads stale host data
there and writes values the device never sees. Before 2026-10-26 the end-of-step
`update self` then overwrote them (dd_eco step 0: 440 NaN-pattern differences).
`acc_sync.plan(fut, roots)` handles this for the routines of a call tree that run device
work (own kernels, or calls into routines that run them).

- A host segment is a maximal run of consecutive statements of one block. None of them
  starts inside a compute construct or calls a routine that runs kernels. Constructs that
  hold kernels are recursed into, and their headers (if conditions, do bounds) get an
  `update self` of their own.
- Before a segment: `update self(...) if_present` of the data it reads or writes that
  device code writes. A write needs it too, since the push after the segment sends the
  whole array.
- After the segment, and before each `return` inside it: `update device(...) if_present`
  of everything it writes. That includes module variables, which the end-of-step update
  would otherwise overwrite.
- Accesses come from `AccessMaps`, so host-only callees are covered on their call lines
  (their globals, argument statuses) and get no syncs of their own.
- Pointer components with recorded targets are expanded (`transport_ptr_list(i)%conc_ptr`
  → `col_cs%core%decomp_pools_vr`, ...).
- "Device-written" means written in a kernel, or by a routine-seq callee on its call line.
  It is propagated both ways through the argument bindings (dummy ↔ actual), and keyed by
  (owner type, member), so `this%x` and `col_cf%x` are the same thing.
- Calls resolve through `AccessMapper.callee`, which handles generic interfaces. `p2c` is
  generic, and missing it once pushed stale host `leafc_col` over the kernel's result.
- Only intrinsic-typed data is updated. A whole derived-type object would overwrite the
  host's pointer components with device addresses.
- A callee's global that the routine doesn't see gets `use mod, only : x  ! spel acc-sync`.
  Private module variables and private components are skipped and listed in
  `RoutineSync.skipped`.
- Correct, not fast. Each port that removes a host segment removes its syncs. Grep the tag
  to see what is left.

Unit tests: with `--device-data`, `read_elmtypes` loads each step's inputs into the device
buffers only. `main.F90` therefore calls `elm_device_update(to_device=.false.)` after the
read (`!#ACC_STEP_INPUTS`), so the host matches the device at the start of the step, as at a
transition. Without it, `CropPlantDate`'s host loop read step-0 forcing.

Validation (dd_eco = EcosystemDynNoLeaching1/2, 27 steps, E3SM 0c640e391b): the GPU matches
ACC=OFF to roundoff (max |diff|/max |value| is 5e-16 at step 0). The few large relative
errors are in variables whose whole range is 1e-16 or less (cancellation residues) or are
denormals the GPU flushes to zero. OFF is bit-identical to before, and MULTICORE equals OFF.
Driver: `$SCRATCH/spelval/port_eco.sh` (acc-apply → acc-locals → acc-sync → `spel create`,
with a reanalysis after each step); comparison: `$SCRATCH/spelval/tools/cmp_steps.py`.

## Porting the remaining host loops (2026-10-26)

Each host segment that gets ported removes its syncs. To measure what is left, run the unit
test with `NVCOMPILER_ACC_NOTIFY=2` and sum the transfers per source line
(`$SCRATCH/spelval/tools/notify_sum.sh`). For dd_eco this went from 9.4 GB per 27-step run
to about 0.2 GB. The pipeline now runs these steps before acc-apply, plus hand ports kept
as a patch against the base commit:

- `spel/transforms/blocked_loops.py` rewrites the SetValues idiom (`associate (idx =>
  filter(fi:...))` blocks) into one element per iteration. It shares its detection with
  `spel compact`'s `--setvalues-loops`. It is repeated until nothing is left, because
  routines first reached after an unblock appear only in the next analysis.
- acc-apply interchanges perfect (level, column) nests that the planner calls
  `interchange` (`_swappable`, `_swap`). The column loop becomes the compute loop and the
  level loop runs `seq`, so each column's order of operations is unchanged.
- Only assignments through a vector subscript block a loop (`x(idx) =` with an index
  array). A plain section `x(c,:) =` belongs to one iteration.
- acc-apply takes parse trees from the preprocessed module (`routine_tree`), so an
  `#ifdef` inside an associate list (FireArea) no longer makes the whole routine
  "unsafe". Abort calls come from the parse tree (`abort_calls`), because the analysis
  drops `endrun` calls from its edited sources.
- acc-sync puts the syncs of an if construct that can abort inside its branches: an
  error check runs its host code, and pays for the copies, only when the check fails.
  Hand-ported checks count failures on the device (`reduction(+:nbad)`) and run the
  original diagnostic loop on the host only when `nbad > 0` (VerticalProfile).
- Patch-to-column scatters are ported by hand as column-parallel loops over each
  column's patches in filter order (bit-for-bit, since filters are ascending). The soil
  patches of a soil column are its active patches (FireFluxes).

## Usage

```
spel device-data --dry-run            # plan: instances, transitions, edits
spel device-data --print              # the module texts
spel device-data [--reanalyze]        # write the modules to data_types/, edit the two driver files in place
spel device-data -o CASE/SourceMods/src.elm
```

Debug and optimized builds use the same generated code (plain data directives, runtime
guards, no reliance on optimization); the presence check runs in both.

## Local variables in simpler forms (`spel/transforms/local_forms.py`)

A report, made from the record and the parse tree's if guards; the edits are made by hand
(NitrifDenitrif, 2026-10-26, was the model). Run it with `$SCRATCH/spelval/tools/eco_locals_forms.py
[routine ...]`.

- `scalar`: a local array whose every use is in one loop nest, always the same element made of
  the surrounding loop indices (`r_min(fc,j)`), with a write before every read in the
  iteration (same loops, enclosing ifs a prefix of the read's). It becomes a scalar, `private`
  in the compute loop, so it needs no device storage.
- `parameter`: an initializer that nothing writes, or a single unconditional assignment
  from a constant expression outside loops, before every read (`pH(:) = 6.5` becomes a scalar
  parameter). Findings with transcendental intrinsics are marked, because the compiler's
  value may differ from the run time's in the last bit.
- `dead`: written, never read.

Actual-argument, pointer, allocation and `size` uses disqualify a variable. The walker
records an initializer as an `init` write.

## 2026-10-29: the remaining host syncs

- `generate.HOST_SEGMENTS` is empty: dynSubgrid_driver is a device call. Its heat/water content
  (dynConsBiogeophysMod, TotalWaterAndHeatMod) and weight bookkeeping (set_prior/old/new_*_weights)
  run on the device; the weight-change path (transient land use, dyn_cnbal_*) stays host code under
  `if (weights_updated)`. `tools/scope.py` lists what stays out.
- acc-sync `cold` has a fourth rule, `flag_condition`: a block `if` whose condition reads only host
  flags (scalars no device code writes, or local scalars such as a reduction count) gets its syncs
  inside its branches. `if (weights_updated)`, `if (create_glacier_mec_landunit)`, `if (nbad > 0)`
  cost nothing when not taken. A condition on an array element or a derived-type object keeps the
  whole construct as one segment.
- `spel/transforms/section_loops.py`: array-section assignments outside kernels become explicit loop
  nests (`x(begc:endc,1:n) = 0` -> `do spel_i2; do spel_i1; x(spel_i1,spel_i2) = 0`), which acc-apply
  then ports. Runs before acc-apply in the pipeline (`tools/section_loops_apply.py`).
- `edit_files` keeps the write of an assignment whose right-hand side used an excluded module as the
  no-op `lhs = lhs` (the walker records a self-assignment as a write only): pdep_interp's
  `forc_pdep_grc` write used to vanish with the statement, so nothing pushed it. ndepStreamMod is no
  longer excluded; the whole-type `update_atm2lnd_type` around ndep_interp goes away.
- An internal procedure's record links its host (`SubroutineRecord.host`); the planner resolves
  host-associated arrays through it (store_tsl_moisture's `h2osoi_liq_saved` was a "host scalar race").
- Write statements no longer block loops: PrecisionControl, ch4_tran and DivideExtraSnowLayers count
  on the device and report on the host only when the count is non-zero (a cold branch).

## 2026-10-30: collapsed loops must not take their bounds from inquiry functions (`spel/transforms/collapse_bounds.py`)

nvfortran 25.9 (cc80) runs `parallel loop collapse(n)` as an empty kernel on the GPU when a
collapsed `do` header's bound is `lbound`/`ubound` of a pointer (a derived-type component,
an associate name of one, or a module pointer): no error, nothing written. A 1-D loop with
the same bounds runs, a collapsed nest with host-scalar bounds runs, and the device
descriptor holds the right bounds (`$SCRATCH/spelval/assoc_test/t_assoc2.F90`); a module
allocatable with `size()` bounds is fine (`t_alloc.F90`), so histComputeMod's hand-written
`collapse(2)` over `size(snow_fields)` stays. The
section-loop rewrite (`x(:,:) = y(:,:)` -> `do spel_i2 = lbound(x,2), ubound(x,2)` ...)
produced 29 such nests; the first casualty was UrbanRadiation's `forc_solad_pp` copies,
which left `top_af%solad_pp` = spval on the device and surfaced two routines later as
"t_soisno > 400 K in urban columns" (a day of file-group bisecting; the probe method of
memory `probe-in-elm-drv-not-bisect` found it in two runs).

`collapse_bounds.apply(fut, routine_ids)` (tool: `spelval/tools/collapse_bounds_apply.py`)
hoists every inquiry bound of a collapsed nest into `spel_lb<k>`/`spel_ub<k>` integer scalars
assigned right before the directive, k counting the nest outer-first, and rebuilds the
headers from the record's loop nodes. It needs the analysis of the sources it edits, so the
pipeline is: section-loops -> acc-apply -> `spel analyze` -> collapse-bounds (then
bounds_copyin / device_ready as before). Rule for hand-written kernels: the same.

## 2026-10-30: calls into excluded modules inside device routines; genf90 templates

A call to a routine of a module the analysis excludes (`edit_files.bad_modules`) has no events,
so `acc_sync` syncs nothing around it. `dynSubgrid_driver`'s `dyncrop_interp` writes the crop
landunit and column weights and the fertilizer patches on the host; once dynSubgrid_driver ran
device kernels itself (the 10-29 host-sync removal), those writes never reached the device and
the GPU restart diverged from the host (GDD accumulators, soil temperature). Two measures:

- `acc_sync.plan` reports every such call (`SyncPlan.notes`, `excluded_calls_in`; printed by
  `tools/tree_sync.py`). A note means the routine is not safe on the device until the callee is
  analyzable.
- `spel.scripts.genf90` generates the modules E3SM builds from genf90 templates (`dynVar*`,
  `ncdio_pio`, `restUtilMod`, `shr_*`) into the analysis cache (`<cache>/genf90`, line markers
  stripped) and the module lookup searches it, so `dyncropFileMod` and `dynpftFileMod` are
  analyzed instead of excluded, and the sync planner sees their effects on the call lines.

Rule learned the same day: a pull of data the device-data layer does not carry (the
accumulators of accumComputeMod, the history buffers, the budget tables) at a point where the
host takes over (restart write, a host-mode update) must be unconditional. `elm_device_ready`
is already false there, so `if(elm_device_ready)` turns the pull into a no-op and the restart
gets stale host copies (every accumulator was 10 steps short, 2026-10-30). The clause belongs
only on acc-sync's own updates inside device routines, whose host copies are current whenever
the flag is off.

Same day, third rule: a local the host initializes in the specification part (an initializer
or a DATA statement) and never writes again is `enter data copyin`, not `create`
(`acc_locals.RoutineReport.host_initialized`): `create` hands the device an unfilled copy,
and a push placed right after the DATA statement is a no-op because the device copy does
not exist yet (acc_sync now puts pushes of specification-part writes after the enter-data
block, `after_enter_data`). CombineSnowLayers' dzmin thresholds were zero on the device, so
the GPU never combined thin snow layers (one extra layer in the worst columns, 2026-10-30).

## 2026-10-30: what the four regressions of the host-sync removal have in common

Residency (which copy of a variable is authoritative, where) is implicit in directive text
spread over the tree; each pass (section-loops, acc-apply, acc-locals, acc-sync, targeted,
the host-safe retrofit, hand edits) edits that text once, in sequence, nothing re-derives the
whole from the analysis afterwards, and OpenACC's `if_present`, `create` and `if(flag)` turn
every inconsistency into a wrong number instead of an abort. Rules that follow:

1. Syncs are an output of the analysis, never maintained text: fixed pass order, `spel
   analyze` between passes, `tools/sync_drift.py` (plan vs. present directives, per routine)
   as a mandatory gate before any build. It found 9 drifted routines in a minute.
2. Hand-written syncs are invisible to the planner: tag them or let the planner own them all.
3. Debug build to make the silences loud: no `if_present` on generated updates, NaN-fill of
   created locals, device-lint = collapse-bounds + host-initialized-locals + sync-drift.
4. Validate per pass group (10-step host-vs-GPU restart, 15 min), compare the restarts first
   (per landunit, per layer, pattern of the difference), then probe the driver.

## 2026-10-30: `attach` before the target is on the device is a silent no-op (history buffers wrong on the GPU)

The refactored history (`histComputeMod`) and the subgrid view (`subgridAveViewMod`) reach the
model arrays through pointer components that are `enter data attach`ed. Both were attached when
they were built, in `initialize2` before `elm_device_copyin`: the OpenACC runtime takes *no
action* for an attach whose target is not present (spec 2.7.2), the device descriptor keeps the
host address, and Perlmutter's unified addressing (HMM) lets the kernel read host memory instead
of faulting. The GPU's history buffers were the *host's* stale arrays (TSOI exactly 11 x the
finidat value, fluxes 0, spval-initialized fields spval), at page-fault speed (7.5 s / 10 steps).
The in-tree test copied the data before attaching, so it passed; the 10-step validation compares
the model restart, never the history buffers. Reproducer: `spelval/histtest/repro_attach/t_attach.F90`.

Rules:

1. Attach is a separate, explicit step after the data layer's copy (`subgrid_ave_device_attach`,
   `hist_device_attach`, called by `initialize2` right after `elm_device_copyin`); the builders
   (`init_subgrid_ave`, `hist_compute_build`) no longer attach, and re-attach themselves only
   when rebuilding something that was attached.
2. Every attach checks `acc_is_present` of every target and aborts on a missing one; the device
   path aborts if it runs before the attach. Nothing is allowed to fall through to host memory.
3. Tests of device code copy the data in the *model's* order (after the consumer's setup) and run
   with a host decoy: the host copies are made wrong (negated) while the kernel runs, so a kernel
   reading host memory fails the comparison even where host and device would otherwise agree.
4. An e2e validation of a component must compare that component's output (the history restart
   `elm.rh0` through cprnc, `tools/validate_hist.sh`), not only the model restart.

## 2026-10-30: every history source must be in the device-data layer (`ptr_topo`, private instances)

The first run with the presence checks of rule 2 above aborted: `PCT_LANDUNIT`, `PCT_NAT_PFT` and
`PCT_CFT` (hpindex 1492-1494, the last fields registered) read `subgrid_weights_diagnostics%pct_*`
(`subgridWeightsMod`), which the layer did not carry. Two SPEL deficits hid it:

- `registrations.py` knew `ptr_patch/col/lunit/gcell/lnd/atm` but not `ptr_topo`, so the 17
  topounit-level fields (`TBOT`, `PBOT`, ..., the `PCT_*` fields) were invisible to
  `_history_components` and to every registration-based check. Fixed (test added).
- `subgridWeightsMod` is default-private and did not export the type or the instance, so `plan`
  skipped it ("private module variable"): the layer cannot name it, and acc-sync cannot push it
  after `set_subgrid_diagnostic_fields` (a host loop under `weights_updated`). Fixed in the tree by
  making both public; the layer is regenerated and dynSubgridDriverMod resynced.

Rules: a history field's source is read on the device by `histComputeMod` whatever the analyzed
routines do, so the planner must treat every registration target as device data: the instance must
be in the layer (public, or its module must manage it), and host writes to it need a push. The
history attach check (`hist_compute_attach`) reports *every* missing target before it aborts, so one
GPU run with the production history (`tools/gpuleg_hist_probe.sh`) lists the whole gap; the
hpindex -> field name map is the `hpindex`/`name` variables of the history restart (`elm.rh0`).

The same night, the lnd2atm presence check found the second gap of this kind: `sedflux_vars` was copied only under
`(use_cn .and. use_erosion) .or. (ero_ccycle .and. ...)`, because `access_guards` derived the instance guards from
elm_drv's accesses alone, while `lnd2atm` (called by `lnd_run_mct`, outside elm_drv) averages `sed_yld_col`
unconditionally through the `l2a_add` table. The guards now come from every device root (`DEVICE_ROOTS` =
elm_drv, lnd2atm, lnd2atm_minimal; `merge_guards`: a field stays guarded only if every root that accesses it
guards it). Rule: whenever device code gets a new entry point outside elm_drv's tree, add it to `DEVICE_ROOTS`.
