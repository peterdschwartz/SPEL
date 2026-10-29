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
