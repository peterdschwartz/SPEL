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

`components/elm/src/main/ElmDeviceDataMod.F90` (from `fut.type_dict`: every module-level
instance of every analyzed derived type, its components, nested types, pointer targets),
plus calls inserted into `elm_initializeMod` and `elm_driver`. Everything compiles to
no-ops without `_OPENACC`, so CPU builds are untouched (bit-for-bit).

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
  copy. Per type a routine `dd_copyin_<type>(this)` does `enter data copyin` of every
  array component with storage (`associated`/`allocated` guards), recursing into nested
  types and element-wise into arrays of types; instances without a `declare` get `enter
  data copyin(instance)` first. Pointer components the analysis records as aliases (`=>`
  to another component, e.g. `transport_ptr_list`, compact fields) are not copied but
  attached in a second pass (`dd_attach_<type>`) once their targets are present. Then
  `dd_check_<type>` tests every array with `acc_is_present` and the misses are reported
  per rank (the first 50 by name), so a failed copy is diagnosed at the first step instead
  of at the first kernel.
- `elm_device_update(to_device)`: `update device` or `update self` of every component of
  every instance (`dd_update_<type>`), the whole state.

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

## Usage

```
spel device-data --dry-run            # plan: instances, transitions, edits
spel device-data --print              # the module text
spel device-data [--reanalyze]        # write the module, edit the two driver files in place
spel device-data -o CASE/SourceMods/src.elm
```

Debug and optimized builds use the same generated code (plain data directives, runtime
guards, no reliance on optimization); the presence check runs in both.
