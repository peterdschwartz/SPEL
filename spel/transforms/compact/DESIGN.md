# Compact subgrid layout

Goal: subgrid fields stored on the elements a filter selects, so loops over a
filter read and write contiguous memory (and the per-step zeroing is a
contiguous fill), instead of scattered `x(filter(fp))` accesses into
`begp:endp` arrays. First target: the C/N/P flux types zeroed every step by
`SetValues` in `EcosystemDynNoLeaching1` (`CNZero`). Long term: filters give
`beg:end` ranges for every field.

This is the second design (2026-10-24). The first kept every field's full
`begp:endp` allocation plus a same-sized compact copy, indexed each clump at
`begp-1+fp`, and served history by unpacking after `EcosystemDynLeaching`;
it was net neutral on Intel CPUs. Since the layout is a transform applied at
build time (`e3sm_casegen.sh --compact`), it may be disruptive.

## Commands

```bash
spel compact inventory [--verdict blocked] [--field veg_cf%gpp] [-o inv.json]
spel compact rewrite [--type vegetation_carbon_flux ...] [--roles] --dry-run [--diff] [--excluded]
spel compact rewrite [--roles] [-o CASE/SourceMods/src.elm [--replace]]
```

`inventory` (inventory.py) traces one time step from `elm_drv` and reports,
per field, how every access indexes it (index_forms.py), whether it lives
across steps (history/restart, registrations.py), and a verdict. `rewrite`
(select.py, rewrite.py) applies the layout to the fields of the types that
can take it. `validate_level.sh`/`e3sm_casegen.sh --compact` pass extra
options through `COMPACT_ARGS`.

## Layout

- **Group buffers.** `CompactLayoutMod` (templates/; the E3SM branch carries the
  same file in `components/elm/src/main`, since `histComputeMod` and
  `histFileMod` use it, and the rewrite checks it equals the template) holds
  `compact_groups(:)`, each `data(n, nfld)` allocated over the **process soil
  filter** of its level: `n = compact_num_soilp` (patch) or `compact_num_soilc`
  (column), the sizes of the concatenated `filter(nc)%soilp`/`soilc` over the
  clumps. A group is one holder embedding (`veg_cf%core`, `veg_nf%retrans`) x
  one zeroing guard x one level; the type's pointer components alias its
  columns (`this%x => compact_groups(id)%data(:, slot)`). Memory is the filter
  size, not the subgrid size.
- **Clumps are contiguous slices.** Clumps tile the process range in order and
  filters are built ascending, so the concatenation is ascending and element
  `fp` of clump `nc` is `compact_soilp_clump_off(nc) + fp` (prefix counts).
  `get_clump_bounds` calls `compact_set_clump(n)`, which sets the thread-private
  `compact_soilp_off`/`compact_soilc_off`; every rewritten routine copies them
  once to a local `k0p`/`k0c`, so OpenMP/OpenACC loops see a private scalar.
  Sites keep the form `x(k0p+fp)`. No routine signature changes.
- **Inverse maps.** `compact_cidx_p(begp:endp)` / `compact_cidx_c` hold, per
  point, the compact index (member, > 0), -1 on an active special-landunit
  point, 0 otherwise. History reads a compact field through them inside the
  kernels (`histComputeMod`: `field_level_compact`, `accumulate_samples_compact`,
  `subgridAveViewMod`: `p2g_range_compact`, `c2g_range_compact`): member ->
  value, -1 -> the group's `fill` (0 when the group's guard held at the
  init-time special-landunit zeroing, else spval), 0 -> spval. That is
  bit-for-bit what the old full array held (init-time `SetValues` over the
  special list, spval elsewhere; inactive points are masked by `active_p/c`).
  `hist_addfld1d` finds compact sources with `compact_locate` (pointer
  association against the group columns), records (group, slot) in the field
  table and nullifies the stored pointer; `set_lake/urb/spec` arguments on a
  compact source are an error (the selection excludes such fields). Rank-2
  sources: see "Rank-2 fields" below.
- **Reallocation.** The generated `CompactBindMod` (`compact_filters_set_from_filters`:
  counts, concatenation, `ifspecial` flags from the clumps' filters;
  `compact_filters_updated`: that, `compact_group_resize` for every group,
  `<holder>_compact_bind` for every embedding, `compact_bound = .true.`) runs
  from `initialize2` before `elm_inst_biogeochem` (filters only, so Init can
  bind) and after the final `reweight_wrapup` loop, and from `elm_drv` after
  `t_stopf('dyn_subgrid')` when `get_subgrid_weights_updated() .or. nstep_mod
  == 0 .or. first_restart_step_mod` (the condition CNZero uses). Nothing is
  preserved across a resize: every selected field is zeroed by `SetValues`
  before any read in the step (selection rule below). Groups are resolved per
  step in the history kernels, so nothing dangles.
- **Types.** The holder gets `integer :: cgrp_id(ngroups) = 0` and a plain
  module procedure `<holder>_compact_bind(this)` (creates the groups on first
  use with `compact_group_new(name, level, nfld)`, then points the fields at
  the columns). In the Init routine that allocates the fields (`<base>_init`
  for a nutrient base type, the concrete Init for a role or a pre-nutrient
  type) the `allocate(this%x(beg:end)); this%x(:) = spval` lines and any
  redundant `this%x(...) = spval` re-initialization are dropped and the bind
  call is inserted after the last allocation; a field the Init touches
  anywhere else is dropped from the selection with a note, and a history
  registration before the bind is an error.
- **Zeroing.** `SetValues` drops the compact fields' assignments and appends
  one block: `if (compact_bound)` -> per group under its guard
  `this%x(k0p+1:k0p+num) = value` (explicit per-field assignments so SPEL's
  analysis sees the writes); `else` (the init-time special-landunit call, the
  only call before binding) -> `compact_group_fill(this%cgrp_id(k), value)`
  under the same guard, which is how the group's `fill` gets the value
  history reads on active special points. The blocked loops (`do fi = 1, num,
  32` around `associate (idx => filter(fi:...))`) become one `do concurrent`
  per block, for all six flux types (`--no-setvalues-loops` keeps them):
  nvfortran handles the associate-name subscript very slowly.

## Which fields

A component of a holder is rewritten when **every** instance embedding the
holder zeroes it per step (the inventory reports, and the rewrite checks and
edits, only the accesses of instances whose zeroing writes the field: an
instance without that zeroing would keep full-layout sites on a compact
array, as `col_nf%core%decomp_pools_leached` did until 2026-10-24) and, for
every instance:

- the inventory verdict over **every access after the zeroing, for the rest of
  the step**, is `compact` (or `unused`): there is no unpack point any more
  (`build_inventory(unpack_after=None)`, `NO_UNPACK`),
- nothing accesses it before its zeroing in the step (the storage may have
  been reallocated), and no history registration passes `set_*` values
  (`Registration.set_special`); a restart variable is allowed (see "Restart
  fields"),
- it is rank 1, or rank 2 with levels `1:nlev` (see "Rank-2 fields"), with one
  zeroing guard (the guard is the group),
- every access site is `x(p)` with `p = filter_soilp(fp)` in `do fp = 1, num_soilp`
  (column: `filter_soilc`), a provable member reached through a map (see
  "Map-indexed sites"), or has no subscripts (passed on whole to an analyzed
  routine, or a dummy bound to an element); a whole-array pass to an unanalyzed
  routine excludes it; the history kernels' own reads (`histComputeMod`,
  `subgridAveViewMod`) are ignored, the compact path serves them,
- every site that touches it touches only selected fields (a generic routine
  indexes all arrays bound to a dummy the same way). Applied until nothing changes.

A routine whose array dummies get selected fields in some calls and full
layout arrays in others (`CarbonIsoFluxCalc`) is allowed when every call is
all-compact or all-full on those dummies: it gets an optional `compact_off`
dummy (index `kc = compact_off + fp` if present, else `p`), passed by the
compact calls.

`rewrite` lists uses of the selected components in routines the trace never
visits (a text search; it includes same-named components of other types).

On the branch (2026-10-24, `--roles`, all six flux types): 84 fields;
`vegetation_nutrient_flux` 55, `column_nutrient_flux` 20, `column_mineral_flux`
2, `vegetation_retrans_flux` 6, `vegetation_uptake_flux` 1 (phase 1 alone had
24 + 5 + 1 patch fields; 54 before restart fields and the harvest analysis).
`dynHarvestMod` was in `edit_files.bad_modules` (never analyzed), so every
`hrv_*` field counted as written by an unanalyzed routine; it is analyzed now
(`CNHarvest` runs after CNZero, its sites are filter-indexed), and the unit
test drops the `extends(file_desc_t)` of `dynFileMod`'s type, whose parent
comes from PIO. Bisecting on the stride-8 case: `spel compact rewrite
--fields a,b -o CASE/SourceMods/src.elm --replace`, `--no-map-sites`.

### Map-indexed sites (phase 2a, built)

A site whose subscript is a provable member of the main filter, reached other
than by gathering from it, is rewritten to `x(compact_cidx_p(p))` (site mode
`map`, `select.member_through_map`): an element of a sub-filter
(`filter_pcropp`, `index_forms.SUBSET_FILTERS`), a patch
`p = col_pp%pfti(c) + pi - 1` of a `filter_soilc` column read or written under
`veg_pp%active(p)` (the soil patch filter is exactly the active patches on
soil/crop columns), or a column `c = veg_pp%column(p)` of a `filter_soilp`
patch (an active patch needs an active column, which is then a soil/crop
column in `filter_soilc`). The maps are read-only within the step, so kernels
stay race-free. The `compact+maps` verdict is admitted; the per-site rule
decides. On the branch this takes `vegetation_nutrient_flux` from 24 to 30
fields; what remained excluded then: the `hrv_*` fields (an analysis gap, fixed
later, see "Which fields"), restart fields (supported later, see "Restart
fields"), `veg_cf_summary`'s `p2c_1d_filter` slices and `pfti(c):pftf(c)` range
sums, and a field with two zeroing guards.

### Rank-2 fields (phase 2b, built)

A rank-2 field whose levels start at 1 (`begc:endc,1:nlevdecomp_full`,
`select.level_bounds`) joins a rank-2 group: one group per (guard, level
count), created with `compact_group_new(name, level, nfld, nlev)` and stored
as `data3(n, nlev, nfld)`; the component is bound to `data3(:, :, slot)`. Sites
are `x(k0c+fc, j)`. The zeroing writes the level range of the loop around the
original assignment (`x(k0c+1:k0c+num, 1:nlevdecomp_full) = value`,
`Rewriter.level_range`), in whichever SetValues routine holds that assignment
(the concrete routines zero base-holder fields through `this%core%x` too); a
selected field that no routine zeroes stops the rewrite. History: a rank-2
registration (`hist_addfld2d`) reads level `lev` from `data3(:, lev, slot)`;
a per-level rank-1 registration (`data1dptr => x(:, k)` in Init, then
`hist_addfld1d`) is found by `compact_locate` as a level block and carries
`compact_lev`. In Init such a pointer assignment is not a touch of the field
unless something writes through the pointer. Rank-3 fields and fields with a
level lower bound other than 1 stay excluded. The `do concurrent` rewrite of
the blocked `SetValues` loops is off by default (CNZero 0.94x on Intel).

Measured (stride-8 summer window, 128 ranks x 1 thread, Intel, 5 days, both
runs in one allocation, 2026-10-24): 54 fields give `elm_run` 1.07x,
`ecosysdyn` 1.12x, `CNZero` 1.32x, `CNUpdate2` 1.15x, `SLDA-actual` 1.13x,
`CNPsum` 1.10x over the plain build; land throughput 13.15 vs 12.24 SYPD. The
patch-only selection (36 fields) gave `elm_run` 1.03x.

### Restart fields (built 2026-10-24)

`restartvar(ncid, flag, ..., data=this%x)` needs the full layout: `data` is a
pointer dummy and the restart file holds every patch. The rewrite keeps the
call and swaps its storage (`plan_restart`/`rewrite_restart`): before the call
`call compact_gather1(this%x, spel_rst1)` allocates a full-layout temporary
over `lbound:ubound(compact_cidx_p)` with the member values, the group's fill
on active special-landunit points and `spval` elsewhere (the values the full
array would hold); the call gets `data=spel_rst1`; after it, `if (trim(flag)
== 'read') call compact_scatter1(spel_rst1, this%x)` copies the members back
and the temporary is deallocated (`compact_gather2/scatter2` for rank 2). The
restart routines are the holder's `restart` binding and the concrete types'
`restart` procedures; a field whose call passes a section, that the routine
touches anywhere else, or whose call is not found, is dropped. Order at
initialization: `compact_filters_set_from_filters` (maps and counts, before
`elm_inst_biogeochem`), the Inits bind the fields to freshly allocated groups,
`restFile_read` scatters the restart values into them, the final reweight is
followed by `compact_filters_updated`, which resizes the groups (values
dropped: the step zeroes every selected field before reading it). Verified by
the crop20 restart pair (rest_cmp5: h0 identical after a restart+continue).
Unit tests never run the restart routines. Unlocked the five crop harvest
fields (`grain_to_food`, `pool_to_grain`, `grain_xfer_to_grain`,
`grain_storage_to_xfer`, `pool_to_grain_storage`).

### Still excluded

`p2c_1d_filter` slices and `pfti(c):pftf(c)` range sums in the Summary routines
and the `bounds%begc:bounds%endc` slices the balance checks pass to `c2g` (a map
argument or a compact variant would lift them), fields with several zeroing
guards, fields aliased by pointer components (`transport_ptr_list` →
`decomp_pools_sourcesink`, `decomp_pools_transport_tendency`: the alias is set
once at init and would need re-pointing after every rebind), rank-3 fields.

## Reproducibility

The rewrite is generated: located by content, checked against the analysis
(a site whose text doesn't match stops it), and it refuses sources that
already have `CompactBindMod.F90` or `.spel-compact.json`. The manifest
records the SPEL and E3SM commits, the groups (guard, fields in slot order),
the history fields and the files. After a rebase: drop the generated
changes, `spel analyze`, run `spel compact rewrite` again.

## Unit tests and captures

The unit test reads and writes the full layout (nc_io allocates the type
components; `duplicate_sets` offsets them). `write_compact_module`
(unit_test.py, from the manifest and the case's active instances/components)
writes `SpelCompactMod.F90` into the case: after `elm_init` the IO arrays are
remembered as shadows (`!#COMPACT_INIT`); before `compute` the process filters
are built from `filter(nc)` the first time, the groups created and bound,
then every step the fields are bound and each clump's slice packed from the
shadows (`!#COMPACT_BEFORE`); after `compute` the slices go back and the
fields point at the shadows again (`!#COMPACT_AFTER`). `cgrp_id` is a run-time
component (never captured, `RUNTIME_COMPONENTS`); CompactLayoutMod's
variables are run-time state (`RUNTIME_STATE_MODULES`). OpenACC GPU builds
stop; multicore runs in host memory.

Captures from rewritten sources (`instrument_elm`, `capture_pieces`): before
`write_elmtypes` every compact field is gathered into a full-layout scratch
array through the inverse map (member -> value, -1 -> group fill, else spval)
and the component points at it; after, the embeddings are rebound. The files
hold what a build without the layout holds. Captures need one thread per
process (the components are shared).

## Open items

- `CarbonIsoFluxCalc` gets an optional `compact_off` (see above); the isotope
  path is not exercised by the crop20 case.
- The deferred `ch4_prod` fields (DEFERRED.md); the unity fast path for compact
  history fields (measure first); the exclusions listed under "Still excluded".
- crop20 (1 rank, January; also a 4-rank or OpenMP build) does not exercise
  everything: the `decomp_pools_leached` fault only showed on the stride-8
  summer case (128 ranks). Validate a new selection there too (1-day run).
- GPU: `compact_groups`, `compact_cidx_*` are `declare create`d and the groups
  `enter data`; the generated bind does not `attach` the components (the
  flux instances have no device copies today).
- `hist_compute_test` with nvfortran multicore fails with some random seeds on
  the committed sources too (seeds 3, 4; `HIST_TEST_SEED`): a latent race in
  the test or the multicore history path, independent of the compact additions.

## Nutrient sources (after `spel nutrient ...` and `spel split`)

The shared C/N/P fields then live on a base type (`vegetation_nutrient_flux`) that has no
instances: each concrete type holds one as its component `core` (`veg_cf%core%agnpp`), N/P-only
fields on role components (`veg_nf%retrans%pool`), and the shared code is in plain module
procedures `<base>_<binding>` (`vegetation_nutrient_flux_setvalues`, `_init_history`, `_restart`).
`holders.py` describes both shapes: a *holder* is the type whose fields get compact storage and
its *embeddings* (instance, component path, element: `veg_cf`/`core`/`c12`, `veg_nf`/`retrans`/`n`).

- **Groups per holder embedding** (`veg_cf%core` and `c13_veg_cf%core` have their own groups, in
  each instance's own `cgrp_id`). The inventory
  traces 3-level targets (`Tracer.target`: the longest component path; a whole holder passed to a
  base routine is not a field), zeroing in `<base>_setvalues` counts (`build_inventory` adds it to
  the zeroing routines of the concrete types that embed the holder), and registrations come from
  the concrete Init/restart plus `<holder>_init`, `_init_history`, `_restart`, their fields
  prefixed with the component. Templated names (`nutrient_name(this%species, 'LEAF#')`) are kept
  as labels; species guards (`this%species == CN_SPECIES_N`, `is_carbon(this%species)`) are
  evaluated per element, and a guard the evaluator cannot read admits (conservative: history →
  unpacked, restart → excluded). `hist_default(...)` is not `inactive`.
- **Plain procedure on the base type.** `<base>_compact_bind(this)` with `type(<base>) :: this`,
  placed before `end module` of the holder's file and made public. Not type-bound: `resolve_nested` (DerivedType.py) only flattens a component type
  into `core%field` entries when it has no type-bound procedures, so a binding on the base type
  would stop SPEL's analysis of the rewritten sources.
- **Bind where the fields were allocated**: `<base>_init` for the base holder, the concrete Init
  for a role holder (after its last dropped `allocate`). The concrete module gets `use <holder
  module>, only : <base>_compact_bind` when the holder lives in another module (after `spel split`).
- **Zeroing**: a base holder's fields are dropped from `<base>_setvalues` and zeroed by group there
  (`this%x(k0p+1:k0p+num)` under `compact_bound`); a role holder's from each concrete SetValues under
  `this%<path>%` (`this%retrans%pool`, `this%retrans%cgrp_id(k)`). The zeroing class text of
  the base routine may name `this%species` (`.not. (is_carbon(this%species) .and. use_fates)`):
  it is emitted as is, inside the routine it came from.
- **Selection**: `--type vegetation_nutrient_flux` selects the shared fields; a concrete type's
  own fields (`veg_cf%gpp`) are still selected with `--type vegetation_carbon_flux`, whose nested
  fields are excluded with "lives on <holder>: pass --type <holder>". Both can be passed in one
  run. With a `.spel-nutrient.json` in the sources and no explicit `--type`, the default concrete
  types are mapped to their family base types (`--roles` adds the role holders).
- **CompactBindMod** rebinds every embedding (`call <base>_compact_bind(veg_cf%core)`). The manifest
  records `embedded_in` per type and the nutrient manifest's provenance.
- **Unit tests / captures**: driven by the manifest (`compact_fields`), see above.
- **Order on a checkout**: `spel analyze` → the nutrient steps (`nutrient_refactor.sh`, each
  re-analyzes) → `spel analyze` → `spel compact rewrite` → `spel analyze` for unit tests.

## What is read from parses and records, what is edited by text

Information comes from the analysis and the parser, never from pattern matching on the
source text: the trace and the access records give the fields, index forms and zeroing;
`parse_routine`/`parse_module_file` give registrations (their `if` conditions are kept as
expression trees and evaluated for carbon_type and species), the positions of type
definitions, the last component, `end type`, `end module`, the first use statement, the
module's imports, and routine ends; `untraced_uses` reads the access records of analyzed
routines the trace never visited. The only text search left is for lines outside every
analyzed routine (there is nothing parsed to read there), and it is reported as such.

Regular expressions remain where the source text itself is edited (not read): the
`x(p)` -> `x(k0p+fp)` substitution of a designator the record identified, appending an
argument to a call, the indentation of inserted lines, and the associate-name replacement
of a blocked loop. Those are textual edits of statements whose content is already known.
