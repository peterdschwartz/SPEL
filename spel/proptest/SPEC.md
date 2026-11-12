# `<name>.proptest.toml`: the spec format

A spec describes property-based tests of one SPEL unit-test case: a synthetic domain, one or more
*scenarios* (how the inputs are drawn) and *checks* (what must hold after the routine ran).
`spel proptest init CASE` writes a starter; `spel/proptest/examples/canopyfluxes.proptest.toml`
is a finished one, and `canopyfluxes_tutorial.ipynb` next to it explains how it was arrived at.

Every name in a spec is checked against SPEL's analysis of the case when the file loads. An
unknown name, an unknown key, a rule for something that is not an input, or an input without a
rule stops loading or the run: nothing is defaulted without a word.

- [Names and expressions](#names-and-expressions)
- [`[case]`](#case), [`[settings]`](#settings), [`[domain]`](#domain), [`[namelist]`](#namelist), [`[units]`](#units), [`[functions]`](#functions)
- [`[scenario.<name>]`](#scenarioname): [rules](#rules), [`derive`](#derive), [`assume`](#assume)
- Checks: [`[[property]]`](#property), [`[[relation]]`](#relation), [`[compare]` and `[[differential]]`](#compare-and-differential)
- [Multi-step runs](#multi-step-runs): `steps`, `time`, `forcing`, `between`, `keep`; `prev%`, `step`, `tend`, `child_*`
- [Commands, files and exit codes](#commands-files-and-exit-codes)
- [Limits](#limits), [Extending](#extending)

## Names and expressions

Variables are named as the routine names them:

| Form | Example |
|---|---|
| instance and field | `veg_es%t_veg`, `col_ws%h2osoi_liq` |
| an `associate` name of the routines under test | `t_veg`, `forc_t` |
| a scalar argument `elm_drv` passes to the routine | `nextsw_cday` |
| a module parameter or constant | `nlevgrnd`, `tfrz` |
| a per-step scalar of the case | `dtime_mod`, `nstep_mod` |
| a filter | `filter%nolakeurbanp`, `filter%num_nolakeurbanp` |

`spel proptest show CASE --kind input` (or `inout`, `output`, `parameter`, `topology`) lists them
with dimensions and units. The kinds are:

| Kind | Meaning | In a spec |
|---|---|---|
| `input` | read, never written first | needs a rule |
| `inout` | read before its first unconditional write | needs a rule |
| `output` | written before any read | reset to NaN before every call |
| `parameter` | read, no subgrid dimension (pftcon arrays) | from the reference capture; may take a rule |
| `topology` | subgrid indices, types, weights (integer and logical fields of the `_pp` types, except physical data such as `col_pp%nlevbed`, which is an input: `inventory.PHYSICAL_INTEGERS`) | set by the domain |
| `unused` | captured but not touched, or only used under namelist options that are off | not available |

The kinds depend on the [namelist options](#namelist) of the run: `show CASE` counts every
branch as running, `show SPEC` classifies under each scenario's options.

Expressions (`derive`, `assume`, `expr`, `follow`, `expect`) are Fortran expressions, parsed by
SPEL's parser and evaluated with numpy for all elements at once.

- The element variable of the level a check runs over (`p` for `over = "p"`) is the element.
  `g`, `t`, `l`, `c` are its gridcell, topounit, landunit and column. So `t_grnd(c)` and
  `t_veg(p)` can be used together. A finer level's variable is defined only when it is the same
  element: over `l`, `c` exists only if every landunit has one column (each level has as many
  elements).
- `x(p)` is one value per element, `x(p, :)` all layers of each element, `x(p, 1:nlevsoi)` a
  section. A whole array must be subscripted.
- Operators are Fortran's (`.and.`, `.not.`, `==`, `**`, integer division truncates).
- Intrinsics: `abs sqrt exp log log10 sin cos tan asin acos atan atan2 sinh cosh tanh min max mod
  modulo sign real int nint floor ceiling merge epsilon huge tiny isnan` and the reductions
  `sum product maxval minval any all count` (with `dim=`, `mask=`). A reduction of `x(p, :)`
  reduces the layers of each element.
- In a check, a bare name is the value **after** the run. For a variable the routine reads and
  writes (`inout`), `input%x` is the value it had before.

Functions besides the intrinsics:

| Function | Value |
|---|---|
| `finite(x)` | finite and not ELM's fill value (`abs(x) < 1e35`); for `x(p, :)`, every layer |
| `between(x, lo, hi)` | `lo <= x <= hi` (both ends included) |
| `close(a, b, rtol=1e-12, atol=0)` | `abs(a - b) <= atol + rtol * abs(b)`; NaN is never close |
| `agree(a, b, rtol=0, atol=0)`, `same(a, b)` | equal within the tolerances (default: identical), or both NaN |
| `implies(a, b)` | `.not. a .or. b` |
| `integral(x, dz)` | `sum(x * dz)` over the last dimension |
| `cumsum(x)` | running sum along the last dimension |
| `rel_change(new, old)` | `(new - old) / max(abs(old), 1e-30)` |
| `tend(x)` | x after the step minus x before it (`x - prev%x`; one call: `x - input%x`); any expression of the state, e.g. `tend(child_sum(col_ws%h2osfc, 't'))` |
| `acc(x)` | x summed over the steps so far (each step's value in its own state; one call: x), e.g. `acc(child_wsum(col_wf%qflx_runoff, 'g'))` |
| `elevation(t)`, `lowest(t)`, `uphill_count(t)` | from the domain declaration: a topounit's declared elevation (NaN without one), whether it is the lowest of its gridcell (ties included), how many other topounits point to it with ELM's `top_pp%downhill_ti` |
| `child_sum(x, 't')`, `child_wsum`, `child_mean`, `child_min`, `child_max` | the whole variable x of a finer level reduced over the descendants of each element of level `'g'`, `'t'`, `'l'` or `'c'` (`wsum`: weighted by their weights relative to that level, as ELM's p2c/c2l); the level's element variable must be bound (a check over that level or a finer one) |
| `active_dir(p, "<output>", k=1)` | coordinate of the element's drawn inputs along direction `k` of the saved active subspace of that output (run `spel proptest subspace` first) |

## `[case]`

```toml
[case]
name = "canflux"                             # unit-tests/canflux; or  dir = "path/to/case"
build = { fc = "nvfortran", acc = "MULTICORE" }   # optional; also dbg = true
```

Without `build` the library is the case's default build (`<case>/build-py`: `$SPEL_FC`, else the
machine's default compiler, else gfortran; `ACC=OFF`). Another build goes to
`<case>/build-py-<fc>-<acc>[-dbg]`. `--fc` and `--acc` on the command line take precedence.

## `[settings]`

Passed to Hypothesis: `max_examples` (default 100), `seed`, and any other setting of
`hypothesis.settings` (`derandomize`, `phases`, `verbosity`, ...). `-n` and `--seed` of
`spel proptest run` override them. With a seed, Hypothesis does not use its example database.

## `[domain]`

The routine runs on a synthetic domain with ELM's subgrid hierarchy: gridcells, topounits,
landunits, columns and patches, each with a weight in its parent. Gridcells are declared one by
one (`[[domain.gridcell]]`), each `copies` times, and the whole set is repeated `nsets` times
(`hierarchy.py`; the Python API builds the same tables, see [Python API](#python-api)).

```toml
[domain]
nsets = 2
reference = "cf_val"

[[domain.gridcell]]
name = "mixed"                       # reports, queries; `permute` exchanges gridcells of one name
copies = 2
landunits = [                        # one topounit
  { kind = "soil", weight = 0.7, patches = [{ pft = 1, weight = 0.6 }, 13] },
  { kind = "urban", density = "md", weight = 0.2 },
  { kind = "lake" },
]

[[domain.gridcell]]
name = "hillslope"
topounits = [                        # several topounits
  { weight = 0.5, elevation = 300.0, landunits = [{ kind = "soil", patches = [1, 13] }] },
  { weight = 0.5, elevation = 150.0, landunits = [{ kind = "soil", patches = [13] }] },
  { weight = 0.0, elevation = 50.0, landunits = [{ kind = "soil", patches = [1] }] },   # inactive
]
```

| Landunit | `lun_pp%itype` | Columns (`col_pp%itype`) | Patches |
|---|---|---|---|
| `{ kind = "soil", patches = [...] }` | `istsoil` | one soil column (1), as ELM's natural vegetation | one per pft (`pft` or `{ pft, weight }`) |
| `{ kind = "soil", columns = [{ patches = [...], weight }] }` | `istsoil` | several soil columns | per column |
| `{ kind = "crop", cfts = [...] }` | `istcrop` | one per CFT (`istcrop*100 + cft`); a CFT's weight is its column's | one each |
| `{ kind = "urban", density = "md" }` (`roof`, `road_perv`: 0.5) | `isturb_tbd/hd/md` (`urbpoi`, `ifspecial`) | roof, sunwall, shadewall, impervious and pervious road (71-75), `set_landunit_urban`'s weights for `roof`/`road_perv` | one bare (`noveg`) each |
| `{ kind = "lake" }` | `istdlak` (`lakpoi`, `ifspecial`) | one (`istdlak`, `is_lake`) | one bare |

- **Weights.** A missing weight shares what the given ones leave; the children of every element
  must sum to 1. `wtgcell`, `wttopounit`, `wtlunit`, `wtcol` are the products up the tree.
- **Activity.** A topounit with weight 0 is inactive, and so is everything under it, as are
  elements with weight 0 (subgridWeightsMod without its special cases). `setFilters` skips them.
- **Layout.** Depth first, every element's children contiguous, as ELM lays out a clump; set k of
  every level is a contiguous range. Element counts differ by level.
- **Topounits.** `top_pp%topo_grc_ind` is the topounit's number within its gridcell;
  `max_topounits` and `has_topounit` are decided by the declaration (setting them is an error).
  With declared `elevation`s, `top_pp%downhill_ti` follows `set_topounit`'s algorithm (including
  its quirk: the lowest topounit keeps the previous `dn_index`) and `top_pp%elevation` is pinned
  to them: a scenario cannot rule it.
- **Index fields** (`*_pp%gridcell/topounit/landunit/column`, `topi/topf`, `lndi/lndf`, `coli/colf`,
  `pfti/pftf`, `ntopounits` ... `npfts`, `grc_pp/top_pp%landunit_indices`) come from the layout;
  `spel proptest show SPEC` prints the gridcells and the element counts, and
  `Session.built().check()` compares what the case built with the declaration.

| Key | Default | Meaning |
|---|---|---|
| `gridcell` | required | the `[[domain.gridcell]]` tables: `name`, `copies` (1), and `landunits` (one topounit) or `topounits` |
| `nsets` | 1 | copies of the declared gridcells; relations need an even number (base and follow-up halves) |
| `snow_layers` | 0 | `col_pp%snl = -snow_layers` |
| `reference` | none | a capture (name under `unit-tests/input-data`, or a directory): constants, parameters and level extents come from it |
| `constants` | `{}` | scalar constants the reference lacks or should not decide; namelist options go in [`[namelist]`](#namelist) |
| `time` | `{}` | per-step scalars, e.g. `{ jday_mod = 15, secs_curr = 0 }` |
| `dims` | `{}` | values for names in dimension bounds the reference does not give, e.g. `{ nlevsno = 5 }` |
| `init` | by the case | `"direct"`: the worker allocates the variables and sets their values from Python (no inputs file); `"files"`: the case reads a synthetic inputs file, as a unit test reads a capture. Unset: direct for cases made since direct init existed, files for older ones (`show` says which). Both give the same results |

What is not given:

- A constant the routines read that neither the reference nor `constants` has stops the build.
- A parameter the routines read that the reference lacks (or has with another shape) stops the
  run, unless the scenario gives it a rule.
- Per-step scalars not in `time` take the defaults of `domain.TIME_DEFAULTS` (noon on 1 July 2000,
  step 1, 1800 s), then the constants file, then 0. `spel proptest show SPEC` lists them.
- A subgrid structure field the builder has no rule for stops the build
  (`domain.topology_value`).

## `[namelist]`

```toml
[namelist]
soil_stress_method = 0
use_c13 = false
```

The namelist options of the runs. An entry is a namelist variable the routines test in an `if`
(`show` lists them), or another scalar switch they read (`soil_stress_method`, which ELM sets in
code). The options are ordinary constants of the case, so a run can change them without a
rebuild. The value of an option is, in order: the reference's constants file,
`[domain.constants]`, `[namelist]`, the scenario's `namelist`, and for the follow-up of a
relation its `namelist`.

The spec is read under these options:

- A variable that is only used in branches the options turn off is `unused`: it needs no rule,
  and `init`, `probe`, `stats` and `outputs = "all"` leave it out. With `use_c13` off, the
  twelve C13/C14 fields of CanopyFluxes are gone from the spec.
- Naming such a variable in `derive`, `assume` or a check is an error when the spec loads, and
  the message names the options: "c13_psnsun_patch has no value in scenario snowfree_day: it is
  only used under other namelist options (here use_c13 = .false., use_cn = .true.)". Restrict
  the check to scenarios where it is live (`scenarios = [...]`), or use `follow%` in a relation
  that turns the option on.
- A *rule* for an input that is not read under a scenario's options is allowed and dormant
  there: its value is drawn but not set. `show SPEC` lists dormant rules. This is how the
  follow-up of a namelist relation gets the inputs only it reads.

A scenario runs under other options with `namelist = { use_hydrstress = true }` in its table, and
a relation compares two settings with [`namelist`](#relation). Each distinct set of options is
its own worker process on the same library.

Only namelist variables remove variables from a spec. A switch that is not a namelist variable
(`soil_stress_method`) can be set and compared the same way, but the branches it selects count
as ordinary `if`s. An option that is on in no capture may need inputs or parameters the
reference never recorded: the run then says which rules, `[domain.dims]` entries or parameter
values are missing.

## `[units]`

```toml
[units]
"canopystate_vars%elai_patch" = "m2/m2"
```

Units are documentation (`show`, the `stats` report). They come from ELM's history and restart
registrations; this table adds or replaces them.

## `[functions]`

```toml
[functions]
modules = ["helpers.py"]        # relative to the spec
```

```python
# helpers.py
import numpy as np
from spel.proptest.functions import register

@register
def qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / (p - 0.378 * es)
```

`qsat(forc_t(t), forc_pbot(t))` then works in every expression of that spec. Arguments arrive as
numpy arrays (one value per element, or elements x layers).

## `[scenario.<name>]`

```toml
[scenario.snowfree_day]
over = "p in nolakeurbanp"                   # default "p": every element
namelist = { use_hydrstress = true }         # options of this scenario, over [namelist]
extends = "canopyfluxes.inputs.toml"         # rule files (one or a list), e.g. from `stats`
assume = ["t_grnd(c) >= forc_t(t)"]          # keep only elements where this holds
trap_fpe = false                             # halt on invalid, divide by zero, overflow

[scenario.snowfree_day.inputs]
"top_as%tbot" = { range = [278.0, 313.0] }

[scenario.snowfree_day.derive]
"top_as%thbot" = "top_as%tbot(t)"
```

`over` selects the elements that are checked: an element variable, optionally `in` a filter of
the case. A check may have its own `over`.

Every input the routines read under the scenario's namelist options must have a rule or a
`derive` expression, from the scenario or the files it `extends`. `spel proptest show SPEC` lists the ones
without. A rule in `inputs` replaces the one from a rule file.

### Rules

| Rule | Example | Value |
|---|---|---|
| a bare value | `"col_ws%snow_depth" = 0.0` | constant |
| `range` | `{ range = [0.1, 6.0] }` | uniform |
| `log_range` | `{ log_range = [1e-6, 1e-2] }` | log-uniform |
| `values` | `{ values = [0, 1, 2] }` | one of them |
| `dist` | `{ dist = "lognormal", mean = 0.0, sigma = 1.0, min = 0.1, max = 10.0 }` | `uniform`, `loguniform` (need `min`, `max`), `normal`, `lognormal` (need `mean`, `sigma`; `min`/`max` clip) |
| `scale` | `{ scale = [0.8, 1.2] }` | the variable's own value times a factor (parameters) |

Any rule may add:

- `clip = [lo, hi]`;
- `pft = { 13 = { range = [0.3, 1.0] } }`: another rule for the elements of a pft;
- `levels = { "1" = 0.0, "2:nlevsoi" = { range = [0.1, 0.4] } }`: another rule for a section of
  the first dimension that is not the subgrid one (Fortran indices).

An override without a `clip` of its own keeps the base rule's. 
Parameters take `range`, `values`, `scale` or a value.

For each input Hypothesis draws a *mode*: one value for all elements; independent values for
every entry; edge values (each entry at the low or high end); or, for layered inputs,
independent per element with one value down the layers. Shrinking moves towards the first, so a
minimal failing example has as many constant inputs as possible.

### `derive`

An input computed from other inputs, parameters and constants, per element:

```toml
"canopystate_vars%laisha_patch" = "elai(p) - laisun(p)"
"col_pp%z" = "cumsum(dz(c, :)) - 0.5 * dz(c, :)"
```

Rules are applied in dependency order; a cycle is an error. `derive` is applied again to the
follow-up half of a relation, except to the variables its `follow` sets.

### `assume`

A list of logical expressions of the inputs. Elements where one is false are not checked; an
example with no element left is discarded. (The older form `[scenario.x.assume] exprs = [...]`
still loads.) `derive`, `assume` and `follow` run before the routine, so they cannot use its
outputs.

## `[[property]]`

```toml
[[property]]
name = "soil water stress in [0, 1]"
expr = "between(btran(p), 0.0, 1.0)"
scenarios = ["snowfree_*"]      # default: every scenario (globs)
over = "p in nolakeurbanp"      # default: the scenario's
```

`expr` gives one logical per element.

## `[[relation]]`

```toml
[[relation]]
name = "more absorbed shortwave does not cool the leaf"
follow = { "solarabs_vars%sabv_patch" = "1.1 * solarabs_vars%sabv_patch(p) + 10.0" }
expect = "follow%t_veg(p) >= base%t_veg(p) - 1.0e-3"
```

The first half of the sets (*base*) gets the drawn example. The second half (*follow-up*) gets
a copy, changed by `follow`: input = an expression of the base inputs. One call runs both, and
`expect` compares them per base element through `base%x` and `follow%x`.

- Without `follow` the halves are copies. `same(follow%x(p), base%x(p))` then checks that a
  result does not depend on where the element sits.
- `permute = true` also reorders the follow-up gridcells among those of the same declaration,
  each with everything under it.
  `follow%x(p)` is still the follow-up of base element `p`. This finds a result that depends on
  a neighbouring element, which identical halves hide.
- A variable without a subgrid dimension is shared by both halves and cannot be in `follow`.

With `namelist`, the follow-up is the same example under other namelist options:

```toml
[[relation]]
name = "C13 on: the fluxes are identical and the 13C flux is finite"
namelist = { use_c13 = true }
expect = """same(follow%t_veg(p), base%t_veg(p))
            .and. finite(follow%photosyns_vars%c13_psnsun_patch(p))"""
```

- Options are module scalars, so the two runs cannot share a call: the follow-up runs in a
  second process on the same library. `base%x` and `follow%x` are the two whole runs, so `nsets`
  need not be even. `follow` transforms may be given too; `permute` may not.
- `follow%x` is checked against the follow-up's options, `base%x` and bare names against the
  scenario's. Above, `c13_psnsun_patch` exists only in the follow-up.
- Inputs only the follow-up reads need rules in the scenario, where they are dormant for the
  base run.
- Typical relations: an option that should not matter for these outputs (`same`), two settings
  that select the same code, an invariant that must hold under either setting, a known direction
  of change.

## `[compare]` and `[[differential]]`

```toml
[compare]
case = "canflux_acc"                  # another case (or dir = "..."); omit for this case
build = { fc = "nvfortran", acc = "MULTICORE" }   # and/or another build

[[differential]]
name = "the refactored routine gives the same answers"
outputs = ["t_veg", "eflx_sh_veg"]    # default "all": every output of the case
rtol = 1.0e-9                         # and atol; default: identical
```

Each example runs on both: this case (`ref%x`) and the one `[compare]` names (`dev%x`), on the
same domain. The listed outputs must agree (`agree`, so a value neither wrote is not a
difference). `expect = "..."` replaces `outputs`/`rtol`/`atol` with any expression of `ref%` and
`dev%`. The other case must have every input the scenario sets, with the same shapes, and no
input of its own left without a rule.

Uses: a refactored copy of a routine against the original over a whole scenario, not only the
regime one capture covers; `ACC = "MULTICORE"` against `OFF` as a race test; two compilers.

## Multi-step runs

A scenario with `steps = N` calls the routines N times per example, carrying their state: the
state after step k is what step k+1 starts from, and the time scalars advance by `dtime_mod` on
ELM's no-leap calendar (`timestep.py`: what elm_drv sets each step, `nstep_mod`, the date and
`jday_mod`, and the others consistent with that date). The checks are judged after every step.

```toml
[scenario.season]
over = "c in hydrologyc"
extends = "hs_season.inputs.toml"
steps = 96                                   # default 1: one call
time = { mon_curr = 7, day_curr = 1, secs_curr = 1800, jday_mod = 182, thiscalday_mod = 182.02 }  # step 1
keep = ["col_es%t_soisno", "col_ws%*"]       # boundary variables left to the process

[scenario.season.inputs]                     # drawn once: the state before step 1
"top_ws%from_uphill" = 0.0

[scenario.season.forcing]                    # set before every step
"top_af%rain" = { range = [0.0, 5e-3] }      # a rule: drawn for every step (modes below)
"top_af%snow" = "0.0"                        # or an expression of the state before the step

[scenario.season.between]                    # after every step, in this order, on the state after it
"top_ws%from_uphill" = "max(0.0, top_ws%from_uphill(t) - dtime_mod * child_sum(col_wf%qflx_from_uphill, 't'))"

```

**The boundary.** Between two calls ELM runs the rest of its step, and some of those routines
write variables the case's routines read: CanopyHydrology writes `h2osoi_liq`, ColWaterBalanceCheck
drains `top_ws%from_uphill`, and so on. SPEL finds them in the static trace of the whole step
(`coupling.py`: elm_drv's trace on the full analysis, as the database's call-tree view uses; the
writes outside the case's routines that run under the scenario's namelist options, on filters that
can share an element with the case's own accesses). Every such variable the routines read (an
input or inout) must be declared, in one of:

| Table | Meaning |
|---|---|
| `forcing` | set before every step: a rule (drawn per step) or an expression (bare names and `prev%x`: the state before the step; `step`; time scalars) |
| `between` | assigned after every step from the state after it (bare names; `prev%x` is the state before the step); the checks of the step see the assigned value, the next step starts from it |
| (all three) | a `forcing` or `between` entry whose target is not read under a run's namelist options is dormant there, like a rule (the follow-up of a relation with IM2 off skips the IM2 store's drain) |
| `keep` | left as the process leaves it: an inout is carried, an input keeps its value (fnmatch patterns, e.g. `"col_ws%*"`); the missing writers are assumed absent |

A run with an undeclared boundary variable stops before its first example, with the variable and
its writers (routine, file:line, before/after the case, conditional). `spel proptest show SPEC`
lists the boundary variables of every multi-step scenario and how they are declared. This is the
point of a multi-step scenario: it states what the rest of ELM does between the steps instead of
leaving it to chance. Variables nothing else writes are carried (inouts) or keep their drawn
value (inputs); outputs are set to NaN before every step, as before one call. The analysis needs
the case's E3SM checkout analyzed (`spel analyze --srcroot`); the boundary is cached per case.

**Forcing draws.** A forcing rule draws over `(step, element, ...)` with the input modes: one value
for every step and element, independent values per step and element, edges, or one value per
element (and layer) held over the steps.

**Checks.** A check is judged after every step (`steps = "all"`, the default), `"last"`, one step
(`steps = 3`) or a range (`[2, 10]`); the first failing step is reported (`after step 3 of 96`).
`prev%x` is x before the judged step, `input%x` before step 1, `step` the step number, `tend(x)`
`x - prev%x`, `acc(x)` its sum over steps 1..k. Inside `tend` and `acc`, `base%`/`follow%` are the
halves (or runs) of the step concerned; before step 1 they are not available (`tend(follow%x)` needs
`steps = [2, N]`). A relation compares its two halves (or its two runs, under `namelist`) step by step;
a differential runs the same steps in both cases.

**Python.** `Session.execute` keeps every step: `res.series("top_ws%from_uphill")` (step axis
first), `res.at(k)` (the result after step k, e.g. `res.at(k).frame("t")`), `res.times[k - 1]`.
A saved failure keeps the forcing values (`forcing.npz`), and `replay` runs the steps again.

Multi-step runs need an `ACC = "OFF"` build for now (the time scalars are not copied to the device
between steps).

## Commands, files and exit codes

| Command | Does |
|---|---|
| `spel proptest init CASE [--reference REF] [--where EXPR] [--landunits KIND...] [--namelist N=V] [-o FILE]` | starter spec with a `[namelist]` table and `# TODO` lines. The domain's landunit kinds come from the filters elm_drv passes the routines (`urbanl`... -> urban, `lakec`... -> lake, any other filter or none -> vegetation; `nourbanl` next to `urbanl` adds nothing), or from `--landunits`; one gridcell per kind, two copies: a soil column with the reference's most common pfts, a town with the densities of its active urban landunits, a lake. A domain of one non-vegetated kind runs over its patch filter (`p in urbanp`); an urban one derives the canyon geometry as ELM does (`scaffold.URBAN_DERIVE`) |
| `spel proptest build CASE [--fc FC] [--acc ACC] [--dbg]` | the shim and the library (also done by the others when sources changed) |
| `spel proptest stats CASE --reference REF [--landunits KIND...] [--where EXPR] [--stride N] [--namelist N=V]` | rule file and report from a capture, for the inputs read under those options; `--landunits soil crop urban lake` keeps the patches on landunits of those kinds (and their coarser elements) |
| `spel proptest show SPEC\|CASE [--kind KIND]` | inventory; per scenario its namelist options and where each value comes from, the variables they leave unused, dormant rules, inputs without a rule; defaults in use |
| `spel proptest probe SPEC [-s SCENARIO] [-n N]` | runs a few examples: outputs never written, variables never changed, aborts |
| `spel proptest run SPEC [-s SCENARIO] [-k CHECK] [-n N] [--seed N] [--json F] [--junit F]` | the checks |
| `spel proptest replay DIR` | a saved failing example again, without Hypothesis |
| `spel proptest coverage SPEC [-s SCENARIO] [-n N] [--all]` | `if` branches of the routines' call tree the scenario could reach and never entered; those its namelist options turn off are counted (`--all` lists them) |
| `spel proptest subspace SPEC --scenario S --output EXPR [--input NAME] [--parameters]` | active subspace of an output |

`run` exits with 0 when every check passes, 1 when one fails, 2 when the spec or the command
has a problem or no check is selected. `-s` and `-k` take a name or a glob.

Files:

| Where | What |
|---|---|
| `unit-tests/<case>/build-py*/` | the libraries |
| `unit-tests/<case>/proptest/` | scratch: synthetic domains (`domain-<hash>`, and `domain-<hash>-files` when a failure of a direct-init run is saved), logs of aborted runs, the units cache. `spel create` deletes it with the case |
| `unit-tests/proptest-data/<case>/hypothesis/` | Hypothesis' example database: failing examples are tried first in the next run |
| `unit-tests/proptest-data/<case>/failures/<scenario>__<check>/` | the shrunk failing example: `spel-inputs0001.nc` and `spel-constants0001.nc` in capture format, and `example.json`; the follow-up of a namelist relation in `follow/` |
| `unit-tests/proptest-data/<case>/stats/`, `subspace/` | rule files and reports |

A failing run prints the element, the value of every name the expression uses, and the drawn
inputs at that element. If the routine stops (`endrun`, `stop`, a segfault), that is the failure,
with the end of its output.

## Python API

`spel.proptest.api` builds the same spec in Python and writes it as TOML; every piece goes
through the reader's checks when it is added (a bad name fails at once, with the TOML message):

```python
from spel.proptest import api as pt
from spel.proptest.api import Gridcell, Lake, Patch, Soil, Topounit, Urban

case = pt.Case("urbval", reference="cf_val")
dom = pt.Domain([Gridcell.of(Urban("md"), Urban("hd"), copies=2, name="town"),
                 Gridcell(name="hill", topounits=(Topounit((Soil(1, 13),), weight=0.5, elevation=300.0),
                                                  Topounit((Lake(),), weight=0.5, elevation=150.0)))],
                nsets=2, reference="cf_val")
spec = pt.Spec(case, dom, namelist={"implicit_stress": False})
base = spec.scenario("base", over="p in urbanp", extends=case.stats_rules(landunits=["urban"]),
                     inputs={"lun_pp%canyon_hwr": pt.Range(0.5, 1.0)}, derive=pt.URBAN_GEOMETRY)
spec.property("canopy air is taf", "veg_es%t_ref2m(p) == lun_es%taf(l)", scenarios=[base])
spec.to_toml("urbval.proptest.toml")          # pt.Spec.from_toml(path) reads one back
```

| | |
|---|---|
| `pt.Case(name)` | the inventory: `var("ht_roof")`, `vars(kind=, level=)`, `find("roof")`, `stats_rules(landunits=, where=)` |
| rules | `pt.Range`, `LogRange`, `Values`, `Value`, `Scale`, `Dist`, each with `clip=`, `pft={13: ...}`, `levels={"1:3": ...}` |
| `spec.view()` / `dom.view()` | the declared domain: `gridcells[k].topounits[..].landunits[..].columns[..].patches`, `table(level)` (pandas), `select(level, kind=, role=, pft=, name=)`, `expected_filters()` |
| `spec.session()` | a `runner.Session` on the validated spec |
| `s.built()` | what the case built: `filters`, `topology`, `bounds`; `check()` lists differences from the declaration |
| `s.sample(scenario, seed)` | an example drawn like a test run's (same modes), from a seed |
| `s.execute(scenario, ex)` | a `Result`: `frame(level, names=, aggregate={"x": "wsum"})` (ancestors joined, descendants reduced), `verdict(check)` (per element, without raising) |

Scripts that open a session need an `if __name__ == "__main__":` guard (the worker is a spawned
process).

## Limits

- Landunits are soil, crop, urban or lake; no glacier or wetland yet. Activity follows the
  weights only (not ELM's `all_active` or zero-weight urban special cases).
- Per-pft rule overrides (`pft = {...}`) apply to patch variables only (a column holds several
  pfts); there are no overrides per urban column type yet (use `derive` with
  `merge(..., col_pp%itype(c) == 71)`).
- Multi-step runs: CPU builds only (`ACC = "OFF"`); `between`/`forcing` are expressions of the
  state, not ELM code (a scenario states what the rest of the step does).
- Type-bound routines cannot be called by the shim yet.
- `coverage` needs gfortran and its gcov.
- Without a machine file (`spel machine`), build each compiler's library in a shell set up for
  that compiler. A run then loads each library with the netcdf-fortran it was built with.

## Extending

| To add | Write | Where |
|---|---|---|
| a rule kind | a `RuleKind` subclass with `@register` (its table check and quantile map) | `rules.py` |
| a check kind | a `Check` subclass with `@register` (its table, `prepare`, `elements`, `mask`, `namespaces`) | `checks.py` |
| a function | a function with `@register` in a `[functions]` module, or an entry of `FUNCTIONS` | `functions.py` |
| a topology field | an entry of the `TOPOLOGY_*` sets or a case in `topology_value` | `domain.py` |
| another way to run the routine | an object with `get`, `run`, `stop` passed as `Session(..., backend=...)` | `runner.py` |
