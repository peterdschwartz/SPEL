# SPEL — Software Package for E3SM Land

SPEL is a toolkit for **developing, debugging, and understanding complicated (legacy) Fortran
projects**. It began as a generator of *functional unit tests* (FUTs) for any subroutine in the
E3SM Land Model (ELM), and has grown into a general static-analysis platform.

What SPEL does:

- **Functional unit tests** — given one or more subroutine names, SPEL resolves the full
  dependency closure (modules, derived types, globals, call tree), rewrites the sources so they
  compile standalone, and emits a CMake-buildable test case in `unit-tests/<casename>`.
- **Static analysis of Fortran** — a hand-written lexer/parser/AST for Fortran drives call-tree
  construction, read/write analysis of derived-type components, and argument-access tracking
  down to the line number.
- **Configuration-aware analysis** — SPEL tracks `if` blocks guarded by namelist variables and
  their cascading dependents, so analysis results can be **tailored to a specific namelist
  configuration or ELM COMPSET**. Variables and code paths that are only reachable under, e.g.,
  `use_crop = .true.` are identified as such rather than being reported unconditionally.
- **Source-to-source modification** — script-driven edits such as OpenACC pragma insertion.
- **Database export** — analysis results can be serialized and exported as CSV for a Django web
  app that lets you browse ELM's structure and call flow in a browser.

> ELM is the only project supported today.

---

## Installation

SPEL is a normal Python package; a virtual environment is recommended. Dependencies are declared
in `pyproject.toml`.

```bash
git clone https://github.com/peterdschwartz/SPEL_OpenACC.git
cd SPEL_OpenACC
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs the `spel` console script (`[project.scripts]` → `spel.cli:main`).

### Pointing SPEL at E3SM

SPEL needs to know where your E3SM clone lives. The default comes from `SPEL_E3SM_SRCROOT`
(environment or the untracked `.spel.env`), falling back to a `dev_E3SM` checkout next to the
SPEL repo. Set it with:

```bash
spel config --set-srcroot /path/to/E3SM
```

Every command also takes `--srcroot DIR`, which overrides it for that one command (child processes
such as `spel validate`'s `spel create`s inherit it). Everything else (`ELM_SRC`, `SHR_SRC`, the
analysis cache) is derived from it. Verify with:

```bash
spel config
```

### Comparing a development branch against a reference branch

Each E3SM checkout gets its own `elm_drv` analysis cache, so two checkouts can be used side by
side without `--reanalyze`. Use the same case name (`-c`) for both:

```bash
# 1. reference data from the reference checkout (instruments it, runs ELM, then runs/validates the test)
spel create -s canopyfluxes -c canflux --run-case --srcroot ~/E3SM
# 2. the same unit test generated from the development checkout
spel create -s canopyfluxes -c canflux --srcroot ~/dev_E3SM
# 3. build/run it on the reference data and diff against the reference outputs
spel run canflux
```

Step 2 regenerates `unit-tests/canflux/` but leaves `unit-tests/input-data/canflux/` (the
reference data) alone. `spel run` prints which checkout generated the unit test and which one
produced the data (recorded in `input-data/<case>/spel_reference.json`), and notes when they
differ. Then the diffs are the answer differences between the branches. The access check uses the
development branch's analysis. The development unit test can only read the reference data if both
branches use the same variables (derived-type components) as inputs.

### Directory layout

| Path | Purpose |
|------|---------|
| `spel/cli.py` | CLI entry point / subcommand definitions |
| `spel/scripts/` | Analysis engine (parser, call trees, writers, export) |
| `spel/scripts/fortran_parser/` | Fortran lexer, parser, AST, expression evaluation |
| `spel/scripts/nml/` | Namelist / COMPSET-conditional analysis |
| `spel/scripts/ml_training/` | Experimental ML emulator tooling |
| `spel/db/` | Django app for browsing exported analysis |
| `SourceFiles/` | Fortran support files copied into every generated unit test |
| `unit-tests/<casename>/` | Generated unit-test cases |
| `unit-tests/input-data/` | NetCDF input/output data for unit tests |
| `script-output/` | Intermediate generated Fortran |
| `spel/scripts/fut_<casename>.pkl` | Serialized analysis results (see below) |

---

## CLI reference

All commands are subcommands of `spel`. Run `spel <command> --help` for the authoritative list.

| Command | Purpose |
|---------|---------|
| [`spel analyze`](#spel-analyze) | (Re)build the cached whole-driver (`elm_drv`) analysis |
| [`spel create`](#spel-create) | Extract subroutines from the cached analysis and generate a functional unit test |
| [`spel instrument`](#spel-instrument) | Instrument ELM to capture reference data; optionally build/run a CIME case |
| [`spel validate`](#spel-validate) | Create, capture (one ELM run) and validate many unit tests in parallel |
| [`spel run`](#spel-run) | Configure, build, and run a generated unit test, then validate it against ELM's data |
| [`spel diff`](#spel-diff) | Compare two netCDF files and report relative error |
| [`spel export`](#spel-export) | Turn a pickled unit test into database CSVs |
| [`spel config`](#spel-config) | Print the resolved E3SM/ELM source paths |
| [`spel upload`](#spel-upload) | rsync the netCDF interface files to a remote machine |
| [`spel repl`](#spel-repl) | Drop into an IPython shell wired to the Fortran parser |
| [`spel sample`](#spel-sample-and-spel-train-experimental) | *(experimental)* sample unit-test output for ML training |
| [`spel train`](#spel-sample-and-spel-train-experimental) | *(experimental)* train an emulator network |

### `spel analyze`

Parses and analyzes the whole call tree under `elm_driver::elm_drv` once, and caches it in
`unit-tests/.spel-cache/<srcroot name>-<hash>/elm_drv/` (one cache per E3SM checkout; `spel config`
lists them):

- `analysis.pkl` — the pickled analysis (every module, subroutine, type, access map, call tree)
- `src/` — the edited ELM sources
- `meta.json` — E3SM/SPEL commits and branches, the analyzed roots, and the analysis `failures`
  and `incomplete` routines

`elm_drv` is the analysis root, so every routine it reaches, at any depth, is analyzed once. Routines
`elm_drv` calls whose calls the edited driver comments out are analyzed as roots of their own.
Failures are isolated per routine: a routine that fails to parse or analyze is recorded under
`failures` (its callers treat it as an unknown callee), and every routine whose call tree contains
it is listed under `incomplete`. `spel create` refuses to extract a failed or incomplete routine;
everything else stays usable.

`elm_instMod` is parsed *declaration-only*: its contained routines are never analyzed or compiled
(they're commented out), only its module-level declarations, so names other modules import from it
resolve. Each unit test gets a generated `elm_instMod.F90` declaring just the instances it needs and
the names its modules import (re-exports are forwarded from their source modules).

Run it after switching branches or changing ELM/SPEL sources. `spel create` builds the cache
automatically if it is missing, and **warns** (doesn't rebuild) when the recorded commits differ
from the current checkouts. E3SM must not be instrumented (run `spel instrument --undo` first).

```bash
spel analyze
```

### `spel create`

Extracts the given subroutines from the cached `elm_drv` analysis (see [`spel analyze`](#spel-analyze))
— no re-parsing — and writes a standalone unit test to `unit-tests/<casename>`.
The subroutines must be reachable from `elm_drv`. Regardless of mode, the analysis results are **always pickled** at the
end (see [Pickled unit tests](#pickled-unit-tests)).

```bash
spel create -s <subroutine> [<subroutine> ...] [-c <casename>] [-u] [--db]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `-s` | yes | — | One or more subroutine names (case-insensitive) to build the test around |
| `-c` | no | `fut` | Case name; the case directory is `unit-tests/<casename>` |
| `-u` | no | off | Re-use the existing case directory instead of wiping and re-preprocessing it |
| `--db` | no | off | Database mode: run the analysis and pickle it, but **do not** emit Fortran unit-test files |
| `--reanalyze` | no | off | Rebuild the cached `elm_drv` analysis first (same as `spel analyze`) |
| `--srcroot DIR` | no | `SPEL_E3SM_SRCROOT` | E3SM checkout to analyze/generate from (and to instrument/run with `--run-case`); any command accepts it |
| `--instrument` | no | off | Afterwards run [`spel instrument`](#spel-instrument) for the case |
| `--run-case` | no | off | Also create/build/run the CIME case, collect its `spel-*.nc`, then [`spel run`](#spel-run) the unit test (implies `--instrument`) |
| `--freq`, `--case-args` | no | `9`, `""` | See [`spel instrument`](#spel-instrument) |

Without `-u`, an existing case directory is cleared, and stale `.pkl` and `script-output/*.F90`
files are removed before the run.

```bash
# Build a unit test named "canflux" around CanopyFluxes
spel create -s canopyfluxes -c canflux

# Analysis only — produce the pickle for database export, skip Fortran generation
spel create -s canopyfluxes -c canflux --db
```

### `spel instrument`

Prepares `$SPEL_E3SM_SRCROOT` to dump the reference data (`spel-constants/inputs/outputs*.nc`)
for a unit-test case. It needs no source scanning of its own: it reuses the analysis `spel create`
saves in `unit-tests/<case>/fut.pkl` (re-run `create` if that file is missing or predates this feature).

Every case gets a *tag* (its name, lowercased, non-word characters → `_`, ≤ 40 characters), so
several cases can be instrumented for one build (see [`spel validate`](#spel-validate)).

1. Writes `SpelCapture_<tag>.F90` (`spel_capture_inputs_<tag>/outputs_<tag>(bounds)`; capture every
   `--freq` model steps, files of 720 captures, written as `<tag>.spel-*.nc` in the RUNDIR). A copy
   is kept in `unit-tests/<case>/elm-capture/`.
2. Inserts one `use SpelCapture_<tag>` line and a capture call before/after the call site(s) of the
   case's routines found during `create`. Routines `elm_drv` calls directly are captured in
   `elm_driver.F90`. Deeper routines (e.g. `SoilLittVertTransp`) are captured in their caller on the
   shortest call chain from `elm_drv` (`EcosystemDynMod`), using the caller's `bounds`. `main.F90`
   then calls them with the actuals `elm_drv` passes down that chain, via keyword arguments
   (`filter(nc)%...`, `bounds_clump`, globals). Restrictions: all selected routines must have the
   same caller. A dummy bound to a local of an intermediate routine must be an intrinsic scalar;
   `main.F90` passes an uninitialized `spel_<dummy>` for it and `create` warns. Inserted
   lines end in `!#SPEL`, so re-running replaces them and `--undo` removes them. Hand-written capture
   code must be removed first. If the caller's file changed since `create`, it stops and asks you to
   re-run `create`.
3. Makes the module variables/types the IO modules use public and drops `protected`, editing only
   the declaration and `private`/`protected` statement lines recorded during `create`. `spel create`
   makes the same edit in the unit-test copies of those modules, so ELM needs no hand edits.
4. Copies `nc_io` and `nc_allocMod`, the case's `ReadWriteMod` and `FUTConstantsMod` renamed
   `ReadWriteMod_<tag>`/`FUTConstantsMod_<tag>`, and `SpelCapture_<tag>` to
   `components/elm/src/main`. IO modules from earlier instrumentations are deleted first.
   `--undo` deletes them too.

```bash
spel instrument canflux --dry-run                 # report only
spel instrument canflux --run-case --case-args "--stop-n 10 --stop-option ndays"
spel instrument --undo
```

`--run-case` calls the casegen script (`$SPEL_CASEGEN`, default
`e3sm_casegen.sh` in the SPEL repo) with `--case <case> --srcroot ... --build --submit`
plus `--case-args`. The script must print `CASEDIR=<path>`. The `<tag>.spel-*.nc` files in the case's
RUNDIR are then copied to `unit-tests/input-data/<case>` as `spel-*.nc`. **This overwrites any existing reference data there.**
Finally the unit test is built, run, and validated as by [`spel run`](#spel-run); a failed validation
gives a non-zero exit code.

### `spel validate`

The validation harness: one ELM build and run captures reference data for many unit tests.

```bash
spel validate -s canopyfluxes soillittverttransp -j 4
spel validate --list cases.txt --case-args "--stop-n 2 --stop-option ndays"
```

1. Builds the `elm_drv` analysis cache if it is missing (or with `--reanalyze`).
2. Runs `spel create` for every case in parallel (`-j` processes, each with its own
   `SPEL_OUTPUT_DIR`; logs in `unit-tests/.validate-logs/`).
3. Instruments ELM for all cases at once, as in [`spel instrument`](#spel-instrument). A case that
   can't be instrumented is reported and skipped.
4. Runs one CIME case (`--run-name`, default `spel-validate`) with the casegen script. A failed
   run (not build) is resubmitted once with `./case.submit`; a failed build reports the compiler
   errors from the build log. Then it copies each case's `<tag>.spel-*.nc`, plus the run's
   `lnd_in`, to `unit-tests/input-data/<case>`.
5. Removes the instrumentation (unless `--keep-instrumentation`).
6. Runs `spel run <case>` for every case in parallel (logs in `unit-tests/<case>/validate.log`).
7. Prints a summary and writes `unit-tests/validate-report.json`. The exit code is 1 unless every
   case passes.

`-s` makes one case per routine, named after the routine. A `--list` file has one case per line,
either `case: sub1 sub2` or just `sub`, with `#` comments. To split up an enormous routine such as
`ecosystemdynnoleaching2`, list its children as separate cases. Statuses:

| Status | Meaning |
|--------|---------|
| `PASS` | bit-for-bit and access analysis consistent |
| `FAIL` | the unit test failed to build or run, or failed validation |
| `CREATE_FAILED` / `INSTRUMENT_FAILED` | see the log or detail in the report |
| `NOT_CALLED` | ELM never reached the call site at a capture step (inactive for this configuration; check `lnd_in`) |
| `ELM_RUN_FAILED` | the CIME build or run failed |

`--skip-create` reuses existing cases. `--dry-run` creates the cases and reports what would be
instrumented without touching ELM.

### `spel run`

Runs `check_config.sh` (which configures CMake and runs `make`) and then executes
`./build/elmtest` inside the case directory. Then, for each `spel-outputsNNNN.nc` in
`unit-tests/input-data/<case>`:

- **bit-for-bit**: `fut-outputsNNNN.nc` must match ELM's `spel-outputsNNNN.nc` (see `spel diff --test`)
- **static analysis**: `spel-inputsNNNN.nc` vs `spel-outputsNNNN.nc` is checked against `spel_access.json`
  (see `spel diff --inputs`)

It ends with a `PASS`/`FAIL` summary and exits non-zero on any failure. Use `--no-diff` to skip validation.

```bash
spel run [--no-diff] [<casename>] [-- <args passed to elmtest>]
```

If `<casename>` is omitted, the **current working directory** is treated as the case directory —
convenient when you're already inside `unit-tests/<casename>`.

The test executable accepts up to two positional arguments:

| Arg | Meaning | Default |
|-----|---------|---------|
| 1 | number of clump sets (`nsets`) | 1 |
| 2 | sites per clump (`pproc_input`) | 1 |

```bash
spel run canflux            # defaults
spel run canflux 4 2        # 4 clump sets, 2 sites per clump
cd unit-tests/canflux && spel run
```

Compiler, debug, and GPU settings live in the case's `check_config.sh`; it detects a mismatch
with the existing `CMakeCache.txt` and offers to reconfigure.

### `spel diff`

Compares two netCDF files — typically a reference and a test output written by SPEL's
`ReadWriteMod::write_elmtypes` — and reports relative errors. Exits non-zero if any variable differs
(relative error > 1e-10). `spel run` does this automatically for every captured file.

```bash
spel diff --ref <reference>.nc --test <test>.nc [-v <variable>]
```

| Flag | Required | Meaning |
|------|----------|---------|
| `--ref` | yes | Reference netCDF file |
| `--test` | yes | Test netCDF file |
| `-v` | no | Restrict the report to a single variable |

### `spel export`

Loads a pickled unit test and writes one CSV per database table into
`spel/db/app/management/commands/csv/`.

```bash
spel export -c <casename>
```

> **Note:** `-c` is the **casename** of the pickle (`fut_<casename>.pkl`), *not* a git hash.
> A common convention is to name cases after the E3SM commit they were generated from, which is
> why older docs described this flag as a commit — but any casename works.

Tables emitted include modules and module dependencies, subroutines, subroutine arguments, the
call tree, derived-type definitions and instances, active derived-type variables per subroutine,
intrinsic globals, namelist-guarded `if` blocks, namelist cascades, per-line argument access,
call bindings, and propagated access.

### `spel config`

Prints the resolved source roots and analysis cache, and lists the cache of every checkout.

```bash
spel config                      # or: spel config --srcroot /other/E3SM
# E3SM SRCROOT: /path/to/E3SM
# ELM SRC     : /path/to/E3SM/components/elm/src
# SHR SRC     : /path/to/E3SM/share/util
# Cache       : .../unit-tests/.spel-cache/E3SM-2e665a5f/elm_drv
# ...
# elm_drv analysis caches:
#   /path/to/E3SM  master@34d78535b7  built 2026-10-10T17:40:12  -> E3SM-2e665a5f
spel config --set-srcroot /path/to/E3SM   # persist the default in .spel.env
```

### `spel upload`

rsyncs the netCDF interface Fortran files (`FUTConstantsMod.F90`, `nc_allocMod.F90`, `nc_io.F90`,
`ReadWriteMod.F90`) from the current directory to a remote machine.

```bash
spel upload <remote_host> <destination_path>
```

### `spel repl`

Starts an embedded IPython shell with autoreload enabled and `parse_line` from
`spel.scripts.fortran_parser.spel_repl` already in the namespace — handy for interactively
poking at the Fortran lexer/parser.

```bash
spel repl
```

### `spel sample` and `spel train` (experimental)

Early-stage tooling for training neural-network emulators of ELM subroutines from unit-test
output. These paths require PyTorch, which is **not** a declared dependency and must be
installed separately. Interfaces are unstable.

```bash
spel sample -c <casename> -n <samples_per_file>
spel train  -c <casename>
```

---

## Pickled unit tests

Every `spel create` run ends by serializing the whole `FunctionalUnitTest` object — module dict,
subroutine dict, derived-type dict, namelist-guarded usage, and configuration — to:

```
spel/scripts/fut_<casename>.pkl
```

### Why it matters

The pickle is the analysis artifact. Parsing all of ELM is expensive, so downstream tooling
(`spel export`, the ML pipeline, ad-hoc analysis scripts) re-loads the pickle instead of
re-analyzing the source.

### Portability across machines and checkouts

Pickles are designed to be **shared**. On export, every `filepath` on a module, subroutine, or
derived type is rewritten to be *relative to* `E3SM_SRCROOT`; on import, it is re-anchored to the
*local* `E3SM_SRCROOT`. A pickle produced on one machine therefore works on another, even when
the E3SM clone lives somewhere entirely different.

The workflow:

```bash
# Machine A — produce the analysis
spel create -s canopyfluxes -c 4637ab7 --db
# -> spel/scripts/fut_4637ab7.pkl

# Transfer fut_4637ab7.pkl to machine B's spel/scripts/ directory

# Machine B — set E3SM_SRCROOT in spel/scripts/config.py to the local clone, then:
spel config                 # sanity-check the paths
spel export -c 4637ab7      # -> spel/db/app/management/commands/csv/*.csv
```

The only requirement is that the local E3SM checkout contains the same files the pickle refers
to — matching the commit the analysis was run against is the safest choice, which is why naming
the case after the commit hash is a useful convention.

### Using a pickle programmatically

```python
from spel.scripts.export_objects import unpickle_unit_test

fut = unpickle_unit_test(casename="canflux")
fut.subroutine_dict   # dict[str, Subroutine]
fut.module_dict       # dict[str, FortranModule]
fut.type_dict         # dict[str, DerivedType]
```

`spel create` deletes stale `*.pkl` files in `spel/scripts/` when it rebuilds a case without
`-u`; copy any pickle you want to keep somewhere safe.

---

## Database / web interface

The CSVs produced by `spel export` feed a Django app (`spel/db/`) that renders ELM's module
graph, subroutine call trees, derived-type usage, and namelist/COMPSET-conditional behavior in a
browser. Django management commands under
`spel/db/app/management/commands/` import the CSVs (`update_all_data --all` loads everything).

An instance is already hosted on AWS, so most users never need to run the web app or understand
its internals — point your browser at the hosted deployment and use `spel export` only if you
are contributing new analysis data.

---

## Testing

```bash
pytest
```

Test suites live in `spel/scripts/tests/` and `spel/scripts/fortran_parser/tests/`. Some tests
depend on a pre-existing pickle and a configured `E3SM_SRCROOT`.
