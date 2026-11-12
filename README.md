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
- **Code explorer** — the whole-driver analysis is exported as CSVs for a Django/HTMX web app:
  the time step as a data-dependency graph, a multi-variable trace timeline, a
  prognostic/diagnostic catalog, derived types, module `use` graph, subgrid filter relations, and a
  diff mode between namelist configurations or E3SM branches.

> ELM is the only project supported today.

---

## Installation

SPEL is a normal Python package; a virtual environment is recommended. Dependencies are declared
in `pyproject.toml`.

```bash
git clone https://github.com/peterdschwartz/SPEL.git
cd SPEL
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs the `spel` console script (`[project.scripts]` → `spel.cli:main`).
Django (the database / web app) and PyTorch (`spel sample`/`spel train`) are optional extras:
`pip install -e '.[db]'` (`'.[mysql]'` adds the MariaDB driver, `'.[db-test]'` pytest-django),
`pip install -e '.[ml]'`, or `'.[db,ml]'` for both.

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

## Workflows

SPEL shortens the loop of changing ELM and checking the result. Instead of rebuilding E3SM and
running a CIME case for every edit, you pull the routines you are working on out into a standalone
test that builds in seconds, feed it data captured from a real ELM run, and check every change
against that data bit for bit. When the change is right, SPEL copies it back into ELM.

### 0. Set up SPEL on a machine

Once per machine and checkout. On Perlmutter or Chrysalis everything comes from the bundled
machine file; elsewhere you write one first.

```bash
pip install -e .                                  # in a Python >= 3.11 environment
spel config --set-srcroot ~/E3SM                  # the E3SM checkout to analyze and instrument
spel machine                                      # which machine file applies, its toolchains, scratch, cprnc
spel machine check                                # each toolchain: compiler, cmake, pkg-config netcdf-fortran
spel machine setup                                # unit-tests/ and input-data on scratch (symlinks; --move keeps existing cases)
spel analyze                                      # the elm_drv analysis cache for this checkout (minutes)
```

`spel machine` names the machine it detected (Perlmutter by `$NERSC_HOST`, Chrysalis by hostname).
If it says `none`, the machine is not known yet:

```bash
spel machine init mybox                           # ~/.config/spel/machines/mybox.toml from the template
$EDITOR ~/.config/spel/machines/mybox.toml        # toolchains, [detect], scratch; every key is documented
SPEL_MACHINE=mybox spel machine check             # until it passes; then put SPEL_MACHINE=mybox in .spel.env
```

For each compiler you build with, give the file one `[toolchains.<fc>]` table that makes `<fc>`,
`cmake` and `pkg-config netcdf-fortran` work: on a machine E3SM supports, `e3sm_compiler = "gnu"`
takes the modules from E3SM's `config_machines.xml`; on a laptop, `commands` (module loads, PATH)
or `env_file` plus `pkg_config_path` to the netcdf-fortran built with that compiler (nvfortran needs
its own build: `.mod` files are compiler specific). Fill `[detect]` so the file is picked
automatically, or keep selecting it with `SPEL_MACHINE`. Capturing reference data (`--run-case`,
`spel validate`) also needs the E3SM checkout to build on this machine, i.e. a CIME machine entry
for it; `e3sm_casegen.sh` takes its machine and scratch from the machine file.

After that, the toolchain is applied by every build: `spel run canflux` uses the machine's default
compiler, `SPEL_FC=nvfortran spel run canflux` another one, and
`source <(spel machine env --fc nvfortran)` gives the same environment to an interactive shell.
Details of the file format are under [Machine configuration](#machine-configuration);
`SPEL_MACHINE=none` makes SPEL use the current shell environment, as before machine files existed.

### 1. Extract a routine into a standalone test

```bash
spel create -s canopyfluxes -c canflux
```

SPEL resolves everything the routines need (modules, derived types, globals, the call tree below
them), comments out what they don't, and writes a CMake case in `unit-tests/canflux/`. The case
builds with gfortran or nvfortran and runs only those routines. See [`spel create`](#spel-create).

### 2. Capture reference data from a real ELM run

```bash
spel create -s canopyfluxes -c canflux --run-case        # instrument, build and run ELM, collect the data
spel validate -s canopyfluxes soillittverttransp -j 4    # many cases from one ELM run
```

[Instrumentation](#spel-instrument) inserts capture calls around the routine's call site in ELM,
on the shortest call chain from `elm_drv`. It writes every input the routine reads and every output
it writes, every `--freq` steps, to netCDF. The inserted lines are tagged, so re-running replaces
them and `--undo` removes them, and several cases can be captured in one build.
[`spel validate`](#spel-validate) does all of it for a list of routines: create the cases,
instrument ELM once, run one CIME case, remove the instrumentation, run and check every test, and
write a report.

### 3. Focus on the gridcells that matter

```bash
spel create -s urbanfluxes -c urb_sub --run-case --subset urban
spel validate --list cases.txt --subset 'landunit: lun_pp%itype(l) == istcrop'
spel subset urb_full urb_three --gridcells 3,5,7          # cut existing data offline
```

A large simulation can be captured for just the gridcells you care about: urban, lake, crop,
glacier or wetland presets, or any Fortran expression over ELM fields. The selection runs inside
ELM at the first capture, and the files look like the capture of a smaller domain (whole gridcells,
renumbered from 1), so the unit test needs no special handling. A test of the urban code can run on
three urban gridcells of a global run instead of the whole domain. See
[Gridcell subsets](#gridcell-subsets).

### 4. Change the code and verify it

```bash
spel run canflux            # build, run on the captured inputs, diff against ELM's outputs
spel run canflux 8          # the same data replicated 8 times: must stay bit-identical
```

Edit the routines inside `unit-tests/canflux/` and re-run. [`spel run`](#spel-run) diffs every
output against what ELM produced and checks that the routine reads and writes what the analysis
says it does. With a gfortran build the target is bit-for-bit. `nsets` copies of the data catch
indexing and race bugs that a single copy hides. [`spel diff`](#spel-diff) compares any two netCDF
files.


### 5. Merge the change back into ELM

```bash
spel restore -c canflux --dry-run     # list the files that would be copied back
spel restore -c canflux               # copy them into E3SM_SRCROOT
```

[`spel restore`](#spel-restore) finds the source files you changed in the case since
`spel create`. It undoes SPEL's edits (dependencies commented out with a `!#py ` token are
uncommented) and copies them back to their place in the E3SM checkout. Files that SPEL wrote for
the test harness itself (`SourceFiles/`) are never copied. The ELM sources then contain exactly
your change, ready for a full E3SM build and a pull request.

### 6. Compare a development branch against a reference branch

Each E3SM checkout gets its own analysis cache, so two checkouts can be used side by side. Use the
same case name for both:

```bash
spel create -s canopyfluxes -c canflux --run-case --srcroot ~/E3SM     # reference data from master
spel create -s canopyfluxes -c canflux --srcroot ~/dev_E3SM            # the test from your branch
spel run canflux                                                       # your branch on master's data
```

The second `create` regenerates `unit-tests/canflux/` but leaves the reference data in
`unit-tests/input-data/canflux/` alone. `spel run` reports which checkout generated the test and
which one produced the data. The diffs are then the answer differences between the branches. This
works when both branches read the same derived-type components as inputs.

### 7. Understand the code

The whole-driver analysis answers questions that otherwise take reading dozens of files: which
routines a configuration actually runs, which fields a routine reads and writes (through any depth
of calls, with dummy arguments resolved), which state is carried from step to step and whether it
is in the restart file. Browse it in the [web explorer](#database--web-interface), or query it
directly with [`spel repl`](#spel-repl).

### 8. Large structural refactors

Some changes touch a large number of files and are only practical with the analysis behind them.
SPEL's formal parsing approach facillitates programmatic querying and transformation of ELM source code.
Examples include:
- [`spel compact`](#spel-compact) inventories the flux fields zeroed every step;
  `spel transform compact-rewrite` rewrites a type's fields to a compact subgrid layout.
- `spel transform split` moves type definitions and their procedures out of large modules into new
  ones and rewrites every `use` of a moved name.
- `spel nutrient` finds the fields the carbon, nitrogen and phosphorus variants of a type share,
  as candidates for a common base type; the `spel transform nutrient-*` steps extract and merge them.

All source rewrites go through [`spel transform`](#spel-transform).

### 9. Property-based tests of a process

```bash
spel create -s canopyfluxes -c canflux                           # an ordinary unit-test case
spel proptest init canflux --namelist soil_stress_method=0       # a starter canflux.proptest.toml
spel proptest probe canflux.proptest.toml                        # a few examples: what does the routine write?
spel proptest run   canflux.proptest.toml                        # the checks, on inputs Hypothesis draws
```

[`spel proptest`](#spel-proptest) runs an extracted routine from Python on synthetic inputs that
Hypothesis draws from a *scenario*, and checks invariants, metamorphic relations (also between
two namelist settings) and agreement between two builds, all written in Fortran syntax in a TOML
file. The spec is read under its namelist options: variables only used under options that are
off (the C13 fields with `use_c13` off) do not appear. A capture checks one regime bit
for bit; a scenario covers the regimes you describe, including ones no capture reached. Reference
captures only help to design the scenarios. The tutorial
[`spel/proptest/examples/canopyfluxes_tutorial.ipynb`](spel/proptest/examples/canopyfluxes_tutorial.ipynb)
goes through all of it with CanopyFluxes and ends with a checklist for another routine; the file
format is in [`spel/proptest/SPEC.md`](spel/proptest/SPEC.md).

## Directory layout

| Path | Purpose |
|------|---------|
| `spel/cli.py` | CLI entry point / subcommand definitions |
| `spel/scripts/` | Analysis engine (parser, call trees, writers, export) |
| `spel/scripts/fortran_parser/` | Fortran lexer, parser, AST, expression evaluation |
| `spel/scripts/nml/` | Namelist / COMPSET-conditional analysis |
| `spel/scripts/ml_training/` | Experimental ML emulator tooling |
| `spel/db/` | Django app for browsing exported analysis |
| `spel/proptest/` | Property-based tests of unit-test cases (`spel proptest`) |
| `SourceFiles/` | Fortran support files copied into every generated unit test |
| `unit-tests/<casename>/` | Generated unit-test cases |
| `unit-tests/input-data/` | NetCDF input/output data for unit tests |
| `unit-tests/proptest-data/` | Results of `spel proptest` that outlive a regenerated case (failing examples, rule files) |
| `script-output/` | Intermediate generated Fortran |
| `spel/scripts/fut_<casename>.pkl` | Serialized analysis results (see below) |

---
### Machine configuration

Building a unit test needs a Fortran compiler, `cmake` and a matching `netcdf-fortran` (found with
`pkg-config`). On a machine SPEL knows, `spel run`, `spel profile` and `spel validate` set that up
themselves, and `e3sm_casegen.sh --preset ...` and `profile_compare.sh` take their machine
defaults from the same place:

```bash
spel machine            # what applies here: E3SM machine, scratch, cprnc, toolchains
spel machine check      # set up every toolchain and verify compiler, cmake, netcdf-fortran
spel machine setup      # put unit-tests/ and unit-tests/input-data on scratch (symlinks)
spel run canflux        # builds with the machine's default toolchain (SPEL_FC=nvfortran for another)
```

Perlmutter (`gfortran`, `nvfortran`, `ifx`) and Chrysalis (`ifx`, `gfortran`) are bundled
(`spel/machines/*.toml`) and detected automatically. Their toolchains take the modules and
environment of the corresponding compiler from the E3SM checkout's
`cime_config/machines/config_machines.xml`, so they follow E3SM's own module versions; the
machine file only adds glue (where unit tests go, `pkg-config` paths, `ACC_DEVICE_TYPE`).

For another machine:

```bash
spel machine init mybox           # ~/.config/spel/machines/mybox.toml from the documented template
$EDITOR ~/.config/spel/machines/mybox.toml
SPEL_MACHINE=mybox spel machine check
```

A toolchain is a table `[toolchains.<fc>]` (`<fc>` = the compiler executable, what `SPEL_FC`
selects) with either `e3sm_compiler = "gnu"` (modules from E3SM's machine entry, which needs
`e3sm_machine`), `env_file = "~/env.sh"` (a file to source, e.g. a CIME case's
`.env_mach_specific.sh`) or `commands = ["module load ...", ...]`, plus `pkg_config_path`,
`ld_library_path` and `env`. `[detect]` (`hostname` regex or `env` variables) selects the file
automatically; without it, E3SM's `NODENAME_REGEX` is used, or `SPEL_MACHINE=<name>` (environment
or `.spel.env`) picks it. `SPEL_MACHINE=none` turns the machine layer off and SPEL uses the
current shell environment, as before. `spel machine env --fc nvfortran` prints the shell script a
toolchain produces (`source <(spel machine env --fc nvfortran)` gives you the same environment
interactively). User files override bundled ones of the same name, so a copy of
`perlmutter.toml` in `~/.config/spel/machines/` is the way to change it.

## CLI reference

All commands are subcommands of `spel`. Run `spel <command> --help` for the authoritative list.

| Command | Purpose |
|---------|---------|
| [`spel analyze`](#spel-analyze) | (Re)build the cached whole-driver (`elm_drv`) analysis |
| [`spel create`](#spel-create) | Extract subroutines from the cached analysis and generate a functional unit test |
| [`spel instrument`](#spel-instrument) | Instrument ELM to capture reference data; optionally build/run a CIME case |
| [`spel validate`](#spel-validate) | Create, capture (one ELM run) and validate many unit tests in parallel |
| [`spel subset`](#gridcell-subsets) | Write some gridcells of a case's reference data as a new case's |
| [`spel merge-subset`](#gridcell-subsets) | Merge the per-rank files of a subset capture into a case's reference data |
| [`spel run`](#spel-run) | Configure, build, and run a generated unit test, then validate it against ELM's data |
| [`spel profile`](#spel-profile) | Build and run a unit test without outputs under Nsight Systems/Compute |
| [`spel proptest`](#spel-proptest) | Property-based, metamorphic and differential tests of a unit-test case (Hypothesis scenarios, Fortran-syntax checks in TOML) |
| [`spel transform`](#spel-transform) | Source-to-source transforms: OpenACC porting, `acc2omp`, compact layout, nutrient refactor, `split`, `prune`, ... (`spel transform list`) |
| [`spel compact`](#spel-compact) | Inventory flux fields for a compact subgrid layout (the rewrite is `spel transform compact-rewrite`) |
| [`spel accplan`](#spel-accplan) | Propose an OpenACC loop structure (collapse, column-parallel, interchange, scatter) per loop |
| `spel nutrient` | Find the fields C/N/P type variants share (candidates for a common base type) |
| [`spel restore`](#spel-restore) | Copy the files changed in a unit-test case back into the E3SM checkout |
| [`spel diff`](#spel-diff) | Compare two netCDF files and report relative error |
| [`spel export`](#spel-export) | Write the `elm_drv` analysis as database CSVs (a snapshot) |
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
| `--subset`, `--subset-max` | no | all gridcells | Capture a [gridcell subset](#gridcell-subsets) |

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
   The original lines are recorded in `components/elm/src/main/.spel-access-edits.json` and
   `--undo` (also run by `spel validate`) puts them back. A line you've changed since
   `instrument` is left alone and reported. Edits made before this ledger existed must be
   reverted with git.
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
6. Runs `spel run <case>` for every case in parallel (logs in `unit-tests/<case>/validate.log`),
   each in the machine's toolchain (`--test-env FILE` sources a file of your own instead; the ELM
   build keeps the clean environment either way).
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
instrumented without touching ELM. `--subset MASK` captures a [gridcell subset](#gridcell-subsets)
for every case; in a list file, `case[MASK]: subs` (optionally `[MASK; max=N]`) sets it per case.

### Gridcell subsets

`spel instrument`, `spel create --instrument/--run-case` and `spel validate` take
`--subset MASK [--subset-max N]`: the capture then writes only up to `N` (default 100) gridcells
matching `MASK`, with all their topounits, landunits, columns and patches. The subgrid index
fields (`lun_pp%gridcell`, `col_pp%landunit`, `grc_pp%coli`, ...) and the bounds are renumbered
from 1, so the files look like the capture of a smaller domain and the unit test runs on them
unchanged. Use it to pull a small, targeted test out of a large simulation.

| `MASK` | A gridcell matches if it has ... |
|--------|-----------------------------------|
| `urban`, `lake`, `crop`, `glacier`, `wetland` | a landunit of that type with weight > 0 (`crop` is `istcrop`) |
| `all` | (every gridcell) |
| `urban,lake` | any of the listed presets |
| `<level>: <expr>` | a `<level>` entry (gridcell, topounit, landunit, column, patch) for which the Fortran expression is true |

Custom expressions use the index `g`, `t`, `l`, `c` or `p` of their level, the instances
`grc_pp`, `top_pp`, `lun_pp`, `col_pp`, `veg_pp`, and anything public in `landunit_varcon`,
`column_varcon` and `pftvarcon`; the ELM build is what checks them. The presets require a weight
because zero-weight landunits exist (every gridcell has urban landunits when
`run_zero_weight_urban` is on).

```bash
spel instrument urbanfluxes --subset urban --run-case
spel validate --list cases.txt          # e.g. a line `soil_tall[patch: veg_pp%itype(p) == ndllf_evr_tmp_tree; max=20]: canopyfluxes`
spel validate -s canopyfluxes --subset "gridcell: grc_pp%latdeg(g) > 60._r8" --subset-max 10
```

The selection runs in ELM at the first capture (`spel_select_subset` in `SpelCapture_<tag>`), so
masks can use fields the unit test never reads. Matches are taken evenly spaced when there are more
than `N`. With several MPI ranks, every rank with matches writes its share of `N` (in proportion to
its matches) to its own files, `<tag>.rankNNNNN.spel-*.nc` in the run directory, and the master
rank lists those ranks in `<tag>.spel-subset-ranks.txt`. `spel validate` and `spel instrument
--run-case` merge them into one capture: gridcells ordered by global index, cut to `N`, with all
subgrid index fields renumbered. To merge by hand (e.g. after a run you submitted yourself):

```bash
spel merge-subset /path/to/run hs_drain     # -> unit-tests/input-data/hs_drain/spel-*.nc
```

Variables without a subgrid dimension must be identical on every rank; the merge stops and names
any that are not (`--allow-differ NAME` keeps the first rank's values). The run's `e3sm.log` has
`SPEL subset` lines with the counts, and each file records the mask and the kept gridcells' global
index, latitude, longitude and rank as global attributes (`spel_subset_*`).
`spel_reference.json` records the mask and `spel run` prints it.

`spel subset SRC DST --gridcells 3,5,7` does the same offline: it writes those gridcells (1-based
positions in SRC's files) of `unit-tests/input-data/SRC` to `unit-tests/input-data/DST`, using
the same rules (`spel/scripts/subset.py`). Masks need ELM fields that the capture usually lacks, so
offline subsets take explicit gridcells.

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

The test executable accepts one optional positional argument:

| Arg | Meaning | Default |
|-----|---------|---------|
| 1 | copies of the input sites (`nsets`) | 1 |

By default the unit test runs a single clump spanning the whole process, so filters and loops
cover every site at once. With `nsets > 1`, the netCDF reads replicate each subgrid-dimensioned input
`nsets` times and the generated `duplicateMod` offsets the subgrid index fields
(`veg_pp%column`, `col_pp%landunit`, ...) of each copy. Outputs are written for the first copy
only, so `fut-outputs*.nc` stay comparable with `spel-outputs*.nc`. A
`WARNING (spel_nsets)` line means a copy produced results that differ from the first copy.

`SPEL_CLUMPS=N` (run time, `1 <= N <= nsets`) splits the sets into `N` clumps of whole sets
(as evenly as possible), builds the filters of each clump, and calls the routines in the same
`!$OMP PARALLEL DO` over clumps that `elm_drv` uses, so a build with `ACC=OPENMP` threads the
unit test the way ELM threads it on CPUs (one thread per clump, `OMP_NUM_THREADS=N`). Without
OpenMP the clumps run one after the other. OpenACC builds need `SPEL_CLUMPS=1`. The NVTX range
`compute` times the whole clump loop.

```bash
SPEL_ACC=OPENMP SPEL_CLUMPS=8 OMP_NUM_THREADS=8 spel run soiltemperature 16
```

Time-step inputs are read in chunks: `read_elmtypes` loads a whole input file (720 timesteps by
default) per variable into set-1-shaped buffers, then each step copies its slice into the
variables, replicating it into the `nsets` copies. In OpenACC builds the buffers live on the
device, so the copy (and the `duplicateMod` index offsets) run there and the inputs never
pass through the host. Set `SPEL_IO_CHUNK=<n>` at run time to read at most `n` timesteps at
once when host or device memory is tight.

```bash
spel run canflux            # one copy of the sites
spel run canflux 8          # 8 copies of the sites, all in one clump
cd unit-tests/canflux && spel run
```

Compiler, debug, and OpenACC settings live in the case's `check_config.sh` (`compiler`, `debug`,
`acc`); it detects a mismatch with the existing `CMakeCache.txt` and offers to reconfigure. The
environment variables `SPEL_FC`, `SPEL_DBG` and `SPEL_ACC` override them without editing the script.
On a known machine (see [Machine configuration](#machine-configuration)) `spel run` builds and runs
inside the toolchain of `SPEL_FC` (default: the machine's `default_fc`): modules, `PKG_CONFIG_PATH`
and the like come from the machine file, and `make` gets `-j<make_jobs>`. Running
`./check_config.sh` by hand needs the same environment: `source <(spel machine env --fc nvfortran)`.

`acc` sets the generated CMake's `ACC` cache option:

| `ACC` | Flags | Notes |
|-------|-------|-------|
| `OFF` (default) | none | `!$acc` directives are comments |
| `MULTICORE` | nvfortran: `-acc=multicore -Minfo=accel` | compute regions run on host threads; set the count with `ACC_NUM_CORES` (default: all cores). gfortran only gets `-fopenacc -foffload=disable` and a warning: GCC has no multicore target, so regions run on one thread |
| `GPU` | nvfortran: `-acc=gpu -gpu=deepcopy -Minfo=accel -cuda` | needs nvfortran (`-DGPU=ON` still works as an alias) |
| `OPENMP` | gfortran: `-fopenmp -U_OPENMP`; nvfortran: `-mp -Minfo=mp -U_OPENMP` | threads the clump loop (`SPEL_CLUMPS`) and sources translated by [`spel transform acc2omp`](#acc2omp); set the thread count with `OMP_NUM_THREADS`. `-U_OPENMP` keeps ELM's own `#ifdef _OPENMP` code off, as in the analysis |

```bash
# nvfortran needs its own netcdf-fortran build (.mod files are compiler specific) on
# PKG_CONFIG_PATH and ahead of any gfortran build on LD_LIBRARY_PATH: the machine file's
# nvfortran toolchain provides that on a known machine
SPEL_FC=nvfortran SPEL_ACC=MULTICORE SPEL_DBG=OFF spel run canflux
ACC_NUM_CORES=4 ./build/elmtest   # rerun with 4 threads
```

Comparing `ACC=OFF` and `ACC=MULTICORE` outputs (including `ACC_NUM_CORES=1` vs all cores) catches
missing `private`/`reduction` clauses and races without a GPU.

### `spel profile`

Builds the case like `spel run`, then runs `build/elmtest` with `SPEL_PROFILE=1` under Nsight
Systems (default) or Nsight Compute. No validation is done. In profile mode `main.F90` skips the
per-step device-to-host copy (`spel_acc_update_host`) and `write_elmtypes`, because outputs only
matter for correctness checks.

```bash
SPEL_FC=nvfortran SPEL_ACC=GPU SPEL_DBG=OFF spel profile canflux 8 --steps 100 --stats
spel profile canflux 8 --tool ncu --steps 2 --tool-args "--set full --launch-count 20"
spel profile canflux --tool none --steps 10          # profile mode, no profiler
```

| Option | Meaning |
|--------|---------|
| `--tool nsys` | `nsys profile --trace=cuda,nvtx,openacc,osrt`; `--stats` prints the summary tables |
| `--tool ncu` | `ncu --nvtx --nvtx-include compute/`: only kernels from the routines under test |
| `--steps N` | run at most `N` time steps (`SPEL_MAX_STEPS`) |
| `-o PATH` | report path; default `unit-tests/profiles/<case>/<case>-nsets<N>-<tool>`, outside the case dir so `spel create` doesn't delete it |
| `--tool-args "..."` | extra profiler options |

nvfortran builds mark NVTX ranges (`spel_profMod`, compiled with `-DSPEL_NVTX`, linked with
`-cudalib=nvtx`; configure with `-DNVTX=OFF` to drop them). The ranges are `spel_init` and
`spel_timesteps`; inside the latter, each `step` contains `read_inputs`, `duplicate_sets`,
`compute` (one range per generated call, e.g. `canopyfluxes`), and, outside profile mode,
`update_host` and `write_outputs`. For gfortran builds the ranges are no-ops.
You can also set `SPEL_PROFILE` and `SPEL_MAX_STEPS` yourself when running `build/elmtest`
directly (e.g. under `nsys` or `ncu` with your own options).
`ncu` needs access to the GPU performance counters (`ERR_NVGPUCTRPERM` otherwise; see NVIDIA's
instructions for that error).

To compare a refactored routine against its reference across thread counts, run
`profile_compare.sh` (repo root) on the machine you want numbers for. It builds both cases with
nvfortran (`-fast`) and times the NVTX range `compute` (wall time per time step) for:

| Config | Build | Threading |
|--------|-------|-----------|
| `master_off`, `acc_off` | `ACC=OFF` | serial |
| `master_cN`, `acc_cN` | `ACC=OPENMP`, `SPEL_CLUMPS=N` | `N` clumps x 1 thread, as `elm_drv` threads ELM |
| `acc_mcN` | `ACC=MULTICORE` | 1 clump x `N` OpenACC threads |
| `ompN` (`--omp-case`) | `ACC=OPENMP` on an `acc2omp` case | 1 clump x `N` OpenMP threads |

The summary gives each configuration's speed-up over master serial, over master with clump
threading on the same `N` threads (the fair comparison with ELM's CPU threading), and over the
same code on one thread. `acc_cN` vs `master_cN` isolates the code changes; `acc_mcN` vs
`master_cN` compares the threading strategies. Clump runs need `N <= nsets`.

```bash
./profile_compare.sh --env nvhpc_env.sh -n 128 soiltemperature soiltemp_acc
./profile_compare.sh -c "1 8 32 64 128" --omp-case soiltemp_omp soiltemperature soiltemp_acc
OMP_PROC_BIND=spread OMP_PLACES=cores LAUNCH="srun -n 1 -c 128 --cpu-bind=cores" \
    ./profile_compare.sh -n 128 soiltemperature soiltemp_acc
```

`--env` sources a module/env file first, `--skip` leaves out configurations, `--omp-case` adds
an `acc2omp` case, `--link-inputs` points a new case at the reference case's input data, and
`LAUNCH` prefixes every run. Results go to `unit-tests/profiles/compare-<case>-nsets<N>-<date>/`
(`results.csv`, `summary.txt`, the `.nsys-rep` reports and build/run logs). For weak scaling,
run it again with `nsets` proportional to the core count.

### `spel transform`

One front end for SPEL's source-to-source transforms (`spel/transforms/`). Each transform is a
registered class that states what it matches, what it changes and what it refuses, and builds a
plan (edits, refusals, notes); the shared driver reports, diffs, writes and verifies it.

```bash
spel transform list                                   # the transforms, one line each
spel transform explain collapse-bounds                # what it does, and its options
spel transform section-loops --dry-run --report       # plan only: what would change where
spel transform acc-apply --where SoilTemperature      # before/after at matching locations
spel transform acc-sync --diff -o /tmp/out --verify   # write flat into DIR, re-parse the result
spel transform --spec steps.toml -o /tmp/out          # several steps, each on the previous one's output
```

Options of each transform are flags (`spel transform NAME --help`). A spec file names one
`transform` with an `[options]` table, or a list of `[[step]]`s:

```toml
[[step]]
transform = "section-loops"
[[step]]
transform = "acc-apply"
options = { roots = ["elm_driver::elm_drv"] }
```

The steps of a spec run as a pipeline: each plans on the text the earlier ones produced (the
analysis of the edited files is refreshed in memory, routine by routine), and the net result is
written once; `--dry-run` shows each step's plan on its predecessors' output. A step whose output
the in-memory analysis can't follow (new declarations or `use`s in a module head, a new Fortran
file, a moved routine header) stops the pipeline before anything is written; `--reanalyze-steps`
then runs the steps one at a time in place, with a full reanalysis in between.

Each write appends to `.spel-transforms.json` in the output directory (or the srcroot). A transform
that refuses the whole input reports why and writes nothing. `-o DIR` writes by basename and doesn't
clear `DIR`: use a fresh directory (e.g. a new case's `SourceMods/src.elm`). The API and its
status are in `spel/transforms/DESIGN.md`.

The transforms replace the former commands `spel acc2omp`, `spel compact rewrite`,
`spel device-data`, `spel split` and `spel nutrient extract|species|merge|groups|history|restart`,
which now only print their replacement:

| Former command | Now |
|----------------|-----|
| `spel acc2omp MOD ... [--case CASE]` | `spel transform acc2omp [--modules MOD ...] [--case CASE]` |
| `spel compact rewrite [opts]` | `spel transform compact-rewrite [opts]` (same option names; `--replace` is gone: use an empty `-o` directory) |
| `spel device-data [--print]` | `spel transform device-data [--dry-run --diff]` |
| `spel nutrient STEP [opts]` | `spel transform nutrient-STEP [opts]` |
| `spel split` | `spel transform split` |

The reports stay commands: `spel compact inventory`, `spel nutrient common|procedures` and
`spel accplan`. The nutrient refactor of a whole checkout, one commit per step, is
`nutrient_refactor.sh`.

#### `prune`

`spel transform prune` deletes the derived-type members (of every analyzed type) that nothing
uses: every `%name` reference under `components/elm` only declares, allocates, deallocates,
nullifies or resets the member to a constant. Those statements go with it. References are
matched by component name across types, so a name used anywhere keeps every member of that name.
The EMI stub/demo and git submodules (FATES) are searched but never edited. `--report --verbose`
lists the kept members with the reason. It edits type declarations, so in a spec it needs
`--reanalyze-steps`.

#### `acc2omp`

Translates the OpenACC directives of analyzed modules to OpenMP, so a unit test can run
multithreaded with gfortran (GCC has no OpenACC multicore target). The data-sharing clauses
come from the analysis, not from the OpenACC clauses:

- the loop each `!$acc parallel loop` applies to, and its `collapse` nest, come from the
  routine's loop nest; collapsed loops must be perfectly nested and rectangular
- local scalars written in the loop (including through intent(out)/(inout) call arguments)
  and inner loop indices become `private`; `firstprivate` if the loop reads them before
  writing them. OpenACC reductions are kept; arrays stay shared unless OpenACC privatizes them
- written dummy or global scalars, privatized values read after the loop, and callees that
  write SAVEd locals or global scalars are reported

The report lists each loop's clauses and how they differ from the OpenACC `private` list
(`added`: privatized by the analysis but not by OpenACC, a likely race in the OpenACC
version too). Data directives (`data`, `enter/exit data`, `update`, `declare`, ...),
`routine` and inner `loop` directives are removed; `atomic` becomes `!$omp atomic`.
`!$acc parallel`/`kernels` regions without `loop` are left as OpenACC (serial) and reported.

```bash
spel transform acc2omp --case soiltemp_acc --diff                     # rewrite the case's copies in place
SPEL_ACC=OPENMP OMP_NUM_THREADS=6 spel run soiltemp_acc
spel transform acc2omp --modules soiltemperaturemod -o /tmp/omp       # translated copies of the E3SM sources
spel transform acc2omp --modules soiltemperaturemod --dry-run --report # report only
```

With `--case` and no `--modules`, every analyzed module of the case with OpenACC compute
loops is translated. Edits are located by content, so they apply to the case copies even
though `spel create` shifted their lines. `spel create` regenerates the case without them.
The analysis must include loop-nest information (rebuild an older cache with `spel analyze`).

### `spel compact`

Stores subgrid fields on the elements of their filter (contiguous per clump) instead of the
whole `begp:endp` range. `inventory` traces one time step from `elm_drv` and reports, for every
field the per-step `SetValues` zeroes, how each access indexes it, whether its value lives across
steps (history, restart), and whether it can use the compact index directly, needs maps, or is
blocked. The transform `compact-rewrite` gives the fields of one type that qualify compact
storage in the E3SM sources (or writes the rewritten files to `-o DIR`, e.g. a case's
`SourceMods/src.elm`; `e3sm_casegen.sh --compact` does this). The design, rules and open items
are in `spel/transforms/compact/DESIGN.md`.

```bash
spel compact inventory --verdict blocked -o inventory.json
spel transform compact-rewrite --type vegetation_carbon_flux --dry-run --diff --excluded
spel transform compact-rewrite --type vegetation_carbon_flux -o $CASE/SourceMods/src.elm
```

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

Writes the cached `elm_drv` analysis (see [`spel analyze`](#spel-analyze)) as one CSV per table of
the database, plus `snapshot.json` (E3SM/SPEL commits, analysis date, row counts), into a directory:

```bash
spel export                                   # -> script-output/db-export/<branch>@<commit>/
spel export --srcroot ~/E3SM -o /tmp/x --label master-2026-10
spel export --case canflux                    # a case's pickle (unit-tests/canflux/fut.pkl) instead
```

The label names the *snapshot* in the database; several snapshots (branches, commits) can be
loaded side by side. The tables are the modules, derived types and their components, module
variables (with namelist membership), routines (with their source text), arguments, call sites and
argument bindings, per-line accesses (direct and through callees), if blocks (namelist-gated ones
flagged), `hist_addfld*`/`restartvar` registrations, the static step trace (every routine call of
one `elm_drv` time step in execution order, and every read/write of a module variable along it,
with dummies resolved to the actuals bound at each call, and the subgrid filter each access runs
under), the filters the step builds and the disjoint/subset relations derived from them, and the
derived per-variable *field state* (see [Database / web interface](#database--web-interface)). The
exporter is `spel/scripts/db_export/`; `schema.py` lists the columns. Export takes a few seconds; a
fresh `spel analyze` of an E3SM checkout takes about a minute.

### `spel restore`

Copies the ELM source files you changed inside a unit-test case back into `E3SM_SRCROOT`.

```bash
spel restore -c canflux --dry-run   # list them
spel restore -c canflux -y          # copy without prompting
```

A file counts as changed when its modification time differs from the one most files in the case
share (the time `spel create` wrote them). The `!#py ` token SPEL used to comment out unneeded code
is removed again. Files from `SourceFiles/` (the netCDF interface, decomposition and driver stubs of
the test harness) are never copied back. Review the result with `git diff` in the E3SM checkout.

### `spel accplan`

Proposes an OpenACC loop structure for each loop of the analyzed routines: collapse a perfect nest,
parallelize over columns with an inner sequential loop, interchange, or scatter. It reads the loop
nests and per-loop accesses from the analysis and reports only.

```bash
spel accplan --routine soiltemperature --min-lines 5
```

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

### `spel machine`

The machine configuration (see [Machine configuration](#machine-configuration)): machine files
`spel/machines/*.toml` (bundled: `perlmutter`, `chrysalis`; `template.toml` documents every key)
and `~/.config/spel/machines/*.toml` (`$SPEL_MACHINES_DIR`), the current one from `$SPEL_MACHINE`
or host detection.

```bash
spel machine [show]                 # machine, E3SM machine, scratch, cprnc, toolchains, casegen defaults
spel machine list                   # every machine file (* = this machine)
spel machine check [--fc FC]        # set up the toolchain(s), verify compiler, cmake, pkg-config netcdf-fortran
spel machine env [--fc FC]          # the toolchain's shell script: source <(spel machine env --fc nvfortran)
spel machine vars                   # SPEL_E3SM_MACHINE=..., SPEL_SCRATCH=..., SPEL_CPRNC=... for shell scripts
spel machine init NAME [--from M]   # new user machine file from the template (or a copy of machine M)
spel machine setup [--move]         # symlink unit-tests/ and unit-tests/input-data to $scratch/spel-*
```

### `spel upload` (deprecated)

rsyncs the netCDF interface Fortran files (`FUTConstantsMod.F90`, `nc_allocMod.F90`, `nc_io.F90`,
`ReadWriteMod.F90`) from the current directory to a remote machine.

```bash
spel upload <remote_host> <destination_path>
```

### `spel proptest`

Property-based, metamorphic and differential testing of an extracted routine. It needs the
`proptest` extra (`pip install -e '.[proptest]'`: Hypothesis). The routine runs in a worker
subprocess, so a crash, `endrun` or `stop` counts as a failing example, which Hypothesis then
shrinks. The spec format, every rule and function, the files written and the limits are in
[`spel/proptest/SPEC.md`](spel/proptest/SPEC.md). For a guided start, see the CanopyFluxes
tutorial notebook `spel/proptest/examples/canopyfluxes_tutorial.ipynb` (it needs
`pip install notebook pandas` and the `canflux` case).

```bash
spel create -s canopyfluxes -c canflux                      # an ordinary unit-test case
spel proptest init  canflux --where "frac_veg_nosno(p) > 0" \
    --namelist soil_stress_method=0                        # starter spec + rule file from the case's capture
spel proptest show  canflux.proptest.toml [--kind input]    # inventory, scenarios, inputs without a rule
spel proptest probe canflux.proptest.toml                   # a few examples: outputs never written, aborts
spel proptest run   canflux.proptest.toml [-s SCENARIO] [-k CHECK] [-n EXAMPLES] [--seed N] [--json F] [--junit F]
spel proptest replay unit-tests/proptest-data/canflux/failures/<scenario>__<check>   # a saved failure again
spel proptest coverage canflux.proptest.toml                # if-branches the scenario never enters (gfortran + gcov)
spel proptest subspace canflux.proptest.toml --scenario S --output "t_veg(p)" [--input NAME ...] [--parameters]
spel proptest stats canflux --reference canflux [--landunits soil crop]   # only the rule file and report
spel proptest build canflux [--fc nvfortran] [--acc MULTICORE] [--dbg]   # a build; `run` rebuilds when sources changed
```

- **Inventory.** Every variable comes from SPEL's analysis of the case (`fut.pkl`), never from
  the source text. Each is classified as topology (subgrid index fields, types), parameter
  (constants), input, inout or output. The read/write category comes from the static trace the
  database export uses: a variable whose first write is conditional and that is read later is an
  inout. Outputs are reset to NaN before every call, so elements the routine skips stay visible.
- **Domain.** `[domain]` declares ELM's subgrid hierarchy gridcell by gridcell
  (`[[domain.gridcell]]`): topounits (several per gridcell, with weights and elevations),
  landunits (soil with one column of weighted pfts, crop with a column per CFT, urban with its
  five columns, lake), each element weighted in its parent; `copies` per gridcell and `nsets`
  sets. A reference capture supplies the constants, parameters and level extents. `init` picks
  the landunit kinds from the filters the routine takes (UrbanFluxes: a town over `p in urbanp`,
  statistics of urban patches only); `--landunits` overrides it.
- **Python API.** `spel.proptest.api` builds the same spec in Python (`pt.Case`, `pt.Domain`,
  `pt.Spec`, rules as `pt.Range(...)`), writes it as TOML, and inspects it: the declared domain
  (`spec.view()`: per-element tables, filters to expect), what the case built
  (`session.built().check()`), and drawn examples and their outputs as pandas tables
  (`session.sample`, `session.execute(...).frame(level)`, `.verdict(check)`).
- **Scenarios.** `[scenario.<name>]` gives a rule for every input the routine reads; a run refuses
  when one is missing, and when a parameter the routine reads has no value. The rules are
  `range`, `log_range`, `values`, `dist`, `scale`, a constant, with per-pft and per-level
  overrides. `extends` reads a rule file such as the one `stats` writes. `derive` sets dependent
  inputs from Fortran expressions, `assume` filters elements and `over` picks them
  (`p in nolakeurbanp`).
- **Namelist options.** `[namelist]` sets the options of the runs over the reference's, and a
  scenario or a relation can change them (`namelist = { use_c13 = true }`). The inventory is
  classified under them from the analysis' namelist ifs: a variable only used in branches that are
  off needs no rule, and naming it in a check is an error that names the option. Options are
  run-time constants of the case, so no rebuild is needed.
- **Checks.** `[[property]]` is a logical Fortran expression per element. `[[relation]]` runs a
  base and a follow-up on the two halves of the sets in one call and compares them (`base%x`,
  `follow%x`); without `follow` the halves are copies, and `permute` reorders them. With
  `namelist` the follow-up is the same example under other options (a second process).
  `[[differential]]` runs each example on two builds or two cases (`[compare]`), e.g. a refactored
  routine against the original, or `ACC=MULTICORE` against `OFF`. Unknown names stop loading.
- **Failures.** A failing check prints the element, the values its expression used and the drawn
  inputs there. The shrunk example is saved in capture format and `replay` runs it again.
- **Design aids.** `probe` and `coverage` show what a scenario reaches; `stats` proposes rules
  from a capture; `subspace` ranks the inputs by their effect on an output (active subspace).
- **Edits.** After editing the case's Fortran, the next command rebuilds the library (an
  incremental make). Results that must survive `spel create` are in
  `unit-tests/proptest-data/<case>/`.

The worked example is `spel/proptest/examples/canopyfluxes.proptest.toml`. It has four
properties and six relations, three of them between namelist settings, and the relation "more
absorbed shortwave does not cool the leaf" fails in weak wind.

### `spel repl`

Starts an embedded IPython shell with autoreload enabled and `parse_line` from
`spel.scripts.fortran_parser.spel_repl` already in the namespace — handy for interactively
poking at the Fortran lexer/parser.

```bash
spel repl
```

### `spel sample` and `spel train` (experimental)

Early-stage tooling for training neural-network emulators of ELM subroutines from unit-test
output. These paths need the `ml` extra (`pip install -e '.[ml]'`: PyTorch, scikit-learn).
Interfaces are unstable.

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
(`spel export --case`, the ML pipeline, ad-hoc analysis scripts) re-loads the pickle instead of
re-analyzing the source. `spel export` without `--case` reads the `elm_drv` cache instead.

### Portability across machines and checkouts

Pickles are designed to be shared. On export, every `filepath` on a module, subroutine, or
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
spel export --case 4637ab7  # -> script-output/db-export/<label>+4637ab7/
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

A Django app (`spel/db/`, HTMX plus a little d3) for scientists, developers and new team members to
browse one ELM time step. Every page follows a *snapshot* (an analyzed E3SM checkout) and a
*namelist configuration*, both picked in the toolbar, so what is shown is what that configuration
actually runs.

| Page | What it shows |
|---|---|
| Tree graph | A routine's callees as boxes, one row below the last routine each depends on; a line runs from a routine to a later one that reads what it wrote. Click into any routine. Book-keeping routines (summaries, zeroing, balance checks, budgets) are hidden by default, and variables written and read under disjoint filters (lake vs. non-lake columns) don't count as a dependency. |
| Trace | Several variables on one timeline: where in the step each is read and written, arcs from a write to its next read, a dashed arc for state carried into the next step. |
| Call tree | The step as an expandable tree pruned by the configuration, with each routine's arguments, accesses, namelist gates and source. |
| Variables | Every field classified as **prognostic** (read before the step's first unconditional write: its value carries over), **diagnostic** (written first, registered for history), **intermediate** (written first, not output), **input** (only read) or **unused**, with restart/history registrations and their mismatches (restart-registered but recomputed every step; carried but not in the restart file). |
| Types | Derived types and their members by subgrid level, instances, type-bound procedures; members can be added to the trace from here. |
| Modules | What each module `use`s and what uses it, as a layered graph, plus per-module contents. |
| Filters | The subgrid filters the step builds, where and under which conditions, and which pairs are disjoint or nested. |
| Diff (experimental) | Two configurations or two snapshots compared: reachable routines, flipped if blocks, access sets, categories. |

Search boxes are fuzzy throughout (`canpyflx` finds `canopyfluxes`).

### Running it locally

```bash
pip install -e '.[db]'                       # Django ('.[mysql]' adds the MariaDB driver)
cp spel/db/.env.example spel/db/.env          # optional; SQLite needs no settings
spel analyze && spel export                   # the analysis as CSVs
python spel/db/manage.py migrate
python spel/db/manage.py update_all_data script-output/db-export/<label>   # --replace to reload
python spel/db/manage.py import_presets       # namelist presets (COMPSET lnd_in values)
python spel/db/manage.py runserver
```

Several snapshots can be loaded side by side, e.g. E3SM master and a development branch:

```bash
spel analyze --srcroot ~/E3SM-master && spel export --srcroot ~/E3SM-master
python spel/db/manage.py update_all_data script-output/db-export/master@<commit>
```

`update_all_data --list` shows the loaded snapshots, and `update_all_data --set-default LABEL`
chooses the one visitors see first (otherwise the most recently loaded). Namelist configurations are evaluated on
first use with SPEL's if evaluator; flags a configuration does not set take the default declared in
the Fortran source. Every page takes `?snap=<label>&cfg=<preset slug>`, so views are shareable.
Signed-in users can create their own configurations in the namelist editor and override a
variable's category by hand (overrides survive a `--replace`). Results are cached for ten minutes:
restart `runserver` after reloading a snapshot.

### Deploying

The app needs a server (it queries the database and evaluates configurations per request); static
hosting such as GitHub Pages is not enough. Step-by-step notes for nginx, gunicorn and SQLite on
an Ubuntu VM (AWS Lightsail) are in [`spel/db/DEPLOY.md`](spel/db/DEPLOY.md). In short:

- Set `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=0`, `DJANGO_ALLOWED_HOSTS=<host>` and
  `DJANGO_CSRF_TRUSTED_ORIGINS=https://<host>` in `spel/db/.env` (template: `.env.example`, never
  commit the real file), plus the production TLS settings listed there.
- `python spel/db/manage.py collectstatic`, then serve `STATIC_ROOT` from the web server and run
  Django under gunicorn or similar.
- Give the VM a fixed (Elastic) IP, point a DNS `A` record at it, and get a certificate with
  Let's Encrypt (`certbot --nginx -d <host>`, or Caddy does it itself).
- The database is a single SQLite file (`SPEL_DB_PATH`), which suits a read-mostly site on one
  instance. MariaDB (`DB_ENGINE=mysql`, `reset_db.sh`) is supported for many concurrent writers;
  switching needs no code changes.

### Reporting problems

ELM has many configurations, and the analysis is conservative (an unknown condition counts as "may
happen"). A routine that looks wrongly pruned, a dependency that cannot be real or a category that
disagrees with the model are worth an [issue](https://github.com/peterdschwartz/SPEL/issues); please
include the snapshot, the configuration and the page URL.

## Testing

```bash
pytest
```

Test suites live in `spel/scripts/tests/`, `spel/scripts/fortran_parser/tests/`,
`spel/transforms/` and `spel/db/app/tests/` (pytest-django, skipped without the `db-test` extra). Tests that need
a whole-ELM analysis use the `elm_drv_analysis` fixture (`conftest.py`). It loads a frozen copy at
`.pytest-fixtures/elm_drv_analysis.pkl`, which is seeded once from the `E3SM_SRCROOT` analysis
cache. Delete that file to refresh it, or point `SPEL_TEST_ANALYSIS` at another `analysis.pkl`. If
neither the copy nor the cache exists, these tests are skipped.

## Contributing

`AGENTS.md` collects the working notes for developers and coding agents: machine setup, generated
vs. hand-edited files, how to verify a refactor bit-for-bit, and the rule that new analyses are
built on SPEL's Fortran parser and scope walker rather than on regexes over source text (extend the
parser when it lacks something).
