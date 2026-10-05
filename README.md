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

SPEL needs to know where your E3SM clone lives. Edit `spel/scripts/config.py`:

```python
E3SM_SRCROOT = spel_dir / "../E3SM"   # relative to the SPEL repo root
```

Everything else (`ELM_SRC`, `SHR_SRC`) is derived from it. Verify with:

```bash
spel config
```

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
| [`spel create`](#spel-create) | Analyze subroutines and generate a functional unit test |
| [`spel run`](#spel-run) | Configure, build, and run a generated unit test |
| [`spel diff`](#spel-diff) | Compare two netCDF files and report relative error |
| [`spel export`](#spel-export) | Turn a pickled unit test into database CSVs |
| [`spel config`](#spel-config) | Print the resolved E3SM/ELM source paths |
| [`spel upload`](#spel-upload) | rsync the netCDF interface files to a remote machine |
| [`spel repl`](#spel-repl) | Drop into an IPython shell wired to the Fortran parser |
| [`spel sample`](#spel-sample-and-spel-train-experimental) | *(experimental)* sample unit-test output for ML training |
| [`spel train`](#spel-sample-and-spel-train-experimental) | *(experimental)* train an emulator network |

### `spel create`

Analyzes the given subroutines, resolves every dependency, and writes a standalone unit test to
`unit-tests/<casename>`. Regardless of mode, the analysis results are **always pickled** at the
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

Without `-u`, an existing case directory is cleared, and stale `.pkl` and `script-output/*.F90`
files are removed before the run.

```bash
# Build a unit test named "canflux" around CanopyFluxes
spel create -s canopyfluxes -c canflux

# Analysis only — produce the pickle for database export, skip Fortran generation
spel create -s canopyfluxes -c canflux --db
```

### `spel run`

Runs `check_config.sh` (which configures CMake and runs `make`) and then executes
`./build/elmtest` inside the case directory.

```bash
spel run [<casename>] [-- <args passed to elmtest>]
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
`ReadWriteMod::write_elmtypes` — and reports relative errors.

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

Prints the resolved source roots so you can confirm your `config.py` edit took effect.

```bash
spel config
# E3SM SRCROOT: /path/to/E3SM
# ELM SRC     : /path/to/E3SM/components/elm/src
# SHR SRC     : /path/to/E3SM/share/util
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
