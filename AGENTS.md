# AGENTS.md

Notes for AI agents working on SPEL. Read `README.md` for the CLI and the
reference-vs-development workflow; this file only covers what the README doesn't.

## Environment (local workstation)

- `spel` is on `PATH` from `~/python-envs/py3.12` (there is no `~/climate-projects/py12env`).
- nvfortran lives in `/opt/nvidia/hpc_sdk/Linux_x86_64/25.9`. Its netcdf-fortran build is
  `~/.local/netcdf-fortran` (+ `~/.local/netcdf-c`, `~/.local/hdf5`); the default `PKG_CONFIG_PATH`
  points at the gfortran/mpich builds in `~/.local/gcc/`. There is no `~/.local/nvhpc-libs/env.sh`;
  before any nvfortran build set:

  ```bash
  export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin:$PATH
  export PKG_CONFIG_PATH=$HOME/.local/netcdf-fortran/lib/pkgconfig:$HOME/.local/netcdf-c/lib/pkgconfig:$HOME/.local/hdf5/lib/pkgconfig
  export LD_LIBRARY_PATH=$HOME/.local/netcdf-fortran/lib:$HOME/.local/netcdf-c/lib:$HOME/.local/hdf5/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/lib:$LD_LIBRARY_PATH
  ```

  Put that in a file and pass it to `profile_compare.sh --env`.
- Hardware: Ryzen 5 3600 (6 cores / 12 threads), RTX 4070. Production profiling runs on
  Perlmutter; only test locally unless the user asks otherwise.

## Generated vs. hand-edited files

- `SourceFiles/nc_io.F90` is **generated**. Edit `spel/scripts/io/netcdf_io.py`, then
  regenerate with `python -m spel.scripts.io.netcdf_io`.
- Other `SourceFiles/*.F90` (`main.F90`, `elm_initializeMod.F90`, `nc_allocMod.F90`, ...) are
  templates/static files copied into each case; `!#...` markers in them are filled in by the generator.
- `duplicateMod.F90`, `ReadWriteMod.F90` and the case's ELM sources are generated per case.
  Everything under `unit-tests/<case>/` is overwritten by `spel create`; fix the generator, not the case.
  The case `CMakeLists.txt` comes from `generate_cmake` in `spel/scripts/functional_unit_test.py`.

## Testing

```bash
python -m pytest -q
```

Known failures that predate current work (don't chase them unless asked):
`test_sample_and_compress`, `test_sub_parse` (missing fixture file),
`test_use_statements_drop_unavailable_names`, `test_resolver_use_stmts_follow_pruned_imports`,
`test_pickle_roundtrip_matches_fut` (error).

Tests that need a whole-ELM analysis use the session fixture `elm_drv_analysis` (root
`conftest.py`). It loads the frozen copy `.pytest-fixtures/elm_drv_analysis.pkl`, which is gitignored
and seeded once from the `E3SM_SRCROOT` cache, so tests never re-analyze. Delete the frozen copy to
refresh it, or set `SPEL_TEST_ANALYSIS=/path/to/analysis.pkl`. Don't put binary fixtures under
`spel/scripts/tests/`; `test_sub_parse` greps that directory.

For analysis features, test on small snippets instead: `walk()` in
`spel/scripts/fortran_parser/tests/test_scope_walk.py` parses and walks a routine, and the `StubSub`
pattern in `spel/scripts/tests/test_record_access.py` (or `Sub` in
`spel/transforms/tests/test_acc_to_omp.py`) feeds it to `AccessMapper`.

Compile-check Fortran templates with both compilers (netcdf-fortran `.mod` files are compiler specific):

```bash
gfortran -c -ffree-line-length-none -I$HOME/.local/gcc/netcdf-fortran/include SourceFiles/nc_io.F90 SourceFiles/nc_allocMod.F90
nvfortran -c -I$HOME/.local/netcdf-fortran/include SourceFiles/nc_io.F90 SourceFiles/nc_allocMod.F90
```

Do this in a scratch directory so `.mod`/`.o` files don't land in the repo.

## Building and running unit tests

- Select the toolchain with env vars, which override `check_config.sh`:
  `SPEL_FC=nvfortran|gfortran SPEL_ACC=OFF|MULTICORE|GPU|OPENMP SPEL_DBG=ON|OFF spel run <case> [nsets]`.
- `spel run` prompts for confirmation; use `yes | spel run ...` when running non-interactively.
  In scripts with `set -o pipefail`, `yes` exits with SIGPIPE: check the status of the command
  it feeds, not the pipeline's.
- gfortran builds: `DBG=ON` (the default) is `-O0` plus gcov coverage, and `DBG=OFF` adds no `-O`
  flag, so gfortran timings are meaningless (coverage counters also contend under threads).
  Profile with nvfortran (`-fast`).
- gfortran `ACC=MULTICORE` runs on one thread (GCC has no multicore OpenACC target). To run a
  case multithreaded with gfortran, translate it with `spel acc2omp --case <case>` and build with
  `SPEL_ACC=OPENMP` (`-fopenmp -U_OPENMP`; threads from `OMP_NUM_THREADS`). `-U_OPENMP` keeps
  ELM's own `#ifdef _OPENMP` clump-threading code off, as in the analysis (it needs
  `shr_sys_abort` and aborts outside threaded regions). `spel create` undoes the translation, so
  keep a separate case for it (e.g. `<case>_omp`).
- `spel profile <case> [nsets] --steps N` runs with `SPEL_PROFILE=1` (no outputs written) under
  `nsys`; it needs the same `SPEL_FC`/`SPEL_ACC` vars as the build, or `check_config.sh` reconfigures.
  NVTX ranges come from `SourceFiles/spel_profMod.F90` (nvfortran only); `main.F90` pushes one
  range per routine under test inside `compute`, so `nsys stats --report nvtx_sum` times just that
  routine.
- `SPEL_CLUMPS=N` (run time) splits the `nsets` sets into `N` clumps of whole sets and runs the
  routines in `elm_drv`'s `!$OMP PARALLEL DO` over clumps (`SourceFiles/main.F90`; clumps are set
  up by `init_clumps` in `SourceFiles/elm_initializeMod.F90`). With `ACC=OPENMP` that is ELM's CPU
  threading; it is the fair baseline for master, whose routines have no loop-level threading.
  The clump loop has `IF (nclumps > 1)` so a single-clump OpenMP build doesn't nest the
  `acc2omp` loops. OpenACC builds need one clump. IO uses the process bounds (`bounds_io`);
  `read_elmtypes` overwrites its bounds argument with one set's bounds.
- `nsets=N` with `SPEL_CLUMPS=M` must stay bit-identical to `nsets=1` (no `WARNING (spel_nsets)`).
- Time the NVTX range `compute` (wall time of the clump loop). The per-routine ranges run inside
  the clump threads, so with several clumps they time one clump.
- `profile_compare.sh MASTER_CASE ACC_CASE` (repo root) times `compute` for master serial, master
  and refactor with N clumps x 1 thread (`ACC=OPENMP`), the refactor with 1 clump x N OpenACC
  threads (`ACC=MULTICORE`), and optionally an `acc2omp` case (`--omp-case`); the summary includes
  the speed-up over master on the same thread count. Cases made before `SPEL_CLUMPS` must be
  regenerated. It leaves the cases configured for nvfortran.

## Analysis

- `spel create` reuses the per-srcroot cache in `unit-tests/.spel-cache/<srcroot>-<hash>/elm_drv/`.
  After editing E3SM sources, pass `--reanalyze`; otherwise the generated test may contain
  stale routine code. SPEL warns when HEAD or uncommitted `components/elm` changes differ
  from the cache, but it does not rebuild automatically. The analysis is pickled, so changes to
  the parser, AST or walker (`spel/scripts/fortran_parser/`) also need `--reanalyze`.
- `meta.json` in the cache lists `failures`/`incomplete` routines; a refactor that doesn't parse
  shows up there (e.g. `SemanticError: Undeclared symbol ...`) and `spel create` refuses it.
- **Line numbers are 0-based indices** into the source file everywhere (`LineTuple.ln`,
  `Subroutine.startline`, the record's events and loops). Add 1 when printing file lines.
  (Files with `#include "shr_assert.h"` used to be off by one; fixed in `remove_cpp_directives`.)
- `Subroutine.record` (`SubroutineRecord`, `scope_walk.py`) is the source of truth for a routine:
  declarations (rank, attributes, intent), every access with line, status and origin, calls,
  associate frames, and `loop_nest` (`LoopInfo`: extent, index, parent/children, variables read by
  the bounds, `perfect_nest(n)` for collapse). `Subroutine.record_access` (`record_access.py`) has
  per-line read/write statuses of locals, dummies and globals, with actual arguments resolved
  through the callee's dummy intent. Build new analyses on these rather than on regexes.
- OpenACC directives are comments to the parser; `spel/transforms/acc_directives.py` parses them
  from the source and `acc_to_omp.py` attaches them to the analyzed loops.

## Refactoring a routine against a reference

- Use a new case name for the refactored test (e.g. `soiltemp_acc`) and make
  `unit-tests/input-data/<new>/` with symlinks to the reference case's `spel-*.nc` and a copy of
  its `spel_reference.json` (`profile_compare.sh --link-inputs` does the symlinks). Don't symlink
  the directory: `spel run` writes `fut-outputs*.nc` there.
- `spel diff` skips (and lists) reference variables the development branch no longer uses.
- Target: gfortran `ACC=OFF` bit-for-bit with the reference; nvfortran `ACC=MULTICORE` identical to
  nvfortran `ACC=OFF`; `acc2omp` + gfortran `OPENMP` bit-for-bit at any thread count (a race shows
  up there as a difference from the reference).
- To stay bit-for-bit, keep the reference's summation order: accumulate column sums (over layers or
  patches) in a per-column parallel loop with `!$acc loop seq` inside, not with OpenACC
  `reduction` clauses or reordered partial sums. List every scalar assigned in a compute loop in
  `private`, and give local work arrays a `!$acc data create(...)` region.
- `spel acc2omp --dry-run` doubles as a race check for OpenACC code: `added vs OpenACC` lists
  scalars the analysis privatizes but the OpenACC clauses don't.

## Verifying changes

- `spel run` diffs `fut-outputs*.nc` against `spel-outputs*.nc` in `unit-tests/input-data/<case>/`.
  nvfortran builds differ from master reference data by roundoff (up to about 5e-7 relative);
  compare against a prior SPEL build to check bit-for-bit.
- OpenACC: an `ACC=MULTICORE` build should match `ACC=OFF` to about 1e-9 relative. Larger
  differences mean a race, usually a scalar out-argument missing from `private`.
- `nsets=N` must give output bit-identical to `nsets=1` and print no `WARNING (spel_nsets)` lines.
- After changing line-number handling, `spel instrument <case> --dry-run` checks that call sites
  and declarations are still found (it reports only).

## Don'ts

- Don't modify uncommitted work in the E3SM checkouts (`~/climate-projects/E3SM`, `~/climate-projects/dev_E3SM`)
  unless the user asks; if they do, back the file up first.
- `~/climate-projects/E3SM` (master) may be instrumented: run `spel instrument --undo` before
  analyzing it, and don't reanalyze it to test SPEL changes (use `dev_E3SM`).
- Don't name scratch Python files after stdlib modules (e.g. `/tmp/copy.py`).
