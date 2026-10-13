# AGENTS.md

Notes for AI agents working on SPEL. Read `README.md` for the CLI and the
reference-vs-development workflow; this file only covers what the README doesn't.

## Generated vs. hand-edited files

- `SourceFiles/nc_io.F90` is **generated**. Edit `spel/scripts/io/netcdf_io.py`, then
  regenerate with `python -m spel.scripts.io.netcdf_io`.
- Other `SourceFiles/*.F90` (`main.F90`, `elm_initializeMod.F90`, `nc_allocMod.F90`, ...) are
  templates/static files copied into each case; `!#...` markers in them are filled in by the generator.
- `duplicateMod.F90`, `ReadWriteMod.F90` and the case's ELM sources are generated per case.
  Everything under `unit-tests/<case>/` is overwritten by `spel create`; fix the generator, not the case.

## Testing

```bash
source ~/climate-projects/py12env/bin/activate
python -m pytest -q
```

Tests that need a whole-ELM analysis use the session fixture `elm_drv_analysis` (root
`conftest.py`). It loads the frozen copy `.pytest-fixtures/elm_drv_analysis.pkl`, which is gitignored
and seeded once from the `E3SM_SRCROOT` cache, so tests never re-analyze. Delete the frozen copy to
refresh it, or set `SPEL_TEST_ANALYSIS=/path/to/analysis.pkl`. Don't put binary fixtures under
`spel/scripts/tests/`; `test_sub_parse` greps that directory.

Compile-check Fortran templates with both compilers (netcdf-fortran `.mod` files are compiler specific):

```bash
gfortran -c -ffree-line-length-none -I$HOME/.local/gcc/netcdf-fortran/include SourceFiles/nc_io.F90 SourceFiles/nc_allocMod.F90
source ~/.local/nvhpc-libs/env.sh
nvfortran -c -I$HOME/.local/nvhpc-libs/netcdf-fortran/include SourceFiles/nc_io.F90 SourceFiles/nc_allocMod.F90
```

Do this in a scratch directory so `.mod`/`.o` files don't land in the repo.

## Building and running unit tests

- Run `source ~/.local/nvhpc-libs/env.sh` before any nvfortran build.
- Select the toolchain with env vars, which override `check_config.sh`:
  `SPEL_FC=nvfortran|gfortran SPEL_ACC=OFF|MULTICORE|GPU SPEL_DBG=ON|OFF spel run <case> [nsets]`.
- `spel run` prompts for confirmation; use `yes | spel run ...` when running non-interactively.
- `spel profile <case> [nsets] --steps N` runs with `SPEL_PROFILE=1` (no outputs written) under
  `nsys`; it needs the same `SPEL_FC`/`SPEL_ACC` vars as the build, or `check_config.sh` reconfigures.
  NVTX ranges come from `SourceFiles/spel_profMod.F90` (nvfortran only).
- Only test on the local machine unless the user asks otherwise.

## Analysis cache

`spel create` reuses the per-srcroot cache in `unit-tests/.spel-cache/<srcroot>-<hash>/elm_drv/`.
After editing E3SM sources, pass `--reanalyze`; otherwise the generated test may contain
stale routine code. SPEL warns when HEAD or uncommitted `components/elm` changes differ
from the cache, but it does not rebuild automatically.

## Verifying changes

- `spel run` diffs `fut-outputs*.nc` against `spel-outputs*.nc` in `unit-tests/input-data/<case>/`.
  nvfortran builds differ from master reference data by roundoff (up to about 5e-7 relative);
  compare against a prior SPEL build to check bit-for-bit.
- OpenACC: an `ACC=MULTICORE` build should match `ACC=OFF` to about 1e-9 relative. Larger
  differences mean a race, usually a scalar out-argument missing from `private`.
- `nsets=N` must give output bit-identical to `nsets=1` and print no `WARNING (spel_nsets)` lines.

## Don'ts

- Don't modify uncommitted work in the E3SM checkouts (`~/climate-projects/E3SM`, `~/climate-projects/dev_E3SM`).
- Run `spel instrument --undo` before analyzing an instrumented checkout.
- Don't name scratch Python files after stdlib modules (e.g. `/tmp/copy.py`).
