# Compact layout: tests on Perlmutter

Instructions for an agent on Perlmutter. Goal: measure, on the crop
performance configuration, what the generated `SetValues` loop rewrite and
the compact layout of the patch C/N/P flux types do to ELM's run time, and
check that both give results bit-for-bit identical to the unmodified model.
Read `AGENTS.md` (repo root), `spel/transforms/compact/DESIGN.md` and
`DEFERRED.md` first. Record Perlmutter's paths/modules in the AGENTS.md
"Perlmutter" section as you go (it is empty; only edit that section).

## What is being tested

Three variants, built from one E3SM checkout:

| Variant | How | What changes |
|---|---|---|
| `base` | the checkout as is | nothing |
| `loops` | `spel transform compact-rewrite --loops-only -o CASE/SourceMods/src.elm` | the six flux types' `SetValues`: blocked `associate (idx => filter(fi:...))` loops become `do concurrent` (2 files: VegetationDataType.F90, ColumnDataType.F90) |
| `compact` | `spel transform compact-rewrite -o CASE/SourceMods/src.elm` | the same loops, plus compact storage for 75 `veg_cf`, 33 `veg_nf`, 33 `veg_pf` fields (15 files, plus `CompactLayoutMod.F90`) |

Background (laptop, 21-gridcell case): EcosystemDynNoLeaching1 unit test with
nvfortran -fast went from ~400 ms to ~6 ms per step with the loop rewrite
alone, and to ~5 ms with the compact layout (10-16% more). The loop speedup
comes from nvfortran: the associate-name vector subscript is ~1000x slower
there than written out; gfortran shows no such cost. So the compiler matters:
**record which compiler each case uses**, and if the default isn't nvidia, also
run the three variants with the nvidia compiler if pm-cpu supports it for ELM.

## Sources

- SPEL: branch `compact-layout` of `git@github.com:peterdschwartz/SPEL.git`
  (it has to be pushed first; ask the user if it's missing). Install with
  `pip install -e .` in a Python 3.12 venv; `spel config --set-srcroot $E3SM`.
- E3SM: branch `peterdschwartz/lnd/gpu-refactor-with-crop-perf-changes`,
  commit `002008f571` (what the rewrite was developed and checked against),
  with submodules (`git submodule update --init --recursive`; the analysis needs
  `components/elm/src/external_models/fates`). A later commit is fine, but then
  the counts below may differ; report them.
- Optional: `spel/transforms/compact/elm-time-flags.patch` (uncommitted on the
  user's laptop): `timeInfoMod%first_restart_step_mod`, set in elm_drv, replaces
  `is_first_step()`/`is_first_restart_step()` in EcosystemDynNoLeaching1 and
  three ColumnDataType SetValues. Not needed for these runs (behavior is
  identical); **don't apply it** unless asked, so `base` is the branch as pushed.

## Steps

1. Analyze the checkout (about a minute; it must not be instrumented, see
   AGENTS.md):

   ```bash
   spel analyze --srcroot $E3SM
   ```

2. Check the selection and dry runs. Expected at `002008f571`:

   ```bash
   spel compact inventory -o $SCRATCH/compact/inventory.json
   #   instance  fields compact +maps blocked unused
   #   veg_cf       125     105    17       1      2
   #   veg_nf        75      61    14       0      0
   #   veg_pf        74      62    11       1      0
   #   col_cf        65      55     7       3      0
   #   col_nf       106      90     7       9      0
   #   col_pf        96      85     7       4      0
   spel transform compact-rewrite --dry-run
   #   vegetation_carbon_flux: 75 fields, 626 access sites
   #   vegetation_nitrogen_flux: 33 fields, 281 access sites
   #   vegetation_phosphorus_flux: 33 fields, 278 access sites
   #   <type>_setvalues: N blocked loops -> do concurrent   (4+3+3+5+6+7 = 28)
   spel transform compact-rewrite --loops-only --dry-run      # the 28 loops only
   ```

   The rewrite is refused (nothing written, the reason in the report) if a
   site's text doesn't match the analysis (sources changed since `spel
   analyze`): re-run `spel analyze`.
   "uses outside the elm_drv trace" lines marked `(another type)` are expected
   (same-named components of col/grc types).

3. Create the three cases with the crop preset. Each in its own `--group`
   (the preset's `--auto-name` gives all three the same name). Use the
   resolution/compset the user uses for crop performance runs; if unsure, ask.
   The preset builds right away; SourceMods are added afterwards and the case
   rebuilt (incremental).

   ```bash
   cd SPEL
   for v in base loops compact; do
     ./e3sm_casegen.sh --preset crop-perf --srcroot $E3SM --res t-lr25 --with-crops \
         --group E3SM_testings/compact_$v | tee $SCRATCH/compact/casegen_$v.log
   done
   CASE_base=$(grep '^CASEDIR=' $SCRATCH/compact/casegen_base.log | cut -d= -f2)     # likewise loops, compact
   spel transform compact-rewrite --loops-only -o $CASE_loops/SourceMods/src.elm
   spel transform compact-rewrite              -o $CASE_compact/SourceMods/src.elm
   (cd $CASE_loops && ./case.build) && (cd $CASE_compact && ./case.build)
   ```

   - To regenerate SourceMods, first remove exactly the files of the earlier
     rewrite: those `.spel-compact.json` lists under `files` (by basename),
     `CompactBindMod.F90`, `CompactLayoutMod.F90` if written, and the two
     manifests `.spel-compact.json`/`.spel-transforms.json`. Never delete with
     globs and never edit the generated files by hand.
   - Never run `spel transform compact-rewrite` without `-o` on the shared checkout.
   - Changing `GMAKE_J` or other env_build settings after setup needs
     `./case.build --clean-all`.

4. History for the comparison: in each case's `user_nl_elm` (same in all three),
   daily averages so a 5-day run has 5 records:

   ```
   hist_nhtfrq = -24
   hist_mfilt  = 5
   ```

   If the preset's `user_nl_elm` already sets these, replace them (duplicates
   are a namelist error). Then submit all three (`./case.submit`), same queue
   and node type.

5. Correctness:

   ```bash
   python spel/transforms/compact/tools/compare_history.py $RUN_base $RUN_loops
   python spel/transforms/compact/tools/compare_history.py $RUN_base $RUN_compact
   ```

   Expected: 0 variables differ. The rewrite only moves where values are
   stored and how zeros are written; no arithmetic changes, so optimized builds
   should be bit-for-bit too. If anything differs, report the variables and
   sizes, then rerun the two cases with `DEBUG=TRUE` for 1 day (bounds and
   pointer checks catch a wrong compact index) before investigating further.
   Each compact run logs its buffers in `lnd.log` (`veg_cf: compact buffer: 75
   fields ...`); check it's there.

6. Timings:

   ```bash
   python spel/transforms/compact/tools/timers.py $CASE_base $CASE_loops $CASE_compact
   ```

   Timers of interest: `elm_run`, `ecosysdyn`, `CNZero` and `CNZero-<type>`,
   `CNPsum`, `SoilLittDecompAlloc`, `hbuf`, and `compact_unpack` (the copy of the
   compact fields to their full layout for history, compact case only; it is
   overhead to subtract when judging the layout itself). Run each case at
   least twice (resubmit) and report both; the spread matters more than one
   number.

## What to report back

- Commits used (SPEL, E3SM), compiler(s), resolution/compset, NTASKS/NTHRDS.
- The counts from step 2 if they differ from the expected ones.
- `compare_history.py` results for loops and compact vs base.
- The `timers.py` tables for every run.
- Anything that failed, with the error text.

## Notes and limits

- CPU only. The compact layout has no OpenACC support yet (device copies of the
  buffers); don't build it for pm-gpu or with `ACC`. The loop rewrite alone is
  fine for any build.
- `compact_soilp_off` is thread-private and set by `get_clump_bounds`; runs with
  NTHRDS > 1 should work (tested on the laptop with 2 OpenMP clumps in unit
  tests). If time allows, one compact run with NTHRDS=2 vs base is useful.
- Isotopes (`use_c13/c14`) compile but are untested at run time; FATES is not
  supported (veg flux fields aren't allocated under FATES; registration skips them).
- CNHarvest (`dynHarvestMod`) is not reached by SPEL's analysis; the harvest
  fields it writes are excluded from compaction automatically.
- Laptop results to compare against: 3-day 21-gridcell runs, gfortran debug,
  all variants bit-for-bit; unit tests of ecosystemdynnoleaching1 and
  maintenanceresp from the rewritten sources bit-for-bit against ELM data.
- Optional, only if asked: unit tests on crop data. Capture with
  `spel validate --srcroot $E3SM --list cases.txt` where cases.txt has
  `eco1[crop; max=20]: ecosystemdynnoleaching1`, then make a scratch worktree of
  the checkout with `spel transform compact-rewrite` applied in place (commit it there),
  `spel analyze --srcroot $WT`, `spel create -s ecosystemdynnoleaching1 -c eco1_c
  --srcroot $WT`, link `unit-tests/input-data/eco1_c/` to eco1's `spel-*.nc`
  (symlinks) plus a copy of `spel_reference.json`, and `spel run eco1_c`.
  Time with `SPEL_FC=nvfortran SPEL_ACC=OFF SPEL_DBG=OFF spel profile <case> <nsets>`.
