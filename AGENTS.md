# AGENTS.md

Notes for AI agents working on SPEL. Read `README.md` for the CLI and the
reference-vs-development workflow; this file only covers what the README doesn't.

## Environment

The user works on several machines with this repo, and this file is shared between them.
**Find out which one you're on** (`lscpu`, `nvidia-smi`, `ls ~/climate-projects/py12env`) and
edit only that machine's notes. Don't replace another machine's section with what you find
locally. Earlier edits made that mistake.

Common to all: the gfortran netcdf/mpich builds live in `~/.local/gcc/` (the default
`PKG_CONFIG_PATH`). Production profiling runs on Perlmutter; only test locally unless the user
asks otherwise.

Machine specifics (toolchains = modules/`PKG_CONFIG_PATH` per compiler, scratch, cprnc, casegen
defaults) are in machine files: `spel/machines/{perlmutter,chrysalis}.toml` (bundled, detected
automatically; toolchains take their modules from the E3SM checkout's `config_machines.xml`) or
`~/.config/spel/machines/<name>.toml` (`spel machine init`; the laptop and workstation need one,
written from the notes below). `spel machine` shows what applies, `spel machine check` verifies
the toolchains, `spel run`/`spel profile`/`spel validate` build inside the toolchain of `SPEL_FC`,
`profile_compare.sh` and `e3sm_casegen.sh --preset` read the same file. `SPEL_MACHINE=none`
falls back to the current shell environment. When a machine's setup changes, change its machine
file, not these notes alone.

### Perlmutter

Production profiling and large (e.g. crop, `--preset crop-perf`) reference runs. Recorded on
2026-10-19.

- Login nodes are shared and have A100 GPUs. Build and run unit tests on an interactive CPU node
  (`salloc -N 1 -q interactive -C cpu -t 2:00:00 --no-shell`, then
  `srun --jobid=JOB --overlap -N1 -n1 -c 64 ...`); Slurm jobs need the user's OK.
- `spel` comes from the NERSC python module (`nersc-python` 3.13, conda env 26.8.1); the user
  install is `~/.local/perlmutter/python-3.13/26.8.1/bin/spel`.
- Crop performance runs: `./e3sm_casegen.sh --preset crop-perf --res t-lr25 --with-crops
  --compressed-cft` (pm-cpu default compiler = Intel, 1024 tasks, 5 days, debug queue).
- Performance cases must all be the same kind of build: the crop-perf preset makes threaded builds
  (`FORCE_BUILD_SMP=TRUE`, `-qopenmp`, NTHRDS=1; machine file `casegen.nthrds = 1`). A non-threaded Intel
  build of identical code ran 3-8% slower on untouched BGC routines (summer stride-8, 2026-10-22), which
  looked like a code regression. Don't `xmlchange` a case while its `case.build` runs ("env_run.xml appears
  to have changed without a corresponding invalidation"; recover with `case.setup --reset`).
- Stride-8 summer window: `--finidat $SCRATCH/stride8/spin30m.elm.r.0003-07-01.nc --start-date 0003-07-01
  --datm-years 1901 1903` (symlink: the real path exceeds the namelist string limit). A case without the
  start date silently runs January from the July restart (it happened once: canflux -20%, surfalb +10%).
- Unit-test toolchains (`spel/machines/perlmutter.toml`, all verified 2026-10-22 with
  `spel machine check`): `gfortran` (PrgEnv-gnu, gcc-native 14, the default), `nvfortran`
  (PrgEnv-nvidia, nvidia/25.9 = `/opt/nvidia/hpc_sdk/Linux_x86_64/25.9`) and `ifx`; modules from
  E3SM's pm-cpu entry plus `$CRAY_MPICH_DIR/lib/pkgconfig` (the Cray netcdf-fortran `.pc` needs
  the mpich one). The nvfortran toolchain sets `ACC_DEVICE_TYPE=multicore`: otherwise the
  OpenACC runtime may try the login node's GPU and loop forever printing `Failing in Thread:1`.
  `SPEL_DBG` defaults to ON (`-O0` + coverage); use `SPEL_DBG=OFF` for timing.
- `cprnc` is `/global/cfs/cdirs/e3sm/tools/cprnc/cprnc`; restart files need `-m`.

### Chrysalis (LCRC, AMD EPYC, 64 cores/node)

CPU crop-performance runs, which moved here from Perlmutter. Recorded on 2026-10-21.

- `spel` comes from `~/spel-py314` (spack Python 3.14; `~/myenv` is 3.11, too old). Unit tests build with
  Intel: the `ifx` toolchain of `spel/machines/chrysalis.toml` (modules from E3SM's chrysalis intel/openmpi
  entry; written from `config_machines.xml`, not yet run there: check with `spel machine check` and fix the
  machine file if `pkg-config netcdf-fortran` is not found). The older manual route is
  `source ~/spel-intel-env.sh` (the ELM case's `.env_mach_specific.sh`, `SPEL_FC=ifx`), also usable as
  `spel validate --test-env` or as the toolchain's `env_file`. There is no gfortran netcdf build here.
- E3SM checkouts: `~/integration/E3SM` (upstream bishtgautam/lnd/add-t-meshes-v3 @ e5aa4e3094,
  the baseline) and its worktree `~/E3SM-gpu-refactor` (peterdschwartz/lnd/gpu-refactor-nutrient-types,
  which contains the crop-perf branch). Only these two are compared.
- Handoff notes and user mods: `/lcrc/group/e3sm/ac.schwartzpd/crop_case_files`
  (`CHRYSALIS_HANDOFF.md`, `user_mods/stride8_chrys`). Inputs: `/lcrc/group/e3sm/ac.schwartzpd/crop_inputdata`
  (`stride8/` in inputdata layout, `spin30m_restart/` = the 0003-07-01 stride-8 restart; the paramfile
  is a symlink into `crop_reduced_inputdata/`).
- Stride-8 cases are the workhorse. Full 1024-task t-lr25 cases (16 nodes, debug queue) run only when the
  user asks: `./e3sm_casegen.sh --preset crop-perf --res t-lr25 --no-crops` and `--with-crops
  --compressed-cft --remove-zero-weight-urban`. The non-crop and 50-pft t-lr25 surfdata are in shared
  inputdata; the domain and `MOSART_u.oi6to18.lr25.nc` aren't, and casegen takes them from
  `COMPRESSED_CFT_DIR` for any t-lr25 case.
- `e3sm_casegen.sh --preset crop-perf` detects Chrysalis by hostname: `chrysalis`, Intel, cases under
  `/lcrc/globalscratch/$USER/E3SM_testings`, debug queue (account `e3sm`, up to 20 nodes).
  It forces NTHRDS=1 and 64 MPI tasks per node: the default layout for these compsets is 32 tasks x 2 threads.
- Stride-8 summer window (128 tasks = 2 nodes, the same gridcells per rank as Perlmutter's 1024-task case):
  `./e3sm_casegen.sh --preset crop-perf --res ELM_USRDAT --with-crops --user-mods
  /lcrc/group/e3sm/ac.schwartzpd/crop_case_files/user_mods/stride8_chrys --ntasks 128 --no-daily-history
  --finidat <spin30m restart> --name-suffix stride8.spin30m --srcroot <checkout>`. CIME accepts the
  absolute `finidat`. Builds run on the login node; batch jobs need the user's OK. CIME's
  case-directory git can race with the job right after `case.submit` (`index.lock` exists): resubmit.
- Put CIME cases and all SPEL unit tests on `/lcrc/globalscratch/ac.schwartzpd` (the user's choice,
  2026-10-22); `/lcrc/group/e3sm` (fs2) is nearly full and briefly ran out of space on 2026-10-21.
  `SPEL/unit-tests` is a symlink to `/lcrc/globalscratch/ac.schwartzpd/spel-unit-tests`; keep only
  symlinks in the SPEL tree. The earlier cases (stride-8, baseline t-lr25) are still under
  `/lcrc/group/e3sm/ac.schwartzpd/scratch/chrys/E3SM_testings`.
- `cprnc` is `/lcrc/group/e3sm/tools/cprnc/cprnc`.

## Generated vs. hand-edited files

- `SourceFiles/nc_io.F90` is **generated**. Edit `spel/scripts/io/netcdf_io.py`, then
  regenerate with `python -m spel.scripts.io.netcdf_io`.
- Other `SourceFiles/*.F90` (`main.F90`, `elm_initializeMod.F90`, `nc_allocMod.F90`, ...) are
  templates/static files copied into each case; `!#...` markers in them are filled in by the generator.
- `duplicateMod.F90`, `ReadWriteMod.F90` and the case's ELM sources are generated per case.
  Everything under `unit-tests/<case>/` is overwritten by `spel create`; fix generator output (harness
  files) in the generator, not the case. ELM source changes are prototyped in the case and brought back
  with `spel restore` (see "Prototype in an existing case" below).
  The case `CMakeLists.txt` comes from `generate_cmake` in `spel/scripts/functional_unit_test.py`.

## Testing

```bash
python -m pytest -q
```
Tests that need a whole-ELM analysis use the session fixture `elm_drv_analysis` (root
`conftest.py`). It loads the frozen copy `.pytest-fixtures/elm_drv_analysis.pkl`, which is gitignored
and seeded once from the `E3SM_SRCROOT` cache, so tests never re-analyze. Delete the frozen copy to
refresh it, or set `SPEL_TEST_ANALYSIS=/path/to/analysis.pkl`. Don't put binary fixtures under
`spel/scripts/tests/`; `test_sub_parse` greps that directory.

For analysis features, test on small snippets instead: `walk()` in
`spel/scripts/fortran_parser/tests/test_scope_walk.py` parses and walks a routine, and the `StubSub`
pattern in `spel/scripts/tests/test_record_access.py` (or `Sub` in
`spel/transforms/tests/test_acc_to_omp.py`) feeds it to `AccessMapper`.

Compile-check Fortran templates with both compilers (netcdf-fortran `.mod` files are compiler
specific; the nvfortran include path is per machine, see Environment):

```bash
gfortran -c -ffree-line-length-none -I$HOME/.local/gcc/netcdf-fortran/include SourceFiles/nc_io.F90 SourceFiles/nc_allocMod.F90
nvfortran -c -I$HOME/.local/nvhpc-libs/netcdf-fortran/include SourceFiles/nc_io.F90 SourceFiles/nc_allocMod.F90  # laptop
nvfortran -c -I$HOME/.local/netcdf-fortran/include SourceFiles/nc_io.F90 SourceFiles/nc_allocMod.F90             # GPU workstation
```

Do this in a scratch directory so `.mod`/`.o` files don't land in the repo.

## Building and running unit tests

**Prototype in an existing case; don't regenerate it.** To try a source change (a directive, a loop
rewrite, a kernel restructure), edit the case's copy (`unit-tests/<case>/<Module>.F90`) and rerun
`spel run`/`spel profile`: `build/` is kept, so make recompiles only what changed (`make -j`). Bring the
result back with `spel restore -c <case>` (it applies only your edits, diffed against the pristine
copies `spel create` keeps in `<case>/.spel-orig/`, to the srcroot files, and refuses on conflicts).
Run `spel create` only when the case itself must change (other routines, captured variables, IO,
generator fixes): it deletes the case, `build/` included, and costs a full rebuild. Larger domains
come from `nsets` (`spel run <case> 128`), not from a new capture.

- Select the toolchain with env vars, which override `check_config.sh`:
  `SPEL_FC=nvfortran|gfortran SPEL_ACC=OFF|MULTICORE|GPU|OPENMP SPEL_DBG=ON|OFF spel run <case> [nsets]`.
  On a machine with a machine file, `spel run` sets up `SPEL_FC`'s toolchain (modules, pkg-config)
  itself (`spel/scripts/run_fut.py::toolchain_environment`); `spel machine env --fc X` prints it for
  running `./check_config.sh` by hand.
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

## Time-step state in unit tests

- Module variables are captured once (constants file); derived-type components every step.
  State that changes every step is captured per step instead, through `spel_t_<name>(1)` in
  ReadWriteMod, and copied back after each read (`spel/scripts/io/time_scalars.py`, main.F90
  `!#TIME_SYNC_INIT/STEP`): every `timeInfoMod` scalar, and the globals a case reads that
  something in elm_drv's tree writes (`step_written_globals`, e.g. `subgrid_weights_updated`).
- `elm_time_manager` stays out of unit tests (it needs ESMF). A `use elm_time_manager, only :`
  of just `is_first_step`/`is_first_restart_step` is pointed at `SourceFiles/spel_timeMod.F90`,
  whose values the capture records from `get_nstep()`/`is_first_restart_step()`.
- Adding a per-step variable changes the IO: reference data captured before it must be recaptured.

## Developing SPEL: build on the parser, not on regexes

SPEL has its own Fortran lexer, parser, AST and scope walker (`spel/scripts/fortran_parser/`:
`lexer.py`, `spel_parser.py`, `spel_ast.py`, `scope_walk.py`, `symbols.py`), and analyses built on
them (`record_access.py`, `registrations.py`, `spel/transforms/compact/index_forms.py`). A new
feature that needs to know what a statement *is* or which names it uses must get that from them:

- Take structure from the parse tree and the record: statements and constructs from the AST
  (`registrations.parse_routine`, `iter_statements`/`line_guards` for enclosing ifs), accesses,
  loops, assignments and calls from `Subroutine.record` (`SubroutineRecord`: `events`, `loop_nest`,
  `associations`, `lookup`), read/write status and argument bindings from `Subroutine.record_access`,
  index forms and filters from `IndexContext`. Expressions are AST nodes; walk them instead of
  re-parsing their text.
- Don't scan source text with regexes to recognise statements, declarations, calls, loop bounds or
  variable names. Regex scans have caused real bugs here (`endif` not recognised, names inside
  strings and keyword arguments counted as variables, continuation lines missed). Regexes are fine
  only for pure text work that classifies nothing: placing an edit at a known line, renaming a
  token the parser has already identified, or reading non-Fortran text.
- If the parser or walker lacks what you need (a construct it rejects, a node that drops
  information, an event the walker doesn't record), **extend the parser** rather than working around
  it: add the node/field/event, make any behaviour change opt-in when existing callers depend on the
  old output (as `keep_string_case`/`keep_directives` were), and add a snippet test in
  `spel/scripts/fortran_parser/tests/` (`walk()` in `test_scope_walk.py` parses and walks a routine).
- Parser, AST or walker changes alter the pickled analysis: rerun with `--reanalyze` and refresh the
  frozen test fixture (delete `.pytest-fixtures/elm_drv_analysis.pkl`).
- Older modules (`LoopConstructs.py`, parts of `utilityFunctions.py`) are line- and regex-based.
  Don't extend them; port what you need onto the record instead.

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
  (Files with `#include "shr_assert.h"` used to be off by one; fixed in `remove_cpp_directives`.
  Lines after a removed cpp block of about 8+ lines, which cpp collapses into one line marker, were
  shifted until 2026-10-28; fixed in `modify_file`/`check_sections`. Edits placed by earlier analyses
  in such files may be misplaced.)
- `Subroutine.record` (`SubroutineRecord`, `scope_walk.py`) is the source of truth for a routine:
  declarations (rank, attributes, intent), every access with line, status and origin, calls,
  associate frames, and `loop_nest` (`LoopInfo`: extent, index, parent/children, variables read by
  the bounds, `perfect_nest(n)` for collapse). `Subroutine.record_access` (`record_access.py`) has
  per-line read/write statuses of locals, dummies and globals, with actual arguments resolved
  through the callee's dummy intent. Build new analyses on these rather than on regexes.
- OpenACC directives are comments to the parser; `spel/transforms/acc_directives.py` parses them
  from the source and `acc_to_omp.py` attaches them to the analyzed loops.

## Database and web app (`spel export`, `spel/db`)

- Pipeline: `spel analyze` → `spel export` (CSVs + `snapshot.json` in `script-output/db-export/<label>/`,
  code in `spel/scripts/db_export/`, columns in `schema.py`) → `python spel/db/manage.py
  update_all_data DIR [--replace]` (`spel/db/app/management/loaders.py`) → `import_presets`.
  `spel/scripts/export_objects.py` only holds the pickle helpers now.
- Everything in the DB belongs to a `Snapshot`; the models keep their old names (`Modules`,
  `Subroutines`, `SubroutineCalltree`, ...) plus `Snapshot`, `SubroutineCode`, `TypeProcedure`,
  `Registration`, `TraceNode`/`TraceAccess` (the static step: a DFS from `elm_drv` in source order,
  dummies resolved to the actuals bound at each call) and `FieldState`. Line numbers stay 0-based.
- Categories (`spel/scripts/db_export/fieldstate.py::categorize`, also used by the views per
  configuration): prognostic = a read before the step's first *unconditional* write (or only
  conditional writes); input = read only; diagnostic = unconditional write first + history
  registration; intermediate = unconditional write first, no history; unused. "Conditional" =
  inside a plain (non-namelist) if block or in a routine called from one (`TraceAccess.conditional`).
  Registration flags are facts; `restart_unneeded` / `restart_missing` are findings for review.
- Accesses through pointer components of module instances (`transport_ptr_list(i)%trcr_tend_ptr`)
  are expanded to the component's recorded targets (`DerivedType.components[x].pointer`, filled
  from every `=>` in the module by `allocations.scan_module`), transitively, in the exporter
  (`GlobalResolver.expand`); `TraceAccess.resolved_from == "pointer"` marks such rows. Pointers
  assigned at run time to something the module scan never sees stay unexpanded (conservative).
- Filters: every trace access carries `index_kind` (the form of its subgrid subscript, from
  `spel/transforms/compact/index_forms.py`: `IndexContext`, shared with the compact tracer through
  `callee_context`) and `filter` (`filter%lakec` when the index is an element of a resolved filter, or
  a range/mapping drawn from one; empty = may touch any element). Assignments into a `clumpfilter`
  member (`StaticTrace.filter_site`) are the filter-building sites; `spel/scripts/db_export/filters.py`
  derives the `filters` and `filter_relations` tables from them (disjoint from if/else branches, also
  across levels when the condition is on a coarser level; subset from nested conditions or a loop over
  another filter; closure; plus the curated landunit-family facts in `FAMILIES`/`CURATED_DISJOINT`).
  Unknown pairs overlap. The tree graph (`q.dependency_graph`) drops a shared variable only when every
  write and read run under known disjoint filters; `/filters/` shows the matrices and the sources.
- The "Tree graph" page is a layered box diagram of one routine's callees: rows = data-dependency
  depth, left to right = call order; book-keeping routines (`q.BOOKKEEPING`) hidden by default; rows
  and direct (transitively reduced) edges are recomputed in the browser from the routines shown.
- Namelist cascades (`spel/scripts/nml/namelist_cascade.py::NML_CASCADES`) give the value of a
  non-namelist variable that follows from namelist variables, so the ifs testing it are evaluated per
  configuration. A cascade may have several triggers (`carbon_only` from `suplnitro`, `suplphos`,
  `spinup_state`): each pair's `nml_val` is then a tuple and applies only when all match, and in the
  DB the rows of one mapping share `CascadePair.pair_key`. Combinations no pair lists stay unknown
  (the if stays active). A multi-trigger cascade implies nothing about a single trigger for unit-test
  guards. Adding one changes which ifs are namelist-gated: re-run `spel analyze` and `spel export`.
- `hist_addfld*`/`restartvar` calls are recorded by the analysis (`spel/scripts/registrations.py`,
  `Subroutine.registrations`) by parsing the original routine bodies; the edited `src/` copies keep
  them commented out. Analysis attributes the export reads need `spel analyze` again after changes.
- Settings: SQLite by default (`spel/db/explorer.sqlite3`, gitignored), also for the hosted site
  (`SPEL_DB_PATH`, deployment in `spel/db/DEPLOY.md`); `DB_ENGINE=mysql` switches to MariaDB. All
  queries go through the ORM; the only backend-specific code is SQLite PRAGMAs guarded by
  `connection.vendor`. Keep it that way. `spel/db/.env` is gitignored (template `.env.example`); never commit secrets.
- Tests: `spel/scripts/tests/test_db_export.py` (no Django), `spel/db/app/tests/` (pytest-django,
  in-memory SQLite; skipped without the `db-test` extra). Evaluation and reachability are cached
  in LocMem for 10 minutes: restart `runserver` after reloading a snapshot.
- UI: HTMX 2 partials + bespoke JS (`static/js/{explorer,tracelist,timeline,layered,calltree,modgraph,toggle}.js`);
  d3 v7 (cdnjs) only for the timeline and the two layered graphs. `layered.js` is the shared engine
  of the tree graph and the module graph (rows by dependency depth, channel-routed edges, transitive
  reduction in the browser). Pages take `?snap=&cfg=`. Templates load JS/CSS with `{% static_v %}`
  (`templatetags/staticv.py`, mtime query string) so edits reach browsers without a hard reload; a
  new templatetag library needs a `runserver` restart.
- Every search box is fuzzy (`q.fuzzy_filter`/`fuzzy_score`); keep new ones that way. The graph
  root boxes take the best fuzzy match when the name is not exact.
- The loader must load every table in `schema.TABLES` (it checks the CSVs exist, not that it reads
  them: `module_deps` was once silently skipped). Add a loader step with any new table.

## Compact layout (`spel compact`)

- Since 2026-10-22 the E3SM branch no longer carries the compact rewrite: the layout is applied per
  case at build time, `e3sm_casegen.sh --compact` (= `spel compact rewrite --srcroot SRC --reanalyze
  -o CASE/SourceMods/src.elm`, case name gets `.compact`; `COMPACT_ARGS` env adds options such as
  `--roles`). The branch does carry `CompactLayoutMod.F90` (the history readers use it); the rewrite
  requires it to equal the SPEL template `spel/transforms/compact/templates/CompactLayoutMod.F90` and
  does not write it again. Captures and hand edits work on unrewritten sources.
- v2 (2026-10-24): fields live only in per-group buffers over the process soil filters, clumps are
  contiguous slices, history reads them through the inverse map in the kernels (`histComputeMod`), a
  generated `CompactBindMod` resizes and rebinds when the filters change. No unpack point any more.
  Design and rules: `spel/transforms/compact/DESIGN.md`; deferred fields: `DEFERRED.md`.
- The inventory is deliberately conservative: false positives (e.g. value-dependent guards like
  `ch4_prod`'s `lake`) are reviewed by hand, not analyzed away.
- `rewrite` needs the analysis of exactly the sources it edits (`spel analyze` first) and refuses
  sources that already have `CompactBindMod.F90` or `.spel-compact.json`. Test it with
  `-o $CASE/SourceMods/src.elm` against a baseline case built from the same commit (bit-for-bit
  history files; `$SCRATCH/prsplit/validate_level.sh LABEL SRC REF` with `COMPACT_ARGS` does this).
  Don't run analysis probes on a srcroot while a `--reanalyze` of it runs (truncated pickle).
- `CompactLayoutMod` must only use share modules (`shr_*`): `decompMod` uses it, so ELM modules
  like `abortutils` would make a module cycle.
- Unit tests of rewritten sources get a generated `SpelCompactMod.F90` (shadows packed around
  `compute`); `cgrp_id` is never captured. Captures gather the full layout through the inverse map and
  need one thread per process.

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
- `private` arrays need compile-time bounds: nvfortran `ACC=MULTICORE` corrupts the heap
  (`free(): invalid pointer`, even on one thread) with automatic ones such as `x(-nlevsno+1:0)`.
  Size them with a parameter (`elm_varpar::nlevsno_max`) and assign explicit sections
  (`x(-nlevsno+1:0) = ...`). Private elements a thread never sets are NaN on the GPU but stale
  finite values on the CPU, so a GPU-only NaN often means an uninitialized private or a local that
  is never copied to the device.

## Verifying changes

- `spel run` diffs `fut-outputs*.nc` against `spel-outputs*.nc` in `unit-tests/input-data/<case>/`.
  nvfortran builds differ from master reference data by roundoff (up to about 5e-7 relative);
  compare against a prior SPEL build to check bit-for-bit.
- OpenACC: an `ACC=MULTICORE` build should match `ACC=OFF` to about 1e-9 relative. Larger
  differences mean a race, usually a scalar out-argument missing from `private`.
- `nsets=N` must give output bit-identical to `nsets=1` and print no `WARNING (spel_nsets)` lines.
- After changing line-number handling, `spel instrument <case> --dry-run` checks that call sites
  and declarations are still found (it reports only).

## Gridcell subsets

Purpose: extract reference data for a few gridcells (up to `--subset-max`, default 100) of a
large simulation, e.g. the urban, lake or crop gridcells, without capturing the whole domain.
Usage is in the README ("Gridcell subsets"): `--subset MASK` on `spel instrument`, `spel create
--instrument/--run-case` and `spel validate`, or `case[MASK; max=N]: subs` in a validate list file;
`spel subset SRC DST --gridcells 3,5,7` cuts existing data offline.

Design (keep these properties when changing it):

- The subset files must look like the capture of a smaller domain: whole gridcells with all their
  topounits/landunits/columns/patches, in their original order, renumbered from 1 (bounds,
  `lbounds`/`ubounds` attributes and index fields). The unit-test side has no subset code;
  don't add any.
- Selection runs in ELM at the first capture (`spel_select_subset`, generated by
  `capture_select_routine` in `spel/scripts/subset.py` into `SpelCapture_<tag>`), so masks may
  use any ELM field, e.g. `lun_pp%wtgcell`, which captures usually lack. Presets need weight > 0:
  every gridcell has zero-weight urban landunits when `run_zero_weight_urban` is on.
  Presets live in `PRESETS`; custom masks are `<level>: <Fortran expression>` with index g/t/l/c/p.
- `nc_io` does the writing while a `spel_subset_type` is active (`nc_subset_activate`, one subset
  per case, so several cases in one ELM build can use different masks): `nc_subset_select`
  gathers along the first subgrid dimension and renumbers the fields of `SUBGRID_INDEX_FIELDS`
  (`netcdf_io.py`), which `duplicateMod` also uses. A new subgrid index field must go there, or
  subsets and `nsets` copies point at the wrong entries. A variable whose subgrid extent isn't the
  process's stops the run (`nc_subset_n`).
- Without `--subset` the capture module is byte-identical to before; keep it that way.
- MPI: the rank with the most matches writes, the others write nothing. Merging ranks into one
  subset is not implemented.
- Provenance: global attributes `spel_subset_mask/gindex/latdeg/londeg` in every file, `subset` in
  `spel_reference.json` (printed by `spel run`), and `SPEL subset` lines in the run's `e3sm.log`.

Testing:

- `spel/scripts/tests/test_subset.py`: mask parsing, capture-module generation, and a gfortran
  program writing a synthetic capture (non-unit process bounds) through `nc_io`, compared
  bit-for-bit with the offline subsetter (`subset_dataset`, `diff_datasets`).
- End to end (done on the laptop on 2026-10-18, all PASS): one `spel validate` run with a full
  case and a subset case of the same routine (e.g. `urb_full: urbanfluxes` and
  `urb_sub[urban]: urbanfluxes`), then check that `subset_dataset` of the full files at the
  `spel_subset_gindex` gridcells is bit-for-bit the subset files, and that `[all]` reproduces the
  full capture. On the 21-gridcell case, urban = gridcells 3, 5, 7 and lakes = 1-9 (from
  `PCT_URBAN`/`PCT_LAKE` in its surface dataset). That case has no crops: the `crop` preset
  (`lun_pp%itype(l) == istcrop`) still needs a test on a crop configuration (Perlmutter).
- To instrument E3SM master without touching uncommitted work in it, use a scratch `git worktree`
  of its HEAD, rsync the submodule trees and the untracked `mylaptop_gnu.cmake` into it, and pass
  `--srcroot` (its analysis takes about 45 s). Remove the worktree and its `.spel-cache` entry after.

## Don'ts

- Don't modify uncommitted work in the E3SM checkouts 
  unless the user asks; if they do, back the file up first.
- Don't name scratch Python files after stdlib modules (e.g. `/tmp/copy.py`).
