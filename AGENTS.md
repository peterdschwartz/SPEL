# AGENTS.md

Notes for AI agents working on SPEL. Read `README.md` for the CLI and the
reference-vs-development workflow; this file only covers what the README doesn't.

## Environment

Machine specifics (toolchains = modules/`PKG_CONFIG_PATH` per compiler, scratch, cprnc, casegen
defaults) are in machine files: `spel/machines/{perlmutter,chrysalis}.toml` (bundled, detected
automatically; toolchains take their modules from the E3SM checkout's `config_machines.xml`) or
`~/.config/spel/machines/<name>.toml` (`spel machine init`). `spel machine` shows what applies,
`spel machine check` verifies the toolchains, `spel run`/`spel profile`/`spel validate` build
inside the toolchain of `SPEL_FC`, and `profile_compare.sh` and `e3sm_casegen.sh --preset` read
the same file. `SPEL_MACHINE=none` falls back to the current shell environment. When a machine's
setup changes, change its machine file. Batch (Slurm) jobs need the user's OK.

Pitfalls worth knowing:

- Perlmutter login nodes are shared and have A100 GPUs: build and run unit tests on an
  interactive CPU node. The nvfortran toolchain sets `ACC_DEVICE_TYPE=multicore`; otherwise the
  OpenACC runtime may try the login node's GPU and loop forever printing `Failing in Thread:1`.
  The Cray netcdf-fortran `.pc` needs `$CRAY_MPICH_DIR/lib/pkgconfig` on `PKG_CONFIG_PATH`.
- Chrysalis' gfortran toolchain (E3SM's chrysalis_gnu modules) has a pkg-config that omits `-I`
  (the directory is on `CPATH`); tests add `fmoddir` themselves.
- `cprnc` compares restart files only with `-m`.
- Performance comparisons need builds of the same kind: a threaded (`FORCE_BUILD_SMP`) and a
  non-threaded Intel build of identical code differ by several percent on untouched routines.
- CIME: don't `xmlchange` a case while its `case.build` runs ("env_run.xml appears to have changed
  without a corresponding invalidation"; recover with `case.setup --reset`); the case directory's
  git can race with the job right after `case.submit` (`index.lock` exists): resubmit. A case
  started from a restart without the matching `--start-date` silently runs from the default date.

## Generated vs. hand-edited files

- `SourceFiles/nc_io.F90` and `SourceFiles/spel_varMod.F90` are **generated**. Edit
  `spel/scripts/io/netcdf_io.py`, then regenerate with `python -m spel.scripts.io.netcdf_io`.
- Allocation goes through descriptors: `spel_varMod::spel_var_allocate(var, d)` allocates from
  `d`'s bounds (`spel_var_desc`); the case's read routines fill `d` from the capture with
  `nc_allocMod::nc_desc` (attributes, `nsets` scaling), the proptest shim from Python.
  `spel_varMod` is shared with instrumented ELM (`instrument_elm.SHARED_IO_MODULES`): Fortran 2008
  only, no netCDF. `elm_init` = read + `elm_setup` (sets, clumps, filters), which the shim calls alone.
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

Compile-check Fortran templates with both compilers, each against its own netcdf-fortran
(`.mod` files are compiler specific; `spel machine env --fc FC` sets up a toolchain):

```bash
gfortran -c -ffree-line-length-none -I$NETCDF_FORTRAN_GNU/include SourceFiles/nc_io.F90 SourceFiles/nc_allocMod.F90
nvfortran -c -I$NETCDF_FORTRAN_NV/include SourceFiles/nc_io.F90 SourceFiles/nc_allocMod.F90
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
  case multithreaded with gfortran, translate it with `spel transform acc2omp --case <case>` and build with
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
- In-line edits (rewrite a subscript, add an argument, replace an expression) take their columns
  from the parser: `parse_routine(..., keep_spans=True)` (or `Parser(keep_spans=True)`) gives every
  token and expression node a `span` `(ln0, c0, ln1, c1)` into the file's physical lines (0-based,
  `c1` exclusive; continuations, comment lines and `;` handled), `SubroutineDefinitionConstruct
  .args_close` the `)` of the dummy list. Edit with `core.edits.replace_span`/`replace_cols`
  (`TextEdit`), not a regex over the line (`compact/rewrite.py::rewrite_site` is the example). Spans
  are off by default and not in the pickled record: re-parse the routine where you need them.
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
  the parser, AST or walker (`spel/scripts/fortran_parser/`) also need `--reanalyze`. So do changes
  to `spel/scripts/edit_files.py`: the edited ELM sources `spel create` copies into a case are cached
  too (`<cache>/elm_drv/src/`), so a generator fix shows up only after `--reanalyze`.
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

## Source transforms (`spel transform`, `spel/transforms/core`)

- New transforms subclass `core.transform.Transform` (`name`, `summary`, an `Options` dataclass,
  `build(ctx, opts) -> Plan`), register with `@transform` and get listed in
  `core/registry.py::BUILTINS`. Build edits with `core.edits` (`insert`/`replace`/`stacked`), find
  code with `core.query` and the record, emit Fortran with `core.fortran`; never write files in a
  transform (the driver does). Status and the
  remaining phases are in `spel/transforms/DESIGN.md` §7.
- Every source rewrite is a `spel transform NAME` (`spel transform list`). The former commands
  (`spel acc2omp`, `compact rewrite`, `device-data`, `split`, `nutrient extract|species|merge|groups|
  history|restart`) are removed and only print their replacement (`cli.py::REMOVED`); don't add new
  rewrite commands outside the registry. `spel compact inventory`, `spel nutrient common|procedures`
  and `spel accplan` are reports and stay commands. `spel accplan MODULES --lint` judges the loops that already carry a
  directive (level-outer gang loops with a filter loop inside, coarser-level arrays written from a filter loop).
- Filter-to-filter maps: `core/filter_maps.py` knows clumpfilter's `soilc_soilp_beg/end` (the soil patches
  of soil column fc are `soilp(beg(fc):end(fc))`), which filter instance holds which patches, and finds the
  loops over a filtered column's patches (`pfti(c)..pftf(c)`, `pfti(c)+pi-1`) with an exactness verdict
  (`ctx.child_loops` in a transform). `spel transform filter-map` rewrites the exact ones and threads the map
  (and `filter_soilp` where missing) through the analyzed callers as keyword actuals; callers outside the
  analysis (elm_interface_funcsMod) need the same by hand.
- Transforms that replace hand edits of an OpenACC port (bit-for-bit on RD, ECA and FATES
  regression runs): `scalarize-locals` (local_forms' per-iteration
  arrays -> scalars), `filter-locals` (clump-sized work arrays stored on their filter, across calls as
  groups: filter_locals_ip.py), `drop-dummies` (unused `bounds` dummies and their actuals),
  `acc-routines` (`!$acc routine seq` exactly on the routines kernels reach), `acc-declares`
  (`acc declare create` for module data device code uses; scalars in kernels are firstprivate),
  `acc-strip` (remove directives; `--sync-only` = acc-sync's updates). Dropping clump sections of
  type components (`x(bounds%begc:bounds%endc)` -> `x`) is NOT safe: components span the process
  bounds, a clump is a slice of them.
- `spel gates [--config PRESET|lnd_in] [--allow TEXT...]`: collapse-bounds, acc-declares and
  acc-routines have nothing left to do, acc-sync's syncs are all present (config-aware), no stale
  pushes; exit 1 on a failure. Replaces spelval/tools sync_drift/stale_updates/collapse_inquiry_scan/
  missing_declares and scope.py.
- The record can miss uses the file has: the analysis walks copies whose namelist-inactive
  branches are commented out (`!#py`, e.g. DUSTMod's `dust_emis_scheme == 1` block), and I/O
  statements are blanked. A transform that changes every use of a name (filter-locals) must check
  the span-parsed original routine (`parse_routine(..., keep_spans=True, skip_io=True, cpp=True)`)
  and refuse when it holds uses the plan did not cover; an uncovered use segfaulted all three
  regression runs on 2026-11-03.
- The driver keeps each file's line endings (CRLF stays CRLF: `driver.newline_of`); transforms see
  `\n`-split lines.
- A refused transform writes nothing and still exits 0 (exit 1 only on `--verify` problems, 2 on a
  pipeline stop); scripts must check for the output they expect (`e3sm_casegen.sh --compact` checks
  for `.spel-compact.json`).
- Test a transform on a snippet with `core/testing.py` (`check_transform(T, before, after,
  tmp_path)`, `check_pipeline` for several steps): it analyzes one file (undeclared names become
  module globals) and checks the result parses and that a second run changes nothing.
- A multi-step `--spec` runs as `core/pipeline.py::Pipeline` (each step plans on the previous one's
  text, the touched routines re-parsed in memory, written once). Anything that reads sources in a
  transform must go through `ctx.lines`, `Path.read_text`/`open` or `apply_preprocessor` so the
  pipeline's overlay (`VirtualFS`, `edit_files.SOURCE_OVERRIDES`) serves it the edited text; a new
  parse cache must key on that text, as `module_spec._key` does.
- Refactoring a transform: compare golden outputs before and after, on a fixed srcroot, with
  `python -m spel.transforms.tools.golden OUT --srcroot SRC --only PASS ...` (old code from a
  `git worktree` of HEAD) and `diff -r`; `--chain nutrient` replays the nutrient refactor (through `spel transform`). Plans must
  not depend on set order: run a pass under two `PYTHONHASHSEED`s if outputs look flaky.
- The Django settings default to `DJANGO_DEBUG=True` (a dev secret), so plain `pytest` works; a
  production `.env` with `DJANGO_DEBUG=false` needs `DJANGO_SECRET_KEY`.

## Compact layout (`spel compact`)

- The compact rewrite is not carried in E3SM sources: the layout is applied per
  case at build time, `e3sm_casegen.sh --compact` (= `spel transform compact-rewrite --srcroot SRC --reanalyze
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
- `compact-rewrite` needs the analysis of exactly the sources it edits (`spel analyze` first) and refuses
  sources that already have `CompactBindMod.F90` or `.spel-compact.json`. Test it with
  `-o $CASE/SourceMods/src.elm` against a baseline case built from the same commit (bit-for-bit
  history files).
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
- `spel transform acc2omp --dry-run` doubles as a race check for OpenACC code: `added vs OpenACC` lists
  scalars the analysis privatizes but the OpenACC clauses don't.
- `private` arrays need compile-time bounds: nvfortran `ACC=MULTICORE` corrupts the heap
  (`free(): invalid pointer`, even on one thread) with automatic ones such as `x(-nlevsno+1:0)`.
  Size them with a parameter (`elm_varpar::nlevsno_max`) and assign explicit sections
  (`x(-nlevsno+1:0) = ...`). Private elements a thread never sets are NaN on the GPU but stale
  finite values on the CPU, so a GPU-only NaN often means an uninitialized private or a local that
  is never copied to the device.

## Property-based tests (`spel proptest`, `spel/proptest`)

User documentation: README (`spel proptest`) and `spel/proptest/SPEC.md` (the spec format, files, limits).
Keep SPEC.md in step with the code: it is the only reference of the format.

- Layout:
  - `inventory.py`: variables of a case, from `fut.pkl`;
  - `shim.py`: generates `SpelPyMod.F90`, a bind(C) shim, plus `spel_py_manifest.json` with
    `core.fortran` emitters. The routine call's actuals come from `ut.driver_bindings`;
  - `build.py`: builds `<case>/build-py/libspelcase.so` (CMake `SPEL_PYLIB=ON` from
    `generate_cmake`); other builds (`Variant`: fc, acc, dbg) go to `build-py-<fc>-<acc>[-dbg]`;
  - `hierarchy.py`: ELM's subgrid hierarchy (declarations, `Layout`); `domains.py`: drawn domains
    and their shrinking;
  - `domain.py`: the synthetic domain under `<case>/proptest/domain-<hash>`: the constants file,
    and either an inputs file (`[domain] init = "files"`) or, by default for cases made with
    spel_varMod, `Domain.direct` (bounds + non-fill values), which the worker allocates through the
    shim's descriptors (`spel_py_set_bounds`/`spel_py_allocate`/`spel_py_setup`, runtime.Case.init).
    Both give bit-identical state and outputs (checked on canflux 2026-11-10; the snow/lake filters
    `setFilters` doesn't build hold uninitialized memory either way). Saved failures are written
    as captures (`Session.capture_dir` writes the files domain once when needed);
  - `runtime.py`/`worker.py`: ctypes, in a spawned subprocess;
  - `fexpr.py`: Fortran expressions through `spel_parser`, evaluated with numpy;
  - `spec.py`: loads the TOML; `rules.py` (rule kinds), `checks.py` (check kinds) and
    `functions.py` have a registry each (`@register`), and the spec reader and runner know only
    the interfaces. Add a kind there, not as a branch in `spec.py`/`runner.py`;
  - `strategies.py`, `runner.py`: Hypothesis; `Session` runs the routine through a backend
    (`get`/`run`/`stop`, `begin`/`step`: a `Worker`, or a test's fake);
  - `coupling.py`, `timestep.py`: multi-step runs (the boundary, the calendar);
  - `api.py` (Python builder of specs), `tomlw.py` (TOML writer), `results.py` (inspection);
  - `scaffold.py` (`init`), `stats.py`, `subspace.py`, `coverage.py`: design aids; `paths.py`: where files go.

  No module recognises Fortran by text: names, dims, guards and expressions all go through the
  analysis and the parser (user's rule for this feature).
- **Nothing is defaulted without a word.** An input without a rule, a parameter the routines read
  that the reference lacks (`Domain.filled`), a constant they read that nobody gave, a misspelt
  `[domain.time]`/`[domain.dims]` key, an unknown topology field: each stops the load or the run
  with the fix in the message. elm_drv's scalar arguments (`fut.driver_args`) are scalar inputs
  that need a rule. Keep it that way when adding inputs of another sort.
- **Namelist options.** `Inventory.accesses` keeps every traced access with its *gates*: the
  namelist conditions of the branches around it and around the calls leading to it
  (`transforms/core/config.py::gates`, from `flat_ifs`; the same branch logic as
  `Configuration.dead_branches`). `Inventory.configured(values)` classifies from the accesses
  whose gates are not decided false (`evaluate_guard`; undecided = live). A `Session` has one
  configuration (its inventory, its domain's constants file); a scenario with `namelist`, and the
  follow-up of a relation with `namelist`, run in `Session.sibling(...)`, a second worker on the
  same library. Rules of variables dead in a configuration are dormant there (`Session.live_inputs`).
  Liveness of names is checked per scenario at load (`spec._check_configurations`,
  `Check.check_scenario`). `Inventory.load` alone counts every branch as running.
  Non-namelist switches (`soil_stress_method`) can be set in `[namelist]` but prune nothing.
- Files: scratch in `<case>/proptest/` (domains, logs of aborted workers, units cache); what must
  survive `spel create` (which deletes the case directory's contents) in
  `unit-tests/proptest-data/<case>/` (Hypothesis database, `failures/`, `stats/`, `subspace/`).
- A library is stale when a case source or `fut.pkl` is newer than `build-py*/.spel-built`, or the
  generated shim differs; `Session` then rebuilds (incremental). The stamp also records the
  netcdf-fortran directory of the build, which the worker puts first in `LD_LIBRARY_PATH`, so a
  gfortran and an nvfortran build of one case run in the same session (differential checks).
  Without a machine file, build each variant in a shell set up for its compiler.
- Input/output classification uses `db_export.trace.StaticTrace` over the case routines
  (`inventory.trace_status`), with the root env seeded from `driver_bindings`, and
  `fieldstate.categorize`: prognostic → inout, input → input, diagnostic/intermediate → output.
  The access manifest's first access is only the fallback. A first-access-only classification
  once made `t_veg` (conditional first write, read later) a NaN-poisoned output: every output was
  NaN.
- Expressions: whether a value is per element is decided from the expression (`Evaluator.elemental`:
  it uses the element variable), never from a shape; `FArray.take` puts the element axis first and
  `Evaluator.aligned` lines operands up on it. A level count can equal the element count.
  In a check a bare inout name is the value after the run; `input%x` is the value before.
- Scripts that create a `runner.Session`/worker need an `if __name__ == "__main__":` guard
  (multiprocessing spawn).
- One Hypothesis example draws every element: per input a mode (one constant, iid from a drawn
  seed, edges, or per element with one value down the layers), so shrinking works on a few
  integers/floats, not on arrays. Adding a mode or a draw changes which examples a seed gives.
  With `--seed`, Hypothesis does not use its example database.
- Check and scenario names may contain `[...]`. Selection (`runner._selected`) tries the exact
  name before the fnmatch glob.
- CanopyFluxes on master needs `soil_stress_method = 0` in `[namelist]` (the reference
  constants file lacks it, and its declaration has no default). `firrig` is copied from the reference (known limitation). The derived
  `rootfr` must sum to 1 over `nlevgrnd`: 15 × 0.0666667 gave `btran` = 1.0000005.
- `coverage` uses the gcov of the gfortran in use (`gcov-<major>`; the plain `gcov` here is GCC 13's
  and cannot read GCC 14 files) and takes counts as before/after differences, so nothing in the
  build directory is deleted.
- Domains are ELM's hierarchy (`hierarchy.py`: `Gridcell` > `Topounit` > `Landunit` > `Column` >
  `Patch`, weights in the parent; constructors for every landunit kind ELM builds, following
  initGridCellsMod: `Soil`, `Polygon`, `Crop`, `Urban`, `Lake`, `Wetland`, `Glacier`, `GlacierMec`;
  `glcmask`/`icemask` per gridcell; `[[domain.gridcell]]` in TOML). Activity is subgridWeightsMod's
  is_active_l/c/p (`all_active` off). `Layout` lays a set out in ELM's clump order (`order = "elm"`,
  the default since 2026-11-12): landunits in one pass per type over all topounits, so a gridcell's
  or topounit's landunits, columns and patches are NOT contiguous (`"depth"` keeps them so). Take
  element counts from `layout.count(level)`, ancestors from `parent`/`ancestor_of`, descendants
  from `descendants` (index arrays; `first_child`/`last_child` are ranges only for l -> c -> p),
  relation halves from `Session.half_count(level)`; `row_permutation` exchanges whole gridcells of
  one declaration. `[domain] subgrid` adds ELM's implicit elements (the soil landunit on every
  topounit, every pft/CFT, every glacier class in the glcmask). Topology values
  (`domain.topology_value`) all come from the layout: flags (`Layout.flag`), `topo_grc_ind`,
  `downhill_ti` (set_topounit's algorithm, quirk included), the rank-2 `landunit_indices`
  (polygons under polygontype + 9); ELM leaves the topounit/gridcell landunit ranges at ispval and
  never sets `col_pp%hydrologically_active` (always .false., though TotalWaterAndHeatMod reads it),
  and so does the domain. `max_topounits`/`has_topounit` are `DOMAIN_CONSTANTS`; declared
  topounit elevations pin `top_pp%elevation` (`Domain.pinned`). ELM type codes are
  `hierarchy.TYPE_CODES`, checked against the case's parameters.
- Checked against ELM: `experiments/elm_layout_oracle.py RESTART` rebuilds every gridcell of a
  restart file (land1d/cols1d/pfts1d) as a declaration and compares types, weights, activity and
  order (every gridcell of a 95,000-gridcell run matched); `Session.built().check()`
  compares what a case built (topology, bounds, `domain.EXPECTED_FILTERS`) with the declaration.
  Urban geometry inputs need ELM's derivations (`scaffold.URBAN_DERIVE`), or about 30% of drawn
  landunits take logs of negative heights (NaN).
- Drawn domains (`domains.py`, `[domain.draw]`): `runner.run` runs every check on `count` domains
  drawn from the seed, one Session each (the shim cannot deallocate, so each domain is a new
  worker); Hypothesis draws only the inputs. Failures of one domain are shrunk together
  (`runner._shrink`: candidates from `shrink_candidates`, quick runs without input shrinking, then
  each check alone, then one full run per final domain) and saved to
  `failures/<scenario>__<check>__d<k>/` with `domain.toml`, which `replay` reads. Known weakness:
  a shrink step can change which bug fails (e.g. a weight-0 topounit made active).
- Python API (`api.py`): builds the TOML-shaped dict `spec.parse` reads, checking each piece with
  the reader's own code; `tomlw.py` writes it. Inspection: `results.py` (`BuiltDomain`, `Result`
  frames, `Verdict`), `Session.sample` (the test-run draw through `strategies.RngChooser`).
  `init` takes the landunit kinds from the routine's filter arguments
  (`scaffold.routine_landunits`, `domain.FILTER_KINDS`/`FILTER_COMPLEMENTS`); `stats --landunits`
  restricts the statistics. Kinds, urban densities and polygon types are enums (`LandunitKind`,
  `UrbanDensity`, `PolygonType`).
- Multi-step runs (SPEC.md "Multi-step runs"): worker ops `begin`/`step` (no reset between steps;
  `run` = one call, unchanged), `Session.run_steps` (cached per example; `compute` returns the
  state of the step being judged, `_step` is shared with siblings and `other`, so relations and
  differentials compare the same step), `timestep.advance` (ELM's no-leap calendar). The boundary
  (`coupling.py`: elm_drv's StaticTrace over the full analysis, writers of case variables outside
  the case's subtrees, gates and filters as `inventory.trace_accesses`/`db_export.filters`) must be
  declared per scenario (`forcing`/`between`/`keep`). Design choice (2026-11-11, with the user):
  proptest tests a process; what the rest of the step does is stated in the spec, not pulled in as
  more routines (that closes over elm_drv and becomes an end-to-end test). The shim calls the case's
  filter builders before the routines, as main.F90 does, so dynamic filters follow the state.
- Outputs are NaN before every step; `carry` (SPEC.md) exempts outputs written on a filter and read
  elsewhere (`qflx_from_uphill`: SurfaceRunoff on hydrologyc, ColWaterBalanceCheck on every column), so
  a `between` sees ELM's stale value, not a NaN ELM never has. `child_wsum` skips inactive/weight-0
  children as subgridAveMod does. Fortran `MAX(0, NaN)` is 0 with gfortran and NaN with ifx
  fexpr's `max` propagates NaN (ifx-like), which matters when a state can hold a NaN.
- Tutorial: `spel/proptest/examples/canopyfluxes_tutorial.ipynb` (executed, with outputs). Change it through
  its narrative and re-execute it (`nbclient`) after API changes; its text quotes actual results.
- Tests (`spel/proptest/tests/`): `test_proptest.py` (snippets; the CanopyFluxes end-to-end test is
  skipped unless the `canflux` case exists and `spel proptest build canflux` has been run),
  `test_runner_fake.py` and `test_multistep.py` (the runner on `fake.py`: a synthetic inventory and a
  numpy "routine", no build), `test_domain.py` (hierarchy, layout, topology, filters),
  `test_domains.py` (drawing, shrinking, the drawn-domain run on the fake), `test_api.py`,
  `test_runtime.py` (ctypes and the worker on a hand-written gfortran library).

## Comparing E3SM timings

Use `./elm_timing_compare.py CASE_A CASE_B --labels a b --top 30 [--format markdown]` (repo root; takes a case,
run or timing directory, or a saved stats file). Don't write a new parser. For a repeated comparison, copy
`RUNDIR/timing/model_timing_stats` after each run (the next run replaces it) and compare the saved files. Timers
are inclusive and `max` is the slowest process. Repeat on an otherwise idle node before quoting a timer (a build
sharing the node invalidates it), alternate the two cases, and name the configuration, including gridcells per
task. Timers a change doesn't touch are the control: they should come out near 1.00x.

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
- MPI: every rank with matches writes its share of `--subset-max` (proportional to its matches,
  at least one; one clump per rank) to `<tag>.rankNNNNN.spel-*.nc`, and the master rank lists the
  writing ranks in `<tag>.spel-subset-ranks.txt`. `subset.merge_rank_files` (run by `spel validate`,
  `spel instrument --run-case`, and `spel merge-subset RUNDIR CASE`) reads exactly the listed ranks,
  orders gridcells by global index, cuts to the max evenly, concatenates every subgrid dimension and
  renumbers the index fields per rank. Variables without a subgrid dimension must be bit-identical
  across ranks (else it refuses; `--allow-differ NAME` keeps the lowest writing rank's). `spel_subset_rank` records
  each gridcell's rank. Tested in `test_subset.py` (merge of a partition = the whole capture; the
  generated `spel_select_subset` compiled against stub ELM modules).
- Provenance: global attributes `spel_subset_mask/gindex/latdeg/londeg` in every file, `subset` in
  `spel_reference.json` (printed by `spel run`), and `SPEL subset` lines in the run's `e3sm.log`.

Testing:

- `spel/scripts/tests/test_subset.py`: mask parsing, capture-module generation, and a gfortran
  program writing a synthetic capture (non-unit process bounds) through `nc_io`, compared
  bit-for-bit with the offline subsetter (`subset_dataset`, `diff_datasets`).
- End to end: one `spel validate` run with a full
  case and a subset case of the same routine (e.g. `urb_full: urbanfluxes` and
  `urb_sub[urban]: urbanfluxes`), then check that `subset_dataset` of the full files at the
  `spel_subset_gindex` gridcells is bit-for-bit the subset files, and that `[all]` reproduces the
  full capture. Pick the expected gridcells from the surface dataset (`PCT_URBAN`, `PCT_LAKE`, ...).
- To instrument E3SM master without touching uncommitted work in it, use a scratch `git worktree`
  of its HEAD, rsync the submodule trees and any untracked machine files into it, and pass
  `--srcroot` (its analysis takes about 45 s). Remove the worktree and its `.spel-cache` entry after.

## Don'ts

- Don't modify uncommitted work in the E3SM checkouts 
  unless the user asks; if they do, back the file up first.
- Don't name scratch Python files after stdlib modules (e.g. `/tmp/copy.py`).
