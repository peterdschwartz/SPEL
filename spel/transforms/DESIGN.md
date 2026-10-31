# A common transform API (`spel.transforms.core`)

Status (2026-11-01): phases 0-3 are built; the large passes are registered as adapters (§7). The
survey (§1) describes the code before the work; §7 lists what exists now, what changed on purpose
and what is left.

Goal: one small, readable API for source-to-source transforms, so that a user can read what a
transform matches, what it changes and when it refuses, adjust it (usually by editing a table, not
code), and write a new one in ~50 lines on top of shared infrastructure.

## 1. Survey

About 16k lines in ~35 modules, four families. Every family re-solved the same five problems
(read source, find the code, build new lines, apply edits, write/report) in its own way.

| Family | Modules | Matching | Edit model | Plan / dry-run / manifest |
|---|---|---|---|---|
| OpenACC | `acc_directives`, `acc_plan`, `acc_apply`, `acc_locals`, `acc_sync`, `acc_to_omp` | record + `find_directives`; `DO_RE` in acc_to_omp | `Insertion` (acc_apply:656, acc_locals:283), `(ln, order, lines)` tuples (acc_sync), `Edit(start,end,old,new)` (acc_to_omp:134) | `PortPlan`/`SyncPlan`/`FileTranslation`; dry-run; no manifest |
| Device | `device/generate`, `device/targeted` | analysis + record; `_locate` anchors | generated modules + `Edit` (generate:1193) + tuples (targeted) | `Plan`; dry-run; no manifest |
| Compact / loops | `compact/*`, `blocked_loops`, `collapse_bounds`, `section_loops`, `defer_checks`, `local_forms` | record + `IndexContext`; regex fallback select.py:500-522, rewrite.py:285/418-427/666 | `FileEdits` (rewrite:190), direct `lines[i] = ...` (blocked_loops, collapse_bounds, section_loops), tuples (defer_checks) | compact: dry-run/diff/replace + `.spel-compact.json`; others: dry-run only |
| Nutrient / split | `nutrient/*`, `split/modules` | record + `tree.Routine`; regex declaration/use/type scans in extract (16 regex calls), species (11) | whole-file arrays, insert/drop maps, a second `FileEdits` (merge:203) | extract: `.spel-nutrient.json`; others none |
| Legacy | `refactor_tools`, `compress_filter`, `apply_filter_compress` | regex grammar (compress_filter:50-60 and statement regexes after it; refactor_tools) | direct writes | none; `refactor_tools` is reached only from the ad-hoc `apply_filter_compress`; nothing imports `compress_filter` |

What already works and should be kept:

- `Subroutine.record` (loops, accesses, assignments, calls, associations; 0-based lines),
  `record_access`, `IndexContext`, `registrations.parse_routine(keep_directives=...)`,
  `acc_directives.find_directives`. Planning on these is solid.
- Plan/apply separation in the newer transforms (`PortPlan`, `SyncPlan`, `Merger.build`, `Splitter.build`).
- Content-checked edits (acc_to_omp's `Edit` keeps `old` and detects shifted targets), idempotence
  tags (acc_sync `dedupe_sync_blocks`, defer_checks `PREFIX`), manifests that refuse re-application
  (compact), `--diff`.

What is duplicated (each line is one concept, implemented N times):

| Concept | Implementations |
|---|---|
| indentation of a line | `_indent` acc_apply:662, acc_locals:289, acc_sync:646; `re.match(r"\s*")` ×8 in compact/rewrite, acc_plan:547 |
| wrap a directive / list over continuation lines | `format_directive` acc_to_omp:156, `_wrap_directive` acc_locals:293, `_directive` acc_sync:627, `_wrap`/`_wrap_cond` device/generate:545/392, `wrap` split/modules:179 |
| logical statement at a line (continuations) | `statement_at` nutrient/tree:87, `statement_start/end` compact/select:394, `logical_statements` compress_filter:104, `_continuation_end` device/targeted:121, `_end_of` acc_sync:386 |
| code/comment split | `split_code` nutrient/tree:55, `strip_comment` acc_directives:189 and scripts/instrument_elm:72, `_code` compact/select:390 |
| first executable line | `first_executable` acc_locals:305, `first_executable_line` nutrient/species:180 |
| `use` statement editing | nutrient/species `add_imports`, split/modules `client_lines`, scripts/edit_files |
| edit list + apply | 6 classes (above) + ad-hoc tuples in 4 modules |
| write files / dry-run / diff | ~16 modules call `write_text`/`writelines` themselves |
| CLI boilerplate | each handler in `cli.py` repeats load_analysis → build → report → diff → write |

Hidden coupling: private names imported across modules (`acc_sync._spans/_Types/_end_of`,
`acc_apply._Facts/_path`, `acc_locals._above_directives`, `compact.select._code`,
`procedures._split_code`). These are de-facto shared APIs without a home.

Configuration lives in module constants: `split.PLANS`, `nutrient.groups.GROUPS/FIELD_NAMES`,
`history.HISTORY/RESTART`, `merge.DEFAULT_BINDINGS`, `species.NAMES/SPECIES`, `device.GROUPS`,
`acc_sync.OFF_BY_DEFAULT`, `compact.inventory.FLUX_TYPES`, `acc_apply` abort/host-only names. These
are the parts users most want to adjust, and the hardest to find.

Parser gap: AST nodes carry only `lineno` (+ `end_ln` for constructs). There are no column spans,
so in-line substitutions (rename a designator, rewrite a subscript) fall back to regexes over the
line text. That is the root cause of most remaining regex use in compact/rewrite and nutrient.

## 2. Design

Five layers, each usable on its own. A transform author normally touches only layers 3-4.
(The layout below is the proposal; §7 lists what was built: sinks live in `driver.py`, file
capture for the adapters in `capture.py`, `testing.py` is not written yet.)

```
spel/transforms/core/
  source.py    SourceFile: lines, logical statements, indentation, comments   (layer 1)
  edits.py     Edit, EditSet: content-checked, conflict-checked patches       (layer 1)
  fortran.py   emitters: directive(), use_stmt(), decl(), do_loop(), wrap()   (layer 2)
  query.py     Matches over the analysis: routines, loops, assigns, calls,   (layer 3)
               directives, decls, uses, all carrying a Span
  transform.py Transform protocol, Plan/Change/Refusal, Context               (layer 4)
  sink.py      where output goes: in place, out_dir, SourceMods, case       (layer 5)
  driver.py    run(): select → plan → report/diff → apply → verify → manifest
  registry.py  @transform registration, CLI generation, `spel transform ...`
  testing.py   check_transform(before, after) snippet harness
```

### Layer 1: source and edits

As built (`core/edits.py`, `core/source.py`): an edit is line-based, which is what every pass
needed; column spans (§4) are still to come.

```python
@dataclass(frozen=True)
class Edit:
    ln: int                     # 0-based first line
    n: int                      # lines replaced; 0 = insertion before `ln`
    new: tuple[str, ...]
    order: float = 0            # insertions at one line: lower order ends up above
    expect: tuple[str, ...] | None = None   # original text, checked at apply time
    why: str = ""

insert(ln, new, order=0), replace(ln, new, n=1, expect=None), delete(ln, n=1)
stacked([(ln, lines, order), ...])   # the older passes' splice order (later one above at a tie)
changed_lines(old, new)              # line-for-line replacements between two same-length texts

class EditSet:                       # one per file
    def add(*edits)                  # EditConflict on overlapping replacements (identical ones merge)
    def apply(lines) -> list[str]    # bottom-up; EditMismatch when `expect` doesn't match
    def diff(lines) -> str

class SourceFile / Sources           # lines read once; indent, code, statement(ln) -> Span
split_code, code, comment, indent, statement_start/end/text, matching_paren, split_top_level
```

Every existing edit form maps onto `Edit`: `Insertion(ln, lines, order)` = `insert` (n = 0); acc_sync
tuples = same; `FileEdits.before/after/subs` = insert/insert/replace; `lines[i] = x` = replace one
line. Conflicts become errors instead of silent order-dependence. Two transforms (or two passes of
one) can share an `EditSet` and be applied together.

### Layer 2: Fortran emitters

One implementation each of: `directive(sentinel, head, clauses, indent, width, cond=None)`,
`use_stmt(module, names)`, `add_use(src, scope, module, names) -> Edit` (merges into an existing
`use`), `decl(type, attrs, names)`, `do_loop(index, lo, hi, body)`, `if_block(cond, body)`,
`continuation_wrap(head, items, width)`, `module(name, uses, spec, contains)`. Expressions are
printed from AST nodes (`str(expr)` today; promote to an explicit `to_fortran()` with a round-trip
test). This replaces the five wrap helpers and the hand-built module text in device/generate,
compact/unit_test, nutrient/history `HELPERS` and split.

### Layer 3: queries

A thin, readable facade over what the analysis already has; no new analysis. Every result is a
small object with `.span`, `.routine`, and the underlying record object (`.info`, `.node`, `.event`)
for anything the facade does not expose.

```python
q = Query(ctx)                        # ctx holds fut, AccessMapper, sources (lazy, cached)
for r in q.routines(module="SoilTemperatureMod", has_acc=True):
    for loop in r.loops(outermost=True, over_filter="filter_nolakec"):
        loop.span, loop.index, loop.bounds, loop.perfect_nest(2), loop.directive
    r.assignments(lhs="col_es%t_soisno"), r.calls(name="endrun"), r.locals(rank=1, runtime_sized=True)
    r.directives(kind="parallel loop"), r.first_executable, r.uses, r.declarations
```

Promote the privately imported helpers here: `acc_sync._spans` (device regions), `_end_of`,
`acc_apply.abort_calls/call_tree/_Facts`, `acc_locals.first_executable/_above_directives`,
`compact.select` statement helpers, `nutrient.tree.read_routine`. Directive data comes from
`find_directives` (stays the single OpenACC parser; generalise its sentinel to `!$omp` for acc2omp
checks).

### Layer 4: the transform protocol

```python
class Transform(Protocol):
    name: str
    summary: str                     # one line, shown by `spel transform list`
    Options: type                    # dataclass; fields become CLI flags and spec keys
    spec: Spec | None                # declarative table (below), overridable from a TOML file

    def select(self, ctx, opts) -> Iterable[Match]         # what it looks at
    def plan(self, ctx, opts, m: Match) -> Change | Refusal # what it would do there
    def generate(self, ctx, opts, plan) -> dict[Path, str]  # optional: new files (modules)
```

`Change(match, edits: list[Edit], notes)`, `Refusal(match, reason)`; `Plan` is the collection,
with `report()` (text/JSON), `diff()`, and `explain(match)` (before/after snippet + `why`). The
driver owns everything else, so individual transforms stop doing it:

1. load analysis (`--srcroot`, `--reanalyze`), build `Context`;
2. `select` → `plan` per match; `--only ROUTINE/MODULE` filters matches;
3. report / `--diff` / `--dry-run` stop here;
4. apply all edits through `EditSet` (conflicts → error), write through a `Sink`;
5. verify: re-parse every touched routine with `parse_routine` (+ scope walk when cheap); optional
   idempotence check (plan again on the result → no changes);
6. write a manifest `.spel-transforms.json` in the sink root: transform, options, spec hash, source
   commit, files, edit tags. Replaces `.spel-compact.json`/`.spel-nutrient.json` (kept readable
   as legacy) and drives refuse-on-reapply and `--replace`.

Sinks: `InPlace(backup=True)`, `OutDir(path)` (SourceMods/src.elm), `Case(case)` (writes into
`unit-tests/<case>/`, keeps `.spel-orig/` so `spel restore` works unchanged).

Composition: `Pipeline([SectionLoops(), DeferChecks(), AccLocals(), AccSync()])` plans in sequence on
a virtual source (edits of step n are visible to step n+1 via an in-memory overlay), which is what
the device pipeline and compact rewrite (which embeds blocked_loops) do by hand today.

### Declarative specs

Tables that encode policy move out of code into typed specs with a documented TOML form, shipped as
defaults and overridable with `--spec FILE` (same pattern as the machine files):

```toml
# spel transform nutrient-groups --spec my_groups.toml
[groups.mineral]
types  = ["column_nitrogen_state", "column_phosphorus_state"]
fields = ["sminn_vr", "smin_no3_vr", "solutionp_vr"]
```

Candidates: `split.PLANS`, `nutrient.groups.GROUPS/FIELD_NAMES`, `history.HISTORY/RESTART`,
`merge.DEFAULT_BINDINGS`, `species.NAMES/SPECIES`, `device.GROUPS`, `acc_sync.OFF_BY_DEFAULT`,
`compact.inventory.FLUX_TYPES`, acc_apply abort/host-only names. Rules that are logic (index forms,
race checks) stay Python.

### CLI

`@transform` registers a class; the registry builds `spel transform <name>` with the shared flags
(`--srcroot --reanalyze --only --dry-run --diff --report {text,json} -o/--case --spec --replace`) plus
the `Options` fields. `spel transform list` and `spel transform explain <name> [--only X]` make
transforms discoverable. Existing commands (`acc2omp`, `compact rewrite`, `nutrient *`, `split`,
`device-data`, `accplan`) stay as aliases with unchanged flags.

### Readability contract

Each transform module starts with a fixed docstring: **Matches**, **Changes**, **Refuses**,
**Spec**, **Example** (a before/after snippet that is also its first test). `explain` prints the
same for a concrete match.

### What a small transform looks like

`collapse_bounds` (hoist inquiry bounds out of collapsed nests) today: 184 lines including its own
regex for `do` headers, writes and dry-run. In the new API:

```python
@transform
class CollapseBounds:
    """Matches: perfectly nested loops under `collapse(n)` whose bounds call size/ubound/lbound.
    Changes: hoists each such bound into a local scalar assigned before the nest.
    Refuses: bounds that read variables written inside the nest."""
    name, summary = "collapse-bounds", "hoist inquiry-function loop bounds out of collapsed nests"

    def select(self, ctx, opts):
        for r in ctx.query.routines(has_acc=True):
            for loop in r.loops(outermost=True):
                if (d := loop.directive) and d.collapse > 1 and loop.inquiry_bounds(d.collapse):
                    yield loop

    def plan(self, ctx, opts, loop):
        if loop.bounds_written_in_nest():
            return Refusal(loop, "bound depends on the nest")
        hoisted = {b: ctx.fresh_name(loop.routine, f"spel_n{i}") for i, b in enumerate(loop.inquiry_bounds())}
        return Change(loop, [
            add_decl(loop.routine, "integer", hoisted.values()),
            *(insert_before(loop.directive.span, f"{n} = {b}") for b, n in hoisted.items()),
            *(replace_expr(b.span, n) for b, n in hoisted.items()),
        ])
```

## 3. Testing

- `core/testing.py::check_transform(T, before, after, opts=...)`: builds a context from a snippet
  (reusing `walk()` / `StubSub`), runs the driver in memory, compares text, re-parses the result and
  checks idempotence. Each transform's docstring example becomes its first test.
- Regression gate for the migration: `scripts/transform_golden.sh` runs every CLI transform on a
  fixed E3SM commit into an out-dir, before and after each port; outputs must be byte-identical
  (differences only where a port deliberately fixes a bug, listed in the commit).
- Layer-1/2 helpers get their own tests (continuations with `&` comments, directives inside
  continuations, strings containing `!`, cpp lines), replacing the per-module variants.

## 4. Parser extension: column spans (opt-in)

To remove the remaining regex substitutions (compact/rewrite designator/subscript patterns at
418-427/666, nutrient extract `rename_in_lines`, species renames, compact/select fallback
500-522), tokens and expression nodes need `(line, col_start, col_end)` in physical-line terms.
Make it opt-in like `keep_directives` (`keep_spans=True`), carry spans through
`LogicalLineIterator` joins (map logical offset → physical line/col), and expose
`Access.span`/`AssignEvent.span` in the record. Then `replace_expr(span, text)` is an exact edit.
Snippet tests in `fortran_parser/tests/`; pickled analysis must be refreshed (`--reanalyze`,
delete `.pytest-fixtures/elm_drv_analysis.pkl`).

## 5. Migration (each phase ships independently, golden outputs unchanged)

| Phase | Work | Size |
|---|---|---|
| 0 | Delete or move to `legacy/`: `refactor_tools`, `apply_filter_compress`, `compress_filter` (regex grammar, no CLI, no tests, no importers of `compress_filter`); confirm with the user first | S |
| 1 | `core/source.py`, `edits.py`, `fortran.py` + tests; replace the duplicated helpers (table in §1) by imports; no behaviour change | M |
| 2 | `transform.py`, `driver.py`, `sink.py`, `registry.py`, `testing.py`; port the small, self-contained ones: `collapse_bounds`, `blocked_loops`, `section_loops`, `defer_checks`, `acc_locals`; `local_forms`/`acc_plan` as report-only transforms | M |
| 3 | `query.py`; promote the private cross-imports; port `acc_apply`, `acc_sync`, `acc_to_omp`, `device/targeted`, then `device/generate` (codegen through emitters) | L |
| 4 | Specs (TOML + dataclasses); port `split`, `nutrient/*` (merge `FileEdits` → `EditSet`, manifests unified) | L |
| 5 | Parser column spans (§4); port `compact/select` + `compact/rewrite`, removing their regexes; compact `--roles`, `e3sm_casegen.sh --compact` unchanged | L |
| 6 | Docs: README "Transforms" section generated from the registry (`spel transform list --markdown`); AGENTS.md pointer | S |

Order rationale: phases 1-2 give the most deduplication for the least risk and prove the API on
small transforms; the acc/device family (phase 3) benefits most from the query layer; compact is
last because it needs the parser extension and has the largest test surface.

## 6. Open questions

- Spec format: TOML (consistent with machine files) vs. Python dataclass modules. Proposal: TOML
  for user overrides, dataclasses as the typed schema and defaults.
    - Decision:  Go with the proposal.
- Keep legacy command names permanently, or deprecate after phase 6? 
  - Decision: Deprecate
- Should `Pipeline` overlays plan on edited text by re-parsing (exact, slower) or by mapping line
  numbers through pending edits (fast, needs every transform to use `Span`s)? Proposal: re-parse the
  touched routines only.
  - Decision: go with proposal
- Phase 0 deletions need the user's OK (`compress_filter` may still be wanted as a reference).

## 7. Status (2026-11-01)

Built:

- `core/`: `source.py`, `edits.py`, `fortran.py` (directive/use/declaration/do/if emitters),
  `query.py` (the formerly private cross-module helpers: `call_tree`, `abort_calls`,
  `device_spans`, `construct_end`, `first_executable`, `above_directives`, `source_path`,
  `outside_routine`, ...), `transform.py` (`Transform`, `Plan`/`Change`/`Refusal`, `Context`,
  `describe()`), `driver.py` (`run`/`apply_plan`, sinks `InPlace`/`OutDir`/`Memory`, `--verify`
  re-parse, the `.spel-transforms.json` manifest), `registry.py` (`@transform`, flags from the
  Options dataclass, TOML specs, `spel transform ...`), `capture.py` (`VirtualFS`).
- Ported onto core edits (their old functions are thin wrappers, so external scripts keep
  working): `collapse-bounds`, `unblock` (blocked_loops), `section-loops`, `defer-checks`,
  `acc-locals`, `acc-apply`; `acc-sync` and `acc2omp` give whole-file outputs (acc-sync
  deduplicates its earlier runs, acc2omp maps onto a shifted unit-test copy).
- Adapters (`adapters.py`): `compact-rewrite`, `device-data`, `device-targeted`,
  `nutrient-{extract,species,merge,groups,history,restart}`, `split`. Each runs the existing pass
  on the context's analysis with its writes captured and returns the files as the plan's outputs,
  so it gets the shared `--dry-run/--diff/-o/--verify/--spec`. The old commands are unchanged.
- Deleted (phase 0): `refactor_tools`, `apply_filter_compress`, `compress_filter`.
- Golden harness: `python -m spel.transforms.tools.golden OUT --srcroot SRC [--only PASS ...]`
  (`--chain nutrient` replays the nutrient refactor). Every port was checked byte-identical
  against the pre-port code on a frozen E3SM commit, master, and master plus the removal of the
  unused nutrient types, except for the fixes below.

Deliberate behaviour changes:

- compact `rewrite_initialize` finds `elm_initializemod::initialize2` outside the analysis
  (`query.outside_routine`); before it could miss it.
- collapse-bounds rewrites a nest whole or not at all (a partial header edit could leak).
- collapse-bounds/unblock log their routines in id order.
- acc_plan `_aliases`: an index alias built from several loop indices (`l =
  landunit_indices(ltype, t)`, `c = fc + j`) depends on all of them (`Alias.also`), and
  arithmetic over several is not injective. Before, one index was picked in set order, so the
  plan depended on `PYTHONHASHSEED` (dynLandunitAreaMod's `update_landunit_weights` was ported or
  not). It now stays on the host as a scatter.

Left (in order):

1. Replace the remaining private `_indent`/`_directive` copies in acc_sync, acc_to_omp,
   device/generate and split by `core.source`/`core.fortran`.
2. Policy tables as TOML specs (split `PLANS`, nutrient `GROUPS`/`HISTORY`, device `GROUPS`,
   acc_sync `OFF_BY_DEFAULT`); `load_spec` already reads step lists and options.
3. Port the adapters onto core edits one at a time (device-targeted and split first; compact
   last), each checked with the golden harness.
4. Parser column spans (§4), then drop the regex substitutions in compact/nutrient.
5. `core/testing.py` snippet harness and a `Pipeline` overlay (today `--spec` steps reanalyze
   between steps).
