"""
Translate a module's OpenACC directives to OpenMP, so that a unit test can
run multithreaded with gfortran (which has no OpenACC multicore target).

The data-sharing clauses are derived from SPEL's analysis rather than copied
from the OpenACC clauses:

* the loop a directive applies to, and its `collapse` nest, come from the
  record's loop nest (SubroutineRecord.loop_nest);
* every variable accessed in the loop is classified from the access maps
  (Subroutine.record_access), where actual arguments already carry the
  callee's dummy intent:
    - indices of the parallel (collapsed) loops: predetermined private
    - indices of inner sequential loops, and local scalars written in the
      loop: private (firstprivate if the loop reads them before writing them)
    - variables of an OpenACC reduction clause: same reduction
    - arrays: shared, unless the OpenACC directive privatizes them
    - written dummy or global scalars: reported as races
* callees in the loop are checked for writes to SAVEd locals or global scalars.

The result is compared with the OpenACC private list; differences are
reported (a variable the analysis privatizes but OpenACC does not is a likely
race in the OpenACC version too).

Directive mapping:
    parallel/kernels loop   -> !$omp parallel do [collapse] [private] [firstprivate] [reduction] [if]
    serial loop             -> removed (the loop runs serially)
    loop (orphaned or inside a translated loop) -> removed (sequential)
    data, enter/exit data, update, declare, host_data, wait, routine, cache -> removed
    atomic [clause] / end atomic -> !$omp atomic [clause] / !$omp end atomic
    parallel/kernels/serial regions (non-loop) -> left untouched (run serially), reported

Line numbers are 0-based indices into the analyzed source (Subroutine.filepath),
as in the analysis; messages print 1-based file lines. The edits
can be applied to a copy of that file whose lines were shifted or commented
(a unit-test case directory): lines are matched by content.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

from spel.scripts.fortran_parser.scope_walk import (
    CallEvent,
    FunctionRef,
    LoopInfo,
    SubroutineRecord,
)
from spel.transforms.acc_directives import (
    DATA_CONSTRUCTS,
    AccDirective,
    find_directives,
    next_code_line,
)

if TYPE_CHECKING:
    from spel.scripts.record_access import AccessMapper, AccessMaps

DO_RE = re.compile(r"^\s*(\w+\s*:\s*)?do\b", re.IGNORECASE)
MAX_LINE = 100

# access-map attribute -> origin kind used in the classification
MAP_KINDS = (("locals", "local"), ("args", "dummy"), ("globals", "global"), ("host", "host"))


class TranslationError(Exception):
    pass


@dataclass
class Note:
    level: str  # "info" | "warn" | "error"
    routine: str
    ln: int
    msg: str

    def __str__(self) -> str:
        return f"{self.level.upper():5s} {self.routine} line {self.ln + 1}: {self.msg}"


@dataclass
class LoopPlan:
    """OpenMP clauses for one OpenACC compute loop."""

    routine: str
    directive: AccDirective
    nest: list[LoopInfo] = field(repr=False)
    private: list[str] = field(default_factory=list)
    firstprivate: list[str] = field(default_factory=list)
    reductions: list[tuple[str, list[str]]] = field(default_factory=list)
    if_clause: Optional[str] = None
    # privatized here but absent from the OpenACC private/firstprivate lists
    added: list[str] = field(default_factory=list)
    # in the OpenACC private list but needing no privatization
    unneeded: list[str] = field(default_factory=list)

    @property
    def collapse(self) -> int:
        return len(self.nest)

    @property
    def loop_ln(self) -> int:
        return self.nest[0].start_ln if self.nest else self.directive.end_ln + 1

    def clauses(self) -> list[str]:
        out = []
        if self.collapse > 1:
            out.append(f"collapse({self.collapse})")
        out.extend(_var_clauses("private", self.private))
        out.extend(_var_clauses("firstprivate", self.firstprivate))
        out.extend(f"reduction({op}:{', '.join(vs)})" for op, vs in self.reductions)
        if self.if_clause is not None:
            out.append(f"if({self.if_clause})")
        return out

    def omp_lines(self) -> list[str]:
        return format_directive(self.directive.indent, "parallel do", self.clauses())

    def summary(self) -> str:
        text = f"{self.routine} line {self.loop_ln + 1}: !$omp parallel do {' '.join(self.clauses())}"
        if self.added:
            text += f"  [added vs OpenACC: {', '.join(self.added)}]"
        if self.unneeded:
            text += f"  [not needed: {', '.join(self.unneeded)}]"
        return text


@dataclass
class Edit:
    """Replace lines start_ln..end_ln (0-based, inclusive) with `new`."""

    start_ln: int
    end_ln: int
    old: list[str]
    new: list[str]


def _var_clauses(name: str, names: list[str], width: int = 60) -> list[str]:
    """`name(a, b, ...)`, split into several clauses when the list is long."""
    out, cur = [], []
    for v in names:
        if cur and len(", ".join(cur + [v])) > width:
            out.append(f"{name}({', '.join(cur)})")
            cur = []
        cur.append(v)
    if cur:
        out.append(f"{name}({', '.join(cur)})")
    return out


def format_directive(indent: str, head: str, clauses: list[str]) -> list[str]:
    """`!$omp head clauses`, continued with `&` / `!$omp` when too long."""
    lines = [f"{indent}!$omp {head}"]
    for c in clauses:
        if len(lines[-1]) + 1 + len(c) <= MAX_LINE:
            lines[-1] += f" {c}"
        else:
            lines[-1] += " &"
            lines.append(f"{indent}!$omp   {c}")
    return lines


# ---------------------------------------------------------------------------
# analysis-driven classification of a compute loop
# ---------------------------------------------------------------------------


# OpenACC data clauses: a scalar listed there is deliberately shared
SHARED_CLAUSES = (
    "copy", "copyin", "copyout", "create", "present", "no_create", "deviceptr", "attach",
    "pcopy", "pcopyin", "pcopyout", "pcreate", "present_or_copy", "present_or_copyin",
    "present_or_copyout", "present_or_create",
)

Accesses = list[tuple[int, str, str]]  # (ln, status, path)


def _region_accesses(
    maps: AccessMaps, lo: int, hi: int
) -> tuple[dict[str, tuple[str, Accesses]], dict[str, Accesses]]:
    """
    Per base variable: (origin kind, [(ln, status, path)]) for lines lo..hi,
    and the accesses after hi (for locals, to detect values live after the loop).
    """
    inside: dict[str, tuple[str, Accesses]] = {}
    after: dict[str, Accesses] = {}
    for attr, kind in MAP_KINDS:
        for path, rws in getattr(maps, attr).items():
            base = path.split("%")[0]
            for rw in rws:
                if lo <= rw.ln <= hi:
                    entry = inside.setdefault(base, (kind, []))
                    entry[1].append((rw.ln, rw.status, path))
                elif rw.ln > hi and kind == "local":
                    after.setdefault(base, []).append((rw.ln, rw.status, path))
    for _, accesses in inside.values():
        accesses.sort()
    for accesses in after.values():
        accesses.sort()
    return inside, after


def _is_saved(rec: SubroutineRecord, name: str) -> bool:
    """A local with the SAVE attribute or an initializer (implicit SAVE)."""
    decl = rec.declaration(name)
    if decl is None or "parameter" in decl.attrs:
        return False
    if "save" in decl.attrs:
        return True
    node = decl.node
    for ent in getattr(node, "entities", []) or []:
        if ent.token.literal == name and ent.init is not None:
            return True
    return False


def callee_issues(
    sub,
    mapper: AccessMapper,
    seen: Optional[set[str]] = None,
) -> list[str]:
    """
    Reasons why `sub` may not be safe to call from several threads at once:
    SAVEd locals it writes, global scalars it (or its callees) writes.
    """
    seen = set() if seen is None else seen
    if sub.id in seen:
        return []
    seen.add(sub.id)
    rec: SubroutineRecord = sub.record
    maps = mapper.maps(sub)
    if rec is None or maps is None:
        return [f"{sub.id} could not be analyzed"]
    issues = []
    for path, rws in maps.locals.items():
        base = path.split("%")[0]
        if any("w" in rw.status for rw in rws) and _is_saved(rec, base):
            issues.append(f"{sub.id} writes SAVEd local '{base}'")
    for path, rws in (maps.globals | maps.child_globals).items():
        if any("w" in rw.status for rw in rws):
            issues.append(f"{sub.id} writes global '{path}'")
    for event in [*rec.calls, *rec.function_calls]:
        if isinstance(event, FunctionRef) and event.intrinsic:
            continue
        callee = mapper.callee(sub, rec, event)
        if callee is not None and not callee.library:
            issues.extend(callee_issues(callee, mapper, seen))
    return issues


@dataclass
class AtomicRegion:
    """The statement lines of an `!$acc atomic [kind]` construct."""

    lines: frozenset
    kind: str  # read | write | update | capture


def atomic_regions(directives: list[AccDirective], lines: list[str]) -> list[AtomicRegion]:
    """Atomic constructs: a block up to `end atomic`, or the next statement."""
    out = []
    for i, d in enumerate(directives):
        if d.construct != "atomic":
            continue
        kind = next(
            (c.name for c in d.clauses if c.name in {"read", "write", "update", "capture"}),
            "update",
        )
        nxt = directives[i + 1] if i + 1 < len(directives) else None
        if nxt is not None and nxt.construct == "end atomic":
            body = range(d.end_ln + 1, nxt.start_ln)
        else:
            stmt = next_code_line(lines, d.end_ln)
            body = [] if stmt is None else [stmt]
        out.append(AtomicRegion(frozenset(body), kind))
    return out


def _atomic_targets(maps: AccessMaps, atomics: list[AtomicRegion], lo: int, hi: int) -> set[str]:
    """
    Shared variables updated atomically in lines lo..hi: read and written by an
    update/capture construct (`x = x + 1`, `v = x`), or written by a write
    construct. The captured value (`v`, written only) is not a target.
    """
    targets: set[str] = set()
    for region in atomics:
        if not region.lines or not all(lo <= ln <= hi for ln in region.lines):
            continue
        status: dict[str, str] = {}
        for attr, _ in MAP_KINDS:
            for path, rws in getattr(maps, attr).items():
                base = path.split("%")[0]
                for rw in rws:
                    if rw.ln in region.lines:
                        status[base] = status.get(base, "") + rw.status
        for var, st in status.items():
            if ("r" in st and "w" in st) or (region.kind == "write" and "w" in st):
                targets.add(var)
    return targets


INTRINSIC_TYPES = {"integer", "real", "logical", "character", "complex", "double precision", "doubleprecision"}


def plan_compute_loop(
    sub,
    directive: AccDirective,
    loop: LoopInfo,
    mapper: AccessMapper,
    atomics: Optional[list[AtomicRegion]] = None,
) -> tuple[Optional[LoopPlan], list[Note]]:
    """
    OpenMP clauses for the OpenACC compute loop `directive` applied to `loop`
    of `sub` (a Subroutine with record and record_access). `atomics`: the
    file's atomic constructs. Returns (None, notes) if the loop cannot be
    translated.
    """
    rec: SubroutineRecord = sub.record
    maps = mapper.maps(sub)
    name = sub.name
    notes: list[Note] = []
    ln = loop.start_ln

    def note(level: str, msg: str, at: int = ln) -> None:
        notes.append(Note(level, name, at, msg))

    if maps is None:
        note("error", "no access maps for the routine")
        return None, notes

    try:
        nest = loop.perfect_nest(directive.collapse())
    except ValueError as err:
        note("error", f"collapse: {err}")
        return None, notes
    outer: set[str] = set()
    for inner in nest:
        if inner.index is None:
            note("error", f"loop on line {inner.start_ln + 1} has no index")
            return None, notes
        if dependent := inner.bound_paths & outer:
            note(
                "error",
                f"collapse: bounds of loop on line {inner.start_ln + 1} depend on {sorted(dependent)}",
            )
            return None, notes
        outer.add(inner.index)

    lo, hi = loop.start_ln, loop.end_ln
    par_idx = {l.index for l in nest}
    seq_idx = {l.index for l in loop.walk() if l not in nest and l.index}

    reductions = directive.reductions()
    red_vars = {v for _, vs in reductions for v in vs}
    acc_private = directive.var_list("private")
    acc_first = directive.var_list("firstprivate")
    acc_shared = {v for name in SHARED_CLAUSES for v in directive.var_list(name)}

    plan = LoopPlan(name, directive, nest, reductions=reductions)
    if (c := directive.clause("if")) is not None:
        plan.if_clause = c.args

    inside, after = _region_accesses(maps, lo, hi)
    call_lines = {c.ln for c in rec.calls}
    atomic = _atomic_targets(maps, atomics or [], lo, hi)
    for var, (kind, accesses) in sorted(inside.items(), key=lambda kv: kv[1][1][0]):
        if var in par_idx or var in red_vars:
            continue
        decl = rec.declaration(var)
        if decl is not None and "parameter" in decl.attrs:
            continue
        written = any("w" in s for _, s, _ in accesses)
        # writes of the whole variable, not only of derived-type components
        whole = any("w" in s and path == var for _, s, path in accesses)
        is_array = decl is not None and decl.dim > 0
        derived = decl is not None and decl.type.lower().split("(")[0].strip() not in INTRINSIC_TYPES
        if not written:
            if var in acc_private or var in acc_first:
                plan.unneeded.append(var)
            continue
        if var in atomic:
            continue  # shared, updated atomically
        if var in acc_shared and var not in acc_private and not is_array:
            note("warn", f"scalar '{var}' is written but shared (OpenACC data clause): race unless atomic")
            continue
        if kind == "local" and (not is_array or var in acc_private or var in acc_first):
            first_ln, first_status, _ = accesses[0]
            reads_first = "r" in first_status and var not in seq_idx
            if reads_first or var in acc_first:
                plan.firstprivate.append(var)
                if var not in acc_first:
                    note("warn", f"'{var}' is read before it is written in the loop: firstprivate", first_ln)
            else:
                plan.private.append(var)
            if var not in acc_private and var not in acc_first and var not in seq_idx:
                plan.added.append(var)
            later = after.get(var)
            if later and "r" in later[0][1]:
                note("warn", f"'{var}' is privatized but read after the loop (line {later[0][0] + 1})")
            if any(
                a.via is not None and a.base == var and lo <= a.ln <= hi
                for a in rec.accesses()
            ):
                note("warn", f"'{var}' is privatized but accessed through an associate name")
        elif kind == "local":
            continue  # array written per iteration: shared
        elif is_array and var not in acc_private:
            continue  # dummy array: shared
        elif var in acc_private:
            plan.private.append(var)
            note("warn", f"{kind} '{var}' is privatized (OpenACC private)")
        elif whole and not (derived and all(ln in call_lines for ln, st, path in accesses if "w" in st and path == var)):
            # (a derived-type object passed whole to a callee: the callee's component writes are
            # mapped as component paths; a whole-object status on a call line only repeats the
            # dummy's intent when the callee does not touch it)
            note("warn", f"{kind} scalar '{var}' is written in the parallel loop (race)")
    for var in acc_private + acc_first:
        if var not in inside and var not in plan.unneeded:
            plan.unneeded.append(var)

    for event in [*rec.calls, *rec.function_calls]:
        if not (lo <= event.ln <= hi):
            continue
        if isinstance(event, FunctionRef) and event.intrinsic:
            continue
        callee = mapper.callee(sub, rec, event)
        if callee is None or callee.library:
            note("warn", f"cannot check that '{event.name}' is thread safe", event.ln)
            continue
        for issue in callee_issues(callee, mapper):
            note("warn", f"call {event.name}: {issue}", event.ln)

    return plan, notes


def fallback_plan(routine: str, directive: AccDirective) -> LoopPlan:
    """Clauses copied from the OpenACC directive (routine not analyzed)."""
    plan = LoopPlan(routine, directive, nest=[], reductions=directive.reductions())
    plan.private = directive.var_list("private")
    plan.firstprivate = directive.var_list("firstprivate")
    if (c := directive.clause("if")) is not None:
        plan.if_clause = c.args
    return plan


# ---------------------------------------------------------------------------
# file translation
# ---------------------------------------------------------------------------


@dataclass
class FileTranslation:
    path: Path
    lines: list[str]  # the analyzed source (with line endings)
    edits: list[Edit] = field(default_factory=list)
    plans: list[LoopPlan] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)

    def apply(self, target: Optional[list[str]] = None) -> list[str]:
        """
        The edits applied to `target` (default: the analyzed source). `target`
        may be a copy with lines inserted or changed elsewhere: each edited
        line is located by matching the analyzed source against it.
        """
        if target is None:
            target = self.lines
        mapping = _line_map(self.lines, target)
        out = list(target)
        for edit in sorted(self.edits, key=lambda e: e.start_ln, reverse=True):
            idx = [mapping.get(ln) for ln in range(edit.start_ln, edit.end_ln + 1)]
            if None in idx or idx != list(range(idx[0], idx[0] + len(idx))):
                raise TranslationError(
                    f"{self.path.name}: lines {edit.start_ln + 1}-{edit.end_ln + 1} "
                    "not found unchanged in the target file"
                )
            out[idx[0] : idx[-1] + 1] = [line + "\n" for line in edit.new]
        return out

    def diff(self, target: Optional[list[str]] = None) -> str:
        before = self.lines if target is None else target
        return "".join(
            difflib.unified_diff(before, self.apply(target), str(self.path), str(self.path))
        )

    def report(self) -> str:
        out = [f"== {self.path}"]
        out.extend(f"  {p.summary()}" for p in self.plans)
        out.extend(f"  {n}" for n in self.notes)
        return "\n".join(out)


def _line_map(src: list[str], dst: list[str]) -> dict[int, int]:
    """0-based index in `src` -> index in `dst` for lines matched unchanged."""
    a = [l.rstrip() for l in src]
    b = [l.rstrip() for l in dst]
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    mapping = {}
    for tag, i1, i2, j1, _ in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
    return mapping


def _routine_at(subs: list, ln: int):
    """The innermost analyzed routine whose lines contain `ln`."""
    hits = [s for s in subs if s.startline <= ln <= s.end_stmt_ln]
    return max(hits, key=lambda s: s.startline) if hits else None


def translate_file(path: Path, subs: list, mapper: AccessMapper) -> FileTranslation:
    """
    Translates every OpenACC directive in `path`. `subs`: the analyzed
    routines of the file (Subroutine objects with record/record_access).
    """
    for sub in subs:
        if sub.record is not None and not hasattr(sub.record, "loop_nest"):
            raise TranslationError(
                f"{sub.id}: the cached analysis predates loop-nest information; "
                "rebuild it with `spel analyze` (or --reanalyze)"
            )
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    plain = [l.rstrip("\n") for l in lines]
    result = FileTranslation(path, lines)
    directives = find_directives(plain)
    atomics = atomic_regions(directives, plain)

    def old(d: AccDirective) -> list[str]:
        return plain[d.start_ln : d.end_ln + 1]

    def replace(d: AccDirective, new: list[str]) -> None:
        result.edits.append(Edit(d.start_ln, d.end_ln, old(d), new))

    def note(level: str, d: AccDirective, msg: str) -> None:
        sub = _routine_at(subs, d.start_ln)
        result.notes.append(Note(level, sub.name if sub else "<module>", d.start_ln, msg))

    # compute regions left as OpenACC (and the directives inside them)
    untouched: list[tuple[int, int]] = []
    open_regions: list[AccDirective] = []
    for d in directives:
        if d.is_compute_region:
            open_regions.append(d)
        elif d.construct.startswith("end ") and d.construct[4:] in {"parallel", "kernels", "serial"}:
            if not open_regions:
                raise TranslationError(f"{path.name} line {d.start_ln + 1}: unmatched !$acc {d.construct}")
            start = open_regions.pop()
            untouched.append((start.start_ln, d.end_ln))
            note("warn", start, f"!$acc {start.construct} region left as OpenACC (runs serially)")

    def in_untouched(ln: int) -> bool:
        return any(a <= ln <= b for a, b in untouched)

    for d in directives:
        if in_untouched(d.start_ln):
            continue
        sub = _routine_at(subs, d.start_ln)
        if d.is_compute_loop:
            if d.construct == "serial loop" or d.clause("seq") is not None:
                replace(d, [])  # the loop runs sequentially
                continue
            target = next_code_line(plain, d.end_ln)
            if target is None or not DO_RE.match(plain[target]):
                note("error", d, "no DO loop follows the directive; left as OpenACC")
                continue
            if sub is None or sub.record is None:
                plan = fallback_plan(sub.name if sub else "<module>", d)
                note("warn", d, "routine not analyzed: private/reduction copied from OpenACC")
                result.plans.append(plan)
                replace(d, plan.omp_lines())
                continue
            loop = sub.record.loop_at(target)
            if loop is None:
                raise TranslationError(
                    f"{sub.id}: no analyzed loop on line {target + 1} of {path.name}; "
                    "the source changed since the analysis (rebuild it)"
                )
            plan, notes = plan_compute_loop(sub, d, loop, mapper, atomics)
            result.notes.extend(notes)
            if plan is None:
                note("error", d, "left as OpenACC (runs serially)")
                continue
            result.plans.append(plan)
            replace(d, plan.omp_lines())
        elif d.construct in {"end parallel loop", "end kernels loop", "end serial loop"}:
            replace(d, [])
        elif d.construct == "loop":
            # inner loops of a translated loop and orphaned loops run sequentially
            replace(d, [])
        elif d.construct in DATA_CONSTRUCTS or d.construct == "routine":
            replace(d, [])
        elif d.construct == "atomic":
            replace(d, format_directive(d.indent, "atomic", [str(c) for c in d.clauses]))
        elif d.construct == "end atomic":
            replace(d, [f"{d.indent}!$omp end atomic"])
        elif not d.is_compute_region and not d.construct.startswith("end "):
            note("warn", d, f"!$acc {d.construct} left untouched")
    result.edits.sort(key=lambda e: e.start_ln)
    return result


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def module_files(fut, modules: Iterable[str]) -> dict[Path, list]:
    """Source file of each module -> its analyzed routines."""
    files: dict[Path, list] = {}
    for mod in modules:
        mod = mod.lower()
        fmod = fut.module_dict.get(mod)
        if fmod is None:
            raise TranslationError(f"module '{mod}' is not in the analysis")
        subs = [s for s in fut.subroutine_dict.values() if s.module == mod]
        files.setdefault(Path(fmod.filepath), []).extend(subs)
    return files


def translate_modules(fut, modules: Iterable[str]) -> list[FileTranslation]:
    from spel.scripts.record_access import AccessMapper

    mapper = AccessMapper(fut.subroutine_dict)
    return [translate_file(path, subs, mapper) for path, subs in module_files(fut, modules).items()]


def case_modules(fut, case_dir: Path) -> list[str]:
    """Analyzed modules of a unit-test case whose source has OpenACC compute loops."""
    names = {p.name for p in case_dir.glob("*.F90")}
    out = []
    for name, fmod in fut.module_dict.items():
        path = Path(fmod.filepath)
        if path.name in names and path.is_file():
            if any(d.is_compute_loop for d in find_directives(path.read_text().splitlines())):
                out.append(name)
    return sorted(out)
