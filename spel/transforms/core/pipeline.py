"""
Pipelines: several transforms planned in sequence on a virtual source, written once.

    Pipeline([(SectionLoops, opts1), (AccApply, opts2), (AccLocals, opts3)]).run(fut, srcroot, sink)

Step n+1 plans on the text step n produced, without writing it and without reanalyzing the
srcroot: the edited files are served from memory to everything that reads sources (Sources,
Path.read_text, open, the C preprocessor; `capture.VirtualFS`), and the analysis is refreshed
for the touched files only (`AnalysisOverlay`): their routines' line numbers are mapped onto the
new text, and the analyzed ones are re-parsed and re-walked (record, record_access,
registrations). Everything else keeps the cached analysis.

What the overlay can't represent stops the pipeline with a PipelineError before a later step
plans on it (run those steps one at a time with a reanalysis in between, `--reanalyze-steps`):
a step that edits the code of a module head (declarations, `use`s; comments and directives
such as `!$acc declare` are fine), creates a Fortran file the analysis doesn't know, moves or
rewrites a routine's header or END, or leaves a routine that no longer parses or walks.
"""

from __future__ import annotations

import copy
import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from spel.transforms.core.capture import VirtualFS
from spel.transforms.core.driver import Result, Sink, apply_plan, verify_parse, write_manifest, InPlace
from spel.transforms.core.source import Sources, statement_end
from spel.transforms.core.transform import Context, Transform

log = logging.getLogger("spel.transforms.pipeline")


class PipelineError(RuntimeError):
    """A step's output the analysis overlay can't represent for the steps after it."""


# --- line maps ---------------------------------------------------------------------------------


@dataclass
class LineMap:
    """Old -> new line numbers of an edited file (0-based)."""

    fwd: dict[int, int]  # lines kept or replaced one for one
    same: set[int]  # new lines whose text is unchanged
    near_: dict[int, int] = field(default_factory=dict)  # every old line -> its new position

    def strict(self, ln: Optional[int]) -> Optional[int]:
        return None if ln is None else self.fwd.get(ln)

    def near(self, ln: int) -> int:
        """The new line of `ln`, or of the start of the block that replaced it."""
        return self.near_.get(ln, ln)


def line_map(old: list[str], new: list[str]) -> LineMap:
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    fwd: dict[int, int] = {}
    same: set[int] = set()
    near: dict[int, int] = {}
    for tag, i0, i1, j0, j1 in sm.get_opcodes():
        one_for_one = tag == "equal" or (tag == "replace" and i1 - i0 == j1 - j0)
        for k in range(i0, i1):
            near[k] = j0 + (k - i0 if one_for_one else 0)
            if one_for_one:
                fwd[k] = j0 + k - i0
                if tag == "equal":
                    same.add(j0 + k - i0)
    return LineMap(fwd, same, near)


# --- logical lines (what the analysis parses) --------------------------------------------------


def preprocessed_lines(path: Path, n_lines: int):
    """The file's logical lines as the analysis builds them (cpp, continuations joined, lower
    case), from the text the active VirtualFS serves."""
    from spel.scripts.edit_files import apply_comments, apply_preprocessor, remove_cpp_directives
    from spel.scripts.utilityFunctions import unwrap_section

    work = apply_comments(remove_cpp_directives(apply_preprocessor(path), str(path), log))
    parsed = [""] * max(n_lines, max((lt.ln for lt in work), default=-1) + 1)
    for lt in work:
        parsed[lt.ln] = lt.line
    return unwrap_section(parsed, -1)  # startln is exclusive


def merge_logical(old_ll, new_ll, old: list[str], new: list[str], m: LineMap):
    """Logical lines of the new text: the analysis' own lines (which may hold its edits, such as
    `!#py` comments and shimmed `use`s) where the statement is unchanged, `new_ll` elsewhere."""
    from spel.scripts.types import LineTuple

    inv = {n: o for o, n in m.fwd.items()}
    old_by_ln: dict[int, list] = {}
    for lt in old_ll:
        old_by_ln.setdefault(lt.ln, []).append(lt)
    out = []
    for lt in new_ll:
        n0 = lt.ln
        n1 = statement_end(new, n0) if n0 < len(new) else n0
        o0 = inv.get(n0)
        kept = (o0 is not None and all(k in m.same and inv.get(k) == o0 + k - n0 for k in range(n0, n1 + 1))
                and statement_end(old, o0) == o0 + n1 - n0)
        if not kept:
            out.append(lt)
            continue
        for o in range(o0, o0 + n1 - n0 + 1):
            out += [LineTuple(line=x.line, ln=x.ln - o0 + n0) for x in old_by_ln.get(o, ())]
    return out


# --- the analysis overlay ----------------------------------------------------------------------


class OverlayAnalysis:
    """The cached analysis with some routines replaced (`subroutine_dict`)."""

    def __init__(self, base, subs: dict):
        self._base = base
        self.subroutine_dict = subs

    def __getattr__(self, name):
        return getattr(self._base, name)


class AnalysisOverlay:
    """Keeps the analysis of a srcroot in step with edited, unwritten text (see module doc)."""

    def __init__(self, fut, srcroot: Path):
        self.base = fut
        self.srcroot = Path(srcroot)
        self.subs = dict(fut.subroutine_dict)
        self.fut = OverlayAnalysis(fut, self.subs)
        self._logical: dict[Path, list] = {}
        self._head: dict[str, int] = {}  # module -> its `contains` (end of head) line in the current text
        self.refreshed: list[str] = []

    def path(self, obj) -> Path:
        p = Path(obj.filepath)
        return p if p.is_absolute() else self.srcroot / p

    def _modules_of(self, path: Path) -> list:
        return [m for m in self.base.module_dict.values()
                if getattr(m, "filepath", None) and self.path(m) == path]

    def update(self, old_texts: dict[Path, str], new_texts: dict[Path, str], step: str) -> None:
        """Refresh the analysis of the files whose text went from `old_texts` to `new_texts`.
        Call with the new texts served (VirtualFS active)."""
        from spel.scripts.module_resolver import ModuleScopes
        from spel.scripts.record_access import AccessMapper
        from spel.scripts.registrations import find_registrations

        scopes = ModuleScopes(self.base.module_dict, self.subs)
        walked = []
        for path, new in sorted(new_texts.items()):
            old = old_texts.get(path)
            if old == new:
                continue
            if Path(path).suffix.lower() not in (".f90", ".f"):
                continue
            mods = self._modules_of(Path(path))
            if old is None or not mods:
                raise PipelineError(f"{step} writes {path}, which the analysis doesn't cover")
            walked += self._update_file(Path(path), old, new, mods, scopes, step)

        mapper = AccessMapper(self.subs, scopes, failures={})
        for c in walked:
            if mapper.maps(c) is None:
                raise PipelineError(f"after {step}: no access maps for {c.id}: "
                                    f"{mapper.failures.get(c.id, 'unknown callee cycle')}")
        find_registrations({c.id: c for c in walked}, log)
        self.refreshed += [c.id for c in walked]

    def _update_file(self, path: Path, old: str, new: str, mods, scopes, step: str) -> list:
        from spel.scripts.fortran_parser.scope_walk import walk_subroutine
        from spel.scripts.module_resolver import ModuleResolver

        old_l, new_l = old.split("\n"), new.split("\n")
        m = line_map(old_l, new_l)
        for mod in mods:
            # comments (directives) may change in the head; declarations and uses may not
            head = self._head.get(mod.name, getattr(mod, "end_of_head_ln", 0) or 0)
            new_head = m.strict(head)
            if new_head is None or _code(old_l[: head + 1]) != _code(new_l[: new_head + 1]):
                raise PipelineError(f"{step} edits the head of module {mod.name} ({path.name}); "
                                    "later steps would see the old module analysis")
            self._head[mod.name] = new_head
        old_ll = self._logical.get(path)
        if old_ll is None:
            old_ll = mods[0].get_mod_lines()
        new_ll = merge_logical(old_ll, preprocessed_lines(path, len(new_l)), old_l, new_l, m)
        self._logical[path] = new_ll

        copies = {}
        for s in self.subs.values():
            if getattr(s, "library", False) or not getattr(s, "filepath", None) or self.path(s) != path:
                continue
            c = copy.copy(s)
            for attr in ("startline", "end_stmt_ln", "contains_ln", "endline"):
                ln = getattr(s, attr, None)
                if ln is None:
                    continue
                new_ln = m.strict(ln)
                if new_ln is None:
                    raise PipelineError(f"{step} rewrites the {attr} line of {s.id} ({path.name}:{ln + 1})")
                setattr(c, attr, new_ln)
            _remap_lines(c, m)
            c.internal_subs = {}
            copies[s.id] = c
        for c in copies.values():
            if getattr(c, "is_internal", False) and c.host_id:
                c.host = copies.get(c.host_id) or self.subs[c.host_id]
                c.host.internal_subs[c.name] = c
        self.subs.update(copies)

        walked = []
        for c in sorted(copies.values(), key=lambda c: _depth(c)):
            if c.record is None:  # not analyzed: lines mapped, nothing to walk
                continue
            full = [lt for lt in new_ll if c.startline <= lt.ln <= c.end_stmt_ln]
            c.sub_lines = [lt for lt in full if lt.ln <= c.endline]
            c.syntax_tree, c.record, c.record_access = None, None, None
            family, x = [], c
            while x is not None:
                family.append(self.base.subroutine_dict.get(x.id, x))
                x = x.host if getattr(x, "is_internal", False) else None
            # `use`s of unavailable modules are recovered from the module's (unchanged) use
            # statements by their lines in the analyzed text
            ranges = [(f.startline, f.endline) for f in family]
            try:
                c.parse(full)
                resolver = ModuleResolver(c.module, scopes.mod_dict, scopes.sub_dict, line_ranges=ranges,
                                          scopes=scopes)
                host = c.host.record if getattr(c, "is_internal", False) and c.host is not None else None
                c.record = walk_subroutine(c.syntax_tree, resolver=resolver, host=host)
            except BaseException as err:  # noqa: BLE001 (the parser exits on errors)
                raise PipelineError(f"after {step}: {c.id} no longer parses or walks: "
                                    f"{type(err).__name__}: {err}") from None
            walked.append(c)
        return walked


def _code(lines: list[str]) -> list[str]:
    """The non-comment, non-blank lines (directives are comments to the analysis)."""
    return [l.rstrip() for l in lines if l.strip() and not l.lstrip().startswith("!")]


def _depth(sub) -> int:
    d = 0
    while getattr(sub, "is_internal", False) and getattr(sub, "host", None) is not None:
        sub, d = sub.host, d + 1
    return d


def _remap_lines(c, m: LineMap) -> None:
    """Line-keyed analysis data other than the record (which is rebuilt) onto the new text."""
    for attr in ("associate_start", "associate_end", "last_decl_ln"):
        ln = getattr(c, attr, -1)
        if ln is not None and ln >= 0:
            setattr(c, attr, m.near(ln))
    if getattr(c, "flat_ifs", None):
        ifs = []
        for x in c.flat_ifs:
            y = copy.copy(x)
            y.start_ln, y.end_ln = m.near(x.start_ln), m.near(x.end_ln)
            ifs.append(y)
        c.flat_ifs = ifs
    if getattr(c, "sub_call_desc", None):
        c.sub_call_desc = {m.near(ln): d for ln, d in c.sub_call_desc.items()}
    if getattr(c, "elmtype_access_by_ln", None):
        c.elmtype_access_by_ln = {
            k: [copy.copy(rw) for rw in v] for k, v in c.elmtype_access_by_ln.items()}
        for v in c.elmtype_access_by_ln.values():
            for rw in v:
                if getattr(rw, "ln", None) is not None and rw.ln >= 0:
                    rw.ln = m.near(rw.ln)


# --- the pipeline ------------------------------------------------------------------------------


@dataclass
class StepResult:
    transform: Transform
    opts: object
    result: Result
    ctx: Context  # its sources are the text the step planned on (diffs, explain)


@dataclass
class PipelineResult:
    steps: list[StepResult] = field(default_factory=list)
    outputs: dict[Path, str] = field(default_factory=dict)  # net new text of every file changed
    written: list[Path] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


class Pipeline:
    """Transforms planned in sequence; step n+1 sees step n's output (see the module doc)."""

    def __init__(self, steps: Sequence):
        self.steps = []
        for st in steps:
            t, opts = st if isinstance(st, tuple) else (st, None)
            t = t() if isinstance(t, type) else t
            self.steps.append((t, opts if opts is not None else t.Options()))

    def run(self, fut, srcroot: Optional[Path] = None, sink: Optional[Sink] = None, dry_run: bool = False,
            verify: bool = False, manifest: bool = False,
            on_step: Optional[Callable[[int, StepResult], None]] = None,
            overlay: Optional[Callable] = None, config=None) -> PipelineResult:
        """Plan every step, then (unless `dry_run`) write the net result through `sink` once.
        `overlay(fut, srcroot)` makes the analysis overlay (default AnalysisOverlay; the snippet
        harness passes its own)."""
        if srcroot is None:
            from spel.scripts.config import E3SM_SRCROOT

            srcroot = Path(E3SM_SRCROOT)
        overlay = (overlay or AnalysisOverlay)(fut, srcroot)
        texts: dict[Path, str] = {}  # current text of every file a step changed
        original: dict[Path, Optional[str]] = {}
        out = PipelineResult()
        per_step: list[set[Path]] = []
        for i, (t, opts) in enumerate(self.steps):
            sources = Sources()
            for p, text in texts.items():
                sources.put(p, text)
            vfs = VirtualFS(texts, capture=False)
            with vfs.active():
                ctx = Context(overlay.fut, srcroot, sources, config.fresh() if config is not None else None)
                res = apply_plan(t.build(ctx, opts), ctx, dry_run=True)
            changed = {Path(p): s for p, s in res.outputs.items() if texts.get(Path(p), _disk(p)) != s}
            for p in changed:
                if p not in original:
                    original[p] = _disk(p)
            old = {p: texts.get(p, original[p]) for p in changed}
            texts.update(changed)
            per_step.append(set(changed))
            step = StepResult(t, opts, res, ctx)
            out.steps.append(step)
            if on_step:
                on_step(i, step)
            if changed and i < len(self.steps) - 1:
                with VirtualFS(texts, capture=False).active():
                    overlay.update(old, changed, t.name)
        out.outputs = {p: s for p, s in texts.items() if s != original.get(p)}
        if verify:
            out.problems = verify_parse(out.outputs)
        if not dry_run and out.outputs:
            sink = sink or InPlace()
            out.written = sink.write(out.outputs)
            if manifest:
                for (t, opts), step, files in zip(self.steps, out.steps, per_step):
                    mine = [sink.target(p) for p in sorted(files) if p in out.outputs]
                    if mine:
                        write_manifest(sink, t, opts, step.result.plan, mine, Path(srcroot),
                                       config.source if config is not None else None)
        return out


def _disk(path) -> Optional[str]:
    p = Path(path)
    return p.read_text() if p.exists() else None
