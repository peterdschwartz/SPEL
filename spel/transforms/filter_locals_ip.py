"""
filter-locals across calls: work arrays that are passed to other routines.

ELM passes clump-sized work arrays down the call tree (`call SolveTemperature(..., tk(begc:endc,:),
...)` with the dummy `tk(bounds%begc:, -nlevsno+1:)`). Such an array can be stored on the filter
only together with the dummies it reaches. The unit is a *group*: locals and dummies whose first
dimension spans the clump, linked by the call bindings (`Subroutine.record_access.bindings`).
A group is converted entirely or not at all. Every member must be

- used at filter members inside the filter loop (`x(c)`, `c = F(fc)`; filter_locals._check), or
- passed whole or as the clump section (`x(begc:endc, ...)`) to a dummy that is itself a member;

and the filter must be the same along every call: the callee's filter and count dummies are bound
to the caller's (a pass-through member takes its filter from the calls it makes). Every call of a
member dummy's routine, in the analysis and in the ELM sources' text (other callers, generic
interfaces, type-bound procedures), must pass a member. Uses the analysis did not record refuse
the member (filter_locals.TextUses).

Edits: a local's or explicit-shape dummy's first dimension becomes the count (`x(N, ...)`), an
assumed-shape dummy's `bounds%begc:` becomes `:`, filter elements `x(fc, ...)`, clump-section
actuals `x(:, ...)`. Values and results are unchanged.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spel.scripts.fortran_parser.scope_walk import Access
from spel.scripts.fortran_parser.symbols import Origin
from spel.transforms.compact.index_forms import IndexContext, norm
from spel.transforms.core.edits import replace_cols
from spel.transforms.core.source import statement_start
from spel.transforms.filter_locals import DISQUALIFY, INTRINSIC_TYPES, LEVELS, TextUses

Key = tuple[str, str]  # (routine id, variable)


@dataclass
class Member:
    rid: str
    name: str
    dummy: bool
    decl_ln: int
    bound: object  # the first declared dimension (BoundsExpression, with spans)
    level: str  # c, p, l, g, t
    assumed: bool  # dummy `x(bounds%begc:)` (assumed shape)
    sites: dict[int, str] = field(default_factory=dict)  # statement line -> filter position (fc)
    filt: Optional[str] = None  # filter dummy of the routine
    count: Optional[str] = None  # its count dummy
    own: bool = False  # filter from its own element uses (else from its calls)
    calls: list = field(default_factory=list)  # (ln, callee id, dummy, "whole" | "section")
    why: str = ""


def _clump_dim(b, dummy: bool) -> Optional[tuple[str, bool]]:
    """(level, assumed shape) when the declared dimension `b` spans the clump."""
    if b is None or getattr(b, "start", None) is None:
        return None
    s = norm(b.start).lower()
    e = norm(b.end).lower() if getattr(b, "end", None) is not None else None
    for x in LEVELS:
        if e is None and dummy and s == f"bounds%beg{x}":
            return x, True
        if (s, e) in ((f"bounds%beg{x}", f"bounds%end{x}"), (f"beg{x}", f"end{x}")):
            return x, False
    return None


def _clump_section(sub_expr, level: str) -> bool:
    """`begc:endc` / `bounds%begc:bounds%endc` as an actual's first subscript."""
    if getattr(sub_expr, "start", None) is None or getattr(sub_expr, "end", None) is None:
        return False
    s, e = norm(sub_expr.start).lower(), norm(sub_expr.end).lower()
    return s in (f"beg{level}", f"bounds%beg{level}") and e in (f"end{level}", f"bounds%end{level}")


class Planner:
    def __init__(self, fut, subs: list, lines_of, tree_of, srcroot: Path):
        self.fut, self.lines_of, self.tree_of = fut, lines_of, tree_of
        self.subs = {s.id: s for s in subs}
        self.members: dict[Key, Member] = {}
        self.uses: dict[str, TextUses] = {}
        self.trees: dict[str, object] = {}
        self.refused: dict[str, str] = {}
        self.srcroot = srcroot
        self._text_calls: dict[str, int] = {}
        self.bindings: dict[tuple[str, int], dict[str, tuple[str, Origin]]] = defaultdict(dict)  # (caller, ln) -> callee dummy -> actual
        self.callee_of: dict[tuple[str, int, str], str] = {}  # (caller, ln, actual) -> callee
        self.incoming: dict[Key, set] = defaultdict(set)  # (callee, dummy) -> {(caller, ln, actual)}
        for s in fut.subroutine_dict.values():
            ra = getattr(s, "record_access", None)
            for b in ra.bindings if ra is not None else []:
                self.bindings[(b.caller, b.ln, b.callee)][b.dummy.lower()] = (b.actual.lower(), b.origin)
                self.callee_of[(b.caller, b.ln, b.actual.lower())] = b.callee
                self.incoming[(b.callee, b.dummy.lower())].add((b.caller, b.ln, b.actual.lower()))

    # --- members of one routine ---
    def collect(self, sub) -> None:
        from spel.scripts.fortran_parser.spel_ast import VariableDecl
        from spel.scripts.registrations import iter_statements

        tree = self.tree_of(sub)
        if tree is None:
            return
        if getattr(tree, "contains", None):
            self.refused[sub.id] = "has internal procedures"
            return
        self.trees[sub.id] = tree
        lines = self.lines_of(sub)
        self.uses[sub.id] = TextUses(sub, lines, tree)
        rec = sub.record
        ents = {}
        for stmt, _ in iter_statements(tree.body):
            if isinstance(stmt, VariableDecl):
                for ent in stmt.entities:
                    if ent.bounds:
                        ents[ent.token.literal.lower()] = ent
        accs: dict[str, list] = defaultdict(list)
        for pos, e in enumerate(rec.events):
            if isinstance(e, Access) and e.origin in (Origin.LOCAL, Origin.DUMMY):
                accs[e.base.lower()].append((pos, e))
        ictx = IndexContext(sub.id, rec, {}, {})
        loops = rec.all_loops()
        for name, ent in ents.items():
            d = rec.declaration(name)
            if d is None or d.dim < 1 or not d.type.lower().startswith(INTRINSIC_TYPES):
                continue
            dummy = d.intent is not None
            cd = _clump_dim(ent.bounds[0], dummy)
            if cd is None:
                continue
            m = Member(sub.id, name, dummy, d.ln, ent.bounds[0], cd[0], cd[1])
            self.members[(sub.id, name)] = m
            attrs = {a.lower() for a in d.attrs}
            if attrs & {"parameter", "pointer", "target", "allocatable", "save", "optional", "value"}:
                m.why = "/".join(sorted(attrs))
                continue
            self._uses(m, accs.get(name, []), ictx, loops, rec)

    def _uses(self, m: Member, accs, ictx, loops, rec) -> None:
        gathers = set()
        for pos, a in accs:
            if a.context == "init" or a.via is not None:
                m.why = f"{a.context}{' via ' + a.via if a.via else ''} at line {a.ln + 1}"
                return
            if a.context == "arg" or a.status == "arg":
                callee = self.callee_of.get((m.rid, a.ln, a.path.lower()))
                if callee is None:
                    m.why = f"line {a.ln + 1}: passed to a routine the analysis has no binding for"
                    return
                if not a.subscripts:
                    mode = "whole"
                elif _clump_section(a.subscripts[0], m.level):
                    mode = "section"
                else:
                    m.why = f"line {a.ln + 1}: passed as {a.path}({','.join(a.subs)})"
                    return
                dummy = next((dn for dn, (act, _) in self.bindings[(m.rid, a.ln, callee)].items()
                              if act == a.path.lower()), None)
                if dummy is None:
                    m.why = f"line {a.ln + 1}: no binding of {a.path}"
                    return
                m.calls.append((a.ln, callee, dummy, mode))
                continue
            if a.context in DISQUALIFY:
                m.why = f"{a.context} at line {a.ln + 1}"
                return
            f = ictx.access_form(a, pos)
            sub0 = a.subs[0] if a.subs else ""
            inner = f.inner
            if (f.kind != "filter" or f.var != sub0 or inner is None or inner.kind != "count"
                    or not inner.var or not inner.text):
                m.why = f"line {a.ln + 1}: subscript {sub0 or '(whole)'} is {f}"
                return
            if not any(lp.index == inner.var and lp.contains(a.ln) for lp in loops):
                m.why = f"line {a.ln + 1}: {sub0} used outside the loop over {inner.var}"
                return
            if m.sites.get(a.ln, inner.var) != inner.var:
                m.why = f"line {a.ln + 1}: two different filter positions on one line"
                return
            m.sites[a.ln] = inner.var
            gathers.add((f.filter, inner.text.lower()))
        if len(gathers) > 1:
            m.why = "gathers through several filters/counts"
            return
        if gathers:
            (filt, count), = gathers
            if len(filt) != 1 or not next(iter(filt)).startswith("dummy:"):
                m.why = f"filter {sorted(filt)} is not a dummy"
                return
            fname = next(iter(filt)).split(":", 1)[1].split("%")[0].lower()
            if any(a.base.lower() == fname and "w" in a.status for a in rec.accesses()):
                m.why = f"filter {fname} is written in the routine"
                return
            m.filt, m.count, m.own = fname, count, True
        if not m.sites and not m.calls:
            m.why = "never used"

    # --- the groups ---
    def resolve(self) -> None:
        changed = True
        while changed:
            changed = False
            for m in self.members.values():
                if m.why:
                    continue
                why = self._check(m)
                if why:
                    m.why, changed = why, True
                elif m.filt is None:
                    changed |= self._infer(m)
        for m in self.members.values():
            if not m.why and m.filt is None:
                m.why = "never used at a filter element (directly or below)"

    def _actuals(self, m: Member, ln: int, callee: str, t: Member) -> Optional[tuple[str, str]]:
        """The caller's actuals bound to the callee member's filter and count at this call."""
        b = self.bindings[(m.rid, ln, callee)]
        f, n = b.get(t.filt), b.get(t.count)
        if f is None or n is None or f[1] is not Origin.DUMMY or n[1] is not Origin.DUMMY:
            return None
        return f[0], n[0]

    def _infer(self, m: Member) -> bool:
        for ln, callee, dummy, _ in m.calls:
            t = self.members.get((callee, dummy))
            if t is not None and not t.why and t.filt is not None:
                fn = self._actuals(m, ln, callee, t)
                if fn is not None:
                    m.filt, m.count = fn
                    return True
        return False

    def _check(self, m: Member) -> str:
        rec = self.subs[m.rid].record
        for ln, callee, dummy, _ in m.calls:
            t = self.members.get((callee, dummy))
            if t is None:
                return f"line {ln + 1}: passed to {callee}'s {dummy}, which is not clump-sized or not analyzed"
            if t.why:
                return f"line {ln + 1}: passed to {callee}'s {dummy} ({t.why})"
            if t.filt is not None and m.filt is not None and self._actuals(m, ln, callee, t) != (m.filt, m.count):
                return f"line {ln + 1}: {callee} gets another filter than {m.filt}/{m.count}"
        if m.count is not None and (not m.dummy or not m.assumed):
            d = rec.declaration(m.count)
            if d is None or d.intent is None or d.dim != 0 or d.ln >= m.decl_ln:
                return f"count {m.count} is not a scalar dummy declared before {m.name}"
        if m.dummy:
            callers = self.incoming.get((m.rid, m.name), set())
            if not callers:
                return "no analyzed caller"
            for caller, ln, actual in callers:
                src = self.members.get((caller, actual))
                if src is None or src.why or not any(c[0] == ln and c[1] == m.rid and c[2] == m.name for c in src.calls):
                    return f"{caller} line {ln + 1} passes {actual}, not a member"
            ncalls = len({(c, ln) for c, ln, _ in callers})
            if self.text_calls(m.rid) != ncalls:
                return f"{m.rid.split('::')[-1]} is called {self.text_calls(m.rid)} times in the sources, {ncalls} analyzed"
        return ""

    def text_calls(self, rid: str) -> int:
        if rid not in self._text_calls:
            self._text_calls[rid] = text_call_count(self.srcroot, self.subs.values(), rid.split("::")[-1])
        return self._text_calls[rid]

    # --- edits ---
    def plan(self) -> tuple[dict[str, list], dict[str, list[str]], dict[str, list[str]]]:
        """Resolve the groups and plan the edits; a member refused while planning its edits
        refuses its group too (resolve again until nothing changes)."""
        while True:
            self.resolve()
            edits, done, why, refused = self.edits()
            if not refused:
                return edits, done, why

    def edits(self):
        """routine id -> edits, -> converted names, -> reasons (left alone); and whether a member
        was refused here."""
        from spel.transforms.compact.rewrite import subscripted_designators

        edits, done, why, refused = defaultdict(list), defaultdict(list), defaultdict(list), False
        for (rid, name), m in sorted(self.members.items()):
            if m.why:
                why[rid].append(f"{name}: {m.why}")
                continue
            sub, tree, lines = self.subs[rid], self.trees[rid], self.lines_of(self.subs[rid])
            mine, ok = self._decl_edit(m), True
            if mine is None:
                m.why, refused = "declaration bounds without a span", True
                continue
            sections = {ln for ln, _, _, mode in m.calls if mode == "section"}
            uses = self.uses[rid]
            nsub = 0
            for (l0, col0), (path, first) in subscripted_designators(tree, sub.startline, sub.endline):
                if path != name:
                    continue
                st = statement_start(lines, l0)
                span = first.span if first.span is not None else _bounds_span(first)
                if span is None or span[0] != span[2]:
                    ok = False
                    break
                if st in m.sites:
                    mine.append(replace_cols(span[0], span[1], span[3], m.sites[st], why=f"filter-locals {name}"))
                elif st in sections:
                    mine.append(self._section_edit(uses, name, first, span, l0, col0))
                else:
                    ok = False
                    break
                nsub += 1
            whole = sum(1 for _, _, _, mode in m.calls if mode == "whole")
            if not ok or len(uses.subscripted.get(name, [])) != nsub or uses.bare.get(name, 0) != whole:
                m.why, refused = "the file has uses the analysis did not record", True
                continue
            problem = uses.uncovered(name, nsub) if not whole else ""
            if whole and (_in(name, uses.io_text) or _in(name, uses.directive_text)):
                problem = "used in an I/O statement or a directive"
            if problem:
                m.why, refused = problem, True
                continue
            edits[rid] += mine
            done[rid].append(name)
        return edits, done, why, refused

    @staticmethod
    def _section_edit(uses, name: str, first, span, l0: int, col0: int):
        """A clump-section actual becomes the whole array: `x(begc:endc)` and `x(begc:endc, :)` -> `x`
        (the member is filter-sized now), `x(begc:endc, j)` -> `x(:, j)`."""
        node = next((e for e in uses.subscripted.get(name, []) if e.args and e.args[0] is first), None)
        rest = node.args[1:] if node is not None else None
        if (node is not None and node.span is not None and node.span[2] == l0
                and all(getattr(a, "start", 0) is None and getattr(a, "end", 0) is None
                        and type(a).__name__ == "BoundsExpression" for a in rest)):
            return replace_cols(l0, col0, node.span[3], "", why=f"filter-locals {name}")
        return replace_cols(span[0], span[1], span[3], ":", why=f"filter-locals {name}")

    def _decl_edit(self, m: Member) -> Optional[list]:
        b = m.bound
        if m.dummy and m.assumed:
            s = b.start.span
            if s is None or s[0] != s[2]:
                return None
            return [replace_cols(s[0], s[1], s[3], "", why=f"filter-locals {m.name}")]
        if b.start.span is None or b.end.span is None or b.start.span[0] != b.end.span[2]:
            return None
        return [replace_cols(b.start.span[0], b.start.span[1], b.end.span[3], m.count, why=f"filter-locals {m.name}")]


def _bounds_span(b):
    s, e = getattr(b, "start", None), getattr(b, "end", None)
    if s is None or e is None or s.span is None or e.span is None:
        return None
    return (s.span[0], s.span[1], e.span[2], e.span[3])


def _in(name: str, text: str) -> bool:
    return re.search(rf"(?<![\w%]){re.escape(name)}(?!\w)", text) is not None


def text_call_count(srcroot: Path, subs, name: str) -> int:
    """`call name(` statements in the ELM sources' text (the files of `subs` when the srcroot has
    no ELM tree); -1 if the name is also bound as a procedure (generic interface, type-bound
    procedure, procedure pointer). A plain text count, used only to refuse."""
    n, other = 0, False
    call_re = re.compile(rf"^\s*(?:\d+\s+)?(?:if\s*\(.*\)\s*)?call\s+{re.escape(name)}\b", re.I)
    proc_re = re.compile(rf"(procedure\b.*\b{re.escape(name)}\b|=>\s*{re.escape(name)}\b)", re.I)
    elm = srcroot / "components" / "elm" / "src"
    files = (sorted(elm.rglob("*.F90")) if elm.is_dir()
             else sorted({Path(s.filepath) if Path(s.filepath).is_absolute() else srcroot / s.filepath for s in subs}))
    for f in files:
        if "external_models" in f.parts:
            continue
        for line in f.read_text(errors="replace").split("\n"):
            code = line.split("!", 1)[0]
            if call_re.match(code):
                n += 1
            elif proc_re.search(code):
                other = True
    return -1 if other else n
