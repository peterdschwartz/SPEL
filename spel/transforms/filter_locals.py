"""
Local work arrays stored on the filter (`spel transform filter-locals`).

ELM routines declare work arrays over the whole clump, `tk(bounds%begc:bounds%endc, nlevgrnd)`,
and use them only at filter members, `tk(c, j)` with `c = filter_nolakec(fc)` inside
`do fc = 1, num_nolakec`. This pass stores such an array on the filter:

    real(r8) :: tk(num_nolakec, 1:nlevgrnd)
    ...
    tk(fc, j) = ...

The values are the same, the memory is the filter's size and the accesses are contiguous
(on the GPU a kernel over fc reads consecutive elements). Both the GPU branch and the
t-meshes performance commits ("Index ... work arrays by filter position") did this by hand.

A local qualifies when all of these hold (anything else is reported and left alone):
- an intrinsic-type array, not a parameter/pointer/target/allocatable/save variable, whose first
  dimension is declared `bounds%beg<x>:bounds%end<x>` (or `beg<x>:end<x>`);
- every access subscripts it with a plain index `c` whose form (compact/index_forms.py) is
  `c = F(fc)`, a gather from one filter array F through a loop index `fc` over 1..N, and the
  access lies inside that `do fc` loop (after the loop fc is N+1, not the member's position);
- the same F and N everywhere; N is a dummy (the automatic array's size); F is not written in
  the routine;
- never passed as an actual argument, reached through an associate name, allocated, inquired
  (size/lbound) or used whole; the routine has no internal procedures.
Rank > 1 arrays keep their other dimensions.

Edits come from the parser's spans (`parse_routine(..., keep_spans=True)`), never from regexes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from spel.scripts.fortran_parser.scope_walk import Access
from spel.scripts.fortran_parser.spel_ast import BoundsExpression, Expression
from spel.scripts.fortran_parser.symbols import Origin
from spel.transforms.compact.index_forms import IndexContext, norm
from spel.transforms.core.edits import replace_cols, span_text
from spel.transforms.core.registry import transform
from spel.transforms.core.source import statement_start
from spel.transforms.core.transform import Change, Plan, Refusal, Transform

if TYPE_CHECKING:
    from spel.scripts.analyze_subroutines import Subroutine
    from spel.scripts.fortran_parser.spel_ast import SubroutineDefinitionConstruct

INTRINSIC_TYPES = ("integer", "real", "logical", "double precision", "doubleprecision", "complex")
LEVELS = ("c", "p", "l", "g", "t")
DISQUALIFY = {"arg", "ptr-assign", "allocate", "deallocate", "size", "inquiry", "nullify", "read",
              "stat", "iostat", "errmsg", "iomsg"}


@dataclass
class Candidate:
    name: str
    count: str  # N: the declaration's new first dimension
    sites: dict[int, str] = field(default_factory=dict)  # line -> loop index (fc) of its accesses


def _clump_bounds(b: Optional[Expression]) -> bool:
    """`bounds%begc:bounds%endc` (or `begc:endc`) as a declared dimension."""
    if not isinstance(b, BoundsExpression) or b.start is None or b.end is None:
        return False
    s, e = norm(b.start).lower(), norm(b.end).lower()
    for x in LEVELS:
        if (s, e) in ((f"bounds%beg{x}", f"bounds%end{x}"), (f"beg{x}", f"end{x}")):
            return True
    return False


def candidates(sub_id: str, rec, decl_bounds: dict[str, object]) -> tuple[list[Candidate], list[str]]:
    """The locals to store on their filter, and why the other clump-sized locals are not."""
    ctx = IndexContext(sub_id, rec, {}, {})
    loops = rec.all_loops()
    accs: dict[str, list[tuple[int, Access]]] = defaultdict(list)
    for pos, e in enumerate(rec.events):
        if isinstance(e, Access) and e.origin is Origin.LOCAL:
            accs[e.base.lower()].append((pos, e))
    out, why = [], []
    for name, decl in sorted(rec.local_variables.items()):
        name = name.lower()
        if decl.dim < 1 or name not in decl_bounds or not _clump_bounds(decl_bounds[name]):
            continue
        attrs = {a.lower() for a in decl.attrs}
        if attrs & {"parameter", "pointer", "target", "allocatable", "save"}:
            why.append(f"{name}: {'/'.join(sorted(attrs))}")
            continue
        if not decl.type.lower().startswith(INTRINSIC_TYPES):
            continue
        cand, reason = _check(name, accs.get(name, []), ctx, loops, rec, decl.ln)
        if cand is None:
            why.append(f"{name}: {reason}")
        else:
            out.append(cand)
    return out, why


def _check(name, accs, ctx, loops, rec, decl_ln: int) -> tuple[Optional[Candidate], str]:
    if not accs:
        return None, "never used"
    gathers: set[tuple[frozenset, str]] = set()
    cand = Candidate(name, "")
    for pos, a in accs:
        if a.context == "init":
            return None, "initialized in its declaration"
        if a.context in DISQUALIFY or a.via is not None:
            return None, f"{a.context}{' via ' + a.via if a.via else ''} at line {a.ln + 1}"
        f = ctx.access_form(a, pos)
        sub0 = a.subs[0] if a.subs else ""
        inner = f.inner
        if (f.kind != "filter" or f.var != sub0 or inner is None or inner.kind != "count"
                or not inner.var or not inner.text):
            return None, f"line {a.ln + 1}: subscript {sub0 or '(whole)'} is {f}"
        if not any(l.index == inner.var and l.contains(a.ln) for l in loops):
            return None, f"line {a.ln + 1}: {sub0} used outside the loop over {inner.var}"
        if cand.sites.get(a.ln, inner.var) != inner.var:
            return None, f"line {a.ln + 1}: two different filter positions on one line"
        cand.sites[a.ln] = inner.var
        gathers.add((f.filter, inner.text.lower()))
    if len(gathers) != 1:
        return None, f"gathers through several filters/counts {sorted((sorted(f), n) for f, n in gathers)}"
    filt, count = gathers.pop()
    d = rec.declaration(count)
    if d is None or d.intent is None or d.dim != 0:
        return None, f"count {count} is not a scalar dummy"
    if d.ln >= decl_ln:
        # a specification expression may only use a variable typed before it
        return None, f"count {count} is declared after {name}"
    for path in filt:
        base = path.split(":")[-1] if path.startswith(("dummy:", "local:")) else path
        written = [a for a in rec.accesses() if a.base.lower() == base.split("%")[0].lower() and "w" in a.status]
        if written or not path.startswith("dummy:"):
            return None, f"filter {path} is written in the routine or is not a dummy"
    cand.count = count
    return cand, ""


def plan_routine(sub: Subroutine, lines: list[str], tree: SubroutineDefinitionConstruct) -> tuple[list, list[str], list[str]]:
    """(edits, changed names, reasons) for one routine; `tree` parsed with keep_spans."""
    from spel.scripts.fortran_parser.spel_ast import EntityDecl, VariableDecl
    from spel.scripts.registrations import iter_statements
    from spel.transforms.compact.rewrite import subscripted_designators

    rec = sub.record
    if tree.contains:
        return [], [], ["has internal procedures"]
    uses = TextUses(sub, lines, tree)
    ents: dict[str, EntityDecl] = {}
    for stmt, _ in iter_statements(tree.body):
        if isinstance(stmt, VariableDecl):
            for ent in stmt.entities:
                ents[ent.token.literal.lower()] = ent
    decl_bounds = {n: e.bounds[0] for n, e in ents.items() if e.bounds}
    cands, why = candidates(sub.id, rec, decl_bounds)
    edits, done = [], []
    for c in cands:
        b = decl_bounds[c.name]
        if b.start.span is None or b.end.span is None or b.start.span[0] != b.end.span[2]:
            why.append(f"{c.name}: declaration bounds without a span")
            continue
        ln, c0, c1 = b.start.span[0], b.start.span[1], b.end.span[3]
        site_edits = [replace_cols(ln, c0, c1, c.count, why=f"filter-locals {c.name}")]
        hits = defaultdict(int)
        lo, hi = sub.startline, sub.endline
        for (l0, col0), (path, first) in subscripted_designators(tree, lo, hi):
            st = statement_start(lines, l0)  # the record's line of a continued statement
            if path != c.name or st not in c.sites:
                continue
            if first.span[0] != first.span[2]:
                continue  # a subscript over continuation lines: reported below as missing
            site_edits.append(replace_cols(first.span[0], first.span[1], first.span[3], c.sites[st],
                                           why=f"filter-locals {c.name}"))
            hits[st] += 1
        missing = sorted(set(c.sites) - set(hits))
        if missing:
            why.append(f"{c.name}: no subscript found on line(s) {[m + 1 for m in missing]}")
            continue
        problem = uses.uncovered(c.name, len(site_edits) - 1)
        if problem:
            why.append(f"{c.name}: {problem}")
            continue
        edits += site_edits
        done.append(c.name)
    return edits, done, why


class TextUses:
    """
    Every use of each name in a routine's file text, not only the record's: the analysis works on
    copies whose inactive branches are commented out (`!#py`), so its record can miss uses that the
    file has. `tree` is the routine parsed with keep_spans/skip_io/cpp; I/O statements (blanked
    before parsing) and OpenACC/OpenMP directives (comments to the parser) are searched as text.
    """

    def __init__(self, sub, lines: list[str], tree):
        from spel.scripts.fortran_parser.spel_ast import FuncExpression, Identifier, VariableDecl
        from spel.scripts.registrations import _blank_io_statements, iter_statements
        from spel.scripts.types import LineTuple
        from spel.transforms.compact.rewrite import iter_expressions

        self.bare: dict[str, int] = defaultdict(int)  # name -> uses without subscripts
        self.subscripted: dict[str, list] = defaultdict(list)  # name -> its FuncExpression nodes
        for stmt, _ in iter_statements(tree.body):
            if isinstance(stmt, VariableDecl):
                continue
            heads: set[int] = set()
            for e in iter_expressions(stmt):
                if isinstance(e, FuncExpression) and isinstance(e.function, Identifier):
                    heads.add(id(e.function))
                    self.subscripted[e.function.value.lower()].append(e)
                elif isinstance(e, Identifier) and id(e) not in heads:
                    self.bare[e.value.lower()] += 1
        rng = range(sub.startline, min(sub.endline, len(lines) - 1) + 1)
        blanked = _blank_io_statements([LineTuple(line=lines[i], ln=i) for i in rng])
        self.io_text = " ".join(lines[lt.ln].lower() for lt in blanked if not lt.line and lines[lt.ln].strip())
        self.directive_text = " ".join(lines[i].lower() for i in rng
                                       if lines[i].lstrip().lower().startswith(("!$acc", "!$omp")))

    def uncovered(self, name: str, planned: int) -> str:
        """Why the planned edits (`planned` subscripted uses) do not cover every use, or ''."""
        n = len(self.subscripted.get(name, []))
        if self.bare.get(name) or n != planned:
            return (f"the file has uses the analysis did not record ({n} subscripted, "
                    f"{self.bare.get(name, 0)} bare; {planned} planned)")
        if _word_in(name, self.io_text):
            return "used in an I/O statement"
        if _word_in(name, self.directive_text):
            return "named in an OpenACC/OpenMP directive"
        return ""


def _word_in(name: str, text: str) -> bool:
    """`name` as a whole word in `text` (plain text search, used only to refuse)."""
    import re

    return re.search(rf"(?<![\w%]){re.escape(name)}(?!\w)", text) is not None


@dataclass
class Options:
    routines: list[str] = field(default_factory=list)  # module::name (default: the analyzed/active ones)
    report_all: bool = False  # also list why clump-sized locals were left alone
    interprocedural: bool = True  # also arrays passed to other routines (filter_locals_ip)


@transform
class FilterLocals(Transform):
    """
    Matches: local work arrays declared over the clump (`x(bounds%begc:bounds%endc, ...)`) and used
             only at filter members inside the filter loop (`x(c, ...)`, `c = filter(fc)`).
    Changes: their first dimension becomes the filter count and every subscript the filter
             position: `x(num_f, ...)`, `x(fc, ...)`. Values and results are unchanged.
    Refuses: locals passed as arguments, used whole or through an associate name, allocatable,
             gathered through several filters, used after their loop; routines with internal
             procedures. `--report-all` lists them.
    """

    name = "filter-locals"
    summary = "store local work arrays on their filter (x(c) -> x(fc), sized by the filter count)"
    Options = Options

    def build(self, ctx, opts) -> Plan:
        from spel.scripts.registrations import parse_routine

        out = Plan(self.name)
        if opts.interprocedural:
            return self._build_groups(ctx, opts, out)
        for sub in ctx.routines(opts.routines or None):
            rec = sub.record
            if rec is None or not any(
                    d.dim >= 1 for d in rec.local_variables.values()):
                continue
            path = ctx.path(sub)
            try:
                tree = parse_routine(path, sub.startline, sub.endline, keep_spans=True, skip_io=True, cpp=True)
            except (Exception, SystemExit) as err:  # noqa: BLE001 - the parser exits on errors; reported, left alone
                out.add(Refusal(sub.id, f"does not parse with spans: {err}"))
                continue
            edits, done, why = plan_routine(sub, ctx.lines(path), tree)
            if edits:
                out.add(Change(path, edits, where=sub.id, why=f"on the filter: {', '.join(done)}"))
            if opts.report_all:
                out.notes += [f"{sub.id}: {w}" for w in why]
        return out

    def _build_groups(self, ctx, opts, out: Plan) -> Plan:
        from spel.scripts.registrations import parse_routine
        from spel.transforms.filter_locals_ip import Planner

        def tree_of(sub):
            try:
                return parse_routine(ctx.path(sub), sub.startline, sub.endline, keep_spans=True, skip_io=True, cpp=True)
            except (Exception, SystemExit) as err:  # noqa: BLE001 - the parser exits on errors
                out.add(Refusal(sub.id, f"does not parse with spans: {err}"))
                return None

        subs = [s for s in ctx.routines(opts.routines or None) if s.record is not None]
        planner = Planner(ctx.fut, subs, lambda sub: ctx.lines(ctx.path(sub)), tree_of, ctx.srcroot)
        for sub in subs:
            planner.collect(sub)
        edits, done, why = planner.plan()
        for rid in sorted(edits):
            sub = planner.subs[rid]
            out.add(Change(ctx.path(sub), edits[rid], where=rid, why=f"on the filter: {', '.join(done[rid])}"))
        if opts.report_all:
            out.notes += [f"{rid}: {w}" for rid in sorted(why) for w in why[rid]]
        for rid, w in sorted(planner.refused.items()):
            out.notes.append(f"{rid}: {w}")
        return out
