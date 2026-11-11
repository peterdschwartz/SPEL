"""
Per-iteration local arrays become scalars (`spel transform scalarize-locals`).

`local_forms` finds local arrays whose every use is one element inside one loop nest, written
before it is read in each iteration (NitrifDenitrif's `r_min(fc,j)`, a work array per filter
member and level): their values never live from one iteration to the next. This pass applies
that finding:

    real(r8) :: r_min(num_soilc, nlevdecomp)   ->   real(r8) :: r_min
    r_min(fc,j) = ...                          ->   r_min = ...

A scalar assigned in a compute loop is privatized by acc-apply (and acc2omp), so the GPU needs
no device storage, enter/exit data or copy for it. Results are unchanged.

Every use in the file's text must be one the analysis judged (filter_locals.TextUses): uses in
branches the analysis commented out (`!#py`), in I/O statements or in directives refuse the
variable. Declarations with an initializer or a `dimension(...)` attribute are left alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import TYPE_CHECKING, Optional

from spel.scripts.fortran_parser.scope_walk import Access
from spel.transforms.core.edits import replace_cols
from spel.transforms.core.registry import transform
from spel.transforms.core.transform import Change, Plan, Refusal, Transform
from spel.transforms.filter_locals import TextUses

if TYPE_CHECKING:
    from spel.scripts.analyze_subroutines import Subroutine
    from spel.scripts.fortran_parser.spel_ast import EntityDecl, SubroutineDefinitionConstruct


def _one_line(span: Optional[tuple[int, int, int, int]]) -> bool:
    return span is not None and span[0] == span[2]


def plan_routine(sub: Subroutine, lines: list[str], tree: SubroutineDefinitionConstruct,
                 findings) -> tuple[list, list[str], list[str]]:
    """(edits, changed names, reasons) for the `scalar` findings of one routine."""
    from spel.scripts.fortran_parser.spel_ast import VariableDecl
    from spel.scripts.registrations import iter_statements

    if tree.contains:
        return [], [], ["has internal procedures"]
    uses = TextUses(sub, lines, tree)
    ents: dict[str, EntityDecl] = {}
    for stmt, _ in iter_statements(tree.body):
        if isinstance(stmt, VariableDecl):
            for ent in stmt.entities:
                ents[ent.token.literal.lower()] = ent
    recorded: dict[str, int] = {}
    for e in sub.record.events:
        if isinstance(e, Access) and e.context != "init":
            recorded[e.base.lower()] = recorded.get(e.base.lower(), 0) + 1
    edits, done, why = [], [], []
    for f in findings:
        name = f.name.lower()
        ent = ents.get(name)
        if ent is None or not ent.bounds or ent.init is not None or not _one_line(ent.span):
            why.append(f"{name}: declaration is not `{name}(dims)` on one line")
            continue
        found = uses.subscripted.get(name, [])
        problem = uses.uncovered(name, recorded.get(name, 0))
        if problem:
            why.append(f"{name}: {problem}")
            continue
        if not all(_one_line(e.span) and _one_line(e.function.span) for e in found):
            why.append(f"{name}: a use spans continuation lines")
            continue
        ln, c0, _, c1 = ent.span
        mine = [replace_cols(ln, c0, c1, lines[ln][c0:c0 + len(name)], why=f"scalarize-locals {name}")]
        for e in found:
            l0, a0, _, a1 = e.span
            fl, f0, _, f1 = e.function.span
            mine.append(replace_cols(l0, a0, a1, lines[fl][f0:f1], why=f"scalarize-locals {name}"))
        edits += mine
        done.append(name)
    return edits, done, why


@dataclass
class Options:
    routines: list[str] = field(default_factory=list)  # module::name (default: the analyzed/active ones)
    report_all: bool = False  # also list why findings were left alone


@transform
class ScalarizeLocals(Transform):
    """
    Matches: local arrays used one element per loop iteration, written before read in each
             iteration (local_forms' `scalar` findings).
    Changes: the declaration loses its dimensions and every use its subscripts.
    Refuses: uses the analysis did not see (`!#py` branches, I/O, directives), initialized or
             `dimension(...)` declarations, uses over continuation lines, routines with internal
             procedures. `--report-all` lists them.
    """

    name = "scalarize-locals"
    summary = "per-iteration local arrays become scalars (local_forms' scalar findings applied)"
    Options = Options

    def build(self, ctx, opts) -> Plan:
        from spel.scripts.registrations import parse_routine
        from spel.transforms.local_forms import check_routine

        out = Plan(self.name)
        for sub in ctx.routines(opts.routines or None):
            if sub.record is None:
                continue
            findings = [f for f in check_routine(sub) if f.kind == "scalar"]
            if not findings:
                continue
            path = ctx.path(sub)
            try:
                tree = parse_routine(path, sub.startline, sub.endline, keep_spans=True, skip_io=True, cpp=True)
            except (Exception, SystemExit) as err:  # noqa: BLE001 - the parser exits on errors
                out.add(Refusal(sub.id, f"does not parse with spans: {err}"))
                continue
            edits, done, why = plan_routine(sub, ctx.lines(path), tree, findings)
            if edits:
                out.add(Change(path, edits, where=sub.id, why=f"scalars: {', '.join(done)}"))
            if opts.report_all:
                out.notes += [f"{sub.id}: {w}" for w in why]
        return out
