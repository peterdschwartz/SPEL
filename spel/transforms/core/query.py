"""
Questions transforms ask of the analysis: routines (also ones outside it), their statements,
loops and accesses. Promoted from the private helpers the transforms used to import from each
other, so every transform asks them the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional


@dataclass
class OutsideRoutine:
    """A routine the analysis does not contain (elm_drv never reaches it, e.g. initialize2):
    located and parsed on demand. Has the attributes the transforms read from a Subroutine."""

    id: str
    module: str
    name: str
    filepath: str
    startline: int
    endline: int
    syntax_tree: object

    record = None


def routine(fut, rid: str):
    """`fut.subroutine_dict[rid]`, or the routine parsed from its module when the analysis lacks
    it; None when neither has it."""
    sub = fut.subroutine_dict.get(rid)
    if sub is not None:
        return sub
    return outside_routine(rid)


def outside_routine(rid: str) -> Optional[OutsideRoutine]:
    from spel.scripts.driver_callsites import CallerRoutine, load_module, routine_lines
    from spel.scripts.fortran_modules import get_filename_from_module

    module, name = rid.lower().split("::")
    try:
        _, lines = load_module(module)
        body = routine_lines(lines, name)
        caller = CallerRoutine(module, name, body)
    except Exception:
        return None
    return OutsideRoutine(
        rid.lower(), module, name, get_filename_from_module(module), body[0].ln, body[-1].ln, caller.syntax_tree
    )


# --- files and parse trees -----------------------------------------------------------------


def source_path(sub) -> Path:
    """Absolute source file of a routine (`sub.filepath` may be relative to the srcroot)."""
    from spel.scripts.config import E3SM_SRCROOT

    p = Path(sub.filepath)
    return p if p.is_absolute() else Path(E3SM_SRCROOT) / p


def routine_tree(sub):
    """
    A routine's SubroutineDefinitionConstruct with original line numbers: from the
    preprocessed module (module_spec.parse_modules, cached), so cpp blocks inside a
    statement parse, else parsed from the raw lines (raises when that fails too).
    """
    from spel.scripts.fortran_parser.module_spec import ModuleSpecError, parse_modules
    from spel.scripts.registrations import parse_routine

    name = sub.id.split("::")[-1].lower()
    try:
        found = [p for m in parse_modules(source_path(sub)) for p in m.subprograms
                 if p.name.lower() == name and p.lineno == sub.startline]
    except (ModuleSpecError, OSError):
        found = []
    return found[0] if found else parse_routine(source_path(sub), sub.startline, sub.endline)


def first_executable(tree, rec) -> int:
    """
    Line of the routine's first executable statement (its own body, not nested ones).
    Statement-function definitions (`f(x) = ...` with `f` declared scalar) belong to the
    specification part although the parser reads them as assignments.
    """
    from spel.scripts.fortran_parser import spel_ast as A

    spec_kinds = (A.UseStatement, A.ImplicitNoneStatement, A.VariableDecl, A.TypeDef, A.AccessStatement,
                  A.ProcedureStatement)

    def statement_function(st) -> bool:
        e = getattr(st, "expression", None)
        if not (isinstance(st, A.ExpressionStatement) and isinstance(e, A.InfixExpression) and e.operator.strip() == "="):
            return False
        lhs = e.left_expr
        if not isinstance(lhs, A.FuncExpression) or not isinstance(lhs.function, A.Identifier):
            return False
        decl = rec.declaration(lhs.function.value.lower()) if rec is not None else None
        return decl is not None and decl.dim == 0

    for st in tree.body.statements:
        if not isinstance(st, spec_kinds) and not statement_function(st):
            return st.lineno
    return tree.end_ln


def construct_end(st, lines: list[str]) -> int:
    """Last line of a statement: a construct's end, else the last continuation line."""
    end = getattr(st, "construct_end_ln", -1)  # an if construct's `end if` (its end_ln ends the first branch)
    if end is None or end < 0:
        end = getattr(st, "end_ln", -1)
    if end is not None and end >= 0:
        return end
    ln = st.lineno
    while ln + 1 < len(lines) and lines[ln].split("!")[0].rstrip().endswith("&"):
        ln += 1
    return ln


# --- call trees and aborts -----------------------------------------------------------------

ABORTS = {"endrun", "shr_sys_abort", "abort", "mpi_abort"}


def call_tree(fut, roots: Iterable[str],
              exclude: Optional[Callable[[str], bool]] = None) -> dict[str, object]:
    """The routines reachable from `roots` through `child_subroutines`.

    `exclude(routine_id)` cuts the tree: an excluded routine is left out together with
    everything reachable only through it. A routine kept on the host keeps its callees on
    the host: a kernel in such a callee finds the host routine's data absent (2026-10-29:
    template_col_from_natveg_array, called only by the host set_old_column_weights)."""
    out: dict[str, object] = {}
    stack = [r for r in roots]
    while stack:
        sid = stack.pop()
        if sid in out or sid not in fut.subroutine_dict:
            continue
        if exclude is not None and exclude(sid):
            continue
        sub = fut.subroutine_dict[sid]
        out[sid] = sub
        stack.extend(ch.id for ch in getattr(sub, "child_subroutines", {}).values())
    return out


def abort_calls(stmts) -> list[int]:
    """Lines of `call endrun(...)` & co. in a statement list (parse tree). The analysis drops
    these calls from its edited sources, so a routine's record does not list them."""
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.scripts.registrations import iter_statements

    out = []
    for st, _ in iter_statements(stmts):
        if isinstance(st, A.SubCallStatement):
            f = st.function.function if isinstance(st.function, A.FuncExpression) else st.function
            if str(f).split("%")[-1].lower() in ABORTS:
                out.append(st.lineno)
    return out


def cold_branches(stmts) -> list[tuple[int, int, int]]:
    """
    (first line, last line, line of the first statement) of each branch of the outermost
    if constructs that can abort (error checks): host code there runs only when a check fails.
    """
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.scripts.registrations import iter_statements

    out: list[tuple[int, int, int]] = []
    done_to = -1
    for st, _ in iter_statements(stmts):
        if st.lineno <= done_to or not isinstance(st, A.IfConstruct):
            continue
        branches = [st.consequence] + [e.consequence for e in st.else_ifs]
        headers = [e.lineno for e in st.else_ifs]
        if st.else_ is not None:
            branches.append(st.else_.alternative)
            headers.append(st.else_.lineno)
        if not abort_calls(Block(branches)):
            continue
        done_to = st.construct_end_ln
        for b, nxt in zip(branches, headers + [st.construct_end_ln]):
            if b.statements:
                out.append((b.statements[0].lineno, nxt - 1, b.statements[0].lineno))
    return out


class Block:
    """Several blocks as one statement list (for iter_statements)."""

    def __init__(self, blocks):
        self.statements = [x for b in blocks for x in b.statements]


# --- loops and OpenACC constructs ------------------------------------------------------------

END_OF = {"parallel": "end parallel", "kernels": "end kernels", "serial": "end serial", "data": "end data"}


def all_loops(loops):
    """Every LoopInfo of a loop nest list, depth first."""
    for lp in loops:
        yield lp
        yield from all_loops(lp.children)


def loop_at(loops, ln: int):
    """The LoopInfo starting on line `ln`, or None."""
    for lp in loops:
        if lp.start_ln == ln:
            return lp
        hit = loop_at(lp.children, ln)
        if hit is not None:
            return hit
    return None


def compute_ranges(directives, rec) -> Iterator[tuple[object, int, int]]:
    """(construct, first line, last line) of every OpenACC compute construct of a routine."""
    stack: list = []
    for d in directives:
        if d.is_compute_loop:
            # the loop follows the directive: the first analyzed loop starting after it
            nxt = min((lp.start_ln for lp in all_loops(rec.loop_nest) if lp.start_ln > d.end_ln), default=None)
            lp = loop_at(rec.loop_nest, nxt) if nxt is not None else None
            if lp is not None:
                yield d, d.start_ln, lp.end_ln
        elif d.is_compute_region:
            stack.append(d)
        elif stack and d.construct == END_OF.get(stack[-1].construct):
            start = stack.pop()
            yield start, start.start_ln, d.end_ln


def device_spans(directives, rec) -> list[tuple[int, int]]:
    """(first, last) lines of the routine's compute constructs: the code that runs on the device."""
    return [(lo, hi) for _, lo, hi in compute_ranges(directives, rec)]


def above_directives(ln: int, directives) -> int:
    """
    The first line of the directives that end right above line `ln` (a `!$acc parallel
    loop` and its continuations belong to the statement after them): inserting there
    keeps them attached to it.
    """
    ends = {d.end_ln: d.start_ln for d in directives}
    while ln - 1 in ends:
        ln = ends[ln - 1]
    return ln
