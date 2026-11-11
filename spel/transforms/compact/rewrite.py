"""
Rewrite ELM sources so the selected fields of a derived type use compact
storage (see select.py for the selection and CompactLayoutMod.F90 for the
storage).

The layout: every selected field lives in a group buffer allocated over the
process soil filter (``1:compact_num_soilp`` or ``_soilc``); a group is one
holder embedding x one zeroing guard x one subgrid level, and the type's
pointer components alias the group's columns. The clump offsets
(``compact_soilp_off``) make each clump a contiguous slice. History reads the
fields through the inverse map (``compact_cidx_p``), in histComputeMod.

Edits:

  * each access site ``x(p)``, p = filter_soilp(fp), becomes ``x(k0p+fp)``
    (``k0c`` for column fields); the routine gets
    ``use CompactLayoutMod, only : compact_soilp_off, ...`` and a local
    ``k0p = compact_soilp_off`` (copied once, so it is safe inside OpenMP/OpenACC loops)
  * the holder type gets ``integer :: cgrp_id(ngroups)`` and a plain module
    procedure ``<holder>_compact_bind(this)`` that creates its groups on first
    use and points the fields at the group columns
  * in the Init routine the ``allocate(this%x(beg:end)); this%x(:) = spval``
    lines of the compact fields are dropped and the bind call is inserted after
    the last of them (before the history registrations)
  * SetValues zeroes the fields by group: ``x(k0p+1:k0p+num) = value`` once the
    layout is bound; before that (the special-landunit call in Init) it records
    the value as the group's fill (what history reads on active special points)
  * a generated ``CompactBindMod`` builds the process filters from
    ``filter(nc)`` (``compact_filters_set_from_filters``) and, when they change,
    resizes every group and rebinds every instance (``compact_filters_updated``);
    elm_initializeMod calls them around the type initialization, elm_drv after
    ``dynSubgrid_driver`` when the weights were updated
  * get_clump_bounds sets the clump offsets (``compact_set_clump(n)``)
  * CompactLayoutMod.F90 and CompactBindMod.F90 are added to components/elm/src/main

Edits are located by content and checked: a site whose text doesn't match
the analysis stops the rewrite. The sources must be the ones analyzed and
not yet rewritten; `.spel-compact.json` in components/elm/src/main records
what was applied. To reapply after a rebase: drop the generated changes,
`spel analyze`, run the rewrite again.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from spel.scripts.analyze_subroutines import Subroutine
    from spel.scripts.fortran_parser.spel_ast import SubroutineDefinitionConstruct

from spel.transforms.core.registry import transform
from spel.transforms.core.transform import Change, Plan as TPlan, Refusal, Transform
from spel.transforms.compact.holders import Holder, base_routine, embedded_holders, nutrient_manifest
from spel.transforms.compact.inventory import DRIVER_ID
from spel.scripts.registrations import (
    HIST_CALLS, RESTART_CALL, _keyword_args, iter_statements, parse_routine, registrations,
)
from spel.transforms.compact.select import Polymorphic, Selection, statement_end, statement_start
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.types import LineTuple
from spel.scripts.fortran_parser.scope_walk import AssignEvent, flatten_designator
from spel.scripts.fortran_parser.spel_ast import (
    AllocateStatement,
    AssignmentStatement,
    AssociateConstruct,
    BoundsExpression,
    DoLoop,
    Expression,
    ExpressionStatement,
    FieldAccessExpression,
    FuncExpression,
    Identifier,
    IfConstruct,
    InfixExpression,
    KeywordArgument,
    PointerAssignment,
    Statement,
    SubCallStatement,
    TypeDef,
    UseStatement,
    VariableDecl,
    to_actual_arg,
)
from spel.scripts.fortran_parser.symbols import SymbolKind
from spel.transforms.core.edits import span_text
from spel.transforms.split.modules import parse_module_file, use_items

TEMPLATE = Path(__file__).parent / "templates" / "CompactLayoutMod.F90"
MANIFEST = ".spel-compact.json"
BIND_MODULE = "CompactBindMod"
BIND_FILE = f"{BIND_MODULE}.F90"
K0 = {"p": "k0p", "c": "k0c"}  # local copy of the clump offset, per subgrid level
OFFSET = {"p": "compact_soilp_off", "c": "compact_soilc_off"}
CIDX = {"p": "compact_cidx_p", "c": "compact_cidx_c"}  # inverse maps, for the "map" sites
LEVEL = {"p": "compact_lev_patch", "c": "compact_lev_col"}
MAIN = {"p": "soilp", "c": "soilc"}
OFF = "compact_off"  # optional dummy of a Polymorphic routine
KC = "kc"  # its index for the compact-capable dummies
TAG = "! spel compact"
DECOMP_ROUTINES = ("get_clump_bounds_new", "get_clump_bounds_old")
GROUP_NAME_LEN = 64  # compact_group_type%name
INIT_BIOGEOCHEM = "elm_inst_biogeochem"
REWEIGHT = "reweight_wrapup"
DYN_SUBGRID = "dynsubgrid_driver"
RESIZE_CONDITION = "get_subgrid_weights_updated() .or. nstep_mod == 0 .or. first_restart_step_mod"
REGISTRATION_CALLS = set(HIST_CALLS) | {RESTART_CALL}


class RewriteError(Exception):
    pass


def known(rec, name: str) -> bool:
    """Whether `name` resolves in the routine (unknown names resolve as EXTERNAL)."""
    sym = rec.lookup(name)
    return sym is not None and sym.kind is not SymbolKind.EXTERNAL


def designator_text(expr) -> Optional[str]:
    """`this % retrans % x(begp:endp)` -> 'this%retrans%x'; None if not a designator."""
    try:
        return "%".join(n.lower() for n, _ in flatten_designator(expr))
    except Exception:
        return None


def iter_expressions(node, root: bool = True) -> Iterator[Expression]:
    """
    Every Expression node of `node` itself (its condition, bounds, arguments...),
    depth first; nested statements (a construct's body) are not entered: iterate
    them with `iter_statements`.
    """
    if isinstance(node, Statement) and not root:
        return
    if isinstance(node, Expression):
        yield node
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from iter_expressions(item, False)
        return
    if isinstance(node, dict):
        for item in node.values():
            yield from iter_expressions(item, False)
        return
    if not hasattr(node, "__dict__"):
        return
    if root and isinstance(node, IfConstruct):
        # the `else if (...)` conditions belong to the construct's own lines (ElseIf is a
        # Statement, so the generic walk below would skip them)
        for elif_ in node.else_ifs:
            yield from iter_expressions(elif_.condition, False)
    for key, value in vars(node).items():
        if key in ("token", "parent", "span", "args_close") or isinstance(value, (str, int, float, bool, Enum, type(None))):
            continue
        yield from iter_expressions(value, False)


def name_end(expr) -> Optional[tuple[int, int]]:
    """(line, column) just after the name of a designator, before its last subscript list."""
    while isinstance(expr, FieldAccessExpression):
        expr = expr.field
    if isinstance(expr, FuncExpression):
        expr = expr.function
    return tuple(expr.span[2:]) if expr.span else None


def subscripted_designators(tree, ln0: int, ln1: int) -> Iterator:
    """
    `(name_end, (path, first subscript))` of every designator `a%b(i, ...)`
    (subscripts only on its last part) that starts on lines ln0..ln1 of `tree`
    (parsed with keep_spans); `path` is lowercase, without spaces.
    """
    fields: set[int] = set()  # the parts of a designator: not designators of their own
    for stmt, _ in iter_statements(tree.body):
        for e in iter_expressions(stmt):
            if isinstance(e, FieldAccessExpression):
                fields.add(id(e.field))
            if id(e) in fields or not isinstance(e, (FuncExpression, FieldAccessExpression)):
                continue
            if e.span is None or not ln0 <= e.span[0] <= ln1:
                continue
            try:
                comps = flatten_designator(e)
            except Exception:
                continue
            if any(subs for _, subs in comps[:-1]) or not comps[-1][1] or comps[-1][1][0].span is None:
                continue
            end = name_end(e)
            if end is not None:
                yield end, ("%".join(n.lower() for n, _ in comps), comps[-1][1][0])


def data_arguments(tree, ln0: int, ln1: int) -> list[KeywordArgument]:
    """The `data=` arguments of the calls starting on lines ln0..ln1 of `tree` (parsed with keep_spans)."""
    out = []
    for stmt, _ in iter_statements(tree.body):
        if isinstance(stmt, SubCallStatement) and stmt.function.span and ln0 <= stmt.function.span[0] <= ln1:
            for a in stmt.function.args:
                a = to_actual_arg(a)
                if isinstance(a, KeywordArgument) and a.keyword.lower() == "data" and a.span:
                    out.append(a)
    return out


def assignment_lhs(stmt) -> Optional[Expression]:
    """The left side of an assignment statement (parsed as `a = b` infix or as AssignmentStatement)."""
    if isinstance(stmt, AssignmentStatement):
        return stmt.left
    if isinstance(stmt, ExpressionStatement):
        expr = stmt.expression
        if isinstance(expr, InfixExpression) and expr.operator == "=":
            return expr.left_expr
    return None


def pointer_target(stmt) -> Optional[tuple[Expression, Expression]]:
    """(pointer, target) of a pointer assignment statement, else None."""
    if isinstance(stmt, PointerAssignment):
        return stmt.pointer, stmt.target
    if isinstance(stmt, ExpressionStatement):
        expr = stmt.expression
        if isinstance(expr, InfixExpression) and expr.operator == "=>":
            return expr.left_expr, expr.right_expr
    return None


def is_spval_fill(stmt: Statement) -> bool:
    """`x(...) = spval` (any section) on its own."""
    expr = stmt.expression if isinstance(stmt, ExpressionStatement) else None
    if isinstance(expr, InfixExpression):
        rhs = expr.right_expr
    else:
        rhs = stmt.right if isinstance(stmt, AssignmentStatement) else None
    return rhs is not None and str(rhs).strip().lower() == "spval"


def designators_in(node) -> set[str]:
    """The designator paths (lowercase, `a%b%c`) of every variable reference below `node`."""
    out: set[str] = set()
    for expr in iter_expressions(node):
        if isinstance(expr, (Identifier, FieldAccessExpression, FuncExpression)):
            text = designator_text(expr)
            if text:
                out.add(text)
    return out


@dataclass
class FileEdits:
    path: Path
    lines: list[str]
    subs: dict[int, list[tuple[re.Pattern, str]]] = field(default_factory=lambda: defaultdict(list))
    after: dict[int, list[str]] = field(default_factory=lambda: defaultdict(list))
    before: dict[int, list[str]] = field(default_factory=lambda: defaultdict(list))
    deletes: set[int] = field(default_factory=set)
    # column edits (from parser spans): line -> {(c0, c1): new text}, applied before `subs`
    cols: dict[int, dict[tuple[int, int], str]] = field(default_factory=lambda: defaultdict(dict))

    def replace_cols(self, ln: int, c0: int, c1: int, new: str) -> None:
        """Replace columns [c0, c1) of line `ln` (an insertion if c0 == c1)."""
        edits = self.cols[ln]
        if edits.get((c0, c1), new) != new:
            raise RewriteError(f"{self.path}:{ln + 1}: two different edits of columns {c0}-{c1}")
        for a, b in edits:
            if (a, b) != (c0, c1) and (a < c1 and c0 < b or a == b == c0 or c0 == c1 == a):
                raise RewriteError(f"{self.path}:{ln + 1}: overlapping edits at columns {a}-{b} and {c0}-{c1}")
        edits[(c0, c1)] = new

    def line(self, i: int) -> str:
        """Line `i` with its column edits and substitutions."""
        line = self.lines[i]
        for (c0, c1), new in sorted(self.cols.get(i, {}).items(), reverse=True):
            line = line[:c0] + new + line[c1:]
        for pattern, repl in self.subs.get(i, []):
            line = pattern.sub(repl, line)
        return line

    def result(self) -> list[str]:
        out: list[str] = []
        for i in range(len(self.lines)):
            out.extend(self.before.get(i, []))
            if i not in self.deletes:
                out.append(self.line(i))
            out.extend(self.after.get(i, []))
        return out


@dataclass(frozen=True)
class Group:
    """One buffer: the fields of a holder with the same zeroing guard (at the holder's level)."""

    index: int  # 1-based position in cgrp_id
    guard: Optional[str]  # Fortran condition, None = always
    key: str  # the selection's guard text
    fields: list[str]  # components, in slot order
    nlev: Optional[str] = None  # rank-2 group: the fields' level count (text), else None

    def name(self, base: str) -> str:
        tag = f"{base}|{self.key}" + (f"|{self.nlev}" if self.nlev else "")
        return tag[:GROUP_NAME_LEN]


@dataclass
class RestartPlan:
    """The `restartvar(... data=this%<path>%x)` calls of one restart routine for compact fields."""

    sub_id: str
    path: str
    calls: list[tuple[int, int, str, int]]  # (call line, statement end, component, rank)


@dataclass
class InitPlan:
    """The edits of one Init routine: the allocate lines to drop and where the bind call goes."""

    sub_id: str
    path: str  # component path of the holder in `this` ("" | "core" | "retrans")
    deletes: set[int]
    insert_after: int  # line of the last dropped allocate (the bind goes after it); -1 = no bind here
    indent: str


class Rewriter:
    """
    The edits for one or more types (`sels`). Edits shared by the types (a
    routine's k0 prologue, a module's use of CompactLayoutMod, the bind
    module, the driver and initialization calls, the clump offset) are made once.
    """

    # None on a Rewriter made without __init__ (snippet tests): no analysis, no cache yet
    subs: Optional[dict[str, Subroutine]] = None
    _routines: Optional[dict[str, SubroutineDefinitionConstruct]] = None

    def __init__(self, fut, sels: Selection | list[Selection]):
        self.fut = fut
        self.sels = sels if isinstance(sels, list) else [sels]
        self.subs = fut.subroutine_dict
        self.files: dict[str, FileEdits] = {}
        self.notes: list[str] = []
        self._prologued: dict[str, set[str]] = defaultdict(set)  # routine -> subgrids with a k0 local
        self._module_use: set = set()
        self._trees: dict[str, object] = {}  # file -> parsed ModuleConstruct
        self._routines = {}  # routine -> parse tree with spans
        self.init_plans: dict[str, list[InitPlan]] = {}  # selection type -> plans
        self.restart_plans: dict[str, list[RestartPlan]] = {}
        self.bind_module: str = ""

    def file(self, path) -> FileEdits:
        key = str(path)
        if key not in self.files:
            lines = Path(path).read_text().split("\n")
            self.files[key] = FileEdits(Path(path), lines)
        return self.files[key]

    def module_tree(self, fe: FileEdits):
        """The file's parsed module (its specification part and line spans)."""
        key = str(fe.path)
        if key not in self._trees:
            _, self._trees[key] = parse_module_file(fe.path)
        return self._trees[key]

    def head_uses(self, fe: FileEdits) -> list[UseStatement]:
        return [s for s in self.module_tree(fe).spec.statements if isinstance(s, UseStatement)]

    def module_name(self, fe: FileEdits) -> str:
        """The module's name as written (the parser lowercases names)."""
        tree = self.module_tree(fe)
        m = re.match(r"\s*module\s+(\w+)", fe.lines[tree.lineno], re.IGNORECASE)
        return m.group(1) if m else str(tree.name)

    @staticmethod
    def routine_end(sub) -> int:
        """Line of the routine's `end subroutine`, from its parse."""
        return parse_routine(sub.filepath, sub.startline, sub.endline).end_ln

    def module_use_line(self, fe: FileEdits) -> int:
        """The module's first use statement (new uses go after it)."""
        uses = self.head_uses(fe)
        if not uses:
            raise RewriteError(f"{fe.path}: the module has no use statement to add to")
        return uses[0].lineno

    def add_routine_use(self, fe: FileEdits, sub, module: str, names: list[str]) -> None:
        """`use module, only : names` after the routine header, once per (routine, module)."""
        key = (sub.id, module.lower(), tuple(n.lower() for n in names))
        if key in self._module_use:
            return
        self._module_use.add(key)
        header = statement_end(fe.lines, sub.startline)
        fe.after[header].append(f"    use {module}, only : {', '.join(names)}  {TAG}")

    def add_module_use(self, fe: FileEdits, module: str, names: list[str]) -> None:
        """`use module, only : names` at module level, once per (file, module)."""
        key = (str(fe.path), module.lower(), tuple(n.lower() for n in names))
        if key in self._module_use:
            return
        self._module_use.add(key)
        fe.after[self.module_use_line(fe)].append(f"  use {module}, only : {', '.join(names)}  {TAG}")

    # --- groups ---
    @staticmethod
    def groups(sel: Selection) -> list[Group]:
        by_key: dict[tuple, list[str]] = defaultdict(list)
        for comp, zc in sel.components.items():
            by_key[(zc, sel.levels.get(comp))].append(comp)
        out = []
        for k, ((zc, nlev), comps) in enumerate(by_key.items(), 1):
            guards = [g if g.startswith("(") and g.endswith(")") else f"({g})" for g in zc.split(" & ")]
            guard = None if zc == "always" else " .and. ".join(guards)
            out.append(Group(k, guard, zc, comps, nlev))
        return out

    # --- access sites ---
    def rewrite_sites(self) -> None:
        by_sub: dict[str, list] = defaultdict(list)  # routine -> [(site, subgrid)]
        polymorphic: dict = {}
        poly_subgrid: dict[str, str] = {}
        for sel in self.sels:
            for site in sel.sites:
                by_sub[site.sub].append((site, sel.subgrid))
            for sub_id, poly in sel.polymorphic.items():
                prev = polymorphic.get(sub_id)
                if prev is None:
                    polymorphic[sub_id] = poly
                    poly_subgrid[sub_id] = sel.subgrid
                elif prev.dummies == poly.dummies and poly_subgrid[sub_id] == sel.subgrid:
                    # fields of several types through the same dummies (the shared fields on
                    # the nutrient base type, the element-specific ones on the concrete type):
                    # a call is compact when its fields are selected by either selection
                    polymorphic[sub_id] = Polymorphic(sub_id, prev.dummies, prev.compact_calls | poly.compact_calls)
                else:
                    raise RewriteError(f"{sub_id} takes compact fields of several types through different dummies")
        callers = {c for p in polymorphic.values() for c, _ in p.compact_calls}
        for sub_id in sorted(set(by_sub) | callers):
            sub = self.subs[sub_id]
            fe = self.file(sub.filepath)
            if sub_id in polymorphic:
                self.rewrite_polymorphic(fe, sub, [s for s, _ in by_sub[sub_id]])
                continue
            for site, sg in by_sub.get(sub_id, []):
                if site.mode == "map":
                    self.rewrite_site(fe, site, index=f"{CIDX[sg]}({site.sub0})")
                else:
                    self.rewrite_site(fe, site, k0=K0[sg])
            subgrids = {sg for _, sg in by_sub.get(sub_id, [])}
            if sub_id in callers:
                subgrids |= {poly_subgrid[p] for p, poly in polymorphic.items() if any(c == sub_id for c, _ in poly.compact_calls)}
            self.prologue(fe, sub, subgrids)
        for sub_id, poly in polymorphic.items():
            for caller, ln in sorted(poly.compact_calls):
                fe = self.file(self.subs[caller].filepath)
                self.append_argument(fe, self.subs[caller], ln, f"{OFF}={K0[poly_subgrid[sub_id]]}")

    def rewrite_polymorphic(self, fe: FileEdits, sub, sites) -> None:
        """
        An optional `compact_off` dummy; the compact-capable dummies are
        indexed with `kc`: `compact_off + fp` if present, else `p` as before.
        """
        rec = sub.record
        for name in (OFF, KC):
            if rec.declaration(name) is not None or known(rec, name):
                raise RewriteError(f"{sub.id} already has a symbol {name}")
        if any(s.mode != "filter" for s in sites):
            raise RewriteError(f"{sub.id}: a routine taking both layouts cannot have map-indexed sites")
        gathers = {(s.sub0, s.fp) for s in sites}
        if len(gathers) != 1:
            raise RewriteError(f"{sub.id}: sites gather with {sorted(gathers)}; one gather expected")
        var, fp = gathers.pop()
        assigns = [
            e for e in rec.events
            if isinstance(e, AssignEvent) and e.lhs.path == var and not e.lhs.subscripts
            and e.reads and e.reads[-1].subs == (fp,)
        ]
        if len(assigns) != 1:
            raise RewriteError(f"{sub.id}: expected one `{var} = filter({fp})`, found {len(assigns)}")
        for site in sites:
            self.rewrite_site(fe, site, index=KC)
        self.append_argument(fe, sub, sub.startline, OFF)
        last = statement_end(fe.lines, rec.last_decl_ln)
        fe.after[last].extend(
            [
                f"     integer, intent(in), optional :: {OFF}  ! clump offset if the flux arrays are compact  {TAG}",
                f"     integer :: {KC}  ! index of the flux arrays  {TAG}",
            ]
        )
        indent = re.match(r"\s*", fe.lines[assigns[0].ln]).group(0)
        fe.after[statement_end(fe.lines, assigns[0].ln)].extend(
            [
                f"{indent}{KC} = {var}  {TAG}",
                f"{indent}if (present({OFF})) {KC} = {OFF} + {fp}  {TAG}",
            ]
        )

    def tree(self, sub: Subroutine) -> SubroutineDefinitionConstruct:
        """The routine's parse tree with source spans (cached)."""
        if self._routines is None:
            self._routines = {}
        if sub.id not in self._routines:
            try:
                # to the END statement: `endline` stops at the routine's own `contains`, if any
                end = sub.end_stmt_ln or sub.endline
                self._routines[sub.id] = parse_routine(sub.filepath, sub.startline, end, keep_spans=True)
            except BaseException as err:  # the parser exits on fatal errors
                raise RewriteError(f"{sub.id}: does not parse: {err}")
        return self._routines[sub.id]

    def site_tree(self, fe: FileEdits, site, end: int):
        """The tree of the site's routine; without an analysis (snippets), of its statement alone."""
        subs = self.subs
        if subs is not None and site.sub in subs:
            return self.tree(subs[site.sub])
        lines = [LineTuple(line=fe.lines[i], ln=i) for i in range(statement_start(fe.lines, site.ln), end + 1)]
        return SimpleNamespace(body=Parser(lines=lines, keep_spans=True).parse_program())

    def append_argument(self, fe: FileEdits, sub, ln: int, arg: str) -> None:
        """
        Add `arg` as the last argument of the call on line `ln` of `sub`, or of
        its header if `ln` is the routine's first line.
        """
        tree = self.tree(sub)
        if ln == sub.startline:
            close = tree.args_close
        else:
            calls = [s.function.span for s, _ in iter_statements(tree.body)
                     if isinstance(s, SubCallStatement) and s.function.span and s.function.span[0] == ln]
            if len(calls) != 1:
                raise RewriteError(f"{fe.path}:{ln + 1}: expected one call statement for `{arg}`, found {len(calls)}")
            l1, c1 = calls[0][2:]
            close = (l1, c1 - 1, l1, c1)
        if close is None or fe.lines[close[0]][close[1]:close[3]] != ")":
            raise RewriteError(f"{fe.path}:{ln + 1}: no closing parenthesis for `{arg}`")
        fe.replace_cols(close[0], close[1], close[1], f", {arg}")

    def rewrite_site(self, fe: FileEdits, site, index: str | None = None, k0: str = "k0p") -> None:
        """
        `name(sub0, ...)` -> `name(index, ...)` (or `name(k0+fp, ...)`) for every
        reference to the site's designator with that first subscript in its statement.
        """
        repl = f"({index}" if index else f"({k0}+{site.fp}"
        end = statement_end(fe.lines, site.ln)
        want = (site.written.lower(), site.sub0.replace(" ", "").lower())
        hits = 0
        for name_end, first in subscripted_designators(self.site_tree(fe, site, end), site.ln, end):
            path, arg = first
            if (path, span_text(fe.lines, arg.span).replace(" ", "").lower()) != want:
                continue
            (l0, c0), (l1, c1) = name_end, arg.span[2:]
            if l0 != l1:
                raise RewriteError(f"{fe.path}:{site.ln + 1}: `{site.written}({site.sub0}` spans lines")
            code = fe.lines[l1]
            while c1 < len(code) and code[c1] in " \t":  # up to the `,` or `)`
                c1 += 1
            fe.replace_cols(l0, c0, c1, repl)
            hits += 1
        if not hits:
            raise RewriteError(
                f"{fe.path}:{site.ln + 1}: no `{site.written}({site.sub0}` in the statement "
                f"(sources changed since the analysis?)"
            )

    def prologue(self, fe: FileEdits, sub, subgrids, extra: tuple[str, ...] = ()) -> None:
        """
        `use CompactLayoutMod` and a local copy of the clump offset per subgrid
        level the routine indexes (`k0p = compact_soilp_off`). `extra`: more
        names to import (the zeroing routines).
        """
        rec = sub.record
        done = self._prologued[sub.id]
        new = sorted(set(subgrids) - done)
        if not new and not extra:
            return
        header = statement_end(fe.lines, sub.startline)
        last = statement_end(fe.lines, rec.last_decl_ln) if rec.last_decl_ln >= 0 else header
        if not done:
            names = list(OFFSET.values()) + list(CIDX.values()) + list(extra)
            fe.after[header].append(f"    use CompactLayoutMod, only : {', '.join(names)}  {TAG}")
        elif extra:
            fe.after[header].append(f"    use CompactLayoutMod, only : {', '.join(extra)}  {TAG}")
        for sg in new:
            k0 = K0[sg]
            if rec.declaration(k0) is not None or known(rec, k0):
                raise RewriteError(f"{sub.id} already has a symbol {k0}")
            # declarations first (a routine may index both levels, in one or several calls)
            fe.after[last].insert(
                0, f"    integer :: {k0}  ! clump offset: compact index of filter_{MAIN[sg]}(f) is {k0}+f  {TAG}"
            )
            fe.after[last].append(f"    {k0} = {OFFSET[sg]}  {TAG}")
        done.update(new)

    # --- the derived type ---
    def holder_of(self, sel: Selection) -> Holder:
        if sel.holder is None:
            raise RewriteError(f"{sel.dtype}: selection without a holder")
        return sel.holder

    def rewrite_type(self, sel: Selection) -> None:
        holder = self.holder_of(sel)
        base = holder.type
        if "cgrp_id" in self.fut.type_dict[base].components:
            raise RewriteError(f"{base} already has a cgrp_id component: already rewritten?")
        groups = self.groups(sel)
        fe = self.file(holder.dtype.filepath)
        self.add_module_use(
            fe, "CompactLayoutMod",
            ["compact_groups", "compact_group_new", "compact_lev_patch", "compact_lev_col"],
        )
        _, last, end = self.type_span(fe, base)
        fe.after[last].append(
            f"    integer :: cgrp_id({len(groups)}) = 0  ! compact groups of some fields above {TAG}"
        )
        fe.after[end].append(f"  public :: {base}_compact_bind  {TAG}")
        mod_end = self.module_tree(fe).end_ln
        fe.before[mod_end].extend(self.bind_procedure(base, sel, groups))

        # Init: drop the allocations, bind after them
        for plan in self.init_plans[sel.dtype]:
            init = self.subs[plan.sub_id]
            fe_i = self.file(init.filepath)
            fe_i.deletes |= plan.deletes
            if plan.insert_after < 0:
                continue  # only re-initializations dropped (init_history)
            this = f"this%{plan.path}" if plan.path else "this"
            fe_i.after[plan.insert_after].append(f"{plan.indent}call {base}_compact_bind({this})  {TAG}")
            if fe_i.path != fe.path:
                self.add_module_use(fe_i, self.module_name(fe), [f"{base}_compact_bind"])

        # zeroing: the shared routine of a base type and/or the concrete SetValues routines
        # (which may zero holder fields through `this%core%x`); every field is zeroed where
        # its assignments are, and every selected field must be zeroed somewhere
        zeroed: set[str] = set()
        if "setvalues" in holder.routines:
            zeroed |= self.rewrite_zeroing(self.subs[holder.routines["setvalues"]], "", sel, groups)
        seen: set[str] = set()
        for emb in holder.embeddings:
            sid = self.fut.type_dict[emb.concrete].procedures.get("setvalues")
            if not isinstance(sid, str):
                if "setvalues" in holder.routines:
                    continue
                raise RewriteError(f"{emb.concrete} has no SetValues procedure")
            if sid in seen or sid == holder.routines.get("setvalues"):
                continue
            seen.add(sid)
            zeroed |= self.rewrite_zeroing(self.subs[sid], emb.path, sel, groups)
        missing = set(sel.components) - zeroed
        if missing:
            raise RewriteError(f"{base}: no zeroing lines found for {sorted(missing)}")

    def type_span(self, fe: FileEdits, type_name: str) -> tuple[int, int, int]:
        """(first line, last line of the last component, `end type` line) of the type definition."""
        td = next(
            (s for s in self.module_tree(fe).spec.statements
             if isinstance(s, TypeDef) and str(s.name).lower() == type_name.lower()),
            None,
        )
        if td is None:
            raise RewriteError(f"{fe.path}: no definition of type {type_name}")
        members = td.members.statements if td.members is not None else []
        last = statement_end(fe.lines, members[-1].lineno) if members else td.lineno
        return td.lineno, last, td.end_ln

    def bind_procedure(self, base: str, sel: Selection, groups: list[Group]) -> list[str]:
        """`<base>_compact_bind(this)`: create the groups on first use, point the fields at them."""
        level = LEVEL[sel.subgrid]
        width = max(len(c) for g in groups for c in g.fields)
        body: list[str] = []
        for g in groups:
            nlev = f", nlev={g.nlev}" if g.nlev else ""
            body.append(
                f"    if (this%cgrp_id({g.index}) == 0) this%cgrp_id({g.index}) = "
                f"compact_group_new('{g.name(base)}', {level}, {len(g.fields)}{nlev})"
            )
            for slot, comp in enumerate(g.fields, 1):
                if g.nlev:
                    body.append(f"    this%{comp.ljust(width)} => compact_groups(this%cgrp_id({g.index}))%data3(:, :, {slot})")
                else:
                    body.append(f"    this%{comp.ljust(width)} => compact_groups(this%cgrp_id({g.index}))%data(:, {slot})")
        return [
            "",
            "  !------------------------------------------------------------------------",
            f"  subroutine {base}_compact_bind(this)",
            "    !",
            "    ! !DESCRIPTION:",
            "    ! Point the compact fields at their group buffers; the groups are created",
            "    ! on the first call (generated by spel compact rewrite)",
            "    !",
            "    ! !ARGUMENTS:",
            f"    type({base}), intent(inout) :: this",
            "    !------------------------------------------------------------------------",
            *body,
            f"  end subroutine {base}_compact_bind",
        ]

    # --- restart ---
    def restart_routines(self, sel: Selection) -> list[tuple[str, str]]:
        """(routine id, holder path) of the restart routines that may register the holder's fields."""
        holder = self.holder_of(sel)
        out: list[tuple[str, str]] = []
        if "restart" in holder.routines:
            out.append((holder.routines["restart"], ""))
        seen: set[str] = set()
        for emb in holder.embeddings:
            sid = self.fut.type_dict[emb.concrete].procedures.get("restart")
            if isinstance(sid, str) and sid in self.subs and sid not in seen and sid not in holder.routines.values():
                seen.add(sid)
                out.append((sid, emb.path))
        return out

    def plan_restart(self, sel: Selection) -> None:
        """
        Find the `restartvar(..., data=this%<path>%x)` calls of the selected restart
        fields. Each becomes: gather the field into a full-layout temporary, the call
        on the temporary, scatter back on 'read'. A restart field whose call passes a
        section, or that the restart routine touches anywhere else, or whose call is
        not found, is dropped from the selection.
        """
        holder = self.holder_of(sel)
        restart_comps = {c for c in sel.components if sel.restart.get(c)}
        comps = set(sel.components)
        plans: list[RestartPlan] = []
        touched: dict[str, str] = {}
        handled: set[str] = set()
        for sub_id, path in self.restart_routines(sel):
            sub = self.subs[sub_id]
            fe = self.file(sub.filepath)
            tree = parse_routine(sub.filepath, sub.startline, sub.endline)
            prefix = f"this%{path}%" if path else "this%"
            paths = {f"{prefix}{c}": c for c in comps}
            calls: list[tuple[int, int, str, int]] = []
            call_lines: set[int] = set()
            for stmt, _ in iter_statements(tree.body):
                if not (isinstance(stmt, SubCallStatement) and str(stmt.function.function).lower() == RESTART_CALL):
                    continue
                data = _keyword_args(stmt).get("data")
                text = designator_text(data) if data is not None else None
                comp = paths.get(text or "")
                if comp is None:
                    continue
                try:
                    section = bool(flatten_designator(data)[-1][1])
                except Exception:
                    section = True
                if section:
                    touched.setdefault(comp, f"{fe.path.name}:{stmt.lineno + 1} (restartvar of a section)")
                    continue
                rank = 2 if comp in sel.levels else 1
                calls.append((stmt.lineno, statement_end(fe.lines, stmt.lineno), comp, rank))
                call_lines.add(stmt.lineno)
                handled.add(comp)
            for stmt, _ in iter_statements(tree.body):
                if stmt.lineno in call_lines:
                    continue
                for d in designators_in(stmt):
                    for p, c in paths.items():
                        if d == p or d.startswith(p + "%"):
                            touched.setdefault(c, f"{fe.path.name}:{stmt.lineno + 1} (restart routine)")
            if calls:
                plans.append(RestartPlan(sub_id, path, calls))
        for comp in sorted(restart_comps - handled):
            touched.setdefault(comp, "restartvar call not found in the restart routines")
        for comp, where in sorted(touched.items()):
            sel.components.pop(comp, None)
            sel.history.pop(comp, None)
            sel.levels.pop(comp, None)
            sel.restart.pop(comp, None)
            sel.excluded[comp] = f"restart: {where}"
            self.notes.append(f"{sel.dtype}: {comp} dropped: {where}")
        if touched:
            sel.sites = [s for s in sel.sites if self.site_component(holder, s) not in touched]
        self.restart_plans[sel.dtype] = plans

    def rewrite_restart(self, sel: Selection) -> None:
        """Apply the restart plans: gather before each call, scatter after it on 'read'."""
        for plan in self.restart_plans.get(sel.dtype, []):
            sub = self.subs[plan.sub_id]
            fe = self.file(sub.filepath)
            tree = self.tree(sub)
            args = [str(a).lower() for a in tree.args]
            if "flag" not in args:
                raise RewriteError(f"{sub.id}: no `flag` dummy (the restart read/write flag)")
            this = f"this%{plan.path}" if plan.path else "this"
            ranks: set[int] = set()
            for ln, end, comp, rank in plan.calls:
                if comp not in sel.components:
                    continue
                tmp = f"spel_rst{rank}"
                hits = [
                    (a.span[:2], name_end(a.value)) for a in data_arguments(tree, ln, end)
                    if designator_text(a.value) == f"{this}%{comp}".lower()
                ]
                if len(hits) != 1 or hits[0][1] is None or hits[0][0][0] != hits[0][1][0]:
                    raise RewriteError(f"{fe.path}:{ln + 1}: `data={this}%{comp}` not found once in the restartvar call")
                (l0, c0), (_, c1) = hits[0]
                fe.replace_cols(l0, c0, c1, f"data={tmp}")
                indent = re.match(r"\s*", fe.lines[ln]).group(0)
                fe.before[ln].append(f"{indent}call compact_gather{rank}({this}%{comp}, {tmp})  {TAG}")
                fe.after[end].extend(
                    [
                        f"{indent}if (trim(flag) == 'read') call compact_scatter{rank}({tmp}, {this}%{comp})  {TAG}",
                        f"{indent}deallocate({tmp})",
                    ]
                )
                ranks.add(rank)
            if not ranks:
                continue
            decls = [st for st in tree.body.statements if isinstance(st, VariableDecl)]
            if not decls:
                raise RewriteError(f"{sub.id}: no declarations to append the restart temporaries to")
            last = statement_end(fe.lines, max(st.lineno for st in decls))
            for rank in sorted(ranks):
                dims = ":" if rank == 1 else ":,:"
                fe.after[last].append(
                    f"    real(r8), pointer :: spel_rst{rank}({dims}) => null()  ! full-layout copy of a compact field  {TAG}"
                )
            names = [f"compact_gather{r}" for r in sorted(ranks)] + [f"compact_scatter{r}" for r in sorted(ranks)]
            self.add_routine_use(fe, sub, "CompactLayoutMod", names)

    # --- Init ---
    def plan_inits(self, sel: Selection) -> None:
        """
        Find the `allocate(this%x(beg:end)); this%x(:) = spval` lines of the
        compact fields in the Init routines (the holder's own `<base>_init`,
        else the concrete Inits) and where the bind call goes. A field the Init
        touches anywhere else is dropped from the selection (with a note): the
        rewrite doesn't know what that code expects of the full layout.
        """
        holder = self.holder_of(sel)
        routines: list[tuple[str, str]] = []
        if "init" in holder.routines:
            routines.append((holder.routines["init"], ""))
            # the base type's registrations re-initialize the fields over the full range first
            if "init_history" in holder.routines:
                routines.append((holder.routines["init_history"], ""))
        # the concrete Inits: they allocate a role holder's fields, and may re-initialize or
        # register a base holder's fields through `this%core%x`
        seen: set[str] = set()
        for emb in holder.embeddings:
            init_id = self.fut.type_dict[emb.concrete].procedures.get("init")
            if not isinstance(init_id, str):
                if "init" in holder.routines:
                    continue
                raise RewriteError(f"{emb.concrete} has no Init procedure")
            if init_id not in seen and init_id not in holder.routines.values():
                seen.add(init_id)
                routines.append((init_id, emb.path))
        comps = set(sel.components)
        allocated: set[str] = set()
        touched: dict[str, str] = {}
        plans: list[InitPlan] = []
        for sub_id, path in routines:
            init = self.subs[sub_id]
            fe = self.file(init.filepath)
            tree = parse_routine(init.filepath, init.startline, init.endline)
            prefix = f"this%{path}%" if path else "this%"
            by_line: dict[int, list[tuple[str, str]]] = defaultdict(list)  # line -> [(kind, comp)]
            others: dict[int, set[str]] = defaultdict(set)
            for stmt, _ in iter_statements(tree.body):
                ln = stmt.lineno
                if isinstance(stmt, AllocateStatement) and not stmt.is_deallocate:
                    for obj in stmt.objects:
                        text = designator_text(obj)
                        comp = text[len(prefix):] if text and text.startswith(prefix) else None
                        if comp in comps:
                            by_line[ln].append(("allocate", comp))
                        else:
                            others[ln].add(text or str(obj))
                elif (lhs := assignment_lhs(stmt)) is not None:
                    text = designator_text(lhs)
                    comp = text[len(prefix):] if text and text.startswith(prefix) else None
                    if comp in comps and ln in by_line and ("allocate", comp) in by_line[ln]:
                        by_line[ln].append(("assign", comp))
                    elif comp in comps and is_spval_fill(stmt):
                        # a redundant `this%x(begp:endp) = spval` before a registration: the
                        # group buffer holds spval already
                        by_line[ln].append(("reinit", comp))
                    else:
                        others[ln].add(text or "?")
                else:
                    others[ln].add(type(stmt).__name__)
            deletes: set[int] = set()
            last_alloc = -1
            for ln, items in sorted(by_line.items()):
                if statement_end(fe.lines, ln) != ln or (ln in others and others[ln]):
                    raise RewriteError(
                        f"{fe.path}:{ln + 1}: allocation line of {items[0][1]} holds other statements: "
                        f"{fe.lines[ln].strip()}"
                    )
                allocated.update(c for kind, c in items if kind == "allocate")
                if any(kind == "allocate" for kind, _ in items):
                    last_alloc = ln
                deletes.add(ln)
            # any other use of a compact field in this routine (any statement, any line)
            paths = {f"{prefix}{c}": c for c in comps}
            aliases: dict[str, str] = {}  # pointer name -> compact component it points at (a level slice)
            for stmt, _ in iter_statements(tree.body):
                if stmt.lineno in deletes:
                    continue
                if isinstance(stmt, SubCallStatement) and str(stmt.function.function).lower() in REGISTRATION_CALLS:
                    continue  # history registrations are served by the compact path; restart fields are excluded
                target = pointer_target(stmt)
                if target is not None:
                    # `data1dptr => this%core%x(:, k)`: a level slice registered as a rank-1 history
                    # field (compact_locate finds it in the group); a write through the pointer is a touch
                    ptr, tgt = target
                    text = designator_text(tgt)
                    comp = paths.get(text or "")
                    if comp is not None:
                        aliases[designator_text(ptr) or ""] = comp
                        continue
                    aliases.pop(designator_text(ptr) or "", None)
                lhs = assignment_lhs(stmt)
                if lhs is not None:
                    base = (designator_text(lhs) or "").split("%")[0]
                    if base in aliases:
                        touched.setdefault(aliases[base], f"{fe.path.name}:{stmt.lineno + 1} (through {base})")
                for d in designators_in(stmt):
                    for p, c in paths.items():
                        if d == p or d.startswith(p + "%"):
                            touched.setdefault(c, f"{fe.path.name}:{stmt.lineno + 1}")
            if not deletes:
                continue
            last = last_alloc
            indent = re.match(r"\s*", fe.lines[last]).group(0) if last >= 0 else ""
            plans.append(InitPlan(sub_id, path, deletes, last, indent))
            if last < 0:
                continue
            # history registrations of the compact fields must follow the bind
            for reg in registrations(tree):
                if reg.base == "this" and reg.field.startswith(prefix[len("this%"):]) and reg.ln < last:
                    comp = reg.field[len(prefix) - len("this%"):]
                    if comp in comps:
                        raise RewriteError(
                            f"{fe.path}:{reg.ln + 1}: {comp} is registered before its allocations end"
                        )
        missing = comps - allocated
        for comp in sorted(missing):
            touched.setdefault(comp, "no allocate line found in the Init routines")
        for comp, where in sorted(touched.items()):
            sel.components.pop(comp, None)
            sel.history.pop(comp, None)
            sel.excluded[comp] = f"Init touches it outside its allocation ({where})"
            self.notes.append(f"{sel.dtype}: {comp} dropped: Init touches it outside its allocation ({where})")
        if touched:
            sel.sites = [s for s in sel.sites if self.site_component(holder, s) not in touched]
            self._replan_lines(sel, plans, touched)
        self.init_plans[sel.dtype] = plans

    @staticmethod
    def site_component(holder: Holder, site) -> str:
        """The holder component a site accesses (`veg_cf%core%agnpp` -> `agnpp`)."""
        found = holder.split(site.path)
        return found[1] if found is not None else site.path.split("%")[-1]

    def _replan_lines(self, sel: Selection, plans: list[InitPlan], dropped: dict[str, str]) -> None:
        """Keep the allocation lines of dropped fields: recompute each plan's deletes."""
        for plan in plans:
            init = self.subs[plan.sub_id]
            fe = self.file(init.filepath)
            prefix = f"this%{plan.path}%" if plan.path else "this%"
            keep = set()
            for ln in plan.deletes:
                code = fe.lines[ln].lower().replace(" ", "")
                if any(f"{prefix}{c}(" in code for c in dropped):
                    keep.add(ln)
            plan.deletes -= keep
            allocs = [ln for ln in plan.deletes if fe.lines[ln].lower().lstrip().startswith("allocate")]
            if allocs and plan.insert_after >= 0:
                plan.insert_after = max(allocs)
                plan.indent = re.match(r"\s*", fe.lines[plan.insert_after]).group(0)

    # --- zeroing ---
    def rewrite_zeroing(self, setvalues, path: str, sel: Selection, groups: list[Group]) -> set[str]:
        """
        In a zeroing routine: drop the assignments of the compact fields
        (`this%<path>%x(idx) = value`) and zero those fields by group before
        `end subroutine`. Returns the components the routine zeroes (none: no edit).
        """
        fe = self.file(setvalues.filepath)
        this = f"this%{path}" if path else "this"
        zero_lines = self.zero_lines(setvalues, path)
        present = [c for c in sel.components if zero_lines.get(c)]
        if not present:
            return set()
        assigned = {
            (e.ln, e.lhs.path) for e in setvalues.record.events if isinstance(e, AssignEvent)
        }
        ranges: dict[str, str] = {}  # rank-2 component -> the levels its zeroing writes (`1:nlevdecomp_full`)
        for comp in present:
            for ln in zero_lines.get(comp, []):
                # one assignment statement on one line, of this component alone
                if statement_end(fe.lines, ln) != ln or (ln, f"{this}%{comp}") not in assigned:
                    raise RewriteError(
                        f"{fe.path}:{ln + 1}: unexpected zeroing line for {comp}: {fe.lines[ln].strip()}"
                    )
                fe.deletes.add(ln)
                if comp in sel.levels:
                    rng = self.level_range(setvalues, ln, f"{this}%{comp}")
                    if ranges.setdefault(comp, rng) != rng:
                        raise RewriteError(f"{setvalues.id}: {comp} zeroed over different level ranges")
        sv_end = self.routine_end(setvalues)
        if sv_end != setvalues.endline:
            raise RewriteError(f"{fe.path}:{sv_end + 1}: not the end of {setvalues.id}")
        dummies = [d.name for d in setvalues.record.dummy_args]
        if len(dummies) < 4:
            raise RewriteError(f"{setvalues.id}: expected (this, num, filter, value) dummies, got {dummies}")
        num, value = dummies[1], dummies[3]
        here = [
            Group(g.index, g.guard, g.key, [c for c in g.fields if c in present], g.nlev)
            for g in groups if any(c in present for c in g.fields)
        ]
        fe.before[sv_end].extend(self.zero_block(here, num, value, path, sel.subgrid, ranges))
        self.prologue(fe, setvalues, {sel.subgrid}, extra=("compact_bound", "compact_group_fill"))
        return set(present)

    @staticmethod
    def level_range(setvalues, ln: int, lhs_path: str) -> str:
        """
        `lo:hi` of the levels a rank-2 zeroing line writes: the range of the
        enclosing loop over its second subscript (`do j = 1, nlevdecomp_full`
        for `x(idx, j) = value`); `:` when the subscript is not a loop index.
        """
        rec = setvalues.record
        ev = next((e for e in rec.events if isinstance(e, AssignEvent) and e.ln == ln and e.lhs.path == lhs_path), None)
        if ev is None or len(ev.lhs.subscripts) != 2:
            raise RewriteError(f"{setvalues.id}:{ln + 1}: rank-2 zeroing without two subscripts")
        j = str(ev.lhs.subscripts[1]).strip().lower()
        loops = [
            lp for lp in rec.all_loops()
            if lp.index is not None and lp.index.lower() == j and lp.start_ln <= ln <= lp.end_ln
        ]
        if not loops:
            return ":"
        node = max(loops, key=lambda lp: lp.start_ln).node  # the innermost
        if node.step is not None:
            raise RewriteError(f"{setvalues.id}:{ln + 1}: strided level loop in the zeroing")
        return f"{node.start}:{node.end}".replace(" ", "")

    def zero_block(self, groups: list[Group], num: str, value: str, path: str, subgrid: str,
                   ranges: Optional[dict[str, str]] = None) -> list[str]:
        """
        Explicit assignments (visible to SPEL's analysis): a contiguous fill of
        each clump's compact range once the layout is bound; before that (the
        special-landunit call in Init) the value becomes the groups' fill.
        `path`: the component of `this` that holds the fields (`retrans`).
        """
        this = f"this%{path}" if path else "this"
        k0 = K0[subgrid]

        def guarded(guard, body, indent):
            if guard is None:
                return [f"{indent}{line}" for line in body]
            cond = guard if guard.startswith("(") and guard.count("(") == 1 else f"({guard})"
            return [f"{indent}if {cond} then", *(f"{indent}   {line}" for line in body), f"{indent}end if"]

        ranges = ranges or {}

        def section(c: str) -> str:
            return f"({k0}+1:{k0}+{num}, {ranges.get(c, ':')})" if c in ranges else f"({k0}+1:{k0}+{num})"

        out = [f"    ! compact fields, by group  {TAG}", "    if (compact_bound) then"]
        for g in groups:
            out += guarded(g.guard, [f"{this}%{c}{section(c)} = {value}" for c in g.fields], "       ")
        out.append("    else  ! not bound yet: the special-landunit call of Init; history reads this on those points")
        for g in groups:
            out += guarded(g.guard, [f"call compact_group_fill({this}%cgrp_id({g.index}), {value})"], "       ")
        out.append(f"    end if  {TAG}")
        return out

    def zero_lines(self, setvalues, path: str = "") -> dict[str, list[int]]:
        """component -> lines assigning `this%<path>%component` in the routine."""
        want = f"this%{path}%" if path else "this%"
        out: dict[str, list[int]] = defaultdict(list)
        for e in setvalues.record.accesses(status="w"):
            if not e.path.startswith(want):
                continue
            comp = e.path[len(want):]
            if "%" not in comp and e.ln not in out[comp]:
                out[comp].append(e.ln)
        return out

    # --- the bind module, driver, initialization, clump bounds ---
    def instance_module(self, sel: Selection, inst: str) -> str:
        holder = self.holder_of(sel)
        ctype = next((e.concrete for e in holder.embeddings if e.instance == inst), sel.dtype)
        return self.fut.type_dict[ctype].instances[inst].declaration

    def generate_bind_module(self) -> str:
        uses: dict[str, list[str]] = defaultdict(list)
        binds: list[str] = []
        for sel in self.sels:
            holder = self.holder_of(sel)
            fe = self.file(holder.dtype.filepath)
            proc = f"{holder.type}_compact_bind"
            if proc not in uses[self.module_name(fe)]:
                uses[self.module_name(fe)].append(proc)
            for emb in holder.embeddings:
                mod = self.instance_module(sel, emb.instance)
                if emb.instance not in uses[mod]:
                    uses[mod].append(emb.instance)
                binds.append(f"    call {proc}({emb.prefix})")
        use_lines = "\n".join(f"  use {m}, only : {', '.join(n)}" for m, n in sorted(uses.items()))
        return f"""module {BIND_MODULE}
  !-----------------------------------------------------------------------
  ! Generated by spel compact rewrite. Binds the compact flux fields to their
  ! group buffers over the process soil filters, and rebinds them when the
  ! filters change (dynamic subgrid). Nothing is preserved across a rebind:
  ! every compact field is zeroed before it is read in a step.
  !-----------------------------------------------------------------------
  use shr_kind_mod    , only : r8 => shr_kind_r8
  use decompMod       , only : bounds_type, get_proc_clumps
  use filterMod       , only : filter
  use LandunitType    , only : lun_pp
  use VegetationType  , only : veg_pp
  use ColumnType      , only : col_pp
  use CompactLayoutMod, only : compact_filters_set, compact_group_resize, compact_ngroups, &
       compact_lev_patch, compact_lev_col, compact_bound
{use_lines}
  implicit none
  private
  public :: compact_filters_set_from_filters  ! the process filters from the clumps' filters
  public :: compact_filters_updated           ! the above, then resize the groups and rebind
contains

  subroutine compact_filters_set_from_filters(bounds)
    type(bounds_type), intent(in) :: bounds   ! process bounds
    integer :: nc, nclumps, k, p, c
    integer, allocatable :: nump(:), numc(:), idxp(:), idxc(:)
    logical, allocatable :: specp(:), specc(:)

    nclumps = get_proc_clumps()
    allocate(nump(nclumps), numc(nclumps))
    do nc = 1, nclumps
       nump(nc) = filter(nc)%num_soilp
       numc(nc) = filter(nc)%num_soilc
    end do
    allocate(idxp(max(sum(nump), 1)), idxc(max(sum(numc), 1)))
    k = 0
    do nc = 1, nclumps
       idxp(k+1:k+nump(nc)) = filter(nc)%soilp(1:nump(nc))
       k = k + nump(nc)
    end do
    k = 0
    do nc = 1, nclumps
       idxc(k+1:k+numc(nc)) = filter(nc)%soilc(1:numc(nc))
       k = k + numc(nc)
    end do
    allocate(specp(bounds%begp:bounds%endp), specc(bounds%begc:bounds%endc))
    do p = bounds%begp, bounds%endp
       specp(p) = lun_pp%ifspecial(veg_pp%landunit(p))
    end do
    do c = bounds%begc, bounds%endc
       specc(c) = lun_pp%ifspecial(col_pp%landunit(c))
    end do
    call compact_filters_set(compact_lev_patch, bounds%begp, bounds%endp, nump, idxp, specp)
    call compact_filters_set(compact_lev_col, bounds%begc, bounds%endc, numc, idxc, specc)
  end subroutine compact_filters_set_from_filters

  subroutine compact_filters_updated(bounds)
    type(bounds_type), intent(in) :: bounds   ! process bounds
    integer :: g

    call compact_filters_set_from_filters(bounds)
    do g = 1, compact_ngroups
       call compact_group_resize(g)
    end do
{chr(10).join(binds)}
    compact_bound = .true.
  end subroutine compact_filters_updated

end module {BIND_MODULE}
"""

    def rewrite_driver(self) -> None:
        """After `t_stopf('dyn_subgrid')`: resize and rebind when the filters may have changed."""
        drv = self.subs[DRIVER_ID]
        fe = self.file(drv.filepath)
        calls = [c for c in drv.record.calls if c.name.lower() == DYN_SUBGRID]
        if len(calls) != 1:
            raise RewriteError(f"elm_drv: expected one call to {DYN_SUBGRID}, found {len(calls)}")
        for name in ("nstep_mod", "first_restart_step_mod", "bounds_proc"):
            if not known(drv.record, name):
                raise RewriteError(f"elm_drv: {name} is not visible (the resize condition needs it)")
        end = statement_end(fe.lines, calls[0].ln)
        nxt = end + 1
        while nxt < len(fe.lines) and not fe.lines[nxt].split("!")[0].strip():
            nxt += 1
        if fe.lines[nxt].split("!")[0].strip().lower().replace(" ", "").startswith("callt_stopf("):
            end = statement_end(fe.lines, nxt)  # after the timer that closes the dyn_subgrid step
        indent = re.match(r"\s*", fe.lines[calls[0].ln]).group(0)
        fe.after[end].extend(
            [
                f"{indent}! the soil filters may have changed: resize the compact groups and rebind  {TAG}",
                f"{indent}if ({RESIZE_CONDITION}) then",
                f"{indent}   call compact_filters_updated(bounds_proc)",
                f"{indent}end if  {TAG}",
            ]
        )
        if not known(drv.record, "get_subgrid_weights_updated"):
            self.add_routine_use(fe, drv, "dynSubgridControlMod", ["get_subgrid_weights_updated"])
        self.add_routine_use(fe, drv, BIND_MODULE, ["compact_filters_updated"])

    def rewrite_initialize(self) -> None:
        """
        initialize2: the process filters before the types are initialized
        (`elm_inst_biogeochem`), the bind after the final reweight (its
        `!$OMP END PARALLEL DO`).
        """
        from spel.transforms.core.query import outside_routine

        # elm_drv doesn't reach initialize2: parse it from its module when the analysis lacks it
        sub = self.subs.get("elm_initializemod::initialize2") or outside_routine("elm_initializemod::initialize2")
        if sub is None:
            raise RewriteError("elm_initializemod::initialize2 not found")
        fe = self.file(sub.filepath)
        tree = parse_routine(sub.filepath, sub.startline, sub.endline)
        inst = [
            s for s, _ in iter_statements(tree.body)
            if isinstance(s, SubCallStatement) and str(s.function.function).lower() == INIT_BIOGEOCHEM
        ]
        if len(inst) != 1 or len(inst[0].function.args) != 1:
            raise RewriteError(f"initialize2: expected one call {INIT_BIOGEOCHEM}(bounds), found {len(inst)}")
        bounds = str(inst[0].function.args[0]).replace(" ", "")
        indent = re.match(r"\s*", fe.lines[inst[0].lineno]).group(0)
        fe.before[inst[0].lineno].extend(
            [
                f"{indent}! the process soil filters, for the compact storage the types bind in Init  {TAG}",
                f"{indent}call compact_filters_set_from_filters({bounds})",
            ]
        )
        # the last reweight_wrapup call and its clump loop
        loops = [s for s in tree.body.statements if isinstance(s, DoLoop)]
        hosting = [
            loop for loop in loops
            if any(isinstance(s, SubCallStatement) and str(s.function.function).lower() == REWEIGHT
                   for s, _ in iter_statements(loop.body))
        ]
        if not hosting:
            raise RewriteError(f"initialize2: no top-level clump loop calling {REWEIGHT}")
        loop = hosting[-1]
        end = loop.end_ln
        nxt = end + 1
        while nxt < len(fe.lines) and not fe.lines[nxt].strip():
            nxt += 1
        if not fe.lines[nxt].strip().lower().replace(" ", "").startswith("!$ompendparalleldo"):
            raise RewriteError(f"{fe.path}:{nxt + 1}: expected `!$OMP END PARALLEL DO` after the reweight loop")
        indent = re.match(r"\s*", fe.lines[loop.lineno]).group(0)
        fe.after[nxt].extend(
            [
                "",
                f"{indent}! final filters: allocate the compact groups for them and bind the fields  {TAG}",
                f"{indent}call compact_filters_updated({bounds})",
            ]
        )
        self.add_routine_use(fe, sub, BIND_MODULE, ["compact_filters_set_from_filters", "compact_filters_updated"])

    def rewrite_decomp(self) -> None:
        for name in DECOMP_ROUTINES:
            sub = self.subs.get(f"decompmod::{name}")
            if sub is None:
                raise RewriteError(f"decompmod::{name} not in the analysis")
            dummies = [d.name for d in sub.record.dummy_args]
            if not dummies:
                raise RewriteError(f"{sub.id}: no dummy arguments (the clump index is the first)")
            fe = self.file(sub.filepath)
            header = statement_end(fe.lines, sub.startline)
            fe.after[header].append(f"     use CompactLayoutMod, only : compact_set_clump  {TAG}")
            end = self.routine_end(sub)
            if end != sub.endline:
                raise RewriteError(f"{fe.path}:{sub.endline + 1}: not the end of {sub.id}")
            fe.before[end].append(f"     call compact_set_clump({dummies[0]})  {TAG}")

    # --- blocked filter loops ---
    def rewrite_blocked_loops(self, sub) -> int:
        """
        `do fi = 1, num, block` around `associate (idx => filter(fi:min(...)))`
        and assignments `x(idx, ...) = ...` becomes `do concurrent (fi = 1:num)`
        with `x(filter(fi), ...) = ...`: the same fields, one element per
        iteration. nvfortran handles the associate-name subscript very slowly (a
        per-statement cost about 1000x the stores). A block the compact rewrite
        emptied is dropped. Returns the loops rewritten.
        """
        from spel.transforms.blocked_loops import find_blocked_loops, name_pattern

        fe = self.file(sub.filepath)
        try:
            loops = find_blocked_loops(sub.record, fe.lines)
        except ValueError as err:
            raise RewriteError(f"{fe.path}:{err}")
        for b in loops:
            fe.deletes.update(b.assoc_lines)
            use = name_pattern(b.name)
            kept = [ln for ln in b.uses if ln not in fe.deletes]
            for ln in kept:
                fe.subs[ln].append((use, f"{b.filter}({b.index})"))
            if kept:
                fe.subs[b.do_ln].append(
                    (re.compile(r"^(\s*).*$"), rf"\g<1>do concurrent ({b.index} = {b.start}:{b.end})  {TAG}")
                )
            else:  # every field of the block is compact now
                fe.deletes.update(range(b.do_ln, b.end_ln + 1))
        return len(loops)

    # --- run ---
    def build(self, loop_types=()) -> None:
        if self.sels:
            for sel in self.sels:
                self.plan_restart(sel)
                self.plan_inits(sel)
            empty = [sel for sel in self.sels if not sel.components]
            for sel in empty:
                self.notes.append(f"{sel.dtype}: no field left to rewrite")
            self.sels = [sel for sel in self.sels if sel.components]
        if self.sels:
            self.rewrite_sites()
            for sel in self.sels:
                self.rewrite_type(sel)
                self.rewrite_restart(sel)
            self.bind_module = self.generate_bind_module()
            self.rewrite_driver()
            self.rewrite_initialize()
            self.rewrite_decomp()
        done: set[str] = set()
        for dtype in loop_types:
            dt = self.fut.type_dict[dtype]
            sids = [dt.procedures.get("setvalues")]
            sids += [base_routine(self.subs, h, "setvalues") for _, h in embedded_holders(self.fut.type_dict, dt)]
            for sid in sids:
                if isinstance(sid, str) and sid in self.subs and sid not in done:
                    done.add(sid)
                    self.notes.append(f"{sid}: {self.rewrite_blocked_loops(self.subs[sid])} blocked loops -> do concurrent")

    def main_dir(self) -> Path:
        return Path(self.subs[DRIVER_ID].filepath).parent

    def diff(self) -> str:
        out = []
        for fe in self.files.values():
            out.extend(
                difflib.unified_diff(
                    [l + "\n" for l in fe.lines],
                    [l + "\n" for l in fe.result()],
                    fromfile=str(fe.path),
                    tofile=str(fe.path),
                )
            )
        if self.bind_module:
            out.append(f"--- /dev/null\n+++ {self.main_dir() / BIND_FILE}\n")
            out.extend(f"+{l}\n" for l in self.bind_module.split("\n"))
        return "".join(out)


def manifest(rewriter: Rewriter, spel_commit: str, e3sm: dict) -> dict:
    types = {}
    for sel in rewriter.sels:
        holder = sel.holder
        types[sel.dtype] = {
            "level": sel.subgrid,
            "instances": sel.instances,
            "embedded_in": {e.instance: e.path for e in holder.embeddings} if holder and holder.embedded else {},
            "groups": [
                {"index": g.index, "guard": g.guard, "fields": g.fields, "rank": 2 if g.nlev else 1, "nlev": g.nlev}
                for g in rewriter.groups(sel)
            ],
            "history": sorted(c for c, h in sel.history.items() if h),
            "restart": sorted(c for c, r in sel.restart.items() if r),
            "sites": len(sel.sites),
        }
    out = {
        "generated_by": "spel compact rewrite",
        "layout": "v2: group buffers over the process soil filters, map-indexed history",
        "spel_commit": spel_commit,
        "e3sm": e3sm,
        "types": types,
        "files": sorted(str(f.path) for f in rewriter.files.values()),
    }
    nm = nutrient_manifest(rewriter.fut)
    if nm is not None:  # the nutrient transform state the rewrite assumed
        out["nutrient"] = {k: nm.get(k) for k in ("spel_commit", "e3sm", "generated_by") if k in nm}
    return out


DEFAULT_TYPES = ["vegetation_carbon_flux", "vegetation_nitrogen_flux", "vegetation_phosphorus_flux",
                 "column_carbon_flux", "column_nitrogen_flux", "column_phosphorus_flux"]


def select_types(fut, types: Optional[list[str]] = None, roles: bool = False, no_map_sites: bool = False,
                 exclude_fields: str = "", fields: str = "", loops_only: bool = False,
                 list_excluded: bool = False) -> tuple[list, list[str]]:
    """
    The field selections of `types` (default `DEFAULT_TYPES`, mapped to their family base types,
    and with `roles` the role holders, on nutrient-transformed sources), narrowed by
    `exclude_fields`/`fields` (comma-separated components); and the report lines.
    """
    import spel.transforms.compact.select as _sel
    from spel.transforms.compact.holders import family_base, nutrient_manifest, role_types

    report: list[str] = []
    type_given = bool(types)
    types = list(types) if type_given else list(DEFAULT_TYPES)
    nm = nutrient_manifest(fut)
    if nm is not None and not type_given:
        # nutrient-transformed sources: the shared fields live on the family base types
        mapped = []
        for dtype in types:
            base = family_base(nm, dtype)
            if base and base not in mapped:
                mapped.append(base)
            elif not base and dtype not in mapped:
                mapped.append(dtype)
        if roles:
            mapped += [r for r in role_types(nm, types) if r not in mapped]
        report.append(f"types: {', '.join(mapped)} (from .spel-nutrient.json)")
        types = mapped
    saved, _sel.MAP_SITES = _sel.MAP_SITES, _sel.MAP_SITES and not no_map_sites
    try:
        sels = []
        for dtype in [] if loops_only else types:
            sel = _sel.select_fields(fut, dtype)
            if exclude_fields:
                for comp in exclude_fields.split(","):
                    if comp in sel.components:
                        sel.components.pop(comp)
                        sel.history.pop(comp, None)
                        sel.levels.pop(comp, None)
                        sel.restart.pop(comp, None)
                        sel.excluded[comp] = "--exclude-fields"
                drop = set(exclude_fields.split(","))
                sel.sites = [st for st in sel.sites if st.components - drop]
            if fields:
                keep = set(fields.split(","))
                for comp in list(sel.components):
                    if comp not in keep:
                        sel.components.pop(comp)
                        sel.history.pop(comp, None)
                        sel.levels.pop(comp, None)
                        sel.excluded[comp] = "not in --fields"
                sel.sites = [st for st in sel.sites if st.components & keep]
                if not sel.components:
                    report.append(f"{dtype}: no selected field in --fields; skipped")
                    continue
            sels.append(sel)
            report.append(f"{dtype}: {len(sel.components)} fields, {len(sel.sites)} access sites")
            nested = [c for c, why in sel.excluded.items() if why.startswith("lives on")]
            if nested:
                report.append(f"  {len(nested)} shared fields live on an embedded holder; pass --type <that type> (see --excluded)")
            for comp, zc in sel.components.items():
                report.append(f"  {comp:32s} zeroed: {zc:20s} history: {sel.history[comp]}")
            if list_excluded:
                for comp, why in sorted(sel.excluded.items()):
                    report.append(f"  excluded {comp:32s} {why}")
            else:
                report.append(f"  ({len(sel.excluded)} fields excluded; --excluded lists them)")
            if sel.untraced:
                report.append("  uses outside the elm_drv trace (text search; fields with a use of this type are excluded):")
                report += [f"    {u}" for u in sel.untraced]
    finally:
        _sel.MAP_SITES = saved
    return sels, report


def core_edits(fe: FileEdits, why: str = "") -> list:
    """A FileEdits as core edits: the same text as `fe.result()`."""
    from spel.transforms.core.edits import insert, replace, replace_cols

    # at one position: the previous line's `after`, this line's `before`, then the line itself
    out = [insert(i + 1, new, order=-2, why=why) for i, new in sorted(fe.after.items()) if new]
    out += [insert(i, new, order=-1, why=why) for i, new in sorted(fe.before.items()) if new]
    for i in sorted({i for i, v in fe.subs.items() if v} | {i for i, v in fe.cols.items() if v} | fe.deletes):
        line = fe.lines[i]
        if i in fe.deletes:
            out.append(replace(i, [], expect=[line], why=why))
        elif not fe.subs.get(i):  # column edits only
            out += [replace_cols(i, c0, c1, new, expect=line[c0:c1], why=why) for (c0, c1), new in sorted(fe.cols[i].items())]
        elif fe.line(i) != line:
            out.append(replace(i, fe.line(i), expect=[line], why=why))
    return out


@dataclass
class Options:
    type: list[str] = field(default_factory=list, metadata={
        "help": "the derived types to rewrite (default: the patch/column C/N/P flux types, or their family base types on nutrient-transformed sources)"})
    roles: bool = field(default=False, metadata={"help": "also the role holders of the default types (nutrient-transformed sources)"})
    loops_only: bool = field(default=False, metadata={"help": "only the SetValues loops, no compact layout"})
    setvalues_loops: bool = field(default=False, metadata={"help": "also turn the flux types' blocked SetValues loops into do concurrent"})
    exclude_fields: str = field(default="", metadata={"help": "comma-separated components to drop from every selection"})
    fields: str = field(default="", metadata={"help": "comma-separated components to keep (only these)"})
    no_map_sites: bool = field(default=False, metadata={"help": "only filter-gathered sites"})
    excluded: bool = field(default=False, metadata={"help": "list every excluded field and why in the report"})


@transform
class CompactRewrite(Transform):
    """
    Matches: the per-step zeroed fields of the selected flux types that every access site
             reaches through a soil filter (`spel compact inventory` explains each verdict).
    Changes: those fields move to per-group compact buffers; access sites, history, restart and
             the SetValues loops are rewritten; CompactBindMod and the manifest are generated
             (and CompactLayoutMod, unless the sources carry it).
    Refuses: sources rewritten already (CompactBindMod or the manifest exist), a carried
             CompactLayoutMod that differs from SPEL's template.
    Design: compact/DESIGN.md.
    """

    name = "compact-rewrite"
    summary = "compact (filter-packed) storage for the per-step zeroed flux fields"
    Options = Options

    def build(self, ctx, opts) -> TPlan:
        from spel.scripts.analysis_cache import git_state
        from spel.transforms.compact.inventory import FLUX_TYPES

        out = TPlan(self.name)
        sels, report = select_types(ctx.fut, opts.type or None, roles=opts.roles, no_map_sites=opts.no_map_sites,
                                    exclude_fields=opts.exclude_fields, fields=opts.fields,
                                    loops_only=opts.loops_only, list_excluded=opts.excluded)
        rw = Rewriter(ctx.fut, sels)
        try:
            rw.build(loop_types=FLUX_TYPES if opts.setvalues_loops else ())
        except RewriteError as err:
            return out.add(Refusal(self.name, str(err)))
        main = rw.main_dir()
        if (main / MANIFEST).exists() or (main / BIND_FILE).exists():
            return out.add(Refusal(str(main), f"already has {BIND_FILE} or {MANIFEST}: already rewritten?"))
        carried = (main / TEMPLATE.name).exists()
        if carried and (main / TEMPLATE.name).read_text() != TEMPLATE.read_text():
            return out.add(Refusal(TEMPLATE.name, f"{main / TEMPLATE.name} differs from SPEL's template {TEMPLATE}"))
        out.notes += report + rw.notes
        for fe in rw.files.values():
            out.add(Change(fe.path, core_edits(fe, self.name), where=fe.path.name))
        spel_root = Path(__file__).resolve().parents[3]
        out.new_files[main / MANIFEST] = json.dumps(
            manifest(rw, git_state(spel_root).get("commit"), git_state(ctx.srcroot)), indent=1) + "\n"
        if sels:
            if not carried:
                out.new_files[main / TEMPLATE.name] = TEMPLATE.read_text()
            out.new_files[main / BIND_FILE] = rw.bind_module
        return out
