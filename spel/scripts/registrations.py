"""
History and restart registrations (`hist_addfld*`, `restartvar` calls).

Those calls are commented out of the analyzed sources so unit tests need not
compile the history/restart machinery; the routines that contain them are
parsed once more from the original files (`find_registrations`, run by the
analysis; the result is `Subroutine.registrations`). A registration through a
temporary pointer (``data2dptr => this%x(:,1:n)`` then ``ptr_col=data2dptr``)
is attributed to the pointer's target.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from spel.scripts.fortran_parser.scope_walk import flatten_designator
from spel.scripts.fortran_parser.spel_ast import (
    AssociateConstruct,
    BlockStatement,
    DoLoop,
    DoWhile,
    ExpressionStatement,
    IfConstruct,
    InfixExpression,
    FuncExpression,
    MacroIf,
    PrefixExpression,
    SelectCaseConstruct,
    Statement,
    StringLiteral,
    SubCallStatement,
    SubroutineDefinitionConstruct,
    WhereConstruct,
)
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.types import LineTuple

HIST_CALLS = ("hist_addfld1d", "hist_addfld2d", "hist_addfld_decomp")
HIST_POINTERS = ("ptr_patch", "ptr_col", "ptr_lunit", "ptr_gcell", "ptr_lnd", "ptr_atm")
RESTART_CALL = "restartvar"
SET_OPTIONS = ("set_lake", "set_nolake", "set_urb", "set_nourb", "set_spec", "set_noglcmec")
REGISTRATION_RX = re.compile(r"\b(hist_addfld1d|hist_addfld2d|hist_addfld_decomp|restartvar)\b", re.I)


@dataclass(frozen=True)
class Registration:
    kind: str  # "hist" | "restart"
    field: str  # component path below `base` (x, core%x); "" for a bare variable
    name: str  # history field / restart variable name
    ln: int
    guards: tuple  # Guard (normalized text with its parsed condition) per enclosing if
    inactive: bool = False  # history field with default='inactive'
    base: str = "this"  # designator base: the passed object, an instance or a bare variable
    long_name: str = ""
    units: str = ""
    set_special: bool = False  # history field registered with set_lake/set_urb/set_spec/... (writes into the array)

    @property
    def path(self) -> str:
        return f"{self.base}%{self.field}" if self.field else self.base

    def applies_to(self, instance: str, element: str = "") -> bool:
        """
        Whether the registration's guards admit `instance`: `carbon_type` guards
        (c12/c13/c14 from the instance prefix) and, for a registration of a
        nutrient base type, species guards (`this%species == CN_SPECIES_N`,
        `is_carbon(this%species)`) for `element` (c12, c13, c14, n, p).
        Evaluated on the parsed conditions; a guard that cannot be evaluated
        admits (conservative: the field counts as registered).
        """
        iso = next((t for t in ("c13", "c14") if instance.startswith(f"{t}_")), "c12")
        for g in self.guards:
            value = g.evaluate(iso, element) if isinstance(g, Guard) else None
            if value is False:
                return False
        return True


class Guard(str):
    """
    A normalized `if` condition (the text, for grouping and reports) with its
    parsed expression; `negated` for the `not(...)` of an else branch.
    """

    node: object
    negated: bool

    def __new__(cls, text: str, node=None, negated: bool = False):
        obj = super().__new__(cls, text)
        obj.node = node
        obj.negated = negated
        return obj

    def evaluate(self, iso: str, element: str):
        """True/False if the condition is decided by `iso`/`element`, else None."""
        value = _eval(self.node, iso, element) if self.node is not None else None
        if value is None or not self.negated:
            return value
        return not value


def _designator(expr) -> str | None:
    """`this%species` -> 'this%species', `carbon_type` -> 'carbon_type'; None if not a designator."""
    try:
        return "%".join(n.lower() for n, _ in flatten_designator(expr))
    except Exception:
        return None


def _eval(node, iso: str, element: str):
    """Three-valued evaluation of a guard's carbon_type and species terms."""
    match node:
        case PrefixExpression() if node.operator.strip().lower() == ".not.":
            v = _eval(node.right_expr, iso, element)
            return None if v is None else not v
        case InfixExpression():
            op = node.operator.strip().lower()
            if op in (".and.", ".or."):
                a, b = _eval(node.left_expr, iso, element), _eval(node.right_expr, iso, element)
                values = {a, b}
                if op == ".and.":
                    return False if False in values else (True if values == {True} else None)
                return True if True in values else (False if values == {False} else None)
            if op in ("==", ".eq.", "/=", ".ne."):
                equal = op in ("==", ".eq.")
                for left, right in ((node.left_expr, node.right_expr), (node.right_expr, node.left_expr)):
                    name = _designator(left)
                    if name == "carbon_type" and isinstance(right, StringLiteral):
                        hit = str(right.value).strip("'\"").lower() == iso
                        return hit if equal else not hit
                    if name == "this%species":
                        other = _designator(right) or ""
                        if other.startswith("cn_species_") and element:
                            hit = other[len("cn_species_"):] == element
                            return hit if equal else not hit
                        return None
            return None
        case FuncExpression():
            if _designator(node.function) == "is_carbon" and element:
                return element in ("c12", "c13", "c14")
            return None
    return None


def iter_statements(
    block: BlockStatement, guards: tuple[str, ...] = ()
) -> Iterator[tuple[Statement, tuple[str, ...]]]:
    """Every statement in `block` with the conditions of its enclosing ifs."""
    for stmt in block.statements:
        yield stmt, guards
        match stmt:
            case IfConstruct():
                cond = Guard(_norm(stmt.condition), stmt.condition)
                yield from iter_statements(stmt.consequence, guards + (cond,))
                prev = [cond]
                for elif_ in stmt.else_ifs:
                    c = Guard(_norm(elif_.condition), elif_.condition)
                    yield from iter_statements(
                        elif_.consequence,
                        guards + tuple(Guard(f"not({p})", p.node, True) for p in prev) + (c,),
                    )
                    prev.append(c)
                if stmt.else_ is not None:
                    yield from iter_statements(
                        stmt.else_.alternative, guards + tuple(Guard(f"not({p})", p.node, True) for p in prev)
                    )
            case DoLoop() | DoWhile() | AssociateConstruct():
                yield from iter_statements(stmt.body, guards)
            case WhereConstruct():
                yield from iter_statements(stmt.body, guards)
                for ew in stmt.elsewheres:
                    yield from iter_statements(ew.body, guards)
            case SelectCaseConstruct():
                sel = _norm(stmt.selector)
                for case in stmt.cases:
                    vals = ",".join(_norm(v) for v in case.values or []) or "default"
                    yield from iter_statements(case.body, guards + (Guard(f"{sel}=={vals}"),))
            case MacroIf():
                yield from iter_statements(stmt.body, guards)
                for branch in stmt.branches:
                    yield from iter_statements(branch.body, guards)
            case BlockStatement():
                yield from iter_statements(stmt, guards)


def line_guards(block: BlockStatement) -> dict[int, tuple[str, ...]]:
    return {stmt.lineno: g for stmt, g in iter_statements(block)}


def _norm(expr) -> str:
    return str(expr).replace(" ", "").lower()


def _component(expr) -> str | None:
    """`this%x(...)` -> 'x', `this%core%x(...)` -> 'core%x'; anything else -> None."""
    parts = _designator_parts(expr)
    if parts is None or parts[0] != "this" or not parts[1]:
        return None
    return parts[1]


def _designator_parts(expr) -> tuple[str, str] | None:
    """`veg_cf%core%x(...)` -> ('veg_cf', 'core%x'); `x` -> ('x', ''); not a designator -> None."""
    try:
        names = [n.lower() for n, _ in flatten_designator(expr)]
    except Exception:
        return None
    if not names:
        return None
    return names[0], "%".join(names[1:])


def _text(expr) -> str:
    """A string literal's value, else the normalized expression text."""
    if expr is None:
        return ""
    if isinstance(expr, StringLiteral):
        return str(expr.value).strip("'\"")
    return _norm(expr)


def _keyword_args(call: SubCallStatement) -> dict[str, object]:
    out = {}
    for a in call.function.args:
        if isinstance(a, InfixExpression) and a.operator == "=":
            out[_norm(a.left_expr)] = a.right_expr
    return out


def parse_routine(
    filepath: Path,
    startline: int,
    endline: int,
    keep_directives: bool = False,
    keep_string_case: bool = False,
    skip_io: bool = False,
):
    """
    The routine's SubroutineDefinitionConstruct, parsed from the original file.
    skip_io: blank out I/O statements (write/print/read, which may hold implied-do
             lists the parser doesn't support) so the rest of the routine parses.
    """
    src = Path(filepath).read_text(errors="replace").split("\n")
    lines = [LineTuple(line=src[i], ln=i) for i in range(startline, endline + 1)]
    if skip_io:
        lines = _blank_io_statements(lines)
    parser = Parser(lines=lines, keep_directives=keep_directives, keep_string_case=keep_string_case)
    return parser.parse_program().statements[0]


_IO_START = re.compile(r"^\s*(?:\d+\s+)?(write|print|read)\b", re.I)


def _blank_io_statements(lines: list[LineTuple]) -> list[LineTuple]:
    """Replace I/O statements (and their continuation lines) with empty lines."""
    out: list[LineTuple] = []
    in_io = False
    for lt in lines:
        text = lt.line
        code = text.split("!", 1)[0]
        if in_io or _IO_START.match(code):
            in_io = code.rstrip().endswith("&")
            out.append(LineTuple(line="", ln=lt.ln))
        else:
            out.append(lt)
    return out


def registrations(tree) -> list[Registration]:
    out: list[Registration] = []
    pointers: dict[str, tuple[str, str]] = {}
    for stmt, guards in iter_statements(tree.body):
        if isinstance(stmt, ExpressionStatement):
            expr = stmt.expression
            if isinstance(expr, InfixExpression) and expr.operator == "=>":
                ptr = _norm(expr.left_expr)
                parts = _designator_parts(expr.right_expr)
                if parts is not None and parts[1]:
                    pointers[ptr] = parts
                else:
                    pointers.pop(ptr, None)
            continue
        if not isinstance(stmt, SubCallStatement):
            continue
        name = _norm(stmt.function.function)
        kwargs = _keyword_args(stmt)
        if name in HIST_CALLS:
            keys, kind = HIST_POINTERS, "hist"
        elif name == RESTART_CALL:
            keys, kind = ("data",), "restart"
        else:
            continue
        for key in keys:
            if key not in kwargs:
                continue
            value = kwargs[key]
            parts = pointers.get(_norm(value)) or _designator_parts(value)
            if parts is None:
                continue
            base, comp = parts
            label = kwargs.get("fname", kwargs.get("varname"))
            default = _norm(kwargs.get("default", ""))
            out.append(
                Registration(
                    kind,
                    comp,
                    _norm(label) if label is not None else "?",
                    stmt.lineno,
                    guards,
                    inactive=(default == "inactive"),
                    base=base,
                    long_name=_text(kwargs.get("long_name")),
                    units=_text(kwargs.get("units")),
                    set_special=any(k in kwargs for k in SET_OPTIONS),
                )
            )
    return out


def find_registrations(sub_dict: dict, logger: Optional[logging.Logger] = None) -> int:
    """
    Set `sub.registrations` for every routine in `sub_dict` whose original
    source mentions a registration call (type init/restart routines are
    usually not in elm_drv's call tree, so this covers unanalyzed routines
    too). Returns the number of registrations found.
    """
    by_file: dict[Path, list] = defaultdict(list)
    for sub in sub_dict.values():
        sub.registrations = []
        if getattr(sub, "library", False) or not getattr(sub, "filepath", None):
            continue
        if sub.startline is None or sub.startline < 0:
            continue
        by_file[Path(sub.filepath)].append(sub)
    total = 0
    for fp, subs in by_file.items():
        try:
            src = fp.read_text(errors="replace").split("\n")
        except OSError as err:
            if logger:
                logger.warning(f"registrations: cannot read {fp}: {err}")
            continue
        for sub in subs:
            end = getattr(sub, "end_stmt_ln", None) or sub.endline
            if not REGISTRATION_RX.search("\n".join(src[sub.startline : end + 1])):
                continue
            tree = None
            for skip_io in (False, True):
                try:
                    tree = parse_routine(fp, sub.startline, end, skip_io=skip_io)
                except (Exception, SystemExit):  # a parse failure must not fail the analysis
                    tree = None
                    continue
                if not isinstance(tree, SubroutineDefinitionConstruct):
                    tree = None
                    continue
                break
            if tree is None:
                if logger:
                    logger.warning(f"registrations: could not parse {sub.id} ({fp.name} L{sub.startline + 1})")
                continue
            sub.registrations = registrations(tree)
            total += len(sub.registrations)
    return total
