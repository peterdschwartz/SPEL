"""
Allocation bounds and pointer targets of derived-type components, from the
parse trees of a module's routines.

`allocate(this%core%leaf(begp:endp))` in a routine whose `this` is declared
`class(vegetation_carbon_state)`: the object's designator path is resolved
through the declared type of its base (a dummy, local or module variable)
and the types' components, so the bounds go to every type on the path —
`vegetation_carbon_state%core%leaf` and `vegetation_nutrient_state%leaf` —
whatever module the allocation is in. Types with the same component names
(the column and vegetation nutrient bases in one module) each get their own.
An object whose base type isn't known (an associate name) is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

from spel.scripts.fortran_parser.scope_walk import flatten_designator
from spel.scripts.fortran_parser.spel_ast import (
    AllocateStatement,
    BoundsExpression,
    ExpressionStatement,
    FuncExpression,
    InfixExpression,
    PointerAssignment,
    PrefixExpression,
    Statement,
    VariableDecl,
)

if TYPE_CHECKING:
    from spel.scripts.DerivedType import DerivedType

INTRINSIC = {"integer", "real", "logical", "character", "complex", "double", "doubleprecision"}


@dataclass
class Allocation:
    owner: str  # type name
    path: str  # component path in the owner: "leaf", "core%leaf"
    bounds: str  # "begp:endp,1:nlevdecomp"
    sub_id: str
    ln: int


@dataclass
class PointerInit:
    owner: str
    path: str
    target: str
    sub_id: str
    ln: int


def statements(node) -> Iterator[Statement]:
    """Every statement under `node`, in source order (blocks, branches, loops, associates)."""
    for value in vars(node).values():
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, Statement):
                yield item
                yield from statements(item)
            elif hasattr(item, "statements") or hasattr(item, "consequence") or hasattr(item, "alternative"):
                yield from statements(item)


PRECEDENCE = {"**": 6, "*": 5, "/": 5, "+": 4, "-": 4, "//": 3,
              "==": 2, "/=": 2, "<": 2, "<=": 2, ">": 2, ">=": 2, ".eq.": 2, ".ne.": 2, ".lt.": 2,
              ".le.": 2, ".gt.": 2, ".ge.": 2, ".and.": 1, ".or.": 0, ".eqv.": -1, ".neqv.": -1}


def text(expr) -> str:
    """Fortran for an expression with only the parentheses precedence needs (bounds as written)."""

    def prec(e) -> int:
        return PRECEDENCE.get(e.operator.lower(), 7) if isinstance(e, InfixExpression) else 8

    match expr:
        case InfixExpression():
            p = prec(expr)
            left, right = text(expr.left_expr), text(expr.right_expr)
            lp, rp = prec(expr.left_expr), prec(expr.right_expr)
            if lp < p or (lp == p and expr.operator == "**"):
                left = f"({left})"
            if rp < p or (rp == p and expr.operator in ("-", "/")):
                right = f"({right})"
            return f"{left}{expr.operator}{right}"
        case PrefixExpression():
            inner = text(expr.right_expr)
            if isinstance(expr.right_expr, InfixExpression):
                inner = f"({inner})"
            return f"{expr.operator}{inner}" if expr.operator in ("-", "+") else f"{expr.operator} {inner}"
        case BoundsExpression():
            return f"{text(expr.start) if expr.start is not None else ''}:{text(expr.end) if expr.end is not None else ''}"
        case FuncExpression():
            return f"{text(expr.function)}({','.join(text(a) for a in expr.args)})"
    return str(expr)


def declared_types(tree) -> dict[str, str]:
    """name -> derived type of the routine's declarations (not intrinsic types)."""
    out = {}
    for st in tree.body.statements:
        if isinstance(st, VariableDecl):
            t = st.var_type.token.literal.lower()
            if t not in INTRINSIC and t not in ("type", "class"):
                for e in st.entities:
                    out[e.token.literal.lower()] = t
    return out


def owners(names: list[str], base_type: Optional[str], types: dict[str, DerivedType]) -> list[tuple[str, str]]:
    """(type, component path) for every type on the designator `names` whose type is known."""
    out = []
    t = base_type
    for k in range(len(names) - 1):
        if t is None or t not in types:
            break
        out.append((t, "%".join(names[k + 1 :])))
        comp = types[t].components.get(names[k + 1])
        t = str(comp.type).lower() if comp is not None else None
    return out


def scan_routine(sub_id: str, tree, types: dict[str, DerivedType], module_types: dict[str, str]):
    """Allocations and pointer initializations of components in one routine."""
    allocs: list[Allocation] = []
    ptrs: list[PointerInit] = []
    local = declared_types(tree)

    def base_type(name: str) -> Optional[str]:
        return local.get(name) or module_types.get(name)

    def path_of(expr):
        try:
            comps = flatten_designator(expr)
        except Exception:
            return None, None
        return [n.lower() for n, _ in comps], comps

    for st in statements(tree.body):
        if isinstance(st, AllocateStatement) and not st.is_deallocate:
            for obj in st.objects:
                names, comps = path_of(obj)
                if not names or len(names) < 2 or not comps[-1][1]:
                    continue
                bounds = ",".join(text(a) for a in comps[-1][1])
                for owner, path in owners(names, base_type(names[0]), types):
                    allocs.append(Allocation(owner, path, bounds, sub_id, st.lineno))
        pointer = target = None
        if isinstance(st, PointerAssignment):
            pointer, target = st.pointer, st.target
        elif isinstance(st, ExpressionStatement) and isinstance(st.expression, InfixExpression) \
                and st.expression.operator == "=>":
            pointer, target = st.expression.left_expr, st.expression.right_expr
        if pointer is not None:
            names, _ = path_of(pointer)
            tnames, tcomps = path_of(target)
            # a target rooted in an array element (`compact_groups(id)%data(:, s)`, the compact
            # layout's storage) is not an alias of another instance's component
            storage = bool(tcomps) and bool(tcomps[0][1])
            if names and len(names) > 1 and not storage:
                for owner, path in owners(names, base_type(names[0]), types):
                    ptrs.append(PointerInit(owner, path, str(target).strip(), sub_id, st.lineno))
    return allocs, ptrs


def scan_module(mod, sub_dict: dict, types: dict[str, DerivedType]):
    """(allocations, pointer initializations) in the routines of FortranModule `mod`."""
    module_types = {n: str(v.type).lower() for n, v in mod.global_vars.items()}
    allocs, ptrs = [], []
    for sub_id in sorted(mod.subroutines, key=lambda s: getattr(sub_dict.get(s), "startline", 0)):
        sub = sub_dict.get(sub_id)
        tree = getattr(sub, "syntax_tree", None)
        if tree is None:
            continue
        a, p = scan_routine(sub_id, tree, types, module_types)
        allocs += a
        ptrs += p
    return allocs, ptrs


def rank(bounds: str) -> int:
    """Number of top-level comma separated dimensions in `bounds`."""
    depth, n = 0, 1
    for ch in bounds:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            n += 1
    return n


def apply_allocations(types: dict[str, DerivedType], allocs: list[Allocation], ptrs: list[PointerInit] = ()) -> None:
    """Set component bounds (rank must match; the last allocation wins), init routines, pointer targets."""
    for a in allocs:
        dtype = types.get(a.owner)
        comp = dtype.components.get(a.path) if dtype is not None else None
        if comp is None or comp.dim == 0 or rank(a.bounds) != comp.dim:
            continue
        comp.bounds = a.bounds
        dtype.init_sub_name = a.sub_id
    for p in ptrs:
        dtype = types.get(p.owner)
        comp = dtype.components.get(p.path) if dtype is not None else None
        if comp is None:
            continue
        if p.target not in comp.pointer:
            comp.pointer.append(p.target)
            dtype.init_sub_name = p.sub_id
