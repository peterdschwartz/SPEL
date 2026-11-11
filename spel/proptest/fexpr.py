"""
Fortran expressions for properties, scenarios and dimension bounds.

Text is parsed with SPEL's parser (`spel_parser.Parser`, standard unary-sign precedence)
into AST nodes; `Evaluator` walks those nodes with numpy. Nothing here inspects the text.

Element-wise evaluation: the loop variable of a property (e.g. `p`) is bound to the vector
of element indices, so `x(p)` is a vector, `x(p, :)` an (elements x levels) array and
`x(veg_pp%column(p))` a gather. Reductions (`sum`, `maxval`, ...) of a value whose leading
axis is the element axis reduce per element; Fortran's `dim=k` names axis k of `x(p, ...)`'s
trailing dimensions.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import numpy as np

from spel.proptest.errors import ProptestError
from spel.scripts.fortran_parser.spel_ast import (
    ArrayInit,
    BoundsExpression,
    Expression,
    ExpressionStatement,
    FieldAccessExpression,
    FloatLiteral,
    FuncExpression,
    Identifier,
    InfixExpression,
    IntegerLiteral,
    KeywordArgument,
    LogicalLiteral,
    PrefixExpression,
    StringLiteral,
)
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.types import LineTuple


class ExprError(ProptestError):
    """A property/scenario expression that does not parse or cannot be evaluated."""


def parse_expr(text: str) -> Expression:
    """One Fortran expression (raises ExprError otherwise); line breaks (TOML multi-line strings)
    count as blanks."""
    out = io.StringIO()
    line = " ".join(str(text).split("\n")).strip()
    try:
        with redirect_stdout(out):
            parser = Parser(lines=[LineTuple(line=line, ln=0)], fortran_unary=True)
            prog = parser.parse_program()
    except (SystemExit, Exception) as exc:  # the parser exits on syntax errors
        raise ExprError(
            f"cannot parse {text!r}: {exc} {out.getvalue().strip()}"
        ) from None
    stmts = prog.statements
    if (
        parser.errors
        or len(stmts) != 1
        or not isinstance(stmts[0], ExpressionStatement)
    ):
        raise ExprError(f"not a single expression: {text!r} {parser.errors or ''}")
    expr = stmts[0].expression
    if isinstance(expr, InfixExpression) and expr.operator == "=":
        raise ExprError(f"an assignment, not an expression: {text!r}")
    return expr


def parse_section(text: str) -> tuple[Optional[Expression], Optional[Expression]]:
    """One subscript of a dimension, `3`, `1:3` or `:nlevsoi` -> its (start, end) expressions
    (None = open; a single index is both)."""
    ref = parse_expr(f"x({text})")
    args = getattr(ref, "args", None)
    if args is None or len(args) != 1:
        raise ExprError(f"not a subscript of one dimension: {text!r}")
    a = args[0]
    if isinstance(a, BoundsExpression):
        if getattr(a, "stride", None) is not None:
            raise ExprError("strided sections are not supported")
        return a.start, a.end
    return a, a


def designator(node: Expression) -> Optional[list[str]]:
    """`a%b%c` -> ['a', 'b', 'c'] (lower case); None for anything else."""
    if isinstance(node, Identifier):
        # the parser may give a component reference as one identifier (`a%b`)
        return node.value.lower().split("%")
    if isinstance(node, FieldAccessExpression):
        left, right = designator(node.left), designator(node.field)
        if left is not None and right is not None:
            return left + right
    return None


def keyword_of(arg: Expression) -> Optional[tuple[str, Expression]]:
    """A `name=value` actual argument (parsed as `=` infix, or classified)."""
    if isinstance(arg, KeywordArgument):
        return arg.keyword.lower(), arg.value
    if (
        isinstance(arg, InfixExpression)
        and arg.operator == "="
        and isinstance(arg.left_expr, Identifier)
    ):
        return arg.left_expr.value.lower(), arg.right_expr
    return None


def children(node: Expression) -> list[Expression]:
    """Direct sub-expressions."""
    match node:
        case InfixExpression():
            return [node.left_expr, node.right_expr]
        case PrefixExpression():
            return [node.right_expr]
        case FieldAccessExpression():
            return [node.left, node.field]
        case FuncExpression():
            return [node.function, *node.args]
        case BoundsExpression():
            return [e for e in (node.start, node.end) if e is not None]
        case KeywordArgument():
            return [node.value]
        case ArrayInit():
            return list(node.elements)
    return []


@dataclass(frozen=True)
class Use:
    """A name an expression uses: `path` (designator) and whether it is called/indexed."""

    path: tuple[str, ...]
    applied: bool  # f(...) / x(...)


def uses(node: Expression) -> Iterator[Use]:
    """Every designator in `node` (keyword names excluded)."""
    if (kw := keyword_of(node)) is not None:
        yield from uses(kw[1])
        return
    if isinstance(node, FuncExpression):
        path = designator(node.function)
        if path is not None:
            yield Use(tuple(path), True)
        else:
            yield from uses(node.function)
        for a in node.args:
            yield from uses(a)
        return
    path = designator(node)
    if path is not None:
        yield Use(tuple(path), False)
        return
    for c in children(node):
        yield from uses(c)


@dataclass
class FArray:
    """A Fortran array: data in Fortran index order and its lower bounds."""

    data: np.ndarray
    lb: tuple[int, ...]
    name: str = ""

    @property
    def ub(self) -> tuple[int, ...]:
        return tuple(l + n - 1 for l, n in zip(self.lb, self.data.shape))

    def take(self, subs: list) -> Any:
        if len(subs) != self.data.ndim:
            raise ExprError(
                f"{self.name}: {len(subs)} subscripts for rank {self.data.ndim}"
            )
        idx = []
        indexed = []  # positions of the non-section subscripts
        vector = False
        for k, s in enumerate(subs):
            lb = self.lb[k]
            if isinstance(s, slice):
                lo = 0 if s.start is None else int(s.start) - lb
                hi = None if s.stop is None else int(s.stop) - lb + 1
                idx.append(slice(lo, hi))
            else:
                i = np.asarray(s)
                if i.dtype.kind not in "iu":
                    raise ExprError(f"{self.name}: non-integer subscript")
                j = i - lb
                if np.any(j < 0) or np.any(j >= self.data.shape[k]):
                    raise ExprError(
                        f"{self.name}: subscript out of bounds {self.lb[k]}:{self.ub[k]} in dim {k + 1}"
                    )
                idx.append(j)
                indexed.append(k)
                vector = vector or j.ndim > 0
        out = self.data[tuple(idx)]
        if vector and indexed[-1] - indexed[0] + 1 == len(indexed) and indexed[0] > 0:
            # numpy keeps the axis of adjacent index subscripts in place (`x(:, p)` -> (levels,
            # elements)); the element axis goes first, as every other element-wise value has it
            out = np.moveaxis(out, indexed[0], 0)
        return out


class Namespace:
    """A value whose components are looked up by name (`base`, `follow`, an instance)."""

    def __init__(self, name: str, get: Callable[[list[str]], Any]):
        self.name = name
        self.get = get


def _real(x):
    return np.asarray(x, dtype=float)


def _sign(a, b):
    a = np.abs(a)
    return np.where(np.asarray(b) >= 0, a, -a)


def _min(*a):
    return np.minimum.reduce(np.broadcast_arrays(*a))


def _max(*a):
    return np.maximum.reduce(np.broadcast_arrays(*a))


ELEMENTAL: dict[str, Callable] = {
    "abs": np.abs,
    "dabs": np.abs,
    "sqrt": np.sqrt,
    "dsqrt": np.sqrt,
    "exp": np.exp,
    "dexp": np.exp,
    "log": np.log,
    "dlog": np.log,
    "alog": np.log,
    "log10": np.log10,
    "alog10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "min": _min,
    "max": _max,
    "amin1": _min,
    "amax1": _max,
    "dmin1": _min,
    "dmax1": _max,
    "mod": np.fmod,
    "modulo": np.mod,
    "sign": _sign,
    "hypot": np.hypot,
    "real": _real,
    "dble": _real,
    "float": _real,
    "sngl": _real,
    "int": lambda x: np.trunc(x).astype(np.int64),
    "nint": lambda x: np.rint(x).astype(np.int64),
    "floor": lambda x: np.floor(x).astype(np.int64),
    "ceiling": lambda x: np.ceil(x).astype(np.int64),
    "merge": lambda t, f, mask: np.where(mask, t, f),
    "epsilon": lambda x=1.0: np.finfo(float).eps,
    "huge": lambda x=1.0: np.finfo(float).max,
    "tiny": lambda x=1.0: np.finfo(float).tiny,
    "isnan": np.isnan,
    "ieee_is_nan": np.isnan,
    "isfinite": np.isfinite,
    "ieee_is_finite": np.isfinite,
}

REDUCTIONS: dict[str, Callable] = {
    "sum": np.sum,
    "product": np.prod,
    "maxval": np.max,
    "minval": np.min,
    "any": np.any,
    "all": np.all,
    "count": np.count_nonzero,
}

RELATIONAL = {
    "==": np.equal,
    ".eq.": np.equal,
    "/=": np.not_equal,
    ".ne.": np.not_equal,
    "!=": np.not_equal,
    "<": np.less,
    ".lt.": np.less,
    "<=": np.less_equal,
    ".le.": np.less_equal,
    ">": np.greater,
    ".gt.": np.greater,
    ">=": np.greater_equal,
    ".ge.": np.greater_equal,
}
LOGICAL = {
    ".and.": np.logical_and,
    ".or.": np.logical_or,
    ".eqv.": np.equal,
    ".neqv.": np.not_equal,
}


def _divide(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "iu" and b.dtype.kind in "iu":
        return np.trunc(a / b).astype(np.int64)  # Fortran integer division
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


ARITHMETIC = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": _divide,
    "**": lambda a, b: np.power(
        (
            np.asarray(a, dtype=float)
            if np.asarray(b).dtype.kind == "f" or np.any(np.asarray(b) < 0)
            else a
        ),
        b,
    ),
}


class Evaluator:
    """
    resolve(path) -> value of a designator that is not a local (FArray, Namespace, scalar or
    ndarray); raise KeyError when unknown. functions: name -> callable(*args, **kwargs) (args
    evaluated; an unsubscripted array arrives as an FArray). n_elem: the element count
    reductions keep (None outside element-wise evaluation).
    """

    def __init__(
        self,
        resolve: Callable[[list[str]], Any],
        functions: Optional[dict[str, Callable]] = None,
    ):
        self.resolve = resolve
        self.functions = functions or {}

    def __call__(
        self,
        node: Expression,
        local: Optional[dict[str, Any]] = None,
        n_elem: Optional[int] = None,
    ) -> Any:
        self.local = local or {}
        self.n_elem = n_elem
        return self.eval(node)

    def elemental(self, node: Expression) -> bool:
        """Whether `node` has one value per element: it uses a local bound to a vector (the element
        variable). Its value then has the element axis first (`FArray.take`). Decided from the
        expression, never from a shape: a level dimension may have as many entries as there are
        elements."""
        for u in uses(node):
            v = self.local.get(u.path[0])
            if isinstance(v, np.ndarray) and v.ndim > 0:
                return True
        return False

    # -- names
    def lookup(self, path: list[str]) -> Any:
        head = path[0]
        if head in self.local:
            value = self.local[head]
            if len(path) == 1:
                return value
            if isinstance(value, Namespace):
                return value.get(path[1:])
            raise ExprError(f"{head} has no components")
        try:
            return self.resolve(path)
        except KeyError:
            raise ExprError(f"unknown name {'%'.join(path)}") from None

    # -- nodes
    def eval(self, node: Expression) -> Any:
        match node:
            case IntegerLiteral():
                return np.int64(int(node.value))
            case FloatLiteral():
                return np.float64(float(node.value))
            case LogicalLiteral():
                return np.bool_(bool(node.value))
            case StringLiteral():
                v = str(node.value)
                return v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"" else v
            case Identifier() | FieldAccessExpression():
                path = designator(node)
                if path is None:
                    raise ExprError(f"unsupported designator {node}")
                return self.lookup(path)
            case PrefixExpression():
                right = self.value(node.right_expr)
                match node.operator.lower():
                    case "-":
                        return np.negative(right)
                    case "+":
                        return right
                    case ".not.":
                        return np.logical_not(right)
                raise ExprError(f"unsupported prefix operator {node.operator}")
            case InfixExpression():
                return self.infix(node)
            case FuncExpression():
                return self.apply(node)
            case ArrayInit():
                return np.array([self.value(e) for e in node.elements])
        raise ExprError(f"unsupported expression {node} ({type(node).__name__})")

    def value(self, node: Expression) -> Any:
        """A value usable in arithmetic (not a whole array or a namespace)."""
        v = self.eval(node)
        if isinstance(v, FArray):
            raise ExprError(
                f"{v.name or node}: subscript the array (e.g. with the element variable)"
            )
        if isinstance(v, Namespace):
            raise ExprError(f"{v.name}: a namespace is not a value")
        return v

    def infix(self, node: InfixExpression) -> Any:
        op = node.operator.lower()
        if op == "=":
            raise ExprError(f"assignment/keyword outside an argument list: {node}")
        a, b = self.aligned(
            [
                (self.value(node.left_expr), node.left_expr),
                (self.value(node.right_expr), node.right_expr),
            ]
        )
        with np.errstate(all="ignore"):
            if op in RELATIONAL:
                return RELATIONAL[op](a, b)
            if op in LOGICAL:
                return LOGICAL[op](np.asarray(a, dtype=bool), np.asarray(b, dtype=bool))
            if op in ARITHMETIC:
                return ARITHMETIC[op](a, b)
        raise ExprError(f"unsupported operator {node.operator}")

    def aligned(self, operands: list[tuple[Any, Expression]]) -> list[Any]:
        """Operands of one elemental operation, the element-wise ones lined up on the element axis:
        `x(p) * w(p, :)` is each element's x times its row of w, as in the Fortran loop over p
        (numpy alone would line a vector up with the last axis)."""
        ranks = [
            (
                np.ndim(v)
                if not isinstance(v, (FArray, Namespace, str)) and self.elemental(n)
                else 0
            )
            for v, n in operands
        ]
        top = max(ranks, default=0)
        out = []
        for (v, _), r in zip(operands, ranks):
            if 0 < r < top:
                v = np.asarray(v)
                v = v.reshape(v.shape + (1,) * (top - r))
            out.append(v)
        return out

    def subscript(self, arg: Expression) -> Any:
        if isinstance(arg, BoundsExpression):
            if getattr(arg, "stride", None) is not None:
                raise ExprError("strided sections are not supported")
            lo = None if arg.start is None else int(self.value(arg.start))
            hi = None if arg.end is None else int(self.value(arg.end))
            return slice(lo, hi)
        return self.value(arg)

    def apply(self, node: FuncExpression) -> Any:
        path = designator(node.function)
        if path is None:
            raise ExprError(f"unsupported call/subscript {node}")
        target = None
        name = path[0] if len(path) == 1 else ""
        if (
            not (
                name
                and (name in self.functions or name in ELEMENTAL or name in REDUCTIONS)
            )
            or name in self.local
        ):
            target = self.lookup(path)
        if isinstance(target, FArray):
            return target.take([self.subscript(a) for a in node.args])
        if target is not None:
            raise ExprError(f"{'%'.join(path)} is not an array")
        args, kwargs = [], {}
        nodes = []
        first = None  # the array argument of a reduction
        for a in node.args:
            kw = keyword_of(a)
            if kw is not None:
                kwargs[kw[0]] = self.eval(kw[1])
                if kw[0] == "array":
                    first = kw[1]
            else:
                args.append(self.eval(a))
                nodes.append(a)
                first = a if first is None else first
        if name in self.functions:
            return self.functions[name](*self.aligned(list(zip(args, nodes))), **kwargs)
        args = [self._plain(name, a) for a in args]
        kwargs = {k: self._plain(name, v) for k, v in kwargs.items()}
        with np.errstate(all="ignore"):
            if name in ELEMENTAL:
                if kwargs:
                    raise ExprError(f"{name}: keyword arguments are not supported")
                return ELEMENTAL[name](*self.aligned(list(zip(args, nodes))))
            if "array" in kwargs:
                args.insert(0, kwargs.pop("array"))
            return self.reduce(
                name,
                *args,
                per_elem=first is not None and self.elemental(first),
                **kwargs,
            )

    def _plain(self, name: str, v: Any) -> Any:
        if isinstance(v, (FArray, Namespace)):
            raise ExprError(f"{name}: subscript the array argument")
        return v

    def reduce(self, name: str, x, dim=None, mask=None, per_elem: bool = False) -> Any:
        """per_elem: x is element-wise (axis 0 = elements); with trailing dimensions the reduction
        is over those, per element."""
        x = np.asarray(x)
        if mask is not None:
            if name not in (
                "sum",
                "product",
                "maxval",
                "minval",
                "count",
                "any",
                "all",
            ):
                raise ExprError(f"{name}: mask= is not supported")
            fill = {"sum": 0, "product": 1, "maxval": -np.inf, "minval": np.inf}.get(
                name, False
            )
            x = np.where(mask, x, fill)
        per_elem = per_elem and x.ndim >= 2
        if dim is not None:
            axis = int(dim) if per_elem else int(dim) - 1
        else:
            axis = tuple(range(1, x.ndim)) if per_elem else None
        return REDUCTIONS[name](x, axis=axis)


def evaluate_bound(node: Expression, env: dict[str, int]) -> int:
    """An integer bound expression (`nlevgrnd`, `-nlevsno+1`, `numpft`) over `env`."""

    def resolve(path):
        name = "%".join(path)
        if name not in env:
            raise KeyError(name)
        return env[name]

    try:
        return int(Evaluator(resolve)(node))
    except ExprError as exc:
        raise ExprError(f"bound {node}: {exc}") from None
