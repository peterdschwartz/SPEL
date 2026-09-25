from enum import Enum, auto

from spel.scripts.fortran_parser.spel_ast import (
    FloatLiteral,
    FuncExpression,
    Identifier,
    InfixExpression,
    IntegerLiteral,
    LogicalLiteral,
    PrefixExpression,
    StringLiteral,
)


class TruthValues(Enum):
    TRUE = auto()
    FALSE = auto()
    UNKNOWN = auto()


def tv_from_bool(b):
    return TruthValues.TRUE if b else TruthValues.FALSE


def tv_not(x):
    return TruthValues.FALSE if x is TruthValues.TRUE else TruthValues.TRUE if x is TruthValues.FALSE else TruthValues.UNKNOWN


def tv_and(a, b):
    if a is TruthValues.FALSE or b is TruthValues.FALSE:
        return TruthValues.FALSE
    if a is TruthValues.TRUE and b is TruthValues.TRUE:
        return TruthValues.TRUE
    return TruthValues.UNKNOWN


def tv_or(a, b):
    if a is TruthValues.TRUE or b is TruthValues.TRUE:
        return TruthValues.TRUE
    if a is TruthValues.FALSE and b is TruthValues.FALSE:
        return TruthValues.FALSE
    return TruthValues.UNKNOWN


LOGICAL_OPS = {".and.": tv_and, ".or.": tv_or}
REL_OPS = {
    ".eq.",
    "==",
    "/=",
    ".ne.",
    "<",
    "<=",
    ">",
    ">=",
    ".lt.",
    ".le.",
    ".gt.",
    ".ge.",
}


class UnknownValue:
    pass


UNKNOWN = UnknownValue()


def get(env, name):
    return env.get(name, UNKNOWN)


def to_tv(x):
    # Accept already-TV, bool → TV, Unknown → UNKNOWN sentinel, other scalars pass through
    if isinstance(x, TruthValues):
        return x
    if isinstance(x, bool):
        return tv_from_bool(x)
    if x is UNKNOWN:
        return TruthValues.UNKNOWN
    return x  # numeric/str etc.


def cmp_tv(op, L, R):
    if L is UNKNOWN or R is UNKNOWN:
        return TruthValues.UNKNOWN
    if isinstance(L, TruthValues) or isinstance(R, TruthValues):
        return TruthValues.UNKNOWN
    # symbolic + dot-ops
    try:
        if op in (".eq.", "=="):
            return tv_from_bool(L == R)
        if op in (".ne.", "/="):
            return tv_from_bool(L != R)
        if op in (".lt.", "<"):
            return tv_from_bool(L < R)
        if op in (".le.", "<="):
            return tv_from_bool(L <= R)
        if op in (".gt.", ">"):
            return tv_from_bool(L > R)
        if op in (".ge.", ">="):
            return tv_from_bool(L >= R)
    except Exception:
        return TruthValues.UNKNOWN
    return TruthValues.UNKNOWN


def eval_expr(node, env):
    if isinstance(node, Identifier):
        v = get(env, node.value)
        return to_tv(v)

    # literals → return their native values
    if isinstance(node, LogicalLiteral):
        return TruthValues.TRUE if node.value else TruthValues.FALSE
    if isinstance(node, IntegerLiteral):
        return node.value
    if isinstance(node, FloatLiteral):
        # ensure your parser fills .value; if not, treat as UNKNOWN (or compute once)
        return node.value if node.value is not None else UNKNOWN
    if isinstance(node, StringLiteral):
        return node.value
    # Prefix .NOT.
    if isinstance(node, PrefixExpression):
        if node.operator == ".not.":
            x = eval_expr(node.right_expr, env)
            x = to_tv(x)
            return tv_not(x) if isinstance(x, TruthValues) else TruthValues.UNKNOWN
        return TruthValues.UNKNOWN

    # Function/array call
    if isinstance(node, FuncExpression):
        cal = eval_expr(node.function, env)
        if cal is UNKNOWN or cal is TruthValues.UNKNOWN or not callable(cal):
            return TruthValues.UNKNOWN
        args = []
        for a in node.args:
            av = eval_expr(a, env)
            if av is TruthValues.UNKNOWN or av is UNKNOWN:
                return TruthValues.UNKNOWN  # conservatively unknown if any arg unknown
            args.append(True if av is TruthValues.TRUE else False if av is TruthValues.FALSE else av)
        try:
            res = cal(*args)
            return to_tv(res)
        except Exception:
            return TruthValues.UNKNOWN

    # Infix (logical/relational)
    if isinstance(node, InfixExpression):
        op = node.operator
        L = eval_expr(node.left_expr, env)
        R = eval_expr(node.right_expr, env)

        if op in LOGICAL_OPS:
            Ltv = to_tv(L)
            Rtv = to_tv(R)
            if not isinstance(Ltv, TruthValues):
                Ltv = TruthValues.UNKNOWN
            if not isinstance(Rtv, TruthValues):
                Rtv = TruthValues.UNKNOWN
            return LOGICAL_OPS[op](Ltv, Rtv)

        if op in REL_OPS:
            return cmp_tv(op, L, R)

        return TruthValues.UNKNOWN

    # booleans, numbers, strings passed through if you have literal nodes
    if isinstance(node, bool):
        return tv_from_bool(node)

    return TruthValues.UNKNOWN


def eval_if_condition(node, env):
    """Top-level IF policy: UNKNOWN ⇒ treat as True (could be true)."""
    v = eval_expr(node, env)
    if isinstance(v, TruthValues):
        return v is not TruthValues.FALSE
    if isinstance(v, bool):
        return v
    return True
