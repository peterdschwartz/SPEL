from enum import Enum, auto

from scripts.fortran_parser.spel_ast import (
    FloatLiteral,
    FuncExpression,
    Identifier,
    InfixExpression,
    IntegerLiteral,
    LogicalLiteral,
    PrefixExpression,
    StringLiteral,
)


class TV(Enum):
    TRUE = auto()
    FALSE = auto()
    UNKNOWN = auto()


def tv_from_bool(b):
    return TV.TRUE if b else TV.FALSE


def tv_not(x):
    return TV.FALSE if x is TV.TRUE else TV.TRUE if x is TV.FALSE else TV.UNKNOWN


def tv_and(a, b):
    if a is TV.FALSE or b is TV.FALSE:
        return TV.FALSE
    if a is TV.TRUE and b is TV.TRUE:
        return TV.TRUE
    return TV.UNKNOWN


def tv_or(a, b):
    if a is TV.TRUE or b is TV.TRUE:
        return TV.TRUE
    if a is TV.FALSE and b is TV.FALSE:
        return TV.FALSE
    return TV.UNKNOWN


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
    if isinstance(x, TV):
        return x
    if isinstance(x, bool):
        return tv_from_bool(x)
    if x is UNKNOWN:
        return TV.UNKNOWN
    return x  # numeric/str etc.


def cmp_tv(op, L, R):
    if L is UNKNOWN or R is UNKNOWN:
        return TV.UNKNOWN
    if isinstance(L, TV) or isinstance(R, TV):
        return TV.UNKNOWN
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
        return TV.UNKNOWN
    return TV.UNKNOWN


# ---- AST evaluators (use your concrete classes) ----


def eval_expr(node, env):
    if isinstance(node, Identifier):
        v = get(env, node.value)
        return to_tv(v)

    # literals → return their native values
    if isinstance(node, LogicalLiteral):
        return TV.TRUE if node.value else TV.FALSE
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
            return tv_not(x) if isinstance(x, TV) else TV.UNKNOWN
        return TV.UNKNOWN

    # Function/array call
    if isinstance(node, FuncExpression):
        cal = eval_expr(node.function, env)
        if cal is UNKNOWN or cal is TV.UNKNOWN or not callable(cal):
            return TV.UNKNOWN
        args = []
        for a in node.args:
            av = eval_expr(a, env)
            if av is TV.UNKNOWN or av is UNKNOWN:
                return TV.UNKNOWN  # conservatively unknown if any arg unknown
            args.append(True if av is TV.TRUE else False if av is TV.FALSE else av)
        try:
            res = cal(*args)
            return to_tv(res)
        except Exception:
            return TV.UNKNOWN

    # Infix (logical/relational)
    if isinstance(node, InfixExpression):
        op = node.operator
        L = eval_expr(node.left_expr, env)
        R = eval_expr(node.right_expr, env)

        if op in LOGICAL_OPS:
            Ltv = to_tv(L)
            Rtv = to_tv(R)
            if not isinstance(Ltv, TV):
                Ltv = TV.UNKNOWN
            if not isinstance(Rtv, TV):
                Rtv = TV.UNKNOWN
            return LOGICAL_OPS[op](Ltv, Rtv)

        if op in REL_OPS:
            return cmp_tv(op, L, R)

        return TV.UNKNOWN

    # booleans, numbers, strings passed through if you have literal nodes
    if isinstance(node, bool):
        return tv_from_bool(node)

    return TV.UNKNOWN


def eval_if_condition(node, env):
    """Top-level IF policy: UNKNOWN ⇒ treat as True (could be true)."""
    v = eval_expr(node, env)
    if isinstance(v, TV):
        return v is not TV.FALSE
    if isinstance(v, bool):
        return v
    return True
