from scripts.fortran_parser.spel_ast import (
    Expression,
    Identifier,
    IfConstruct,
    InfixExpression,
    PrefixExpression,
)

_LOGICAL_AND = {".and.", "and"}
_LOGICAL_OR = {".or.", "or"}
_LOGICAL_NOT = {".not.", "not"}

_NEGATED_COMPARISON = {
    "==": "/=",
    "=": "/=",
    ".eq.": ".ne.",
    "/=": "==",
    "!=": "==",
    ".ne.": ".eq.",
    ">": "<=",
    ".gt.": ".le.",
    ">=": "<",
    ".ge.": ".lt.",
    "<": ">=",
    ".lt.": ".ge.",
    "<=": ">",
    ".le.": ".gt.",
}

_REVERSED_COMPARISON = {
    "==": "==",
    "=": "=",
    ".eq.": ".eq.",
    "/=": "/=",
    "!=": "!=",
    ".ne.": ".ne.",
    ">": "<",
    ".gt.": ".lt.",
    ">=": "<=",
    ".ge.": ".le.",
    "<": ">",
    ".lt.": ".gt.",
    "<=": ">=",
    ".le.": ".ge.",
}


def _negate_comparison_operator(op: str) -> str:
    return _NEGATED_COMPARISON.get(op.lower(), f".not. ({op})")


def _reverse_comparison_operator(op: str) -> str:
    return _REVERSED_COMPARISON.get(op.lower(), op)


def _merge_expectation_maps(
    left: dict[str, list[str]], right: dict[str, list[str]]
) -> dict[str, list[str]]:
    merged = {name: values.copy() for name, values in left.items()}
    for name, values in right.items():
        merged.setdefault(name, []).extend(values)
    return merged


def _combine_expectation_alternatives(
    left: list[dict[str, list[str]]], right: list[dict[str, list[str]]]
) -> list[dict[str, list[str]]]:
    return [_merge_expectation_maps(a, b) for a in left for b in right]


def infer_condition_expectations(
    expr: Expression,
    variables: set[str],
    truth: bool = True,
) -> list[dict[str, list[str]]]:
    """
    Return possible expectation alternatives for tracked variables.

    Example:
        a > 3 .and. flag

    returns:
        [{"a": ["> 3"], "flag": ["True"]}]

    For `.or.`, multiple alternatives are returned because either side may make
    the condition true.
    """
    if isinstance(expr, PrefixExpression) and expr.operator.lower() in _LOGICAL_NOT:
        return infer_condition_expectations(expr.right_expr, variables, not truth)

    if isinstance(expr, InfixExpression):
        op = expr.operator.lower()

        if op in _LOGICAL_AND:
            left = infer_condition_expectations(expr.left_expr, variables, truth)
            right = infer_condition_expectations(expr.right_expr, variables, truth)

            if truth:
                return _combine_expectation_alternatives(left, right)

            return left + right

        if op in _LOGICAL_OR:
            left = infer_condition_expectations(expr.left_expr, variables, truth)
            right = infer_condition_expectations(expr.right_expr, variables, truth)

            if truth:
                return left + right

            return _combine_expectation_alternatives(left, right)

        if op in _NEGATED_COMPARISON:
            expectations: dict[str, list[str]] = {}

            if (
                isinstance(expr.left_expr, Identifier)
                and expr.left_expr.value in variables
            ):
                expected_op = op if truth else _negate_comparison_operator(op)
                expectations.setdefault(expr.left_expr.value, []).append(
                    f"{expected_op} {expr.right_expr}"
                )

            if (
                isinstance(expr.right_expr, Identifier)
                and expr.right_expr.value in variables
            ):
                reversed_op = _reverse_comparison_operator(op)
                expected_op = (
                    reversed_op if truth else _negate_comparison_operator(reversed_op)
                )
                expectations.setdefault(expr.right_expr.value, []).append(
                    f"{expected_op} {expr.left_expr}"
                )

            return [expectations] if expectations else [{}]

    if isinstance(expr, Identifier) and expr.value in variables:
        return [{expr.value: [str(truth)]}]

    return [{}]


def log_if_condition_expectations(
    if_construct: IfConstruct,
    variables: dict[str, object],
    logger=print,
) -> list[dict[str, list[str]]]:
    """
    Log tracked-variable expectations required for an IfConstruct condition
    to evaluate True.
    """
    alternatives = infer_condition_expectations(
        if_construct.condition,
        { v for v in variables },
        truth=True,
    )

    useful_alternatives = [alt for alt in alternatives if alt]
    for idx, alternative in enumerate(useful_alternatives, start=1):
        alt_suffix = (
            f" alternative {idx}/{len(useful_alternatives)}"
            if len(useful_alternatives) > 1
            else ""
        )

        for name, expected_values in alternative.items():
            current_value = variables.get(name)
            for expected in expected_values:
                logger(
                    f"if line {if_construct.lineno}{alt_suffix}: "
                    f"{name} current={current_value!r}, expected {expected}"
                )

    return alternatives
