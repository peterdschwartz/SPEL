from __future__ import annotations
from collections.abc import Iterable
from itertools import combinations

from dataclasses import dataclass
from typing import TypeAlias

from spel.scripts.fortran_parser.spel_ast import (
    Expression,
    Identifier,
    IfConstruct,
    InfixExpression,
    PrefixExpression,
)


@dataclass(frozen=True)
class Expectation:
    variable: str
    constraint: str

    def to_fortran(self)->str:
        if self.constraint == 'False':
            return f".not. {self.variable}"
        elif self.constraint == 'True':
            return self.variable
        else:
            return f"{self.variable} {self.constraint}"

@dataclass(frozen=True)
class AllOf:
    items: tuple["ConditionExpectation", ...]

    def to_fortran(self)->str:
        return ' .and. '.join([item.to_fortran() for item in self.items])

    def __eq__(self,other)->bool:
        if not isinstance(other,AllOf):
            return False
        my_set = frozenset(self.items)
        other_set = frozenset(other.items)
        return my_set == other_set

    def __hash__(self)->int:
        return hash((AllOf,frozenset(self.items)))


@dataclass(frozen=True)
class AnyOf:
    items: tuple["ConditionExpectation", ...]

    def to_fortran(self)->str:
        return ' .or. '.join([item.to_fortran() for item in self.items])

    def __eq__(self,other)->bool:
        if not isinstance(other,AnyOf):
            return False
        my_set = frozenset(self.items)
        other_set = frozenset(other.items)
        return my_set == other_set
    
    def __hash__(self)->int:
        return hash((AnyOf,frozenset(self.items)))




ConditionExpectation: TypeAlias = Expectation | AllOf | AnyOf


def expected_constraints(
    condition: ConditionExpectation,
    variable: str,
) -> set[Expectation]:
    """
    Helper function to pick out the constraints from a specific variable of interest
    from a ConditionExpectation instance
    """
    match condition:
        case Expectation(var, constraint):
            if var == variable:
                return {Expectation(variable=var,constraint=constraint)}
            return set()

        case AllOf(items):
            constraints: set[Expectation] = set()
            for item in items:
                constraints |= expected_constraints(item, variable)
            return constraints

        case AnyOf(items):
            constraints: set[Expectation] = set()
            for item in items:
                constraints |= expected_constraints(item, variable)
            return constraints


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

NO_EXPECTATION = AllOf(())



def simplify_expectations(items: set[Expectation])->ConditionExpectation:
    """
    """
    res: set[Expectation] = items.copy()
    to_remove: set[Expectation] = set()
    for left, right in combinations(items,2):
        if _are_complete(left,right):
            to_remove.add(left)
            to_remove.add(right)

    res.difference_update(to_remove)
    return _any_of(*res)

def _are_complete(left: Expectation, right: Expectation)->bool:
    """
    Checks if two Expectations for the same variable form the complete
    range of possible values for the variable
    """

    if left.variable != right.variable:
        return False

    if left.constraint in ['True', 'False']:
        lvalue = left.constraint
        rvalue = right.constraint
        return not (rvalue == lvalue)
    return False


def _negate_comparison_operator(op: str) -> str:
    return _NEGATED_COMPARISON.get(op.lower(), f".not. ({op})")


def _reverse_comparison_operator(op: str) -> str:
    return _REVERSED_COMPARISON.get(op.lower(), op)


def _is_no_expectation(expectation: ConditionExpectation) -> bool:
    return isinstance(expectation, AllOf) and not expectation.items


def _all_of(*items: ConditionExpectation) -> ConditionExpectation:
    """
    """
    flattened: list[ConditionExpectation] = []

    for item in items:
        if _is_no_expectation(item):
            continue

        if isinstance(item, AllOf):
            flattened.extend(item.items)
        else:
            flattened.append(item)

    if not flattened:
        return NO_EXPECTATION

    if len(flattened) == 1:
        return flattened[0]

    return AllOf(tuple(flattened))


def _any_of(*items: ConditionExpectation) -> ConditionExpectation:
    """
    _any_of(*A):
    * For A = _NO_EXPECTATION, returns _NO_EXPECTATION
    * For A = A, AnyOf(B,C), return AnyOf(A,B,C)
    else returns AnyOf(A,X)
    """
    flattened: list[ConditionExpectation] = []

    for item in items:
        if isinstance(item, AnyOf):
            flattened.extend(item.items)
        else:
            flattened.append(item)

    if not flattened:
        return NO_EXPECTATION

    if len(flattened) == 1:
        return flattened[0]

    return AnyOf(tuple(flattened))


def _expectation_alternatives(
    expectation: ConditionExpectation,
) -> tuple[tuple[Expectation, ...], ...]:
    if isinstance(expectation, Expectation):
        return ((expectation,),)

    if isinstance(expectation, AnyOf):
        alternatives: list[tuple[Expectation, ...]] = []
        for item in expectation.items:
            alternatives.extend(_expectation_alternatives(item))
        return tuple(alternatives)

    alternatives = [()]
    for item in expectation.items:
        item_alternatives = _expectation_alternatives(item)
        alternatives = [
            alternative + item_alternative
            for alternative in alternatives
            for item_alternative in item_alternatives
        ]

    return tuple(alternatives)


def infer_condition_expectations(
    expr: Expression,
    variables: set[str],
    truth: bool = True,
) -> ConditionExpectation:
    """
    Return tracked-variable expectations required for an expression.

    `AllOf` means every child expectation must hold.
    `AnyOf` means any child expectation may make the condition hold.
    `AllOf(())` means no useful tracked-variable expectation was inferred.

    Example:
        a > 3 .and. flag

    returns:
        AllOf((
            Expectation("a", "> 3"),
            Expectation("flag", "True"),
        ))

    Example:
        a > 3 .or. flag

    returns:
        AnyOf((
            Expectation("a", "> 3"),
            Expectation("flag", "True"),
        ))
    """
    if isinstance(expr, PrefixExpression) and expr.operator.lower() in _LOGICAL_NOT:
        return infer_condition_expectations(expr.right_expr, variables, not truth)

    if isinstance(expr, InfixExpression):
        op = expr.operator.lower()

        if op in _LOGICAL_AND:
            left = infer_condition_expectations(expr.left_expr, variables, truth)
            right = infer_condition_expectations(expr.right_expr, variables, truth)

            if truth:
                return _all_of(left, right)

            return _any_of(left, right)

        if op in _LOGICAL_OR:
            left = infer_condition_expectations(expr.left_expr, variables, truth)
            right = infer_condition_expectations(expr.right_expr, variables, truth)

            if truth:
                return _any_of(left, right)

            return _all_of(left, right)

        if op in _NEGATED_COMPARISON:
            expectations: list[ConditionExpectation] = []

            if (
                isinstance(expr.left_expr, Identifier)
                and expr.left_expr.value in variables
            ):
                expected_op = op if truth else _negate_comparison_operator(op)
                expectations.append(
                    Expectation(
                        expr.left_expr.value,
                        f"{expected_op} {expr.right_expr}",
                    )
                )

            if (
                isinstance(expr.right_expr, Identifier)
                and expr.right_expr.value in variables
            ):
                reversed_op = _reverse_comparison_operator(op)
                expected_op = (
                    reversed_op if truth else _negate_comparison_operator(reversed_op)
                )
                expectations.append(
                    Expectation(
                        expr.right_expr.value,
                        f"{expected_op} {expr.left_expr}",
                    )
                )

            return _all_of(*expectations)

    if isinstance(expr, Identifier) and expr.value in variables:
        return Expectation(expr.value, str(truth))

    return NO_EXPECTATION


def log_if_condition_expectations(
    if_construct: IfConstruct,
    variables: dict[str, object],
    logger=print,
) -> ConditionExpectation:
    """
    Log tracked-variable expectations required for an IfConstruct condition
    to evaluate True.
    """
    expectation = infer_condition_expectations(
        if_construct.condition,
        set(variables),
        truth=True,
    )

    alternatives = _expectation_alternatives(expectation)
    useful_alternatives = [alternative for alternative in alternatives if alternative]

    for idx, alternative in enumerate(useful_alternatives, start=1):
        alt_suffix = (
            f" alternative {idx}/{len(useful_alternatives)}"
            if len(useful_alternatives) > 1
            else ""
        )

        for item in alternative:
            current_value = variables.get(item.variable)
            logger(
                f"if line {if_construct.lineno}{alt_suffix}: "
                f"{item.variable} current={current_value!r}, "
                f"expected {item.constraint}"
            )

    return expectation
