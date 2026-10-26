from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import TypeAlias

from spel.scripts.fortran_parser.spel_ast import (
    Expression,
    FuncExpression,
    Identifier,
    IfConstruct,
    InfixExpression,
    PrefixExpression,
    StringLiteral,
)


@dataclass(frozen=True)
class Expectation:
    variable: str
    constraint: str

    def variable_names(self) -> set[str]:
        return {self.variable}

    def to_fortran(self) -> str:
        if self.constraint == "False":
            return f".not. {self.variable}"
        elif self.constraint == "True":
            return self.variable
        op, _, rhs = self.constraint.partition(" ")
        if _is_string_literal(rhs) and op.lower() in _STRING_EQ | _STRING_NE:
            # the analysis lowercases sources, so compare strings case-insensitively
            streq = f"spel_streq({self.variable}, {rhs})"
            return streq if op.lower() in _STRING_EQ else f".not. {streq}"
        return f"{self.variable} {self.constraint}"


@dataclass(frozen=True)
class AllOf:
    items: tuple["ConditionExpectation", ...]

    def variable_names(self) -> set[str]:
        return {name for item in self.items for name in item.variable_names()}

    def to_fortran(self) -> str:
        if not self.items:  # no conditions: always true
            return ".true."
        return (
            "(" + " .and. ".join([f"{item.to_fortran()}" for item in self.items]) + ")"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, AllOf):
            return False
        my_set = frozenset(self.items)
        other_set = frozenset(other.items)
        return my_set == other_set

    def __hash__(self) -> int:
        return hash((AllOf, frozenset(self.items)))


@dataclass(frozen=True)
class AnyOf:
    items: tuple["ConditionExpectation", ...]

    def variable_names(self) -> set[str]:
        return {name for item in self.items for name in item.variable_names()}

    def to_fortran(self) -> str:
        if not self.items:
            return ".false."
        return (
            "(" + " .or. ".join([f"({item.to_fortran()})" for item in self.items]) + ")"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, AnyOf):
            return False
        my_set = frozenset(self.items)
        other_set = frozenset(other.items)
        return my_set == other_set

    def __hash__(self) -> int:
        return hash((AnyOf, frozenset(self.items)))


ConditionExpectation: TypeAlias = Expectation | AllOf | AnyOf

_STRING_EQ = {"==", "=", ".eq."}
_STRING_NE = {"/=", "!=", ".ne."}


def _is_string_literal(text: str) -> bool:
    return len(text) >= 2 and text[0] == text[-1] and text[0] in "'\""


def _operand_str(expr: Expression) -> str:
    """Fortran source for a comparison operand (StringLiteral.__str__ drops the quotes)"""
    if isinstance(expr, StringLiteral):
        return f"'{expr.value}'"
    return str(expr)


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
                return {Expectation(variable=var, constraint=constraint)}
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

NO_EXPECTATION = AllOf(())  # TRUE
IMPOSSIBLE_EXPECTATION = AnyOf(())  # False


def negate(
    condition: ConditionExpectation,
) -> ConditionExpectation:
    match condition:
        case Expectation(variable, constraint):
            if constraint == "True":
                return Expectation(variable, "False")

            if constraint == "False":
                return Expectation(variable, "True")

            parts = constraint.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Cannot negate constraint: {constraint!r}")

            op, rhs = parts

            try:
                negated_op = _NEGATED_COMPARISON[op.lower()]
            except KeyError:
                raise ValueError(f"Unknown comparison operator {op!r}") from None

            return Expectation(
                variable,
                f"{negated_op} {rhs}",
            )

        case AllOf(items):
            # !(A && B) == !A || !B
            return AnyOf(tuple(negate(item) for item in items))

        case AnyOf(items):
            # !(A || B) == !A && !B
            return AllOf(tuple(negate(item) for item in items))


def simplify(
    condition: ConditionExpectation,
) -> ConditionExpectation:
    # one pass can leave reducible terms, e.g. combining
    # (a .and. b) .or. (a .and. .not. b) .or. .not. a  ->  a .or. .not. a
    for _ in range(100):
        simplified = _simplify_once(condition)
        if simplified == condition:
            return simplified
        condition = simplified
    return condition


def _simplify_once(
    condition: ConditionExpectation,
) -> ConditionExpectation:
    match condition:
        case Expectation():
            return condition

        case AllOf(items):
            return _simplify_all_of(items)

        case AnyOf(items):
            return _simplify_any_of(items)

    raise TypeError(f"Unknown condition type: {type(condition)}")


def _are_complete(left: Expectation, right: Expectation) -> bool:
    """
    Checks if two Expectations for the same variable form the complete
    range of possible values for the variable
    """

    if left.variable != right.variable:
        return False

    if left.constraint in ["True", "False"]:
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
    """ """
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
                        f"{expected_op} {_operand_str(expr.right_expr)}",
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
                        f"{expected_op} {_operand_str(expr.left_expr)}",
                    )
                )

            return _all_of(*expectations)

    if isinstance(expr, Identifier) and expr.value in variables:
        return Expectation(expr.value, str(truth))

    # logical arrays/components, e.g. lun_pp%ispolygon(l) for the cascade var ispolygon
    if isinstance(expr, FuncExpression):
        expr = expr.function
    if isinstance(expr, Identifier) and "%" in expr.value:
        name = expr.value.split("%")[-1]
        if name in variables:
            return Expectation(name, str(truth))

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


def _simplify_all_of(
    items: tuple[ConditionExpectation, ...],
) -> ConditionExpectation:
    simplified: set[ConditionExpectation] = set()

    # Recursively simplify and flatten nested ANDs.
    for item in items:
        item = simplify(item)

        # A && False == False
        if item == IMPOSSIBLE_EXPECTATION:
            return IMPOSSIBLE_EXPECTATION

        # A && True == A
        if item == NO_EXPECTATION:
            continue

        if isinstance(item, AllOf):
            simplified.update(item.items)
        else:
            simplified.add(item)

    if not simplified:
        return NO_EXPECTATION

    # A && !A == False
    for item in simplified:
        if simplify(negate(item)) in simplified:
            return IMPOSSIBLE_EXPECTATION

    # Absorption:
    #
    # A && (A || B) == A
    absorbed: set[ConditionExpectation] = set()

    for item in simplified:
        if isinstance(item, AnyOf):
            if any(subitem in simplified for subitem in item.items):
                absorbed.add(item)

    simplified -= absorbed

    if len(simplified) == 1:
        return next(iter(simplified))

    return AllOf(tuple(simplified))


def _simplify_any_of(
    items: tuple[ConditionExpectation, ...],
) -> ConditionExpectation:
    simplified: set[ConditionExpectation] = set()

    # Recursively simplify and flatten nested ORs.
    for item in items:
        item = simplify(item)

        # A || True == True
        if item == NO_EXPECTATION:
            return NO_EXPECTATION

        # A || False == A
        if item == IMPOSSIBLE_EXPECTATION:
            continue

        if isinstance(item, AnyOf):
            simplified.update(item.items)
        else:
            simplified.add(item)

    if not simplified:
        return IMPOSSIBLE_EXPECTATION

    # A || !A == True
    for item in simplified:
        if simplify(negate(item)) in simplified:
            return NO_EXPECTATION

    # Absorption:
    #
    # A || (A && B) == A
    absorbed: set[ConditionExpectation] = set()

    for item in simplified:
        if isinstance(item, AllOf):
            if any(subitem in simplified for subitem in item.items):
                absorbed.add(item)

    simplified -= absorbed

    # (A && B) || (A && !B) -> A
    simplified = _combine_anyof_terms(simplified)

    if len(simplified) == 1:
        return next(iter(simplified))

    return AnyOf(tuple(simplified))


def _combine_anyof_terms(
    terms: set[ConditionExpectation],
) -> set[ConditionExpectation]:
    changed = True

    while changed:
        changed = False
        term_list = list(terms)

        for i, lhs in enumerate(term_list):
            for rhs in term_list[i + 1 :]:
                combined = _combine_complementary_terms(lhs, rhs)

                if combined is None:
                    continue

                terms.remove(lhs)
                terms.remove(rhs)
                terms.add(simplify(combined))

                changed = True
                break

            if changed:
                break

    return terms


def _combine_complementary_terms(
    lhs: ConditionExpectation,
    rhs: ConditionExpectation,
) -> ConditionExpectation | None:
    lhs_terms = set(lhs.items) if isinstance(lhs, AllOf) else {lhs}

    rhs_terms = set(rhs.items) if isinstance(rhs, AllOf) else {rhs}

    common = lhs_terms & rhs_terms

    lhs_remaining = lhs_terms - common
    rhs_remaining = rhs_terms - common

    # They must differ by exactly one term each.
    if len(lhs_remaining) != 1 or len(rhs_remaining) != 1:
        return None

    lhs_term = next(iter(lhs_remaining))
    rhs_term = next(iter(rhs_remaining))

    # Those differing terms must be complements.
    if simplify(negate(lhs_term)) != rhs_term:
        return None

    # X || !X -> True
    if not common:
        return NO_EXPECTATION

    # (A && X) || (A && !X) -> A
    if len(common) == 1:
        return next(iter(common))

    # (A && B && X) || (A && B && !X) -> A && B
    return AllOf(tuple(common))


def _atom_key(e: Expectation) -> tuple[tuple[str, str], bool]:
    """A boolean atom and the polarity of `e` in it: a logical flag (`x` / `.not. x`) or a
    comparison paired with its negation (`n .eq. 1` / `n .ne. 1`) under one canonical form."""
    if e.constraint in ("True", "False"):
        return (e.variable, "True"), e.constraint == "True"
    neg = negate(e).constraint
    canon = min(e.constraint, neg)
    return (e.variable, canon), e.constraint == canon


def _evaluate(condition: ConditionExpectation, values: dict) -> bool:
    match condition:
        case Expectation():
            key, positive = _atom_key(condition)
            return values[key] == positive
        case AllOf(items):
            return all(_evaluate(i, values) for i in items)
        case AnyOf(items):
            return any(_evaluate(i, values) for i in items)
    raise TypeError(f"Unknown condition type: {type(condition)}")


def _atoms(condition: ConditionExpectation, out: dict) -> dict:
    match condition:
        case Expectation():
            out.setdefault(_atom_key(condition)[0], None)
        case AllOf(items) | AnyOf(items):
            for i in items:
                _atoms(i, out)
    return out


def minimize(condition: ConditionExpectation, max_atoms: int = 12) -> ConditionExpectation:
    """
    An equivalent minimal sum of products (Quine-McCluskey over the truth table of the
    condition's atoms), e.g. `use_cn .or. use_fates .or. (use_c13 .and. use_cn)` ->
    `use_cn .or. use_fates`. Comparisons count as independent atoms, so the result is
    equivalent on every assignment (minimal up to their dependence). Conditions with
    more than `max_atoms` atoms are returned unchanged.
    """
    keys = list(_atoms(condition, {}))
    n = len(keys)
    if n > max_atoms:
        return condition
    minterms = []
    for bits in range(1 << n):
        values = {k: bool(bits >> i & 1) for i, k in enumerate(keys)}
        if _evaluate(condition, values):
            minterms.append(tuple(bool(bits >> i & 1) for i in range(n)))
    if not minterms:
        return IMPOSSIBLE_EXPECTATION
    if len(minterms) == 1 << n:
        return NO_EXPECTATION
    # prime implicants: merge terms that differ in one position (None = don't care)
    terms = set(minterms)
    primes: set[tuple] = set()
    while terms:
        merged: set[tuple] = set()
        used: set[tuple] = set()
        tl = list(terms)
        for a, b in combinations(tl, 2):
            diff = [i for i in range(n) if a[i] != b[i]]
            if len(diff) == 1 and a[diff[0]] is not None and b[diff[0]] is not None:
                merged.add(tuple(None if i == diff[0] else a[i] for i in range(n)))
                used |= {a, b}
        primes |= terms - used
        terms = merged

    def covers(p, m):
        return all(p[i] is None or p[i] == m[i] for i in range(n))

    # essential primes first, then greedily the prime covering most remaining minterms
    left = set(minterms)
    chosen: list[tuple] = []
    for m in minterms:
        cov = [p for p in primes if covers(p, m)]
        if len(cov) == 1 and cov[0] not in chosen:
            chosen.append(cov[0])
    left -= {m for m in left if any(covers(p, m) for p in chosen)}
    while left:
        best = max(sorted(primes, key=str), key=lambda p: (sum(covers(p, m) for m in left), -sum(x is not None for x in p)))
        chosen.append(best)
        left -= {m for m in left if covers(best, m)}

    def literal(key, positive: bool) -> Expectation:
        var, constraint = key
        if constraint == "True":
            return Expectation(var, "True" if positive else "False")
        e = Expectation(var, constraint)
        return e if positive else negate(e)

    products = []
    for p in sorted(chosen, key=lambda p: (sum(x is not None for x in p), str(p))):
        lits = [literal(keys[i], p[i]) for i in range(n) if p[i] is not None]
        products.append(lits[0] if len(lits) == 1 else AllOf(tuple(lits)))
    return products[0] if len(products) == 1 else AnyOf(tuple(products))
