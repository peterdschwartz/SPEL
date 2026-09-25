import pytest

from spel.scripts.fortran_parser.boolen_expression import (
    IMPOSSIBLE_EXPECTATION,
    NO_EXPECTATION,
    AllOf,
    AnyOf,
    Expectation,
    negate,
    simplify,
)


def test_negate_true_expectation():
    cond = Expectation("option", "True")

    assert negate(cond) == Expectation("option", "False")


def test_negate_false_expectation():
    cond = Expectation("option", "False")

    assert negate(cond) == Expectation("option", "True")


@pytest.mark.parametrize(
    ("constraint", "expected"),
    [
        ("== 1", "/= 1"),
        ("= 1", "/= 1"),
        (".eq. 1", ".ne. 1"),
        ("/= 1", "== 1"),
        ("!= 1", "== 1"),
        (".ne. 1", ".eq. 1"),
        ("> 1", "<= 1"),
        (".gt. 1", ".le. 1"),
        (">= 1", "< 1"),
        (".ge. 1", ".lt. 1"),
        ("< 1", ">= 1"),
        (".lt. 1", ".ge. 1"),
        ("<= 1", "> 1"),
        (".le. 1", ".gt. 1"),
    ],
)
def test_negate_comparison(constraint, expected):
    cond = Expectation("nlev", constraint)

    assert negate(cond) == Expectation("nlev", expected)


def test_double_negation():
    cond = Expectation("option", "True")

    assert negate(negate(cond)) == cond


def test_negate_allof_uses_demorgan():
    a = Expectation("a", "True")
    b = Expectation("b", "True")

    cond = AllOf((a, b))

    assert negate(cond) == AnyOf(
        (
            Expectation("a", "False"),
            Expectation("b", "False"),
        )
    )


def test_negate_anyof_uses_demorgan():
    a = Expectation("a", "True")
    b = Expectation("b", "True")

    cond = AnyOf((a, b))

    assert negate(cond) == AllOf(
        (
            Expectation("a", "False"),
            Expectation("b", "False"),
        )
    )


def test_double_negation_nested_expression():
    cond = AnyOf(
        (
            Expectation("a", "True"),
            AllOf(
                (
                    Expectation("b", "True"),
                    Expectation("c", "False"),
                )
            ),
        )
    )

    assert negate(negate(cond)) == cond


def test_simplify_expectation_is_identity():
    cond = Expectation("a", "True")

    assert simplify(cond) == cond


def test_simplify_empty_allof_is_true():
    assert simplify(AllOf(())) == NO_EXPECTATION


def test_simplify_empty_anyof_is_false():
    assert simplify(AnyOf(())) == IMPOSSIBLE_EXPECTATION


def test_simplify_allof_removes_true():
    a = Expectation("a", "True")

    cond = AllOf((a, NO_EXPECTATION))

    assert simplify(cond) == a


def test_simplify_anyof_removes_false():
    a = Expectation("a", "True")

    cond = AnyOf((a, IMPOSSIBLE_EXPECTATION))

    assert simplify(cond) == a


def test_simplify_allof_with_false_is_false():
    a = Expectation("a", "True")

    cond = AllOf((a, IMPOSSIBLE_EXPECTATION))

    assert simplify(cond) == IMPOSSIBLE_EXPECTATION


def test_simplify_anyof_with_true_is_true():
    a = Expectation("a", "True")

    cond = AnyOf((a, NO_EXPECTATION))

    assert simplify(cond) == NO_EXPECTATION


def test_simplify_duplicate_and_terms():
    a = Expectation("a", "True")

    cond = AllOf((a, a))

    assert simplify(cond) == a


def test_simplify_duplicate_or_terms():
    a = Expectation("a", "True")

    cond = AnyOf((a, a))

    assert simplify(cond) == a


def test_simplify_flattens_nested_allof():
    a = Expectation("a", "True")
    b = Expectation("b", "True")
    c = Expectation("c", "True")

    cond = AllOf(
        (
            a,
            AllOf((b, c)),
        )
    )

    assert simplify(cond) == AllOf((a, b, c))


def test_simplify_flattens_nested_anyof():
    a = Expectation("a", "True")
    b = Expectation("b", "True")
    c = Expectation("c", "True")

    cond = AnyOf(
        (
            a,
            AnyOf((b, c)),
        )
    )

    assert simplify(cond) == AnyOf((a, b, c))


def test_simplify_a_and_not_a_is_false():
    cond = AllOf(
        (
            Expectation("option", "True"),
            Expectation("option", "False"),
        )
    )

    assert simplify(cond) == IMPOSSIBLE_EXPECTATION


def test_simplify_a_or_not_a_is_true():
    cond = AnyOf(
        (
            Expectation("option", "True"),
            Expectation("option", "False"),
        )
    )

    assert simplify(cond) == NO_EXPECTATION


def test_simplify_comparison_and_negation_is_false():
    a = Expectation("nlev", "> 1")

    cond = AllOf(
        (
            a,
            negate(a),
        )
    )

    assert simplify(cond) == IMPOSSIBLE_EXPECTATION


def test_simplify_comparison_or_negation_is_true():
    a = Expectation("nlev", "> 1")

    cond = AnyOf(
        (
            a,
            negate(a),
        )
    )

    assert simplify(cond) == NO_EXPECTATION


def test_simplify_or_absorption():
    """
    A OR (A AND B) -> A
    """
    a = Expectation("a", "True")
    b = Expectation("b", "True")

    cond = AnyOf(
        (
            a,
            AllOf((a, b)),
        )
    )

    assert simplify(cond) == a


def test_simplify_and_absorption():
    """
    A AND (A OR B) -> A
    """
    a = Expectation("a", "True")
    b = Expectation("b", "True")

    cond = AllOf(
        (
            a,
            AnyOf((a, b)),
        )
    )

    assert simplify(cond) == a


def test_simplify_complementary_terms():
    """
    (A AND B) OR (A AND NOT B) -> A
    """
    a = Expectation("a", "True")
    b = Expectation("b", "True")
    not_b = Expectation("b", "False")

    cond = AnyOf(
        (
            AllOf((a, b)),
            AllOf((a, not_b)),
        )
    )

    assert simplify(cond) == a


def test_simplify_complementary_terms_with_multiple_common_terms():
    """
    (A AND B AND C) OR (A AND B AND NOT C) -> A AND B
    """
    a = Expectation("a", "True")
    b = Expectation("b", "True")
    c = Expectation("c", "True")
    not_c = Expectation("c", "False")

    cond = AnyOf(
        (
            AllOf((a, b, c)),
            AllOf((a, b, not_c)),
        )
    )

    assert simplify(cond) == AllOf((a, b))


def test_simplify_complementary_single_terms_to_true():
    """
    A OR NOT A -> True
    """
    a = Expectation("a", "True")

    cond = AnyOf((a, negate(a)))

    assert simplify(cond) == NO_EXPECTATION


def test_simplify_nested_expression_recursively():
    """
    A OR ((A AND B) OR False) -> A
    """
    a = Expectation("a", "True")
    b = Expectation("b", "True")

    cond = AnyOf(
        (
            a,
            AnyOf(
                (
                    AllOf((a, b)),
                    IMPOSSIBLE_EXPECTATION,
                )
            ),
        )
    )

    assert simplify(cond) == a


def test_simplify_is_idempotent():
    cond = AnyOf(
        (
            AllOf(
                (
                    Expectation("a", "True"),
                    Expectation("b", "True"),
                )
            ),
            AllOf(
                (
                    Expectation("a", "True"),
                    Expectation("b", "False"),
                )
            ),
        )
    )

    once = simplify(cond)
    twice = simplify(once)

    assert twice == once


def test_simplification_preserves_unrelated_or_terms():
    a = Expectation("a", "True")
    b = Expectation("b", "True")

    cond = AnyOf((a, b))

    assert simplify(cond) == AnyOf((a, b))


def test_simplification_preserves_unrelated_and_terms():
    a = Expectation("a", "True")
    b = Expectation("b", "True")

    cond = AllOf((a, b))

    assert simplify(cond) == AllOf((a, b))
