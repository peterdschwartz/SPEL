from spel.scripts.fortran_parser.boolen_expression import (
    AllOf,
    AnyOf,
    ConditionExpectation,
    Expectation,
    expected_constraints,
    infer_condition_expectations,
    log_if_condition_expectations,
)
from spel.scripts.fortran_parser.spel_ast import (
    BlockStatement,
    FloatLiteral,
    Identifier,
    IfConstruct,
    InfixExpression,
    PrefixExpression,
)
from spel.scripts.fortran_parser.tokens import Token, TokenTypes

tok = Token(token=TokenTypes.EOF, literal="temp")


def test_infer_condition_expectations_steps_boolean_expression():
    # (temperature > 273.15 .and. active) .or. .not. frozen
    expr = InfixExpression(
        tok=tok,
        op=".or.",
        left=InfixExpression(
            tok=tok,
            op=".and.",
            left=InfixExpression(
                tok=tok,
                op=">",
                left=Identifier(tok=tok, value="temperature"),
                right=FloatLiteral(tok=tok, val=273.15, prec=""),
            ),
            right=Identifier(tok=tok, value="active"),
        ),
        right=PrefixExpression(
            tok=tok,
            op=".not.",
            right=Identifier(tok=tok, value="frozen"),
        ),
    )

    tracked_variables = {
        "temperature": 250.0,
        "active": False,
        "frozen": True,
        "ignored": 1,
    }

    # (temperature > 273.15 .and. active) .or. .not. frozen
    expectations = infer_condition_expectations(
        expr,
        tracked_variables,
        truth=True,
    )

    print("constraints for temperature: ", expected_constraints(expectations,'frozen'))

    answer = AnyOf(
        (
            AllOf(
                (
                    Expectation("temperature", "> 273.15"),
                    Expectation("active", "True"),
                )
            ),
            Expectation("frozen","False"),
        )
    )

    assert expectations == answer


def test_log_if_condition_expectations_logs_each_true_alternative():
    """"""
    messages = []

    if_construct = IfConstruct(
        tok=tok,
        cond=InfixExpression(
            tok=tok,
            op=".or.",
            left=Identifier(tok=tok, value="active"),
            right=PrefixExpression(
                tok=tok,
                op=".not.",
                right=Identifier(tok=tok, value="frozen"),
            ),
        ),
        consequence=BlockStatement(tok=tok),
    )

    if_construct.lineno = 42
    log_if_condition_expectations(
        if_construct,
        {"active": False, "frozen": True},
        logger=messages.append,
    )

    assert messages == [
        "if line 42 alternative 1/2: active current=False, expected True",
        "if line 42 alternative 2/2: frozen current=True, expected False",
    ]

def test_boolean_equality():

    expr_1 = AnyOf(
        (
            Expectation("frozen","False"),
            AllOf(
                (
                    Expectation("temperature", "> 273.15"),
                    Expectation("active", "True"),
                    Expectation("blah","False"),
                )
            ),
            Expectation("this", "test"),
        )
    )

    expr_2 = AnyOf(
        (
            AllOf(
                (
                    Expectation("temperature", "> 273.15"),
                    Expectation("blah","False"),
                    Expectation("active", "True"),
                )
            ),
            Expectation("this", "test"),
            Expectation("frozen","False"),
        )
    )

    assert expr_1 == expr_2

