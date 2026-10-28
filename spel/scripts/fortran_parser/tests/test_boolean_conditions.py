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



def test_string_comparison_guard_is_quoted_fortran():
    # nu_com .eq. 'rd'  (analysis lowercases sources, so the literal is 'rd')
    from spel.scripts.fortran_parser.spel_ast import StringLiteral

    expr = InfixExpression(
        tok=tok,
        op=".eq.",
        left=Identifier(tok=tok, value="nu_com"),
        right=StringLiteral(tok=tok, val="rd"),
    )
    variables = {"nu_com"}
    assert infer_condition_expectations(expr, variables, True) == Expectation(
        "nu_com", ".eq. 'rd'"
    )
    assert (
        infer_condition_expectations(expr, variables, True).to_fortran()
        == "spel_streq(nu_com, 'rd')"
    )
    assert (
        infer_condition_expectations(expr, variables, False).to_fortran()
        == ".not. spel_streq(nu_com, 'rd')"
    )


def test_empty_condition_renders_as_valid_fortran():
    # an unconditional access alongside a guarded one: (guard) .or. (always)
    guard = AllOf((Expectation("tw_irr", "True"), Expectation("irrigate", "True")))
    assert AnyOf((guard, AllOf(()))).to_fortran() == "(((tw_irr .and. irrigate)) .or. (.true.))"
    assert AnyOf(()).to_fortran() == ".false."


def test_simplify_reaches_fixed_point():
    # CanopyHydrology accesses qflx_supply under every irrigate/tw_irr branch
    from spel.scripts.fortran_parser.boolen_expression import NO_EXPECTATION, simplify

    irr, twi, gw = (Expectation(v, "True") for v in ("irrigate", "tw_irr", "extra_gw_irr"))
    branches = AnyOf(
        (
            AllOf((irr, twi)),
            AllOf((irr, twi, gw)),
            AllOf((irr, Expectation("tw_irr", "False"))),
            Expectation("irrigate", "False"),
        )
    )
    assert simplify(branches) == NO_EXPECTATION


def test_minimize_absorbs_and_merges():
    from spel.scripts.fortran_parser.boolen_expression import (
        AllOf, AnyOf, Expectation, NO_EXPECTATION, _evaluate, _atoms, minimize,
    )
    import itertools

    cn, fates, c13 = (Expectation(v, "True") for v in ("use_cn", "use_fates", "use_c13"))
    not_fates = Expectation("use_fates", "False")
    spin = Expectation("spinup_state", ".eq. 1")
    # absorption: c13_col_cf's guard in the device data layer
    cond = AnyOf((cn, fates, AllOf((c13, AnyOf((fates, cn)))), AllOf((not_fates, c13, AnyOf((fates, cn))))))
    assert minimize(cond) == AnyOf((cn, fates))
    # merging: (a and b) or (a and not b) -> a; a or not a -> true
    assert minimize(AnyOf((AllOf((cn, c13)), AllOf((cn, Expectation("use_c13", "False")))))) == cn
    assert minimize(AnyOf((fates, not_fates))) == NO_EXPECTATION
    # comparisons keep their operator; the result is equivalent on every assignment
    cond = AnyOf((AllOf((spin, cn)), AllOf((negate_spin := Expectation("spinup_state", ".ne. 1"), cn)), c13))
    m = minimize(cond)
    assert m == AnyOf((cn, c13))
    cond = AnyOf((AllOf((spin, fates)), AllOf((cn, c13)), AllOf((spin, cn, c13))))
    m = minimize(cond)
    keys = list(_atoms(cond, {}))
    for bits in itertools.product((False, True), repeat=len(keys)):
        vals = dict(zip(keys, bits))
        assert _evaluate(cond, vals) == _evaluate(m, vals)
    assert m.to_fortran().count(".and.") == 2


def test_minimize_does_not_depend_on_the_input_order():
    # the device data layer's guards were regenerated with their terms in a different order
    from spel.scripts.fortran_parser.boolen_expression import AllOf, AnyOf, Expectation, minimize

    lch4, vert, fates, cn = (Expectation(v, "True") for v in ("use_lch4", "use_vertsoilc", "use_fates", "use_cn"))
    a = AnyOf((lch4, AllOf((vert, fates)), AllOf((vert, cn))))
    b = AnyOf((AllOf((cn, vert)), lch4, AllOf((fates, vert))))
    assert minimize(a).to_fortran() == minimize(b).to_fortran()
