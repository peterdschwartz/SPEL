from pprint import pprint

from scripts.analyze_subroutines import Subroutine
from scripts.export_objects import unpickle_unit_test
from scripts.fortran_parser import lexer
from scripts.fortran_parser.evaluate_ifs import eval_if_condition
from scripts.fortran_parser.spel_ast import (
    ExpressionStatement,
    PrefixExpression,
    expr_from_json,
    expr_to_json,
)
from scripts.fortran_parser.spel_parser import Parser
from scripts.types import LineTuple, LogicalLineIterator


def test_ifs():
    mod_dict, sub_dict, type_dict = unpickle_unit_test("4637ab7")
    env_strings: dict[str, str] = {
        "use_lch4": ".false.",
        "no_frozen_nitrif_denitrif": ".true.",
        "nu_com": "eca",
        "do_budgets": ".false.",
    }
    line = [LineTuple(line=f"{key} = {val}", ln=1) for key, val in env_strings.items()]

    def _get_expr_value(expr):
        if isinstance(expr, PrefixExpression):
            if expr.operator == "-":
                return -1 * expr.right_expr.value
            else:
                return expr.right_expr.value
        else:
            # should jsut be a xxLiteral
            return expr.value

    lex = lexer.Lexer(LogicalLineIterator(line))
    parser = Parser(lex=lex)
    program = parser.parse_program()
    env = {
        stmt.expression.left_expr.value: _get_expr_value(stmt.expression.right_expr)
        for stmt in program.statements
    }
    for sub in sub_dict.values():
        assert isinstance(sub, Subroutine)
        for if_ in sub.flat_ifs:
            cond = if_.condition
            rebuilt = assert_expr_roundtrip(cond)
            if if_.nml_vars:
                active_og = eval_if_condition(if_.condition, env)
                active_rebuilt = eval_if_condition(rebuilt, env)
                assert active_og == active_rebuilt, (
                    f"Eval mismatch for {str(cond)}" f"\n   rebuilt: {str(rebuilt)}"
                )
                if (
                    "nu_com " in str(if_.condition)
                    and sub.name == "soillittdecompalloc2"
                ):
                    print("=" * 10)
                    print("nu_com = ", env.get("nu_com"))
                    print(str(if_.condition), active_og)
                    print(str(rebuilt), active_rebuilt)
                    print(if_.condition.to_dict())
                    print(rebuilt.to_dict())

    return


def assert_expr_roundtrip(expr):
    """Assert structure and printable form survive JSON round-trip."""
    js = expr_to_json(expr)
    rebuilt = expr_from_json(js)

    # Structural equality: dicts must match exactly
    assert (
        rebuilt is None if expr is None else rebuilt.to_dict() == expr.to_dict()
    ), f"Mismatch:\n OG:= {expr.to_dict()}\n RB:= {rebuilt.to_dict()}"

    # String form equality (after you improve __str__ with precedence parens)
    if expr is not None:
        assert str(rebuilt) == str(
            expr
        ), f"Pretty-print mismatch:\n{str(rebuilt)}\nvs\n{str(expr)}"

    return rebuilt
