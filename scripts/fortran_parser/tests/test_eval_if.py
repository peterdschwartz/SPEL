from pprint import pprint

from scripts.analyze_subroutines import Subroutine
from scripts.export_objects import unpickle_unit_test
from scripts.fortran_parser import lexer
from scripts.fortran_parser.evaluate_ifs import eval_if_condition
from scripts.fortran_parser.spel_ast import ExpressionStatement
from scripts.fortran_parser.spel_parser import Parser
from scripts.types import LineTuple, LogicalLineIterator


def test_ifs():
    mod_dict, sub_dict, type_dict = unpickle_unit_test("a812c11")
    env_strings: dict[str, str] = {
        "use_lch4": ".false.",
        "no_frozen_nitrif_denitrif": ".true.",
        "nu_com": "RD",
    }
    line = [LineTuple(line=f"{key} = {val}", ln=1) for key, val in env_strings.items()]

    lex = lexer.Lexer(LogicalLineIterator(line))
    parser = Parser(lex=lex)
    program = parser.parse_program()
    env = {}
    for stmt in program.statements:
        varname = stmt.expression.left_expr.value
        value = stmt.expression.right_expr.value
        env[varname] = value

    for sub in sub_dict.values():
        assert isinstance(sub, Subroutine)
        for if_ in sub.flat_ifs:
            if if_.nml_vars:
                print(str(if_.condition), eval_if_condition(if_.condition, env))

    return
