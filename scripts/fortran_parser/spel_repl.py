import os
from pprint import pprint

from formatting import ast_to_plantuml

import scripts.fortran_parser.lexer as lexer
from scripts.fortran_parser.spel_parser import Parser
from scripts.fortran_parser.tracing import Trace


def start_repl():
    os.system("banner SPEL REPL")

    Trace.enabled = True
    while True:
        user_input = input(">>> ")
        lex = lexer.Lexer(user_input)
        parser = Parser(lex=lex)

        program = parser.parse_program()

        for stmt in program.statements:
            print(stmt)
            pprint(stmt.to_dict(), sort_dicts=False)
            puml = ast_to_plantuml(stmt.to_dict())
            with open("spel_ast.puml", "w") as f:
                f.write(puml)


if __name__ == "__main__":
    start_repl()
