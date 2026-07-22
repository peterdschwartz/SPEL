import os
import sys
import unittest
from pprint import pprint

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

import scripts.fortran_parser.lexer as lexer
from scripts.fortran_parser.spel_parser import Parser
from scripts.fortran_parser.tokens import Token, TokenTypes


class LexTests(unittest.TestCase):

    def test_tokens(self):
        input = """x = 5.d0
        call func(1._r8,2)
        """

        expected_tokens = [
            # line 1
            Token(token=TokenTypes.IDENT, literal="x"),
            Token(token=TokenTypes.ASSIGN, literal="="),
            Token(token=TokenTypes.FLOAT, literal="5.d0"),
            Token(token=TokenTypes.NEWLINE, literal="\n"),
            # line2
            Token(token=TokenTypes.CALL, literal="call"),
            Token(token=TokenTypes.IDENT, literal="func"),
            Token(token=TokenTypes.LPAREN, literal="("),
            Token(token=TokenTypes.FLOAT, literal="1._r8"),
            Token(token=TokenTypes.COMMA, literal=","),
            Token(token=TokenTypes.INT, literal="2"),
            Token(token=TokenTypes.RPAREN, literal=")"),
            Token(token=TokenTypes.NEWLINE, literal="\n"),
        ]

        lex = lexer.Lexer(input)

        for expected_tok in expected_tokens:
            tok = lex.next_token()
            with self.subTest(ans=expected_tok):
                self.assertEqual(tok, expected_tok)

    def test_parser(self):

        input = """1+2*x_1
        -2*_x%y-1/2
        (x-y)/(2*y+1)
        add(1,min(2*x,arg=4.0))
        """
        lex = lexer.Lexer(input)
        parser = Parser(lex=lex)
        program = parser.parse_program()

        expected_stmts = [
            "(1+(2*x_1))",
            "(((-2)*_x%y)-(1/2))",
            "((x-y)/((2*y)+1))",
            "[add(1,[min((2*x),(arg=4.0))])]",
        ]

        if parser.errors:
            for err in parser.errors:
                print("err: ", err)

        for n, ans in enumerate(expected_stmts):
            with self.subTest(ans=ans):
                stmt = str(program.statements[n])
                self.assertEqual(stmt, ans)

        new_input = """call dynamic_plant_alloc(min(1.0_r8-N_lim_factor(p),1.0_r8-P_lim_factor(p)),W_lim_factor(p),laisun(p)+laisha(p), allocation_leaf(p), allocation_stem(p), allocation_froot(p), woody(ivt(p)))"""

        newlex = lexer.Lexer(new_input)
        parser.lexer = newlex
        parser.next_token()
        parser.next_token()
        # parser = Parser(lex=newlex)
        program1 = parser.parse_program()

        for stmt in program1.statements:
            pprint(stmt.to_dict(), sort_dicts=False)


if __name__ == "__main__":
    unittest.main()
