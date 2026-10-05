import unittest
from pprint import pprint

import spel.scripts.fortran_parser.lexer as lexer
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.fortran_parser.tokens import Token, TokenTypes
from spel.scripts.types import LineTuple, LogicalLineIterator


def make_lexer(text: str) -> lexer.Lexer:
    lines = [LineTuple(line=line, ln=i) for i, line in enumerate(text.splitlines())]
    return lexer.Lexer(line_it=LogicalLineIterator(lines))


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

        lex = make_lexer(input)

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
        lex = make_lexer(input)
        parser = Parser(lex=lex)
        program = parser.parse_program()

        expected_stmts = [
            "(1+(2*x_1))",
            "(((-2)*_x%y)-(1/2))",
            "((x-y)/((2*y)+1))",
            "add(1,min((2*x),(arg=4.0)))",
        ]

        if parser.errors:
            for err in parser.errors:
                print("err: ", err)

        for n, ans in enumerate(expected_stmts):
            with self.subTest(ans=ans):
                stmt = str(program.statements[n])
                self.assertEqual(stmt, ans)

        new_input = """call dynamic_plant_alloc(min(1.0_r8-N_lim_factor(p),1.0_r8-P_lim_factor(p)),W_lim_factor(p),laisun(p)+laisha(p), allocation_leaf(p), allocation_stem(p), allocation_froot(p), woody(ivt(p)))"""

        newlex = make_lexer(new_input)
        parser.lexer = newlex
        parser.next_token()
        parser.next_token()
        # parser = Parser(lex=newlex)
        program1 = parser.parse_program()

        for stmt in program1.statements:
            pprint(stmt.to_dict(), sort_dicts=False)

    def test_number_before_dot_operator(self):
        """`0.and.` is INT 0 then .and.; the '.' only belongs to real literals"""
        cases = {
            "pio_stride>0.and.n<0": ["pio_stride", ">", "0", ".and.", "n", "<", "0"],
            "x.eq.1.or.y": ["x", ".eq.", "1", ".or.", "y"],
            "a=1.eq.b": ["a", "=", "1", ".eq.", "b"],
            "a=2..not.b": ["a", "=", "2.", ".not.", "b"],
            "a=1.e5+1.d-3*1._r8-1.5": ["a", "=", "1.e5", "+", "1.d-3", "*", "1._r8", "-", "1.5"],
            "a=3.": ["a", "=", "3."],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                lex = make_lexer(text)
                lits = []
                tok = lex.next_token()
                while tok.token not in (TokenTypes.NEWLINE, TokenTypes.EOF):
                    lits.append(tok.literal)
                    tok = lex.next_token()
                self.assertEqual(lits, expected)
        lex = make_lexer("0.and.")
        self.assertEqual(lex.next_token(), Token(TokenTypes.INT, "0"))

    def test_unterminated_delimiter_raises(self):
        for text in ["x = 'abc", 'x = "abc', "x = a .and b"]:
            with self.subTest(text=text):
                lex = make_lexer(text)
                with self.assertRaisesRegex(lexer.LexError, "Unterminated"):
                    for _ in range(10):
                        lex.next_token()

    def test_unterminated_delimiter_is_a_parse_error(self):
        lines = [LineTuple(line="x = 'abc", ln=0)]
        parser = Parser(lines=lines)
        with self.assertRaises(SystemExit):
            parser.parse_program()
        self.assertTrue(any("Unterminated" in e for e in parser.errors))


if __name__ == "__main__":
    unittest.main()
