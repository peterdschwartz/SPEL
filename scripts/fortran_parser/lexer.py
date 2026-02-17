import scripts.fortran_parser.tokens as tokens
from scripts.fortran_parser.tracing import Trace
from scripts.types import LogicalLineIterator


class Lexer:
    def __init__(self, line_it: LogicalLineIterator):
        self.input: str = ""
        self.position: int = 0
        self.read_position: int = 0
        self.ch: str = ""
        self.line_iter: LogicalLineIterator = line_it
        self.cur_ln: int = 0
        # position of current token in current line
        self.token_pos: int = -1
        self._fetch_next_line()

    def _fetch_next_line(self) -> None:
        """Get the next line from the iterator, or set EOF."""
        try:
            fline = next(self.line_iter)
            next_line = fline.line
            self.input = next_line + "\n"
            self.position = 0
            self.read_position = 0
            self.read_char()  # initialize ch for new line
            self.cur_ln = self.line_iter.get_start_ln()
            self.token_pos = -1
        except StopIteration:
            self.input = ""
            self.ch = ""
            self.cur_ln = -1  # no more lines

    def read_to_delim(self, delim: str):
        """
        Func to read to delim (i.e. '  or ") for string tokens
        """
        pos = self.position
        while self.peek_char() != delim:
            self.read_char()
        self.read_char()
        return self.input[pos + 1 : self.position]

    def read_char(self):
        if self.read_position >= len(self.input):
            self.ch = ""
        else:
            self.ch = self.input[self.read_position]
        self.position = self.read_position
        self.read_position += 1

    def read_identifier(self) -> str:
        # Identifiers may contain digits but can't start with digits
        pos = self.position
        while is_valid_ident(self.ch):
            self.read_char()
        return self.input[pos : self.position]

    def read_num(self) -> str:
        pos = self.position
        while is_number(self.ch):
            self.read_char()
            if self.ch in ["."]:
                self.read_char()
        self.check_precision()

        return self.input[pos : self.position]

    def check_precision(self):
        """
        Need to retrieve scientific notation/precision.
        EX: 1.D-10, 1.E+3, 1._r8
        """
        peek_ch = self.peek_char()
        if peek_ch == " " or peek_ch == "\t":
            return
        if self.ch in ["d", "e"]:
            self.read_char()  # cur token is now one of the above.
            if self.ch in ["-", "+"]:
                self.read_char()
            # self.ch is either a letter or +/-
            self.read_char()
            self.read_num()  # Advance position to end of exponent
        elif self.ch == "_":
            self.read_identifier()
        return

    def skip_white_space(self) -> None:
        space_chars: list[str] = [" ", "\t"]
        while self.ch in space_chars:
            self.read_char()
        return

    def peek_char(self):
        if self.read_position >= len(self.input):
            return ""
        else:
            return self.input[self.read_position]

    def next_token(self) -> tokens.Token:
        """
        Get next tokens
        """
        self.skip_white_space()

        match self.ch:
            case "=":
                next_ch = self.peek_char()
                if next_ch == "=":
                    tok = new_token(tokens.TokenTypes.EQUIV, "==")
                    self.read_char()
                elif next_ch == ">":
                    tok = new_token(tokens.TokenTypes.PTR, "=>")
                    self.read_char()
                else:
                    tok = new_token(tokens.TokenTypes.ASSIGN, self.ch)
            case "(":
                next_ch = self.peek_char()
                if next_ch == "/":
                    tok = new_token(tokens.TokenTypes.ARRAY_INIT_START, "(/")
                    self.read_char()
                else:
                    tok = new_token(tokens.TokenTypes.LPAREN, self.ch)
            case ")":
                tok = new_token(tokens.TokenTypes.RPAREN, self.ch)
            case ",":
                tok = new_token(tokens.TokenTypes.COMMA, self.ch)
            case "+":
                tok = new_token(tokens.TokenTypes.PLUS, self.ch)
            case "-":
                tok = new_token(tokens.TokenTypes.MINUS, self.ch)
            case "*":
                next_ch = self.peek_char()
                if next_ch == "*":
                    lit = "**"
                    self.read_char()
                    tok = new_token(tokens.TokenTypes.EXP, lit)
                else:
                    tok = new_token(tokens.TokenTypes.ASTERISK, self.ch)
            case "/":
                next_ch = self.peek_char()
                if next_ch == "=":
                    tok = new_token(tokens.TokenTypes.NOT_EQUIV, ".ne.")
                    self.read_char()
                elif next_ch == "/":
                    tok = new_token(tokens.TokenTypes.CONCAT, "//")
                    self.read_char()
                elif next_ch == ")":
                    tok = new_token(tokens.TokenTypes.ARRAY_INIT_END, "/)")
                    self.read_char()
                else:
                    tok = new_token(tokens.TokenTypes.SLASH, self.ch)
            case ">":
                next_ch = self.peek_char()
                if next_ch == "=":
                    tok = new_token(tokens.TokenTypes.GTEQ, ">=")
                    self.read_char()
                else:
                    tok = new_token(tokens.TokenTypes.GT, self.ch)
            case "<":
                next_ch = self.peek_char()
                if next_ch == "=":
                    tok = new_token(tokens.TokenTypes.LTEQ, "<=")
                    self.read_char()
                else:
                    tok = new_token(tokens.TokenTypes.LT, self.ch)
            case "":
                self._fetch_next_line()
                if not self.input:
                    tok = new_token(tokens.TokenTypes.EOF, "")
                else:
                    return self.next_token()
            case "\n":
                tok = new_token(tokens.TokenTypes.NEWLINE, self.ch)
            case ":":
                next_ch = self.peek_char()
                if next_ch == ":":
                    tok = new_token(tokens.TokenTypes.DOUBLE_COLON, "::")
                    self.read_char()
                else:
                    tok = new_token(tokens.TokenTypes.COLON, self.ch)
            case ";":
                tok = new_token(tokens.TokenTypes.SEMICOLON,self.ch)
            case "%":
                tok = new_token(tokens.TokenTypes.PERCENT, self.ch)
            case "'":
                delim = self.ch
                lit: str = self.read_to_delim(delim)
                tok = new_token(tokens.TokenTypes.STRING, lit)
            case '"':
                delim = self.ch
                lit: str = self.read_to_delim(delim)
                tok = new_token(tokens.TokenTypes.STRING, lit)
            case ".":
                next_ch = self.peek_char()
                if is_number(next_ch):
                    self.read_char()
                    lit = self.read_num()
                    tok = new_token(tokens.TokenTypes.FLOAT, f"0.{lit}")
                    return tok
                else:
                    delim = self.ch
                    lit: str = self.read_to_delim(delim)
                    tok_type = tokens.lookup_identifer(f".{ lit }.")
                    tok = new_token(tok_type, f".{ lit }.")
            case "#":
                tok = new_token(tokens.TokenTypes.MACRO, self.ch)
            case "[":
                tok = new_token(tokens.TokenTypes.ARRAY_LBRACKET, self.ch)
            case "]":
                tok = new_token(tokens.TokenTypes.ARRAY_RBRACKET, self.ch)
            case _:
                cur_ch = self.ch
                if cur_ch.isalpha() or cur_ch == "_":
                    lit: str = self.read_identifier()
                    tok_type = tokens.lookup_identifer(lit)
                    tok = new_token(tok_type, lit)
                    return tok
                elif is_number(cur_ch):
                    lit: str = self.read_num()
                    if "." in lit or "e" in lit or "d" in lit:
                        tok = new_token(tokens.TokenTypes.FLOAT, lit)
                    else:
                        tok = new_token(tokens.TokenTypes.INT, lit)
                    return tok
                else:
                    tok = new_token(tokens.TokenTypes.ILLEGAL, self.ch)

        self.read_char()
        return tok


def is_valid_ident(ch: str) -> bool:
    "FORTRAN allows numbers, _, and % (for derived types) in identifier names"
    return ch.isalnum() or ch == "_" or ch == "%"


def is_number(ch: str) -> bool:
    return ch.isdigit()


def new_token(tok_type, lit) -> tokens.Token:
    token = tokens.Token(
        token=tok_type,
        literal=lit,
    )
    return token
