from dataclasses import dataclass
from enum import Enum
from typing import Dict


class TokenTypes(Enum):
    IDENT = "IDENT"
    EOF = "EOF"
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    ILLEGAL = "ILLEGAL"
    # Operators
    ASSIGN = "="
    PLUS = "+"
    MINUS = "-"
    ASTERISK = "*"
    BANG = "!"
    SLASH = "/"
    EXP = "**"
    EQUIV = "=="
    # delimiters
    DOT = "."
    COMMA = ","
    LPAREN = "("
    RPAREN = ")"
    NEWLINE = "\n"
    COLON = ":"
    PERCENT = "%"
    # keywords
    SUBROUTINE = "SUBROUTINE"
    FUNCTION = "FUNCTION"
    LOGICAL = "LOGICAL"
    TRUE = ".true."
    FALSE = ".false."
    RETURN = "RETURN"
    CALL = "CALL"
    DO = "DO"
    DOWHILE = "DO WHILE"


keywords: Dict[str, TokenTypes] = {
    ".true.": TokenTypes.LOGICAL,
    ".false.": TokenTypes.LOGICAL,
    "call": TokenTypes.CALL,
    "subroutine": TokenTypes.SUBROUTINE,
    "function": TokenTypes.FUNCTION,
    "while": TokenTypes.DOWHILE,
    "do": TokenTypes.DO,
}


@dataclass
class Token:
    token: TokenTypes
    literal: str

    def __str__(self):
        return f"{self.literal}"


def lookup_indentifer(ident: str) -> TokenTypes:
    if ident in keywords:
        return keywords[ident]
    return TokenTypes.IDENT
