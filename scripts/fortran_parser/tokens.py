from dataclasses import dataclass
from enum import Enum


class TokenTypes(Enum):
    IDENT = "IDENT"
    EOF = "EOF"
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    ILLEGAL = "ILLEGAL"
    TYPE = "TYPE"
    TYPE_DEF = "type"
    # Operators
    ASSIGN = "="
    PLUS = "+"
    MINUS = "-"
    ASTERISK = "*"
    BANG = "!"
    SLASH = "/"
    EXP = "**"
    EQUIV = "=="
    GT = ">"
    GTEQ = ">="
    LT = "<"
    LTEQ = "<="
    NOT_EQUIV = "!="
    OR = "or"
    AND = "and"
    CONCAT = "//"
    PTR = "=>"
    # delimiters
    DOT = "."
    COMMA = ","
    LPAREN = "("
    RPAREN = ")"
    NEWLINE = "\n"
    COLON = ":"
    PERCENT = "%"
    MACRO = "#"
    DOUBLE_COLON = "::"
    ARRAY_INIT_START = "(/"
    ARRAY_INIT_END = "/)"
    ARRAY_LBRACKET = "["
    ARRAY_RBRACKET = "]"

    # keywords
    CONTAINS = "contains"
    SUBROUTINE = "SUBROUTINE"
    FUNCTION = "FUNCTION"
    LOGICAL = "LOGICAL"
    TRUE = ".true."
    FALSE = ".false."
    RETURN = "RETURN"
    CALL = "CALL"
    DO = "DO"
    DOWHILE = "DO WHILE"
    IF = "IF"
    THEN = "THEN"
    ELSE = "ELSE"
    ELSEIF = "ELSE IF"
    END = "END"
    ENDIF = "ENDIF"
    ENDDO = "ENDDO"
    ENDSUB = "END SUB"
    ENDFUNC = "END FUNC"
    PRINT = "PRINT"
    WRITE = "WRITE"
    IFDEF = "ifdef"
    DEF = "def"
    IFNDEF = "ifndef"
    M_ENDIF = "#endif"
    ENDTYPE = "end type"
    PROC = "procedure"


keywords: dict[str, TokenTypes] = {
    ".true.": TokenTypes.LOGICAL,
    ".false.": TokenTypes.LOGICAL,
    "call": TokenTypes.CALL,
    "subroutine": TokenTypes.SUBROUTINE,
    "function": TokenTypes.FUNCTION,
    "while": TokenTypes.DOWHILE,
    "do": TokenTypes.DO,
    "if": TokenTypes.IF,
    "else": TokenTypes.ELSE,
    "end": TokenTypes.END,
    "enddo": TokenTypes.ENDDO,
    "endif": TokenTypes.ENDIF,
    "elseif": TokenTypes.ELSEIF,
    "then": TokenTypes.THEN,
    ".eq.": TokenTypes.EQUIV,
    ".gt.": TokenTypes.GT,
    ".lt.": TokenTypes.LT,
    ".not.": TokenTypes.BANG,
    ".ne.": TokenTypes.NOT_EQUIV,
    ".ge.": TokenTypes.GTEQ,
    ".le.": TokenTypes.LTEQ,
    ".and.": TokenTypes.AND,
    ".or.": TokenTypes.OR,
    "print": TokenTypes.PRINT,
    "write": TokenTypes.WRITE,
    "ifdef": TokenTypes.IFDEF,
    "define": TokenTypes.DEF,
    "ifndef": TokenTypes.IFNDEF,
    "#endif": TokenTypes.M_ENDIF,
    "type": TokenTypes.TYPE_DEF,
    "class": TokenTypes.TYPE,
    "integer": TokenTypes.TYPE,
    "real": TokenTypes.TYPE,
    "complex": TokenTypes.TYPE,
    "logical": TokenTypes.TYPE,
    "character": TokenTypes.TYPE,
    "contains": TokenTypes.CONTAINS,
    "endtype": TokenTypes.ENDTYPE,
    "procedure": TokenTypes.PROC,
}


@dataclass
class Token:
    token: TokenTypes
    literal: str

    def __str__(self):
        return f"{self.literal}"


def lookup_identifer(ident: str) -> TokenTypes:
    if ident in keywords:
        return keywords[ident]
    return TokenTypes.IDENT
