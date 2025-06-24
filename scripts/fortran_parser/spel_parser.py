from enum import Enum
from logging import Logger
from typing import Callable, Dict, List, Optional

import scripts.fortran_parser.lexer as lexer
from scripts.fortran_parser.spel_ast import (BoundsExpression, DoLoop, DoWhile,
                                             Expression, ExpressionStatement,
                                             FieldAccessExpression,
                                             FloatLiteral, FuncExpression,
                                             Identifier, InfixExpression,
                                             IntegerLiteral, LogicalLiteral,
                                             PrefixExpression, Program,
                                             Statement, StringLiteral,
                                             SubCallStatement)
from scripts.fortran_parser.tokens import Token, TokenTypes
from scripts.fortran_parser.tracing import Trace
from scripts.logging_configs import get_logger


class Precedence(Enum):
    _ = 0
    LOWEST = 1
    EQUALS = 2
    LESSGREATER = 3
    SUM = 4
    PRODUCT = 5
    PREFIX = 6
    BOUNDS = 7
    CALL = 8


precedences = {
    TokenTypes.ASSIGN: Precedence.EQUALS,
    TokenTypes.PLUS: Precedence.SUM,
    TokenTypes.MINUS: Precedence.SUM,
    TokenTypes.SLASH: Precedence.PRODUCT,
    TokenTypes.ASTERISK: Precedence.PRODUCT,
    TokenTypes.LPAREN: Precedence.CALL,
    TokenTypes.COLON: Precedence.BOUNDS,
    TokenTypes.PERCENT: Precedence.BOUNDS,
    TokenTypes.EXP: Precedence.PRODUCT,
    TokenTypes.EQUIV: Precedence.EQUALS,
}

PrefixParseFn = Callable[[], Expression]
InfixParseFn = Callable[[Expression], Expression]


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, lex: lexer.Lexer):
        self.lexer: lexer.Lexer = lex
        self.errors: List[str] = []
        self.cur_token: Token = Token(token=TokenTypes.ILLEGAL, literal="")
        self.peek_token: Token = Token(token=TokenTypes.ILLEGAL, literal="")
        self.prefix_parse_fns: Dict[TokenTypes, PrefixParseFn] = {}
        self.infix_parse_fns: Dict[TokenTypes, InfixParseFn] = {}
        self.logger: Logger = get_logger("Parser")

        self.register_prefix_fns(TokenTypes.IDENT, self.parse_identifier)
        self.register_prefix_fns(TokenTypes.INT, self.parseIntegerLiteral)
        self.register_prefix_fns(TokenTypes.FLOAT, self.parse_FloatLiteral)
        self.register_prefix_fns(TokenTypes.STRING, self.parseStringLiteral)
        self.register_prefix_fns(TokenTypes.LOGICAL, self.parseLogicalLiteral)
        self.register_prefix_fns(TokenTypes.BANG, self.parse_prefix_expr)
        self.register_prefix_fns(TokenTypes.MINUS, self.parse_prefix_expr)
        self.register_prefix_fns(TokenTypes.LPAREN, self.parse_grouped_expr)
        self.register_prefix_fns(TokenTypes.COLON, self.parse_prefix_bounds_expr)

        self.register_infix_fns(TokenTypes.PLUS, self.parse_infix_expr)
        self.register_infix_fns(TokenTypes.MINUS, self.parse_infix_expr)
        self.register_infix_fns(TokenTypes.SLASH, self.parse_infix_expr)
        self.register_infix_fns(TokenTypes.ASTERISK, self.parse_infix_expr)
        self.register_infix_fns(TokenTypes.EXP, self.parse_infix_expr)
        self.register_infix_fns(TokenTypes.ASSIGN, self.parse_infix_expr)
        self.register_infix_fns(TokenTypes.LPAREN, self.parse_func_expr)
        self.register_infix_fns(TokenTypes.COLON, self.parse_infix_bounds_expr)
        self.register_infix_fns(TokenTypes.EQUIV, self.parse_infix_expr)
        self.register_infix_fns(TokenTypes.PERCENT, self.parse_field_access_expr)

        self.next_token()
        self.next_token()

    def reset_lexer(self, lex: lexer.Lexer):
        """
        Function to reuse parser with new input/lexer
        """
        self.lexer = lex
        self.next_token()
        self.next_token()

    def next_token(self):
        self.cur_token = self.peek_token
        self.peek_token = self.lexer.next_token()

    def register_prefix_fns(self, tok_type: TokenTypes, fn: PrefixParseFn):
        self.prefix_parse_fns[tok_type] = fn

    def register_infix_fns(self, tok_type: TokenTypes, fn: InfixParseFn):
        self.infix_parse_fns[tok_type] = fn

    def parse_identifier(self) -> Expression:
        return Identifier(tok=self.cur_token, value=self.cur_token.literal)

    @Trace.trace_decorator("parseIntegerLiteral")
    def parseIntegerLiteral(self) -> Expression:
        try:
            val = int(self.cur_token.literal)
            return IntegerLiteral(tok=self.cur_token, val=val)
        except ValueError:
            self.errors.append(
                "Could not parse IntegerLiteral: " + self.cur_token.literal
            )
            raise ParseError(
                "Could not parse IntegerLiteral: " + self.cur_token.literal
            )

    @Trace.trace_decorator("parseStringLiteral")
    def parseStringLiteral(self) -> Expression:
        return StringLiteral(tok=self.cur_token, val=self.cur_token.literal)

    @Trace.trace_decorator("parseStringLiteral")
    def parseLogicalLiteral(self) -> Expression:
        val_str = self.cur_token.literal
        if val_str == ".true.":
            val = True
        else:
            val = False
        return LogicalLiteral(tok=self.cur_token, val=val)

    def parse_FloatLiteral(self) -> Expression:
        num = FloatLiteral(tok=self.cur_token)
        lit = self.cur_token.literal
        if "_" in lit:
            val, prec = lit.split("_")
            num.precision = "_" + prec
        else:
            val = lit
        num.value = float(val)
        return num

    def curTokenIs(self, etype: TokenTypes):
        return self.cur_token.token == etype

    def peekTokenIs(self, etype: TokenTypes):
        return self.peek_token.token == etype

    def expect_peek(self, etype: TokenTypes) -> bool:
        if self.peekTokenIs(etype):
            self.next_token()
            return True
        else:
            return False

    def peek_precedence(self) -> Precedence:
        try:
            prec = precedences[self.peek_token.token]
            return prec
        except KeyError:
            return Precedence.LOWEST

    def cur_precedence(self) -> Precedence:
        try:
            prec = precedences[self.cur_token.token]
            return prec
        except KeyError:
            return Precedence.LOWEST

    def parse_program(self) -> Program:
        program = Program()
        while self.cur_token.token != TokenTypes.EOF:
            try:
                stmt = self.parse_statement()
                program.statements.append(stmt)
            except ParseError as e:
                print(f"Error: {e}")
            self.next_token()
        return program

    def check_label(self) -> Optional[Token]:
        """
        If the current token is an identifier followed by a colon,
        treat it as a named block label.
        This function should only be called on the first Token of a Statement
        """
        label = None
        if self.cur_token.token == TokenTypes.IDENT and self.peek_token.token == TokenTypes.COLON:
            label = self.cur_token
            self.next_token()  # skip IDENT
            self.next_token()  # skip COLON
        return label

    @Trace.trace_decorator("parse_statement")
    def parse_statement(self) -> Statement:
        label = self.check_label()

        if self.curTokenIs(TokenTypes.DO) and self.peekTokenIs(TokenTypes.DOWHILE):
            self.next_token() # skip DO token

        match self.cur_token.token:
            case TokenTypes.CALL:
                stmt = self.parse_subcall_statement()
            case TokenTypes.DO:
                stmt = self.parse_do_block(label)
            case TokenTypes.DOWHILE:
                stmt = self.parse_dowhile_block(label)
            case _:
                stmt = self.parse_expression_statement()
        return stmt

    @Trace.trace_decorator("parse_subcall_statement")
    def parse_subcall_statement(self)->Statement:
        stmt = SubCallStatement(tok=self.cur_token)
        self.next_token()
        # Parse identifier expression:
        stmt.function = self.parse_expression(Precedence.LOWEST)
        return stmt

    @Trace.trace_decorator("parse_do_block")
    def parse_do_block(self, label: Optional[Token]) -> Statement:
        """
        Upon function call, cur_token = 'DO'
        """
        tok = self.cur_token
        self.next_token() # cur_token is the loop index identifier
        index = self.cur_token
        assert index.token == TokenTypes.IDENT

        if not self.expect_peek(TokenTypes.ASSIGN): # expect_peek advances token
            self.errors.append(f"failed to parse DO LOOP {self.peek_token}")
            raise ParseError("Couldn't Parse Do LOOP")

        self.next_token() 
        start_expr, end_expr, step = self.parse_do_bounds()

        # advance to end of statement
        self.next_token()
        return DoLoop(tok,index,start_expr,end_expr, step)

    @Trace.trace_decorator("parse_do_bounds")
    def parse_do_bounds(self) -> tuple[Expression, Expression, Optional[Expression]]:
        """
        Parse the bounds expression: start, end [, step]
        Assumes current token is just past '='
        """
        start_expr = self.parse_expression(Precedence.LOWEST)

        self.expect_peek(TokenTypes.COMMA)
        self.next_token()  # skip the comma

        end_expr = self.parse_expression(Precedence.LOWEST)

        step_expr = None
        if self.peekTokenIs(TokenTypes.COMMA):
            self.next_token()  # skip comma
            self.next_token()
            step_expr = self.parse_expression(Precedence.LOWEST)

        return start_expr, end_expr, step_expr

    def parse_dowhile_block(self, label: Token)->Statement:
        """
        Parse do while block. cur_token is expected to be WHILE
        """
        assert self.curTokenIs(TokenTypes.DOWHILE), "(parse_dowhile_block) expects DOWHILE"
        tok = self.cur_token
        self.next_token() # token should be RPAREN
        cond = self.parse_expression(Precedence.LOWEST)
        self.next_token()
        return DoWhile(tok,cond)

    @Trace.trace_decorator("parse_expression_statement")
    def parse_expression_statement(self) -> ExpressionStatement:
        stmt = ExpressionStatement(tok=self.cur_token)
        stmt.expression = self.parse_expression(Precedence.LOWEST)
        self.next_token()
        return stmt

    @Trace.trace_decorator("parse_expression")
    def parse_expression(self, prec: Precedence) -> Expression:
        cur_type = self.cur_token.token
        prefix = self.prefix_parse_fns[cur_type]
        left_expr: Expression = prefix()

        while (
            not self.peekTokenIs(TokenTypes.NEWLINE)
            and prec.value < self.peek_precedence().value
        ):
            peek_type = self.peek_token.token
            if peek_type not in self.infix_parse_fns:
                return left_expr
            infix = self.infix_parse_fns[peek_type]
            self.next_token()
            left_expr = infix(left_expr)
        return left_expr

    @Trace.trace_decorator("parse_grouped_expr")
    def parse_grouped_expr(self) -> Expression:
        self.next_token()

        expr = self.parse_expression(Precedence.LOWEST)
        if not self.expect_peek(TokenTypes.RPAREN):
            self.errors.append("Failed to Parse Grouped Expression" + str(expr))
        return expr

    @Trace.trace_decorator("parse_prefix_expr")
    def parse_prefix_expr(self) -> Expression:
        expr = PrefixExpression(tok=self.cur_token, op=self.cur_token.literal)
        self.next_token()
        expr.right_expr = self.parse_expression(Precedence.PREFIX)

        return expr

    @Trace.trace_decorator("parse_infix_expr")
    def parse_infix_expr(self, left: Expression) -> Expression:
        expression = InfixExpression(
            tok=self.cur_token,
            op=self.cur_token.literal,
            left=left,
        )
        prec = self.cur_precedence()
        self.next_token()

        expression.right_expr = self.parse_expression(prec)
        return expression

    @Trace.trace_decorator("parse_field_access_expr")
    def parse_field_access_expr(self, left: Expression) -> Expression:
        tok = self.cur_token

        prec = self.cur_precedence()
        self.next_token()

        right_expr: Expression = self.parse_expression(prec)

        return FieldAccessExpression(
            tok=tok,
            left=left,
            field=right_expr,
        )

    @Trace.trace_decorator("parse_func_expr")
    def parse_func_expr(self, func: Expression) -> Expression:
        func_expr = FuncExpression(tok=self.cur_token, fn=func)
        func_expr.args = self.parse_args()
        return func_expr

    def parse_args(self) -> List[Expression]:
        args: List[Expression] = []

        if self.peekTokenIs(TokenTypes.RPAREN):
            self.next_token()
            return args
        self.next_token()
        args.append(self.parse_expression(Precedence.LOWEST))
        while self.peekTokenIs(TokenTypes.COMMA):
            self.next_token()
            self.next_token()
            # Current token is now start of next arg
            args.append(self.parse_expression(Precedence.LOWEST))

        # Note expect peek advances tokens, to cur_token = RPAREN at return
        if not self.expect_peek(TokenTypes.RPAREN):
            raise ParseError("Couldn't Parse Arguments")
        return args

    @Trace.trace_decorator("parse_infix_bounds_expr")
    def parse_infix_bounds_expr(self, start: Expression) -> Expression:
        """
        Function to parse bounds.  curent token should be ":"
        """
        bounds_expr = BoundsExpression(tok=self.cur_token)
        bounds_expr.start = start

        if not self.peekTokenIs(TokenTypes.RPAREN) and not self.peekTokenIs(
            TokenTypes.COMMA
        ):
            self.next_token()
            bounds_expr.end = self.parse_expression(Precedence.LOWEST)
        return bounds_expr

    @Trace.trace_decorator("parse_prefix_bounds_expr")
    def parse_prefix_bounds_expr(self) -> Expression:
        """
        Function to parse bounds.  curent token should be ":"
        """
        bounds_expr = BoundsExpression(tok=self.cur_token)
        if not self.peekTokenIs(TokenTypes.RPAREN) and not self.peekTokenIs(
            TokenTypes.COMMA
        ):
            self.next_token()
            bounds_expr.end = self.parse_expression(Precedence.LOWEST)
        return bounds_expr
