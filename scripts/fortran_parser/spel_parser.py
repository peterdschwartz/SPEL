import sys
from logging import Logger
from types import FunctionType
from typing import Callable, List, Optional

import scripts.fortran_parser.lexer as lexer
from scripts.fortran_parser.spel_ast import (ArrayInit, AttributeSpec,
                                             BlockStatement, BoundsExpression,
                                             DoLoop, DoWhile, Else, ElseIf,
                                             EntityDecl, Expression,
                                             ExpressionStatement,
                                             FieldAccessExpression,
                                             FloatLiteral, FuncExpression, GenericOperatorExpression,
                                             Identifier, IfConstruct,
                                             ImpliedDo, InfixExpression,
                                             IntegerLiteral, IOExpression,
                                             LogicalLiteral, MacroDefine,
                                             MacroIf, PrefixExpression,
                                             PrintStatement,
                                             ProcedureStatement, Program,
                                             Statement, StringLiteral,
                                             SubCallStatement, TypeDef,
                                             TypeSpec, UseStatement,
                                             VariableDecl, WriteStatement)
from scripts.fortran_parser.tokens import Token, TokenTypes
from scripts.fortran_parser.tracing import Trace
from scripts.logging_configs import get_logger
from scripts.types import LineTuple, LogicalLineIterator, Precedence

precedences = {
    TokenTypes.ASSIGN: Precedence.EQUALS,
    TokenTypes.PTR: Precedence.EQUALS,
    TokenTypes.PLUS: Precedence.SUM,
    TokenTypes.MINUS: Precedence.SUM,
    TokenTypes.SLASH: Precedence.PRODUCT,
    TokenTypes.ASTERISK: Precedence.PRODUCT,
    TokenTypes.LPAREN: Precedence.CALL,
    TokenTypes.COLON: Precedence.BOUNDS,
    TokenTypes.PERCENT: Precedence.BOUNDS,
    TokenTypes.EXP: Precedence.PRODUCT,
    TokenTypes.EQUIV: Precedence.EQUALS,
    TokenTypes.NOT_EQUIV: Precedence.EQUALS,
    TokenTypes.GT: Precedence.LESSGREATER,
    TokenTypes.GTEQ: Precedence.LESSGREATER,
    TokenTypes.LT: Precedence.LESSGREATER,
    TokenTypes.LTEQ: Precedence.LESSGREATER,
    TokenTypes.AND: Precedence.EQUALS,
    TokenTypes.OR: Precedence.EQUALS,
    TokenTypes.CONCAT: Precedence.PRODUCT,
    TokenTypes.MACRO: Precedence.PREFIX,
    TokenTypes.BANG: Precedence.PREFIX,
}

PrefixParseFn = Callable[[], Expression]
InfixParseFn = Callable[[Expression], Expression]

Tok = TokenTypes


class ParseError(Exception):
    pass

class Parser:
    def __init__(self, lex: Optional[lexer.Lexer]=None, logger: str = "Parser",lines: list[LineTuple]=None,):
        if lex:
            self.lexer: lexer.Lexer = lex
        elif lines:
            line_it = LogicalLineIterator(lines=lines)
            self.lexer = lexer.Lexer(line_it)
        else:
            sys.exit("Error - Need either lexer or lines to create Parser")
        self.errors: list[str] = []
        self.cur_token: Token = Token(token=Tok.ILLEGAL, literal="")
        self.peek_token: Token = Token(token=Tok.ILLEGAL, literal="")
        self.prefix_parse_fns: dict[Tok, PrefixParseFn] = {}
        self.infix_parse_fns: dict[Tok, InfixParseFn] = {}
        self.logger: Logger = get_logger(logger)

        self.lineno: int = 0

        self.register_prefix_fns(Tok.IDENT, self.parse_identifier)
        self.register_prefix_fns(Tok.TYPE, self.parse_identifier)
        self.register_prefix_fns(Tok.PRINT, self.parse_identifier)
        self.register_prefix_fns(Tok.INT, self.parseIntegerLiteral)
        self.register_prefix_fns(Tok.FLOAT, self.parse_FloatLiteral)
        self.register_prefix_fns(Tok.STRING, self.parseStringLiteral)
        self.register_prefix_fns(Tok.LOGICAL, self.parseLogicalLiteral)
        self.register_prefix_fns(Tok.BANG, self.parse_prefix_expr)
        self.register_prefix_fns(Tok.MINUS, self.parse_prefix_expr)
        self.register_prefix_fns(Tok.PLUS, self.parse_prefix_expr)
        self.register_prefix_fns(Tok.LPAREN, self.parse_grouped_expr)
        self.register_prefix_fns(Tok.COLON, self.parse_prefix_bounds_expr)
        self.register_prefix_fns(Tok.ASTERISK, self.stdout_or_fmt)
        self.register_prefix_fns(Tok.ARRAY_INIT_START, self.parse_array_init)
        self.register_prefix_fns(Tok.ARRAY_LBRACKET, self.parse_array_init)

        # Infix Operators
        self.register_infix_fns(Tok.PLUS, self.parse_infix_expr)
        self.register_infix_fns(Tok.MINUS, self.parse_infix_expr)
        self.register_infix_fns(Tok.SLASH, self.parse_infix_expr)
        self.register_infix_fns(Tok.ASTERISK, self.parse_infix_expr)
        self.register_infix_fns(Tok.EXP, self.parse_infix_expr)
        self.register_infix_fns(Tok.ASSIGN, self.parse_infix_expr)
        self.register_infix_fns(Tok.PTR, self.parse_infix_expr)
        self.register_infix_fns(Tok.CONCAT, self.parse_infix_expr)
        self.register_infix_fns(Tok.LPAREN, self.parse_func_expr)
        self.register_infix_fns(Tok.COLON, self.parse_infix_bounds_expr)
        # Logical operators
        self.register_infix_fns(Tok.EQUIV, self.parse_infix_expr)
        self.register_infix_fns(Tok.GT, self.parse_infix_expr)
        self.register_infix_fns(Tok.LT, self.parse_infix_expr)
        self.register_infix_fns(Tok.GTEQ, self.parse_infix_expr)
        self.register_infix_fns(Tok.LTEQ, self.parse_infix_expr)
        self.register_infix_fns(Tok.NOT_EQUIV, self.parse_infix_expr)
        self.register_infix_fns(Tok.AND, self.parse_infix_expr)
        self.register_infix_fns(Tok.OR, self.parse_infix_expr)
        #
        self.register_infix_fns(Tok.PERCENT, self.parse_field_access_expr)

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
        """
        function to advance tokens
        """
        self.cur_token = self.peek_token
        self.peek_token = self.lexer.next_token()
        self.lexer.token_pos += 1

        pair = (self.cur_token.token, self.peek_token.token)

        # combos that *should* consume the peek
        combos = {
            (Tok.END, Tok.IF): (Tok.ENDIF, "ENDIF"),
            (Tok.END, Tok.DO): (Tok.ENDDO, "ENDDO"),
            (Tok.END, Tok.SUBROUTINE): (Tok.ENDSUB, "ENDSUB"),
            (Tok.END, Tok.FUNCTION): (Tok.ENDFUNC, "ENDFUNC"),
            (Tok.ELSE, Tok.IF): (Tok.ELSEIF, "ELSEIF"),
            (Tok.MACRO, Tok.ENDIF): (Tok.M_ENDIF, "#endif"),
            (Tok.END, Tok.TYPE_DEF): (Tok.ENDTYPE, "end type"),
        }
        if pair in combos:
            new_type, lit = combos[pair]
            self.cur_token = Token(new_type, lit)
            self.peek_token = self.lexer.next_token()
        elif self.curTokenIs(Tok.END):
            self.cur_token = Token(Tok.IDENT, "end")
        elif (
            self.curTokenIs(Tok.TYPE_DEF)
            and self.peekTokenIs(Tok.LPAREN)
            and self.is_first_token()
        ):
            self.cur_token = Token(Tok.TYPE, "type")
        elif not self.is_first_token() and self.curTokenIs(Tok.TYPE_DEF):
            self.cur_token = Token(Tok.IDENT, "type")
        # self.logger.debug(f"{self.cur_token} -> {self.peek_token}")
        self.lineno = self.lexer.cur_ln
        return

    def is_first_token(self) -> bool:
        return self.lexer.token_pos == 1

    def register_prefix_fns(self, tok_type: Tok, fn: PrefixParseFn):
        self.prefix_parse_fns[tok_type] = fn

    def register_infix_fns(self, tok_type: Tok, fn: InfixParseFn):
        self.infix_parse_fns[tok_type] = fn

    def parse_identifier(self) -> Expression:
        return Identifier(tok=self.cur_token, value=self.cur_token.literal)

    def parseIntegerLiteral(self) -> Expression:
        lit = self.cur_token.literal
        if "_" in lit:
            val, prec = lit.split("_")
            val = int(val)
        else:
            val = int(lit)
            prec = ""
        return IntegerLiteral(tok=self.cur_token, val=val, prec=prec)

    def parseStringLiteral(self) -> Expression:
        return StringLiteral(tok=self.cur_token, val=self.cur_token.literal)

    def parseLogicalLiteral(self) -> Expression:
        val_str = self.cur_token.literal
        if val_str == ".true.":
            val = True
        else:
            val = False
        return LogicalLiteral(tok=self.cur_token, val=val)

    def parse_FloatLiteral(self) -> Expression:
        lit = self.cur_token.literal
        precision = ""
        if "_" in lit:
            val_prec = lit.split("_")
            val = val_prec[0]
            prec = "_".join(val_prec[1:])
            precision = "_" + prec
        else:
            val = lit
        value = float(val.replace("d", "e"))
        num = FloatLiteral(tok=self.cur_token,val=value,prec=precision)
        return num

    def curTokenIs(self, etype: Tok):
        return self.cur_token.token == etype

    def peekTokenIs(self, etype: Tok):
        return self.peek_token.token == etype

    def expect_peek(self, etype: Tok) -> bool:
        if self.peekTokenIs(etype):
            self.next_token()
            return True
        else:
            err = f"Expected: {etype}, Got: {self.peek_token} @{self.lineno} {self.lexer.input}"
            self.errors.append(err)
            self.logger.error(f"{err}")
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
        while self.cur_token.token != Tok.EOF:
            try:
                stmt = self.parse_statement()
                if stmt:
                    program.statements.append(stmt)
            except ParseError as e:
                self.logger.error(f"Error: {e}")
            self.next_token()
            if self.errors:
                self.error_exit()
        return program

    def error_exit(self):
        for err in self.errors:
            self.logger.error(f"{err}")
        sys.exit(1)

    def check_label(self) -> Optional[Token]:
        """
        If the current token is an identifier followed by a colon,
        treat it as a named block label.
        This function should only be called on the first Token of a Statement
        """
        label = None
        if self.cur_token.token == Tok.IDENT and self.peek_token.token == Tok.COLON:
            label = self.cur_token
            self.next_token()  # skip IDENT
            self.next_token()  # skip COLON
        return label

    def parse_macro(self) -> Optional[Statement]:
        # Current token is tok.MACRO
        if self.peekTokenIs(Tok.IFDEF) or self.peekTokenIs(Tok.IFNDEF):
            # consume IFDEF or IFNDEF
            self.next_token()
            token = self.cur_token
            self.expect_peek(Tok.IDENT)
            symbol = self.cur_token.literal
            self.next_token()
            body = self.parse_block_statement(token, [Token(Tok.M_ENDIF, "#endif")])
            stmt = MacroIf(token=token, symbol=symbol, body=body)
            self.next_token()
        elif self.peekTokenIs(Tok.DEF):
            self.next_token()
            token = self.cur_token
            if not self.expect_peek(Tok.IDENT):
                self.logger.error("Expected macro name after #define")
                return None
            macro_name = self.cur_token.literal
            macro_value = None
            self.next_token()
            # Optional value after macro name
            if not self.curTokenIs(Tok.NEWLINE):
                macro_value = self.parse_expression(Precedence.LOWEST)
            stmt = MacroDefine(token=token, symbol=macro_name, value=macro_value)
        else:
            err = f"UNKNOWN MACRO {self.lexer.input}"
            self.logger.error(err)
            self.errors.append(err)
            stmt = None

        return stmt

    def stdout_or_fmt(self) -> Expression:
        return IOExpression(self.cur_token)

    @Trace.trace_decorator("parse_statement")
    def parse_statement(self) -> Optional[Statement]:
        label = self.check_label()

        if self.curTokenIs(Tok.DO) and self.peekTokenIs(Tok.DOWHILE):
            self.next_token()  # skip DO token

        startln = self.lineno
        match self.cur_token.token:
            case Tok.CALL:
                stmt = self.parse_subcall_statement()
            case Tok.DO:
                stmt = self.parse_do_block(label)
            case Tok.DOWHILE:
                stmt = self.parse_dowhile_block(label)
            case Tok.IF:
                stmt = self.parse_if_statement()
            case Tok.NEWLINE:
                stmt = None
            case Tok.SEMICOLON:
                stmt = None
            case Tok.PRINT:
                stmt = self.parse_print_statement()
            case Tok.WRITE:
                stmt = self.parse_write_statement()
            case Tok.MACRO:
                stmt = self.parse_macro()
            case Tok.TYPE:
                stmt = self.parse_var_decl()
            case Tok.TYPE_DEF:
                stmt = self.parse_type_def()
            case Tok.PROC:
                stmt = self.parse_procedure_stmt()
            case Tok.USE:
                stmt = self.parse_use_statement()
            case _:
                stmt = self.parse_expression_statement()
        if stmt:
            stmt.lineno = startln
        return stmt

    @Trace.trace_decorator("parse_uses")
    def parse_use_statement(self)->Statement:
        """
        Parses fortran statements: use <mod_name> [, only : obj1,obj2, x => obj3]
        """
        tok = self.cur_token
        self.next_token()
        assert self.curTokenIs(Tok.IDENT), f"Un-expected module name: {self.cur_token}"
        module = self.parse_expression(Precedence.LOWEST) # must be Identifier
        objs = []
        only_clause = False
        if self.peekTokenIs(Tok.COMMA):
            self.next_token() # COMMA
            self.expect_peek(Tok.IDENT) # 'only'
            assert self.cur_token.literal == "only", f"Expect 'only' identifier got: {self.cur_token}"
            self.expect_peek(Tok.COLON) # ':'
            self.next_token()
            only_clause = True

        if only_clause:
            op = self.check_operator_def()
            if op is None:
                objs.append(self.parse_expression(Precedence.LOWEST))
            else:
                objs.append(op)
            while self.peekTokenIs(Tok.COMMA):
                self.next_token()
                self.next_token()
                op = self.check_operator_def()
                if op is None:
                    objs.append(self.parse_expression(Precedence.LOWEST))
                else:
                    objs.append(op)
        return UseStatement(tok=tok,mod_name=module,objs=objs)


    @Trace.trace_decorator("check_operator")
    def check_operator_def(self)->Optional[GenericOperatorExpression]:
        if not self.peekTokenIs(Tok.LPAREN):
            return None
        if self.cur_token.literal == "assignment" :
            tok = Token(token=Tok.IDENT,literal="assignment")
            _ = self.expect_peek(Tok.LPAREN) # not needed but advances tokens
            _ = self.expect_peek(Tok.ASSIGN)
            spec: Identifier = Identifier(tok=self.cur_token,value='=')
            _ = self.expect_peek(Tok.RPAREN)
        elif self.cur_token.literal == "operator":
            tok = Token(token=Tok.IDENT,literal="operator")
            _ = self.expect_peek(Tok.LPAREN)
            self.next_token()
            spec: Identifier = self.parse_identifier()
            _ = self.expect_peek(Tok.RPAREN)
        else:
            self.errors.append(f"(check_operator_def) Unexpected token {self.cur_token}")
            self.error_exit()
        return GenericOperatorExpression(tok=tok,spec=spec)


    def parse_type_def(self) -> Statement:
        """
        Function to parse a fortran type definition
        - cur_token = Tok.TYPE_DEF
        """
        tok = self.cur_token
        attrs = None
        methods = None
        self.next_token()  # comma, double colon, type_name?
        if self.curTokenIs(Tok.COMMA):
            attrs = self.parse_attr_spec()
        elif self.curTokenIs(Tok.DOUBLE_COLON):
            self.next_token()

        assert self.curTokenIs(Tok.IDENT), f"Expected type name: {self.cur_token}"
        type_name = self.cur_token.literal
        self.expect_peek(Tok.NEWLINE)
        body = self.parse_block_statement(
            tok=tok,
            terminators=[
                Token(Tok.CONTAINS, "contains"),
                Token(Tok.ENDTYPE, "end type"),
            ],
        )
        if self.curTokenIs(Tok.CONTAINS):
            self.next_token()
            methods = self.parse_block_statement(
                tok=Token(Tok.CONTAINS, "contains"),
                terminators=[Token(Tok.ENDTYPE, "end type")],
            )
        self.next_token()  # consume end type

        return TypeDef(
            tok=tok,
            name=type_name,
            attr_spec=attrs,
            body=body,
            methods=methods,
        )

    @Trace.trace_decorator("parse_var_decl")
    def parse_var_decl(self) -> Statement:
        """
        Enter:  cur_token is .TYPE
        """
        type_tok = self.cur_token
        kind = ""
        len_ = None
        if type_tok.literal in ["class", "type"]:
            self.next_token()
            type_expr = self.parse_expression(Precedence.LOWEST)
            assert isinstance(
                type_expr, Identifier
            ), f"Not Identifier {type_expr}, {self.lexer.input}"
            base_type = Token(Tok.TYPE, type_expr.value)
        else:
            # intrinsic types
            type_expr = self.parse_expression(Precedence.LOWEST)
            if isinstance(type_expr, FuncExpression):
                base_type = Token(Tok.TYPE, type_expr.function.value)
                if type_tok.literal == "character":
                    len_ = str(type_expr.args[0])
                elif type_tok.literal in ["real", "integer", "complex"]:
                    kind = str(type_expr.args[0])
            elif isinstance(type_expr, Identifier):
                base_type = Token(Tok.TYPE, type_expr.value)
            elif isinstance(type_expr, InfixExpression):
                assert (
                    type_expr.operator == "*"
                ), f"Unexpected expression for type {type_expr} / {type(type_expr)}:\n{self.lexer.input}"
                base_type = Token(Tok.TYPE, type_expr.left_expr.value)
                kind = type_expr.right_expr.value
        self.next_token()

        type_spec = TypeSpec(base_type=base_type, kind=kind, len_=len_)
        attrs = None
        if self.curTokenIs(Tok.COMMA):
            attrs = self.parse_attr_spec()
        entities = self.parse_entity_list()

        return VariableDecl(
            tok=type_tok,
            type_spec=type_spec,
            attrs=attrs,
            entities=entities,
        )

    @Trace.trace_decorator("parse_entity_list")
    def parse_entity_list(self) -> list[EntityDecl]:
        """
        helper function for comma separated expressions in variable declarations
        """
        if self.peekTokenIs(Tok.DOUBLE_COLON):
            self.next_token()  # '::'
            self.next_token()  # now at identifier

        if self.curTokenIs(Tok.DOUBLE_COLON):
            self.next_token()

        entities: list[EntityDecl] = []
        # parse first ident expression
        expr = self.parse_expression(Precedence.LOWEST)
        entities.append(self.create_entity(expr))
        self.next_token()

        while self.curTokenIs(Tok.COMMA):
            self.next_token()
            expr = self.parse_expression(Precedence.LOWEST)
            entities.append(self.create_entity(expr))
            self.next_token()

        return entities

    def create_entity(self, expr: Expression) -> EntityDecl:
        bounds: list[Expression] = []
        init: Optional[Expression] = None
        if isinstance(expr, FuncExpression):
            # the "args" in are the Bounds
            name = expr.function.value
            bounds.extend(expr.args)
        elif isinstance(expr, Identifier):
            name = expr.value
        elif isinstance(expr, InfixExpression):
            if isinstance(expr.left_expr, FuncExpression):
                bounds.extend(expr.left_expr.args)
            name = expr.left_expr.get_name()
            init = expr.right_expr

        return EntityDecl(tok=Token(Tok.IDENT, name), bounds=bounds, init=init)

    def parse_attr_spec(self) -> AttributeSpec:
        """
        Cur token is COMMA.  If an attribute spec exists, there must be
        a '::' separator!
        """
        fn = "(parse_attr_spec)"
        attrs: list[Expression] = []
        while not self.curTokenIs(Tok.DOUBLE_COLON):
            self.next_token()
            expr = self.parse_expression(Precedence.LOWEST)
            attrs.append(expr)
            self.next_token()
        self.next_token()  # Consume Double Colon
        return AttributeSpec(tok=self.cur_token, attrs=attrs)

    @Trace.trace_decorator("parse_subcall_statement")
    def parse_subcall_statement(self) -> Statement:
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
        if self.peekTokenIs(Tok.NEWLINE):
            index = None
            start_expr = None
            end_expr = None
            step = None
            self.next_token()  # cur token is newline

        elif self.peekTokenIs(Tok.IDENT):
            self.next_token()  # Ident = index
            index = self.cur_token
            if not self.expect_peek(Tok.ASSIGN):  # expect_peek advances token
                raise ParseError("Couldn't Parse Do LOOP")
            self.next_token()
            start_expr, end_expr, step = self.parse_do_bounds()
        else:
            err = f"Unexpected Token {self.cur_token} @{self.lineno} {self.lexer.input}"
            self.logger.error(err)
            self.errors.append(err)
            return None
        # advance to end of statement
        self.next_token()
        block = self.parse_block_statement(tok, [Token(Tok.ENDDO, "ENDDO")])
        return DoLoop(tok, index, start_expr, end_expr, block, step)

    @Trace.trace_decorator("parse_do_bounds")
    def parse_do_bounds(self) -> tuple[Expression, Expression, Optional[Expression]]:
        """
        Parse the bounds expression: start, end [, step]
        Assumes current token is just past '='
        """
        start_expr = self.parse_expression(Precedence.LOWEST)

        self.expect_peek(Tok.COMMA)
        self.next_token()  # skip the comma

        end_expr = self.parse_expression(Precedence.LOWEST)

        step_expr = None
        if self.peekTokenIs(Tok.COMMA):
            self.next_token()  # skip comma
            self.next_token()
            step_expr = self.parse_expression(Precedence.LOWEST)

        return start_expr, end_expr, step_expr

    @Trace.trace_decorator("parse_dowhile_block")
    def parse_dowhile_block(self, label: Token) -> Statement:
        """
        Parse do while block. cur_token is expected to be WHILE
        """
        assert self.curTokenIs(Tok.DOWHILE), "(parse_dowhile_block) expects DOWHILE"
        token = self.cur_token
        self.next_token()  # token should be LPAREN
        cond = self.parse_expression(Precedence.LOWEST)

        self.next_token()
        blk = self.parse_block_statement(token, [Token(Tok.ENDDO, "enddo")])
        self.next_token()  # consume ENDDO

        return DoWhile(tok=token, cond=cond, body=blk)

    @Trace.trace_decorator("parse_block_statement")
    def parse_block_statement(
        self, tok: Token, terminators: list[Token]
    ) -> BlockStatement:
        block = BlockStatement(tok)

        while not self.curTokenIs(Tok.EOF) and not any(
            self.curTokenIs(end_token.token) for end_token in terminators
        ):
            stmt = self.parse_statement()
            if stmt:
                block.statements.append(stmt)
            self.next_token()

        return block

    @Trace.trace_decorator("parse_if_construct")
    def parse_if_statement(self) -> Statement:
        """
        cur_token = IF
        """
        cur_token = self.cur_token
        blk_terminators = [
            Token(Tok.ENDIF, "ENDIF"),
            Token(Tok.ELSEIF, "ELSEIF"),
            Token(Tok.ELSE, "ELSE"),
        ]

        if not self.expect_peek(Tok.LPAREN):
            self.logger.error(f"Expected IF condition Got: {self.peek_token}")

        condition = self.parse_expression(Precedence.LOWEST)
        self.next_token()

        if self.curTokenIs(Tok.THEN):
            tok = self.cur_token
            self.next_token()
            consequence = self.parse_block_statement(tok, blk_terminators)
            if_block = IfConstruct(
                tok=cur_token, cond=condition, consequence=consequence
            )
            if_block.end_ln = self.lineno
        else:
            # simple if statement
            consequence = BlockStatement(tok=self.cur_token)
            stmt = self.parse_statement()
            assert stmt
            consequence.statements.append(stmt)
            if_block = IfConstruct(
                tok=cur_token, cond=condition, consequence=consequence
            )
            return if_block

        while self.curTokenIs(Tok.ELSEIF):
            tok = self.cur_token
            elif_ln = self.lineno
            self.expect_peek(Tok.LPAREN)
            elif_cond = self.parse_expression(Precedence.LOWEST)
            self.expect_peek(Tok.THEN)
            self.next_token()

            consequence = self.parse_block_statement(tok, blk_terminators)

            elif_block = ElseIf(cond=elif_cond, blk=consequence)
            elif_block.lineno = elif_ln
            elif_block.end_ln = self.lineno

            if_block.else_ifs.append(elif_block)

        if self.curTokenIs(Tok.ELSE):
            else_lineno = self.lineno
            else_tok = self.cur_token
            self.expect_peek(Tok.NEWLINE)
            self.next_token()
            alternative = self.parse_block_statement(
                else_tok, [Token(Tok.ENDIF, "ENDIF")]
            )
            else_ = Else(tok=else_tok, alt=alternative)
            else_.lineno = else_lineno
            else_.end_ln = self.lineno
            if_block.else_ = else_

        if self.curTokenIs(Tok.ENDIF):
            self.next_token()

        return if_block

    def parse_expression_statement(self) -> ExpressionStatement:
        stmt = ExpressionStatement(tok=self.cur_token)
        stmt.expression = self.parse_expression(Precedence.LOWEST)
        # self.next_token()
        return stmt

    @Trace.trace_decorator("parse_expression")
    def parse_expression(self, prec: Precedence) -> Expression:
        cur_type = self.cur_token.token
        prefix = self.prefix_parse_fns.get(cur_type, None)
        if not prefix:
            err = f"Unexpected Token {cur_type} at {self.lineno} {self.lexer.input}"
            self.logger.error(err)
            self.errors.append(err)
            sys.exit(1)
        left_expr: Expression = prefix()

        while (
            not self.peekTokenIs(Tok.NEWLINE)
            and prec.value < self.peek_precedence().value
        ):
            peek_type = self.peek_token.token
            if peek_type not in self.infix_parse_fns:
                return left_expr
            infix = self.infix_parse_fns[peek_type]
            self.next_token()
            left_expr = infix(left_expr)
        return left_expr

    def parse_grouped_expr(self) -> Expression:
        self.next_token()

        expr = self.parse_expression(Precedence.LOWEST)
        if not self.expect_peek(Tok.RPAREN):
            self.errors.append("Failed to Parse Grouped Expression" + str(expr))
        return expr

    def parse_prefix_expr(self) -> Expression:
        tok = self.cur_token
        op = tok.literal
        self.next_token()
        right_expr = self.parse_expression(Precedence.PREFIX)
        expr = PrefixExpression(tok=tok, op=op, right=right_expr)

        return expr

    def parse_infix_expr(self, left: Expression) -> Expression:
        """
        (parse_infix_expr)
        """
        tok = self.cur_token
        op = self.cur_token.literal
        prec = self.cur_precedence()
        self.next_token()

        right_expr = self.parse_expression(prec)
        expression = InfixExpression(
            tok=tok,
            op=op,
            left=left,
            right=right_expr,
        )
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
        args = self.parse_args()
        func_expr = FuncExpression(tok=self.cur_token, fn=func,args=args)

        return func_expr

    def parse_args(self) -> list[Expression]:
        args: list[Expression] = []

        if self.peekTokenIs(Tok.RPAREN):
            self.next_token()
            return args
        self.next_token()
        args.append(self.parse_expression(Precedence.LOWEST))
        while self.peekTokenIs(Tok.COMMA):
            self.next_token()
            self.next_token()
            # Current token is now start of next arg
            args.append(self.parse_expression(Precedence.LOWEST))

        # Note expect peek advances tokens, to cur_token = RPAREN at return
        if not self.expect_peek(Tok.RPAREN):
            raise ParseError("Couldn't Parse Arguments")
        return args

    @Trace.trace_decorator("parse_infix_bounds_expr")
    def parse_infix_bounds_expr(self, start: Expression) -> Expression:
        """
        Function to parse bounds.  curent token should be ":"
        """
        tok = self.cur_token
        start_ = start
        end_= None
        if not self.peekTokenIs(Tok.RPAREN) and not self.peekTokenIs(Tok.COMMA):
            self.next_token()
            end_ = self.parse_expression(Precedence.LOWEST)
        return BoundsExpression(tok=tok,start=start_,end=end_)

    @Trace.trace_decorator("parse_prefix_bounds_expr")
    def parse_prefix_bounds_expr(self) -> Expression:
        """
        Function to parse bounds.  curent token should be ":"
        """
        tok = self.cur_token
        end_ = None
        if not self.peekTokenIs(Tok.RPAREN) and not self.peekTokenIs(Tok.COMMA):
            self.next_token()
            end_ = self.parse_expression(Precedence.LOWEST)
        return BoundsExpression(tok=tok, start=None,end=end_)

    @Trace.trace_decorator("parse_write_statement")
    def parse_write_statement(self) -> WriteStatement:
        tok = self.cur_token  # 'write'
        self.expect_peek(Tok.LPAREN)
        self.next_token()

        # parse log unit
        unit = self.parse_expression(Precedence.LOWEST)

        if self.peekTokenIs(Tok.COMMA):
            self.expect_peek(Tok.COMMA)
            self.next_token()
            # parse format
            fmt = self.parse_expression(Precedence.LOWEST)
        else:
            # statement is of form write(logunit) expr
            fmt = IOExpression(Token(Tok.ASTERISK, "*"))

        self.expect_peek(Tok.RPAREN)
        self.next_token()
        # parse expressions
        exprs: list[Expression] = []
        exprs.append(self.parse_expression(Precedence.LOWEST))
        while self.peekTokenIs(Tok.COMMA):
            self.next_token()
            self.next_token()
            exprs.append(self.parse_expression(Precedence.LOWEST))

        return WriteStatement(tok, unit, fmt, exprs)

    def parse_print_statement(self) -> PrintStatement:
        tok = self.cur_token
        self.next_token()
        fmt = self.parse_expression(Precedence.LOWEST)

        self.expect_peek(Tok.COMMA)
        self.next_token()

        exprs: list[Expression] = []
        exprs.append(self.parse_expression(Precedence.LOWEST))
        while self.peekTokenIs(Tok.COMMA):
            self.next_token()
            self.next_token()
            exprs.append(self.parse_expression(Precedence.LOWEST))
        return PrintStatement(token=tok, fmt=fmt, exprs=exprs)

    @Trace.trace_decorator("parse_array_init")
    def parse_array_init(self) -> Expression:
        start_tok = self.cur_token.token
        end_tok = (
            Tok.ARRAY_INIT_END
            if self.curTokenIs(Tok.ARRAY_INIT_START)
            else Tok.ARRAY_RBRACKET
        )
        elements: list[Expression] = []
        implied_do = None
        while not self.curTokenIs(end_tok):
            self.next_token()
            if self.curTokenIs(Tok.IDENT) and self.peekTokenIs(Tok.ASSIGN):
                implied_do = self.parse_implied_do()
            else:
                item_expr = self.parse_expression(Precedence.LOWEST)
                elements.append(item_expr)
            self.next_token()

        return ArrayInit(
            start_tok=start_tok,
            elements=elements,
            end_tok=end_tok,
            implied_do=implied_do,
        )

    @Trace.trace_decorator("parse_implied_do")
    def parse_implied_do(self) -> ImpliedDo:
        var_tok = self.cur_token
        self.expect_peek(Tok.ASSIGN)
        self.next_token()
        start_expr, end_expr, step = self.parse_do_bounds()

        return ImpliedDo(
            index=var_tok,
            start_expr=start_expr,
            end_expr=end_expr,
            step_expr=step,
        )

    @Trace.trace_decorator("parse_procedure_stmt")
    def parse_procedure_stmt(self) -> Statement:
        """
        -cur token is Tok.PROC
        """
        tok = self.cur_token
        attr_spec = None
        alias = ""
        self.next_token()
        if self.curTokenIs(Tok.LPAREN):
            expr = self.parse_expression(Precedence.LOWEST)
            # cur token = Tok.RPAREN
            self.next_token()

        if self.curTokenIs(Tok.COMMA):
            attr_spec = self.parse_attr_spec()
        elif self.curTokenIs(Tok.DOUBLE_COLON):
            self.next_token()

        expr = self.parse_expression(Precedence.LOWEST)
        if isinstance(expr, Identifier):
            name = expr.value
        elif isinstance(expr, InfixExpression):
            assert expr.operator == "=>", f"unexpected operator {self.lexer.input}"
            alias = expr.left_expr.value
            name = expr.right_expr.value
        return ProcedureStatement(
            tok=tok,
            attr_spec=attr_spec,
            name=name,
            alias=alias,
        )
