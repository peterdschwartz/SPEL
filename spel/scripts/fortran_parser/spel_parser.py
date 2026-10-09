import sys
from logging import Logger
from types import FunctionType
from typing import Callable, List, Optional

import spel.scripts.fortran_parser.lexer as lexer
from spel.scripts.fortran_parser.spel_ast import (
    AllocateStatement,
    ArrayInit,
    AssociateConstruct,
    AttributeSpec,
    BlockStatement,
    BoundsExpression,
    CaseBlock,
    ContinueStatement,
    CycleStatement,
    DataImpliedDo,
    DataSet,
    DataStatement,
    DataValue,
    DoLoop,
    DoWhile,
    ElseWhere,
    WhereConstruct,
    Else,
    ElseIf,
    EntityDecl,
    ExitStatement,
    Expression,
    ExpressionStatement,
    FieldAccessExpression,
    FloatLiteral,
    FormatStatement,
    FuncExpression,
    GenericOperatorExpression,
    GotoStatement,
    Identifier,
    IfConstruct,
    ImplicitNoneStatement,
    ImpliedDo,
    ImportStatement,
    InfixExpression,
    IntegerLiteral,
    IntrinsicStatement,
    IOExpression,
    LogicalLiteral,
    MacroCallStatement,
    MacroBranch,
    MacroDefine,
    MacroIf,
    NameListStatement,
    NullifyStatement,
    PrefixExpression,
    PrintStatement,
    ProcedureStatement,
    Program,
    ReadStatement,
    ReturnStatement,
    SelectCaseConstruct,
    Statement,
    StopStatement,
    StringLiteral,
    SubCallStatement,
    SubroutineDefinitionConstruct,
    TypeDef,
    TypeSpec,
    UseStatement,
    VariableDecl,
    WriteStatement,
)
from spel.scripts.fortran_parser.tokens import (
    KEYWORDS_THAT_CAN_BE_IDENTIFIERS,
    Token,
    TokenTypes,
)
from spel.scripts.fortran_parser.tracing import Trace
from spel.scripts.logging_configs import get_logger
from spel.scripts.types import LineTuple, LogicalLineIterator, Precedence

precedences = {
    TokenTypes.ASSIGN: Precedence.ASSIGN,
    TokenTypes.PTR: Precedence.ASSIGN,
    TokenTypes.PLUS: Precedence.SUM,
    TokenTypes.MINUS: Precedence.SUM,
    TokenTypes.SLASH: Precedence.PRODUCT,
    TokenTypes.ASTERISK: Precedence.PRODUCT,
    TokenTypes.LPAREN: Precedence.CALL,
    TokenTypes.COLON: Precedence.BOUNDS,
    TokenTypes.PERCENT: Precedence.BOUNDS,
    TokenTypes.EXP: Precedence.EXP,
    TokenTypes.EQUIV: Precedence.EQUALS,
    TokenTypes.NOT_EQUIV: Precedence.EQUALS,
    TokenTypes.GT: Precedence.LESSGREATER,
    TokenTypes.GTEQ: Precedence.LESSGREATER,
    TokenTypes.LT: Precedence.LESSGREATER,
    TokenTypes.LTEQ: Precedence.LESSGREATER,
    TokenTypes.AND: Precedence.AND,
    TokenTypes.OR: Precedence.OR,
    TokenTypes.CONCAT: Precedence.CONCAT,
    TokenTypes.MACRO: Precedence.PREFIX,
    TokenTypes.BANG: Precedence.PREFIX,
}

PrefixParseFn = Callable[[], Expression]
InfixParseFn = Callable[[Expression], Expression]

Tok = TokenTypes

SUBPROGRAM_PREFIXES = {"pure", "elemental", "recursive", "impure"}
ALLOC_OPTIONS = {"stat", "errmsg", "source", "mold"}
USE_MODULE_NATURES = {"intrinsic", "non_intrinsic"}
# CPP function-like macros used as statements (share/include/shr_assert.h)
FUNCTION_LIKE_MACROS = {
    "shr_assert",
    "shr_assert_all",
    "shr_assert_fl",
    "shr_assert_all_fl",
}


class ParseError(Exception):
    pass


class Parser:
    def __init__(
        self,
        lex: Optional[lexer.Lexer] = None,
        logger: str = "Parser",
        lines: list[LineTuple] = None,
    ):
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
        prev_type = self.cur_token.token
        self.cur_token = self.peek_token
        self.peek_token = self.lex_next()
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
            (Tok.END, Tok.ASSOCIATE): (Tok.ENDASSOCIATE, "end associate"),
        }
        if pair in combos:
            new_type, lit = combos[pair]
            self.cur_token = Token(new_type, lit)
            self.peek_token = self.lex_next()
        elif self.curTokenIs(Tok.END) and self.peek_token.literal == "select":
            self.cur_token = Token(Tok.ENDSELECT, "end select")
            self.peek_token = self.lex_next()
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

        # Statement start is resolved here, before block loops test for
        # terminators like `else`/`contains`
        if self.is_first_token() or prev_type is Tok.SEMICOLON:
            self.cur_keyword_as_variable()

        self.lineno = self.lexer.cur_ln
        return

    def lex_next(self) -> Token:
        try:
            return self.lexer.next_token()
        except lexer.LexError as e:
            self.fatal(str(e))
            raise

    def cur_keyword_as_variable(self) -> None:
        """
        A keyword at statement start that is the target of an assignment
        (`use = 1`, `type(i)%x = 2`, `p => t`) is a variable name.
        """
        if self.cur_token.token not in KEYWORDS_THAT_CAN_BE_IDENTIFIERS:
            return
        if self.peekTokenIs(Tok.LPAREN):
            rest = self.text_after_peek_paren()
            is_var = rest.startswith(("=", "%")) and not rest.startswith("==")
        else:
            is_var = (
                self.peekTokenIs(Tok.ASSIGN)
                or self.peekTokenIs(Tok.PTR)
                or self.peekTokenIs(Tok.PERCENT)
            )
        if is_var:
            self.cur_token = Token(Tok.IDENT, self.cur_token.literal.lower())

    def text_after_peek_paren(self) -> str:
        """
        Source text following the parenthesized group opened by peek '('
        (the lexer sits just past it), leading whitespace stripped.
        """
        text = self.lexer.input
        pos = self.lexer.position
        depth = 1
        while pos < len(text) and depth > 0:
            ch = text[pos]
            if ch in ("'", '"'):
                # skip a quoted string ('it''s' is two adjacent strings)
                close = text.find(ch, pos + 1)
                pos = len(text) if close < 0 else close
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            pos += 1
        return text[pos:].lstrip()

    def peek_keyword_as_identifier(self):
        """
        Used in instances where it's ok for a keyword to be an identifier
        """
        if self.peek_token.token in KEYWORDS_THAT_CAN_BE_IDENTIFIERS:
            self.peek_token = Token(
                token=Tok.IDENT, literal=self.peek_token.literal.lower()
            )
        return

    def is_first_token(self) -> bool:
        return self.lexer.token_pos == 1

    def register_prefix_fns(self, tok_type: Tok, fn: PrefixParseFn):
        self.prefix_parse_fns[tok_type] = fn

    def register_infix_fns(self, tok_type: Tok, fn: InfixParseFn):
        self.infix_parse_fns[tok_type] = fn

    def parse_identifier(self) -> Identifier:
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
        num = FloatLiteral(tok=self.cur_token, val=value, prec=precision)
        return num

    def curTokenIs(self, etype: Tok):
        return self.cur_token.token == etype

    def peekTokenIs(self, etype: Tok):
        return self.peek_token.token == etype

    def expect_peek_and_advance(self, etype: Tok) -> bool:
        if self.peekTokenIs(etype):
            self.next_token()
            return True
        else:
            err = f"Expected: {etype}, Got: {self.peek_token} @{self.lineno} {self.lexer.input}"
            self.fatal(err)
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
            self.expect_peek_and_advance(Tok.IDENT)
            stmt = self.parse_macro_if(token, symbol=self.cur_token.literal)
        elif self.peekTokenIs(Tok.DEF):
            self.next_token()
            token = self.cur_token
            if not self.expect_peek_and_advance(Tok.IDENT):
                self.fatal("Expected macro name after #define")
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
            self.fatal(err)

        return stmt

    def parse_macro_if(
        self, token: Token, symbol: Optional[str] = None, condition: Optional[str] = None
    ) -> MacroIf:
        """
        Entry: cur_token ends the directive line (#ifdef symbol / #if token)
        Exit: the token after #endif
        """
        terminators = [
            Token(Tok.M_ENDIF, "#endif"),
            Token(Tok.M_ELIF, "#elif"),
            Token(Tok.M_ELSE, "#else"),
        ]
        self.next_token()
        body = self.parse_block_statement(token, terminators)
        branches: list[MacroBranch] = []
        while self.curTokenIs(Tok.M_ELIF) or self.curTokenIs(Tok.M_ELSE):
            branch_tok = self.cur_token
            cond = branch_tok.literal if self.curTokenIs(Tok.M_ELIF) else None
            if branches and branches[-1].condition is None:
                self.fatal("#elif/#else after #else")
            self.next_token()
            blk = self.parse_block_statement(branch_tok, terminators)
            branches.append(MacroBranch(branch_tok, cond, blk))
        if not self.curTokenIs(Tok.M_ENDIF):
            self.fatal(f"Unterminated {token.literal} block (missing #endif)")
        self.next_token()
        return MacroIf(token, symbol, body, condition=condition, branches=branches)

    def parse_cpp_if(self) -> MacroIf:
        """cur_token is M_IF (literal = raw condition)"""
        return self.parse_macro_if(self.cur_token, condition=self.cur_token.literal)

    def stdout_or_fmt(self) -> Expression:
        return IOExpression(self.cur_token)

    @Trace.trace_decorator("parse_statement")
    def parse_statement(self) -> Optional[Statement]:
        # an integer followed by a keyword/name at statement start is a numeric label
        stmt_label: Optional[int] = None
        if self.curTokenIs(Tok.INT) and self.peek_token.literal[:1].isalpha():
            stmt_label = int(self.cur_token.literal)
            self.next_token()
        label = self.check_label()
        # statement start after a numeric label or a one-line `if (c)`
        self.cur_keyword_as_variable()

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
            case Tok.M_IF:
                stmt = self.parse_cpp_if()
            case Tok.TYPE:
                stmt = self.parse_var_decl()
            case Tok.TYPE_DEF:
                stmt = self.parse_type_def()
            case Tok.PROC:
                stmt = self.parse_procedure_stmt()
            case Tok.USE:
                stmt = self.parse_use_statement()
            case Tok.NAMELIST:
                stmt = self.parse_namelist_statement()
            case Tok.ASSOCIATE:
                stmt = self.parse_associate_construct()
            case Tok.ALLOCATE | Tok.DEALLOCATE:
                stmt = self.parse_allocate_statement()
            case Tok.SUBROUTINE | Tok.FUNCTION:
                stmt = self.parse_subprogram(prefixes=[], return_type=None)
            case Tok.IDENT if self.is_subprogram_prefix():
                stmt = self.parse_prefixed_subprogram()
            case Tok.IDENT if self.is_import_statement():
                stmt = self.parse_import_statement()
            case Tok.IDENT if self.is_implicit_statement():
                stmt = self.parse_implicit_statement()
            case Tok.IDENT if self.is_branch_statement():
                stmt = self.parse_branch_statement()
            case Tok.IDENT if self.is_select_statement():
                stmt = self.parse_select_case()
            case Tok.IDENT if self.is_where_statement():
                stmt = self.parse_where()
            case Tok.IDENT if self.is_io_or_nullify_statement("nullify"):
                stmt = self.parse_nullify_statement()
            case Tok.IDENT if self.is_io_or_nullify_statement("read"):
                stmt = self.parse_read_statement()
            case Tok.IDENT if self.is_macro_call():
                stmt = self.parse_macro_call()
            case Tok.IDENT if self.is_stop_statement():
                stmt = self.parse_stop_statement()
            case Tok.IDENT if self.is_goto_statement():
                stmt = self.parse_goto_statement()
            case Tok.IDENT if (
                self.cur_token.literal == "continue" and self.peek_ends_statement()
            ):
                stmt = ContinueStatement(self.cur_token)
            case Tok.IDENT if self.is_intrinsic_statement():
                stmt = self.parse_intrinsic_statement()
            case Tok.IDENT if self.is_data_statement():
                stmt = self.parse_data_statement()
            case Tok.IDENT if stmt_label is not None and self.is_format_statement():
                stmt = self.parse_format_statement()
            case _:
                stmt = self.parse_expression_statement()
        if stmt:
            stmt.lineno = startln
            if stmt_label is not None:
                stmt.label = stmt_label
        return stmt

    @Trace.trace_decorator("parse_nml_stmt")
    def parse_namelist_statement(self) -> Statement:
        """
        Parses fortran statements: namelist /group/ <comma sep list of vars>
        """

        tok = self.cur_token

        self.expect_peek_and_advance(Tok.SLASH)  # cur token become SLASH
        self.next_token()  # cur token is group name

        group_name: Identifier = self.parse_identifier()  # doesn't advance tokens
        self.expect_peek_and_advance(Tok.SLASH)
        self.next_token()

        vars: list[Identifier] = [self.parse_identifier()]
        while self.peekTokenIs(Tok.COMMA):
            self.next_token()  # comma
            self.next_token()  # identifier
            vars.append(self.parse_identifier())

        return NameListStatement(tok=tok, namelist_group=group_name, vars=vars)

    @Trace.trace_decorator("parse_uses")
    def parse_use_statement(self) -> Statement:
        """
        Parses fortran statements:
            use [[, intrinsic|non_intrinsic] ::] <mod_name> [, rename-list]
            use [[, intrinsic|non_intrinsic] ::] <mod_name>, only : [obj1, x => obj2, operator(+)]
        Exit: cur_token is the last token of the statement
        """
        tok = self.cur_token
        nature: Optional[str] = None
        if self.peekTokenIs(Tok.COMMA):
            self.next_token()  # COMMA
            self.next_token()  # module-nature
            if self.cur_token.literal not in USE_MODULE_NATURES:
                self.fatal(f"Invalid module nature in use statement: {self.cur_token}")
            nature = self.cur_token.literal
            if not self.peekTokenIs(Tok.DOUBLE_COLON):
                self.fatal("Expected '::' after module nature in use statement")
        if self.peekTokenIs(Tok.DOUBLE_COLON):
            self.next_token()
        if not self.peekTokenIs(Tok.IDENT):
            self.fatal(f"Un-expected module name: {self.peek_token}")
        self.next_token()
        module = self.parse_identifier()

        objs: list[Expression] = []
        renames: list[Expression] = []
        has_only = False
        if self.peekTokenIs(Tok.COMMA):
            self.next_token()  # COMMA
            self.next_token()
            if self.cur_token.literal == "only" and self.peekTokenIs(Tok.COLON):
                has_only = True
                self.next_token()  # ':'
                if not self.peekTokenIs(Tok.NEWLINE) and not self.peekTokenIs(Tok.EOF):
                    self.next_token()
                    objs.append(self.parse_only_item())
                    while self.peekTokenIs(Tok.COMMA):
                        self.next_token()
                        self.next_token()
                        objs.append(self.parse_only_item())
            elif self.cur_token.literal == "only":
                self.fatal("Expected ':' after 'only' in use statement")
            else:
                renames.append(self.parse_use_rename())
                while self.peekTokenIs(Tok.COMMA):
                    self.next_token()
                    self.next_token()
                    renames.append(self.parse_use_rename())

        return UseStatement(
            tok=tok,
            mod_name=module,
            objs=objs,
            nature=nature,
            has_only=has_only,
            renames=renames,
        )

    def peek_ends_statement(self) -> bool:
        return (
            self.peekTokenIs(Tok.NEWLINE)
            or self.peekTokenIs(Tok.EOF)
            or self.peekTokenIs(Tok.SEMICOLON)
        )

    def is_branch_statement(self) -> bool:
        """exit/cycle [construct-name] | return. None are reserved words."""
        lit = self.cur_token.literal
        if lit in ("exit", "cycle"):
            return self.peek_ends_statement() or self.peekTokenIs(Tok.IDENT)
        return lit == "return" and self.peek_ends_statement()

    @Trace.trace_decorator("parse_branch")
    def parse_branch_statement(self) -> Statement:
        """Exit: cur_token is the last token of the statement"""
        tok = self.cur_token
        if tok.literal == "return":
            return ReturnStatement(tok)
        name = None
        if self.peekTokenIs(Tok.IDENT):
            self.next_token()
            name = self.cur_token.literal
        cls = ExitStatement if tok.literal == "exit" else CycleStatement
        return cls(tok, name)

    def is_select_statement(self) -> bool:
        return self.cur_token.literal == "select" and self.peekTokenIs(Tok.IDENT)

    def is_where_statement(self) -> bool:
        return self.cur_token.literal == "where" and self.peekTokenIs(Tok.LPAREN)

    def at_elsewhere(self) -> bool:
        lit = self.cur_token.literal.lower()
        return lit == "elsewhere" or (lit == "else" and self.peek_token.literal == "where")

    def at_end_where(self) -> bool:
        lit = self.cur_token.literal.lower()
        return lit == "endwhere" or (lit == "end" and self.peek_token.literal == "where")

    @Trace.trace_decorator("parse_where")
    def parse_where(self) -> WhereConstruct:
        """
        Entry: cur_token = `where`, peek = LPAREN
        Exit: last token of the WHERE statement / `end where`
        """
        tok = self.cur_token
        startln = self.lineno
        self.next_token()
        mask = self.parse_expression(Precedence.LOWEST)
        self.next_token()
        if not self.curTokenIs(Tok.NEWLINE):
            body = BlockStatement(tok)
            stmt = self.parse_statement()
            if stmt is None:
                self.fatal(f"Expected a statement after where (...) @{startln}")
            body.statements.append(stmt)
            return WhereConstruct(tok, mask, body)

        def at_branch() -> bool:
            return self.at_elsewhere() or self.at_end_where()

        self.next_token()
        body = self.parse_block_statement(tok, [], stop=at_branch)
        elsewheres: list[ElseWhere] = []
        while self.at_elsewhere():
            ew_tok = self.cur_token
            ew_ln = self.lineno
            if self.cur_token.literal.lower() == "else":
                self.next_token()  # `where`
            ew_mask = None
            if self.peekTokenIs(Tok.LPAREN):
                self.next_token()
                ew_mask = self.parse_expression(Precedence.LOWEST)
            self.expect_peek_and_advance(Tok.NEWLINE)
            self.next_token()
            blk = self.parse_block_statement(ew_tok, [], stop=at_branch)
            elsewhere = ElseWhere(ew_tok, ew_mask, blk)
            elsewhere.lineno = ew_ln
            elsewheres.append(elsewhere)
        if not self.at_end_where():
            self.fatal(f"Unterminated where construct @{startln}")
        if self.cur_token.literal.lower() == "end":
            self.next_token()  # `where`
        construct = WhereConstruct(tok, mask, body, elsewheres)
        construct.end_ln = self.lineno
        return construct

    def is_case_statement(self) -> bool:
        return (
            self.curTokenIs(Tok.IDENT)
            and self.cur_token.literal == "case"
            and (self.peekTokenIs(Tok.LPAREN) or self.peek_token.literal == "default")
        )

    @Trace.trace_decorator("parse_select_case")
    def parse_select_case(self) -> SelectCaseConstruct:
        """
        select case (expr)
        [case (value-list) | case default
           block]...
        end select [name]
        Exit: cur_token is ENDSELECT (or the trailing construct name)
        """
        tok = self.cur_token
        self.next_token()
        if self.cur_token.literal != "case":
            self.fatal(f"Unsupported construct: select {self.cur_token.literal}")
        if not self.expect_peek_and_advance(Tok.LPAREN):
            self.fatal("Expected '(' after select case")
        self.next_token()
        selector = self.parse_expression(Precedence.LOWEST)
        if not self.expect_peek_and_advance(Tok.RPAREN):
            self.fatal("Expected ')' after select case selector")
        self.next_token()
        while self.curTokenIs(Tok.NEWLINE):
            self.next_token()

        cases: list[CaseBlock] = []
        while self.is_case_statement():
            case_tok = self.cur_token
            case_ln = self.lineno
            values: Optional[list[Expression]] = None
            if self.peek_token.literal == "default":
                self.next_token()
            else:
                self.next_token()  # LPAREN
                values = []
                while not self.curTokenIs(Tok.RPAREN):
                    self.next_token()
                    values.append(self.parse_expression(Precedence.LOWEST))
                    self.next_token()  # COMMA or RPAREN
                    if not (self.curTokenIs(Tok.COMMA) or self.curTokenIs(Tok.RPAREN)):
                        self.fatal(
                            f"Unexpected token in case selector: {self.cur_token}"
                        )
            self.next_token()
            body = self.parse_block_statement(
                case_tok,
                [Token(Tok.ENDSELECT, "end select")],
                stop=self.is_case_statement,
            )
            case = CaseBlock(case_tok, values, body)
            case.lineno = case_ln
            cases.append(case)

        if not self.curTokenIs(Tok.ENDSELECT):
            self.fatal(f"Expected 'end select', got {self.cur_token}")
        stmt = SelectCaseConstruct(tok, selector, cases)
        stmt.end_ln = self.lineno
        if self.peekTokenIs(Tok.IDENT):
            self.next_token()  # construct name
        return stmt

    def is_io_or_nullify_statement(self, keyword: str) -> bool:
        """`read(...)`/`nullify(...)`; keywords are not reserved."""
        return self.cur_token.literal == keyword and self.peekTokenIs(Tok.LPAREN)

    def parse_paren_list(self) -> list[Expression]:
        """
        Entry: cur_token is LPAREN. Exit: cur_token is the matching RPAREN
        """
        items: list[Expression] = []
        if self.peekTokenIs(Tok.RPAREN):
            self.next_token()
            return items
        while not self.curTokenIs(Tok.RPAREN):
            self.next_token()
            items.append(self.parse_expression(Precedence.LOWEST))
            self.next_token()
            if not (self.curTokenIs(Tok.COMMA) or self.curTokenIs(Tok.RPAREN)):
                self.fatal(f"Expected ',' or ')' got {self.cur_token}")
        return items

    @Trace.trace_decorator("parse_nullify")
    def parse_nullify_statement(self) -> NullifyStatement:
        tok = self.cur_token
        self.next_token()
        return NullifyStatement(tok, self.parse_paren_list())

    @Trace.trace_decorator("parse_read")
    def parse_read_statement(self) -> ReadStatement:
        """
        read(control-list) [item-list]
        Exit: cur_token is the last token of the statement
        """
        tok = self.cur_token
        self.next_token()
        controls = self.parse_paren_list()
        items: list[Expression] = []
        if not self.peek_ends_statement():
            self.next_token()
            items.append(self.parse_expression(Precedence.LOWEST))
            while self.peekTokenIs(Tok.COMMA):
                self.next_token()
                self.next_token()
                items.append(self.parse_expression(Precedence.LOWEST))
        return ReadStatement(tok, controls, items)

    def is_macro_call(self) -> bool:
        return self.cur_token.literal in FUNCTION_LIKE_MACROS and self.peekTokenIs(
            Tok.LPAREN
        )

    def is_stop_statement(self) -> bool:
        """stop [code] | error stop [code]. Neither word is reserved."""
        lit = self.cur_token.literal
        if lit == "error":
            return self.peek_token.literal == "stop"
        return lit == "stop" and (
            self.peek_ends_statement()
            or self.peekTokenIs(Tok.INT)
            or self.peekTokenIs(Tok.STRING)
            or self.peekTokenIs(Tok.IDENT)
        )

    @Trace.trace_decorator("parse_stop")
    def parse_stop_statement(self) -> StopStatement:
        """Exit: cur_token is the last token of the statement"""
        tok = self.cur_token
        error = tok.literal == "error"
        if error:
            self.next_token()
        code = None
        if not self.peek_ends_statement():
            self.next_token()
            code = self.parse_expression(Precedence.LOWEST)
        return StopStatement(tok, code=code, error=error)

    def is_goto_statement(self) -> bool:
        lit = self.cur_token.literal
        if lit == "go":
            return self.peek_token.literal == "to"
        return lit == "goto" and self.peekTokenIs(Tok.INT)

    @Trace.trace_decorator("parse_goto")
    def parse_goto_statement(self) -> GotoStatement:
        """go to <label>. Exit: cur_token is the label"""
        tok = self.cur_token
        if tok.literal == "go":
            self.next_token()
        if not self.peekTokenIs(Tok.INT):
            self.fatal(
                f"Only unconditional `go to <label>` is supported @{self.lineno}"
            )
        self.next_token()
        return GotoStatement(tok, int(self.cur_token.literal))

    def is_intrinsic_statement(self) -> bool:
        return self.cur_token.literal == "intrinsic" and (
            self.peekTokenIs(Tok.IDENT) or self.peekTokenIs(Tok.DOUBLE_COLON)
        )

    @Trace.trace_decorator("parse_intrinsic")
    def parse_intrinsic_statement(self) -> IntrinsicStatement:
        """intrinsic [::] name-list. Exit: cur_token is the last name"""
        tok = self.cur_token
        if self.peekTokenIs(Tok.DOUBLE_COLON):
            self.next_token()
        names: list[str] = []
        while True:
            self.next_token()
            names.append(self.cur_token.literal)
            if not self.peekTokenIs(Tok.COMMA):
                break
            self.next_token()
        return IntrinsicStatement(tok, names)

    def is_data_statement(self) -> bool:
        """
        `data` is not reserved: `data x /1/` and `data (a(i),i=1,n) /.../` are
        DATA statements, while `data = 1` and `data(i) = x` are assignments.
        """
        if self.cur_token.literal.lower() != "data":
            return False
        self.peek_keyword_as_identifier()
        if self.peekTokenIs(Tok.IDENT):
            return True
        if not self.peekTokenIs(Tok.LPAREN):
            return False
        # a DATA implied-do is followed by its value list, `(...) /`
        rest = self.text_after_peek_paren()
        return rest.startswith("/") and not rest.startswith("/=")

    def is_format_statement(self) -> bool:
        """`<label> format (...)` with nothing after the parenthesized spec"""
        return (
            self.cur_token.literal.lower() == "format"
            and self.peekTokenIs(Tok.LPAREN)
            and self.text_after_peek_paren() == ""
        )

    @Trace.trace_decorator("parse_format")
    def parse_format_statement(self) -> FormatStatement:
        """
        The spec is taken as raw text: edit descriptors are not tokenized.
        Exit: cur_token is the closing ')'
        """
        tok = self.cur_token
        raw = self.lexer.read_rest_of_line().rstrip()
        if not raw.endswith(")"):
            self.fatal(f"Malformed FORMAT statement: {raw}")
        self.cur_token = Token(Tok.RPAREN, ")")
        self.peek_token = self.lex_next()
        return FormatStatement(tok, raw[:-1].strip())

    @Trace.trace_decorator("parse_data")
    def parse_data_statement(self) -> DataStatement:
        """
        data obj-list /value-list/ [[,] obj-list /value-list/]...
        Exit: cur_token is the closing '/' of the last set
        """
        tok = self.cur_token
        sets: list[DataSet] = []
        while True:
            self.next_token()
            objects = [self.parse_data_object()]
            while self.peekTokenIs(Tok.COMMA):
                self.next_token()
                self.next_token()
                objects.append(self.parse_data_object())
            self.expect_peek_and_advance(Tok.SLASH)
            values: list[DataValue] = []
            while True:
                self.next_token()
                values.append(self.parse_data_value())
                if not self.peekTokenIs(Tok.COMMA):
                    break
                self.next_token()
            self.expect_peek_and_advance(Tok.SLASH)
            sets.append(DataSet(objects, values))
            if self.peek_ends_statement():
                break
            if self.peekTokenIs(Tok.COMMA):
                self.next_token()
        return DataStatement(tok, sets)

    def parse_data_object(self) -> Expression:
        """variable | array element/section | data-implied-do"""
        if self.curTokenIs(Tok.LPAREN):
            return self.parse_data_implied_do()
        # PRODUCT stops before the '/' that opens the value list
        return self.parse_expression(Precedence.PRODUCT)

    def parse_data_implied_do(self) -> DataImpliedDo:
        """(obj-list, i = start, end[, step]). Exit: cur_token is ')'"""
        tok = self.cur_token
        objects: list[Expression] = []
        while True:
            self.next_token()
            if self.curTokenIs(Tok.IDENT) and self.peekTokenIs(Tok.ASSIGN):
                loop = self.parse_implied_do()
                self.expect_peek_and_advance(Tok.RPAREN)
                return DataImpliedDo(tok, objects, loop)
            objects.append(self.parse_data_object())
            self.expect_peek_and_advance(Tok.COMMA)

    def parse_data_value(self) -> DataValue:
        """[repeat*]constant. Exit: cur_token is the last token of the value"""
        value = self.parse_expression(Precedence.PRODUCT)
        if self.peekTokenIs(Tok.ASTERISK):
            self.next_token()
            self.next_token()
            return DataValue(self.parse_expression(Precedence.PRODUCT), repeat=value)
        return DataValue(value)

    @Trace.trace_decorator("parse_macro_call")
    def parse_macro_call(self) -> MacroCallStatement:
        tok = self.cur_token
        self.next_token()
        return MacroCallStatement(tok, tok.literal, self.parse_paren_list())

    def is_implicit_statement(self) -> bool:
        """`implicit` is not reserved; `implicit = 1` is an assignment."""
        return self.cur_token.literal == "implicit" and (
            self.peekTokenIs(Tok.IDENT)
            or self.peekTokenIs(Tok.TYPE)
            or self.peekTokenIs(Tok.TYPE_DEF)
        )

    @Trace.trace_decorator("parse_implicit")
    def parse_implicit_statement(self) -> ImplicitNoneStatement:
        """
        Parses `implicit none`. Implicit typing is rejected: SPEL's static
        analysis requires every symbol to be declared.
        Exit: cur_token is the last token of the statement
        """
        tok = self.cur_token
        self.next_token()
        if self.cur_token.literal != "none":
            self.fatal(
                f"Implicit typing is not supported: implicit {self.cur_token.literal}"
            )
        if not (self.peekTokenIs(Tok.NEWLINE) or self.peekTokenIs(Tok.EOF)):
            self.fatal(f"Unexpected token after 'implicit none': {self.peek_token}")
        return ImplicitNoneStatement(tok=tok)

    def is_import_statement(self) -> bool:
        """`import` is not reserved; only treat it as a statement by its follow token."""
        return self.cur_token.literal == "import" and (
            self.peekTokenIs(Tok.DOUBLE_COLON)
            or self.peekTokenIs(Tok.COMMA)
            or self.peekTokenIs(Tok.IDENT)
            or self.peekTokenIs(Tok.NEWLINE)
            or self.peekTokenIs(Tok.EOF)
        )

    @Trace.trace_decorator("parse_import")
    def parse_import_statement(self) -> ImportStatement:
        """
        Parses fortran statements:
            import [[::] name-list]
            import, only : name-list
            import, none | all
        Exit: cur_token is the last token of the statement
        """
        tok = self.cur_token
        spec: Optional[str] = None
        if self.peekTokenIs(Tok.COMMA):
            self.next_token()  # COMMA
            self.next_token()
            spec = self.cur_token.literal
            if spec == "only":
                if not self.peekTokenIs(Tok.COLON):
                    self.fatal("Expected ':' after 'import, only'")
                self.next_token()
            elif spec in ("none", "all"):
                if not (self.peekTokenIs(Tok.NEWLINE) or self.peekTokenIs(Tok.EOF)):
                    self.fatal(
                        f"Unexpected token after 'import, {spec}': {self.peek_token}"
                    )
                return ImportStatement(tok=tok, names=[], spec=spec)
            else:
                self.fatal(f"Invalid import specifier: {self.cur_token}")
        elif self.peekTokenIs(Tok.DOUBLE_COLON):
            self.next_token()
            if not self.peekTokenIs(Tok.IDENT):
                self.fatal(f"Expected name after 'import ::', got {self.peek_token}")

        names: list[str] = []
        if self.peekTokenIs(Tok.IDENT) or spec == "only":
            names.append(self.next_import_name())
            while self.peekTokenIs(Tok.COMMA):
                self.next_token()
                names.append(self.next_import_name())
        return ImportStatement(tok=tok, names=names, spec=spec)

    def next_import_name(self) -> str:
        if not self.peekTokenIs(Tok.IDENT):
            self.fatal(f"Expected name in import statement, got {self.peek_token}")
        self.next_token()
        return self.cur_token.literal

    def parse_only_item(self) -> Expression:
        op = self.check_operator_def()
        return op if op is not None else self.parse_expression(Precedence.LOWEST)

    def parse_use_rename(self) -> Expression:
        rename = self.parse_expression(Precedence.LOWEST)
        if not (
            isinstance(rename, InfixExpression)
            and rename.operator == "=>"
            and isinstance(rename.left_expr, Identifier)
            and isinstance(rename.right_expr, Identifier)
        ):
            self.fatal(f"Expected 'local => use_name' in use statement, got {rename}")
        return rename

    @Trace.trace_decorator("check_operator")
    def check_operator_def(self) -> Optional[GenericOperatorExpression]:
        if not self.peekTokenIs(Tok.LPAREN):
            return None
        if self.cur_token.literal == "assignment":
            tok = Token(token=Tok.IDENT, literal="assignment")
            _ = self.expect_peek_and_advance(
                Tok.LPAREN
            )  # not needed but advances tokens
            _ = self.expect_peek_and_advance(Tok.ASSIGN)
            spec: Identifier = Identifier(tok=self.cur_token, value="=")
            _ = self.expect_peek_and_advance(Tok.RPAREN)
        elif self.cur_token.literal == "operator":
            tok = Token(token=Tok.IDENT, literal="operator")
            _ = self.expect_peek_and_advance(Tok.LPAREN)
            self.next_token()
            spec: Identifier = self.parse_identifier()
            _ = self.expect_peek_and_advance(Tok.RPAREN)
        else:
            self.errors.append(
                f"(check_operator_def) Unexpected token {self.cur_token}"
            )
            self.error_exit()
        return GenericOperatorExpression(tok=tok, spec=spec)

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
        self.expect_peek_and_advance(Tok.NEWLINE)
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
        Also dispatches typed function definitions, e.g. `real(r8) function f(x)`
        """
        type_tok = self.cur_token
        type_spec = self.parse_type_spec()
        if self.curTokenIs(Tok.FUNCTION) or self.is_subprogram_prefix():
            return self.parse_prefixed_subprogram(return_type=type_spec)

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

    def parse_type_spec(self) -> TypeSpec:
        """
        Enter: cur_token is .TYPE
        Exit:  cur_token is the token following the type-spec
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

        return TypeSpec(base_type=base_type, kind=kind, len_=len_)

    def is_subprogram_prefix(self) -> bool:
        """
        cur_token is an IDENT like `pure`/`elemental` that begins a subprogram header
        (and not e.g. a variable named `pure` in an assignment)
        """
        if not self.curTokenIs(Tok.IDENT):
            return False
        if self.cur_token.literal not in SUBPROGRAM_PREFIXES:
            return False
        if self.peek_token.token == Tok.IDENT:
            return self.peek_token.literal in SUBPROGRAM_PREFIXES
        return self.peek_token.token in {
            Tok.TYPE,
            Tok.TYPE_DEF,
            Tok.FUNCTION,
            Tok.SUBROUTINE,
        }

    def fatal(self, msg: str):
        """
        Unrecoverable error: parse_program exits once the ParseError unwinds
        """
        err = f"{msg} @{self.lineno}\n{self.lexer.get_current_line()}"
        self.errors.append(err)
        # print(self.lexer.line_iter.lines)
        raise ParseError(err)

    @Trace.trace_decorator("parse_prefixed_subprogram")
    def parse_prefixed_subprogram(
        self, return_type: Optional[TypeSpec] = None
    ) -> Statement:
        """
        Collects prefix-specs (pure, elemental, ...) and at most one type-spec,
        in any order, up to FUNCTION/SUBROUTINE.
        Enter: cur_token is a prefix IDENT or, if return_type is given, the token after it
        """
        prefixes: list[str] = []
        while not (self.curTokenIs(Tok.FUNCTION) or self.curTokenIs(Tok.SUBROUTINE)):
            if (
                self.curTokenIs(Tok.IDENT)
                and self.cur_token.literal in SUBPROGRAM_PREFIXES
            ):
                prefixes.append(self.cur_token.literal)
                self.next_token()
                continue
            # `type` not at start of line is demoted to IDENT by next_token
            if self.curTokenIs(Tok.IDENT) and self.cur_token.literal == "type":
                self.cur_token = Token(Tok.TYPE, "type")
            if self.curTokenIs(Tok.TYPE) and return_type is None:
                return_type = self.parse_type_spec()
                continue
            self.fatal(f"Unexpected {self.cur_token} in subprogram prefix {prefixes}")

        if return_type is not None and self.curTokenIs(Tok.SUBROUTINE):
            self.fatal(f"SUBROUTINE cannot have a type-spec ({return_type})")
        return self.parse_subprogram(prefixes=prefixes, return_type=return_type)

    @Trace.trace_decorator("parse_subprogram")
    def parse_subprogram(
        self, prefixes: list[str], return_type: Optional[TypeSpec]
    ) -> Statement:
        """
        Enter: cur_token is SUBROUTINE or FUNCTION
        Exit:  cur_token is the END token, or the trailing name of the end statement
        """
        tok = self.cur_token
        start_ln = self.lineno
        is_function = self.curTokenIs(Tok.FUNCTION)
        end_tok = (
            Token(Tok.ENDFUNC, "ENDFUNC")
            if is_function
            else Token(Tok.ENDSUB, "ENDSUB")
        )

        self.peek_keyword_as_identifier()

        if not self.expect_peek_and_advance(Tok.IDENT):
            self.fatal(f"Expected name after {tok.literal}")
        name = self.cur_token.literal

        args: list[str] = []
        if self.peekTokenIs(Tok.LPAREN):
            self.next_token()
            for arg in self.parse_args():
                if not isinstance(arg, Identifier):
                    self.fatal(f"Invalid dummy argument '{arg}' in {name}")
                args.append(arg.value)

        result: Optional[str] = None
        if is_function:
            result = name
            if self.peekTokenIs(Tok.IDENT) and self.peek_token.literal == "result":
                self.next_token()
                if not self.expect_peek_and_advance(Tok.LPAREN):
                    self.fatal(f"Expected '(' after result in {name}")
                res_args = self.parse_args()
                if len(res_args) != 1 or not isinstance(res_args[0], Identifier):
                    self.fatal(f"Invalid result clause in {name}")
                result = res_args[0].value
        self.next_token()

        contains_tok = Token(Tok.CONTAINS, "contains")
        body = self.parse_block_statement(tok, [contains_tok, end_tok])

        contains: list[SubroutineDefinitionConstruct] = []
        if self.curTokenIs(Tok.CONTAINS):
            self.next_token()
            internal = self.parse_block_statement(contains_tok, [end_tok])
            for stmt in internal.statements:
                if not isinstance(stmt, SubroutineDefinitionConstruct):
                    self.fatal(
                        f"Only subprograms allowed after CONTAINS in {name}: {stmt}"
                    )
                contains.append(stmt)

        if not self.curTokenIs(end_tok.token):
            self.fatal(f"Missing END for {tok.literal} {name} (started @{start_ln})")
        end_ln = self.lineno
        self.peek_keyword_as_identifier()
        if self.peekTokenIs(Tok.IDENT):
            self.next_token()
            if self.cur_token.literal != name:
                self.fatal(
                    f"END name '{self.cur_token.literal}' does not match '{name}'"
                )

        stmt = SubroutineDefinitionConstruct(
            tok=tok,
            name=name,
            prefixes=prefixes,
            args=args,
            return_type=return_type,
            result=result,
            body=body,
            contains=contains,
        )
        stmt.end_ln = end_ln
        return stmt

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
            if not self.expect_peek_and_advance(
                Tok.ASSIGN
            ):  # expect_peek advances token
                raise ParseError("Couldn't Parse Do LOOP")
            self.next_token()
            start_expr, end_expr, step = self.parse_do_bounds()
        else:
            err = f"Unexpected Token {self.cur_token} @{self.lineno} {self.lexer.input}"
            self.fatal(msg=err)
            self.errors.append(err)
            return None
        # advance to end of statement
        self.next_token()
        block = self.parse_block_statement(tok, [Token(Tok.ENDDO, "ENDDO")])
        self.skip_construct_name()
        return DoLoop(tok, index, start_expr, end_expr, block, step)

    def skip_construct_name(self) -> None:
        """cur_token ends a construct (`end do`): consume an `end do <name>`"""
        if self.peekTokenIs(Tok.IDENT):
            self.next_token()

    @Trace.trace_decorator("parse_do_bounds")
    def parse_do_bounds(self) -> tuple[Expression, Expression, Optional[Expression]]:
        """
        Parse the bounds expression: start, end [, step]
        Assumes current token is just past '='
        """
        start_expr = self.parse_expression(Precedence.LOWEST)

        self.expect_peek_and_advance(Tok.COMMA)
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
        self,
        tok: Token,
        terminators: list[Token],
        stop: Optional[Callable[[], bool]] = None,
    ) -> BlockStatement:
        """
        Parses statements until a terminator token (or `stop()` is True)
        is the current token at the start of a statement.
        """
        block = BlockStatement(tok)

        while not self.curTokenIs(Tok.EOF) and not any(
            self.curTokenIs(end_token.token) for end_token in terminators
        ):
            if stop is not None and stop():
                break
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

        if not self.expect_peek_and_advance(Tok.LPAREN):
            self.fatal(f"Expected IF condition Got: {self.peek_token}")

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
            self.expect_peek_and_advance(Tok.LPAREN)
            elif_cond = self.parse_expression(Precedence.LOWEST)
            self.expect_peek_and_advance(Tok.THEN)
            self.next_token()

            consequence = self.parse_block_statement(tok, blk_terminators)

            elif_block = ElseIf(cond=elif_cond, blk=consequence)
            elif_block.lineno = elif_ln
            elif_block.end_ln = self.lineno

            if_block.else_ifs.append(elif_block)

        if self.curTokenIs(Tok.ELSE):
            else_lineno = self.lineno
            else_tok = self.cur_token
            self.expect_peek_and_advance(Tok.NEWLINE)
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

    @Trace.trace_decorator("parse_associate_construct")
    def parse_associate_construct(self) -> Statement:
        """
        cur_token = ASSOCIATE
        associate ( name => selector [, name => selector]... )
           block
        end associate
        """
        tok = self.cur_token
        if not self.expect_peek_and_advance(Tok.LPAREN):
            raise ParseError("Expected '(' after ASSOCIATE")

        associations: dict[str, Expression] = {}
        for arg in self.parse_args():
            if not (
                isinstance(arg, InfixExpression)
                and arg.operator == "=>"
                and isinstance(arg.left_expr, Identifier)
            ):
                raise ParseError(f"Invalid association '{arg}' @{self.lineno}")
            name = arg.left_expr.value
            if name in associations:
                raise ParseError(f"Duplicate associate-name '{name}' @{self.lineno}")
            associations[name] = arg.right_expr
        self.next_token()  # move past RPAREN

        body = self.parse_block_statement(
            tok, [Token(Tok.ENDASSOCIATE, "end associate")]
        )
        stmt = AssociateConstruct(tok=tok, associations=associations, body=body)
        stmt.end_ln = self.lineno
        return stmt

    @Trace.trace_decorator("parse_allocate_statement")
    def parse_allocate_statement(self) -> AllocateStatement:
        """
        allocate([type-spec ::] obj-list [, alloc-opt-list])
        deallocate(obj-list [, dealloc-opt-list])
        Enter: cur_token is ALLOCATE/DEALLOCATE
        Exit:  cur_token is the closing RPAREN
        """
        tok = self.cur_token
        is_dealloc = tok.token == Tok.DEALLOCATE
        allowed_opts = (
            ALLOC_OPTIONS - {"source", "mold"} if is_dealloc else ALLOC_OPTIONS
        )
        if not self.expect_peek_and_advance(Tok.LPAREN):
            self.fatal(f"Expected '(' after {tok.literal}")
        if self.peekTokenIs(Tok.RPAREN):
            self.fatal(f"Empty {tok.literal} statement")
        self.next_token()

        type_spec: Optional[TypeSpec] = None
        first: Optional[Expression] = None
        if self.curTokenIs(Tok.TYPE):
            # intrinsic type-spec: parse_type_spec leaves cur on the token after it
            type_spec = self.parse_type_spec()
            if not self.curTokenIs(Tok.DOUBLE_COLON):
                self.fatal(f"Expected '::' after type-spec in {tok.literal}")
        else:
            first = self.parse_expression(Precedence.LOWEST)
            if self.peekTokenIs(Tok.DOUBLE_COLON):
                # derived type-spec is a bare type name
                if not isinstance(first, Identifier):
                    self.fatal(f"Invalid type-spec '{first}' in {tok.literal}")
                type_spec = TypeSpec(base_type=Token(Tok.TYPE, first.value), kind="")
                first = None
                self.next_token()
        if type_spec is not None and is_dealloc:
            self.fatal("deallocate does not take a type-spec")

        items: list[Expression] = [first] if first is not None else []
        if first is None:
            self.next_token()  # cur on first item after '::'
            items.append(self.parse_expression(Precedence.LOWEST))
        while self.peekTokenIs(Tok.COMMA):
            self.next_token()
            self.next_token()
            items.append(self.parse_expression(Precedence.LOWEST))
        if not self.expect_peek_and_advance(Tok.RPAREN):
            self.fatal(f"Expected ')' to close {tok.literal}")

        objects: list[Expression] = []
        options: dict[str, Expression] = {}
        for item in items:
            is_opt = (
                isinstance(item, InfixExpression)
                and item.operator == "="
                and isinstance(item.left_expr, Identifier)
            )
            if not is_opt:
                if options:
                    self.fatal(f"{tok.literal} object '{item}' follows options")
                objects.append(item)
                continue
            name = item.left_expr.value
            if name not in allowed_opts:
                self.fatal(f"Invalid {tok.literal} option '{name}'")
            if name in options:
                self.fatal(f"Duplicate {tok.literal} option '{name}'")
            options[name] = item.right_expr
        if not objects:
            self.fatal(f"{tok.literal} has no objects")

        return AllocateStatement(
            tok=tok, type_spec=type_spec, objects=objects, options=options
        )

    def parse_expression_statement(self) -> ExpressionStatement:
        stmt = ExpressionStatement(tok=self.cur_token)
        stmt.expression = self.parse_expression(Precedence.LOWEST)
        # self.next_token()
        return stmt

    @Trace.trace_decorator("parse_expression")
    def parse_expression(self, prec: Precedence) -> Expression:
        cur_type = self.cur_token.token
        prefix = self.prefix_parse_fns.get(cur_type, None)
        if not prefix and cur_type in KEYWORDS_THAT_CAN_BE_IDENTIFIERS:
            # no reserved words: a keyword in operand position is a name
            self.cur_token = Token(Tok.IDENT, self.cur_token.literal.lower())
            prefix = self.parse_identifier
        if not prefix:
            err = f"Unexpected Token {cur_type} at {self.lineno} {self.lexer.input}"
            self.fatal(err)
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
        if not self.expect_peek_and_advance(Tok.RPAREN):
            self.errors.append("Failed to Parse Grouped Expression" + str(expr))
        return expr

    def parse_prefix_expr(self) -> Expression:
        tok = self.cur_token
        op = tok.literal
        self.next_token()
        prec = Precedence.NOT if tok.token == Tok.BANG else Precedence.PREFIX
        right_expr = self.parse_expression(prec)
        expr = PrefixExpression(tok=tok, op=op, right=right_expr)

        return expr

    @Trace.trace_decorator("parse_infix_expr")
    def parse_infix_expr(self, left: Expression) -> Expression:
        """
        (parse_infix_expr)
        """
        tok = self.cur_token
        op = self.cur_token.literal
        prec = self.cur_precedence()
        if tok.token == Tok.EXP:
            # ** is right-associative
            prec = Precedence(prec.value - 1)
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
        func_expr = FuncExpression(tok=self.cur_token, fn=func, args=args)

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
        if not self.expect_peek_and_advance(Tok.RPAREN):
            raise ParseError("Couldn't Parse Arguments")
        return args

    @Trace.trace_decorator("parse_infix_bounds_expr")
    def parse_infix_bounds_expr(self, start: Expression) -> Expression:
        """
        Function to parse bounds.  curent token should be ":"
        """
        tok = self.cur_token
        start_ = start
        end_ = None
        if not self.peekTokenIs(Tok.RPAREN) and not self.peekTokenIs(Tok.COMMA):
            self.next_token()
            end_ = self.parse_expression(Precedence.LOWEST)
        return BoundsExpression(tok=tok, start=start_, end=end_)

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
        return BoundsExpression(tok=tok, start=None, end=end_)

    @Trace.trace_decorator("parse_write_statement")
    def parse_write_statement(self) -> WriteStatement:
        tok = self.cur_token  # 'write'
        self.expect_peek_and_advance(Tok.LPAREN)
        self.next_token()

        # parse log unit
        unit = self.parse_expression(Precedence.LOWEST)

        if self.peekTokenIs(Tok.COMMA):
            self.expect_peek_and_advance(Tok.COMMA)
            self.next_token()
            # parse format
            fmt = self.parse_expression(Precedence.LOWEST)
        else:
            # statement is of form write(logunit) expr
            fmt = IOExpression(Token(Tok.ASTERISK, "*"))

        self.expect_peek_and_advance(Tok.RPAREN)
        # parse expressions (none for namelist output)
        exprs: list[Expression] = []
        if self.peek_ends_statement():
            return WriteStatement(tok, unit, fmt, exprs)
        self.next_token()
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

        self.expect_peek_and_advance(Tok.COMMA)
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
        self.expect_peek_and_advance(Tok.ASSIGN)
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
