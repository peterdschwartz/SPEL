import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Tuple

from spel.scripts.fortran_parser.tokens import Token, TokenTypes

if TYPE_CHECKING:
    from spel.scripts.fortran_parser.symbols import Scope


# helpers
def NOT(x):
    return PrefixExpression(tok=Token(TokenTypes.BANG, ".not."), op=".not.", right=x)


def AND(a, b):
    return InfixExpression(
        tok=Token(TokenTypes.AND, ".and."), left=a, op=".and.", right=b
    )


class SemanticError(Exception):
    pass


# Base interface: Node
class Node(ABC):
    @abstractmethod
    def token_literal(self) -> str:
        """Return the literal value of the token."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def classify(self, scope: "Scope") -> "Node":
        """
        Refine a syntactically ambiguous node using the symbol table.
        Shallow: only this node is refined; children are left as parsed.
        """
        return self


# Derived interface: Statement
class Statement(Node):
    # numeric statement label (e.g. `30 continue`); set by the parser
    label: Optional[int] = None

    def __init__(self, lineno: int = -1):
        self.lineno: int = lineno

    @abstractmethod
    def statement_node(self) -> None:
        """Marker method for statement nodes."""
        pass

    def to_dict(self):
        return {"Node": self.__class__.__name__}


# Derived interface: Expression
class Expression(Node):
    @abstractmethod
    def expression_node(self) -> None:
        """Marker method for expression nodes."""
        pass

    def to_dict(self):
        return {"Node": self.__class__.__name__}


class Program(Statement):
    def __init__(self):
        self.statements: List[Statement] = []

    def token_literal(self) -> str:
        if len(self.statements) > 0:
            return self.statements[0].token_literal()
        else:
            return ""

    def statement_node(self) -> None:
        pass

    def __str__(self):
        return "\n".join(str(stmt) for stmt in self.statements)


class Identifier(Expression):
    def __init__(self, tok: Token, value: str):
        self.token = tok
        self.value: str = value

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return self.value

    def __repr__(self):
        return f"Ident({self.value})"

    def expression_node(self) -> None:
        pass

    def __eq__(self, other):
        return isinstance(other, Identifier) and self.value == other.value

    def get_name(self) -> str:
        return f"{self.value}"

    def to_dict(self):
        return {"Node": "Ident", "Val": str(self)}


# Statement Classes
class ExpressionStatement(Statement):
    def __init__(self, tok: Token):
        self.token = tok
        self.expression: Expression

    def statement_node(self):
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        return str(self.expression)

    def __eq__(self, other):
        if not isinstance(other, ExpressionStatement):
            return False
        else:
            return self.token == other.token and self.expression == other.expression

    def to_dict(self):
        return {"Node": "ExpressionStatement", "Expr": self.expression.to_dict()}

    def copy(self):
        return deepcopy(self)

    def classify(self, scope: "Scope") -> Statement:
        expr = self.expression
        if isinstance(expr, InfixExpression) and expr.operator in ("=", "=>"):
            if expr.operator == "=":
                stmt = AssignmentStatement(self.token, expr.left_expr, expr.right_expr)
            else:
                stmt = PointerAssignment(self.token, expr.left_expr, expr.right_expr)
            stmt.lineno = self.lineno
            return stmt
        raise SemanticError(f"Expression is not a statement @{self.lineno}: {expr}")


class SubCallStatement(Statement):
    def __init__(self, tok):
        self.token: Token = tok  # "CALL"
        self.function: FuncExpression  # FuncExpression

    def statement_node(self):
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        return "CALL " + str(self.function)

    def __eq__(self, other):
        if not isinstance(other, SubCallStatement):
            return False
        else:
            return self.token == other.token and self.function == other.function

    def to_dict(self):
        return {"Node": "SubCallStatement", "Sub": self.function.to_dict()}

    def copy(self):
        return deepcopy(self)

    def classify(self, scope: "Scope") -> "SubCallStatement":
        """Converts keyword actual arguments. The callee is resolved by the walk."""
        stmt = SubCallStatement(self.token)
        stmt.lineno = self.lineno
        fn = self.function
        stmt.function = FuncExpression(
            fn.token, fn.function, [to_actual_arg(a) for a in fn.args]
        )
        return stmt


class IntegerLiteral(Expression):
    def __init__(self, tok: Token, val: int, prec: str):
        self.token: Token = tok
        self.value: int = val
        self.precision: str = prec

    def token_literal(self) -> str:
        return str(self.value)

    def expression_node(self) -> None:
        pass

    def __str__(self):
        return self.token_literal()

    def __eq__(self, other):
        return isinstance(other, IntegerLiteral) and self.value == other.value

    def to_dict(self):
        return {"Node": "IntegerLiteral", "Val": self.value, "Prec": self.precision}


class StringLiteral(Expression):
    def __init__(self, tok: Token, val: str):
        self.token: Token = tok  # SQUOTE or DQUOTE
        self.value: str = val

    def token_literal(self) -> str:
        return str(self.value)

    def expression_node(self) -> None:
        pass

    def __str__(self):
        return f"{self.value}"

    def __eq__(self, other):
        return isinstance(other, StringLiteral) and self.value == other.value

    def to_dict(self):
        return {"Node": "StringLiteral", "Val": self.value}


class LogicalLiteral(Expression):
    def __init__(self, tok: Token, val: bool):
        self.token: Token = tok
        self.value: bool = val

    def token_literal(self) -> str:
        return str(self.value)

    def expression_node(self) -> None:
        pass

    def __str__(self):
        return self.token_literal()

    def __eq__(self, other):
        return isinstance(other, LogicalLiteral) and self.value == other.value

    def to_dict(self):
        return {"Node": "LogicalLiteral", "Val": self.value}


class FloatLiteral(Expression):
    def __init__(self, tok: Token, val: float, prec: str):
        self.token: Token = tok
        self.value: float = val
        self.precision: str = prec

    def token_literal(self) -> str:
        return self.token.literal

    def expression_node(self) -> None:
        pass

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        return isinstance(other, FloatLiteral) and self.value == other.value

    def to_dict(self):
        return {"Node": "FloatLiteral", "Val": self.value}


class IOExpression(Expression):
    def __init__(self, tok: Token, expr=None):
        self.token = tok
        self.expr: Optional[Expression] = expr

    def token_literal(self) -> str:
        return self.token.literal

    def expression_node(self) -> None:
        pass

    def __str__(self):
        return str(self.expr) if self.expr else self.token.literal

    def __eq__(self, other):
        return (
            isinstance(other, IOExpression)
            and self.token == other.token
            and self.expr == other.expr
        )

    def to_dict(self):
        return {"Node": "IOExpression", "token": self.token, "expr": self.expr}

    def copy(self):
        return deepcopy(self)


class PrefixExpression(Expression):
    def __init__(self, tok: Token, op: str, right: Expression):
        self.token: Token = tok
        self.right_expr: Expression = right
        self.operator: str = op

    def token_literal(self) -> str:
        return self.token.literal

    def expression_node(self) -> None:
        pass

    def __str__(self):
        return f"({self.operator}{str(self.right_expr)})"

    def __eq__(self, other):
        if not isinstance(other, PrefixExpression):
            return False
        else:
            return (
                self.token == other.token
                and self.right_expr == other.right_expr
                and self.operator == other.operator
            )

    def to_dict(self):
        return {
            "Node": "PrefixExpression",
            "Op": self.operator,
            "Right": self.right_expr.to_dict(),
        }

    def copy(self):
        return deepcopy(self)


class InfixExpression(Expression):
    def __init__(
        self,
        tok: Token,
        left: Expression,
        op: str,
        right: Expression,
    ):
        self.token: Token = tok
        self.left_expr: Expression = left
        self.operator: str = op
        self.right_expr: Expression = right

    def expression_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def decompose(self) -> Tuple[Expression, str, Expression]:
        return (self.left_expr, self.operator, self.right_expr)

    def __str__(self):
        return "("+str(self.left_expr) + f"{self.operator}" + str(self.right_expr)+")"

    def __eq__(self, other):
        if not isinstance(other, InfixExpression):
            return False
        else:
            return (
                self.token == other.token
                and self.right_expr == other.right_expr
                and self.operator == other.operator
                and self.left_expr == other.left_expr
            )

    def copy(self):
        return deepcopy(self)

    def to_dict(self):
        return {
            "Node": "InfixExpression",
            "Left": self.left_expr.to_dict(),
            "Op": self.operator,
            "Right": self.right_expr.to_dict(),
        }


class FieldAccessExpression(Expression):
    def __init__(
        self,
        tok: Token,
        left: Expression,
        field: Expression,
    ):
        self.token: Token = tok  # '%'
        self.left: Expression = left
        self.field: Expression = field

    def expression_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        return f"{self.left}%{self.field}"

    def __eq__(self, other):
        if not isinstance(other, FieldAccessExpression):
            return False
        else:
            return (
                self.token == other.token
                and self.left == other.left
                and self.field == other.field
            )

    def copy(self):
        return deepcopy(self)

    def to_dict(self):
        return {
            "Node": "FieldAccessExpression",
            "Left": self.left.to_dict(),
            "Field": self.field.to_dict(),
        }


class FuncExpression(Expression):
    """
    Expression for functions or arrays. Infix operator expression
    """

    def __init__(self, tok: Token, fn: Expression, args: list[Expression]):
        self.token: Token = tok  # '('
        self.function: Expression = fn  # Identifier
        self.args: list[Expression] = args

    def expression_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def get_name(self) -> str:
        return self.function.value

    def __str__(self):
        args = ",".join(str(arg) for arg in self.args)
        return str(self.function) + "(" + args + ")"

    def __eq__(self, other):
        if not isinstance(other, FuncExpression):
            return False
        else:
            return (
                self.token == other.token
                and self.function == other.function
                and self.args == other.args
            )

    def to_dict(self):
        return {
            "Node": "FuncExpression",
            "Func": str(self.function),
            "Args": [arg.to_dict() for arg in self.args],
        }

    def copy(self):
        return deepcopy(self)

    def classify(self, scope: "Scope") -> "ArrayRef | FunctionCall":
        """
        name(...) is an array element/section if `name` (or the base of
        `a%b%name`) is a data object, otherwise a function reference.
        Component names (a%b(i)) are treated as array components: type-bound
        function references are not distinguished without type information.
        """
        from spel.scripts.fortran_parser.symbols import SymbolKind

        name = str(self.function)
        base = name.split("%")[0]
        sym = scope.resolve(base, context=str(self))
        if sym.is_data or (sym.kind is SymbolKind.EXTERNAL and "%" in name):
            return ArrayRef(self.token, self.function, self.args)
        if "%" in name:
            raise SemanticError(f"'{base}' is not a data object in {self}")
        return FunctionCall(
            self.token,
            self.function,
            [to_actual_arg(a) for a in self.args],
            intrinsic=sym.kind is SymbolKind.INTRINSIC,
        )


class ArrayRef(FuncExpression):
    """Array element or section: a(i), x%y(:, j)"""

    def classify(self, scope: "Scope") -> "ArrayRef":
        return self


class FunctionCall(FuncExpression):
    """Function reference; args may contain KeywordArgument."""

    def __init__(
        self, tok: Token, fn: Expression, args: list[Expression], intrinsic: bool
    ):
        super().__init__(tok, fn, args)
        self.name: str = str(fn)
        self.intrinsic: bool = intrinsic

    def classify(self, scope: "Scope") -> "FunctionCall":
        return self


class KeywordArgument(Expression):
    """keyword = value actual argument"""

    def __init__(self, tok: Token, keyword: str, value: Expression):
        self.token = tok
        self.keyword: str = keyword
        self.value: Expression = value

    def expression_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return f"{self.keyword}={self.value}"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, KeywordArgument)
            and self.keyword == other.keyword
            and self.value == other.value
        )

    def to_dict(self):
        return {
            "Node": "KeywordArgument",
            "Keyword": self.keyword,
            "Value": self.value.to_dict(),
        }


def to_actual_arg(arg: Expression) -> Expression:
    if (
        isinstance(arg, InfixExpression)
        and arg.operator == "="
        and isinstance(arg.left_expr, Identifier)
    ):
        return KeywordArgument(arg.token, arg.left_expr.value, arg.right_expr)
    return arg


class BoundsExpression(Expression):
    def __init__(self, tok, start: Optional[Expression], end=Optional[Expression]):
        self.token: Token = tok  # Colon
        self.start = start
        self.end = end

    def expression_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return super().token_literal()

    def __str__(self):
        s = "" if self.start is None else self.start
        e = "" if self.end is None else self.end
        return f"{s}:{e}"

    def __eq__(self, other):
        if not isinstance(other, BoundsExpression):
            return False
        else:
            return (
                self.token == other.token
                and self.start == other.start
                and self.end == other.end
            )

    def to_dict(self):
        return {
            "Node": "BoundsExpression",
            "Val": str(self),
            "Start": self.start.to_dict() if self.start else None,
            "End": self.end.to_dict() if self.end else None,
        }

    def copy(self):
        return deepcopy(self)


class GenericOperatorExpression(Expression):
    def __init__(self, tok: Token, spec: Identifier):
        assert tok.literal in {"operator", "assignment"}
        self.token: Token = tok
        self.interface: Identifier = spec

    def expression_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return f"{self.token_literal()}({self.interface})"

    def to_dict(self):
        return {
            "Node": "GenericOperatorExpression",
            "Token": self.token_literal(),
            "Interface": self.interface.value,
        }


class BlockStatement(Statement):
    def __init__(self, tok: Token):
        self.token: Token = tok
        self.statements: list[Statement] = []

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        str_list = [f"{stmt.lineno}   {stmt}" for stmt in self.statements]
        str_list.append("   }")
        return "\n".join(str_list)

    def __eq__(self, other) -> bool:
        if not isinstance(other, BlockStatement):
            return False
        elif len(self.statements) != len(other.statements):
            return False
        else:
            test = [
                self.statements[i] == other.statements[i]
                for i in range(len(self.statements))
            ]
            return all(test)


class DoLoop(Statement):
    def __init__(
        self,
        tok: Token,
        index: Optional[Token],
        start: Optional[Expression],
        end: Optional[Expression],
        body: BlockStatement,
        step: Optional[Expression],
    ):
        self.token: Token = tok
        self.index: Optional[str] = index.literal if index else None
        self.start: Optional[Expression] = start
        self.end: Optional[Expression] = end
        self.step: Optional[Expression] = step
        self.body: BlockStatement = body

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return super().token_literal()

    def __str__(self):
        result = f"{self.token} {self.index} = {self.start}, {self.end} {{\n"
        result += f"   {self.body}\n"
        return result

    def __eq__(self, other):
        if not isinstance(other, DoLoop):
            return False
        else:
            return (
                self.token == other.token
                and self.index == other.index
                and self.start == other.start
                and self.end == other.end
            )

    def to_dict(self):
        return {
            "Node": "DoLoop",
            "Val": str(self),
        }


class DoWhile(Statement):
    def __init__(
        self,
        tok: Token,
        cond: Expression,
        body: BlockStatement,
    ):
        self.token = tok
        self.condition = cond
        self.body: BlockStatement = body

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return super().token_literal()

    def __str__(self):
        return f"{self.token}({self.condition}){{\n {self.body}"

    def __eq__(self, other):
        if not isinstance(other, DoWhile):
            return False
        else:
            return self.token == other.token and self.condition == other.condition

    def to_dict(self):
        return {
            "Node": "DoWhile",
            "Val": str(self),
        }


class MacroBranch(Statement):
    """`#elif <condition>` or `#else` (condition None) branch of a MacroIf"""

    def __init__(self, token: Token, condition: Optional[str], body: BlockStatement):
        self.token = token
        self.condition = condition
        self.body = body

    def statement_node(self):
        pass

    def token_literal(self) -> str:
        return super().token_literal()

    def __str__(self) -> str:
        head = f"#elif {self.condition}" if self.condition is not None else "#else"
        return f"{head} {self.body}"


class MacroIf(Statement):
    """
    #ifdef/#ifndef <symbol>  or  #if <condition>, then optional #elif/#else
    branches. Conditions are raw C-preprocessor text.
    """

    def __init__(
        self,
        token: Token,
        symbol: Optional[str],
        body: BlockStatement,
        condition: Optional[str] = None,
        branches: Optional[list[MacroBranch]] = None,
    ):
        self.token = token  # The #ifdef/#ifndef/#if token
        self.symbol = symbol  # The macro symbol (#ifdef/#ifndef)
        self.condition = condition  # raw condition (#if)
        self.body = body
        self.branches: list[MacroBranch] = branches or []

    def statement_node(self):
        pass

    def token_literal(self) -> str:
        return super().token_literal()

    def __str__(self) -> str:
        head = self.symbol if self.condition is None else self.condition
        tail = "".join(f" {b}" for b in self.branches)
        return f"{self.token} {head} {self.body}{tail}"


class MacroDefine(Statement):
    def __init__(self, token: Token, symbol: str, value: Optional[Expression] = None):
        self.token = token
        self.symbol = symbol
        self.value = value

    def statement_node(self):
        pass

    def token_literal(self) -> str:
        return super().token_literal()

    def __str__(self) -> str:
        return f"{self.token} {self.symbol} {self.value}"


class ElseIf(Statement):
    def __init__(self, cond: Expression, blk: BlockStatement):
        self.condition = cond
        self.consequence: BlockStatement = blk
        self.end_ln: int = -1

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return super().token_literal()

    def __str__(self):
        return f"{self.lineno} else if {self.condition} {{\n  {self.consequence}\n}}"

    def copy(self):
        return deepcopy(self)


class Else(Statement):
    def __init__(self, tok: Token, alt: BlockStatement):
        self.token = tok
        self.alternative = alt
        self.end_ln: int = -1

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return super().token_literal()

    def __eq__(self, other):
        if not isinstance(other, Else):
            return False
        else:
            return self.token == other.token and self.alternative == other.alternative

    def __str__(self) -> str:
        return f"{self.lineno}  else {{\n {self.alternative}"

    def copy(self):
        return deepcopy(self)


class IfConstruct(Statement):
    def __init__(
        self,
        tok: Token,
        cond: Expression,
        consequence: BlockStatement,
    ):
        self.token = tok
        self.condition = cond
        self.consequence: BlockStatement = consequence
        self.else_ifs: list[ElseIf] = []
        self.else_: Optional[Else] = None
        self.end_ln: int = -1

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return super().token_literal()

    def __str__(self):
        ln = f"{self.lineno}"
        result = f"{ln} if {self.condition} {{\n  {self.consequence}\n  "
        for elif_expr in self.else_ifs:
            result += (
                f"\n{ln} else if {elif_expr.condition} {{\n  {elif_expr.consequence}\n "
            )

        if self.else_:
            result += f"{self.else_}"

        return result

    def __eq__(self, other):
        if not isinstance(other, IfConstruct):
            return False
        else:
            return self.token == other.token and self.condition == other.condition

    def to_dict(self):
        return {
            "Node": "If",
            "condition": self.condition,
        }

    def copy(self):
        return deepcopy(self)

    def build_branch_guards(self) -> tuple[list[Expression], Optional[Expression]]:
        """
        Returns (guards, else_guard)

        guards[0] = C1                      (IF)
        guards[i] = (¬C1 ∧ ... ∧ ¬C_i) ∧ C_{i+1}  for i>=1  (ELSEIFs)
        else_guard = ¬C1 ∧ ¬C2 ∧ ... ∧ ¬C_N       (ELSE), or None if no ELSE
        """
        conds = [self.condition] + [eif.condition for eif in self.else_ifs]

        guards: list[Expression] = []
        P = None  # running prefix P = ∧(¬C_k) for prior branches

        for i, cond in enumerate(conds):
            if i == 0:
                guards.append(cond)
                P = NOT(cond)  # P = ¬C1
            else:
                guards.append(AND(P, cond))  # G_i = P ∧ C_i
                P = AND(P, NOT(cond))  # P = P ∧ ¬C_i

        else_guard = P if self.else_ is not None else None
        return guards, else_guard


class AssociateConstruct(Statement):
    def __init__(
        self,
        tok: Token,
        associations: dict[str, Expression],
        body: BlockStatement,
    ):
        self.token = tok
        self.associations: dict[str, Expression] = associations
        self.body: BlockStatement = body
        self.end_ln: int = -1

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return super().token_literal()

    def __str__(self):
        assoc = ", ".join(f"{k} => {v}" for k, v in self.associations.items())
        return f"{self.lineno} associate({assoc}) {{\n  {self.body}\n}}"

    def __eq__(self, other):
        if not isinstance(other, AssociateConstruct):
            return False
        return (
            self.token == other.token
            and self.associations == other.associations
            and self.body == other.body
        )

    def to_dict(self):
        return {
            "Node": "Associate",
            "associations": {k: str(v) for k, v in self.associations.items()},
        }


class AssignmentStatement(Statement):
    def __init__(self, tok: Token, lhs: Expression, rhs: Expression):
        self.token: Token = tok
        self.left = lhs
        self.right = rhs

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return f"{self.left} = {self.right}"


class PointerAssignment(Statement):
    def __init__(self, tok: Token, pointer: Expression, target: Expression):
        self.token: Token = tok
        self.pointer = pointer
        self.target = target

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return f"{self.pointer} => {self.target}"


class ImplicitNoneStatement(Statement):
    def __init__(self, tok: Token):
        self.token = tok

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return "implicit none"

    def __eq__(self, other) -> bool:
        return isinstance(other, ImplicitNoneStatement)

    def to_dict(self):
        return {"Node": "ImplicitNoneStatement"}


class ExitStatement(Statement):
    """exit [construct-name]"""

    def __init__(self, tok: Token, construct_name: Optional[str] = None):
        self.token = tok
        self.construct_name: Optional[str] = construct_name

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        name = f" {self.construct_name}" if self.construct_name else ""
        return f"{self.token_literal()}{name}"

    def __eq__(self, other) -> bool:
        return type(other) is type(self) and self.construct_name == other.construct_name

    def to_dict(self):
        return {"Node": type(self).__name__, "Name": self.construct_name}


class CycleStatement(ExitStatement):
    """cycle [construct-name]"""


class ReturnStatement(Statement):
    def __init__(self, tok: Token):
        self.token = tok

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return "return"

    def __eq__(self, other) -> bool:
        return isinstance(other, ReturnStatement)

    def to_dict(self):
        return {"Node": "ReturnStatement"}


class StopStatement(Statement):
    """[error] stop [stop-code]"""

    def __init__(self, tok: Token, code: Optional[Expression] = None, error: bool = False):
        self.token = tok
        self.code: Optional[Expression] = code
        self.error: bool = error

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        prefix = "error stop" if self.error else "stop"
        return f"{prefix} {self.code}" if self.code is not None else prefix

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, StopStatement)
            and self.error == other.error
            and self.code == other.code
        )

    def to_dict(self):
        return {
            "Node": "StopStatement",
            "Error": self.error,
            "Code": self.code.to_dict() if self.code is not None else None,
        }


class GotoStatement(Statement):
    """go to <label>"""

    def __init__(self, tok: Token, target: int):
        self.token = tok
        self.target: int = target

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return f"go to {self.target}"

    def __eq__(self, other) -> bool:
        return isinstance(other, GotoStatement) and self.target == other.target

    def to_dict(self):
        return {"Node": "GotoStatement", "Target": self.target}


class ContinueStatement(Statement):
    def __init__(self, tok: Token):
        self.token = tok

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return "continue"

    def __eq__(self, other) -> bool:
        return isinstance(other, ContinueStatement)

    def to_dict(self):
        return {"Node": "ContinueStatement"}


class IntrinsicStatement(Statement):
    """intrinsic [::] name-list"""

    def __init__(self, tok: Token, names: list[str]):
        self.token = tok
        self.names: list[str] = names

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return f"intrinsic :: {', '.join(self.names)}"

    def __eq__(self, other) -> bool:
        return isinstance(other, IntrinsicStatement) and self.names == other.names

    def to_dict(self):
        return {"Node": "IntrinsicStatement", "Names": self.names}


class DataImpliedDo(Expression):
    """data-implied-do: (obj-list, i = start, end[, step]); objects may nest"""

    def __init__(self, tok: Token, objects: list[Expression], loop: "ImpliedDo"):
        self.token = tok
        self.objects: list[Expression] = objects
        self.loop: ImpliedDo = loop

    def expression_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        objs = ", ".join(str(o) for o in self.objects)
        return f"({objs}, {self.loop})"

    def to_dict(self):
        return {
            "Node": "DataImpliedDo",
            "objects": [o.to_dict() for o in self.objects],
            "loop": self.loop.to_dict(),
        }


class DataValue:
    """[repeat*]constant"""

    def __init__(self, value: Expression, repeat: Optional[Expression] = None):
        self.value: Expression = value
        self.repeat: Optional[Expression] = repeat

    def __str__(self) -> str:
        return f"{self.repeat}*{self.value}" if self.repeat is not None else str(self.value)

    def to_dict(self):
        return {
            "Node": "DataValue",
            "repeat": self.repeat.to_dict() if self.repeat is not None else None,
            "value": self.value.to_dict(),
        }


class DataSet:
    """obj-list /value-list/"""

    def __init__(self, objects: list[Expression], values: list[DataValue]):
        self.objects: list[Expression] = objects
        self.values: list[DataValue] = values

    def __str__(self) -> str:
        objs = ", ".join(str(o) for o in self.objects)
        vals = ", ".join(str(v) for v in self.values)
        return f"{objs} /{vals}/"

    def to_dict(self):
        return {
            "Node": "DataSet",
            "objects": [o.to_dict() for o in self.objects],
            "values": [v.to_dict() for v in self.values],
        }


class DataStatement(Statement):
    """data obj-list /value-list/ [[,] obj-list /value-list/]..."""

    def __init__(self, tok: Token, sets: list[DataSet]):
        self.token = tok
        self.sets: list[DataSet] = sets

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return "data " + ", ".join(str(s) for s in self.sets)

    def to_dict(self):
        return {"Node": "DataStatement", "sets": [s.to_dict() for s in self.sets]}


class FormatStatement(Statement):
    """
    <label> format (format-spec)
      * spec: raw text inside the outer parentheses; edit descriptors
        (1x, i10, f21.15, /, 3(...)) are not expressions and are not parsed
    """

    def __init__(self, tok: Token, spec: str):
        self.token = tok
        self.spec: str = spec

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return f"format ({self.spec})"

    def __eq__(self, other) -> bool:
        return isinstance(other, FormatStatement) and self.spec == other.spec

    def to_dict(self):
        return {"Node": "FormatStatement", "Spec": self.spec}


class CaseBlock(Statement):
    """
    case (value-list) | case default
      * values: None for `case default`; items may be BoundsExpression ranges
    """

    def __init__(self, tok: Token, values: Optional[list[Expression]], body: BlockStatement):
        self.token = tok
        self.values: Optional[list[Expression]] = values
        self.body: BlockStatement = body

    @property
    def is_default(self) -> bool:
        return self.values is None

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        sel = "default" if self.values is None else f"({', '.join(map(str, self.values))})"
        return f"{self.lineno} case {sel} {{\n  {self.body}\n}}"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, CaseBlock)
            and self.values == other.values
            and self.body == other.body
        )


class SelectCaseConstruct(Statement):
    def __init__(self, tok: Token, selector: Expression, cases: list[CaseBlock]):
        self.token = tok
        self.selector = selector
        self.cases: list[CaseBlock] = cases
        self.end_ln: int = -1

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        cases = "\n".join(str(c) for c in self.cases)
        return f"{self.lineno} select case ({self.selector}) {{\n{cases}\n}}"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, SelectCaseConstruct)
            and self.selector == other.selector
            and self.cases == other.cases
        )

    def to_dict(self):
        return {
            "Node": "SelectCaseConstruct",
            "Selector": self.selector.to_dict(),
            "Cases": [
                None if c.values is None else [v.to_dict() for v in c.values]
                for c in self.cases
            ],
        }


class NullifyStatement(Statement):
    def __init__(self, tok: Token, objects: list[Expression]):
        self.token = tok
        self.objects: list[Expression] = objects

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return f"nullify({', '.join(map(str, self.objects))})"

    def __eq__(self, other) -> bool:
        return isinstance(other, NullifyStatement) and self.objects == other.objects

    def to_dict(self):
        return {"Node": "NullifyStatement", "Objects": [o.to_dict() for o in self.objects]}


class ReadStatement(Statement):
    """
    read(control-list) [input-item-list]
      * controls: positional (unit, fmt) or keyword (InfixExpression '=') specs
      * items: input items (designators written by the read)
    """

    def __init__(self, tok: Token, controls: list[Expression], items: list[Expression]):
        self.token = tok
        self.controls: list[Expression] = controls
        self.items: list[Expression] = items

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        ctrl = ",".join(map(str, self.controls))
        return f"READ({ctrl}) {', '.join(map(str, self.items))}".rstrip()

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ReadStatement)
            and self.controls == other.controls
            and self.items == other.items
        )

    def to_dict(self):
        return {
            "Node": "ReadStatement",
            "Controls": [c.to_dict() for c in self.controls],
            "Items": [i.to_dict() for i in self.items],
        }


class MacroCallStatement(Statement):
    """
    Function-like CPP macro used as a statement, e.g. SHR_ASSERT(cond, msg).
    Arguments are treated as reads.
    """

    def __init__(self, tok: Token, name: str, args: list[Expression]):
        self.token = tok
        self.name: str = name
        self.args: list[Expression] = args

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        return f"{self.name}({', '.join(map(str, self.args))})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, MacroCallStatement)
            and self.name == other.name
            and self.args == other.args
        )

    def to_dict(self):
        return {
            "Node": "MacroCallStatement",
            "Name": self.name,
            "Args": [a.to_dict() for a in self.args],
        }


class WriteStatement(Statement):
    def __init__(
        self,
        token: Token,
        unit: Expression,
        fmt: Expression,
        exprs: list[Expression],
    ):
        self.token = token  # the 'write' token
        self.unit = unit  # e.g., '*' for default unit
        self.fmt = fmt  # e.g., '*' for default format
        self.exprs = exprs  # expressions to output

    def statement_node(self):
        return super().statement_node()

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        unit_str = self.unit or "default"
        fmt_str = self.fmt or "default"
        exprs_str = ", ".join(str(e) for e in self.exprs)
        return f"WRITE({unit_str},{fmt_str}) {exprs_str}"


class PrintStatement(Statement):
    def __init__(
        self,
        token: Token,
        fmt: Expression,
        exprs: list[Expression],
    ):
        self.token = token  # the 'print' token
        self.fmt = fmt  # e.g., '*' for default format
        self.exprs = exprs

    def statement_node(self):
        return super().statement_node()

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        fmt_str = self.fmt or "default"
        exprs_str = ", ".join(str(e) for e in self.exprs)
        return f"PRINT {fmt_str} {exprs_str}"


class TypeSpec(Expression):
    def __init__(self, base_type: Token, kind: str, len_=None):
        self.token = base_type
        self.kind = kind
        self.len_: Optional[Expression] = len_

    def __str__(self) -> str:
        s = self.token.literal
        if self.kind:
            s += f"({self.kind})"
        elif self.len_ is not None:
            s += f"({self.len_})"  # character length
        return s

    def expression_node(self):
        return super().expression_node()

    def token_literal(self) -> str:
        return self.token.literal

    def to_dict(self):
        return {
            "Node": "TypeSpec",
            "token": self.token.literal,
        }


class AttributeSpec(Expression):
    def __init__(self, tok: Token, attrs: list[Expression]):
        self.token = tok
        self.attrs: list[Expression] = attrs  # e.g., intent(in), dimension(3,3)

    def expression_node(self):
        return super().expression_node()

    def token_literal(self) -> str:
        return self.token.literal

    def get_dim(self) -> int:
        for attr in self.attrs:
            if isinstance(attr, FuncExpression):
                if attr.function.value == "dimension":
                    return len(attr.args)
        return -1

    def __str__(self) -> str:
        if not self.attrs:
            return ""
        else:
            str_ = ",".join([f"{attr}" for attr in self.attrs])
            return str_

    def to_dict(self):
        return {
            "Node": "AttributeSpec",
            "token": self.token.literal,
            "attrs": [attr.to_dict() for attr in self.attrs],
        }


class EntityDecl(Expression):
    def __init__(
        self,
        tok: Token,
        bounds: list[BoundsExpression],
        init: Optional[Expression],
    ):
        self.token: Token = tok
        self.bounds = bounds
        self.init = init

    def expression_node(self):
        return super().expression_node()

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        s = self.token.literal
        if self.bounds:
            s += "(" + ",".join([str(bnds) for bnds in self.bounds]) + ")"
        if self.init is not None:
            s += f" = {self.init}"
        return s

    def get_bounds_str(self) -> str:
        return "(" + ",".join([str(bnds) for bnds in self.bounds]) + ")"

    def to_dict(self):
        return {
            "Node": "EntityDecl",
            "token": self.token.literal,
            "bounds": [bnd.to_dict() for bnd in self.bounds],
            "init": self.init.to_dict() if self.init else None,
        }


class VariableDecl(Statement):
    def __init__(
        self,
        tok: Token,
        type_spec: TypeSpec,
        attrs: Optional[AttributeSpec],
        entities: list[EntityDecl],
    ):
        self.token = tok
        self.var_type = type_spec
        self.attrs = attrs
        self.entities = entities

    def statement_node(self):
        return super().statement_node()

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        ents = ",".join([str(en) for en in self.entities])
        attr_str = f", {self.attrs}" if self.attrs else ""
        return f"{self.var_type}{attr_str} :: {ents}"

    def get_attr_list(self) -> set[str]:
        if self.attrs:
            return {attr.get_name() for attr in self.attrs.attrs}
        else:
            return set()

    def to_dict(self):
        return {
            "Node": "VariableDecl",
            "token": self.token,
            "type_spec": self.var_type.to_dict(),
            "attr": self.attrs.to_dict() if self.attrs else None,
            "entities": [ent.to_dict() for ent in self.entities],
        }


class SubroutineDefinitionConstruct(Statement):
    """
    Subroutine or function definition:
      [prefix...] [type-spec] function name([args]) [result(res)]
      [prefix...] subroutine name[([args])]
        body
      [contains
        internal subprograms]
      end subroutine|function [name]
    """

    def __init__(
        self,
        tok: Token,
        name: str,
        prefixes: list[str],
        args: list[str],
        return_type: Optional[TypeSpec],
        result: Optional[str],
        body: BlockStatement,
        contains: list["SubroutineDefinitionConstruct"],
    ):
        self.token = tok
        self.name = name
        self.is_function: bool = tok.token == TokenTypes.FUNCTION
        self.prefixes = prefixes
        self.args = args
        self.return_type = return_type
        self.result = result
        self.body = body
        self.contains = contains
        self.end_ln: int = -1

    def statement_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        kind = "function" if self.is_function else "subroutine"
        header = " ".join(self.prefixes)
        if self.return_type:
            header += f" {self.return_type}"
        header = f"{header} {kind} {self.name}({', '.join(self.args)})".strip()
        if self.is_function and self.result != self.name:
            header += f" result({self.result})"
        result = f"{self.lineno} {header} {{\n  {self.body}\n"
        for sub in self.contains:
            result += f" contains {sub}\n"
        return result + "}"

    def __eq__(self, other):
        if not isinstance(other, SubroutineDefinitionConstruct):
            return False
        return (
            self.name == other.name
            and self.is_function == other.is_function
            and self.prefixes == other.prefixes
            and self.args == other.args
            and str(self.return_type) == str(other.return_type)
            and self.result == other.result
            and self.body == other.body
            and self.contains == other.contains
        )

    def to_dict(self):
        return {
            "Node": "SubroutineDefinition",
            "name": self.name,
            "is_function": self.is_function,
            "prefixes": self.prefixes,
            "args": self.args,
            "return_type": self.return_type.to_dict() if self.return_type else None,
            "result": self.result,
            "contains": [sub.to_dict() for sub in self.contains],
        }


class ImpliedDo:
    """
    Represents an implied-do array constructor:
      expr, expr, ..., expr_var = start, stop[, step]
    """

    def __init__(
        self,
        index: Token,
        start_expr: Expression,
        end_expr: Expression,
        step_expr: Optional[Expression] = None,
    ):
        self.index = index
        self.start_expr = start_expr
        self.end_expr = end_expr
        self.step_expr = step_expr

    def to_dict(self):
        return {
            "Node": "ImpliedDo",
            "index": self.index.literal,
            "start": self.start_expr.to_dict(),
            "end": self.end_expr.to_dict(),
        }

    def __str__(self) -> str:
        step_str = f", {self.step_expr}" if self.step_expr else ""
        return f"{self.index}={self.start_expr},{self.end_expr}{step_str}"


class ArrayInit(Expression):
    """
    Represents a Fortran array constructor:
      Old style: (/ expr1, expr2, ... /)
      Modern style: [expr1, expr2, ...]
    Can include nested implied-do loops.
    """

    def __init__(
        self,
        start_tok: TokenTypes,
        elements: list[Expression],
        end_tok: TokenTypes,
        implied_do: Optional[ImpliedDo],
    ):
        self.start_tok = start_tok
        self.elements = elements
        self.end_tok = end_tok
        self.implied_do = implied_do

    def expression_node(self) -> None:
        pass

    def token_literal(self) -> str:
        return self.start_tok.value

    def __str__(self) -> str:
        inner = ", ".join(str(elem) for elem in self.elements)
        impl_ = "" if not self.implied_do else str(self.implied_do)
        return f"{self.start_tok.value}{inner}{self.end_tok.value} {impl_}"

    def to_dict(self):
        return {
            "Node": "ArrayInit",
            "elements": [el.to_dict() for el in self.elements],
            "do": str(self.implied_do),
        }


class TypeDef(Statement):
    def __init__(
        self,
        tok: Token,
        name: str,
        attr_spec: Optional[AttributeSpec],
        body: BlockStatement,
        methods: Optional[BlockStatement],
    ):
        self.token = tok
        self.name = name
        self.attrs = attr_spec
        self.members = body
        self.methods = methods

    def statement_node(self) -> None:
        return super().statement_node()

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        attrs = f"{self.attrs}" if self.attrs else ""
        str_ = f"type {attrs} :: {self.name}{{\n{self.members}"
        if self.methods:
            str_ += f"\n contains {{\n {self.methods}"
        return str_


class ProcedureStatement(Statement):
    def __init__(
        self, tok: Token, attr_spec: Optional[AttributeSpec], name: str, alias: str
    ):
        self.token = tok
        self.attrs = attr_spec
        # name => alias
        self.name = name
        self.alias = alias

    def token_literal(self) -> str:
        return self.token.literal

    def statement_node(self) -> None:
        return super().statement_node()

    def __str__(self) -> str:
        attrs = f",{self.attrs}" if self.attrs else ""
        alias = f"=> {self.alias}" if self.alias else ""
        return f"procedure {attrs} :: {self.name} {alias}"


class UseStatement(Statement):
    """
    * module: module name
    * nature: "intrinsic" | "non_intrinsic" | None
    * has_only: True if an `only:` clause is present (it may be empty)
    * objs: the only-list (empty when there is no only clause)
    * renames: `local => use_name` list when there is no only clause
    """

    def __init__(
        self,
        tok: Token,
        mod_name: Identifier,
        objs: list[Expression],
        nature: Optional[str] = None,
        has_only: Optional[bool] = None,
        renames: Optional[list[Expression]] = None,
    ):
        self.token = tok  # Should be 'Ident'
        self.module = mod_name.value
        self.objs = objs
        self.nature: Optional[str] = nature
        self.has_only: bool = bool(objs) if has_only is None else has_only
        self.renames: list[Expression] = renames if renames is not None else []

    def token_literal(self) -> str:
        return self.token.literal

    def statement_node(self) -> None:
        return super().statement_node()

    def __str__(self) -> str:
        nature = f", {self.nature} ::" if self.nature else ""
        if self.has_only:
            use_str = f", only : {','.join(list(map(str,self.objs)))}"
        elif self.renames:
            use_str = f", {', '.join(map(str, self.renames))}"
        else:
            use_str = ""
        return f"{self.lineno}: use{nature} {self.module}{use_str}"

    def to_dict(self):
        return {
            "Node": "UseStatement",
            "Module": self.module,
            "Nature": self.nature,
            "HasOnly": self.has_only,
            "objs": [expr.to_dict() for expr in self.objs],
            "renames": [expr.to_dict() for expr in self.renames],
        }

class ImportStatement(Statement):
    """
    import [[::] name-list]
    import, only : name-list
    import, none | all
      * spec: None | "only" | "none" | "all"
      * names: imported host entity names
    """

    def __init__(self, tok: Token, names: list[str], spec: Optional[str] = None):
        self.token = tok
        self.names: list[str] = names
        self.spec: Optional[str] = spec

    def token_literal(self) -> str:
        return self.token.literal

    def statement_node(self) -> None:
        return super().statement_node()

    def __str__(self) -> str:
        names = ", ".join(self.names)
        if self.spec == "only":
            return f"import, only: {names}"
        if self.spec:
            return f"import, {self.spec}"
        return f"import :: {names}" if names else "import"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ImportStatement)
            and self.names == other.names
            and self.spec == other.spec
        )

    def to_dict(self):
        return {"Node": "ImportStatement", "Spec": self.spec, "Names": self.names}


class AllocateStatement(Statement):
    """
    allocate([type-spec ::] obj-list [, alloc-opt-list])
    deallocate(obj-list [, dealloc-opt-list])
      * type_spec: Optional[TypeSpec]
      * objects: list[Expression]
      * options: dict[str, Expression]  (stat, errmsg, source, mold)
    """

    def __init__(
        self,
        tok: Token,
        type_spec: Optional[TypeSpec],
        objects: list[Expression],
        options: dict[str, Expression],
    ):
        self.token = tok  # ALLOCATE or DEALLOCATE
        self.type_spec: Optional[TypeSpec] = type_spec
        self.objects: list[Expression] = objects
        self.options: dict[str, Expression] = options

    @property
    def is_deallocate(self) -> bool:
        return self.token.token == TokenTypes.DEALLOCATE

    def token_literal(self) -> str:
        return self.token.literal

    def statement_node(self) -> None:
        return super().statement_node()

    def __str__(self) -> str:
        items = ", ".join(str(o) for o in self.objects)
        if self.type_spec is not None:
            items = f"{self.type_spec} :: {items}"
        opts = "".join(f", {k}={v}" for k, v in self.options.items())
        return f"{self.token.literal}({items}{opts})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, AllocateStatement)
            and self.token == other.token
            and str(self.type_spec) == str(other.type_spec)
            and self.objects == other.objects
            and self.options == other.options
        )

    def to_dict(self):
        return {
            "Node": "AllocateStatement",
            "Token": self.token.literal,
            "TypeSpec": self.type_spec.to_dict() if self.type_spec else None,
            "Objects": [o.to_dict() for o in self.objects],
            "Options": {k: v.to_dict() for k, v in self.options.items()},
        }


class NameListStatement(Statement):
    """
    * token
    * namelist_group
    * vars
    """

    def __init__(self, tok: Token, namelist_group: Identifier, vars: list[Identifier]):
        self.token = tok  # Should be namelist
        self.namelist_group: str = namelist_group.value
        self.vars: list[str] = [v.value for v in vars]

    def token_literal(self) -> str:
        return self.token.literal

    def statement_node(self) -> None:
        return super().statement_node()

    def __str__(self) -> str:
        _s = ",\n".join(self.vars)
        return f"namelist /{self.namelist_group}/ {_s}"

    def to_dict(self):
        return {
            "Node": "NameListStatement",
            "nml_group": self.namelist_group,
            "vars": self.vars.copy(),
        }

def expr_from_dict(d: dict | None) -> Optional[Expression]:
    if d is None:
        return None

    node = d.get("Node")
    match node:
        case "Ident":
            return Identifier(None, value=d["Val"])
        case "StringLiteral":
            return StringLiteral(tok=None, val=d["Val"])
        case "FloatLiteral":
            return FloatLiteral(tok=None, val=d["Val"], prec="")
        case "LogicalLiteral":
            return LogicalLiteral(None, val=d["Val"])
        case "IntegerLiteral":
            return IntegerLiteral(tok=None, val=d["Val"], prec=d["Prec"])
        case "PrefixExpression":
            return PrefixExpression(
                tok=None, op=d["Op"], right=expr_from_dict(d["Right"])
            )
        case "InfixExpression":
            return InfixExpression(
                tok=None,
                left=expr_from_dict(d["Left"]),  # <-- recursion
                op=d["Op"],
                right=expr_from_dict(d["Right"]),  # <-- recursion
            )
        case "FuncExpression":
            args = [expr_from_dict(arg) for arg in d["Args"]]
            return FuncExpression(tok=None, fn=d["Func"], args=args)
        case "FieldAccessExpression":
            return FieldAccessExpression(
                left=expr_from_dict(d["Left"]),
                field=expr_from_dict(d["Field"]),
                tok=None,
            )
        case "BoundsExpression":
            return BoundsExpression(
                tok=None,
                start=expr_from_dict(d["Start"]),
                end=expr_from_dict(d["End"]),
            )

    raise ValueError(f"Unknown node type: {node}")


def expr_to_json(expr) -> str:
    """Serialize an Expression to a canonical JSON string."""
    if expr is None:
        return "null"
    return json.dumps(expr.to_dict(), separators=(",", ":"), sort_keys=True)


def expr_from_json(s: str) -> Optional[Expression]:
    """Deserialize a JSON string into an Expression (or None)."""
    d = json.loads(s)
    return expr_from_dict(d)  # your function
