import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Tuple

from scripts.fortran_parser.environment import Environment
from scripts.fortran_parser.tokens import Token, TokenTypes


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

    def classify(self, env: Environment):
        return self


# Derived interface: Statement
class Statement(Node):
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
        return f"'{self.value}'"

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
        return str(self.expr) if self.expr else str(self.token)

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
        return f"{self.operator} ({str(self.right_expr)})"

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
        return str(self.left_expr) + f" {self.operator} " + str(self.right_expr)

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

    def classify(self, env: Environment):
        if self.operator == "=" and isinstance(self.left_expr, Identifier):
            var = env.variables.get(self.left_expr.value)
            if not var:
                raise SemanticError(f"Undeclared variable: {self.left_expr.value}")
            return AssignmentStatement(
                tok=self.token,
                lhs=self.left_expr,
                rhs=self.right_expr,
                env=env,
            )
        return self


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


class MacroIf(Statement):
    def __init__(self, token: Token, symbol: str, body: BlockStatement):
        self.token = token  # The #ifdef/#ifndef token
        self.symbol = symbol  # The macro symbol
        self.body = body

    def statement_node(self):
        pass

    def token_literal(self) -> str:
        return super().token_literal()

    def __str__(self) -> str:
        return f"{self.token} {self.symbol} {self.body}"


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
    def __init__(self, tok: Token, mod_name: Identifier, objs: list[Expression]):
        self.token = tok  # Should be 'Ident'
        self.module = mod_name.value
        self.objs = objs

    def token_literal(self) -> str:
        return self.token.literal

    def statement_node(self) -> None:
        return super().statement_node()

    def __str__(self) -> str:
        use_str = f", only : {','.join(list(map(str,self.objs)))}" if self.objs else ""
        return f"{self.lineno}: use {self.module}{use_str}"

    def to_dict(self):
        return {
            "Node": "UseStatement",
            "Module": self.module,
            "objs": [expr.to_dict() for expr in self.objs],
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
