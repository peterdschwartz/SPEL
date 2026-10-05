"""
classify(scope) refines syntactically ambiguous nodes using the symbol table.
"""

import pytest

from spel.scripts.fortran_parser.spel_ast import (
    ArrayRef,
    AssignmentStatement,
    FunctionCall,
    KeywordArgument,
    PointerAssignment,
    SemanticError,
    SubCallStatement,
)
from spel.scripts.fortran_parser.symbols import (
    DictResolver,
    Origin,
    Scope,
    Symbol,
    SymbolKind,
)
from spel.scripts.fortran_parser.tests.test_Parser import parse_statements


@pytest.fixture
def scope() -> Scope:
    resolver = DictResolver(
        globals={
            "myfunc": Symbol("myfunc", SymbolKind.PROCEDURE, Origin.GLOBAL),
            "garr": Symbol("garr", SymbolKind.VARIABLE, Origin.GLOBAL),
            "r8": Symbol("r8", SymbolKind.VARIABLE, Origin.GLOBAL),
        }
    )
    sc = Scope(kind="routine", resolver=resolver)
    for name in ("a", "x", "i", "p", "t", "this"):
        sc.define(Symbol(name, SymbolKind.VARIABLE, Origin.LOCAL))
    sc.define(Symbol("c", SymbolKind.ALIAS, Origin.LOCAL, target="this%arr"))
    return sc


def expr_of(txt: str):
    # rhs of an assignment: a leading type keyword (real(...)) would start a declaration
    return parse_statements(f"\n zz = {txt}\n").statements[0].expression.right_expr


@pytest.mark.parametrize("txt", ["a(i)", "garr(1, :)", "this%arr(i)", "c(i)"])
def test_array_reference(scope, txt):
    node = expr_of(txt).classify(scope)
    assert isinstance(node, ArrayRef)
    assert str(node) == str(expr_of(txt))


@pytest.mark.parametrize(
    "txt,name,intrinsic",
    [
        ("myfunc(a)", "myfunc", False),
        ("max(a, i)", "max", True),
        ("real(i, r8)", "real", True),
        ("size(a)", "size", True),
    ],
)
def test_function_call(scope, txt, name, intrinsic):
    node = expr_of(txt).classify(scope)
    assert isinstance(node, FunctionCall)
    assert node.name == name
    assert node.intrinsic is intrinsic


def test_declared_variable_shadows_intrinsic(scope):
    scope.define(Symbol("index", SymbolKind.VARIABLE, Origin.LOCAL))
    assert isinstance(expr_of("index(i)").classify(scope), ArrayRef)


def test_keyword_arguments(scope):
    node = expr_of("myfunc(a, k=2)").classify(scope)
    assert isinstance(node, FunctionCall)
    assert not isinstance(node.args[0], KeywordArgument)
    kw = node.args[1]
    assert isinstance(kw, KeywordArgument)
    assert kw.keyword == "k" and str(kw.value) == "2"


def test_unresolved_function_or_array_is_an_error(scope):
    with pytest.raises(SemanticError):
        expr_of("undeclared(i)").classify(scope)


def test_assignment_statement(scope):
    stmt = parse_statements("\n a(i) = x + 1\n").statements[0].classify(scope)
    assert isinstance(stmt, AssignmentStatement)
    assert str(stmt.left) == "a(i)" and str(stmt.right) == "(x+1)"
    assert stmt.lineno == 1


def test_pointer_assignment(scope):
    stmt = parse_statements("\n p => this%arr\n").statements[0].classify(scope)
    assert isinstance(stmt, PointerAssignment)
    assert str(stmt.pointer) == "p" and str(stmt.target) == "this%arr"


@pytest.mark.parametrize("txt", ["x", "x + 1", "myfunc(a)"])
def test_bare_expression_is_not_a_statement(scope, txt):
    with pytest.raises(SemanticError):
        parse_statements(f"\n {txt}\n").statements[0].classify(scope)


def test_subroutine_call_keyword_arguments(scope):
    stmt = parse_statements("\n call foo(a, n=i)\n").statements[0]
    assert isinstance(stmt, SubCallStatement)
    stmt = stmt.classify(scope)
    assert isinstance(stmt, SubCallStatement)
    assert isinstance(stmt.function.args[1], KeywordArgument)
    assert stmt.function.args[1].keyword == "n"
