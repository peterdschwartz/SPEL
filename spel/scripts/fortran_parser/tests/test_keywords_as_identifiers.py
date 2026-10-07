"""
Fortran has no reserved words: keywords used as variable names must parse as
identifiers (at statement start and inside expressions) without breaking the
statements those keywords introduce.
"""

import pytest

from spel.scripts.fortran_parser.spel_ast import (
    ExpressionStatement,
    IfConstruct,
    SubCallStatement,
)
from spel.scripts.fortran_parser.tests.test_control_statements import only_stmt
from spel.scripts.fortran_parser.tests.test_Parser import parse_statements


@pytest.mark.parametrize(
    "txt",
    [
        "type = 1",
        "use = 2",
        "procedure = x",
        "namelist(1) = 2",
        "call = 3",
        "write = 1",
        "print = 1",
        "associate = 1",
        "allocate(2) = 1",
        "deallocate%x = 1",
        "contains = 1",
        "do = 1",
        "if = 1",
        "function = 2",
        "subroutine = 1",
        "type(1) = 3",
        "use(i)%x = 2",
        "use => tgt",
    ],
)
def test_keyword_assigned_at_statement_start(txt):
    stmt = only_stmt(f"\n  {txt}\n")
    assert isinstance(stmt, ExpressionStatement), str(stmt)
    assert str(stmt.expression.left_expr).split("(")[0].split("%")[0] == txt.split(
        "("
    )[0].split("%")[0].split()[0]


@pytest.mark.parametrize(
    "txt,expected",
    [
        ("x = use(2)", "(x=use(2))"),
        ("y = do * if", "(y=(do*if))"),
        ("x = a%use(1)%procedure", None),
        ("x = type + call", "(x=(type+call))"),
    ],
)
def test_keyword_in_expression(txt, expected):
    stmt = only_stmt(f"\n  {txt}\n")
    assert isinstance(stmt, ExpressionStatement)
    if expected is not None:
        assert str(stmt) == expected


def test_keyword_as_actual_and_keyword_argument():
    stmt = only_stmt("\n  call foo(use, write=print)\n")
    assert isinstance(stmt, SubCallStatement)
    assert [str(a) for a in stmt.function.args] == ["use", "(write=print)"]


def test_keyword_after_semicolon_label_and_inline_if():
    stmts = parse_statements("\n  x = 1; use = 2\n").statements
    assert [type(s).__name__ for s in stmts] == ["ExpressionStatement"] * 2

    stmt = only_stmt("\n10 use = 2\n")
    assert isinstance(stmt, ExpressionStatement) and stmt.label == 10

    stmt = only_stmt("\n  if (c) use = 1\n")
    assert isinstance(stmt, IfConstruct)
    assert isinstance(stmt.consequence.statements[0], ExpressionStatement)


def test_block_terminator_keywords_as_variables():
    txt = """
  if (c) then
     else = 1
     contains = 2
  else
     endif = 3
  end if
"""
    stmt = only_stmt(txt)
    assert isinstance(stmt, IfConstruct)
    assert [str(s) for s in stmt.consequence.statements] == ["(else=1)", "(contains=2)"]
    assert [str(s) for s in stmt.else_.alternative.statements] == ["(endif=3)"]


@pytest.mark.parametrize(
    "txt,cls",
    [
        ("type(foo_type), pointer :: x", "VariableDecl"),
        ("type(foo_type) :: x", "VariableDecl"),
        ("use elm_varctl, only: iulog", "UseStatement"),
        ("call foo(x)", "SubCallStatement"),
        ("write(iulog,*) 'a', x", "WriteStatement"),
        ("print *, x", "PrintStatement"),
        ("allocate(x(n))", "AllocateStatement"),
        ("if (a) b(1) = 2", "IfConstruct"),
    ],
)
def test_keyword_statements_unchanged(txt, cls):
    stmt = only_stmt(f"\n  {txt}\n")
    assert type(stmt).__name__ == cls
