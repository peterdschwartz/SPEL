"""
DATA statements: data obj-list /value-list/ [[,] obj-list /value-list/]...
`data` is not a reserved word, so assignments/keyword args named `data` must still parse.
"""

import pytest

from spel.scripts.fortran_parser.spel_ast import (
    DataImpliedDo,
    DataStatement,
    ExpressionStatement,
    FuncExpression,
    Identifier,
    PrefixExpression,
    SubCallStatement,
)
from spel.scripts.fortran_parser.tests.test_control_statements import only_stmt


def test_data_scalar():
    stmt = only_stmt("\n  data x /1._r8/\n")
    assert isinstance(stmt, DataStatement)
    assert len(stmt.sets) == 1
    (objects, values) = stmt.sets[0].objects, stmt.sets[0].values
    assert objects == [Identifier(None, "x")]
    assert len(values) == 1 and values[0].repeat is None
    assert stmt.lineno == 1


def test_data_repeat_and_signed_values():
    stmt = only_stmt("\n  data a /3*0._r8, -1._r8, nmax*2/\n")
    values = stmt.sets[0].values
    assert len(values) == 3
    assert str(values[0].repeat) == "3"
    assert values[1].repeat is None and isinstance(values[1].value, PrefixExpression)
    assert values[2].repeat == Identifier(None, "nmax")


def test_data_array_section_with_continuation():
    txt = """
  data ri  (1,1:NLUse) &
       /1.e36_r8, 60._r8, 120._r8/
"""
    stmt = only_stmt(txt)
    obj = stmt.sets[0].objects[0]
    assert isinstance(obj, FuncExpression) and str(obj.function) == "ri"
    assert len(stmt.sets[0].values) == 3


def test_data_implied_do():
    stmt = only_stmt("\n  data (s_con(1,i),i=1,4) /1898_r8, -110.1_r8, 2.834_r8, -0.02791_r8/ ! CH4\n")
    obj = stmt.sets[0].objects[0]
    assert isinstance(obj, DataImpliedDo)
    assert [str(o.function) for o in obj.objects] == ["s_con"]
    assert obj.loop.index.literal == "i"
    assert str(obj.loop.start_expr) == "1" and str(obj.loop.end_expr) == "4"
    assert len(stmt.sets[0].values) == 4


def test_data_nested_implied_do():
    stmt = only_stmt("\n  data ((a(i,j),i=1,2),j=1,3) /6*0/\n")
    outer = stmt.sets[0].objects[0]
    assert isinstance(outer, DataImpliedDo) and outer.loop.index.literal == "j"
    inner = outer.objects[0]
    assert isinstance(inner, DataImpliedDo) and inner.loop.index.literal == "i"


@pytest.mark.parametrize(
    "txt", ["data x /1/, y /2/", "data x /1/ y /2/", "data x, y /1, 2/ z /3/"]
)
def test_data_multiple_sets(txt):
    stmt = only_stmt(f"\n  {txt}\n")
    names = [[str(o) for o in s.objects] for s in stmt.sets]
    assert names[-1] == (["z"] if "z" in txt else ["y"])
    assert sum(len(s.values) for s in stmt.sets) == (3 if "z" in txt else 2)


@pytest.mark.parametrize(
    "txt",
    ["data = 1", "data(i) = x", "data (i) = x", "data%x = 2"],
)
def test_data_as_variable(txt):
    stmt = only_stmt(f"\n  {txt}\n")
    assert isinstance(stmt, ExpressionStatement)


def test_data_as_keyword_argument():
    stmt = only_stmt("\n  call foo(data=y)\n")
    assert isinstance(stmt, SubCallStatement)
