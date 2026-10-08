"""
Parsing of statements the scope walk needs on ELM code:
exit/cycle/return, select case, nullify, read, and function-like CPP macros.
"""

import pytest

from spel.scripts.fortran_parser.spel_ast import (
    CycleStatement,
    DoLoop,
    ExitStatement,
    ExpressionStatement,
    IfConstruct,
    MacroCallStatement,
    NullifyStatement,
    ReadStatement,
    ReturnStatement,
    SelectCaseConstruct,
    WhereConstruct,
)
from spel.scripts.fortran_parser.tests.test_Parser import parse_statements


def only_stmt(txt: str):
    program = parse_statements(txt)
    assert len(program.statements) == 1, [str(s) for s in program.statements]
    return program.statements[0]


@pytest.mark.parametrize(
    "txt,cls,name",
    [
        ("exit", ExitStatement, None),
        ("EXIT", ExitStatement, None),
        ("exit outer", ExitStatement, "outer"),
        ("cycle", CycleStatement, None),
        ("cycle outer", CycleStatement, "outer"),
        ("return", ReturnStatement, None),
    ],
)
def test_branch_statements(txt, cls, name):
    stmt = only_stmt(f"\n  {txt}\n")
    assert isinstance(stmt, cls) and stmt.lineno == 1
    if cls is not ReturnStatement:
        assert stmt.construct_name == name


@pytest.mark.parametrize("txt", ["exit = 1", "cycle = 2", "return = 3"])
def test_branch_keywords_as_variables(txt):
    stmt = only_stmt(f"\n  {txt}\n")
    assert isinstance(stmt, ExpressionStatement)


def test_branch_statements_in_blocks():
    txt = """
  do i = 1, n
     if (a(i) > 0) exit
     if (a(i) < 0) then
        cycle
     end if
     if (i == 3) return
  end do
"""
    loop = only_stmt(txt)
    assert isinstance(loop, DoLoop)
    s1, s2, s3 = loop.body.statements
    assert isinstance(s1.consequence.statements[0], ExitStatement)
    assert isinstance(s2.consequence.statements[0], CycleStatement)
    assert isinstance(s3, IfConstruct)
    assert isinstance(s3.consequence.statements[0], ReturnStatement)


@pytest.mark.parametrize("end", ["end do loop_columns", "enddo loop_columns", "end do"])
def test_named_do_loop(end):
    txt = f"""
  loop_columns: do fc = 1, num_snowc
     if (h2osno(fc) > 0) then
        exit loop_columns
     endif no_water

  {end}
"""
    loop = only_stmt(txt)
    assert isinstance(loop, DoLoop)
    (stmt,) = loop.body.statements
    assert isinstance(stmt, IfConstruct)
    assert isinstance(stmt.consequence.statements[0], ExitStatement)


SELECT = """
  select case (subgrid_level)
  case (BOUNDS_SUBGRID_GRIDCELL)  ! comment
     beg_index = bounds%begg
  case (1, 3:5, :0)
     beg_index = 2
     beg_index = 3
  case default
     beg_index = -1
  end select
"""


def test_select_case():
    stmt = only_stmt(SELECT)
    assert isinstance(stmt, SelectCaseConstruct)
    assert stmt.lineno == 1 and stmt.end_ln == 9
    assert str(stmt.selector) == "subgrid_level"
    assert [c.lineno for c in stmt.cases] == [2, 4, 7]
    assert [str(v) for v in stmt.cases[0].values] == ["bounds_subgrid_gridcell"]
    assert [str(v) for v in stmt.cases[1].values] == ["1", "3:5", ":0"]
    assert stmt.cases[2].values is None  # case default
    assert stmt.cases[2].is_default
    assert [len(c.body.statements) for c in stmt.cases] == [1, 2, 1]
    assert [s.lineno for s in stmt.cases[1].body.statements] == [5, 6]


def test_nested_select_and_endselect():
    txt = """
  select case (a)
  case (1)
     select case (b)
     case ('x')
        c = 1
     endselect
  case default
     c = 3
  end select
  c = 4
"""
    program = parse_statements(txt)
    assert len(program.statements) == 2
    outer = program.statements[0]
    assert isinstance(outer, SelectCaseConstruct) and outer.end_ln == 9
    inner = outer.cases[0].body.statements[0]
    assert isinstance(inner, SelectCaseConstruct)
    assert (inner.lineno, inner.end_ln) == (3, 6)
    assert program.statements[1].lineno == 10


def test_select_type_is_unsupported():
    with pytest.raises(SystemExit):
        parse_statements("\n  select type (x)\n  type is (foo)\n  end select\n")


def test_nullify():
    stmt = only_stmt("\n  nullify(p, this%q)\n")
    assert isinstance(stmt, NullifyStatement)
    assert [str(o) for o in stmt.objects] == ["p", "this%q"]


@pytest.mark.parametrize(
    "txt,controls,items",
    [
        ("read(10,*) w", ["10", "*"], ["w"]),
        ("read(unit, *) a, b(i)", ["unit", "*"], ["a", "b(i)"]),
        ("read(unitn, nml=elm_inparm, iostat=ierr)", ["unitn", "(nml=elm_inparm)", "(iostat=ierr)"], []),
    ],
)
def test_read_statement(txt, controls, items):
    stmt = only_stmt(f"\n  {txt}\n")
    assert isinstance(stmt, ReadStatement)
    assert [str(c) for c in stmt.controls] == controls
    assert [str(i) for i in stmt.items] == items


def test_function_like_macro_call():
    stmt = only_stmt(
        "\n  SHR_ASSERT((ltype >= 1 .and. ltype <= max_lunit), subname//': bad')\n"
    )
    assert isinstance(stmt, MacroCallStatement)
    assert stmt.name == "shr_assert" and len(stmt.args) == 2
    assert isinstance(only_stmt("\n  SHR_ASSERT_ALL_FL((x > 0), file, line)\n"), MacroCallStatement)


# ---------------------------------------------------------------------------
# stop / go to / labels / continue / intrinsic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "txt,error,code",
    [
        ("stop", False, None),
        ("STOP 1", False, "IntegerLiteral"),
        ("stop 'bad value'", False, "StringLiteral"),
        ("error stop", True, None),
        ("error stop 'x' // trim(msg)", True, "InfixExpression"),
    ],
)
def test_stop_statement(txt, error, code):
    from spel.scripts.fortran_parser.spel_ast import StopStatement

    stmt = only_stmt(f"\n  {txt}\n")
    assert isinstance(stmt, StopStatement) and stmt.lineno == 1
    assert stmt.error is error
    assert (stmt.code and type(stmt.code).__name__) == code


@pytest.mark.parametrize("txt", ["go to 80", "GO TO 80", "goto 80"])
def test_goto_statement(txt):
    from spel.scripts.fortran_parser.spel_ast import GotoStatement

    stmt = only_stmt(f"\n  {txt}\n")
    assert isinstance(stmt, GotoStatement) and stmt.target == 80


def test_labels_and_continue():
    from spel.scripts.fortran_parser.spel_ast import ContinueStatement, GotoStatement

    txt = """
  if (x > 0) go to 30
  x = 1
30  continue
80 x = 2
1000 format (1x,'nstep = ',i10,'   ts = ',f21.15)
"""
    stmts = parse_statements(txt).statements
    assert [type(s).__name__ for s in stmts] == [
        "IfConstruct",
        "ExpressionStatement",
        "ContinueStatement",
        "ExpressionStatement",
        "FormatStatement",
    ]
    assert isinstance(stmts[0].consequence.statements[0], GotoStatement)
    assert [s.label for s in stmts] == [None, None, 30, 80, 1000]
    assert [s.lineno for s in stmts] == [1, 2, 3, 4, 5]
    assert isinstance(stmts[2], ContinueStatement)
    assert stmts[4].spec == "1x,'nstep = ',i10,'   ts = ',f21.15"


@pytest.mark.parametrize(
    "txt,spec",
    [
        ("100 format(a)", "a"),
        ("100 FORMAT (a, 3(i5, 1x), /, 'x(1) = ', es12.4)", "a, 3(i5, 1x), /, 'x(1) = ', es12.4"),
        ("100 format ('it''s )', 2x, a)", "'it''s )', 2x, a"),
        ("100 format (1x, &\n  'a = ', f8.3)", "1x, 'a = ', f8.3"),
    ],
)
def test_format_statement(txt, spec):
    from spel.scripts.fortran_parser.spel_ast import FormatStatement

    stmts = parse_statements(f"\n{txt}\n  x = 1\n").statements
    assert [type(s).__name__ for s in stmts] == ["FormatStatement", "ExpressionStatement"]
    assert isinstance(stmts[0], FormatStatement)
    assert stmts[0].spec == spec and stmts[0].label == 100


@pytest.mark.parametrize("txt", ["format = 1", "format(i) = x", "10 format(i) = x"])
def test_format_as_variable(txt):
    stmt = only_stmt(f"\n{txt}\n")
    assert isinstance(stmt, ExpressionStatement)


@pytest.mark.parametrize(
    "txt,names",
    [
        ("intrinsic erf", ["erf"]),
        ("intrinsic :: erfc, gamma", ["erfc", "gamma"]),
        ("intrinsic erfc_scaled", ["erfc_scaled"]),
    ],
)
def test_intrinsic_statement(txt, names):
    from spel.scripts.fortran_parser.spel_ast import IntrinsicStatement

    stmt = only_stmt(f"\n  {txt}\n")
    assert isinstance(stmt, IntrinsicStatement) and stmt.names == names


@pytest.mark.parametrize("txt", ["stop = 1", "go = 2", "intrinsic = 3", "continue = 4"])
def test_new_keywords_as_variables(txt):
    stmt = only_stmt(f"\n  {txt}\n")
    assert isinstance(stmt, ExpressionStatement)


def test_where_statement():
    stmt = only_stmt("  where(tanpools<0) tanpools = 0.0_r8\n")
    assert isinstance(stmt, WhereConstruct)
    assert str(stmt.mask) == "(tanpools<0)"
    (assign,) = stmt.body.statements
    assert str(assign) == "(tanpools=0.0)"
    assert stmt.elsewheres == []


@pytest.mark.parametrize("end", ["end where", "endwhere"])
def test_where_construct(end):
    txt = f"""
  where (sum_wts == 0.0)
     a = 1.0
     b = 2.0
  elsewhere (sum_wts < 0.)
     a = -1.0
  elsewhere
     a = a / sum_wts
  {end}
"""
    stmt = only_stmt(txt)
    assert isinstance(stmt, WhereConstruct)
    assert len(stmt.body.statements) == 2
    masks = [str(ew.mask) if ew.mask else None for ew in stmt.elsewheres]
    assert masks == ["(sum_wts<0.0)", None]
    assert [len(ew.body.statements) for ew in stmt.elsewheres] == [1, 1]


def test_where_in_do_loop():
    txt = """
  do i = 1, n
     where (x(:,i) > 0) x(:,i) = 0.
     where (y > 0)
        y = 1.
     end where
  end do
"""
    loop = only_stmt(txt)
    assert [type(s) for s in loop.body.statements] == [WhereConstruct, WhereConstruct]


@pytest.mark.parametrize(
    "line,expected",
    [
        ("x = a .or. b", "(x=(a.or.b))"),
        (
            "included = (is_crop(c) .and. f_crop) &\n     .or. (is_soil(c) .and. f_veg)",
            "(included=((is_crop(c).and.f_crop).or.(is_soil(c).and.f_veg)))",
        ),
        ("x = a .and. b .or. c .and. d", "(x=((a.and.b).or.(c.and.d)))"),
        ("x = .not. a == b .and. c", "(x=((.not.(a==b)).and.c))"),
        ("y = a*b**c**d", "(y=(a*(b**(c**d))))"),
        ("s = a // b == c", "(s=((a//b)==c))"),
    ],
)
def test_operator_precedence(line, expected):
    program = parse_statements(line)
    assert [str(s) for s in program.statements] == [expected]
