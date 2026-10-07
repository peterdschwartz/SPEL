import pytest

from spel.scripts.fortran_parser.spel_ast import ImplicitNoneStatement
from spel.scripts.fortran_parser.tests.test_Parser import parse_statements


@pytest.mark.parametrize("txt", ["implicit none", "IMPLICIT NONE"])
def test_implicit_none(txt):
    program = parse_statements(f"\n  {txt}\n")
    assert len(program.statements) == 1
    stmt = program.statements[0]
    assert isinstance(stmt, ImplicitNoneStatement)
    assert stmt.lineno == 1


def test_implicit_as_variable_name_is_not_implicit_statement():
    program = parse_statements("\n  implicit = 1\n")
    assert len(program.statements) == 1
    assert not isinstance(program.statements[0], ImplicitNoneStatement)


@pytest.mark.parametrize(
    "txt",
    [
        "implicit real(r8) (a-h, o-z)",
        "implicit integer (i-n)",
        "implicit double precision (a-h)",
    ],
)
def test_implicit_typing_is_an_error(txt):
    # SPEL requires every symbol to be declared
    with pytest.raises(SystemExit):
        parse_statements(f"\n  {txt}\n")
