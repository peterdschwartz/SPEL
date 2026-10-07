import pytest

from spel.scripts.fortran_parser.spel_ast import (
    ImportStatement,
    SubroutineDefinitionConstruct,
)
from spel.scripts.fortran_parser.tests.test_Parser import parse_statements


def parse_single(txt: str) -> ImportStatement:
    program = parse_statements(txt)
    assert len(program.statements) == 1, [str(s) for s in program.statements]
    stmt = program.statements[0]
    assert isinstance(stmt, ImportStatement), type(stmt)
    return stmt


@pytest.mark.parametrize(
    "txt,names",
    [
        ("import :: soil_water_retention_curve_type", ["soil_water_retention_curve_type"]),
        ("import :: r8, rk", ["r8", "rk"]),
        ("import r8", ["r8"]),
        ("import r8, fire_method_type", ["r8", "fire_method_type"]),
        ("IMPORT :: R8", ["r8"]),
    ],
)
def test_import_names(txt, names):
    stmt = parse_single(f"\n  {txt}\n")
    assert stmt.spec is None
    assert stmt.names == names


def test_bare_import():
    stmt = parse_single("\n  import\n")
    assert stmt.spec is None
    assert stmt.names == []


def test_import_only():
    stmt = parse_single("\n  import, only: a, b\n")
    assert stmt.spec == "only"
    assert stmt.names == ["a", "b"]


@pytest.mark.parametrize("spec", ["none", "all"])
def test_import_none_all(spec):
    stmt = parse_single(f"\n  import, {spec}\n")
    assert stmt.spec == spec
    assert stmt.names == []


def test_import_with_continuation():
    stmt = parse_single("\n  import :: a, &\n     b\n")
    assert stmt.names == ["a", "b"]
    assert stmt.lineno == 1


def test_import_in_interface_body_subroutine():
    txt = """
    subroutine soil_hk_interface(this, smp, hk)
      import :: soil_water_retention_curve_type
      import :: r8
      class(soil_water_retention_curve_type), intent(in) :: this
      real(r8), intent(in) :: smp
      real(r8), intent(out) :: hk
    end subroutine soil_hk_interface
    """
    program = parse_statements(txt)
    assert len(program.statements) == 1
    sub = program.statements[0]
    assert isinstance(sub, SubroutineDefinitionConstruct)
    imports = [s for s in sub.body.statements if isinstance(s, ImportStatement)]
    assert [s.names for s in imports] == [["soil_water_retention_curve_type"], ["r8"]]


def test_import_as_variable_name_is_not_import_statement():
    program = parse_statements("\n  import = 1\n")
    assert len(program.statements) == 1
    assert not isinstance(program.statements[0], ImportStatement)


@pytest.mark.parametrize(
    "txt",
    [
        "import ::",
        "import :: a,",
        "import, only a",
        "import, bogus",
        "import, none :: a",
    ],
)
def test_malformed_import_is_an_error(txt):
    with pytest.raises(SystemExit):
        parse_statements(f"\n  {txt}\n")
