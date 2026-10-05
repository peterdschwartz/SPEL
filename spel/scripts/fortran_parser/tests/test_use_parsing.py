import pytest

from spel.scripts.fortran_parser.spel_ast import (
    SubroutineDefinitionConstruct,
    UseStatement,
)
from spel.scripts.fortran_parser.tests.test_Parser import parse_statements


def parse_single(txt: str) -> UseStatement:
    program = parse_statements(txt)
    assert len(program.statements) == 1, [str(s) for s in program.statements]
    stmt = program.statements[0]
    assert isinstance(stmt, UseStatement), type(stmt)
    return stmt


def strs(exprs):
    return [str(e) for e in exprs]


def test_use_plain():
    stmt = parse_single("\n  use shr_kind_mod\n")
    assert stmt.module == "shr_kind_mod"
    assert stmt.nature is None
    assert not stmt.has_only
    assert stmt.objs == [] and stmt.renames == []


def test_use_only_clause_unchanged():
    stmt = parse_single("\n  use mod3, only: x => y, zs_jf, assignment(=)\n")
    assert stmt.module == "mod3"
    assert stmt.has_only
    assert strs(stmt.objs) == ["(x=>y)", "zs_jf", "assignment(=)"]
    assert stmt.renames == []


def test_use_intrinsic():
    stmt = parse_single("\n  use, intrinsic :: ieee_exceptions\n")
    assert stmt.module == "ieee_exceptions"
    assert stmt.nature == "intrinsic"
    assert not stmt.has_only
    assert stmt.objs == []


def test_use_intrinsic_with_only():
    stmt = parse_single(
        "\n  use, intrinsic :: iso_fortran_env, only: output_unit, error_unit\n"
    )
    assert stmt.module == "iso_fortran_env"
    assert stmt.nature == "intrinsic"
    assert stmt.has_only
    assert strs(stmt.objs) == ["output_unit", "error_unit"]


def test_use_non_intrinsic():
    stmt = parse_single("\n  use, non_intrinsic :: my_mod, only: a\n")
    assert stmt.module == "my_mod"
    assert stmt.nature == "non_intrinsic"
    assert strs(stmt.objs) == ["a"]


def test_use_double_colon_without_nature():
    stmt = parse_single("\n  use :: shr_kind_mod, only: r8 => shr_kind_r8\n")
    assert stmt.module == "shr_kind_mod"
    assert stmt.nature is None
    assert strs(stmt.objs) == ["(r8=>shr_kind_r8)"]


def test_use_rename_list_without_only():
    stmt = parse_single("\n  use mod1, a => b, c => d\n")
    assert stmt.module == "mod1"
    assert not stmt.has_only
    # objs is reserved for the only-list; module is still fully used
    assert stmt.objs == []
    assert strs(stmt.renames) == ["(a=>b)", "(c=>d)"]


def test_use_empty_only_list():
    stmt = parse_single("\n  use mod1, only:\n")
    assert stmt.has_only
    assert stmt.objs == []


def test_use_str_roundtrip():
    stmt = parse_single(
        "\n  use, intrinsic :: iso_fortran_env, only: output_unit\n"
    )
    assert str(stmt).split(": ", 1)[1] == (
        "use, intrinsic :: iso_fortran_env, only : output_unit"
    )


def test_use_intrinsic_in_subroutine_body():
    program = parse_statements(
        """
  subroutine initvertical(bounds)
    use, intrinsic :: ieee_exceptions
    use shr_kind_mod, only: r8 => shr_kind_r8
    type(bounds_type), intent(in) :: bounds
    call ieee_set_halting_mode(ieee_divide_by_zero, .false.)
  end subroutine initvertical
"""
    )
    sub = program.statements[0]
    assert isinstance(sub, SubroutineDefinitionConstruct)
    uses = [s for s in sub.body.statements if isinstance(s, UseStatement)]
    assert [(u.module, u.nature, u.lineno) for u in uses] == [
        ("ieee_exceptions", "intrinsic", 2),
        ("shr_kind_mod", None, 3),
    ]


@pytest.mark.parametrize(
    "txt",
    [
        pytest.param("\n  use, foo :: m\n", id="bad-nature"),
        pytest.param("\n  use, intrinsic m\n", id="nature-missing-double-colon"),
        pytest.param("\n  use, :: m\n", id="comma-without-nature"),
        pytest.param("\n  use m, only x\n", id="only-missing-colon"),
        pytest.param("\n  use m, a\n", id="rename-not-arrow"),
        pytest.param("\n  use\n", id="no-module"),
    ],
)
def test_use_malformed(txt):
    with pytest.raises(SystemExit):
        parse_statements(txt)
