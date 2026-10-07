import pytest

from spel.scripts.fortran_parser.spel_ast import (
    AllocateStatement,
    SubroutineDefinitionConstruct,
)
from spel.scripts.fortran_parser.tests.test_Parser import parse_statements


def parse_single(txt: str) -> AllocateStatement:
    program = parse_statements(txt)
    assert len(program.statements) == 1, [str(s) for s in program.statements]
    stmt = program.statements[0]
    assert isinstance(stmt, AllocateStatement), type(stmt)
    return stmt


def as_strs(stmt: AllocateStatement):
    return [str(o) for o in stmt.objects], {k: str(v) for k, v in stmt.options.items()}


def test_allocate_simple():
    stmt = parse_single("\n  allocate(meg_cmp%emis_factors(numveg))\n")
    assert not stmt.is_deallocate
    assert stmt.type_spec is None
    assert as_strs(stmt) == (["meg_cmp%emis_factors(numveg)"], {})
    assert stmt.lineno == 1


def test_allocate_multiple_objects_with_stat():
    stmt = parse_single(
        "\n  allocate(this%a(begg:endg), this%b(begc:endc,1:nlev), stat=ier)\n"
    )
    assert as_strs(stmt) == (
        ["this%a(begg:endg)", "this%b(begc:endc,1:nlev)"],
        {"stat": "ier"},
    )


def test_allocate_source_and_errmsg():
    stmt = parse_single(
        "\n  allocate(spm, source=create_spm_type(), stat=ier, errmsg=msg)\n"
    )
    assert as_strs(stmt) == (
        ["spm"],
        {"source": "create_spm_type()", "stat": "ier", "errmsg": "msg"},
    )


def test_allocate_typed_character():
    stmt = parse_single("\n  allocate(character(pftname_len) :: pftname(0:mxpft))\n")
    assert stmt.type_spec is not None
    assert stmt.type_spec.token.literal == "character"
    assert str(stmt.type_spec.len_) == "pftname_len"
    assert as_strs(stmt) == (["pftname(0:mxpft)"], {})
    assert str(stmt) == "allocate(character(pftname_len) :: pftname(0:mxpft))"


def test_allocate_typed_real_kind():
    stmt = parse_single("\n  allocate(real(r8) :: buf(n), work(n,m))\n")
    assert stmt.type_spec is not None
    assert stmt.type_spec.token.literal == "real"
    assert stmt.type_spec.kind == "r8"
    assert as_strs(stmt) == (["buf(n)", "work(n,m)"], {})


def test_allocate_typed_derived_type():
    stmt = parse_single(
        "\n  allocate(fates_fire_no_data_type :: fates_fire_data_method)\n"
    )
    assert stmt.type_spec is not None
    assert stmt.type_spec.token.literal == "fates_fire_no_data_type"
    assert as_strs(stmt) == (["fates_fire_data_method"], {})


def test_allocate_continued_line():
    stmt = parse_single("\n  allocate(a(n + &\n     m), &\n     b(n))\n")
    one_line = parse_single("\n  allocate(a(n + m), b(n))\n")
    assert stmt == one_line
    assert len(stmt.objects) == 2 and str(stmt.objects[1]) == "b(n)"


def test_deallocate():
    stmt = parse_single("\n  deallocate(domain%lonv, domain%latv, stat=ier)\n")
    assert stmt.is_deallocate
    assert stmt.type_spec is None
    assert as_strs(stmt) == (["domain%lonv", "domain%latv"], {"stat": "ier"})
    assert str(stmt) == "deallocate(domain%lonv, domain%latv, stat=ier)"


def test_allocate_inside_subroutine_body():
    program = parse_statements(
        """
  subroutine pftconrd()
    character(len=40), allocatable :: pftname(:)
    if (.not. allocated(pftname)) then
      allocate(character(pftname_len) :: pftname(0:mxpft))
    end if
    deallocate(pftname)
  end subroutine pftconrd
"""
    )
    sub = program.statements[0]
    assert isinstance(sub, SubroutineDefinitionConstruct)
    allocs = [s for s in sub.body.statements if isinstance(s, AllocateStatement)]
    assert len(allocs) == 1 and allocs[0].is_deallocate
    assert allocs[0].lineno == 6
    if_stmt = sub.body.statements[1]
    inner = if_stmt.consequence.statements[0]
    assert isinstance(inner, AllocateStatement)
    assert inner.lineno == 4


@pytest.mark.parametrize(
    "txt",
    [
        pytest.param("\n  allocate()\n", id="empty"),
        pytest.param("\n  allocate(stat=ier)\n", id="no-objects"),
        pytest.param("\n  allocate(a(n), stat=ier, b(n))\n", id="object-after-option"),
        pytest.param("\n  allocate(a(n), stat=ier, stat=j)\n", id="duplicate-option"),
        pytest.param("\n  allocate(a(n), foo=1)\n", id="unknown-option"),
        pytest.param("\n  deallocate(a, source=b)\n", id="dealloc-source"),
        pytest.param("\n  deallocate(real(r8) :: a)\n", id="dealloc-typed"),
    ],
)
def test_allocate_malformed(txt):
    with pytest.raises(SystemExit):
        parse_statements(txt)
