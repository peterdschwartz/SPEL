import pytest

from spel.scripts.fortran_parser.spel_ast import (
    AssociateConstruct,
    IfConstruct,
    SubroutineDefinitionConstruct,
    UseStatement,
    VariableDecl,
)
from spel.scripts.fortran_parser.tests.test_Parser import parse_statements


def parse_single(txt: str) -> SubroutineDefinitionConstruct:
    program = parse_statements(txt)
    assert len(program.statements) == 1
    stmt = program.statements[0]
    assert isinstance(stmt, SubroutineDefinitionConstruct)
    return stmt


subroutine_txt = """
  subroutine CanopyTemperature(bounds, num_nolakec, filter_nolakec, &
       atm2lnd_inst, temperature_inst)
    use shr_const_mod, only : SHR_CONST_PI
    implicit none
    type(bounds_type), intent(in) :: bounds
    integer, intent(in) :: num_nolakec
    integer, intent(in) :: filter_nolakec(:)
    type(atm2lnd_type), intent(in) :: atm2lnd_inst
    type(temperature_type), intent(inout) :: temperature_inst
    integer :: fc, c

    associate( &
         forc_t => atm2lnd_inst%forc_t_downscaled_col, &
         t_grnd => temperature_inst%t_grnd_col &
         )
      do fc = 1, num_nolakec
         c = filter_nolakec(fc)
         if (forc_t(c) > tfrz) then
            t_grnd(c) = forc_t(c)
         end if
      end do
    end associate
  end subroutine CanopyTemperature
"""


def test_subroutine_definition():
    sub = parse_single(subroutine_txt)

    assert sub.name == "canopytemperature"
    assert not sub.is_function
    assert sub.prefixes == []
    assert sub.args == [
        "bounds",
        "num_nolakec",
        "filter_nolakec",
        "atm2lnd_inst",
        "temperature_inst",
    ]
    assert sub.return_type is None
    assert sub.result is None
    assert sub.contains == []
    assert sub.lineno == 1
    assert sub.end_ln == 23

    body = sub.body.statements
    assert isinstance(body[0], UseStatement)
    decls = [s for s in body if isinstance(s, VariableDecl)]
    assert len(decls) == 6
    # executable part is parsed into nested constructs, not flattened lines
    assert isinstance(body[-1], AssociateConstruct)


def test_subroutine_without_args():
    sub = parse_single(
        """
  subroutine init_constants
    implicit none
    pi = 3.14_r8
  end subroutine
"""
    )
    assert sub.name == "init_constants"
    assert sub.args == []

    sub = parse_single(
        """
  subroutine init_constants()
  end subroutine init_constants
"""
    )
    assert sub.args == []
    assert sub.body.statements == []


def test_function_with_typed_prefix_and_result():
    func = parse_single(
        """
  pure real(r8) function ft_interp(t, tbl) result(val)
    real(r8), intent(in) :: t
    real(r8), intent(in) :: tbl(:)
    val = tbl(1) + t
  end function ft_interp
"""
    )
    assert func.is_function
    assert func.name == "ft_interp"
    assert func.prefixes == ["pure"]
    assert func.args == ["t", "tbl"]
    assert str(func.return_type) == "real(r8)"
    assert func.result == "val"


def test_function_result_defaults_to_name():
    func = parse_single(
        """
  ELEMENTAL LOGICAL FUNCTION is_active(flag)
    integer, intent(in) :: flag
    is_active = flag > 0
  END FUNCTION is_active
"""
    )
    assert func.is_function
    assert func.prefixes == ["elemental"]
    assert str(func.return_type) == "logical"
    assert func.result == "is_active"


def test_function_untyped_prefix():
    # return type is declared in the body, not the prefix
    func = parse_single(
        """
  function get_dz(c, j) result(dz)
    integer, intent(in) :: c, j
    real(r8) :: dz
    dz = z(c,j+1) - z(c,j)
  end function
"""
    )
    assert func.is_function
    assert func.prefixes == []
    assert func.return_type is None
    assert func.result == "dz"
    assert func.args == ["c", "j"]


def test_internal_subprograms():
    sub = parse_single(
        """
  subroutine outer(x)
    real(r8), intent(inout) :: x
    if (x > 0._r8) then
       call inner(x)
    end if
  contains
    subroutine inner(y)
      real(r8), intent(inout) :: y
      y = helper(y)
    end subroutine inner

    pure function helper(z) result(w)
      real(r8), intent(in) :: z
      real(r8) :: w
      w = 2._r8 * z
    end function helper
  end subroutine outer
"""
    )
    assert sub.name == "outer"
    assert isinstance(sub.body.statements[-1], IfConstruct)

    assert [c.name for c in sub.contains] == ["inner", "helper"]
    inner, helper = sub.contains
    assert not inner.is_function
    assert helper.is_function
    assert helper.result == "w"
    assert sub.end_ln == 17


def test_consecutive_subprograms():
    program = parse_statements(
        """
  subroutine a(x)
    x = 1
  end subroutine a
  integer function b(y)
    integer, intent(in) :: y
    b = y
  end function b
"""
    )
    assert [type(s) for s in program.statements] == [SubroutineDefinitionConstruct] * 2
    a, b = program.statements
    assert (a.name, a.is_function, a.end_ln) == ("a", False, 3)
    assert (b.name, b.is_function, b.lineno, b.end_ln) == ("b", True, 4, 7)
    assert str(b.return_type) == "integer"


@pytest.mark.parametrize(
    "header, end, prefixes, return_type",
    [
        # prefix specs may appear in any order relative to the type-spec
        ("real(r8) elemental function f(a)", "end function f", ["elemental"], "real(r8)"),
        ("elemental real(r8) function f(a)", "end function f", ["elemental"], "real(r8)"),
        ("pure real(r8) elemental function f(a)", "end function", ["pure", "elemental"], "real(r8)"),
        ("real (r8) function f(a)", "endfunction f", [], "real(r8)"),
        ("pure type(foo_type) function f(a)", "end function f", ["pure"], "type(foo_type)"),
        ("character(len=256) function f(a)", "end function f", [], "character(len=256)"),
        ("recursive subroutine f(a)", "end subroutine f", ["recursive"], None),
        ("subroutine f(a)", "endsubroutine f", [], None),
    ],
)
def test_subprogram_header_forms(header, end, prefixes, return_type):
    sub = parse_single(f"\n  {header}\n    a = 1\n  {end}\n")
    assert sub.name == "f"
    assert sub.args == ["a"]
    assert sub.prefixes == prefixes
    if return_type is None:
        assert sub.return_type is None
    else:
        # return type must parse identically to the same type-spec in a declaration
        decl = parse_statements(f"{return_type} :: x").statements[0]
        assert isinstance(decl, VariableDecl)
        assert str(sub.return_type) == str(decl.var_type)
    assert sub.end_ln == 3


def test_continued_function_header():
    func = parse_single(
        """
  real(r8) function qsat(t, &
       p, eps) result(q)
    q = eps * t / p
  end function qsat
"""
    )
    assert func.args == ["t", "p", "eps"]
    assert func.result == "q"
    assert func.end_ln == 4


def test_prefix_named_variable_is_not_a_subprogram():
    program = parse_statements(
        """
  pure = 1
  elemental = pure + 1
"""
    )
    assert len(program.statements) == 2
    assert not any(isinstance(s, SubroutineDefinitionConstruct) for s in program.statements)


@pytest.mark.parametrize(
    "txt",
    [
        # missing END must fail loudly instead of swallowing the rest of the input
        "\n  real(r8) function f(a)\n  integer :: x\n  real(r8) :: y\n",
        "\n  subroutine outer(a)\n  contains\n    subroutine inner(b)\n    end subroutine inner\n",
        # END name must match the definition
        "\n  subroutine f(a)\n  end subroutine g\n",
        "\n  function f(a)\n  end subroutine f\n",
    ],
)
def test_malformed_subprogram_is_an_error(txt):
    with pytest.raises(SystemExit):
        parse_statements(txt)
