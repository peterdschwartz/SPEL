import shutil

import pytest

from spel.scripts.fortran_parser.module_spec import module_spec

pytestmark = pytest.mark.skipif(shutil.which("gfortran") is None, reason="module_spec preprocesses with gfortran")

SRC = """\
module snippet_mod
  implicit none
  private
  public :: pub_t, inst, rootprof, inhemi, days

  type, private :: priv_t
     real, pointer :: p => null()
  end type priv_t

  type pub_t
     real, allocatable :: a(:)
     real, pointer :: s => null()
     real :: x(3)
     integer, private :: hidden
  end type pub_t

  type(pub_t) :: inst
  !$acc declare create(inst)
  real, allocatable :: rootprof(:) ! rooting distribution parameter for C and N inputs
  integer, pointer :: inhemi(:)
  !$acc declare create(inhemi)
  integer, dimension(12) :: days = (/ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 /)
  real :: fracday
  real, parameter :: secs = 3600.
contains
  subroutine build(n)
    integer, intent(in) :: n
    integer :: i
    !$acc parallel loop
    do i = 1, n
    end do
  end subroutine build
  subroutine host_only()
  end subroutine host_only
end module snippet_mod
"""


def test_module_spec_from_the_parser(tmp_path):
    path = tmp_path / "snippet_mod.F90"
    path.write_text(SRC)
    spec = module_spec(path)
    # attributes come from the declaration, not from a comment that says "parameter"
    assert "allocatable" in spec.variables["rootprof"].attrs
    assert spec.storage_test("rootprof") == "allocated"
    assert "parameter" in spec.variables["secs"].attrs
    # a declared module pointer is still a pointer; a fixed-size array with an initializer has no storage test
    assert spec.storage_test("inhemi") == "associated" and "inhemi" in spec.declared
    assert spec.storage_test("days") is None and spec.variables["days"].rank == 1
    # accessibility: default private, public list, type attributes, component attributes
    assert spec.variables["fracday"].access == "private"
    assert spec.variables["inst"].access == "public" and "inst" in spec.declared
    assert spec.types["priv_t"].access == "private"
    pub = spec.types["pub_t"]
    assert pub.access == "public" and pub.components["hidden"] == "private"
    assert pub.storage_test("a") == "allocated" and pub.storage_test("s") == "associated"
    assert pub.storage_test("x") is None
    # compute constructs, per routine
    assert spec.compute_constructs and spec.compute_routines == {"build"}


def test_procedure_name_lists():
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.scripts.fortran_parser.spel_parser import Parser
    from spel.scripts.utilityFunctions import LineTuple

    src = """module m
  implicit none
  interface norm
     module procedure norm_a, norm_b
  end interface norm
  real :: x
contains
  subroutine norm_a()
  end subroutine norm_a
  subroutine norm_b()
  end subroutine norm_b
end module m""".split("\n")
    mod = Parser(lines=[LineTuple(line=l, ln=i) for i, l in enumerate(src)]).parse_program().statements[0]
    procs = [s for s in mod.spec.statements if isinstance(s, A.ProcedureStatement)]
    assert procs[0].name == "norm_a" and procs[0].names == ["norm_a", "norm_b"]
    assert [s for s in mod.spec.statements if isinstance(s, A.VariableDecl)]  # parsing continues after the list
    assert [p.name.lower() for p in mod.subprograms] == ["norm_a", "norm_b"]
