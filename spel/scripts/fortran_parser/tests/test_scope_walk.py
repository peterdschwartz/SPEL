"""
walk_subroutine: single in-order, scope-aware walk of a SubroutineDefinitionConstruct.
The SubroutineRecord it returns is the source of truth for routine data.

Line numbers: the leading newline in each snippet makes the header line 1.
"""

import pytest

from spel.scripts.fortran_parser.scope_walk import SubroutineRecord, walk_subroutine
from spel.scripts.fortran_parser.spel_ast import SemanticError
from spel.scripts.fortran_parser.symbols import DictResolver, Origin, Symbol, SymbolKind
from spel.scripts.fortran_parser.tests.test_Parser import parse_statements


def gvar(name: str) -> Symbol:
    return Symbol(name, SymbolKind.VARIABLE, Origin.GLOBAL)


def proc(name: str) -> Symbol:
    return Symbol(name, SymbolKind.PROCEDURE, Origin.GLOBAL)


def walk(txt: str, globals=(), modules=None, host=None, tree_index=None):
    tree = parse_statements(txt).statements[0]
    if tree_index is not None:
        tree = tree.contains[tree_index]
    resolver = DictResolver(
        globals={s.name: s for s in globals},
        modules=modules or {},
    )
    return walk_subroutine(tree, resolver=resolver, host=host)


def acc(rec: SubroutineRecord, ln=None):
    return [
        (a.status, a.path, a.ln)
        for a in rec.accesses()
        if ln is None or a.ln == ln
    ]


DECLS = """
subroutine s(this, bounds, n, res)
  use elm_varctl, only: iulog, use_cn => use_crop
  implicit none
  class(foo_type), intent(inout) :: this
  type(bounds_type), intent(in) :: bounds
  integer, intent(in) :: n
  real(r8), intent(out) :: res(bounds%begc:bounds%endc)
  integer :: i, j
  real(r8), parameter :: c0 = 0._r8
  real(r8) :: work(n)
  integer :: k = 1
end subroutine s
"""


def test_declarations():
    rec = walk(DECLS, modules={"elm_varctl": {"iulog": gvar("iulog"), "use_crop": gvar("use_crop")}})
    assert [d.name for d in rec.dummy_args] == ["this", "bounds", "n", "res"]
    assert [d.intent for d in rec.dummy_args] == ["inout", "in", "in", "out"]
    assert set(rec.local_variables) == {"i", "j", "c0", "work", "k"}
    assert rec.declaration("res").dim == 1
    assert rec.declaration("c0").ln == 9 and "parameter" in rec.declaration("c0").attrs
    assert rec.last_decl_ln == 11
    assert rec.result is None
    # use-associated names (including renames) resolve as globals
    assert rec.lookup("iulog").origin is Origin.GLOBAL
    assert rec.lookup("use_cn").origin is Origin.GLOBAL
    # specification expressions are reads; an initializer defines the entity,
    # except for named constants (parameters are read-only)
    assert acc(rec) == [
        ("r", "bounds%begc", 7),
        ("r", "bounds%endc", 7),
        ("r", "n", 10),
        ("w", "k", 11),
    ]
    assert rec.accesses(path="k")[0].context == "init"


DATA = """
subroutine s(n)
  use consts, only: gk
  implicit none
  integer, intent(in) :: n
  integer, parameter :: m = 4
  real(r8) :: a(4), b(2,4)
  integer :: k
  data a /4*0._r8/
  data (b(1,j),j=1,m) /1._r8, 2._r8, 3._r8, 4._r8/, k /gk/
  k = k + n
end subroutine s
"""


def test_data_statement():
    rec = walk(DATA, modules={"consts": {"gk": gvar("gk")}})
    # data statements belong to the specification part
    assert rec.last_decl_ln == 9
    # objects are written; implied-do bounds and values are read.
    # The implied-do index `j` is statement-scoped (no declaration needed).
    assert [
        x
        for x in acc(rec)
        if x[1] != "j"
    ] == [
        ("w", "a", 8),
        ("r", "m", 9),
        ("w", "b", 9),
        ("w", "k", 9),
        ("r", "gk", 9),
        ("r", "k", 10),
        ("r", "n", 10),
        ("w", "k", 10),
    ]
    assert {a.context for a in rec.accesses(status="w", ln=9)} == {"data"}
    assert not rec.accesses(path="j", origin=Origin.LOCAL)


def test_format_statement_is_skipped():
    txt = """
subroutine s(x)
  use elm_varctl, only: iulog
  implicit none
  real(r8), intent(in) :: x
  write(iulog,1000) x
1000 format (1x,'x = ',f21.15)
end subroutine s
"""
    rec = walk(txt, modules={"elm_varctl": {"iulog": gvar("iulog")}})
    assert acc(rec) == [("r", "iulog", 5), ("r", "x", 5)]
    assert rec.last_decl_ln == 4


def test_macro_branches_are_all_walked():
    txt = """
subroutine s(x, y)
  implicit none
  real(r8), intent(in) :: x
  real(r8), intent(out) :: y
#if (defined _openmp)
  y = x
#elif defined(other)
  y = 2*x
#else
  y = 0._r8
#endif
end subroutine s
"""
    rec = walk(txt)
    assert acc(rec) == [
        ("r", "x", 6),
        ("w", "y", 6),
        ("r", "x", 8),
        ("w", "y", 8),
        ("w", "y", 10),
    ]


def test_dummy_argument_without_declaration_is_an_error():
    txt = """
subroutine s(a, b)
  implicit none
  real(r8), intent(in) :: a
end subroutine s
"""
    with pytest.raises(SemanticError, match="b"):
        walk(txt)


def test_undeclared_symbol_is_an_error():
    txt = """
subroutine s(a)
  implicit none
  real(r8), intent(inout) :: a
  a = undeclared_thing + 1
end subroutine s
"""
    with pytest.raises(SemanticError, match="undeclared_thing"):
        walk(txt)


LOOP = """
subroutine s(a, b, n)
  implicit none
  integer, intent(in) :: n
  real(r8), intent(inout) :: a(:)
  real(r8), intent(in) :: b(:)
  integer :: i
  do i = 1, n
     a(i) = a(i) + b(i)
  end do
  if (a(1) > b(1)) then
     a(1) = 0.
  else if (n > 2) then
     a(2) = b(2)
  else
     a(3) = 1.
  end if
end subroutine s
"""


def test_source_order_and_statuses():
    rec = walk(LOOP)
    # subscripts are read before the designator they index;
    # the rhs is evaluated before the lhs is written
    assert acc(rec, 7) == [("r", "n", 7), ("w", "i", 7)]
    assert acc(rec, 8) == [
        ("r", "i", 8),
        ("r", "a", 8),
        ("r", "i", 8),
        ("r", "b", 8),
        ("r", "i", 8),
        ("w", "a", 8),
    ]
    # every branch condition is read, in order
    assert acc(rec, 10) == [("r", "a", 10), ("r", "b", 10)]
    assert acc(rec, 11) == [("w", "a", 11)]
    assert acc(rec, 12) == [("r", "n", 12)]
    assert acc(rec, 13) == [("r", "b", 13), ("w", "a", 13)]
    assert acc(rec, 15) == [("w", "a", 15)]
    by_path = rec.access_by_path(origin=Origin.DUMMY)
    assert set(by_path) == {"a", "b", "n"}
    assert [(a.status, a.ln) for a in by_path["n"]] == [("r", 7), ("r", 12)]
    assert rec.accesses(path="i")[1].origin is Origin.LOCAL


DTYPES = """
subroutine s(filter, i_type)
  implicit none
  type(clump_filter), intent(inout) :: filter(:)
  integer, intent(in) :: i_type
  filter(i_type)%num_soilc = 10
  filter(i_type)%soilc(:) = 4
  x%y(i_type)%z(1) = filter(1)%num_soilc
end subroutine s
"""


def test_derived_type_paths_drop_subscripts():
    rec = walk(DTYPES, globals=[gvar("x")])
    assert acc(rec, 5) == [("r", "i_type", 5), ("w", "filter%num_soilc", 5)]
    assert acc(rec, 6) == [("r", "i_type", 6), ("w", "filter%soilc", 6)]
    assert acc(rec, 7) == [
        ("r", "filter%num_soilc", 7),
        ("r", "i_type", 7),
        ("w", "x%y%z", 7),
    ]
    write = rec.accesses(path="x%y%z")[0]
    assert write.base == "x" and write.origin is Origin.GLOBAL


ASSOC = """
subroutine s(this, n)
  implicit none
  class(foo_type), intent(inout) :: this
  integer, intent(in) :: n
  integer :: i
  associate(a => this%x, b => this%y(:, n))
    associate(c => a, d => b(1, :))
      do i = 1, n
         c(i) = d(i) + glob
      end do
    end associate
    a(1) = 0.
  end associate
end subroutine s
"""


def test_nested_associate_scopes():
    # the global `a` is shadowed by the associate name
    rec = walk(ASSOC, globals=[gvar("glob"), gvar("a")])
    assert rec.associate_ranges() == [(6, 13), (7, 11)]
    assert rec.associations(ln=9) == {
        "a": "this%x",
        "b": "this%y",
        "c": "this%x",
        "d": "this%y",
    }
    assert rec.associations(ln=12) == {"a": "this%x", "b": "this%y"}
    assert rec.associations(ln=4) == {}
    # subscripts in associate targets are read at the associate statement
    assert acc(rec, 6) == [("r", "n", 6)]
    assert acc(rec, 9) == [
        ("r", "i", 9),
        ("r", "this%y", 9),
        ("r", "glob", 9),
        ("r", "i", 9),
        ("w", "this%x", 9),
    ]
    write = rec.accesses(path="this%x", status="w")[0]
    assert write.via == "c" and write.base == "this" and write.origin is Origin.DUMMY
    assert acc(rec, 12) == [("w", "this%x", 12)]
    assert rec.lookup("a").origin is Origin.GLOBAL  # outside the associate


def test_associate_expression_names():
    txt = """
subroutine s(x, y)
  implicit none
  real(r8), intent(in) :: x(:)
  real(r8), intent(out) :: y
  associate(m => size(x), t => x(1) + x(2))
    y = m * t
  end associate
end subroutine s
"""
    rec = walk(txt)
    assert acc(rec, 5) == [("r", "x", 5), ("r", "x", 5), ("r", "x", 5)]
    assert acc(rec, 6) == [("r", "m", 6), ("r", "t", 6), ("w", "y", 6)]
    assert rec.accesses(path="m")[0].origin is Origin.ASSOCIATE
    assert rec.associations(ln=6) == {"m": None, "t": None}


CALLS = """
subroutine s(bounds, x)
  implicit none
  type(bounds_type), intent(in) :: bounds
  real(r8), intent(inout) :: x(:)
  integer :: i
  call foo(bounds%begc, x(i), x(1) + 2.0, n=i)
  call col_nf%init(bounds%begc, bounds%endc)
end subroutine s
"""


def test_subroutine_calls():
    rec = walk(CALLS, globals=[proc("foo"), gvar("col_nf")])
    foo, init = rec.calls
    assert (foo.name, foo.ln, foo.passed_object) == ("foo", 6, None)
    assert [(a.keyword, a.path) for a in foo.args] == [
        (None, "bounds%begc"),
        (None, "x"),
        (None, None),  # expression actual argument
        ("n", "i"),
    ]
    # designator actual args are "arg" (resolved later against callee intent)
    assert acc(rec, 6) == [
        ("arg", "bounds%begc", 6),
        ("r", "i", 6),
        ("arg", "x", 6),
        ("r", "x", 6),
        ("arg", "i", 6),
    ]
    assert (init.name, init.passed_object) == ("col_nf%init", "col_nf")
    assert acc(rec, 7) == [
        ("arg", "col_nf", 7),
        ("arg", "bounds%begc", 7),
        ("arg", "bounds%endc", 7),
    ]
    assert rec.accesses(path="col_nf")[0].origin is Origin.GLOBAL


def test_unknown_subroutine_is_an_error():
    txt = """
subroutine s()
  implicit none
  call nowhere()
end subroutine s
"""
    with pytest.raises(SemanticError, match="nowhere"):
        walk(txt)


def test_function_references():
    txt = """
subroutine s(x, z, y)
  implicit none
  real(r8), intent(in) :: x(:), z
  real(r8), intent(out) :: y
  y = myfunc(x, k=2) + max(x(1), z)
end subroutine s
"""
    rec = walk(txt, globals=[proc("myfunc")])
    assert [(f.name, f.intrinsic, f.ln) for f in rec.function_calls] == [
        ("myfunc", False, 5),
        ("max", True, 5),
    ]
    assert [(a.keyword, a.path) for a in rec.function_calls[0].args] == [
        (None, "x"),
        ("k", None),
    ]
    # user-function designator args are "arg"; intrinsic args are reads
    assert acc(rec, 5) == [
        ("arg", "x", 5),
        ("r", "x", 5),
        ("r", "z", 5),
        ("w", "y", 5),
    ]


def test_pointers_and_allocation():
    txt = """
subroutine s(this, n)
  implicit none
  class(foo_type), intent(inout) :: this
  integer, intent(in) :: n
  real(r8), pointer :: p(:)
  integer :: ier
  p => this%arr
  p(1) = 2.
  allocate(this%buf(n), stat=ier)
  deallocate(this%buf)
end subroutine s
"""
    rec = walk(txt)
    assert rec.pointer_targets() == {"p": ["this%arr"]}
    assert acc(rec, 7) == [("w", "p", 7)]
    assert rec.accesses(ln=7)[0].context == "ptr-assign"
    assert acc(rec, 8) == [("w", "p", 8)]
    assert acc(rec, 9) == [("r", "n", 9), ("w", "this%buf", 9), ("w", "ier", 9)]
    assert [a.context for a in rec.accesses(ln=9)] == ["expr", "allocate", "stat"]
    assert acc(rec, 10) == [("w", "this%buf", 10)]
    assert rec.accesses(ln=10)[0].context == "deallocate"


def test_function_result():
    txt = """
real(r8) function f(a) result(r)
  implicit none
  real(r8), intent(in) :: a
  r = a * 2
end function f
"""
    rec = walk(txt)
    assert rec.result is not None and rec.result.name == "r"
    assert [d.name for d in rec.dummy_args] == ["a"]
    assert acc(rec, 4) == [("r", "a", 4), ("w", "r", 4)]
    assert rec.accesses(path="r")[0].origin is Origin.RESULT


def test_function_result_without_result_clause():
    txt = """
function g(a)
  implicit none
  real(r8), intent(in) :: a
  real(r8) :: g
  g = a
end function g
"""
    rec = walk(txt)
    assert rec.result is not None and rec.result.name == "g"
    assert "g" not in rec.local_variables
    assert rec.accesses(path="g")[0].origin is Origin.RESULT


HOST = """
subroutine host(x)
  implicit none
  real(r8), intent(inout) :: x
  real(r8) :: h, shadow
  call inner()
contains
  subroutine inner()
    implicit none
    real(r8) :: shadow
    shadow = h + x
    call sibling()
  end subroutine inner
  subroutine sibling()
  end subroutine sibling
end subroutine host
"""


def test_external_with_component_is_data():
    # names from an unavailable module have unknown kind; a component
    # reference makes them data objects
    txt = """
subroutine s(n)
  implicit none
  integer, intent(in) :: n
  real(r8) :: x
  call ep%init(n)
  x = ep%val
  call ext_sub(ep2, x)
end subroutine s
"""
    ext = Symbol("ep", SymbolKind.EXTERNAL, Origin.EXTERNAL)
    ext2 = Symbol("ep2", SymbolKind.EXTERNAL, Origin.EXTERNAL)
    ext_sub = Symbol("ext_sub", SymbolKind.EXTERNAL, Origin.EXTERNAL)
    rec = walk(txt, globals=[gvar("r8"), ext, ext2, ext_sub])
    assert rec.calls[0].object_path == "ep"
    assert acc(rec, 6) == [("r", "ep%val", 6), ("w", "x", 6)]
    assert rec.accesses(path="ep%val")[0].origin is Origin.EXTERNAL
    # a whole external name passed as an actual argument may be data
    assert [a.path for a in rec.calls[1].args] == ["ep2", "x"]
    assert acc(rec, 7) == [("arg", "ep2", 7), ("arg", "x", 7)]


def test_internal_subprograms_use_host_scope():
    host_rec = walk(HOST)
    # internal subprogram bodies belong to their own records
    assert all(a.ln < 6 for a in host_rec.accesses())
    assert [c.name for c in host_rec.calls] == ["inner"]
    assert host_rec.lookup("inner").kind is SymbolKind.PROCEDURE

    inner_rec = walk(HOST, host=host_rec, tree_index=0)
    assert acc(inner_rec, 10) == [("r", "h", 10), ("r", "x", 10), ("w", "shadow", 10)]
    origins = [a.origin for a in inner_rec.accesses(ln=10)]
    assert origins == [Origin.HOST, Origin.HOST, Origin.LOCAL]
    assert [c.name for c in inner_rec.calls] == ["sibling"]


CONTROL = """
subroutine s(this, n, x)
  implicit none
  class(foo_type), intent(inout) :: this
  integer, intent(in) :: n
  real(r8), intent(inout) :: x(:)
  real(r8), pointer :: p(:)
  integer :: i, ier, unitn
  namelist /elm_inparm/ i, unitn
  do i = 1, n
     if (x(i) > 0) exit
     if (x(i) < 0) cycle
  end do
  select case (n)
  case (1, 2:glob)
     x(1) = 0.
  case default
     return
  end select
  nullify(p, this%q)
  read(unitn, *) x(n), i
  read(unitn, nml=elm_inparm, iostat=ier)
  SHR_ASSERT((n > 0), 'bad n')
end subroutine s
"""


def test_control_statements():
    rec = walk(CONTROL, globals=[gvar("glob")])
    assert acc(rec, 10) == [("r", "i", 10), ("r", "x", 10)]
    assert acc(rec, 11) == [("r", "i", 11), ("r", "x", 11)]
    # selector read at select line; case values read at the case line
    assert acc(rec, 13) == [("r", "n", 13)]
    assert acc(rec, 14) == [("r", "glob", 14)]
    assert acc(rec, 15) == [("w", "x", 15)]
    assert acc(rec, 17) == []
    assert rec.select_ranges() == [(13, 18)]
    # nullify writes the pointer
    assert acc(rec, 19) == [("w", "p", 19), ("w", "this%q", 19)]
    assert [a.context for a in rec.accesses(ln=19)] == ["nullify", "nullify"]
    # read: controls are reads, items are writes
    assert acc(rec, 20) == [("r", "unitn", 20), ("r", "n", 20), ("w", "x", 20), ("w", "i", 20)]
    assert [a.context for a in rec.accesses(ln=20, status="w")] == ["read", "read"]
    # namelist read writes every group member; iostat is written
    assert acc(rec, 21) == [("r", "unitn", 21), ("w", "i", 21), ("w", "unitn", 21), ("w", "ier", 21)]
    assert [a.context for a in rec.accesses(ln=21, status="w")] == ["read", "read", "iostat"]
    assert rec.namelists == {"elm_inparm": ["i", "unitn"]}
    # macro arguments are reads
    assert acc(rec, 22) == [("r", "n", 22)]


def test_namelist_write_reads_members():
    txt = """
subroutine s(unitn)
  implicit none
  integer, intent(in) :: unitn
  integer :: i
  namelist /grp/ i
  write(unitn, nml=grp)
end subroutine s
"""
    rec = walk(txt)
    assert acc(rec, 6) == [("r", "unitn", 6), ("r", "i", 6)]


def test_stop_goto_intrinsic():
    txt = """
subroutine s(x, msg)
  implicit none
  real(r8), intent(inout) :: x
  character(len=*), intent(in) :: msg
  intrinsic erf
  if (x > 0) go to 30
  x = erf(x)
30 continue
  if (x < 0) stop
  error stop msg
end subroutine s
"""
    rec = walk(txt, globals=[gvar("r8")])
    assert rec.lookup("erf").kind is SymbolKind.INTRINSIC
    assert rec.lookup("erfc_scaled").kind is SymbolKind.INTRINSIC
    assert acc(rec, 6) == [("r", "x", 6)]
    assert acc(rec, 7) == [("r", "x", 7), ("w", "x", 7)]
    assert acc(rec, 8) == []
    assert acc(rec, 10) == [("r", "msg", 10)]
