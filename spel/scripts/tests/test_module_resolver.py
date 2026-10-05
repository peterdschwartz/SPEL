"""
ModuleResolver: resolves names visible at module scope from SPEL's module data
(FortranModule: global_vars, defined_types, subroutines, use_stmts, module head).

Policy:
  * accessibility (public/private statements and attributes) is honored
  * names from modules SPEL can't load (bad_modules, missing files, libraries,
    intrinsic modules) are SymbolKind.EXTERNAL
  * a whole-module `use` of such a module makes any otherwise undeclared name
    EXTERNAL (intrinsics still win)
"""

import os
from pathlib import Path

import pytest

from spel.scripts.fortran_modules import FortranModule, parse_use_stmts
from spel.scripts.fortran_parser.scope_walk import walk_subroutine
from spel.scripts.fortran_parser.spel_ast import SemanticError
from spel.scripts.fortran_parser.symbols import Origin, Scope, SymbolKind
from spel.scripts.fortran_parser.tests.test_Parser import parse_statements
from spel.scripts.module_resolver import ModuleResolver, ModuleScopes, scan_module_head
from spel.scripts.types import LineTuple
from spel.scripts.utilityFunctions import Variable


def head_lines(txt: str) -> list[LineTuple]:
    return [
        LineTuple(line=line.strip().lower(), ln=i)
        for i, line in enumerate(txt.strip("\n").splitlines())
        if line.strip()
    ]


def make_mod(name, head="", gvars=(), types=(), subs=(), uses_ln=None):
    lines = head_lines(head)
    mod = FortranModule(name=name, fname=f"{name}.F90", ln=0)
    mod.module_lines = lines
    mod.end_of_head_ln = len(head.strip("\n").splitlines())
    mod.use_stmts = parse_use_stmts([lt for lt in lines if lt.line.startswith("use")])
    mod.global_vars = {v: Variable("integer", v, "?", 0, 0) for v in gvars}
    mod.defined_types = {t: None for t in types}
    mod.subroutines = {f"{name}::{s}" for s in subs}
    return mod


def resolver_for(*mods, name=None, **kw):
    mod_dict = {m.name: m for m in mods}
    return ModuleResolver(name or mods[0].name, mod_dict, sub_dict={}, **kw)


def kind(resolver, name):
    sym = resolver.lookup(name)
    return None if sym is None else (sym.kind, sym.origin)


VAR = (SymbolKind.VARIABLE, Origin.GLOBAL)
PROC = (SymbolKind.PROCEDURE, Origin.GLOBAL)
TYPE = (SymbolKind.DERIVED_TYPE, Origin.GLOBAL)
EXT = (SymbolKind.EXTERNAL, Origin.EXTERNAL)


# ---------------------------------------------------------------------------
# module head scan
# ---------------------------------------------------------------------------
HEAD = """
use shr_kind_mod, only: r8 => shr_kind_r8
implicit none
private
public :: a, my_type, generic_sub
integer, public :: b = 1, c(2) = (/1, 2/)
integer, parameter :: hidden = 3
character(len=*), parameter, private :: sourcefile = 'x'
type, public :: pub_type
  private
  integer :: field
contains
  procedure, public :: init => pub_init
end type pub_type
type :: my_type
  integer :: y
end type
interface generic_sub
  module procedure generic_sub_r, generic_sub_i
end interface generic_sub
interface operator(+)
  module procedure add_t
end interface
abstract interface
  subroutine cb_iface(x)
    integer, intent(in) :: x
  end subroutine cb_iface
  real(r8) function fn_iface(y)
    real(r8) :: y
  end function fn_iface
end interface
interface
  subroutine c_routine(z) bind(c)
    integer :: z
  end subroutine
end interface
"""


def test_scan_module_head():
    head = scan_module_head(head_lines(HEAD))
    assert head.default_private
    assert head.access == {
        "a": True,
        "my_type": True,
        "generic_sub": True,
        "b": True,
        "c": True,
        "sourcefile": False,
        "pub_type": True,
    }
    assert head.generics == {"generic_sub"}
    assert head.interface_procs == {"cb_iface", "fn_iface", "c_routine"}


def test_scan_default_public():
    head = scan_module_head(head_lines("implicit none\npublic\nprivate :: x"))
    assert not head.default_private
    assert head.access == {"x": False}
    assert not scan_module_head(head_lines("implicit none")).default_private


# ---------------------------------------------------------------------------
# lookup at module scope
# ---------------------------------------------------------------------------
def test_own_module_names():
    mod = make_mod(
        "m",
        "interface tri\n module procedure tri_a\nend interface tri",
        gvars=["g"],
        types=["t"],
        subs=["s", "tri_a"],
    )
    r = resolver_for(mod)
    assert kind(r, "g") == VAR
    assert kind(r, "t") == TYPE
    assert kind(r, "s") == PROC
    assert kind(r, "tri") == PROC
    # module procedures know their routine id; generics don't
    assert r.lookup("s").target == "m::s"
    assert r.lookup("tri").target is None
    assert r.lookup("nope") is None


def test_use_only_and_rename():
    b = make_mod("b", gvars=["x", "y"], subs=["bsub"])
    a = make_mod("a", "use b, only: x, z => y, bsub", gvars=[])
    r = resolver_for(a, b)
    assert kind(r, "x") == VAR
    assert kind(r, "z") == VAR
    assert kind(r, "bsub") == PROC
    assert r.lookup("bsub").target == "b::bsub"
    assert r.lookup("y") is None


def test_whole_use_respects_private():
    b = make_mod(
        "b",
        "private :: sourcefile\ninteger :: x\ncharacter(len=*), parameter :: sourcefile='b'",
        gvars=["x", "sourcefile"],
    )
    a = make_mod("a", "use b", gvars=["sourcefile"])
    r = resolver_for(a, b)
    assert kind(r, "x") == VAR
    # own name shadows; b's private name isn't imported
    assert r.lookup("sourcefile").origin is Origin.GLOBAL


def test_use_only_private_name_raises():
    b = make_mod("b", "private\npublic :: x", gvars=["x", "y"])
    a = make_mod("a", "use b, only: y")
    with pytest.raises(SemanticError):
        resolver_for(a, b).lookup("y")


def test_reexport_chain():
    c = make_mod("c", gvars=["deep"])
    b_pub = make_mod("b", "use c", gvars=["mid"])
    a = make_mod("a", "use b")
    assert kind(resolver_for(a, b_pub, c), "deep") == VAR

    b_priv = make_mod("b", "use c\nprivate\npublic :: mid", gvars=["mid"])
    r = resolver_for(a, b_priv, c)
    assert kind(r, "mid") == VAR
    assert r.lookup("deep") is None


def test_rename_in_whole_use():
    b = make_mod("b", gvars=["x", "y"])
    a = make_mod("a", "use b, w => x")
    r = resolver_for(a, b)
    assert kind(r, "w") == VAR
    assert kind(r, "y") == VAR
    assert r.lookup("x") is None


def test_cyclic_use_terminates():
    a = make_mod("a", "use b", gvars=["xa"])
    b = make_mod("b", "use a", gvars=["xb"])
    r = resolver_for(a, b)
    assert kind(r, "xb") == VAR
    assert kind(r, "xa") == VAR


# ---------------------------------------------------------------------------
# unavailable modules -> EXTERNAL
# ---------------------------------------------------------------------------
def test_unavailable_only_list_is_external():
    a = make_mod("a", "use shr_sys_mod, only: shr_sys_abort, abort => shr_sys_flush")
    r = resolver_for(a)
    assert kind(r, "shr_sys_abort") == EXT
    assert kind(r, "abort") == EXT
    assert r.lookup("shr_sys_flush") is None
    assert r.fallback("anything") is None


def test_whole_use_unavailable_falls_back():
    a = make_mod("a", "use iso_c_binding", gvars=["g"])
    r = resolver_for(a)
    assert r.lookup("c_int") is None
    assert r.fallback("c_int").kind is SymbolKind.EXTERNAL
    scope = Scope("routine", resolver=r)
    assert scope.resolve("c_int").kind is SymbolKind.EXTERNAL
    # intrinsics and module names win over the fallback
    assert scope.resolve("size").kind is SymbolKind.INTRINSIC
    assert scope.resolve("g").kind is SymbolKind.VARIABLE


def test_fallback_is_reexported():
    b = make_mod("b", "use netcdf")
    a = make_mod("a", "use b")
    assert resolver_for(a, b).fallback("nf90_open").kind is SymbolKind.EXTERNAL
    b_priv = make_mod("b", "use netcdf\nprivate")
    assert resolver_for(a, b_priv).fallback("nf90_open") is None


def test_no_fallback_raises():
    scope = Scope("routine", resolver=resolver_for(make_mod("a")))
    with pytest.raises(SemanticError):
        scope.resolve("undeclared")


def test_routine_level_unavailable_uses():
    """bad-module uses inside a routine are commented out before parsing; the
    resolver recovers them from the module's use statements by line range."""
    text = """
module a
contains
subroutine s()
use shr_sys_mod, only: shr_sys_abort
use netcdf
end subroutine
subroutine t()
end subroutine
"""
    lines = head_lines(text)
    mod = make_mod("a")
    mod.use_stmts = parse_use_stmts([lt for lt in lines if lt.line.startswith("use")])
    mod.end_of_head_ln = 1
    in_s = resolver_for(mod, line_ranges=[(3, 6)])
    assert kind(in_s, "shr_sys_abort") == EXT
    assert in_s.fallback("nf90_open").kind is SymbolKind.EXTERNAL
    in_t = resolver_for(mod, line_ranges=[(7, 8)])
    assert in_t.lookup("shr_sys_abort") is None
    assert in_t.fallback("nf90_open") is None


def test_use_module_in_routine():
    b = make_mod("b", "private :: y", gvars=["x", "y"])
    a = make_mod("a")
    r = resolver_for(a, b)
    use = parse_statements("\nuse b, only: x\n").statements[0]
    assert set(r.use_module(use)) == {"x"}
    use_all = parse_statements("\nuse b\n").statements[0]
    assert set(r.use_module(use_all)) == {"x"}
    bad = parse_statements("\nuse b, only: y\n").statements[0]
    with pytest.raises(SemanticError):
        r.use_module(bad)
    ext = parse_statements("\nuse shr_log_mod, only: errmsg\n").statements[0]
    assert r.use_module(ext)["errmsg"].kind is SymbolKind.EXTERNAL


EXTERNAL_WALK = """
subroutine s(n)
  use shr_sys_mod, only: shr_sys_abort, shr_sys_flush
  integer, intent(inout) :: n
  n = shr_sys_flush(n) + size((/1, 2/))
  if (n < 0) call shr_sys_abort('neg')
end subroutine s
"""


def test_walk_with_external_symbols():
    r = resolver_for(make_mod("a"))
    rec = walk_subroutine(parse_statements(EXTERNAL_WALK).statements[0], resolver=r)
    assert [c.name for c in rec.calls] == ["shr_sys_abort"]
    assert [f.name for f in rec.function_calls] == ["shr_sys_flush", "size"]
    assert {a.path for a in rec.accesses()} == {"n"}


# ---------------------------------------------------------------------------
# integration with the parsed test fixtures
# ---------------------------------------------------------------------------
def test_fixture_module_resolution():
    import spel.scripts.dynamic_globals as dg
    from spel.scripts.edit_files import process_for_unit_test
    from spel.scripts.tests.test_ParseSubroutine import elm_src_pointing_to, test_dir

    with elm_src_pointing_to(Path(test_dir)):
        dg.populate_interface_list()
        mod_dict, sub_dict = {}, {}
        process_for_unit_test(
            case_dir=test_dir,
            mod_dict=mod_dict,
            mods=[],
            required_mods=[],
            sub_dict=sub_dict,
            sub_name_list=["test_sub_parse::call_sub"],
            overwrite=False,
            verbose=False,
        )

    r = ModuleResolver("test_sub_parse", mod_dict, sub_dict)
    assert kind(r, "tridiagonal") == PROC  # generic interface
    assert kind(r, "tridiagonal_sr") == PROC
    assert kind(r, "bounds_type") == TYPE
    assert kind(r, "col_nf") == VAR
    assert kind(r, "shr_const_spval") == VAR  # whole `use shr_const_mod`
    assert kind(r, "test_type") == TYPE
    assert kind(r, "nlevdecomp_full") == VAR  # use constants_mod, only:
    assert r.lookup("param1") is None  # not in the only-list
    assert kind(r, "elm_fates") == EXT  # use remove_mod (missing file)
    assert kind(r, "r8") == VAR  # shr_const_mod declares r8

    subs = [s for s in sub_dict.values() if s.module == "test_sub_parse" and not s.library]
    scopes = ModuleScopes(mod_dict, sub_dict)
    failures = {}
    for sub in subs:
        try:
            sub.walk_syntax_tree(scopes)
        except SemanticError as e:
            failures[sub.name] = str(e)
    assert failures == {}
