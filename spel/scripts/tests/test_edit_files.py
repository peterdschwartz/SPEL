"""
edit_files: names whose definitions are commented out of a module (declarations
of unavailable types, removed routines) must not survive in the `use` statements
of modules processed later, or the resolver reports them as not public.
"""

from pathlib import Path

import pytest

import spel.scripts.edit_files as ef
from spel.scripts.types import LineTuple, LogicalLineIterator, ParseState

PROVIDER = """
module decompmod
  use shr_kind_mod, only : r8 => shr_kind_r8
  use mct_mod     , only : mct_gsmap
  implicit none
  type decomp_type
     integer, pointer :: gdc2glo(:)
     type(mct_gsmap) :: comp_map
  end type decomp_type
  type(decomp_type), public, target :: ldecomp
  type(mct_gsmap)  ,public,target :: gsmap_lnd_gdc2glo, gsmap_col_gdc2glo(2)
  integer, public :: nclumps
  public :: get_elmlevel_gsmap
  public :: helper
contains
  subroutine get_elmlevel_gsmap(gsmap)
    gsmap => gsmap_lnd_gdc2glo
  end subroutine get_elmlevel_gsmap
  subroutine helper(x)
    integer :: x
    x = nclumps
  end subroutine helper
end module decompmod
"""

USER = """
module firemod
  use decompmod, only : gsmap_lnd_gdc2glo
  use decompmod, only: bounds_type, ldecomp, lnd_map => gsmap_lnd_gdc2glo, &
                       nclumps
  use decompmod, only : get_elmlevel_gsmap, helper
  implicit none
  integer :: comp_map
contains
  subroutine fire_init(n)
    integer :: n
    call mct_gsmap_init(lnd_map, n)
    n = comp_map + nclumps
  end subroutine fire_init
end module firemod
"""


def make_state(txt: str, name: str) -> tuple[ParseState, list[LineTuple]]:
    lines = [LineTuple(line=line + "\n", ln=i) for i, line in enumerate(txt.split("\n"))]
    state = ParseState(
        module_name=name,
        fort_mod=None,
        cpp_file=False,
        work_lines=lines,
        orig_lines=lines,
        path=Path(f"{name}.F90"),
        curr_line=None,
        line_it=LogicalLineIterator(lines),
        logger=ef.get_logger("test_edit_files"),
        sub_init_dict={},
        removed_subs=[],
    )
    return state, lines


def edit(txt: str, name: str) -> list[str]:
    state, lines = make_state(txt, name)
    head_ln = next(lt.ln for lt in lines if lt.line.strip() == "contains")
    ef.comment_bad_references(state, ef.make_pass_manager(), head_ln)
    return [lt.line.rstrip("\n") for lt in ef.apply_comments(lines)]


@pytest.fixture(autouse=True)
def fresh_globals(monkeypatch):
    monkeypatch.setattr(ef, "bad_symbols", set(ef.bad_symbols))
    monkeypatch.setattr(ef, "unavailable_exports", {})


def active(lines: list[str]) -> str:
    return "\n".join(l for l in lines if "!#py" not in l)


def test_commented_module_declarations_become_unavailable_exports():
    out = active(edit(PROVIDER, "decompmod"))
    assert "gsmap_lnd_gdc2glo, gsmap_col_gdc2glo" not in out
    assert "integer, public :: nclumps" in out
    exports = ef.unavailable_exports["decompmod"]
    assert {"gsmap_lnd_gdc2glo", "gsmap_col_gdc2glo"} <= exports
    # removed routines are unavailable too; kept ones are not
    assert "get_elmlevel_gsmap" in exports and "helper" not in exports
    # derived-type components and routine locals are not module exports
    assert not exports & {"comp_map", "gdc2glo", "gsmap", "ldecomp", "nclumps"}


def test_use_statements_drop_unavailable_names():
    edit(PROVIDER, "decompmod")
    out = edit(USER, "firemod")
    kept = active(out)
    # nothing left to import: the statement is commented out
    assert "use decompmod, only : gsmap_lnd_gdc2glo" not in kept
    # only the unavailable names are dropped (continuation folded into one line)
    i = out.index("  use decompmod, only: bounds_type, ldecomp, nclumps")
    assert out[i + 1].lstrip().startswith("!#py")
    assert "  use decompmod, only: helper" in out
    # references to the dropped local name are commented out in this file only
    assert "call mct_gsmap_init(lnd_map, n)" not in kept
    # same-named local of the user module is untouched
    assert "n = comp_map + nclumps" in kept
    assert "integer :: comp_map" in kept


def test_resolver_use_stmts_follow_pruned_imports():
    """fort_mod.use_stmts (from the original source) must match the edited file."""
    from types import SimpleNamespace

    from spel.scripts.fortran_modules import parse_use_stmts

    edit(PROVIDER, "decompmod")
    state, lines = make_state(USER, "firemod")
    it = LogicalLineIterator([LineTuple(lt.line, lt.ln) for lt in lines])
    uses = [
        LineTuple(fl.line, it.get_start_ln()) for fl in it if fl.line.startswith("use")
    ]
    state.fort_mod = SimpleNamespace(use_stmts=parse_use_stmts(uses))
    head_ln = next(lt.ln for lt in lines if lt.line.strip() == "contains")
    ef.comment_bad_references(state, ef.make_pass_manager(), head_ln)

    summary = [
        (s.lineno, sorted(str(o) for o in s.objs)) for s in state.fort_mod.use_stmts
    ]
    assert summary == [
        (3, ["bounds_type", "ldecomp", "nclumps"]),
        (5, ["helper"]),
    ]


def test_untouched_when_nothing_is_unavailable():
    out = edit(USER.replace("decompmod", "othermod"), "firemod")
    assert active(out) == "\n".join(out)
