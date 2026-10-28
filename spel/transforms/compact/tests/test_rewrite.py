from pathlib import Path

from spel.transforms.compact.rewrite import FileEdits, Rewriter
from spel.transforms.compact.select import Site, statement_end, statement_start

LINES = """\
    associate( a => veg_cf%a, &

               b => veg_cf%b , & ! comment
               c => veg_cf%c )
    do fp = 1, num_soilp
       p = filter_soilp(fp)
       a(p) = b(p) + veg_cf%c(p, 1) + &
              veg_cf % c( p ,2) + other%a(p)
    end do""".split("\n")


def test_statement_lines_skip_blank_continuations():
    assert statement_end(LINES, 0) == 3
    assert statement_start(LINES, 3) == 0
    assert statement_end(LINES, 6) == 7
    assert statement_start(LINES, 7) == 6


def test_site_rewrite_keeps_other_subscripts_and_designators():
    fe = FileEdits(Path("x.F90"), LINES)
    rw = Rewriter.__new__(Rewriter)
    rw.rewrite_site(fe, Site("m::s", 6, "veg_cf%a", "a", "p", "fp"))
    rw.rewrite_site(fe, Site("m::s", 6, "veg_cf%c", "veg_cf%c", "p", "fp"))
    out = fe.result()
    assert out[6] == "       a(k0p+fp) = b(p) + veg_cf%c(k0p+fp, 1) + &"
    assert out[7] == "              veg_cf % c(k0p+fp,2) + other%a(p)"


# --- nutrient-transformed sources: fields on an embedded holder (`veg_cf%core%x`) ---
import shutil
import subprocess
from types import SimpleNamespace

from spel.transforms.compact.holders import Embedding, Holder
from spel.transforms.compact.rewrite import TEMPLATE

NESTED = """\
    associate( agnpp => veg_cf%core%agnpp )
    do fp = 1, num_soilp
       p = filter_soilp(fp)
       agnpp(p) = veg_nf % core % agnpp(p) + veg_cf%core%gpp(p)
    end do""".split("\n")


def test_site_rewrite_three_level_paths():
    fe = FileEdits(Path("x.F90"), NESTED)
    rw = Rewriter.__new__(Rewriter)
    rw.rewrite_site(fe, Site("m::s", 3, "veg_cf%core%agnpp", "agnpp", "p", "fp"))
    rw.rewrite_site(fe, Site("m::s", 3, "veg_nf%core%agnpp", "veg_nf%core%agnpp", "p", "fp"))
    assert fe.result()[3] == "       agnpp(k0p+fp) = veg_nf % core % agnpp(k0p+fp) + veg_cf%core%gpp(p)"


def _selection(components, history):
    from spel.transforms.compact.select import Selection

    return Selection(
        dtype="vegetation_nutrient_flux", subgrid="p", instances=["veg_nf"], components=components,
        history=history, sites=[], excluded={}, untraced=[], holder=None,
    )


def test_zero_block_on_a_role_component():
    rw = Rewriter.__new__(Rewriter)
    sel = _selection({"pool": "always", "x": "use_cn"}, {"pool": True, "x": False})
    groups = rw.groups(sel)
    assert [(g.index, g.guard, g.fields) for g in groups] == [(1, None, ["pool"]), (2, "(use_cn)", ["x"])]
    out = rw.zero_block(groups, "num_patch", "value_patch", "retrans", "p")
    assert out[1] == "    if (compact_bound) then"
    assert "       this%retrans%pool(k0p+1:k0p+num_patch) = value_patch" in out
    assert "          this%retrans%x(k0p+1:k0p+num_patch) = value_patch" in out
    assert "       call compact_group_fill(this%retrans%cgrp_id(1), value_patch)" in out
    assert "          call compact_group_fill(this%retrans%cgrp_id(2), value_patch)" in out
    assert any("if (use_cn) then" in l for l in out)
    # column fields use the column offset
    out_c = rw.zero_block(groups, "num_column", "value_column", "", "c")
    assert "       this%pool(k0c+1:k0c+num_column) = value_column" in out_c


def test_generated_bind_procedure_compiles(tmp_path):
    rw = Rewriter.__new__(Rewriter)
    sel = _selection({"leaf_to_litter": "always", "pool_to_leaf": "always", "x": "crop_prog"}, {"leaf_to_litter": True, "pool_to_leaf": False, "x": False})
    groups = rw.groups(sel)
    gen = rw.bind_procedure("vegetation_nutrient_flux", sel, groups)
    text = "\n".join(gen)
    assert "subroutine vegetation_nutrient_flux_compact_bind(this)" in text
    assert "type(vegetation_nutrient_flux), intent(inout) :: this" in text and "class(" not in text
    flat = " ".join(text.split())
    assert "compact_group_new('vegetation_nutrient_flux|always', compact_lev_patch, 2)" in flat
    assert "this%pool_to_leaf => compact_groups(this%cgrp_id(1))%data(:, 2)" in flat
    assert "this%x => compact_groups(this%cgrp_id(2))%data(:, 1)" in flat
    zero = rw.zero_block(groups, "num_patch", "value_patch", "", "p")
    if shutil.which("gfortran") is None:
        return
    module = f"""module NutrientFluxType
  use shr_kind_mod, only : r8 => shr_kind_r8
  use CompactLayoutMod, only : compact_groups, compact_group_new, compact_lev_patch, compact_lev_col
  implicit none
  private
  logical :: crop_prog = .true.
  type, public :: vegetation_nutrient_flux
    integer :: species
    real(r8), pointer :: leaf_to_litter(:) => null()
    real(r8), pointer :: pool_to_leaf(:) => null()
    real(r8), pointer :: x(:) => null()
    integer :: cgrp_id({len(groups)}) = 0
  end type vegetation_nutrient_flux
  public :: vegetation_nutrient_flux_compact_bind, vegetation_nutrient_flux_setvalues
contains
  subroutine vegetation_nutrient_flux_setvalues(this, num_patch, filter_patch, value_patch)
    use CompactLayoutMod, only : compact_soilp_off, compact_soilc_off, compact_bound, compact_group_fill
    type(vegetation_nutrient_flux), intent(inout) :: this
    integer , intent(in) :: num_patch
    integer , intent(in) :: filter_patch(:)
    real(r8), intent(in) :: value_patch
    integer :: k0p
    k0p = compact_soilp_off
{chr(10).join(zero)}
  end subroutine vegetation_nutrient_flux_setvalues
{text}
end module NutrientFluxType
"""
    (tmp_path / "shr_kind_mod.F90").write_text(
        "module shr_kind_mod\n  integer, parameter :: shr_kind_r8 = selected_real_kind(12)\nend module shr_kind_mod\n"
    )
    (tmp_path / "shr_sys_mod.F90").write_text(
        "module shr_sys_mod\ncontains\n  subroutine shr_sys_abort(msg)\n    character(len=*), optional :: msg\n"
        "    stop 1\n  end subroutine shr_sys_abort\nend module shr_sys_mod\n"
    )
    shutil.copy(TEMPLATE, tmp_path / TEMPLATE.name)
    (tmp_path / "NutrientFluxType.F90").write_text(module)
    for f in ("shr_kind_mod.F90", "shr_sys_mod.F90", TEMPLATE.name, "NutrientFluxType.F90"):
        r = subprocess.run(["gfortran", "-c", "-ffree-line-length-none", f], cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode == 0, f"{f}: {r.stderr}"


def test_bind_module_compiles_with_stubs(tmp_path):
    """The generated CompactBindMod against stub ELM modules."""
    from spel.transforms.compact.holders import Embedding, Holder

    rw = Rewriter.__new__(Rewriter)
    rw.files = {}
    rw._trees = {}
    holder = Holder("vegetation_nutrient_flux", SimpleNamespace(filepath=tmp_path / "NutrientFluxType.F90"),
                    [Embedding("veg_cf", "vegetation_carbon_flux", "core", "c12")])
    sel = _selection({"a": "always"}, {"a": False})
    sel.holder = holder
    sel.instances = ["veg_cf"]
    rw.sels = [sel]
    rw.fut = SimpleNamespace(type_dict={"vegetation_carbon_flux": SimpleNamespace(
        instances={"veg_cf": SimpleNamespace(declaration="VegetationDataType")})})
    (tmp_path / "NutrientFluxType.F90").write_text(
        "module NutrientFluxType\n  use shr_kind_mod, only : r8 => shr_kind_r8\n  implicit none\n"
        "  type, public :: vegetation_nutrient_flux\n    real(r8), pointer :: a(:) => null()\n    integer :: cgrp_id(1) = 0\n"
        "  end type\n  public :: vegetation_nutrient_flux_compact_bind\ncontains\n"
        "  subroutine vegetation_nutrient_flux_compact_bind(this)\n    type(vegetation_nutrient_flux), intent(inout) :: this\n"
        "  end subroutine\nend module NutrientFluxType\n"
    )
    text = rw.generate_bind_module()
    assert "call vegetation_nutrient_flux_compact_bind(veg_cf%core)" in text
    assert "use VegetationDataType, only : veg_cf" in text
    if shutil.which("gfortran") is None:
        return
    stubs = {
        "shr_kind_mod.F90": "module shr_kind_mod\n  integer, parameter :: shr_kind_r8 = selected_real_kind(12)\nend module\n",
        "shr_sys_mod.F90": "module shr_sys_mod\ncontains\n  subroutine shr_sys_abort(msg)\n    character(len=*), optional :: msg\n    stop 1\n  end subroutine\nend module\n",
        "decompMod.F90": "module decompMod\n  type bounds_type\n    integer :: begp, endp, begc, endc\n  end type\ncontains\n"
                         "  integer function get_proc_clumps()\n    get_proc_clumps = 1\n  end function\nend module\n",
        "filterMod.F90": "module filterMod\n  type clumpfilter\n    integer, pointer :: soilp(:), soilc(:)\n    integer :: num_soilp, num_soilc\n  end type\n"
                         "  type(clumpfilter), allocatable :: filter(:)\nend module\n",
        "LandunitType.F90": "module LandunitType\n  type lun_t\n    logical, pointer :: ifspecial(:)\n  end type\n  type(lun_t) :: lun_pp\nend module\n",
        "VegetationType.F90": "module VegetationType\n  type veg_t\n    integer, pointer :: landunit(:)\n  end type\n  type(veg_t) :: veg_pp\nend module\n",
        "ColumnType.F90": "module ColumnType\n  type col_t\n    integer, pointer :: landunit(:)\n  end type\n  type(col_t) :: col_pp\nend module\n",
        "VegetationDataType.F90": "module VegetationDataType\n  use NutrientFluxType\n  type vegetation_carbon_flux\n    type(vegetation_nutrient_flux) :: core\n  end type\n"
                                  "  type(vegetation_carbon_flux), target :: veg_cf\nend module\n",
    }
    for name, body in stubs.items():
        (tmp_path / name).write_text(body)
    shutil.copy(TEMPLATE, tmp_path / TEMPLATE.name)
    (tmp_path / "CompactBindMod.F90").write_text(text)
    order = ["shr_kind_mod.F90", "shr_sys_mod.F90", TEMPLATE.name, "decompMod.F90", "filterMod.F90", "LandunitType.F90",
             "VegetationType.F90", "ColumnType.F90", "NutrientFluxType.F90", "VegetationDataType.F90", "CompactBindMod.F90"]
    for f in order:
        r = subprocess.run(["gfortran", "-c", "-ffree-line-length-none", f], cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode == 0, f"{f}: {r.stderr}"
