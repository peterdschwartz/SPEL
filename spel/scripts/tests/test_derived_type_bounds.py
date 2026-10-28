from types import SimpleNamespace

from spel.scripts.allocations import apply_allocations, scan_module
from spel.scripts.DerivedType import rank
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.types import LineTuple
from spel.scripts.utilityFunctions import Variable


def test_rank():
    assert rank("begl:endl") == 1
    assert rank("begg:endg,max_topounits, numurbl") == 3
    assert rank("begp:endp,size(x,1)") == 2


def routine(src: str):
    lines = [LineTuple(line=l, ln=i) for i, l in enumerate(src.strip("\n").split("\n"))]
    return Parser(lines=lines).parse_program().statements[0]


def module(name, routines: dict[str, str], gvars=()):
    sub_dict = {f"{name}::{n}": SimpleNamespace(syntax_tree=routine(src), startline=i) for i, (n, src) in enumerate(routines.items())}
    mod = SimpleNamespace(subroutines=set(sub_dict), global_vars={v: SimpleNamespace(type=t) for v, t in gvars})
    return mod, sub_dict


def dtype(name, **comps):
    return SimpleNamespace(
        type_name=name,
        components={c: Variable(t, c, "", 0, d) for c, (t, d) in comps.items()},
        init_sub_name=None,
    )


def test_bounds_belong_to_the_allocated_object():
    """UrbanParamsType: urbinp%nlev_improad is 3D, this%nlev_improad is 1D."""
    mod, sd = module("urbanparamstype", {
        "init": """
subroutine init(this, begl, endl)
  class(urbanparams_type) :: this
  integer :: begl, endl
  allocate(urbinp%nlev_improad(begg:endg,max_topounits,numurbl))
  allocate(this%nlev_improad(begl:endl)) ; this%nlev_improad(:) = huge(1)
  allocate(this%thick_wall(begl:endl))
end subroutine init
"""}, gvars=[("urbinp", "urbinp_type")])
    t = dtype("urbanparams_type", nlev_improad=("integer", 1), thick_wall=("real", 1))
    u = dtype("urbinp_type", nlev_improad=("integer", 3))
    types = {"urbanparams_type": t, "urbinp_type": u}
    apply_allocations(types, *scan_module(mod, sd, types))
    assert t.components["nlev_improad"].bounds == "begl:endl"
    assert t.components["thick_wall"].bounds == "begl:endl"
    assert u.components["nlev_improad"].bounds == "begg:endg,max_topounits,numurbl"
    assert t.init_sub_name == "urbanparamstype::init"


def test_types_with_the_same_components_get_their_own_bounds():
    """NutrientStateType: column and vegetation bases both have `totveg`."""
    mod, sd = module("nutrientstatetype", {
        "col_init": """
subroutine col_init(this, begc, endc)
  type(column_nutrient_state), intent(inout) :: this
  integer :: begc, endc
  if (.true.) then
     allocate(this%totveg(begc:endc))
  end if
end subroutine col_init
""",
        "veg_init": """
subroutine veg_init(this, begp, endp)
  type(vegetation_nutrient_state), intent(inout) :: this
  integer :: begp, endp
  allocate(this%totveg(begp:endp))
end subroutine veg_init
"""})
    col = dtype("column_nutrient_state", totveg=("real", 1))
    veg = dtype("vegetation_nutrient_state", totveg=("real", 1))
    types = {t.type_name: t for t in (col, veg)}
    apply_allocations(types, *scan_module(mod, sd, types))
    assert col.components["totveg"].bounds == "begc:endc"
    assert veg.components["totveg"].bounds == "begp:endp"


def test_nested_component_allocated_in_another_module():
    """GridcellDataType's grc_cf_init allocates the core of gridcell_nutrient_flux."""
    mod, sd = module("gridcelldatatype", {
        "grc_cf_init": """
subroutine grc_cf_init(this, begg, endg)
  class(gridcell_carbon_flux) :: this
  integer :: begg, endg
  allocate(this%core%dwt_conv_flux(begg:endg))
  this%core%dwt_conv_flux_ptr => this%core%dwt_conv_flux
end subroutine grc_cf_init
"""})
    base = dtype("gridcell_nutrient_flux", dwt_conv_flux=("real", 1), dwt_conv_flux_ptr=("real", 1))
    holder = dtype("gridcell_carbon_flux", core=("gridcell_nutrient_flux", 0))
    holder.components["core%dwt_conv_flux"] = Variable("real", "dwt_conv_flux", "", 0, 1)
    types = {"gridcell_nutrient_flux": base, "gridcell_carbon_flux": holder}
    apply_allocations(types, *scan_module(mod, sd, types))
    assert base.components["dwt_conv_flux"].bounds == "begg:endg"
    assert holder.components["core%dwt_conv_flux"].bounds == "begg:endg"
    assert base.components["dwt_conv_flux_ptr"].pointer == ["this%core%dwt_conv_flux"]


def test_bounds_text_has_only_the_needed_parentheses():
    from spel.scripts.allocations import text

    src = "allocate(this%x(bounds%endp-bounds%begp+1, -nlevsno+1:nlevgrnd, a-(b-c), (a+b)*c, a/(b*c), 0:max(1,n-1)))"
    (st,) = Parser(lines=[LineTuple(line=src, ln=0)]).parse_program().statements
    assert [text(a) for a in st.objects[0].args] == [
        "bounds%endp-bounds%begp+1", "-nlevsno+1:nlevgrnd", "a-(b-c)", "(a+b)*c", "a/(b*c)", "0:max(1,n-1)"]


def test_module_array_takes_the_bounds_of_its_allocation():
    """PhenologyMod's `integer, pointer :: inhemi(:)`: patch-sized, so nsets must duplicate it."""
    mod, sd = module("phenologymod", {
        "cropphenologyinit": """
subroutine cropphenologyinit(bounds)
  type(bounds_type), intent(in) :: bounds
  integer :: work(:)
  allocate( inhemi(bounds%begp:bounds%endp) )
  allocate( work(bounds%begc:bounds%endc) )
end subroutine cropphenologyinit
"""}, gvars=[("inhemi", "integer"), ("work", "integer")])
    mod.global_vars["inhemi"].dim, mod.global_vars["inhemi"].bounds = 1, "(:)"
    mod.global_vars["work"].dim, mod.global_vars["work"].bounds = 1, "(:)"
    scan_module(mod, sd, {})
    assert mod.global_vars["inhemi"].alloc_bounds == "bounds%begp:bounds%endp"
    assert mod.global_vars["inhemi"].bounds == "(:)"  # still deferred-shape: allocated at read time
    assert not hasattr(mod.global_vars["work"], "alloc_bounds")  # a local of the same name is not the module's
