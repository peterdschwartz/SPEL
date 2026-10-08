from types import SimpleNamespace

from spel.scripts.DerivedType import DerivedType, rank
from spel.scripts.types import LineTuple
from spel.scripts.utilityFunctions import Variable


def test_rank():
    assert rank("begl:endl") == 1
    assert rank("begg:endg,max_topounits, numurbl") == 3
    assert rank("begp:endp,size(x,1)") == 2


def test_allocation_bounds_ignore_same_named_field_of_other_type():
    """UrbanParamsType: urbinp%nlev_improad is 3D, this%nlev_improad is 1D."""
    lines = [
        "allocate(urbinp%nlev_improad(begg:endg,max_topounits,numurbl))",
        "allocate(this%nlev_improad        (begl:endl))          ; this%nlev_improad(:) = huge(1)",
        "allocate(this%thick_wall(begl:endl))",
    ]
    mod = SimpleNamespace(module_lines=[LineTuple(l, i) for i, l in enumerate(lines)])
    comps = {
        "nlev_improad": Variable("integer", "nlev_improad", "", 0, 1),
        "thick_wall": Variable("real", "thick_wall", "", 0, 1),
        "nlev_unused": Variable("integer", "nlev_unused", "", 0, 2),
    }
    dtype = SimpleNamespace(declaration="urbanparamstype", components=comps)
    DerivedType.get_allocation_bounds(dtype, {"urbanparamstype": mod})
    assert comps["nlev_improad"].bounds == "begl:endl"
    assert comps["thick_wall"].bounds == "begl:endl"
    assert comps["nlev_unused"].bounds == "nlev_unused_dim1,nlev_unused_dim2"
