from spel.scripts.io.netcdf_io import create_nc_def, create_nc_read, create_nc_write
from spel.scripts.utilityFunctions import Variable

FIXED = Variable("real", "d_con_w", "?", 0, 2, bounds="(ngases,3)")
DEFERRED = Variable("real", "aroot", "?", 0, 1, bounds="(:)")
PTR = Variable("real", "pa", "?", 0, 1, bounds="(:)", ptrscalar=True)
COMP = Variable("real", "col_cf%phr_vr", "?", 0, 2, bounds="(:,:)")
VARS = {v.name: v for v in (FIXED, DEFERRED, PTR, COMP)}


def test_nc_read_allocates_only_deferred_shape_arrays():
    lines = "".join(create_nc_read(VARS, False))
    assert 'nc_alloc(ncid, "d_con_w"' not in lines
    assert "nc_read_var(ncid, 'd_con_w', 2, d_con_w, -1)" in lines
    assert 'nc_alloc(ncid, "aroot", 1, aroot)' in lines
    assert 'nc_alloc(ncid, "col_cf__phr_vr", 2, col_cf%phr_vr)' in lines


def test_unallocated_module_arrays_are_skipped():
    # AllocationMod::aroot is only allocated with crop_prog
    read = "".join(create_nc_read(VARS, False))
    assert "if (nc_has_var(ncid, 'aroot')) then" in read
    assert "nc_has_var(ncid, 'd_con_w')" not in read
    assert "nc_has_var(ncid, 'col_cf__phr_vr')" not in read

    for gen in (create_nc_def, create_nc_write):
        lines = "".join(gen(VARS, False))
        assert "if (allocated(aroot)) then" in lines
        assert "if (associated(pa)) then" in lines
        assert "allocated(d_con_w)" not in lines
        assert "allocated(col_cf%phr_vr)" not in lines


def test_component_with_dimension_name_bounds_is_allocated():
    # components carry their allocate dimension names in `bounds`, not `(:)`
    active = Variable("logical", "top_pp%active", "?", 0, 1, bounds="topo")
    lines = "".join(create_nc_read({active.name: active}, True))
    assert 'nc_alloc(ncid, "top_pp__active", 1, top_pp%active)' in lines
