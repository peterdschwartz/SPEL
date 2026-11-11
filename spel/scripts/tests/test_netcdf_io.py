from spel.scripts.io.netcdf_io import create_nc_def, create_nc_read, create_nc_write
from spel.scripts.utilityFunctions import Variable

FIXED = Variable("real", "d_con_w", "?", 0, 2, bounds="(ngases,3)")
DEFERRED = Variable("real", "aroot", "?", 0, 1, bounds="(:)")
PTR = Variable("real", "pa", "?", 0, 1, bounds="(:)", ptrscalar=True)
COMP = Variable("real", "col_cf%phr_vr", "?", 0, 2, bounds="(:,:)")
VARS = {v.name: v for v in (FIXED, DEFERRED, PTR, COMP)}


def test_nc_read_allocates_only_deferred_shape_arrays():
    lines = "".join(create_nc_read(VARS, False))
    assert "nc_desc(ncid, 'd_con_w'" not in lines
    assert "spel_var_allocate(d_con_w" not in lines
    assert "nc_read_var(ncid, 'd_con_w', 2, d_con_w, -1)" in lines
    assert "call nc_desc(ncid, 'aroot', 1, spel_d)\n" in lines
    assert "call spel_var_allocate(aroot, spel_d)\n" in lines
    assert "call nc_desc(ncid, 'col_cf__phr_vr', 2, spel_d)\n" in lines
    assert "call spel_var_allocate(col_cf%phr_vr, spel_d)\n" in lines
    # bounds first, then the allocation, then the read
    assert lines.index("nc_desc(ncid, 'aroot'") < lines.index("spel_var_allocate(aroot") < lines.index(
        "nc_read_var(ncid, 'aroot'")


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
    assert "call nc_desc(ncid, 'top_pp__active', 1, spel_d)\n" in lines
    assert "call spel_var_allocate(top_pp%active, spel_d)\n" in lines


def test_spel_var_module_has_every_specific():
    from spel.scripts.io.netcdf_io import SPEL_VAR_MAXRANK, gen_spel_var_module

    text = gen_spel_var_module()
    for sfx in ("r8", "int", "lgc"):
        for kind in ("alloc", "ptr"):
            for r in range(SPEL_VAR_MAXRANK + 1):
                name = f"{kind}_{sfx}_{r}"
                assert f"subroutine {name}(var, d)" in text and f"end subroutine {name}" in text
                assert name in text[: text.index("end interface")]
    assert "use netcdf" not in text  # shared with instrumented ELM: no netCDF dependency


def test_derived_constants_are_written_not_read():
    from spel.scripts.io.netcdf_io import add_derived_constants
    from spel.scripts.nml.namelist_cascade import DERIVED_CONSTANTS

    lines = [
        "  subroutine define_vars(ncid)\n",
        "  end subroutine define_vars\n",
        "  subroutine read_constants(io_inst)\n",
        "    call check(nf90_close(ncid))\n",
        "  end subroutine read_constants\n",
        "  subroutine write_constants(io_inst)\n",
        "    call check(nf90_close(ncid))\n",
        "  end subroutine write_constants\n",
    ]
    out = "".join(add_derived_constants(lines, {"urban_hac_int": DERIVED_CONSTANTS["urban_hac_int"]}))
    assert out.count("'urban_hac_int', nf90_int") == 1
    write = out[out.index("subroutine write_constants"):]
    assert "nc_write_var_scalar(ncid, merge(1, merge(2, 0, trim(urban_hac) == 'ON_WASTEHEAT')" in write
    assert "nc_write_var_scalar" not in out[: out.index("subroutine write_constants")]
