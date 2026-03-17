import textwrap
from collections import defaultdict
from pathlib import Path

import xarray as xr

import scripts.io.helper as hio
from scripts.DerivedType import DerivedType
from scripts.export_objects import unpickle_unit_test
from scripts.fortran_modules import FortranModule
from scripts.utilityFunctions import Variable

SUB_NAME = "create_emulator_fields"
EMULATOR_MOD = "emulator_mod"
EMULATOR_TYPE = "emulator_t"
FORT_MOD_NAME = "emulator_setup_mod"


def read_var_order(nc_path: Path, attr_name: str) -> list[str]:
    ds = xr.open_dataset(nc_path)
    order_attr = ds.attrs.get(attr_name)
    if order_attr is None:
        return [str(v) for v in ds.data_vars]
    return [s.replace("__", "%") for s in str(order_attr).split(",") if s]


def gen_subroutine_signature(
    in_order: list[str],
    out_order: list[str],
    type_dict: dict[str, DerivedType],
):
    active_instances = hio.get_active_instances(type_dict)

    needed_instances: dict[str, set[str]] = defaultdict(set)
    for v in in_order:
        inst, field = v.split("%")
        needed_instances[inst].add(field)
    for v in out_order:
        inst, field = v.split("%")
        needed_instances[inst].add(field)

    active_instances = {
        key: val for key, val in active_instances.items() if key in needed_instances
    }

    gv_instances: dict[str, Variable] = {}
    for inst, field in needed_instances.items():
        inst_var = active_instances[inst]
        dtype = type_dict[inst_var.type]
        for f in field:
            field_var = dtype.components[f]
            gv_instances[f"{inst}%{f}"] = field_var.copy()

    stmts = hio.var_type_use_statements(active_instances, type_dict)
    tabs = hio.indent(hio.Tab.shift)
    stmts.append(f"{tabs}use filtermod, only : clumpfilter\n")

    arg_str = ",&\n".join([f"{tabs}{inst}" for inst in needed_instances.keys()])
    if arg_str:
        arg_str = f"{arg_str},&\n"
    arg_decls = (
        [
            f"{tabs}type({var.type}), intent(in) :: {var.name}"
            for var in active_instances.values()
        ]
        if arg_str
        else []
    )

    sig = textwrap.dedent(f"""
    subroutine {SUB_NAME}({arg_str}{tabs}emulator, filter)
{''.join(stmts)}
{'\n'.join(arg_decls)}
       type({EMULATOR_TYPE}), intent(inout) :: emulator
       type(clumpfilter), intent(in)    :: filter
      """)

    return sig


def generate_mapping_module(
    inputs_nc: Path,
    outputs_nc: Path,
    out_f90: Path,
) -> None:

    mod_dict, sub_dict, type_dict = unpickle_unit_test()

    in_order = read_var_order(inputs_nc, "spel_input_vars")
    out_order = read_var_order(outputs_nc, "spel_output_vars")

    tabs = hio.indent(hio.Tab.reset)
    tabs = hio.indent(hio.Tab.shift)
    # Build add() calls in the saved order
    # TODO: choose correct filters from the call signature of elm subroutine
    in_add_lines = []
    for v in in_order:
        in_add_lines.append(
            f"{tabs}call in_list%add({v}, filter%num_soilc, filter%soilc)"
        )

    out_add_lines = []
    for v in out_order:
        out_add_lines.append(
            f"{tabs}call out_list%add({v},filter%num_soilc, filter%soilc)"
        )

    sig = gen_subroutine_signature(in_order, out_order, type_dict)
    body = textwrap.dedent(f"""
    {sig}

    type(field_list_t) :: in_list, out_list

    ! Build input list in the exact order used during training
    {'\n'.join(in_add_lines)}

    ! Build output list in the exact order used during training
    {'\n'.join(out_add_lines)}

      if (.not. emulator%initialized) then
         call emulator%init_from_field_lists("spel_emulator_traced.pt", in_list, out_list)
      end if

    end subroutine {SUB_NAME}
    """.strip("\n"))

    f90_src = textwrap.dedent(f"""
    module {FORT_MOD_NAME}
      use iso_fortran_env, only : real64, int32
      use {EMULATOR_MOD}
      implicit none
      integer, parameter :: rkind = real64, ikind = int32
    contains

    {body}

    end module {FORT_MOD_NAME}""")

    out_f90.write_text(f90_src)


def main():
    from scripts.config import spel_mods_dir, unittests_dir

    root = Path(unittests_dir)
    order_dir = root / "input-data"

    inputs_nc = order_dir / "model-inputs-order.nc"
    outputs_nc = order_dir / "model-outputs-order.nc"

    assert inputs_nc.exists(), f"{inputs_nc} does not exist"
    assert outputs_nc.exists(), f"{outputs_nc} does not exist"

    out_f90 = Path(spel_mods_dir) / "ftorch" / "src" / f"{FORT_MOD_NAME}.F90"
    generate_mapping_module(inputs_nc, outputs_nc, out_f90)
    print("Wrote", out_f90)


if __name__ == "__main__":
    main()
