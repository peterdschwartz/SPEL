import pickle
import sys

import pandas as pd

from scripts.analyze_subroutines import Subroutine
from scripts.DerivedType import DerivedType
from scripts.fortran_modules import FortranModule
from scripts.mod_config import E3SM_SRCROOT, django_database, scripts_dir
from scripts.utilityFunctions import Variable


def pickle_unit_test(
    mod_dict: dict[str, FortranModule],
    sub_dict: dict[str, Subroutine],
    type_dict: dict[str, DerivedType],
):
    """
    Function to dump SPEL's output as pickled objects.
    """

    import subprocess as sp

    func_name = "pickle_unit_test"
    cmd = f"{scripts_dir}/git_commit.sh {E3SM_SRCROOT}"
    output = sp.getoutput(cmd)

    if "ERROR" in output:
        print(f"{func_name}::Couldn't find GIT COMMIT\n{output}")
        sys.exit(1)
    output = output.split()
    output[1] = output[1][0:7]
    commit = output[1]

    for mod in mod_dict.values():
        mod.filepath = mod.filepath.replace(E3SM_SRCROOT, "")

    dbfile = open(f"{scripts_dir}/mod_dict-{commit}.pkl", "ab")
    pickle.dump(mod_dict, dbfile)
    dbfile.close()

    for sub in sub_dict.values():
        sub.filepath = sub.filepath.replace(E3SM_SRCROOT, "")

    dbfile = open(f"{scripts_dir}/sub_dict-{commit}.pkl", "ab")
    pickle.dump(sub_dict, dbfile)
    dbfile.close()

    for dtype in type_dict.values():
        dtype.filepath = dtype.filepath.replace(E3SM_SRCROOT, "")

    dbfile = open(f"{scripts_dir}/type_dict-{commit}.pkl", "ab")
    pickle.dump(type_dict, dbfile)
    dbfile.close()


def unpickle_unit_test(commit):
    """
    Function to load SPEL's output from pickled files.
    """
    mod_dict, sub_dict, type_dict = {}, {}, {}
    dbfile = open(f"{scripts_dir}/mod_dict-{commit}.pkl", "rb")
    mod_dict = pickle.load(dbfile)
    dbfile.close()

    for mod in mod_dict.values():
        mod.filepath = E3SM_SRCROOT + mod.filepath

    dbfile = open(f"{scripts_dir}/sub_dict-{commit}.pkl", "rb")
    sub_dict = pickle.load(dbfile)
    dbfile.close()

    for sub in sub_dict.values():
        sub.filepath = E3SM_SRCROOT + sub.filepath

    dbfile = open(f"{scripts_dir}/type_dict-{commit}.pkl", "rb")
    type_dict = pickle.load(dbfile)
    dbfile.close()
    for dtype in type_dict.values():
        dtype.filepath = E3SM_SRCROOT + dtype.filepath

    return mod_dict, sub_dict, type_dict


def export_table_csv(commit: str):
    """ """

    mod_dict: dict[str, FortranModule] = {}
    sub_dict: dict[str, Subroutine] = {}
    type_dict: dict[str, DerivedType] = {}

    mod_dict, sub_dict, type_dict = unpickle_unit_test(commit)

    inst_to_dtype: dict[str, DerivedType] = {}
    for dtype in type_dict.values():
        for inst in dtype.instances:
            inst_to_dtype[inst] = dtype

    inst_to_dtype["bounds"] = type_dict["bounds_type"]

    prefix = django_database
    export_module_usage(mod_dict, prefix)
    export_subroutines(sub_dict, prefix)
    export_subroutine_args(sub_dict, prefix)
    export_sub_call_tree(sub_dict, prefix)
    export_type_insts(type_dict, prefix)
    export_type_defs(type_dict, prefix)
    export_sub_active_dtypes(sub_dict, inst_to_dtype, prefix)
    return


def export_type_defs(type_dict: dict[str, DerivedType], prefix: str):
    field_names = [
        "module",
        "user_type_name",
        "member_type",
        "member_name",
        "dim",
        "bounds",
    ]
    data = {f: [] for f in field_names}
    csv_file = f"{prefix}type_defs.csv"

    def add_row(mod_name, type_name, field_var):
        data["module"].append(mod_name)
        data["user_type_name"].append(type_name)
        data["member_type"].append(field_var.type)
        data["member_name"].append(field_var.name)
        data["dim"].append(field_var.dim)
        data["bounds"].append(field_var.bounds)
        return

    for dtype in type_dict.values():
        type_name = dtype.type_name
        mod = dtype.declaration
        for field_var in dtype.components.values():
            if "%" in field_var.name:
                field_var.name = field_var.name.split("%")[1]
            add_row(mod, type_name, field_var)

    write_dict_to_csv(data, field_names, csv_file)
    return


def export_subroutines(sub_dict: dict[str, Subroutine], prefix: str):
    """ """
    field_names = ["module", "subroutine"]
    data = {f: [] for f in field_names}

    csv_file = f"{prefix}subroutines.csv"
    for sub in sub_dict.values():
        module = sub.module
        sub_name = sub.name
        data["module"].append(module)
        data["subroutine"].append(sub_name)

    write_dict_to_csv(data, field_names, csv_file)
    return


def export_sub_active_dtypes(
    sub_dict: dict[str, Subroutine],
    inst_to_type_dict: dict[str, DerivedType],
    prefix: str,
):
    field_names = [
        "sub_module",
        "subroutine",
        "type_module",
        "inst_type",
        "inst_name",
        "member_type",
        "member_name",
        "status",
    ]
    data = {f: [] for f in field_names}
    csv_file = f"{prefix}active_dtype_vars.csv"

    def add_row(
        mod_name,
        sub_name,
        type_module,
        inst_type,
        inst_name,
        field_var,
        status,
    ):
        data["sub_module"].append(mod_name)
        data["subroutine"].append(sub_name)
        data["type_module"].append(type_module)
        data["inst_type"].append(inst_type)
        data["inst_name"].append(inst_name)
        data["member_type"].append(field_var.type)
        data["member_name"].append(field_var.name)
        data["status"].append(status)
        return

    for sub in sub_dict.values():
        module = sub.module
        sub_name = sub.name
        for dtype_var, stat in sub.elmtype_access_sum.items():
            if "%" not in dtype_var:
                continue
            inst, field = dtype_var.split("%")
            dtype = inst_to_type_dict[inst]
            field_var = dtype.components[field]
            if "%" in field_var.name:
                field_var.name = field_var.name.split("%")[1]
            add_row(
                module,
                sub_name,
                dtype.declaration,
                dtype.type_name,
                inst,
                field_var,
                stat,
            )

    write_dict_to_csv(data, field_names, csv_file)

    return


def export_type_insts(type_dict: dict[str, DerivedType], prefix: str):
    field_names = ["module", "user_type_name", "instance_name"]
    data = {f: [] for f in field_names}
    csv_file = f"{prefix}user_type_instances.csv"

    def add_row(mod_name, type_name, inst_name):
        data["module"].append(mod_name)
        data["user_type_name"].append(type_name)
        data["instance_name"].append(inst_name)
        return

    for dtype in type_dict.values():
        type_name = dtype.type_name
        mod = dtype.declaration
        for inst in dtype.instances:
            add_row(mod, type_name, inst)

    write_dict_to_csv(data, field_names, csv_file)
    return


def export_sub_call_tree(sub_dict: dict[str, Subroutine], prefix: str):
    field_names = ["mod_parent", "parent_subroutine", "mod_child", "child_subroutine"]
    data = {f: [] for f in field_names}

    csv_file = f"{prefix}subroutine_calltree.csv"

    def add_row(mod_parent, parent, mod_child, child):
        data["mod_parent"].append(mod_parent)
        data["parent_subroutine"].append(parent)
        data["mod_child"].append(mod_child)
        data["child_subroutine"].append(child)
        return

    for sub in sub_dict.values():
        parent = sub.name
        mod_p = sub.module
        for child in sub.child_subroutines:
            mod_c = sub_dict[child].module
            add_row(mod_p, parent, mod_c, child)

    write_dict_to_csv(data, field_names, csv_file)
    return


def export_subroutine_args(sub_dict: dict[str, Subroutine], prefix):

    field_names = ["module", "subroutine", "arg_type", "arg_name", "dim"]
    csv_file = f"{prefix}subroutine_args.csv"

    export_dict = {f: [] for f in field_names}

    def add_row(mod_name, sub_name, arg: Variable):
        export_dict["module"].append(mod_name)
        export_dict["subroutine"].append(sub_name)
        export_dict["arg_type"].append(arg.type)
        export_dict["arg_name"].append(arg.name)
        export_dict["dim"].append(arg.dim)
        return

    for sub in sub_dict.values():
        sub_name = sub.name
        module = sub.module
        for arg in sub.arguments.values():
            add_row(module, sub_name, arg)

    write_dict_to_csv(export_dict, field_names, csv_file)
    return


def export_module_usage(mod_dict: dict[str, FortranModule], prefix):
    """
    Function creates csv file to update Modules/ModuleDependency Tables
    """

    field_names = ["module_name", "dep_module_name", "object_used"]
    csv_file = f"{prefix}module_deps.csv"

    export_dict = {f: [] for f in field_names}

    def add_row(mod_name, dep_name, obj):
        export_dict["module_name"].append(mod_name)
        export_dict["dep_module_name"].append(dep_name)
        export_dict["object_used"].append(obj)
        return

    for mod in mod_dict.values():
        mod_name = mod.name
        for dep_mod, usage in mod.modules.items():
            if usage.all:
                add_row(mod_name, dep_mod, "all")
            else:
                for ptrobj in usage.clause_vars:
                    add_row(mod_name, dep_mod, ptrobj.obj)

    write_dict_to_csv(
        export_dict,
        field_names,
        csv_file,
    )

    return


def write_dict_to_csv(data, fieldnames, csv_file):
    print(f"writing to {csv_file}")

    df = pd.DataFrame(data)
    print(df)
    df.to_csv(f"{csv_file}", index=False)
    print(f"CSV file '{csv_file}' has been created.")
    return


if __name__ == "__main__":

    export_table_csv()
