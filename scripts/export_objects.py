import json
import os
import pickle
import sys

import pandas as pd

from scripts.analyze_subroutines import Subroutine
from scripts.config import E3SM_SRCROOT, django_database, scripts_dir
from scripts.DerivedType import DerivedType
from scripts.fortran_modules import FortranModule
from scripts.fortran_parser.spel_ast import expr_to_json
from scripts.types import CallBinding, ReadWrite, Scope
from scripts.utilityFunctions import Variable

TypeDict = dict[str, DerivedType]
SubDict = dict[str, Subroutine]
ModDict = dict[str, FortranModule]


def _coerce_to_fort(val):
    if isinstance(val, bool):
        return ".true." if val else ".false."
    elif isinstance(val, str):
        return val.replace("True", ".true.").replace("False", ".false.")
    return val


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


def unpickle_unit_test(commit=None) -> tuple[ModDict, SubDict, TypeDict]:
    """
    Function to load SPEL's output from pickled files.
    """
    if not commit:
        import subprocess as sp

        fn = sp.getoutput(f"ls -t {scripts_dir}/*pkl | head -n 1")
        commit = fn.split("-")[1].split(".")[0]
        print(fn, commit)

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
    if not os.path.isdir(django_database):
        os.system(f"mkdir {django_database}")

    export_modules(mod_dict, prefix)
    export_module_usage(mod_dict, prefix)
    export_subroutines(sub_dict, prefix)
    export_subroutine_args(sub_dict, prefix)
    export_sub_call_tree(sub_dict, prefix)
    export_type_defs(type_dict, prefix)
    export_type_insts(type_dict, prefix)
    export_sub_active_dtypes(sub_dict, inst_to_dtype, prefix)
    export_intrinsic_globals(sub_dict, prefix)
    export_nml_ifs(sub_dict, prefix)
    export_cascade_ifs(sub_dict, prefix)
    export_arg_access_by_ln(sub_dict, prefix)
    export_call_binding(sub_dict,type_dict, prefix)
    export_propagated_by_ln(sub_dict, type_dict, prefix)
    return


def export_intrinsic_globals(sub_dict: dict[str, Subroutine], prefix: str):
    field_names = [
        "module",
        "var_name",
        "var_type",
        "dim",
        "bounds",
        "value",
        "sub_module",
        "sub_name",
    ]
    data = {f: [] for f in field_names}
    csv_file = f"{prefix}intrinsic_globals.csv"

    def add_row(var: Variable, sub_mod: str, sub_name: str):
        data["module"].append(var.declaration)
        data["var_name"].append(var.name)
        data["var_type"].append(var.type)
        data["dim"].append(var.dim)
        data["bounds"].append(var.bounds)
        data["value"].append(
            _coerce_to_fort(var.default_value.replace("(", "").replace(")", ""))
        )
        data["sub_module"].append(sub_mod)
        data["sub_name"].append(sub_name)

    for sub in sub_dict.values():
        for var in sub.active_global_vars.values():
            add_row(var, sub.module, sub.name)

    write_dict_to_csv(data, field_names, csv_file)
    return


def export_modules(mod_dict: ModDict, prefix: str):
    """
    exports all the modules
    """
    field_names = ["module"]
    data = {f: [] for f in field_names}
    csv_file = f"{prefix}modules.csv"
    data["module"].extend(list(mod_dict.keys()))
    write_dict_to_csv(data, field_names, csv_file)
    return


def export_cascade_ifs(sub_dict: dict[str, Subroutine], prefix: str):
    field_names = [
        "sub_module",
        "subroutine",
        "nml_var_name",
        "nml_var_module",
        "if_start",
        "if_end",
        "if_cond",
        "cascade_var",
    ]
    data = {f: [] for f in field_names}
    csv_file = f"{prefix}ifs_cascade.csv"

    def add_row(
        sub_module,
        subroutine,
        nml_var_name,
        nml_var_module,
        if_start,
        if_end,
        if_cond,
        cascade_var,
    ):
        data["sub_module"].append(sub_module)
        data["subroutine"].append(subroutine)
        data["nml_var_name"].append(nml_var_name)
        data["nml_var_module"].append(nml_var_module)
        data["if_start"].append(if_start)
        data["if_end"].append(if_end)
        data["if_cond"].append(if_cond)
        data["cascade_var"].append(cascade_var)

    for sub in sub_dict.values():
        for flatif in sub.flat_ifs:
            for cascade_var, dep in flatif.nml_cascades.items():
                nml_module, nml_var = dep.trigger.split("::")
                add_row(
                    sub.module,
                    sub.name,
                    nml_var,
                    nml_module,
                    flatif.start_ln,
                    flatif.end_ln,
                    json.dumps(flatif.condition.to_dict()),
                    cascade_var,
                )

    write_dict_to_csv(data, field_names, csv_file)
    return


def export_nml_ifs(sub_dict: dict[str, Subroutine], prefix: str):
    field_names = [
        "sub_module",
        "subroutine",
        "nml_var_name",
        "nml_var_type",
        "nml_var_dim",
        "nml_var_bounds",
        "nml_var_module",
        "if_start",
        "if_end",
        "if_cond",
        "value",
    ]
    data = {f: [] for f in field_names}
    csv_file = f"{prefix}nml_ifs.csv"

    def add_row(
        modname: str,
        sub_name: str,
        nml_var: Variable,
        if_start: int,
        if_end: int,
        cond: dict,
    ):
        data["sub_module"].append(modname)
        data["subroutine"].append(sub_name)
        data["nml_var_name"].append(nml_var.name)
        data["nml_var_type"].append(nml_var.type)
        data["nml_var_dim"].append(nml_var.dim)
        data["nml_var_bounds"].append(nml_var.bounds)
        data["nml_var_module"].append(nml_var.declaration)
        data["if_start"].append(if_start)
        data["if_end"].append(if_end)
        data["if_cond"].append(json.dumps(cond))
        data["value"].append(
            _coerce_to_fort(nml_var.default_value.replace("(", "").replace(")", ""))
        )

    for sub in sub_dict.values():
        for ifnode in sub.flat_ifs:
            for nml_var in ifnode.nml_vars.values():
                add_row(
                    sub.module,
                    sub.name,
                    nml_var.variable,
                    ifnode.start_ln,
                    ifnode.end_ln,
                    ifnode.condition.to_dict(),
                )
    write_dict_to_csv(data, field_names, csv_file)
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
        "inst_mod",
        "inst_name",
        "member_type",
        "member_name",
        "status",
        "ln",
    ]
    data = {f: [] for f in field_names}
    csv_file = f"{prefix}active_dtype_vars.csv"

    def add_row(
        mod_name,
        sub_name,
        type_module,
        inst_type,
        inst_mod,
        inst_name,
        field_var,
        status,
        ln,
    ):
        data["sub_module"].append(mod_name)
        data["subroutine"].append(sub_name)
        data["type_module"].append(type_module)
        data["inst_type"].append(inst_type)
        data["inst_mod"].append(inst_mod)
        data["inst_name"].append(inst_name)
        data["member_type"].append(field_var.type)
        data["member_name"].append(field_var.name)
        data["status"].append(status)
        data["ln"].append(ln)
        return

    import pprint

    for sub in sub_dict.values():
        module = sub.module
        sub_name = sub.name
        for dtype_var, rw_list in sub.elmtype_access_by_ln.items():
            if "%" not in dtype_var or dtype_var.count("%") > 1:
                continue
            inst, field = dtype_var.split("%")
            if inst not in inst_to_type_dict:
                continue
            dtype = inst_to_type_dict[inst]
            inst_var = dtype.instances[inst]
            field_var = dtype.components[field]
            if "%" in field_var.name:
                field_var.name = field_var.name.split("%")[1]

            for rw in rw_list:
                stat = rw.status
                ln = rw.ln
                add_row(
                    module,
                    sub_name,
                    dtype.declaration,
                    dtype.type_name,
                    inst_var.declaration,
                    inst_var.name,
                    field_var,
                    stat,
                    ln,
                )

    write_dict_to_csv(data, field_names, csv_file)

    return


def export_type_insts(type_dict: dict[str, DerivedType], prefix: str):
    field_names = ["module", "type_mod", "user_type_name", "instance_name"]
    data = {f: [] for f in field_names}
    csv_file = f"{prefix}user_type_instances.csv"

    def add_row(mod_name, type_mod, type_name, inst_name):
        data["module"].append(mod_name)
        data["user_type_name"].append(type_name)
        data["type_mod"].append(type_mod)
        data["instance_name"].append(inst_name)
        return

    for dtype in type_dict.values():
        type_name = dtype.type_name
        type_mod = dtype.declaration
        for inst in dtype.instances.values():
            if inst.name == "this" or not dtype.components:
                continue
            add_row(inst.declaration, type_mod, type_name, inst.name)

    write_dict_to_csv(data, field_names, csv_file)
    return


def export_sub_call_tree(sub_dict: dict[str, Subroutine], prefix: str):
    field_names = [
        "mod_parent",
        "parent_subroutine",
        "mod_child",
        "child_subroutine",
        "ln",
        "args",
    ]
    data = {f: [] for f in field_names}

    csv_file = f"{prefix}subroutine_calltree.csv"

    def add_row(mod_parent, parent, mod_child, child, ln, args):
        data["mod_parent"].append(mod_parent)
        data["parent_subroutine"].append(parent)
        data["mod_child"].append(mod_child)
        data["child_subroutine"].append(child)
        data["ln"].append(ln)
        data["args"].append(args)
        return

    for sub in sub_dict.values():
        parent = sub.name
        mod_p = sub.module
        for ln, call_desc in sub.sub_call_desc.items():
            child_sub = sub_dict[call_desc.fn]
            child = child_sub.name
            mod_c = child_sub.module
            args = call_desc.args_str
            add_row(mod_p, parent, mod_c, child, ln, args)

    write_dict_to_csv(data, field_names, csv_file)
    return


def export_arg_access_by_ln(sub_dict: dict[str, Subroutine], prefix):
    field_names = ["module", "subroutine", "dummy_arg", "member_path", "ln", "status"]
    data = {f: [] for f in field_names}
    csv_file = f"{prefix}arg_access_by_ln.csv"

    def add_row(module, subname, arg, member_path, ln, status):
        data["module"].append(module)
        data["subroutine"].append(subname)
        data["dummy_arg"].append(arg)
        data["member_path"].append(member_path)
        data["ln"].append(ln)
        data["status"].append(status)

    for sub in sub_dict.values():
        subname = sub.name
        module = sub.module
        for arg, rws in sub.arg_access_by_ln.items():
            if "%" in arg:
                argname, member_path = arg.split("%", 1)
            else:
                argname = arg
                member_path = ""
            for stat in rws:
                add_row(module, subname, argname, member_path, stat.ln, stat.status)
    write_dict_to_csv(data, field_names, csv_file)
    return


def export_propagated_by_ln(
    sub_dict: dict[str, Subroutine],
    type_dict: dict[str, DerivedType],
    prefix: str,
):
    field_names = [
        "parent_module",
        "parent_sub",
        "child_module",
        "child_sub",
        "call_ln",
        "dummy_arg",
        "nested_level",
        "bound_member",
        "scope",
        "var_name",
        "member_path",
        "type_name",
        "type_mod",
        "inst_mod",
        "status",
        "rw_ln",
    ]

    data = {f: [] for f in field_names}
    csv_file = f"{prefix}propagated_access.csv"

    def add_row(
        parent_module,
        parent_sub,
        child_module,
        child_sub,
        call_ln,
        dummy,
        nested_level,
        bound_member,
        status,
        rw_ln,
        scope,
        var_name,
        member_path,
        type_name,
        type_mod,
        inst_mod,
    ):
        data["parent_module"].append(parent_module)
        data["parent_sub"].append(parent_sub)
        data["child_module"].append(child_module)
        data["child_sub"].append(child_sub)
        data["call_ln"].append(call_ln)
        data["dummy_arg"].append(dummy)
        data["nested_level"].append(nested_level)
        data["bound_member"].append(bound_member)
        data["status"].append(status)
        data["rw_ln"].append(rw_ln)
        data["scope"].append(scope)
        data["var_name"].append(var_name)
        data["member_path"].append(member_path)
        data["type_name"].append(type_name)
        data["type_mod"].append(type_mod)
        data["inst_mod"].append(inst_mod)

    for sub in sub_dict.values():
        for var, prop_list in sub.propagated_access_by_ln.items():
            if "%" in var:
                varname, member_path = var.split("%", 1)
            else:
                varname = var
                member_path = ""

            for prop in prop_list:
                p_mod, p_sub = prop.tag.caller.split("::")
                c_mod, c_sub = prop.tag.callee.split("::")
                parent_sub = sub_dict[f"{p_mod}::{p_sub}"]
                if member_path and prop.scope == Scope.ELMTYPE:
                    var = parent_sub.dtype_vars.get(varname)
                    type_name = var.type
                    type_mod = type_dict[type_name].declaration
                    inst_mod = var.declaration
                else:
                    type_name = ""
                    type_mod = ""
                    inst_mod = ""

                call_ln = prop.tag.call_ln
                scope = prop.scope.name
                dummy = prop.dummy
                nested_level = prop.binding.nested_level
                for rw in prop.rw_statuses:
                    add_row(
                        parent_module=p_mod,
                        parent_sub=p_sub,
                        child_module=c_mod,
                        child_sub=c_sub,
                        call_ln=call_ln,
                        dummy=dummy,
                        nested_level=nested_level,
                        bound_member=prop.binding.member_path,
                        status=rw.status,
                        rw_ln=rw.ln,
                        scope=scope,
                        var_name=varname,
                        member_path=member_path,
                        type_name=type_name,
                        type_mod=type_mod,
                        inst_mod=inst_mod,
                    )

    write_dict_to_csv(data, field_names, csv_file)
    return


def export_call_binding(
    sub_dict: dict[str, Subroutine],
    type_dict: dict[str, DerivedType],
    prefix: str,
):
    field_names = [
        "parent_module",
        "parent_sub",
        "child_module",
        "child_sub",
        "dummy_arg",
        "scope",
        "var_mod",
        "var_name",
        "member_path",
        "type_name",
        "type_mod",
        "nested_level",
        "ln",
    ]
    data = {f: [] for f in field_names}
    csv_file = f"{prefix}call_bindings.csv"

    def add_row(
        parent_module,
        parent_sub,
        child_module,
        child_sub,
        dummy_arg,
        nested_level,
        ln,
        scope,
        var_mod,
        var_name,
        member_path,
        type_name,
        type_mod,
    ):
        data["parent_module"].append(parent_module)
        data["parent_sub"].append(parent_sub)
        data["child_module"].append(child_module)
        data["child_sub"].append(child_sub)
        data["dummy_arg"].append(dummy_arg)
        data["ln"].append(ln)
        data["scope"].append(scope)
        data["var_name"].append(var_name)
        data["var_mod"].append(var_mod)
        data["member_path"].append(member_path)
        data["nested_level"].append(nested_level)
        data["type_name"].append(type_name)
        data["type_mod"].append(type_mod)

    for sub in sub_dict.values():
        p_subname = sub.name
        p_module = sub.module
        for call_desc in sub.sub_call_desc.values():
            child_sub = sub_dict[call_desc.fn]
            child_sub_name = child_sub.name
            if child_sub.library:
                continue
            child_module = child_sub.module
            bindings = call_desc.export_bindings()
            for bind in bindings:
                print(bind)
                dummy_arg = child_sub.dummy_args_list[bind.argn]
                var_name = bind.var_name
                member_path = bind.member_path
                if bind.scope == Scope.ELMTYPE:
                    if var_name in sub.active_global_vars:
                        scope = "INTRINSIC"
                        var_mod = sub.active_global_vars[var_name].declaration
                        type_name = ""
                        type_mod = ""
                    else:
                        scope = "ELMTYPE"
                        var = sub.dtype_vars[var_name]
                        var_mod = sub.dtype_vars[var_name].declaration
                        type_name = var.type
                        type_mod = type_dict[type_name].declaration
                elif bind.scope == Scope.ARG:
                    scope = "ARG"
                    var_mod = ""
                    type_name = ""
                    type_mod = ""
                elif bind.scope == Scope.LOCAL:
                    scope = "LOCAL"
                    var_mod = ""
                    type_name = ""
                    type_mod = ""
                else:
                    sys.exit(f"Unknown binding: {bind}")

                add_row(
                    parent_module=p_module,
                    parent_sub=p_subname,
                    child_module=child_module,
                    child_sub=child_sub_name,
                    dummy_arg=dummy_arg,
                    ln=call_desc.lpair.ln,
                    var_mod=var_mod,
                    var_name=var_name,
                    member_path=member_path,
                    type_name=type_name,
                    type_mod=type_mod,
                    nested_level=bind.nested_level,
                    scope=scope,
                )
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


# def change_status_on_direct_access(sub_dict: dict[str,Subroutine]):
#     for sub in sub_dict.values():
#         sub.elmtype_access_by_ln = _check_accesses(sub.elmtype_access_by_ln)
#
#     return
#
# def _check_accesses(access_dict: dict[str,list[ReadWrite]]):
#
