from __future__ import annotations

import re
import subprocess as sp
import sys
from pprint import pprint
from typing import TYPE_CHECKING, Dict, Optional

import scripts.dynamic_globals as dg
from scripts.fortran_parser.lexer import Lexer
from scripts.fortran_parser.spel_ast import (
    ProcedureStatement,
    Statement,
    TypeDef,
    VariableDecl,
)
from scripts.fortran_parser.spel_parser import Parser
from scripts.types import LineTuple, LogicalLineIterator

if TYPE_CHECKING:
    from scripts.fortran_modules import FortranModule
    from scripts.analyze_subroutines import Subroutine

from scripts.config import ELM_SRC, _bc, _no_colors
from scripts.fortran_modules import get_module_name_from_file
from scripts.utilityFunctions import (
    Variable,
    create_var_from_decl,
    line_unwrapper,
    parse_line_for_variables,
)

Alias = str
Proc = str
## arrow and tab are strings for writing files or printing readable output
arrow = "|--->"
tab = " " * 2


class DerivedType(object):
    """
    Class to represent Fortran user-derived type
    """

    def __init__(
        self,
        type_name: str,
        vmod: str,
        fpath: Optional[str] = None,
    ):
        self.type_name = type_name
        if fpath:
            self.filepath: str = fpath
        else:
            self.filepath: str = ""
            cmd = rf'find {ELM_SRC}  \( -path "*external_models*" \) -prune -o  -name "{vmod}.F90"'
            output = sp.getoutput(cmd)
            if not output:
                sys.exit(f"Couldn't locate file {vmod}")
            else:
                output = output.split("\n")
                for el in output:
                    if "external_models" not in el:
                        self.filepath = el

        if not self.filepath:
            sys.exit(
                f"Couldn't find file for {type_name}\ncmd:{cmd}\n" f"output: {output}\n"
            )

        self.declaration = vmod
        self.components: dict[str, Variable] = {}
        self.instances: dict[str, Variable] = {}
        self.analyzed: bool = False
        self.active: bool = False
        self.init_sub_name: Optional[str] = ""
        self.procedures: dict[Alias, Proc] = {}

        self.init_sub_ptr: Optional[Subroutine] = None

    def __repr__(self):
        return f"DerivedType({self.type_name})"

    def find_instances(self, mod_dict: dict[str, FortranModule]):
        # Find all instances of the derived type:
        grep = "grep -rin --exclude-dir=external_models/"
        cmd = rf'{grep} "type\s*(\s*{self.type_name}\s*)" {ELM_SRC}* | grep "::" | grep -v "intent"'
        output = sp.getoutput(cmd)

        regex_paren = re.compile(r"\((.+)\)")
        #
        # Each element in output should have the format:
        # <filepath> : <ln> : type(<type_name>) :: <instance_name>
        instance_list = {}
        if not output:
            return

        output = output.split("\n")
        for el in output:
            inst_name = el.split("::")[-1]
            inst_name = inst_name.split("!")[0].strip().lower()

            filepath = el.split(":")[0].strip()
            ln = int(el.split(":")[1].strip())
            _, module_name = get_module_name_from_file(filepath)
            if module_name not in mod_dict:
                end_of_head_ln = find_module_head_end(module_name, filepath)
            else:
                fort_mod = mod_dict[module_name]
                end_of_head_ln = fort_mod.end_of_head_ln
            if ln > end_of_head_ln:
                continue

            dim = inst_name.count(":")
            inst_name = regex_paren.sub("", inst_name)

            inst_var = Variable(
                type=self.type_name,
                name=inst_name,
                subgrid="?",
                ln=ln,
                dim=dim,
                declaration=module_name,
            )
            if inst_var.name not in instance_list:
                instance_list[inst_var.name] = inst_var

        self.instances = instance_list.copy()

    def print_derived_type(self, ofile=sys.stdout, long=False) -> None:
        """
        Function to print info on the user derived type
        """
        if ofile == sys.stdout:
            hl = _bc
        else:
            hl = _no_colors

        ofile.write(hl.HEADER + "Derived Type:" + self.type_name + "\n" + hl.ENDC)
        base_fn = "/".join(self.filepath.split("/")[-2:])
        ofile.write(hl.HEADER + "from Mod: " + base_fn + "\n" + hl.ENDC)
        for v in self.instances.values():
            ofile.write(hl.OKBLUE + f"{v.type} {v.name} {v.declaration}\n" + hl.ENDC)
        ofile.write(hl.WARNING + f"Initialized in {self.init_sub_name} \n" + hl.ENDC)
        if self.procedures:
            ofile.write(hl.OKGREEN + f"Type Procedures:\n")
            for alias, proc in self.procedures.items():
                if alias != proc:
                    ofile.write(f"{alias} => {proc}\n")
                else:
                    ofile.write(f"{alias}\n")
            ofile.write(hl.ENDC)
        if long:
            ofile.write("w/ components:\n")
            for field_var in self.components.values():
                status = field_var.active
                var = field_var
                if var.dim > 0:
                    bounds = field_var.bounds
                else:
                    bounds = ""
                if not var.pointer:
                    str_ = f"  {status} {var.type} {var.name} {bounds} {str(var.dim)}-D"
                else:
                    targets = "|".join(var.pointer)
                    str_ = f"  {status} {var.type} {var.name} => {var.pointer}"
                ofile.write(str_ + "\n")
        return None

    def manual_deep_copy(self, ofile=sys.stdout):
        """
        Function that generates pragmas for manual deepcopy of members
        """
        chunksize = 3
        tabs = " " * 3
        depth = 1
        for inst in self.instances.values():
            inst_name = inst.name
            ofile.write(tabs * depth + f"!$acc enter data copyin({inst_name})\n")
            if inst.dim == 1:
                ofile.write(tabs * depth + f"N = size({inst.name})\n")
                ofile.write(tabs * depth + "do i = 1, N\n")
                inst_name = inst.name + "(i)"
                depth += 1
            elif inst.dim > 1:
                print("Error: multi-dimensional Array of Structs found: ", inst)
                sys.exit(1)
            ofile.write(tabs * depth + "!$acc enter data copyin(&\n")
            for num, member in enumerate(self.components.values()):
                dim_string = ""
                if member.dim > 0:
                    dim_li = [":" for i in range(0, member.dim)]
                    dim_string = ",".join(dim_li)
                    dim_string = f"({dim_string})"

                name = inst_name + "%" + member.name + dim_string
                ofile.write(tabs * depth + f"!$acc& {name}")
                final_num = bool(num == len(self.components.keys()) - 1)
                if not final_num:
                    ofile.write(",&\n")
                else:
                    ofile.write(")")
            ofile.write("\n")
            depth -= 1
            if inst.dim == 1:
                ofile.write(tabs * depth + "end do\n")

        return None


def get_component(
    instance_dict: dict[str, DerivedType], dtype_field: str
) -> Optional[Variable]:
    """
    Function that looks up inst%field in the instance dict.
    """
    regex_paren = re.compile(r"\((.+)\)")  # for removing array of struct index
    if dtype_field.count("%") != 1:
        print(f"Error {dtype_field}")
    inst_name, field = dtype_field.split("%")
    inst_name = regex_paren.sub("", inst_name)
    dtype = instance_dict.get(inst_name, None)
    if not dtype:
        print(f"Error: {inst_name} not known. {dtype_field}")
        return None
    if field in dtype.components:
        var: Variable = dtype.components[field].copy()
        return var
    else:
        if field not in dtype.procedures:
            print(f"Procedures for {dtype.type_name}: {dtype.procedures}")
            print(f"Error- Couldn't categorize {dtype_field}")
        return None


def expand_dtype(
    dtype_vars: list[Variable], type_dict: dict[str, DerivedType]
) -> dict[str, Variable]:
    """Function to take a dtype and create a dict with a key for each var%field"""

    def adj_var_name(var: Variable, inst_var: Variable):
        dim_str = "(index)" if inst_var.dim > 0 else ""
        new_var = var.copy()
        new_var.name = f"{ inst_var.name }{dim_str}%{var.name}"
        return new_var

    result: dict[str, Variable] = {}
    for dtype_var in dtype_vars:
        if dtype_var.type not in type_dict:
            continue
        dtype = type_dict[dtype_var.type]
        fields = dtype.components.values()
        temp: dict[str, Variable] = {
            f"{dtype_var.name}%{field.name}": adj_var_name(field, dtype_var)
            for field in fields
        }
        result.update(temp)
    return result


def find_module_head_end(mod: str, file_path: str) -> int:
    if mod in dg.map_module_head:
        return dg.map_module_head[mod]
    in_type = False
    type_start = re.compile(r"^(type\s+)(?!\()", re.IGNORECASE)
    type_end = re.compile(r"\bend\s+type\b", re.IGNORECASE)
    contains = re.compile(r"\bcontains\b", re.IGNORECASE)
    end_head_ln = -1
    with open(file_path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip().lower()
            end_head_ln = lineno
            # If entering a derived type declaration, set the flag.
            # If we're inside a type and see the end of it, clear the flag.
            if type_start.search(line):
                in_type = True
            if in_type and type_end.search(line):
                in_type = False
            if not in_type and contains.search(line):
                break

    if end_head_ln == -1:
        print("Error -- couldn't iterate through file", file_path)
        sys.exit(1)
    return end_head_ln


TypeDict = dict[str, DerivedType]


def get_fields(typedef: TypeDef) -> dict[str, Variable]:
    components: dict[str, Variable] = {}
    for vardecl in typedef.members.statements:
        if isinstance(vardecl, VariableDecl):
            vars = create_var_from_decl(vardecl)
            d_vars = {v.name: v for v in vars}
            components.update(d_vars)
    return components


def get_type_procedures(typedef: TypeDef) -> dict[Alias, Proc]:
    procs: dict[Alias, Proc] = {}
    for stmt in typedef.methods.statements:
        if isinstance(stmt, ProcedureStatement):
            alias = stmt.alias if stmt.alias else stmt.name
            name = stmt.name
            procs[alias] = name
    return procs


def parse_derived_type_definition(
    lines: list[LineTuple],
    mod_name: str,
    ifile: str,
) -> TypeDict:
    regex_start = re.compile(r"^\s*type\s*(?!\()", re.IGNORECASE)
    regex_skip = re.compile(r"^type\s*\(")
    regex_end = re.compile(r"^\s*end\s*type")

    dtypes: TypeDict = {}
    statements: list[Statement] = []
    line_it = LogicalLineIterator(lines=lines)
    for full_line, _ in line_it:
        m_start = regex_start.search(full_line)
        m_skip = regex_skip.search(full_line)
        if m_start and not m_skip:
            block, _ = line_it.consume_until(regex_end, None)
            blk_iter = LogicalLineIterator(lines=block)
            lexer = Lexer(blk_iter)
            parser = Parser(lexer, logger=f"Looking4Types-{mod_name}")
            program = parser.parse_program()
            if not program.statements:
                str_ = "\n".join([l.line for l in block])
                print(f"Failed to parse {str_}")
                sys.exit(1)
            statements.extend(program.statements)
    # TypeDef
    for typedef in statements:
        assert isinstance(typedef, TypeDef), f"Unexpect statement type {type(typedef)}"
        dtype = DerivedType(type_name=typedef.name, vmod=mod_name, fpath=ifile)
        dtype.components = get_fields(typedef)
        if typedef.methods:
            dtype.procedures = get_type_procedures(typedef)
        dtypes[dtype.type_name] = dtype

    return dtypes
