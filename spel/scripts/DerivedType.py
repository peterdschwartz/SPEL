from __future__ import annotations

import logging
import re
import subprocess as sp
import sys
from collections import defaultdict
from dataclasses import field
from pathlib import Path
from pprint import pformat, pprint
from typing import TYPE_CHECKING, Dict, Optional

import spel.scripts.dynamic_globals as dg
from spel.scripts.fortran_parser.lexer import Lexer
from spel.scripts.fortran_parser.spel_ast import (
    ProcedureStatement,
    Statement,
    TypeDef,
    VariableDecl,
)
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.types import LineTuple, LogicalLineIterator

if TYPE_CHECKING:
    from spel.scripts.fortran_modules import FortranModule
    from spel.scripts.analyze_subroutines import Subroutine

from spel.scripts.config import ELM_SRC, _bc, _no_colors
from spel.scripts.fortran_modules import get_module_name_from_file
from spel.scripts.utilityFunctions import (
    Variable,
    create_var_from_decl,
    parse_variable_decl,
    retrieve_lines,
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
            self.filepath: Path = Path(fpath)
        else:
            cmd = rf'find {ELM_SRC}  \( -path "*external_models*" \) -prune -o  -name "{vmod}.F90"'
            output = sp.getoutput(cmd)
            if not output:
                sys.exit(f"Couldn't locate file {vmod}")
            else:
                output = output.split("\n")
                for el in output:
                    if "external_models" not in el:
                        self.filepath = Path(el)

        assert self.filepath.exists(), (
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
        # extends(parent): resolve_inheritance adds the parent's components and bindings
        self.parent: Optional[str] = None

    def __repr__(self):
        return f"DerivedType({self.type_name})"

    def find_instances(self, mod_dict: dict[str, FortranModule]):
        # Find all instances of the derived type:
        grep = "grep -rin --exclude-dir=external_models/"
        cmd = rf'{grep} "type\s*(\s*{self.type_name}\s*)" {ELM_SRC}* | grep "::" | grep -iv intent'
        output = sp.getoutput(cmd)

        instance_list = {}
        if not output:
            return

        # <filepath> : <ln> : type(<type_name>) :: <instance_name>
        output = output.split("\n")
        split_output = list(map(lambda x: x.split(":", 2), output))

        def mod_ln_helper(modname, fp):
            fort_mod = mod_dict.get(modname, None)
            if fort_mod:
                return fort_mod.end_of_head_ln
            else:
                if modname == "elm_instmod":
                    return find_module_head_end(modname, fp)
                else:
                    return -1

        files = [el[0] for el in split_output]
        lns = [int(el[1]) - 1 for el in split_output]

        io_list = defaultdict(list)
        for f, ln in zip(files, lns):
            _, mod = get_module_name_from_file(f)
            if not mod:  # not a module (e.g. a test program): no instances here
                continue
            if ln < mod_ln_helper(mod, f):
                io_list[f].append(ln)

        lines: list[LineTuple] = []
        mod_for_lines = []
        for f, lns in io_list.items():
            lines.extend(retrieve_lines(f, lns))
            _, mod_name = get_module_name_from_file(f)
            mod_for_lines.extend([mod_name for i in lns])

        assert len(mod_for_lines) == len(
            lines
        ), "Error - Mismatch of modules for each line"
        if not lines:
            return
        line_it = LogicalLineIterator(lines=lines, log_name="find_instances")
        lexer = Lexer(line_it)
        parser = Parser(lexer, "find_instances")
        program = parser.parse_program()

        for i, stmt in enumerate(program.statements):
            assert isinstance(
                stmt, VariableDecl
            ), f"Unexpected Statement: {stmt} from:\n files: {files}\n{pformat(lines)}"
            variables = create_var_from_decl(stmt)
            mod_name = mod_for_lines[i]
            for inst_var in variables:
                if inst_var.name not in instance_list:
                    if inst_var.name == "this":
                        print(pformat(lines))
                        sys.exit(1)
                    inst_var.declaration = mod_name
                    instance_list[inst_var.name] = inst_var

        self.instances = instance_list.copy()

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


from spel.scripts.allocations import rank  # noqa: E402,F401  (re-exported)


def get_component(
    instance_dict: dict[str, DerivedType],
    inst_name: str,
    field: str,
) -> Optional[Variable]:
    """
    Function that looks up inst%field in the instance dict.
    """
    dtype = instance_dict.get(inst_name, None)
    if not dtype:
        print(f"Error: {inst_name} not known. {field}")
        return None
    if "%" in field:
        # inst%core%leaf: the longest component path, else its first component
        field = component_path(dtype.components, field.split("%")) or field.split("%")[0]
    if field in dtype.components:
        var: Variable = dtype.components[field].copy()
        return var
    else:
        if field not in dtype.procedures:
            print(f"Procedures for {dtype.type_name}: {dtype.procedures.items()}")
            print(f"Error- Couldn't categorize {field}")
            sys.exit(10)
        return None


def expand_dtype(
    dtype_vars: list[Variable],
    type_dict: dict[str, DerivedType],
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


def type_parent(typedef: TypeDef) -> Optional[str]:
    """The `extends(parent)` attribute of a type definition, if any."""
    for attr in typedef.attrs.attrs if typedef.attrs else []:
        m = re.fullmatch(r"\s*extends\s*\(\s*(\w+)\s*\)\s*", str(attr), re.IGNORECASE)
        if m:
            return m.group(1).lower()
    return None


def resolve_inheritance(type_dict: dict[str, "DerivedType"]) -> None:
    """
    Give each extended type its parent's components (copies: each type finds its
    own allocation bounds) and type-bound procedures, its own taking precedence.
    """
    done: set[str] = set()

    def resolve(dtype: DerivedType, seen: tuple[str, ...] = ()) -> None:
        if dtype.type_name in done or dtype.parent is None:
            done.add(dtype.type_name)
            return
        parent = type_dict.get(dtype.parent)
        if parent is None:
            # the parent lives in a module the analysis skips (dyn_file_type extends PIO's
            # file_desc_t): keep the type's own components, nothing to inherit here
            logging.getLogger("SPEL").warning(
                f"{dtype.type_name} extends {dtype.parent}, which is not analyzed: no inherited components"
            )
            done.add(dtype.type_name)
            return
        assert dtype.type_name not in seen, f"cyclic extends: {seen}"
        resolve(parent, seen + (dtype.type_name,))
        inherited = {n: v.copy() for n, v in parent.components.items()}
        dtype.components = {**inherited, **dtype.components}
        dtype.procedures = {**parent.procedures, **dtype.procedures}
        done.add(dtype.type_name)

    for dtype in list(type_dict.values()):
        resolve(dtype)


def resolve_nested(type_dict: dict[str, "DerivedType"]) -> None:
    """
    Give each type the components of its data-only derived-type components
    (`type(base) :: core`, a type without type-bound procedures) as
    ``core%leaf`` entries (copies), so `inst%core%leaf` resolves like any
    `inst%field`. The component itself (``core``) stays. Types with bindings
    (e.g. annual_flux_dribbler_type) are left alone: their components are
    reached through their procedures.
    """
    done: set[str] = set()

    def resolve(dtype: DerivedType, seen: tuple[str, ...] = ()) -> None:
        if dtype.type_name in done:
            return
        assert dtype.type_name not in seen, f"recursive component types: {seen}"
        nested = {}
        for name, comp in dtype.components.items():
            inner = type_dict.get(comp.type)
            if inner is None or inner.procedures or comp.pointer or comp.ptrscalar or comp.dim > 0:
                continue
            resolve(inner, seen + (dtype.type_name,))
            for iname, ivar in inner.components.items():
                v = ivar.copy()
                v.name = f"{name}%{iname}"
                v.nested = True  # a path (core%leaf), not inst%field
                nested[v.name] = v
        dtype.components.update(nested)
        done.add(dtype.type_name)

    for dtype in list(type_dict.values()):
        resolve(dtype)


def field_name(var) -> str:
    """A component's name as written after `inst%`: its path for flattened components (core%leaf)."""
    if getattr(var, "nested", False):
        return var.name
    return var.name.split("%")[-1]


def component_path(components: dict, parts: list[str]) -> Optional[str]:
    """The longest `a%b%...` prefix of `parts` that is a component (``core%leaf``), else None."""
    for k in range(len(parts), 0, -1):
        path = "%".join(parts[:k])
        if path in components:
            return path
    return None


def get_type_procedures(modname: str, typedef: TypeDef) -> dict[Alias, Proc]:
    procs: dict[Alias, Proc] = {}
    for stmt in typedef.methods.statements:
        if isinstance(stmt, ProcedureStatement):
            if stmt.token_literal().lower() == "generic":
                continue  # binds other bindings, not routines
            alias = stmt.alias if stmt.alias else stmt.name
            name = f"{modname}::{stmt.name}"
            procs[f"{alias}"] = name
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
    for fline in line_it:
        full_line = fline.line
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
        dtype.parent = type_parent(typedef)
        dtype.components = get_fields(typedef)
        if typedef.methods:
            dtype.procedures = get_type_procedures(mod_name, typedef)
        dtypes[dtype.type_name] = dtype

    return dtypes
