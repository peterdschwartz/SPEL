from __future__ import annotations

import bisect
import re
import subprocess as sp
import sys
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

from scripts.utilityFunctions import Variable

if TYPE_CHECKING:
    from scripts.analyze_subroutines import Subroutine
    from scripts.DerivedType import DerivedType

import scripts.dynamic_globals as dg
from scripts.config import ELM_SRC, SHR_SRC, _bc
from scripts.types import LineTuple, ModUsage, PointerAlias


def get_module_name_from_file(fpath: str) -> tuple[int, str]:
    """
    Given a file path, returns the name of the module
    """
    if fpath not in dg.map_fpath_to_module_name:
        cmd = f'grep -rin -E "^[[:space:]]*module [[:space:]]*[[:alnum:]]+" {fpath}'
        # the module declaration will be the first one. Any others will be interfaces
        module_name = sp.getoutput(cmd).split("\n")[0]
        # grep will have pattern <line_number>:module <module_name>
        linenumber, module_name = module_name.split(":")
        module_name = module_name.split()[1].lower()
        linenumber = int(linenumber)
        dg.map_fpath_to_module_name[fpath] = (linenumber, module_name)
    else:
        linenumber, module_name = dg.map_fpath_to_module_name[fpath]

    return linenumber, module_name


def get_filename_from_module(module_name: str, verbose: bool = False):
    """
    Given a module name, returns the file path of the module
    """
    if module_name in dg.map_module_name_to_fpath:
        return dg.map_module_name_to_fpath[module_name]

    cmd = f'grep -rin --exclude-dir=external_models/ "module {module_name}" {ELM_SRC}*'
    elm_output = sp.getoutput(cmd)
    if not elm_output:
        if verbose:
            print("Checking shared modules...")
        #
        # If file is not an ELM file, may be a shared module in E3SM/share/util/
        #
        cmd = f'grep -rin --exclude-dir=external_models/ "module {module_name}" {SHR_SRC}*'
        shr_output = sp.getoutput(cmd)

        if not shr_output:
            if verbose:
                print(
                    f"Couldn't find {module_name} in ELM or shared source -- adding to removal list"
                )
            file_path = None
        else:
            file_path = shr_output.split("\n")[0].split(":")[0]
    else:
        file_path = elm_output.split("\n")[0].split(":")[0]

    if file_path and file_path.split(".")[-1].lower() != "f90":
        file_path = None
    dg.map_module_name_to_fpath[module_name] = file_path

    return file_path


def unravel_module_dependencies(modtree, mod_dict, mod, depth=0):
    """
    Recursively go through module dependencies and
    return an ordered list with the depth at which it is used.
    """
    depth += 1
    for m in mod.modules.keys():
        modtree.append({"module": m, "depth": depth})
        dep_mod = mod_dict[m]
        if dep_mod.modules:
            modtree = unravel_module_dependencies(
                modtree=modtree, mod_dict=mod_dict, mod=dep_mod, depth=depth
            )

    return modtree


def print_spel_module_dependencies(
    mod_dict: dict[str, FortranModule],
    subs: dict[str, Subroutine],
    depth=0,
):
    """
    Given a dictionary of modules needed for this unit-test
    this prints their dependencies with the modules containing
    subs being the parents
    """
    modtree = []

    for sub in subs.values():
        depth = 0
        module_name = sub.module
        sub_module = mod_dict[module_name]
        modtree.append({"module": module_name, "depth": depth})
        depth += 1
        for mod in sub_module.modules.keys():
            modtree.append({"module": mod, "depth": depth})
            dep_mod = mod_dict[mod]
            if dep_mod.modules.keys():
                modtree = unravel_module_dependencies(
                    modtree=modtree, mod_dict=mod_dict, mod=dep_mod, depth=depth
                )
    return modtree


def parse_only_clause(line: str) -> set[PointerAlias]:
    """
    Input a line of the form: `use modname, only: name1,name2,...`
    """
    # get items after only:
    only_l = line.split(":")[1]
    only_l = only_l.split(",")

    only_objs_list = set()
    # Go through list of objects, determine '=>' usage.
    for ptrobj in only_l:
        if "=>" in ptrobj:
            ptr, obj = ptrobj.split("=>")
            ptr = ptr.strip()
            obj = obj.strip()
            only_objs_list.add(PointerAlias(ptr=ptr, obj=obj))
        else:
            obj = ptrobj.strip()
            only_objs_list.add(PointerAlias(ptr=None, obj=obj))

    return only_objs_list


def build_module_tree(modules: dict[str, FortranModule]) -> list[ModTree]:
    """
    Builds a forest (list of ModTree roots) from a dictionary of FortranModule objects.
    Returns a list because there may be multiple independent module trees.
    """
    mod_nodes: dict[str, ModTree] = {name: ModTree(name) for name in modules.keys()}
    roots = set(mod_nodes.keys())  # Start by assuming all are roots

    for name, module in modules.items():
        for dep in module.modules:
            mod_nodes[name].add_child(mod_nodes[dep])
            if dep in roots:
                roots.remove(dep)

    return [mod_nodes[root] for root in roots]


def insert_header_for_unittest(
    mod_list: list[str], mod_dict: dict[str, FortranModule], casedir: str
):
    """
    Function that will insert the header file into files needed for unit test
    The header file contains definitions to aid in compilation (e.g, override pio types)
    """

    func_name = "insert_header_for_unittest"

    for mod_name in mod_list:

        f = get_filename_from_module(mod_name)
        assert f
        # Change path to unit test case directory
        fpath = casedir + "/" + f.split("/")[-1]
        fort_mod = mod_dict[mod_name]

        ifile = open(fpath, "r")
        lines = ifile.readlines()
        ifile.close()
        lines.insert(fort_mod.ln, '#include "unittest_defs.h"\n')
        with open(fpath, "w") as ofile:
            ofile.writelines(lines)
    return None


class FortranModule:
    """
    A class to represent a Fortran module.
    Main purpose is to store other modules required to
    compile the given file. To be used to determine the
    order in which to compile the modules.
    """

    def __init__(self, name, fname, ln):
        self.name = name  # name of the module
        self.filepath = fname  # the file path of the module
        self.ln: int = ln  # line number of start module block
        self.num_lines: int = -1

        # objects in declared
        self.global_vars: dict[str, Variable] = {}
        self.subroutines: set[str] = set()
        self.defined_types: dict[str, DerivedType] = {}

        # module dependencies
        self.modules: dict[str, ModUsage] = {}
        self.modules_by_ln: dict[str, ModUsage] = {}

        # modules available to all subroutines
        self.head_modules: dict[str, ModUsage] = {}

        self.modified: bool = False  # if module has been through modify_file or not.
        self.variables_sorted: bool = False
        self.end_of_head_ln: int = 99999

        self.module_lines: list[LineTuple] = []

    def __repr__(self):
        return f"FortranModule({self.name})"

    def get_mod_lines(self):
        return self.module_lines

    def add_dependency(self, mod: str, line: str, ln: int):
        """
        Adds module dependency and either only the items accessed
        or "all" for blanket usage of module

        Dependencies here are stored with their linenumber that way module head
        and subroutine specific usages can be easily sorted.
        """
        mod_key = f"{mod}@{ln}"

        curr_usage = self.modules_by_ln.get(mod_key)
        if curr_usage is None:
            self.modules_by_ln[mod_key] = ModUsage(all=False, clause_vars=set())

        # Check if there is an only statement AND that the entire module isn't used.
        if "only" in line and not (self.modules_by_ln[mod_key].all):
            obj_set = parse_only_clause(line)
            self.modules_by_ln[mod_key].clause_vars |= obj_set
        else:
            # Even if only clause was previously used, overwrite
            # and assume it's all used.
            self.modules_by_ln[mod_key].all = True

    def sort_module_deps(self, startln: int, endln: int) -> dict[str, ModUsage]:
        """
        Sort the modules_by_ln into either a dict for all dependencies
        or a dict for only modules in the module head
        """
        sorted_dict: dict[str, ModUsage] = {}
        for modln, muse in self.modules_by_ln.items():
            mod_name, ln = modln.split("@")
            if startln < int(ln) < endln:
                if mod_name in sorted_dict:
                    cur_usage = sorted_dict[mod_name]
                    if cur_usage.all:
                        continue
                    if muse.all:
                        sorted_dict[mod_name].all = True
                    else:
                        sorted_dict[mod_name].clause_vars |= muse.clause_vars
                else:
                    sorted_dict[mod_name] = deepcopy(muse)

        return sorted_dict

    def find_allocations(self, sub_dict: dict[str, Subroutine]):

        if self.name == "columndatatype":
            print(self.defined_types)
        regex_alloc = re.compile(r"^(allocate\b)")
        fline_list = self.module_lines
        alloc_lines = [
            line for line in filter(lambda x: regex_alloc.search(x.line), fline_list)
        ]

        mod_subs = [sub_dict[subname] for subname in self.subroutines]
        Interval = namedtuple("Interval", "tag, xi, xf")
        intervals: list[Interval] = [
            Interval(tag=sub.name, xi=sub.startline, xf=sub.endline) for sub in mod_subs
        ]
        intervals.sort(key=lambda x: x.xi)
        starts = [x.xi for x in intervals]

        def get_init_sub(ln: int) -> Optional[str]:
            idx = bisect.bisect_right(starts, ln) - 1
            if idx >= 0:
                dx = intervals[idx]
                if dx.xi <= ln <= dx.xf:
                    return dx.tag
            return None

        for dtype in self.defined_types.values():
            debug = False
            if dtype.type_name == "column_water_state":
                debug = True
            vars = [
                rf"%{field.name}"
                for field in dtype.components.values()
                if field.dim > 0
            ]
            # Find Allocations
            if not vars:
                continue
            var_str = "|".join(vars)
            regex_var = re.compile(rf"\b({var_str})\b")
            var_lines = [
                line for line in filter(lambda x: regex_var.search(x.line), alloc_lines)
            ]
            for lpair in var_lines:
                line = lpair.line.strip()
                regex_var_and_bounds = re.compile(rf"({var_str})\s*(\(.+?\))")
                for match in regex_var_and_bounds.finditer(line):
                    varname = match.group(1).lstrip("%")
                    bounds = match.group(2)
                    dtype.components[varname].bounds = bounds[1:-1]
                    dtype.init_sub_name = get_init_sub(lpair.ln)

            # Find potential targets for pointers
            regex_ptr_init = re.compile(rf"({var_str})\b\s*(=>)(.+)")
            var_lines = [
                line
                for line in filter(lambda x: regex_ptr_init.search(x.line), fline_list)
            ]
            for lpair in var_lines:
                match_ptrinit = regex_ptr_init.search(lpair.line)
                target = match_ptrinit.groups()[2].strip()
                varname = match_ptrinit.groups()[0].strip().replace("%", "")
                if target not in dtype.components[varname].pointer:
                    dtype.components[varname].pointer.append(target)
                    dtype.init_sub_name = get_init_sub(lpair.ln)

        return


class ModTree:
    def __init__(self, node):
        self.node: str = node
        self.children: list[ModTree] = []
        self.parent: Optional[ModTree] = None

    def add_child(self, child: "ModTree"):
        child.parent = self
        self.children.append(child)

    def traverse_preorder(self):
        yield self
        for child in self.children:
            yield from child.traverse_preorder()

    def traverse_postorder(self):
        for child in self.children:
            yield from child.traverse_postorder()
        yield self

    def __repr__(self):
        return f"ModTree({self.node}, children={len(self.children)})"

    def print_tree(self, level: int = 0):
        if level == 0:
            print("ModTree for", self.node)
        indent = "|--" * level
        print(f"{indent}>{self.node}")
        for child in self.children:
            child.print_tree(level + 1)
