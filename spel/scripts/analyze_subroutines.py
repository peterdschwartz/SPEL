from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

from spel.scripts.DerivedType import DerivedType, get_component
from spel.scripts.fortran_modules import FortranModule
from spel.scripts.fortran_parser.environment import Environment
from spel.scripts.fortran_parser.scope_walk import SubroutineRecord, walk_subroutine
from spel.scripts.fortran_parser.spel_ast import (
    SubroutineDefinitionConstruct,
)
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.fortran_parser.tracing import Trace
from spel.scripts.module_resolver import ModuleResolver, ModuleScopes
from spel.scripts.record_access import AccessMaps
from spel.scripts.helper_functions import combine_many_statuses, find_child_subroutines
from spel.scripts.logging_configs import get_logger, set_logger_level
from spel.scripts.LoopConstructs import Loop
from spel.scripts.process_associate import getAssociateClauseVars
from spel.scripts.types import (
    CallDesc,
    CallTree,
    FileInfo,
    FlatIfs,
    LineTuple,
    PropagatedAccess,
    ReadWrite,
    SubInit,
)
from spel.scripts.utilityFunctions import (
    Variable,
    get_local_variables,
    split_func_line,
)
from spel.scripts.variable_analysis import add_global_vars


class Subroutine(object):
    """
    Class object that holds relevant metadata on a subroutine
    """

    def __init__(
        self,
        init_obj: SubInit,
        lib_func=False,
    ):
        """
        Initalizes the subroutine object:
            1) file for the subroutine is found if not given
            2) calltree is assigned
            3) the associate clause is processed.
        """

        self.name: str = init_obj.name
        self.library: bool = lib_func
        self.func = True if init_obj.function else False

        self.filepath: Path = init_obj.file
        self.startline: int = init_obj.start
        self.module: str = init_obj.mod_name
        self.fort_mod: FortranModule = init_obj.fort_mod

        # endline/cpp_endline: end of this routine's own statements (its `contains`
        # line if it hosts internal subprograms). The regex-based analysis uses these
        # so internal subprograms aren't attributed to the host.
        # end_stmt_ln/cpp_end_stmt_ln: the routine's END statement (used for parsing).
        self.end_stmt_ln: int = init_obj.end
        self.cpp_end_stmt_ln: int | None = init_obj.cpp_end
        self.contains_ln: Optional[int] = init_obj.contains_ln
        self.cpp_contains_ln: Optional[int] = init_obj.cpp_contains_ln
        self.endline: int = (
            init_obj.contains_ln if init_obj.contains_ln is not None else init_obj.end
        )

        self.id: str = f"{self.module}::{self.name}"
        self.mod_deps: set[str] = set()

        # Internal subprograms (after a host's `contains`). Links are set by
        # link_internal_subprograms once every Subroutine has been created.
        self.is_internal: bool = bool(init_obj.parent)
        self.host_id: Optional[str] = (
            f"{self.module}::{init_obj.parent}" if init_obj.parent else None
        )
        self.host: Optional[Subroutine] = None
        self.internal_subs: dict[str, Subroutine] = {}

        # CallTree where repeated child subroutines are not considered.
        self.abstract_call_tree: Optional[CallTree] = None

        # Initialize arguments and local variables
        self.arguments: dict[str, Variable] = {}
        self.local_variables: dict[str, Variable] = {}


        # Compiler preprocessor flags
        self.cpp_startline: int | None = init_obj.cpp_start
        self.cpp_endline: int | None = (
            init_obj.cpp_contains_ln
            if init_obj.cpp_contains_ln is not None
            else init_obj.cpp_end
        )
        self.cpp_filepath: str | None = init_obj.cpp_fn

        # Process the Associate Clause
        self.associate_vars: dict[str, str] = {}
        self.ptr_vars: dict[str, list[str]] = {}

        self.associate_start: int = -1
        self.associate_end: int = -1

        self.dummy_args_list: list[str] = []
        self.return_type = init_obj.function.return_type if init_obj.function else ""
        self.result_name = init_obj.function.result if init_obj.function else ""
        self.result: Variable | None = None

        self.dtype_vars: dict[str, Variable] = {}
        self.sub_lines: list[LineTuple] = []

        self.logger: logging.Logger = get_logger(f"{self.module}::{self.name}")
        self.flat_ifs: list[FlatIfs] = []
        self.ifs_analyzed: bool = False
        self.syntax_tree: Optional[SubroutineDefinitionConstruct] = None

        # Populated by walk_syntax_tree; source of truth for routine data
        self.record: Optional[SubroutineRecord] = None
        # access maps derived from self.record (record_access.AccessMapper)
        self.record_access: Optional[AccessMaps] = None

        if not lib_func:
            full_lines = self.get_sub_lines(init_obj.mod_lines, full=True)
            own_end = self.get_file_info(all=True).endln
            self.sub_lines = [lt for lt in full_lines if lt.ln <= own_end]
            self.parse(full_lines)
            if not self.sub_lines:
                sys.exit(f"FAILED TO GET SUB_LINES FOR { self.name }")

            self.associate_vars, jstart, jend = getAssociateClauseVars(self)
            self.associate_start = jstart
            self.associate_end = jend

            self.dummy_args_list = self._find_dummy_args()
            get_local_variables(self)
            if self.local_variables:
                decl_lns: list[int] = [var.ln for var in self.local_variables.values()]
                self.last_decl_ln = max(decl_lns)

        if init_obj.function:
            if self.result_name in self.arguments:
                self.result = self.arguments.pop(self.result_name)
            else:
                self.result = Variable(
                    type=self.return_type,
                    name=self.result_name,
                    subgrid="?",
                    ln=self.startline,
                    dim=0,
                )
        if self.result_name in self.dummy_args_list:
            self.dummy_args_list.remove(self.result_name)
        if self.arguments:
            sort_args = {}
            for arg in self.dummy_args_list:
                sort_args[arg] = self.arguments[arg]
            self.arguments = sort_args.copy()

        # Access by ln
        self.arg_access_by_ln: dict[str, list[ReadWrite]] = {}
        self.elmtype_access_by_ln: dict[str, list[ReadWrite]] = {}
        self.elmtype_access_summary: dict[str, ReadWrite] = {}
        self.local_vars_access_by_ln: dict[str, list[ReadWrite]] = {}

        self.propagated_access_by_ln: dict[str, list[PropagatedAccess]] = {}

        self.child_subroutines: dict[str, Subroutine] = {}
        self.sub_call_desc: dict[int, CallDesc] = {}

        self.loops: list[Loop] = []

        # non-derived type variables used in the subroutine
        self.active_global_vars: dict[str, Variable] = {}

        ## Section for flags to avoid re-processing subroutines

        # Flag that denotes subroutines that were user requested
        self.unit_test_function: bool = False
        self.preprocessed: bool = False

        self.environment: Optional[Environment] = None

        if not self.library:
            self.get_arg_intent()

    def __repr__(self) -> str:
        name = "Subroutine" if not self.func else "Function"
        return f"{name}({self.get_name()})"

    def get_name(self) -> str:
        return self.id

    def _find_dummy_args(self):
        """
        This function returns the arguments the subroutine takes
        for the s
        And then passes it to the getArguments function
        """
        func_name = "_find_dummy_args"
        tabs = " " * len(func_name)

        lines = self.sub_lines
        regex = re.compile(r"(?<=\()[\w\s,]+(?=\))")

        full_line = lines[0].line
        if self.func:
            _ftype, _f, func_rest = split_func_line(full_line)
            args_and_res = regex.findall(func_rest)
            if args_and_res:
                args = args_and_res[0].split(",")
                if len(args_and_res) != 2:
                    args.append(self.result_name)
                elif len(args_and_res) == 2:
                    args.append(args_and_res[1])
                else:
                    self.logger.error(
                        f"{func_name}Error - wrong function dummy args"
                        + f"{tabs}{args_and_res}\n{tabs}{full_line}"
                    )
                    sys.exit(1)
            else:
                args = [self.result_name] if self.result_name else []
        else:
            args_str = regex.findall(full_line)
            args_str = [_str for _str in args_str if _str.strip()]
            if args_str:
                args = args_str[0].split(",")
            else:
                args = []

        args = [arg.strip() for arg in args]
        return args

    def _get_child_sub_id(
        self, callee: str, sub_dict: dict[str, Subroutine]
    ) -> Optional[str]:
        candidates = {
            id_
            for id_ in sub_dict.keys()
            if re.search(rf"(?<=::){re.escape(callee)}\b", id_)
        }
        actual_id = {id for id in candidates if id.split("::")[0] in self.mod_deps}
        if not actual_id:
            return None
        assert (
            len(actual_id) == 1
        ), f"Error -- couldn't uniquely resolve {callee} in {self.id}\n{actual_id}"
        return actual_id.pop()

    def get_available_dtypes(self, mod_dict: dict[str, FortranModule]):
        intrinsic_types = {"real", "character", "logical", "integer", "complex"}
        sub_mod = self.fort_mod
        variables: dict[str, Variable] = {
            var.name: var
            for var in sub_mod.global_vars.values()
            if var.type not in intrinsic_types
        }
        for mod_name, musage in sub_mod.head_modules.items():
            add_global_vars(
                mod_dict=mod_dict,
                dep_mod=mod_dict[mod_name],
                vars=variables,
                mod_usage=musage,
                mask=lambda x: x not in intrinsic_types,
            )

        fileinfo = self.get_file_info(all=True)
        sub_dep = sub_mod.sort_module_deps(
            startln=fileinfo.startln,
            endln=fileinfo.endln,
        )
        self.mod_deps = (
            self.fort_mod.head_modules.keys() | sub_dep.keys() | {self.module}
        )

        for mod_name, musage in sub_dep.items():
            add_global_vars(
                mod_dict=mod_dict,
                dep_mod=mod_dict[mod_name],
                vars=variables,
                mod_usage=musage,
                mask=lambda x: x not in intrinsic_types,
            )

        return variables

    def replace_associate_in_lines(self):
        def sub_ptr(ptr: str, target: str):
            return lambda lt: LineTuple(
                line=re.sub(rf"(?<!%)\b{ptr}\b", target, lt.line), ln=lt.ln
            )

        passes = [sub_ptr(ptr, target) for ptr, target in self.associate_vars.items()]
        fileinfo = self.get_file_info()
        lines = [
            lt for lt in self.sub_lines if fileinfo.startln <= lt.ln <= fileinfo.endln
        ]
        for func in passes:
            lines = list(map(func, lines))
        return lines

    def find_dtype_vars(
        self,
        instance_dict: dict[str, DerivedType],
    ) -> dict[str, Variable]:
        """
        Function to find the derived types used by a subroutine
        """
        index_str = ""
        regex_paren = re.compile(r"\((.+)\)")  # for removing array of struct index
        regex_dtype_var = re.compile(r"\w+(?:\(\w+\))?%\w+")
        fileinfo = self.get_file_info()
        lines = [
            lt for lt in self.sub_lines if fileinfo.startln <= lt.ln <= fileinfo.endln
        ]
        if self.associate_vars:
            lines = self.replace_associate_in_lines()

        matched_lines = [
            line for line in filter(lambda x: regex_dtype_var.search(x.line), lines)
        ]

        def check_local_decls(my_dict):
            return lambda key: key in my_dict

        def sub_soa(name: str) -> str:
            return regex_paren.sub(index_str, name)

        def replace_associate_ptr(name: str) -> str:
            if name in self.associate_vars:
                name = self.associate_vars[name]
            return name

        local_and_args_dict = self.arguments | self.local_variables
        is_arg_or_local = check_local_decls(local_and_args_dict)

        dtype_vars: dict[str, Variable] = {
            v.name: v
            for v in self.available_dtypes.values()
            if not is_arg_or_local(v.name)
        }

        for lpair in matched_lines:
            m_vars = regex_dtype_var.findall(lpair.line)
            for dtype_w_field in m_vars:
                og_name, og_field = dtype_w_field.split("%", 1)
                actual_inst = replace_associate_ptr(og_name)
                actual_inst = sub_soa(actual_inst).strip()
                if not is_arg_or_local(actual_inst):
                    dtype_var = get_component(instance_dict, actual_inst, og_field)
                    if dtype_var:
                        dtype_var.name = f"{actual_inst}%{og_field}"
                        dtype_vars[dtype_w_field] = dtype_var

        return dtype_vars

    def find_ptr_vars(self):
        """
        Function that finds pointers to derived types either directly or
        through an associated name.
        """
        fileinfo = self.get_file_info()
        regex_ptr = re.compile(r"\w+\s*(=>)\s*\w+(%)\w+")

        sub_lines = self.sub_lines if self.sub_lines else self.get_sub_lines()
        sub_lines = [lpair for lpair in sub_lines if lpair.ln >= fileinfo.startln]

        total_matches: list[LineTuple] = []
        matches = [
            line for line in filter(lambda x: regex_ptr.search(x.line), sub_lines)
        ]
        total_matches.extend(matches)
        if self.associate_vars:
            ptrname_list = [key for key in self.associate_vars.keys()]
            ptrname_str = "|".join(ptrname_list)
            regex_ptr_assoc = re.compile(r"\w+\s*(=>)\s*({})".format(ptrname_str))
            matches = [
                line
                for line in filter(lambda x: regex_ptr_assoc.search(x.line), sub_lines)
            ]
            total_matches.extend(matches)

        for ptr_line in total_matches:
            if ptr_line.line.count("=>") > 1:
                self.logger.warning(f"(get_ptr_vars) {ptr_line}\n fileinfo: {fileinfo}")
                str_ = "\n".join([f"{x.ln} {x.line}" for x in self.sub_lines])
                self.logger.warning(f"{str_}")
                self.logger.warning(
                    f"Associate Start and End: {self.associate_start} - {self.associate_end}"
                )
            ptrname, gv = ptr_line.line.split("=>")
            ptrname = ptrname.strip()
            gv = gv.strip()
            if gv in self.associate_vars:
                gv = self.associate_vars[gv]
            self.ptr_vars.setdefault(ptrname, []).append(gv)

        return None

    def get_sub_lines(
        self, mod_lines: Optional[list[LineTuple]] = None, full: bool = False
    ) -> list[LineTuple]:
        """
        Function that returns lines of a subroutine after trimming comments,
        removing line continuations, and lower-case.
            full: include internal subprograms through the END statement
        """
        fileinfo = self.get_file_info(all=True, full=full)
        regex_all = re.compile(r"(.*)")
        fline_list: list[LineTuple] = []
        if not mod_lines:
            self.logger.error("no mod lines!!!")
        else:
            fline_list = [
                linetuple
                for linetuple in mod_lines
                if fileinfo.startln <= linetuple.ln <= fileinfo.endln
            ]

        return fline_list

    def get_arg_intent(self):
        """
        Attempts to assign intent in/out/inout -> 'r', 'w', 'rw'
         Also check if one of the args is a class
        """
        flines = self.sub_lines if self.sub_lines else self.get_sub_lines()

        lookup_lines = {lpair.ln: lpair.line for lpair in flines}

        regex_intent = re.compile(
            r"intent\s*\(\s*(in\b|inout\b|out\b)\s*\)", re.IGNORECASE
        )
        regex_class = re.compile(r"class\s*\(\s*\w+\s*\)", re.IGNORECASE)
        regex_paren = re.compile(r"(?<=\()\s*\w+\s*(?=\))")

        def set_intent(x: str) -> str:
            match x:
                case "in":
                    return "r"
                case "out":
                    return "w"
                case "inout":
                    return "rw"
                case _:
                    print("Error - Wrong Intent For Argument")
                    sys.exit(1)

        for arg in self.arguments.values():
            line = lookup_lines[arg.ln]
            m_ = regex_intent.search(line.lower())
            cl = regex_class.search(line.lower())
            if m_:
                intent = regex_paren.search(m_.group())
                arg.intent = set_intent(intent.group().strip())
            elif cl:
                class_type = regex_paren.search(cl.group())
                arg.intent = set_intent("inout")
                self.class_method = True
                self.class_type = class_type.group().strip()

        return None

    def get_file_info(self, all: bool = False, full: bool = False):
        """
        Getter that returns tuple for fn, start and stop linenumbers.takes into account cpp files
            full: end at the END statement instead of the host's `contains`
        """
        if self.cpp_filepath:
            fn = self.cpp_filepath
            if self.associate_end == 0 or all:
                start_ln = self.cpp_startline
            else:
                start_ln = self.associate_end
            endline = self.cpp_end_stmt_ln if full else self.cpp_endline
        else:
            fn = self.filepath
            if self.associate_end == 0 or all:
                start_ln = self.startline
            else:
                start_ln = self.associate_end
            endline = self.end_stmt_ln if full else self.endline

        return FileInfo(fpath=fn, startln=start_ln, endln=endline)

    def check_variable_consistency(self) -> bool:
        """
        Checks that the variables in Arguments, LocalVariables, dtype_vars and active_global_vars
        do not overlap (i.e. no variables are improperly shadowed)
        """
        var_set = set()

        var_set.update(self.arguments.keys())

        if var_set & self.local_variables.keys():
            self.logger.error("Error: Local scalar and Argument names overlap.")
            return False
        var_set.update(self.local_variables.keys())

        if var_set & self.dtype_vars.keys():
            self.logger.error(
                f"Error: global dtype names overlap.\n{var_set & self.dtype_vars.keys()}"
            )
            return False
        var_set.update(self.dtype_vars.keys())

        return True

    @Trace.trace_decorator("collect_var_and_call_info")
    def collect_var_and_call_info(
        self,
        sub_dict: dict[str, Subroutine],
        dtype_dict: dict[str, DerivedType],
        mod_dict: dict[str, FortranModule],
        verbose=False,
    ):
        """
        Function that collections usage of global derived type variables,
        pointer variables and any child subroutine calls.
            * main_sub_dict : dict of all subroutines for FUT
            * dtype_dict : dict of user type defintions
            * interface_list : contains names of known interfaces
        """
        func_name = "(collect_var_and_call_info)"
        logger = get_logger(func_name)

        global_vars: dict[str, DerivedType] = {}
        for dtype in dtype_dict.values():
            for inst in dtype.instances.keys():
                if inst not in global_vars:
                    global_vars[inst] = dtype
        for argname, arg in self.arguments.items():
            if arg.type in dtype_dict.keys():
                global_vars[argname] = dtype_dict[arg.type]

        self.available_dtypes = self.get_available_dtypes(mod_dict)
        set_logger_level(self.logger, logging.DEBUG)
        self.dtype_vars = self.find_dtype_vars(global_vars)

        ok = self.check_variable_consistency()
        if not ok:
            logger.error(
                f"Subroutine parsing has inconsistencies for {self.name} exiting..."
            )
            sys.exit(1)

        self.find_ptr_vars()

        find_child_subroutines(self, sub_dict, dtype_dict)

        for call_desc in self.sub_call_desc.values():
            actual_sub_name = call_desc.fn
            if actual_sub_name not in sub_dict:
                childsub: Subroutine = Subroutine(
                    init_obj=SubInit(
                        name=actual_sub_name,
                        mod_name="lib",
                        fort_mod=None,
                        mod_lines=[],
                        file="lib.F90",
                        start=-999,
                        end=-999,
                        cpp_end=None,
                        cpp_start=None,
                        cpp_fn="",
                        function=None,
                        parent="",
                    ),
                    lib_func=True,
                )
                sub_dict[actual_sub_name] = childsub
                self.logger.debug(f"Adding {actual_sub_name} / {childsub} to sub_dict")
            else:
                childsub: Subroutine = sub_dict[actual_sub_name]

            child_sub_names = [s for s in self.child_subroutines.keys()]
            if actual_sub_name not in child_sub_names:
                self.child_subroutines[actual_sub_name] = childsub

        self.preprocessed = True

        return None

    def parse(self, lines: Optional[list[LineTuple]] = None):
        """
        Parse the routine (including internal subprograms) into self.syntax_tree
        """
        lines = lines if lines is not None else self.sub_lines
        self.check_assert(lines is not None, msg="Got Empty lines for subroutine!!")

        parser = Parser(lines=lines, logger=f"Parser-{self.id}")
        program = parser.parse_program()
        cond = len(program.statements) == 1 and isinstance(
            program.statements[0], SubroutineDefinitionConstruct
        )
        self.check_assert(
            cond,
            msg=f"Got {[type(s) for s in program.statements]}\nExpected: SubroutineDefinitionConstruct",
        )
        assert len(program.statements) == 1 and isinstance(
            program.statements[0], SubroutineDefinitionConstruct
        )
        self.syntax_tree = program.statements[0]
        self.check_assert(
            self.name == self.syntax_tree.name,
            msg=f"Name from parsing doesn't match:  {self.name} != {self.syntax_tree.name}",
        )

        return

    def walk_syntax_tree(self, scopes: ModuleScopes) -> SubroutineRecord:
        """
        Scope-aware walk of self.syntax_tree into self.record (once).
        An internal subprogram needs its host's scope, so the host is walked
        first, with its own resolver.
        """
        if self.record is not None:
            return self.record
        self.check_assert(
            self.syntax_tree is not None, msg=f"{self.name}: not parsed before walk"
        )
        host_rec = None
        if self.is_internal:
            self.check_assert(self.host is not None, msg=f"{self.name}: host not linked")
            host_rec = self.host.walk_syntax_tree(scopes)
        resolver = ModuleResolver.for_subroutine(
            self, scopes.mod_dict, scopes.sub_dict, scopes=scopes
        )
        self.record = walk_subroutine(self.syntax_tree, resolver=resolver, host=host_rec)
        return self.record

    def check_assert(self, cond: bool, msg: str):
        if not cond:
            self.logger.error(msg)
            raise RuntimeError

    def summarize_readwrite(self, verbose=False):
        """
        Aggregates the read write status for each variable
        """
        for k, rws in self.elmtype_access_by_ln.items():
            sorted_rws = sorted(rws, key=lambda x: x.ln)
            self.elmtype_access_summary[k] = ReadWrite(
                status=combine_many_statuses([s.status for s in sorted_rws]),
                ln=-1,
                line=None,
            )
        return

    def _get_ptr_targets(self, pot_ptr: str) -> list[str]:
        return self.ptr_vars.get(pot_ptr, [pot_ptr])

    def elmtype_accesses_exclusive_to_namelist_ifs(self) -> dict[str, set[FlatIfs]]:
        """
        Return ELM derived-type fields whose accesses occur only inside namelist-gated
        flat if blocks.
        """

        nml_ifs = [
            flat_if
            for flat_if in self.flat_ifs
            if flat_if.nml_vars or flat_if.nml_cascades
        ]

        def matching_namelist_if(access: ReadWrite) -> Optional[FlatIfs]:
            distance_from_if = [
                (dist, x_if)
                for x_if in nml_ifs
                if (dist := access.ln - x_if.start_ln) >= 0 and access.ln < x_if.end_ln
            ]
            distance_from_if.sort(key=lambda x: x[0])

            if distance_from_if:
                return distance_from_if[0][1].copy()
            return None

        exclusive: dict[str, set[FlatIfs]] = {}

        for elm_field, accesses in self.elmtype_access_by_ln.items():
            guarded_accesses = [matching_namelist_if(rw) for rw in accesses]
            guarded_accesses = [_if for _if in guarded_accesses if _if is not None]

            # check if every access is a guarded access
            if len(guarded_accesses) == len(accesses):
                exclusive[elm_field] = set(guarded_accesses)
        return exclusive


def link_internal_subprograms(sub_dict: dict[str, Subroutine]) -> None:
    """
    Connect internal subprograms to their host routine (and vice versa) so that
    symbol lookups can fall back to the host's scope.
    """
    for sub in sub_dict.values():
        if not sub.is_internal:
            continue
        assert sub.host_id, f"{sub.id} is internal but has no host"
        host = sub_dict.get(sub.host_id)
        if host is None:
            sub.logger.error(f"Host {sub.host_id} not found for internal {sub.id}")
            sys.exit(1)
        sub.host = host
        host.internal_subs[sub.name] = sub
