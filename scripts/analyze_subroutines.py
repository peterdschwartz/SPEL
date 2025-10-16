from __future__ import annotations

import logging
import os.path
import re
import sys
from typing import Any, Dict, Optional

from scripts.config import _bc, _no_colors, spel_dir
from scripts.DerivedType import DerivedType, expand_dtype, get_component
from scripts.fortran_parser.environment import Environment
from scripts.fortran_parser.spel_ast import Statement
from scripts.fortran_parser.tracing import Trace
from scripts.helper_functions import (ReadWrite, SubroutineCall,
                                      analyze_sub_variables, combine_status,
                                      find_child_subroutines, is_derived_type,
                                      merge_status_list, normalize_soa_keys,
                                      replace_elmtype_arg,
                                      summarize_read_write_status,
                                      trace_derived_type_arguments)
from scripts.logging_configs import get_logger
from scripts.LoopConstructs import Loop
from scripts.process_associate import getAssociateClauseVars
from scripts.types import (ArgLabel, CallDesc, CallTree, FileInfo, FlatIfs,
                           LineTuple, SubInit)
from scripts.utilityFunctions import (Variable, determine_filter_access,
                                      find_file_for_subroutine,
                                      get_interface_list, get_local_variables,
                                      getArguments, line_unwrapper,
                                      lineContinuationAdjustment,
                                      search_in_file_section, split_func_line)

# Setup logger
logger = get_logger("SubroutineAnalysis")


class Subroutine(object):
    """
    Class object that holds relevant metadata on a subroutine

    Note: Variables are stored using the Variable Class defined in utilityFunctions.py
    Class Methods:
        * printSubroutineInfo -> method to print to terminal subroutine metadata.
        * _get_dummy_args -> used to find keywords for subroutine arguments
        * parse_subroutine -> Main function called and calls some class methods
                              in order to collect all relevant metadata.
        * _check_acc_status -> checks if subroutine already as !$acc routine seq
        * _get_global_constants -> Find global variables that are parameters (should it be a Class method?)
        * _preprocess -> (rename file?) Gets child subroutines
        * analyze_variables -> loops through global vars, determines r/w status.
        * child_subroutines_analysis -> recursively analyzes child subs. Called after parse_subroutine
        * analyze_calltree -> unravel calltree list to generate proper/readable calltree to terminal or txt file.
        * generate_update_directives -> writes verificationMod subroutines
        * examineLoops -> Not part of primary Unit Test generation; enabled with opt flag.
                This collects all the metadata concerning the Loops of a subroutine (and child subroutines)
                Stores Loops in a Class defined in LoopConstructs.py
                Once loop metadata is collected, OpenACC loop pragmas may be added.
                Data regions are inserted at the beginning and end of subroutines
                If adjust_allocation is enabled, then local variables are compressed as able.
        * exportReadWriteVariables -> writes formatted output for variable status
        * update_arg_tree -> maps dummy args to real args
        * generate_unstructured_data_regions -> inserts OpenACC data regions for local variables
    """

    def __init__(
        self,
        init_obj: SubInit,
        lib_func=False,
        verbose: bool = False,
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

        self.filepath: str = init_obj.file
        self.startline: int = init_obj.start
        self.endline: int = init_obj.end
        self.module: str = init_obj.mod_name
        self.mod_deps: dict[str, str | list[Any]]

        # CallTree where repeated child subroutines are not considered.
        self.abstract_call_tree: Optional[CallTree] = None

        # Initialize arguments and local variables
        self.arguments: dict[str, Variable] = {}
        self.local_variables: dict[str, Variable] = {}
        self.class_method: bool = False
        self.class_type: Optional[str] = None

        # Create another argument dict that stores the read,write status of them:
        self.arguments_read_write: dict[str, ReadWrite] = {}
        self.arg_access_by_ln: dict[str, list[ReadWrite]] = {}

        # Store when the arguments/local variable declarations start and end
        self.var_declaration_startl: int = 0
        self.var_declaration_endl: int = 0

        # Compiler preprocessor flags
        self.cpp_startline: int | None = init_obj.cpp_start
        self.cpp_endline: int | None = init_obj.cpp_end
        self.cpp_filepath: str | None = init_obj.cpp_fn

        # Process the Associate Clause
        self.associate_vars: dict[str, str] = {}
        self.reverse_associate_map: dict[str, str] = {}
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
        self.if_blocks: list[Statement] = []
        self.flat_ifs: list[FlatIfs] = []
        self.nml_ifs: list[FlatIfs] = []
        self.ifs_analyzed: bool = False

        if not lib_func:
            self.sub_lines = self.get_sub_lines(init_obj.mod_lines)
            if not self.sub_lines:
                sys.exit(f"FAILED TO GET SUB_LINES FOR { self.name }")

            self.associate_vars, jstart, jend = getAssociateClauseVars(self)
            self.associate_start = jstart
            self.associate_end = jend
            self.reverse_associate_map: dict[str, str] = {
                val: key for key, val in self.associate_vars.items()
            }

            self.dummy_args_list = self._get_dummy_args()
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
            # print(f"{self.name} / dummy args: {self.dummy_args_list} \n args {self.arguments}\n result name: {self.result_name}")
            for arg in self.dummy_args_list:
                sort_args[arg] = self.arguments[arg]
            self.arguments = sort_args.copy()

        # These 3 dictionaries will be summaries of ReadWrite Status for the FUT subroutines as a whole
        self.elmtype_access_sum: dict[str, str] = {}

        self.elmtype_access_by_ln: dict[str, list[ReadWrite]] = {}

        self.local_vars_access_by_ln: dict[str, list[ReadWrite]] = {}

        self.subroutine_call: list[SubroutineCall] = []
        self.child_subroutines: dict[str, Subroutine] = {}
        self.sub_call_desc: dict[int, CallDesc] = {}

        self.loops: list[Loop] = []

        # non-derived type variables used in the subroutine
        self.active_global_vars: dict[str, Variable] = {}

        ## Section for flags to avoid re-processing subroutines

        # Flag that denotes subroutines that were user requested
        self.unit_test_function: bool = False
        self.acc_status: bool = False
        self.analyzed_child_subroutines: bool = False
        self.preprocessed: bool = False
        self.args_analyzed: bool = False
        self.global_analyzed: bool = False

        self.environment: Optional[Environment] = None


        if not self.library:
            self.get_arg_intent()

    def __repr__(self) -> str:
        name = "Subroutine" if not self.func else "Function"
        return f"{name}({self.get_name()})"

    def get_name(self) -> str:
        return f"{self.module}::{self.name}"

    def print_subroutine_info(self, ofile=sys.stdout, long=False):
        """
        Prints the current subroutine metadata that has been collected.
        If `long` is True, then local scalar variables are printed as well.
        """
        tab = "  "
        base_path = "/".join(self.filepath.split("/")[-2:])

        if ofile != sys.stdout:
            c = _no_colors
        else:
            c = _bc

        title = "Subroutine" if not self.func else "Function"
        ofile.write(
            c.HEADER
            + f"{title} {self.name} in {base_path} L{self.startline}-{self.endline}\n"
            + c.ENDC
        )
        if self.func:
            ofile.write(c.HEADER + f"{tab}Result -> {self.result}\n")

        ofile.write(c.WARNING + "Has Arguments:\n" + c.ENDC)
        for arg in self.arguments.values():
            if arg.optional:
                _str = "~" * len(tab)
            else:
                _str = tab
            ofile.write(c.OKBLUE + _str + f"{arg}\n" + c.ENDC)

        # Print local variables
        ofile.write(c.WARNING + "Local Arrays:\n" + c.ENDC)
        array_dict = {
            key: val for key, val in self.local_variables.items() if val.dim > 0
        }
        for arg in array_dict:
            var = array_dict[arg]
            ofile.write(c.OKGREEN + f"{tab}{var}\n" + c.ENDC)
        if long:
            ofile.write(c.WARNING + "Local Scalars:\n")
            scalars = {
                key: val for key, val in self.local_variables.items() if val.dim == 0
            }
            for arg in scalars:
                ofile.write(f"{tab}{arg}\n")

        # print Child Subroutines
        if self.child_subroutines:
            ofile.write(c.WARNING + "Child Subroutines:\n")
            for s in self.child_subroutines.values():
                ofile.write(f"{tab}{s.name}\n")
        ofile.write(c.ENDC)
        return None

    def _get_dummy_args(self):
        """
        This function returns the arguments the subroutine takes
        for the s
        And then passes it to the getArguments function
        """
        func_name = "_get_dummy_args"
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
                    logger.error(
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

    def get_dtype_vars(
        self, instance_dict: Dict[str, DerivedType]
    ) -> Dict[str, Variable]:
        """
        Function
        """
        index_str = ""
        regex_paren = re.compile(r"\((.+)\)")  # for removing array of struct index
        regex_dtype_var = re.compile(r"\w+(?:\(\w+\))?%\w+")

        lines = self.sub_lines
        matched_lines = [
            line for line in filter(lambda x: regex_dtype_var.search(x.line), lines)
        ]

        def check_local_decls(my_dict):
            return lambda key: key.replace(index_str, "") in my_dict

        def sub_soa(name: str) -> str:
            inst, field = name.split("%")
            inst = regex_paren.sub(index_str, inst)
            return f"{inst}%{field}"

        local_and_args_dict = (
            self.arguments
            | self.local_variables
        )
        is_arg_or_local = check_local_decls(local_and_args_dict)

        dtype_vars: dict[str, Variable] = {}
        for lpair in matched_lines:
            m_vars = regex_dtype_var.findall(lpair.line)
            for dtype in m_vars:
                dtype = sub_soa(dtype)
                if dtype not in dtype_vars and not is_arg_or_local(dtype.split("%")[0]):
                    dtype_var = get_component(instance_dict, dtype)
                    if dtype_var:
                        dtype_var.name = dtype
                        dtype_vars[dtype] = dtype_var

        return dtype_vars

    def get_ptr_vars(self):
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
                str_ = "\n".join([ f"{x.ln} {x.line}" for x in self.sub_lines])
                self.logger.warning(f"{str_}")
                self.logger.warning(f"Associate Start and End: {self.associate_start} - {self.associate_end}")
            ptrname, gv = ptr_line.line.split("=>")
            ptrname = ptrname.strip()
            gv = gv.strip()
            if gv in self.associate_vars:
                gv = self.associate_vars[gv]
            self.ptr_vars.setdefault(ptrname, []).append(gv)

        return None

    def get_sub_lines(
        self, mod_lines: Optional[list[LineTuple]] = None
    ) -> list[LineTuple]:
        """
        Function that returns lines of a subroutine after trimming comments,
        removing line continuations, and lower-case
        """
        fileinfo = self.get_file_info(all=True)
        regex_all = re.compile(r"(.*)")
        if not mod_lines:
            lines = search_in_file_section(
                fpath=fileinfo.fpath,
                start_ln=fileinfo.startln,
                end_ln=fileinfo.endln,
                pattern=regex_all,
            )
            fline_list: list[LineTuple] = []
            ln: int = 0
            while ln < len(lines):
                full_line, new_ln = line_unwrapper(lines, ln)
                if full_line:
                    statements = full_line.split(";")
                    for stmt in statements:
                        fline_list.append(LineTuple(line=stmt.strip(), ln=ln))
                ln = new_ln + 1

            fline_list = [
                LineTuple(line=f.line, ln=f.ln + fileinfo.startln) for f in fline_list
            ]
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

    def get_file_info(self, all: bool = False):
        """
        Getter that returns tuple for fn, start and stop linenumbers.takes into account cpp files
        """
        if self.cpp_filepath:
            fn = self.cpp_filepath
            if self.associate_end == 0 or all:
                start_ln = self.cpp_startline
            else:
                start_ln = self.associate_end
            endline = self.cpp_endline
        else:
            fn = self.filepath
            if self.associate_end == 0 or all:
                start_ln = self.startline
            else:
                start_ln = self.associate_end
            endline = self.endline

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
            self.logger.error("Error: global dtype names overlap.")
            return False
        var_set.update(self.dtype_vars.keys())

        return True

    @Trace.trace_decorator("collect_var_and_call_info")
    def collect_var_and_call_info(
        self,
        sub_dict: dict[str, Subroutine],
        dtype_dict: dict[str, DerivedType],
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

        self.dtype_vars = self.get_dtype_vars(global_vars)

        ok = self.check_variable_consistency()
        if not ok:
            logger.error(
                f"Subroutine parsing has inconsistencies for {self.name} exiting..."
            )
            sys.exit(1)

        self.get_ptr_vars()

        find_child_subroutines(self, sub_dict, dtype_dict)

        for call_desc in self.sub_call_desc.values():
            actual_sub_name = call_desc.fn
            if actual_sub_name not in sub_dict:
                print(
                    _bc.WARNING
                    + f"Warning: {actual_sub_name} not in main_sub_dict."
                    + "\nCould be library function?"
                    + _bc.ENDC
                )
                childsub: Subroutine = Subroutine(
                    init_obj=SubInit(
                        name=actual_sub_name,
                        mod_name="lib",
                        mod_lines=[],
                        file="lib.F90",
                        start=-999,
                        end=-999,
                        cpp_end=None,
                        cpp_start=None,
                        cpp_fn="",
                        function=None,
                    ),
                    lib_func=True,
                )
                sub_dict[actual_sub_name] = childsub
            else:
                childsub: Subroutine = sub_dict[actual_sub_name]

            child_sub_names = [s for s in self.child_subroutines.keys()]
            if actual_sub_name not in child_sub_names:
                self.child_subroutines[actual_sub_name] = childsub

        self.preprocessed = True

        return None

    def _check_acc_status(self):
        """
        checks if subroutine already has !$acc directives
        """
        filename = self.filepath
        file = open(filename, "r")
        lines = file.readlines()  # read entire file
        for ct in range(self.startline, self.endline):
            line = lines[ct]
            # checks for any !$acc directives
            match_acc = re.search(r"^[\s]+(\!\$acc)", line)
            if match_acc:
                self.acc_status = True
                return None

    @Trace.trace_decorator("analyze_variables")
    def analyze_variables(
        self,
        sub_dict: dict[str, Subroutine],
        type_dict: dict[str, DerivedType],
        verbose: bool = False,
    ):
        """
        Function used to determine read and write variables
        If var is written to first, then rest of use is ignored.
        Vars determined to be read are read-in from a file generated by
            full E3SM run called {unit-test}_vars.txt by default.

        Vars that are written to must be checked to verify
        results of unit-testing.

        """
        func_name = "( analyze_variables )"
        self.logger.info(func_name)

        global_vars: Dict[str, Variable] = {}
        for dtype in type_dict.values():
            for dtype_var in dtype.instances.values():
                if dtype_var.name not in global_vars:
                    global_vars[dtype_var.name] = dtype_var


        # remove struct of arrays/field/access from dtype_vars and replace with the instance var
        regex_inst_name = re.compile(r"(?:\(\w+\))?%\w+")
        var_dict: dict[str, Variable] = {
            key: val
            for key, val in self.dtype_vars.items()
        }

        for alias, gv in self.associate_vars.items():
            if gv.split("%")[0] not in self.arguments.keys():
                var = var_dict[gv]
                var_dict[alias] = var.copy()
        # inst_set = {
        #     regex_inst_name.sub("", x)
        #     for x in self.dtype_vars.keys()
        #     if regex_inst_name.search(x)
        # }
        #
        # for inst in inst_set:
        #     inst_var = global_vars[inst]
        #     var_dict[inst] = inst_var.copy()


        globals_accessed = analyze_sub_variables(
            self,
            sub_dict,
            var_dict,
            mode=ArgLabel.globals,
            verbose=verbose,
        )
        norm = normalize_soa_keys(globals_accessed)

        for var_name, stat_list in norm.items():
            if var_name in self.active_global_vars:
                continue
            if var_name in self.associate_vars:
                actual_name = self.associate_vars[var_name]
                inst, _ = actual_name.split("%")
                if inst in self.arguments:
                    merge_status_list(actual_name, self.arg_access_by_ln, stat_list)
                else:
                    merge_status_list(actual_name, self.elmtype_access_by_ln, stat_list)
            else:
                self.elmtype_access_by_ln[var_name] = stat_list.copy()

        # Analyze local variables and add any pointers to elmtypes to elmtype_access_by_ln
        local_var_dict = self.local_variables
        local_accessed = analyze_sub_variables(
            self,
            sub_dict,
            local_var_dict,
            mode=ArgLabel.locals,
            verbose=verbose,
        )

        self.local_vars_access_by_ln = local_accessed

        # For local variables that are pointers, merge their ReadWrite status with their target
        for ptr, gv_list in self.ptr_vars.items():
            stat_list = local_accessed[ptr]
            for gv in gv_list:
                inst, _ = gv.split("%")
                if inst in self.arguments:
                    merge_status_list(gv, self.arg_access_by_ln, stat_list)
                else:
                    merge_status_list(gv, self.elmtype_access_by_ln, stat_list)

        self.global_analyzed = True

        # for var in self.Arguments.values():
        #     if var.type not in intrinsic_types:
        #         inst_set.add(var.name)
        # if inst_set and self.sub_call_desc:
        #     trace_derived_type_arguments(parent_sub=self,sub_dict=sub_dict,inst_set=inst_set,
        #     )

        return

    def match_arg_to_inst(self, type_dict: dict[str, DerivedType]):
        """
        Called for parent subroutine: populate elmtype_access_by_ln
        with instances with status from arg_access_by_ln
        """
        verbose = False
        name_map: dict[str, str] = {}
        for arg in self.arguments.values():
            dtype = type_dict.get(arg.type, None)
            if dtype:
                for inst in dtype.instances:
                    name_map[arg.name] = inst
        if not name_map:
            return
        replace_elmtype_arg(name_map, self, verbose)
        return

    def summarize_readwrite(self):
        summary_elmtype = summarize_read_write_status(self.elmtype_access_by_ln)
        for var, status in summary_elmtype.items():
            self.elmtype_access_sum[var] = status
        return

    def generate_update_directives(self, elmvars_dict, verify_vars):
        """
        This function will create .F90 routine to execute the
        update directives to check the results of the subroutine
        """
        ofile = open(
            f"{spel_dir}scripts/script-output/update_vars_{self.name}.F90", "w"
        )

        spaces = " " * 2
        ofile.write("subroutine update_vars_{}(gpu,desc)\n".format(self.name))

        for dtype in verify_vars.keys():
            mod = elmvars_dict[dtype].declaration
            ofile.write(spaces + f"use {mod}, only : {dtype}\n")

        ofile.write(spaces + "implicit none\n")
        ofile.write(spaces + "integer, intent(in) :: gpu\n")
        ofile.write(spaces + "character(len=*), optional, intent(in) :: desc\n")
        ofile.write(spaces + "character(len=256) :: fn\n")
        ofile.write(spaces + "if(gpu) then\n")
        ofile.write(spaces + spaces + f'fn="gpu_{self.name}"\n')
        ofile.write(spaces + "else\n")
        ofile.write(spaces + spaces + f"fn='cpu_{self.name}'\n")
        ofile.write(spaces + "end if\n")
        ofile.write(spaces + "if(present(desc)) then\n")
        ofile.write(spaces + spaces + "fn = trim(fn) // desc\n")
        ofile.write(spaces + "end if\n")
        ofile.write(spaces + 'fn = trim(fn) // ".txt"\n')
        ofile.write(spaces + 'print *, "Verfication File is :",fn\n')
        ofile.write(spaces + "open(UNIT=10, STATUS='REPLACE', FILE=fn)\n")

        # First insert OpenACC update directives to transfer results from GPU-CPU
        ofile.write(spaces + "if(gpu) then\n")
        acc = "!$acc "

        for v, comp_list in verify_vars.items():
            ofile.write(spaces + acc + "update self(&\n")
            i = 0
            for c in comp_list:
                i += 1
                if i == len(comp_list):
                    name = f"{v}%{c}"
                    c13c14 = bool("c13" in name or "c14" in name)
                    if c13c14:
                        ofile.write(spaces + acc + ")\n")
                    else:
                        ofile.write(spaces + acc + f"{name} )\n")
                else:
                    name = f"{v}%{c}"
                    c13c14 = bool("c13" in name or "c14" in name)
                    if c13c14:
                        continue
                    ofile.write(spaces + acc + f"{name}, &\n")

        ofile.write(spaces + "end if\n")
        ofile.write(spaces + "!! CPU print statements !!\n")
        # generate cpu print statements
        for v, comp_list in verify_vars.items():
            for c in comp_list:
                name = f"{v}%{c}"
                c13c14 = bool("c13" in name or "c14" in name)
                if c13c14:
                    continue
                ofile.write(spaces + f"write(10,*) '{name}',shape({name})\n")
                ofile.write(spaces + f"write(10,*) {name}\n")

        ofile.write(spaces + "close(10)\n")
        ofile.write("end subroutine ")
        ofile.close()
        return

    def examineLoops(
        self,
        global_vars,
        varlist,
        main_sub_dict,
        add_acc=False,
        subcall=False,
        verbose=False,
        adjust_allocation=False,
    ):
        """
        Function that will parse the loop structure of a subroutine
        Add loop parallel directives if desired
        """

        interface_list = (
            get_interface_list()
        )  # move this to a global variable in mod_config?
        if not subcall:
            associate_keys = self.associate_vars.keys()
            temp_global_vars = [key for key in associate_keys]
            global_vars = temp_global_vars[:]
        else:
            if self.name not in main_sub_dict:
                main_sub_dict[self.name] = self

        # Check if associated derived types need to be analyzed
        # convertAssociateDict(self.associate_vars,varlist)

        if adjust_allocation:
            # dict with 'arg' : Subroutine
            # where 'arg' is a local array, and 'sub' is child subroutine that uses it.
            passed_to_sub = {}
            local_array_list = [key for key, val in self.local_variables.items() if val.dim > 0]

        ifile = open(self.filepath, "r")
        lines = ifile.readlines()  # read entire file
        ifile.close()

        loop_start = 0
        loop_end = 0
        regex_do = re.compile(r"\s*(do)\s+\w+\s*(?=[=])", re.IGNORECASE)
        regex_dowhile = re.compile(r"\s*(do while)", re.IGNORECASE)
        regex_enddo = re.compile(r"^\s*(end)\s*(do)", re.IGNORECASE)
        regex_subcall = re.compile(r"^(call)", re.IGNORECASE)

        # Starting parsing lines at the subroutine start
        sublines = lines[self.startline - 1 : self.endline]
        loopcount = 0
        dowhile = False
        lines_to_skip = 0
        for n, line in enumerate(sublines):
            if lines_to_skip > 0:
                lines_to_skip -= 1
                continue
            l, lines_to_skip = lineContinuationAdjustment(sublines, n, verbose)

            # Use RegEx
            m_call = regex_subcall.search(l)
            m_do = regex_do.search(l)
            m_enddo = regex_enddo.search(l)
            m_dowhile = regex_dowhile.search(l)
            if m_dowhile:
                dowhile = True  # Inside do while loop
                if verbose:
                    print("Inside do while loop")
            # If we match subroutine call we should analyze it now so that the
            # DoLoops array has the Loops in order of use.
            #
            if m_call:
                x = l.replace("(", " ").replace(")", " ").split()
                child_sub_name = ""
                ignore_subs = ["cpu_time", "get_curr_date", "endrun", "banddiagonal"]
                if (
                    x[1].lower() not in ignore_subs
                    and "update_vars" not in x[1].lower()
                    and "_oacc" not in x[1].lower()
                ):
                    child_sub_name = x[1]

                    args = getArguments(l)

                    if child_sub_name in interface_list:
                        print("Need to resolve interface!", child_sub_name)
                        sys.exit()
                        child_sub_name, childsub = resolve_interface(
                            self, child_sub_name, args, varlist, verbose=verbose
                        )
                    else:
                        file, startline, endline = find_file_for_subroutine(
                            child_sub_name
                        )
                        childsub = Subroutine(
                            child_sub_name, file, [self.name], startline, endline
                        )

                    # Note that these subroutines have specific versions for
                    # using filter or not using a filter.
                    dont_adjust = [
                        "c2g",
                        "p2c",
                        "p2g",
                        "p2c",
                        "c2l",
                        "l2g",
                        "tridiagonal",
                    ]
                    dont_adjust_string = "|".join(dont_adjust)
                    regex_skip_string = re.compile(
                        f"({dont_adjust_string})", re.IGNORECASE
                    )
                    m_skip = regex_skip_string.search(child_sub_name)

                    if not m_skip:
                        self.update_arg_tree(child_sub_name, args)

                    getLocalVariables(childsub, verbose=verbose)

                    if adjust_allocation:
                        # To adjust memory allocation, need to keep track
                        # of if a Local variable is passed as an argument.
                        # If it's a subcall then must check the arguments as well.
                        for numarg, arg in enumerate(args):
                            if arg in local_array_list:
                                passed_to_sub[arg] = [
                                    childsub,
                                    n + self.startline - 1,
                                    numarg,
                                ]

                        if (
                            subcall
                        ):  # check if an argument is passed to another subroutine
                            arg_list = [v for v in self.arguments]
                            for arg in args:
                                if arg in arg_list:
                                    print(f"{arg} passed to {child_sub_name}")
                    #
                    # get global variable aliases
                    #
                    associate_keys = childsub.associate_vars.keys()
                    temp_global_vars = [key for key in associate_keys]
                    global_vars.extend(temp_global_vars)
                    global_vars = list(dict.fromkeys(global_vars).keys())
                    #
                    # Examine Do loops in child subs
                    #
                    childloops = childsub.examineLoops(
                        global_vars=global_vars,
                        varlist=varlist,
                        main_sub_dict=main_sub_dict,
                        subcall=True,
                        verbose=verbose,
                        adjust_allocation=adjust_allocation,
                        add_acc=False,
                    )
                    if verbose:
                        print(f"Adding {len(childloops)} loops from {childsub.name}")
                    self.loops.extend(childloops)
                    self.child_subroutine_list.append(childsub)
                    self.child_subroutines[child_sub_name] = childsub

            if m_do:
                # get index
                index = m_do.group().split()[1]
                # outer most loop
                if loopcount == 0:
                    if verbose:
                        print(f"============ {self.name} ===============")
                    if verbose:
                        print("New loop at line ", n + self.startline)
                    loop_start = n + self.startline
                    newloop = Loop(loop_start, index, self.filepath, self)
                else:
                    # Found an inner loop
                    newloop.index.append(index)
                    loop_start = n + self.startline
                    newloop.nested += 1
                    newloop.start.append(loop_start)
                    newloop.end.append(0)
                loopcount += 1

            if m_enddo:
                if dowhile:
                    dowhile = False
                    if verbose:
                        print(
                            _bc.WARNING
                            + f"Do while loop in {self.name} ends at {n+self.startline}"
                            + _bc.ENDC
                        )
                    continue
                loopcount -= 1
                if loopcount == 0:
                    if verbose:
                        print("end of outer loop at ", n + self.startline)
                    loop_end = n + self.startline
                    # append loop object:
                    newloop.end[loopcount] = loop_end
                    lstart = newloop.start[0] - self.startline
                    lend = loop_end - self.startline

                    newloop.lines = sublines[lstart : lend + 1]
                    self.loops.append(newloop)
                else:
                    loop_end = n + self.endline
                    newloop.end[loopcount] = loop_end

        # Parse variables used in Loop
        print(f"Parsing variables for Loops in {self.name}")
        for n, loop in enumerate(self.loops):
            if loop.subcall.name is self.name:
                # loops in child subroutines have already been Parsed!
                loop.parseVariablesinLoop(verbose=verbose)
                if loop.reduction:
                    print("loop may contain a race condition:", loop.reduce_vars)

        if adjust_allocation:
            determine_filter_access(self, verbose=False)
            # Check and fix !$acc enter/exit data directives for the subroutine
            self.generate_unstructured_data_regions()

        if subcall:
            return self.loops

        # Create dictionary to hold subroutines found so far
        # and their initial startlines to help adjust for line insertions later.
        sub_dict = {}
        # dictionary of the loops to avoid adding OpenACC directives to the same
        # Loop more than once
        loop_dict = {}
        num_reduction_loops = 0
        ofile = open(f"loops-{self.name}.txt", "w")
        for loop in self.loops:
            loopkey = f"{loop.subcall.name}:L{loop.start[0]}"
            if loopkey not in loop_dict:
                loop_dict[loopkey] = loop
                if loop.reduction:
                    print(
                        _bc.BOLD
                        + _bc.WARNING
                        + f"{loopkey} {loop.reduce_vars}"
                        + _bc.ENDC
                    )
                    ofile.write(f"{loopkey} {loop.reduce_vars}" + "\n")
                    num_reduction_loops += 1
                else:
                    print(loopkey)
                    ofile.write(loopkey + "\n")
            if loop.subcall.name not in sub_dict:
                file, startline, endline = find_file_for_subroutine(loop.subcall.name)
                sub_dict[loop.subcall.name] = startline
        loopids = [k for k in loop_dict.keys()]
        print(
            _bc.OKGREEN
            + f"{len(loopids)} | {num_reduction_loops} w/ Race Conditions"
            + _bc.ENDC
        )
        ofile.write(f"{len(loopids)} | {num_reduction_loops} w/ Race Conditions" + "\n")

        ofile.close()
        count = 0
        for s in main_sub_dict.values():
            arrs = s.LocalVariables["arrays"]
            for arr in arrs:
                v = arrs[arr]
                if "num_" in v.declaration:
                    count += 1
                    print(f"{count}  {s.name}::{v.name} -- {v.declaration}")
        if add_acc:
            lines_adjusted = {}
            for key, loop in loop_dict.items():
                if loop.subcall.name not in lines_adjusted:
                    # This keeps track of the number of line adjustments
                    # inside a given subroutine.  The start of the subroutine
                    # is rechecked inside addOpenACCFlags function
                    lines_adjusted[loop.subcall.name] = 0
                file, startline, endline = find_file_for_subroutine(loop.subcall.name)
                subline_adjust = startline - sub_dict[loop.subcall.name]
                loop.addOpenACCFlags(lines_adjusted, subline_adjust, key)

        # Note: Can't use List Comprehension here with class attributes
        var_list = []
        local_vars_only = []
        for loop in self.loops:
            for key in loop.vars.keys():
                var_list.append(key)
                if key not in global_vars:
                    local_vars_only.append(key)

        var_list = list(dict.fromkeys(var_list).keys())
        local_vars_only = list(dict.fromkeys(local_vars_only).keys())

        # print only the global variable list:
        global_loop_vars = []
        all_array_vars = []
        for v in var_list:
            if v in global_vars or "filter" in v:
                global_loop_vars.append(v)

        return None

    def exportReadWriteVariables(self):
        """
        Writes the variables for read and write to a separate data file
        """
        spaces = "     "
        read_flat = []
        write_flat = []
        all_flat = []
        maxlen = 0
        # read variables
        print("============= exportReadWriteVariables ===================")
        for varname, components in self.elmtype_access_sum.items():
            for c in components:
                var = f"{varname}%{c}"
                read_flat.append(var)
                all_flat.append(var)
                if len(var) > maxlen:
                    maxlen = len(var)

        output_list = []
        # header
        ofile = open(f"{self.name}-ReadWriteVars.dat", "w")
        header = f"{'Variable':<{maxlen}} {'Status':5}"
        output_list.append(header)
        ofile.write(header + "\n")
        for var in all_flat:
            status = ""
            if var in read_flat:
                status += "r"
            if var in write_flat:
                status += "w"
            if len(status) < 2:
                status += "o"
            string = f"{var:<{maxlen}} {status:5}\n"
            ofile.write(string)
        ofile.close()

        return None

    def update_arg_tree(self, childsubname, args):
        """
        Checks if any of the subroutines arguments
        or local variables are passed to any subroutines.
        """
        from config import _bc


        arrays = self.local_variables

        for v in args:
            if "=" in v:
                kw, var = v.split("=")
                kw = kw.strip()
                var = var.strip()
                if var not in arrays:
                    continue
                self.local_variables[var].keyword = kw
                self.local_variables[var].subs.append(childsubname)
                self.VariablesPassedToSubs.setdefault(childsubname, []).append(
                    self.local_variables[var]
                )
            else:
                var = v.strip()
                if var not in arrays:
                    continue
                self.local_variables[var].keyword = ""
                self.local_variables[var].subs.append(childsubname)
                self.VariablesPassedToSubs.setdefault(childsubname, []).append(
                    self.local_variables[var]
                )

    def generate_unstructured_data_regions(self, remove=True) -> None:
        """
        Function generates appropriate enter and exit data
        directives for the local variables of this Subroutine.

        First step is to remove any existing directives
        Next, create new directives from local variable list
        Compare new and old directives and overwrite if they are different
        """
        # Open File:
        if os.path.exists(spel_dir + "modified-files/" + self.filepath):
            print("Modified file found")
            print(
                _bc.BOLD
                + _bc.WARNING
                + f"Opening file "
                + spel_dir
                + "modified-files/"
                + self.filepath
                + _bc.ENDC
            )
            ifile = open(spel_dir + "modified-files/" + self.filepath, "r")
        else:
            print(_bc.BOLD + _bc.WARNING + f"Opening file{self.filepath}" + _bc.ENDC)
            ifile = open(self.filepath, "r")

        lines = ifile.readlines()

        ifile.close()

        regex_enter_data = re.compile(r"^\s*\!\$acc enter data", re.IGNORECASE)
        regex_exit_data = re.compile(r"^\s*\!\$acc exit data", re.IGNORECASE)

        lstart = self.startline - 1
        lend = self.endline
        old_enter_directives = []
        old_exit_directives = []
        if remove:
            ln = lstart
            while ln < lend:
                line = lines[ln]
                match_enter_data = regex_enter_data.search(line)
                match_exit_data = regex_exit_data.search(line)
                if match_enter_data:
                    directive_start = ln
                    old_enter_directives.append(line)  # start of enter data directive
                    line = line.rstrip("\n")
                    line = line.strip()
                    while line.endswith("&"):
                        ln += 1
                        line = lines[ln]
                        old_enter_directives.append(line)  # end of enter data directive
                        line = line.rstrip("\n")
                        line = line.strip()
                    directive_end = ln
                    del lines[directive_start : directive_end + 1]
                    num_lines_removed = directive_end - directive_start + 1
                    lend -= num_lines_removed
                    ln -= num_lines_removed
                    print(f"Removed {num_lines_removed} enter data lines")
                if match_exit_data:
                    directive_start = ln  # start of exit data directive
                    old_exit_directives.append(line)
                    line = line.rstrip("\n")
                    line = line.strip()
                    while line.endswith("&"):
                        ln += 1
                        line = lines[ln]
                        old_exit_directives.append(line)
                        line = line.rstrip("\n")
                        line = line.strip()
                    directive_end = ln  # end of exit data directive
                    del lines[directive_start : directive_end + 1]
                    num_lines_removed = directive_end - directive_start + 1
                    lend -= num_lines_removed
                    ln -= num_lines_removed
                    print(f"Removed {num_lines_removed} exit data lines")
                ln += 1

        # Create New directives
        vars = []  # list to hold all vars needed to be on the device
        arrays_dict = self.local_variables["arrays"]
        for k, v in arrays_dict.items():
            varname = v.name
            dim = v.dim
            li_ = [":"] * dim
            dim_str = ",".join(li_)
            dim_str = "(" + dim_str + ")"
            print(f"adding {varname}{dim_str} to directives")
            vars.append(f"{varname}{dim_str}")

        # Only add scalars to if they are a reduction variables
        # Only denoting that by if it has "sum" in the name
        for v in self.local_variables["scalars"]:
            varname = v.name
            for loop in self.loops:
                if loop.subcall.name == self.name:
                    if varname in loop.reduce_vars and varname not in vars:
                        print(f"Adding scalar {varname} to directives")
                        vars.append(varname)

        num_vars = len(vars)
        if num_vars == 0:
            print(f"No Local variables to make transfer to device, returning")
            return None
        else:
            print(f"Generating create directives for {num_vars} variables")

        # Get appropriate indentation for the new directives:
        padding = ""
        first_line = 0

        for ln in range(lstart, lend):
            line = lines[ln]
            # Before ignoring comments, check if it's an OpenACC directive
            m_acc = re.search(r"\s*(\!\$acc routine seq)", line)
            if m_acc:
                sys.exit("Error: Trying to add data directives to an OpenACC routine")

            m_acc = re.search(
                r"\s*(\!\$acc)\s+(parallel|enter|update)", line, re.IGNORECASE
            )
            if m_acc and first_line == 0:
                first_line = ln

            l = line.split("!")[0]
            l = l.strip()
            if not l:
                continue

            m_use = re.search(
                r"^(implicit|use|integer|real|character|logical|type\()", line.lstrip()
            )
            if m_use and not padding:
                padding = " " * (len(line) - len(line.lstrip()))
            elif padding and not m_use and first_line == 0:
                first_line = ln

            if ln == lend - 1 and not padding:
                sys.exit("Error: Couldn't get spacing")

        new_directives = []

        for v in vars[0 : num_vars - 1]:
            new_directives.append(padding + f"!$acc {v}, &\n")
        new_directives.append(padding + f"!$acc {vars[num_vars-1]})\n\n")

        new_enter_data = [padding + "!$acc enter data create(&\n"]
        new_enter_data.extend(new_directives)
        #
        new_exit_data = [padding + "!$acc exit data delete(&\n"]
        new_exit_data.extend(new_directives)

        if (
            new_enter_data != old_enter_directives
            or new_exit_data != old_exit_directives
        ):
            # Insert the enter data directives
            if self.associate_end != 0:
                # insert new directives just after last associate statement:
                for l in reversed(new_enter_data):
                    lines.insert(self.associate_end + 1, l)
            else:  # use first_line found above
                for l in reversed(new_enter_data):
                    lines.insert(first_line, l)
            lend += len(new_enter_data)
            print(
                _bc.BOLD + _bc.WARNING + f"New Subroutine Ending is {lend}" + _bc.ENDC
            )
            # Inster the exit data directives
            if self.associate_end != 0:
                end_associate_ln = 0
                regex_end = re.compile(r"^(end associate)", re.IGNORECASE)
                for ln in range(lend, lstart, -1):
                    m_end = regex_end.search(lines[ln].lstrip())
                    if m_end:
                        end_associate_ln = ln
                        break
                for l in reversed(new_exit_data):
                    lines.insert(end_associate_ln, l)
            else:
                for l in reversed(new_exit_data):
                    lines.insert(lend - 1, l)
            lend += len(new_exit_data)
            print(
                _bc.BOLD + _bc.WARNING + f"New Subroutine Ending is {lend}" + _bc.ENDC
            )

            # Overwrite File:
            if "modified-files" in self.filepath:
                print(
                    _bc.BOLD
                    + _bc.WARNING
                    + f"Writing to file {self.filepath}"
                    + _bc.ENDC
                )
                ofile = open(self.filepath, "w")
            else:
                print(
                    _bc.BOLD
                    + _bc.WARNING
                    + "Writing to file "
                    + spel_dir
                    + "modified-files/"
                    + self.filepath
                    + _bc.ENDC
                )
                ofile = open(spel_dir + "modified-files/" + self.filepath, "w")

            ofile.writelines(lines)
            ofile.close()
        else:
            print(_bc.BOLD + _bc.WARNING + "NO CHANGE" + _bc.ENDC)
        return None

    def parse_arguments(
        self,
        sub_dict: dict[str, Subroutine],
        type_dict: dict[str, DerivedType],
        verbose=False,
    ):
        """
        Function that will analyze the variable status for only the arguments
            'sub.arguments_read_write' : { `arg` : ReadWrite}
        """
        func_name = "( parse_arguments )"
        associate_set: set[str] = set()
        var_dict = self.arguments.copy()

        for key, val in self.associate_vars.items():
            if val.split("%")[0] in var_dict:
                associate_set.add(key)
        if associate_set:
            var_inst_dict: dict[str, DerivedType] = {
                var.name: type_dict[var.type]
                for var in var_dict.values()
                if is_derived_type(var) and var.type in type_dict
            }
            for key in associate_set:
                field_name = self.associate_vars[key]
                if "%" in field_name:
                    dtype_var = get_component(var_inst_dict, field_name)
                    if dtype_var:
                        var_dict[key] = dtype_var
                else:
                    argvar = self.arguments[field_name].copy()
                    argvar.name = key
                    test = expand_dtype([argvar],type_dict)
                    var_dict.update(test)

        args_accessed = analyze_sub_variables(
            self,
            sub_dict,
            var_dict,
            mode=ArgLabel.dummy,
            verbose=verbose,
        )
        # Substitute any associated pointer names
        for key in associate_set:
            full_name = self.associate_vars[key]
            ptr_status = args_accessed.pop(key, None)
            if ptr_status:
                args_accessed.setdefault(full_name, []).extend(ptr_status)

        for key in associate_set:
            full_name = self.associate_vars[key]
            regex_alias = re.compile(rf"\b{key}\b")
            for arg in list(args_accessed.keys()):
                if regex_alias.search(arg):
                    alias, field = arg.split("%")
                    arg_status = args_accessed.pop(arg)
                    new_alias = regex_alias.sub(full_name,alias)
                    new_name = "%".join([new_alias, field])
                    args_accessed.setdefault(new_name,[]).extend(arg_status)

        self.arg_access_by_ln = args_accessed.copy()
        # All args should have been processed. Store information into Subroutine
        arg_status_summary = summarize_read_write_status(args_accessed)
        for arg, status in arg_status_summary.items():
            self.arguments_read_write[arg] = ReadWrite(
                status, -999, LineTuple(ln=-999, line="")
            )
            if "%" in arg:
                inst, _ = arg.split("%")
                if inst not in self.arguments_read_write:
                    self.arguments_read_write[inst] = ReadWrite(
                        status, -999, LineTuple(ln=-999, line="")
                    )
                else:
                    val = self.arguments_read_write[inst].status
                    cstat = combine_status(status, val)
                    self.arguments_read_write[inst].status = cstat

        if not self.arguments_read_write and not self.arguments:
            self.logger.error(f"{func_name}::ERROR: Failed to analyze arguments")
            sys.exit(1)

        for arg in self.arguments:
            if arg not in self.arguments_read_write:
                self.arguments_read_write[arg] = ReadWrite(
                    "-", -999, LineTuple(ln=-999, line="")
                )
        return None
