import re
import sys
from logging import Logger
from typing import Optional

from scripts.config import spel_output_dir
from scripts.types import FunctionReturn, ParseState, PreProcTuple, SubInit, SubStart
from scripts.utilityFunctions import intrinsic_type, split_func_line


def check_function_start(
    line: str,
    ln_pair: PreProcTuple,
    parent: str,
):
    """
    Function to parse function start for result type and name
    """

    split_line = split_func_line(line)
    regex = re.compile(r"(?<=\()[\w\s,]+(?=\))")
    reg_res = re.compile(r"result\s*\(\w+\)")
    regex_paren = re.compile(r"\((.+)\)")  # for removing array of struct index

    func_type, func_keyword, func_rest = split_line
    if func_type:
        # Definition:  <function type> function <name>(<args>)
        if "type" in func_type:
            func_type = regex.search(func_type).group()
        else:
            func_type = intrinsic_type.search(func_type).group()
        args_and_res = regex.findall(func_rest)
        m_ = reg_res.search(func_rest)
        func_name = func_rest.split("(")[0].strip()
        if m_:
            res = regex_paren.search(m_.group())
            res = res.group().strip()
            res = res[1:-1]
            func_result = res
        else:
            func_result = func_name

    else:
        # is it worth getting this here or after local variables are parsed?
        func_type = ""
        func_name = func_rest.split("(")[0].strip()
        # Definition: function <name>(<args>) result(<result>) || function <name>(<args>)
        args_and_res = regex.findall(func_rest)
        args_and_res = [v.strip() for v in args_and_res]
        if len(args_and_res) == 2:
            args, func_result = args_and_res
        else:
            func_result = func_name.strip()

    return FunctionReturn(
        return_type=func_type,
        name=func_name,
        result=func_result,
        start_ln=ln_pair.ln,
        cpp_start=ln_pair.cpp_ln,
        parent=parent,
    )


def create_init_obj(
    state: ParseState,
    logger: Logger,
    sub_start: Optional[SubStart] = None,
    func_init: Optional[FunctionReturn] = None,
) -> None:
    """
    Function to instantiate function if not in sub_dict
    """
    func_name = "( create_function )"
    fn = state.path
    if state.cpp_file:
        base_fn = fn.split("/")[-1]
        new_fn = f"{spel_output_dir}cpp_{base_fn}"
    else:
        new_fn = ""

    assert state.curr_line

    end_ln = state.curr_line.ln
    cpp_end = state.line_it.i if state.host_program == -1 else state.host_program

    sub_name = None
    start_ln = None
    parent = ""
    if sub_start:
        sub_name = sub_start.subname
        start_ln = sub_start.start_ln
        cpp_start = sub_start.cpp_ln if state.cpp_file else None
        parent = sub_start.parent
    elif func_init:
        sub_name = func_init.name
        start_ln = func_init.start_ln
        cpp_start = func_init.cpp_start if state.cpp_file else None
        parent = func_init.parent

    assert sub_name and start_ln, f"{sub_name} Error {sub_name} or {start_ln} empty"

    keyword = "SUBROUTINE" if not func_init else "FUNCTION"
    if sub_name in state.sub_init_dict:
        logger.error(
            f"{func_name}::{sub_name} Already in dictionary"
            + f"{func_name}Adding {keyword} {sub_name}:\n{' '*len(func_name)} fn: {fn}\n"
            + f"{' '*len(func_name)} ln: L{start_ln}-{end_ln}"
        )
        sys.exit(1)

    init_obj = SubInit(
        name=sub_name,
        mod_name=state.module_name,
        fort_mod=state.fort_mod,
        mod_lines=[],
        file=fn,
        cpp_fn=new_fn,
        start=start_ln,
        end=end_ln,
        cpp_start=cpp_start,
        cpp_end=cpp_end,
        function=func_init,
        parent=parent,
    )

    state.sub_init_dict[f"{state.module_name}::{sub_name}"] = init_obj
    state.host_program = -1 # reset

    return None
