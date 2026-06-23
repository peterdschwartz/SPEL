import re
import subprocess
from collections import defaultdict

from scripts.analyze_subroutines import Subroutine
from scripts.config import ELM_SRC
from scripts.fortran_parser.boolen_expression import infer_condition_expectations
from scripts.fortran_parser.spel_ast import NameListStatement, Program
from scripts.fortran_parser.spel_parser import Parser
from scripts.nml.namelist_cascade import NML_CASCADES
from scripts.types import LineTuple, LogicalLineIterator, NameList


def find_nml_ifs(sub_dict: dict[str, Subroutine], nml_dict: dict[str, NameList]):
    """
    Function to find the namelist vars used in if statements and their expected value
    """
    nml_str = "|".join(list(nml_dict.keys()))
    regex_nml = re.compile(rf"\b({nml_str})\b")

    deps = list(NML_CASCADES.keys())
    regex_deps = re.compile(rf"\b({'|'.join(deps)})\b")
    for sub in sub_dict.values():
        for if_node in sub.flat_ifs:
            cond = str(if_node.condition)
            nml_vars = regex_nml.findall(cond)
            temp = {v: nml_dict[v] for v in nml_vars}
            if_node.nml_vars.update(temp)

            # check nml dependents:
            dep_vars = regex_deps.findall(cond)
            if_node.nml_cascades.update({v: NML_CASCADES[v] for v in dep_vars})
            if temp or dep_vars:
                variables = { v for v in dep_vars + nml_vars }
                infer_condition_expectations(expr=if_node.condition,variables=variables)


    return


def check_calltree_for_nml_guarded_vars(
    root_sub: Subroutine,
    sub_dict: dict[str, Subroutine],
):
    """
    Given a root subroutine node, traverse the calltree and determine
    if any global variables are ONLY accessed under certain NML options
    """
    from pprint import pprint

    exclusive_by_sub: dict[str, dict] = defaultdict(dict)
    if root_sub.abstract_call_tree:
        for subnode in root_sub.abstract_call_tree.traverse_postorder():
            subname = subnode.node.subname
            sub = sub_dict[subname]
            exclusive = sub.elmtype_accesses_exclusive_to_namelist_ifs()
            if exclusive:
                exclusive_by_sub[subname] = {key: val for key, val in exclusive.items()}

    pprint(exclusive_by_sub)

    return


def find_all_namelist() -> dict[str, NameList]:
    """
    Find all namelist variables across ELM.
    NOTE: Grep is 1-based indexed for lineno's
    """

    output = subprocess.getoutput(
        rf'grep -rin --include=*.F90 --exclude-dir=external_models/ "namelist\s*\/" {ELM_SRC}'
    )
    namelist_dict: dict[str, NameList] = {}
    if output.strip() == "":
        return namelist_dict
    entries: dict[str, list[int]] = defaultdict(list)

    for line in output.split("\n"):
        line = line.split(":")
        filename = line[0]
        line_number = int(line[1])
        entries[filename].append(line_number)

    nml_lines: list[LineTuple] = []
    for fn, lns in entries.items():
        in_stream = open(fn, "r")
        lines = in_stream.readlines()
        line_iter = LogicalLineIterator(
            lines=[LineTuple(ln=ln, line=line) for ln, line in enumerate(lines)]
        )
        for ln in lns:
            nml_lines.append(line_iter.get_full_line(ln))

    # Now Parse!
    parser = Parser(lines=nml_lines)
    program = parser.parse_program()
    for stmt in program.statements:
        assert isinstance(stmt, NameListStatement), "Expected only NameListStatement"
        for var in stmt.vars:
            namelist_dict[var] = NameList(name=var, group=stmt.namelist_group)
    return namelist_dict
