import re
import subprocess
from collections import defaultdict
from pprint import pprint

from spel.scripts.analyze_subroutines import Subroutine
from spel.scripts.config import ELM_SRC
from spel.scripts.fortran_parser.boolen_expression import (
    NO_EXPECTATION,
    ConditionExpectation,
    Expectation,
    expected_constraints,
    infer_condition_expectations,
    simplify_expectations,
)
from spel.scripts.fortran_parser.spel_ast import NameListStatement, Program
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.nml.namelist_cascade import NML_CASCADES
from spel.scripts.types import FlatIfs, LineTuple, LogicalLineIterator, NameList


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
                variables = {v for v in dep_vars + nml_vars}
                if_node.condtional_expectation = infer_condition_expectations(
                    expr=if_node.condition,
                    variables=variables,
                )
                for v in variables:
                    if_node.expected_namelist_values |= expected_constraints(
                        if_node.condtional_expectation, v
                    )

    return


def check_sub_for_nml_guarded_vars(root_sub: Subroutine)-> dict[ConditionExpectation,list[str]]:
    """
    Given a root subroutine node, traverse the calltree and determine
    if any global variables are ONLY accessed under certain NML options
    """
    exclusive = root_sub.elmtype_accesses_exclusive_to_namelist_ifs()
    var_dict: dict[str, ConditionExpectation] = {}
    if exclusive:
        for v, flatifs in exclusive.items():
            combined_expectations: set[Expectation] = set()
            for _if in flatifs:
                combined_expectations.update(_if.expected_namelist_values)
            combined = simplify_expectations(combined_expectations)
            if combined != NO_EXPECTATION:
                var_dict[v] = combined

    temp = defaultdict(list)
    for v, cond in var_dict.items():
        temp[cond].append(v)

    return temp


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
