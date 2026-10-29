import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from pprint import pprint
import sys

from spel.scripts.analyze_subroutines import Subroutine
from spel.scripts.config import ELM_SRC
from spel.scripts.DerivedType import DerivedType, get_component
from spel.scripts.fortran_parser.boolen_expression import (
    NO_EXPECTATION,
    AllOf,
    AnyOf,
    ConditionExpectation,
    Expectation,
    expected_constraints,
    infer_condition_expectations,
    simplify,
)
from spel.scripts.fortran_parser.spel_ast import NameListStatement, Program
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.nml.namelist_cascade import NML_CASCADES
from spel.scripts.types import FlatIfs, LineTuple, LogicalLineIterator, NameList
from spel.scripts.utilityFunctions import Variable
from spel.scripts.validate_access import merge_guards


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
                if_node.condtional_expectation = cascade_to_triggers(
                    infer_condition_expectations(
                        expr=if_node.condition,
                        variables=variables,
                    )
                )
                variables |= {t for v in dep_vars for t in NML_CASCADES[v].trigger_names}
                for v in variables:
                    if_node.expected_namelist_values |= expected_constraints(
                        if_node.condtional_expectation, v
                    )

    return


def cascade_to_triggers(condition: ConditionExpectation) -> ConditionExpectation:
    """
    Replace expectations on cascade variables by what they imply for the
    trigger namelist variable: the cascade variable may be an array element
    (lun_pp%ispolygon(l)) or a filter count (num_pcropp) that a guard can't
    evaluate. Negations are already at the leaves, so the weaker condition
    still holds wherever the original does.
    """
    match condition:
        case Expectation(var, constraint) if var in NML_CASCADES:
            dep = NML_CASCADES[var]
            trigger = dep.trigger_constraint(constraint)
            if trigger is None:
                return NO_EXPECTATION
            return Expectation(dep.trigger_name, trigger)
        case AllOf(items):
            return AllOf(tuple(cascade_to_triggers(c) for c in items))
        case AnyOf(items):
            return AnyOf(tuple(cascade_to_triggers(c) for c in items))
    return condition


def nml_guards(
    root_sub: Subroutine,
    _memo: dict[str, dict[str, ConditionExpectation]] | None = None,
    _active: frozenset[str] = frozenset(),
) -> dict[str, ConditionExpectation]:
    """
    inst%field -> namelist condition under which root_sub (incl. its children,
    via their call sites) accesses it, for fields ONLY accessed under namelist ifs.
    An access at a call site is guarded by the enclosing namelist if AND the
    callee's own guard for the field (e.g. FracH2OSfc reads col_ws%iwp_* only
    under lun_pp%ispolygon, i.e. use_polygonal_tundra).
    """
    memo = {} if _memo is None else _memo
    if root_sub.id in memo:
        return memo[root_sub.id]
    active = _active | {root_sub.id}

    callee_guards: dict[int, dict[str, ConditionExpectation]] = {}
    for ln, desc in root_sub.sub_call_desc.items():
        child = root_sub.child_subroutines.get(desc.fn)
        if child is not None and child.id not in active:
            callee_guards[ln] = nml_guards(child, memo, active)

    var_dict: dict[str, ConditionExpectation] = {}
    for v, accesses in root_sub.elmtype_access_by_ln.items():
        conditions: list[ConditionExpectation] = []
        for rw in accesses:
            if_ = root_sub.namelist_if_at(rw.ln)
            local = if_.condtional_expectation if if_ else NO_EXPECTATION
            callee = callee_guards.get(rw.ln, {}).get(v, NO_EXPECTATION)
            conditions.append(AllOf((local, callee)))
        combined = simplify(AnyOf(tuple(conditions)))
        if combined != NO_EXPECTATION:
            var_dict[v] = combined
    memo[root_sub.id] = var_dict
    return var_dict


def check_subs_for_nml_guarded_vars(
    root_subs: Iterable[Subroutine],
    instance_dict: dict[str, DerivedType],
) -> dict[ConditionExpectation, list[Variable]]:
    """
    Given the unit test's root subroutines, find the global variables ONLY
    accessed under certain NML options (guarded in every root that accesses them)
    """
    root_subs = list(root_subs)
    var_dict = merge_guards(
        (root_sub.elmtype_access_by_ln, nml_guards(root_sub)) for root_sub in root_subs
    )
    accessed = {v for root_sub in root_subs for v in root_sub.elmtype_access_by_ln}

    # Guards are looked up by the type's own component (identity, see
    # FunctionalUnitTest.guards_for_variable), which every instance of the type
    # shares: c13_col_cf%x used only under use_c13 must not guard col_cf%x. A
    # component is guarded only if every instance accessing it is, by any of
    # their conditions.
    by_comp: dict[int, tuple[Variable, list]] = {}
    for v in sorted(accessed | set(var_dict)):
        inst, _, field = v.partition("%")
        # the type's own component (get_component returns a copy)
        dtype = instance_dict.get(inst)
        var = dtype.components.get(field) if dtype is not None else None
        if var is None:
            if v not in var_dict:
                continue  # unguarded and not a shared component: nothing to clear
            var = get_component(instance_dict, inst, field)
        if var is None:
            # an instance of a type the analysis does not know (alm_fates: the FATES interface
            # module is excluded): nothing to guard, and no reason to drop every other guard
            continue
        by_comp.setdefault(id(var), (var, []))[1].append(var_dict.get(v))

    temp = defaultdict(list)
    for var, conds in by_comp.values():
        if any(cond is None for cond in conds):
            continue
        unique = list(dict.fromkeys(conds))
        cond = unique[0] if len(unique) == 1 else simplify(AnyOf(tuple(unique)))
        temp[cond].append(var)

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
    file_of_ln: dict[int, str] = {}
    for fn, lns in entries.items():
        in_stream = open(fn, "r")
        lines = in_stream.readlines()
        line_iter = LogicalLineIterator(
            lines=[LineTuple(ln=ln, line=line) for ln, line in enumerate(lines)]
        )
        for ln in lns:
            full = line_iter.get_full_line(ln)
            file_of_ln[full.ln] = fn
            nml_lines.append(full)

    # Now Parse!
    parser = Parser(lines=nml_lines)
    program = parser.parse_program()
    for stmt in program.statements:
        assert isinstance(stmt, NameListStatement), "Expected only NameListStatement"
        for var in stmt.vars:
            nml = NameList(name=var, group=stmt.namelist_group)
            nml.filepath = file_of_ln.get(stmt.lineno, "")
            namelist_dict[var] = nml
    return namelist_dict
