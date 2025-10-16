import re
import subprocess

from scripts.analyze_subroutines import Subroutine
from scripts.config import ELM_SRC
from scripts.nml.analyze_elm import single_line_unwrapper
from scripts.nml.analyze_ifs import set_default
from scripts.types import NameList


def find_nml_ifs(sub_dict: dict[str, Subroutine], nml_dict: dict[str, NameList]):
    nml_str = "|".join(list(nml_dict.keys()))
    regex_nml = re.compile(rf"\b({nml_str})\b")

    for sub in sub_dict.values():
        for if_node in sub.flat_ifs:
            cond = str(if_node.condition)
            nml_vars = regex_nml.findall(cond)
            temp = {v: nml_dict[v] for v in nml_vars}
            if_node.nml_vars.update(temp)

    return


def find_all_namelist() -> dict[str, NameList]:
    """
    Find all namelist variables across ELM
    """

    output = subprocess.getoutput(
        f'grep -rin --include=*.F90 --exclude-dir=external_modules/ "namelist\s*\/" {ELM_SRC}'
    )
    namelist_dict = {}
    if output.strip() == "":
        return namelist_dict
    for line in output.split("\n"):
        line = line.split(":")
        filename = line[0]
        line_number = int(line[1]) - 1
        info = line[2]
        full_line, _ = single_line_unwrapper(filename, line_number)

        group = re.findall(r"namelist\s*\/\s*(\w+)\s*\/", info, re.IGNORECASE)[0]
        flags = full_line.split("/")[-1].split(",")
        for flag in flags:
            f = NameList()
            f.name = flag.strip()
            f.group = group
            f.filepath = filename
            f.ln = line_number

            namelist_dict[flag.strip()] = f
    return namelist_dict


def generate_namelist_dict(mod_dict, namelist_dict, ifs, subroutine_calls):
    """
    Find used namelists variables within the if-condition
    Attach variable objects to each namelist
    """
    for mod in mod_dict.values():
        for namelist in namelist_dict.keys():
            current_namelist_var = namelist_dict[namelist]

            matching_var = [v for v in mod.global_vars if v.name == namelist]
            if matching_var:
                var = matching_var[0]
                if_blocks = []
                for pair in ifs:
                    if re.search(
                        r"\b\s*{}\b\s*".format(re.escape(namelist)), pair.condition
                    ):
                        if_blocks.append(pair)

                    for index, call in enumerate(pair.calls):
                        ln = call[0]
                        name = call[1]
                        if isinstance(name, tuple):
                            continue

                        if subroutine_calls.get(name) != None:
                            for s in subroutine_calls[name]:
                                if ln == s.ln:
                                    pair.calls[index] = (name, s)

                current_namelist_var.if_blocks.extend(if_blocks)
                current_namelist_var.variable = var


def change_default(if_block, namelist, namelist_variable, value):
    """
    Change namelist_variable's default to value
    if_block is parent ifblocks
    """
    namelist[namelist_variable].variable.default_value = value
    set_default(if_block, namelist)
