import subprocess as sp
from typing import Optional, Tuple

from scripts.config import ELM_SRC

interface_list: list[str] = []

map_module_name_to_fpath: dict[str, Optional[str]] = {}
map_fpath_to_module_name: dict[str, Tuple[int, str]] = {}

map_module_lines: dict[str, list[str]] = {}

map_module_head: dict[str, int] = {}


def populate_interface_list():
    """
    returns a list of all interfaces
    """
    cmd = f'grep -rin --exclude-dir={ELM_SRC}external_models/ -E "^[[:space:]]+(interface)" {ELM_SRC}*'
    output = sp.getoutput(cmd)
    output = output.split("\n")
    global interface_list
    if interface_list:
        print("Warning - Trying to recalculate interface list")
    for el in output:
        el = el.split()
        interface = el[2]
        interface_list.append(interface.lower())

    return
