"""
elm_drv (elm_driver.F90) is the top-most parent of every unit-test routine.
A root routine's dummies are bound to whatever elm_drv passes at its call
sites: each global actual gets the dummy's overall status at the call line,
along with the root's own (transitive) global accesses.

elm_driver is parsed read-only, outside the unit test's module dictionary
(its file is not part of the unit test).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

from spel.scripts import edit_files
from spel.scripts.edit_files import (
    apply_preprocessor,
    get_used_mods,
    remove_cpp_directives,
)
from spel.scripts.fortran_modules import get_filename_from_module
from spel.scripts.fortran_parser.scope_walk import (
    CallEvent,
    FunctionRef,
    SubroutineRecord,
    walk_subroutine,
)
from spel.scripts.fortran_parser.spel_ast import (
    SemanticError,
    SubroutineDefinitionConstruct,
)
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.logging_configs import get_logger
from spel.scripts.module_resolver import ModuleResolver, ModuleScopes
from spel.scripts.record_access import (
    AccessDict,
    AccessMapper,
    AccessMaps,
    ArgBinding,
    callsite_globals,
)
from spel.scripts.types import LineTuple
from spel.scripts.utilityFunctions import unwrap_section

if TYPE_CHECKING:
    from spel.scripts.analyze_subroutines import Subroutine
    from spel.scripts.fortran_modules import FortranModule

DRIVER_MODULE = "elm_driver"
DRIVER_ROUTINE = "elm_drv"

regex_routine = re.compile(
    r"^\s*(?:(?:pure|elemental|recursive|module)\s+)*"
    r"(?:(?:integer|real|logical|character|complex|type\s*\([^)]*\))[^:]*?\s+)?"
    r"(subroutine|function)\s+(\w+)",
    re.IGNORECASE,
)


class CallerRoutine:
    """The parts of Subroutine that AccessMapper needs, for a routine outside the unit test."""

    def __init__(self, module: str, name: str, lines: list[LineTuple]):
        self.module = module
        self.name = name
        self.id = f"{module}::{name}"
        self.library = False
        self.host_id: Optional[str] = None
        self.sub_call_desc: dict = {}
        self.sub_lines = lines
        self.record: Optional[SubroutineRecord] = None
        self.record_access: Optional[AccessMaps] = None
        program = Parser(lines=lines, logger=f"Parser-{self.id}").parse_program()
        tree = program.statements[0] if len(program.statements) == 1 else None
        if not isinstance(tree, SubroutineDefinitionConstruct) or tree.name != name:
            raise SemanticError(f"{self.id}: expected a single routine named {name}")
        self.syntax_tree = tree

    def walk_syntax_tree(self, scopes: ModuleScopes) -> SubroutineRecord:
        if self.record is None:
            ranges = [(self.sub_lines[0].ln, self.sub_lines[-1].ln)]
            resolver = ModuleResolver(
                self.module, scopes.mod_dict, scopes.sub_dict, line_ranges=ranges, scopes=scopes
            )
            self.record = walk_subroutine(self.syntax_tree, resolver=resolver)
        return self.record


def load_module(mod_name: str) -> tuple[FortranModule, list[LineTuple]]:
    """
    FortranModule for `mod_name` (not added to any unit test) and its
    preprocessed logical lines, numbered by line in the original file.
    """
    fn = get_filename_from_module(mod_name)
    if fn is None:
        raise SemanticError(f"no source file for module {mod_name}")
    # get_used_mods records unavailable modules in a global set; this module
    # is not part of the unit test, so keep that set unchanged
    saved_bad = set(edit_files.bad_modules)
    try:
        _, parsed = get_used_mods(ifile=fn, mods=[], singlefile=True, mod_dict={})
    finally:
        edit_files.bad_modules.clear()
        edit_files.bad_modules.update(saved_bad)
    fort_mod = parsed[mod_name]

    logger = get_logger("driver_callsites")
    text = [""] * fort_mod.num_lines
    for lt in remove_cpp_directives(apply_preprocessor(Path(fn)), fn, logger):
        text[lt.ln] = lt.line
    lines = unwrap_section(text, startln=-1)
    fort_mod.module_lines = lines
    fort_mod.subroutines = {
        f"{mod_name}::{m.group(2).lower()}"
        for lt in lines
        if lt.ln > fort_mod.end_of_head_ln and (m := regex_routine.match(lt.line))
    }
    return fort_mod, lines


def routine_lines(lines: list[LineTuple], name: str) -> list[LineTuple]:
    start = re.compile(rf"^\s*subroutine\s+{name}\b", re.IGNORECASE)
    end = re.compile(rf"^\s*end\s*subroutine\s+{name}\b", re.IGNORECASE)
    out: list[LineTuple] = []
    for lt in lines:
        if out or start.match(lt.line):
            out.append(lt)
            if end.match(lt.line):
                return out
    raise SemanticError(f"routine {name} not found")


def driver_callsites(
    roots: Iterable[Subroutine],
    sub_dict: dict[str, Subroutine],
    mod_dict: dict[str, FortranModule],
) -> tuple[dict[str, AccessDict], list[ArgBinding]]:
    """
    Global accesses of each root routine at its elm_drv call sites, and
    elm_drv's argument bindings. Only routines already mapped for the unit
    test are bound; other callees of elm_drv are treated as unknown.
    """
    drv_mod, lines = load_module(DRIVER_MODULE)
    drv = CallerRoutine(DRIVER_MODULE, DRIVER_ROUTINE, routine_lines(lines, DRIVER_ROUTINE))
    scopes = ModuleScopes({**mod_dict, DRIVER_MODULE: drv_mod}, sub_dict)
    mapped = {k: s for k, s in sub_dict.items() if s.record_access is not None}
    mapper = AccessMapper(mapped, scopes)
    dmaps = mapper.maps(drv)
    rec = drv.walk_syntax_tree(scopes)

    call_lines: dict[str, set[int]] = {}
    for e in rec.events:
        if isinstance(e, (CallEvent, FunctionRef)):
            callee = mapper.callee(drv, rec, e)
            if callee is not None:
                call_lines.setdefault(callee.id, set()).add(e.ln)
    access = {
        root.id: callsite_globals(dmaps, call_lines[root.id])
        for root in roots
        if root.id in call_lines
    }
    return access, dmaps.bindings
