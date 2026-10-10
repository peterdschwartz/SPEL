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
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

from spel.scripts import edit_files
from spel.scripts.config import E3SM_SRCROOT
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
from spel.scripts.fortran_parser.symbols import Origin
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

# Trailing comment on lines inserted by `spel instrument`
MARKER = "!#SPEL"
# Calls that dump/read SPEL reference data (hand-written instrumentation)
CAPTURE_CALLS = {"write_elmtypes", "read_elmtypes", "write_constants", "spel_io_init"}
IO_OBJECTS = {"io_constants", "io_inputs", "io_outputs"}


@dataclass
class DriverCall:
    """
    ln: 0-based first line of the call statement in the driver file with
        any `spel instrument` (MARKER) lines removed
    """

    callee: str
    ln: int
    bounds: Optional[str]


@dataclass
class DriverSites:
    """
    Where elm_drv calls the unit-test roots; used by `spel instrument`.
    path:        driver file, relative to E3SM_SRCROOT (if under it)
    capture_lns: lines of existing SPEL capture calls in elm_drv
    """

    module: str
    routine: str
    path: str
    calls: dict[str, list[DriverCall]] = field(default_factory=dict)
    capture_lns: list[int] = field(default_factory=list)


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
) -> tuple[dict[str, AccessDict], list[ArgBinding], DriverSites]:
    """
    Global accesses of each root routine at its elm_drv call sites,
    elm_drv's argument bindings, and the call sites themselves. Only routines
    already mapped for the unit test are bound; other callees of elm_drv are
    treated as unknown.
    """
    drv_mod, lines = load_module(DRIVER_MODULE)
    drv = CallerRoutine(DRIVER_MODULE, DRIVER_ROUTINE, routine_lines(lines, DRIVER_ROUTINE))
    scopes = ModuleScopes({**mod_dict, DRIVER_MODULE: drv_mod}, sub_dict)
    mapped = {k: s for k, s in sub_dict.items() if s.record_access is not None}
    mapper = AccessMapper(mapped, scopes)
    dmaps = mapper.maps(drv)
    rec = drv.walk_syntax_tree(scopes)

    call_lines: dict[str, set[int]] = {}
    capture_lns: set[int] = set()
    for e in rec.events:
        if isinstance(e, CallEvent) and (
            e.name in CAPTURE_CALLS or (e.passed_object or "") in IO_OBJECTS
        ):
            capture_lns.add(e.ln)
        if isinstance(e, (CallEvent, FunctionRef)):
            callee = mapper.callee(drv, rec, e)
            if callee is not None:
                call_lines.setdefault(callee.id, set()).add(e.ln)
    access = {
        root.id: callsite_globals(dmaps, call_lines[root.id])
        for root in roots
        if root.id in call_lines
    }
    drv_path = Path(drv_mod.filepath).resolve()
    raw = drv_path.read_text(errors="replace").splitlines()
    markers = [i for i, line in enumerate(raw) if line.rstrip().endswith(MARKER)]

    def unmarked(ln: int) -> int:
        return ln - bisect_left(markers, ln)

    sites = DriverSites(
        module=DRIVER_MODULE,
        routine=DRIVER_ROUTINE,
        path=str(
            drv_path.relative_to(E3SM_SRCROOT)
            if drv_path.is_relative_to(E3SM_SRCROOT)
            else drv_path
        ),
        calls={
            root.id: [
                DriverCall(root.id, unmarked(ln), bounds_actual(root, dmaps.bindings, ln))
                for ln in sorted(call_lines[root.id])
            ]
            for root in roots
            if root.id in call_lines
        },
        capture_lns=sorted(unmarked(ln) for ln in capture_lns),
    )
    return access, dmaps.bindings, sites


def call_statement(sites: DriverSites, call: DriverCall) -> str:
    """The (unwrapped) `call` statement of `call` in the routine of `sites`."""
    fort_mod, lines = load_module(sites.module)
    raw = Path(fort_mod.filepath).resolve().read_text(errors="replace").splitlines()
    markers = [i for i, line in enumerate(raw) if line.rstrip().endswith(MARKER)]
    for lt in routine_lines(lines, sites.routine):
        if (
            lt.ln - bisect_left(markers, lt.ln) == call.ln
            and not raw[lt.ln].rstrip().endswith(MARKER)
            and re.match(r"^\s*call\b", lt.line, re.IGNORECASE)
        ):
            return lt.line.strip()
    raise SemanticError(f"no call statement for {call.callee} at line {call.ln + 1} of {sites.path}")


def bounds_actual(root: Subroutine, bindings: list[ArgBinding], ln: int) -> Optional[str]:
    """Actual passed for root's bounds_type dummy at the call on line `ln`."""
    for b in bindings:
        if b.callee != root.id or b.ln != ln:
            continue
        dummy = root.arguments.get(b.dummy)
        if dummy is not None and dummy.type == "bounds_type":
            return b.actual
    return None


# --------------------------------------------------------------------------
# Nested call sites: routines elm_drv reaches through other routines
# --------------------------------------------------------------------------
def callees_of(sub: Subroutine) -> set[str]:
    out = set(sub.child_subroutines)
    if sub.record_access is not None:
        out |= {b.callee for b in sub.record_access.bindings}
    return out


def call_path(
    sub_dict: dict[str, Subroutine], direct: Iterable[str], target: str
) -> Optional[list[str]]:
    """
    Shortest call chain [elm_drv callee, ..., caller, target] from a routine
    elm_drv calls directly, or None if `target` isn't reached.
    """
    from collections import deque

    starts = sorted(direct)
    prev: dict[str, Optional[str]] = {s: None for s in starts}
    queue = deque(starts)
    while queue:
        cur = queue.popleft()
        if cur == target:
            path = [cur]
            while (p := prev[path[-1]]) is not None:
                path.append(p)
            return path[::-1]
        sub = sub_dict.get(cur)
        if sub is None or sub.library:
            continue
        for nxt in sorted(callees_of(sub)):
            if nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    return None


def callers_of(sub_dict: dict[str, Subroutine], target: str) -> list[str]:
    return sorted(
        s.id for s in sub_dict.values() if not s.library and target in callees_of(s)
    )


def compose_bindings(
    sub_dict: dict[str, Subroutine], driver_bindings: list[ArgBinding], path: list[str]
) -> list[ArgBinding]:
    """
    The target's (path[-1]) dummies bound to what elm_drv passes: an actual
    that is a dummy of the caller is replaced by what the caller's caller
    passes for it, up to elm_drv (first call site at each level). Actuals
    local to an intermediate routine keep their origin (and caller).
    """
    def first_site(bindings: list[ArgBinding], callee: str) -> list[ArgBinding]:
        found = [b for b in bindings if b.callee == callee]
        ln = min((b.ln for b in found), default=None)
        return [b for b in found if b.ln == ln]

    top = first_site(driver_bindings, path[0])
    if not top:
        raise SemanticError(f"no {DRIVER_ROUTINE} call site found for {path[0]}")
    drv_ln = top[0].ln
    # dummy of the current routine -> [(actual at elm_drv, origin)]
    bound: dict[str, list[tuple[str, Origin, str]]] = {}
    for b in top:
        bound.setdefault(b.dummy, []).append((b.actual, b.origin, b.caller))
    for caller, callee in zip(path, path[1:]):
        maps = sub_dict[caller].record_access
        site = first_site(maps.bindings if maps is not None else [], callee)
        new: dict[str, list[tuple[str, Origin, str]]] = {}
        for b in site:
            base = b.actual.split("%")[0]
            if b.origin is Origin.DUMMY and base in bound:
                for actual, origin, at in bound[base]:
                    new.setdefault(b.dummy, []).append(
                        (actual + b.actual[len(base):], origin, at)
                    )
            else:
                new.setdefault(b.dummy, []).append((b.actual, b.origin, b.caller))
        bound = new
    last = sub_dict[path[-2]].record_access if len(path) > 1 else None
    argn = {b.dummy: b.argn for b in first_site(last.bindings if last else top, path[-1])}
    return [
        ArgBinding(at, path[-1], drv_ln, argn.get(dummy, 0), dummy, actual, origin)
        for dummy, items in bound.items()
        for actual, origin, at in items
    ]


def caller_sites(caller: Subroutine, targets: Iterable[Subroutine]) -> DriverSites:
    """
    DriverSites for the `call` statements to `targets` in `caller`, located
    in the (unedited) E3SM source; `bounds` is a bounds_type dummy or local
    of the caller (needed by the capture calls).
    """
    fort_mod, lines = load_module(caller.module)
    body = routine_lines(lines, caller.name)
    bounds = next(
        (n for n, v in caller.arguments.items() if v.type == "bounds_type"),
        next(
            (n for n, v in caller.local_variables.items() if v.type == "bounds_type"),
            None,
        ),
    )
    path = Path(fort_mod.filepath).resolve()
    raw = path.read_text(errors="replace").splitlines()
    markers = [i for i, line in enumerate(raw) if line.rstrip().endswith(MARKER)]
    calls: dict[str, list[DriverCall]] = {}
    for target in targets:
        regex = re.compile(rf"^\s*call\s+{target.name}\b", re.IGNORECASE)
        calls[target.id] = [
            DriverCall(target.id, lt.ln - bisect_left(markers, lt.ln), bounds)
            for lt in body
            if regex.match(lt.line) and not raw[lt.ln].rstrip().endswith(MARKER)
        ]
        if not calls[target.id]:
            raise SemanticError(f"no `call {target.name}` found in {caller.id}")
    return DriverSites(
        module=caller.module,
        routine=caller.name,
        path=str(path.relative_to(E3SM_SRCROOT) if path.is_relative_to(E3SM_SRCROOT) else path),
        calls=calls,
    )
