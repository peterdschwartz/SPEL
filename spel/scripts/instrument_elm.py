"""
Instrument an E3SM/ELM source tree so that a CIME case dumps the reference
data (spel-constants/inputs/outputs) needed by a SPEL functional unit test.

Everything is driven by the FunctionalUnitTest that `spel create` pickles
into the case directory (<case>/fut.pkl):

    1. Generate SpelCapture_<tag> (spel_capture_inputs/outputs_<tag>) for the
       active elm_instMod instances; it writes <tag>.spel-*.nc
    2. Insert one `use` line and two capture calls around the call site(s)
       of the unit-test routines found by driver_callsites. Inserted lines
       carry MARKER so re-instrumenting replaces them.
    3. Make the module variables/types the IO modules import public and not
       protected, editing only the declaration / access-statement lines the
       analysis recorded
    4. Copy the IO modules into components/elm/src/main; ReadWriteMod and
       FUTConstantsMod are renamed <Module>_<tag>

Each case gets its own tag, so `instrument_cases` can instrument many cases
for a single ELM build and run (`spel validate`).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import spel.scripts.io.helper as hio
from spel.scripts.driver_callsites import MARKER, DriverSites
from spel.scripts.module_resolver import ModuleHead, entity_names, module_head

if TYPE_CHECKING:
    from spel.scripts.functional_unit_test import FunctionalUnitTest
    from spel.scripts.subset import SubsetSpec

CAPTURE_PREFIX = "SpelCapture_"
MAX_TAG = 40  # Fortran names are <= 63 characters
# Kept out of the case dir itself: the offline CMakeLists globs *.F90 and
# the capture module needs ELM-only modules (elm_time_manager).
CAPTURE_DIR = "elm-capture"
# Case-independent modules; taken from SourceFiles when available since a
# case dir may hold a copy predating the current generator.
SHARED_IO_MODULES = ("nc_io.F90", "nc_allocMod.F90")
# Per-case modules; renamed <Module>_<tag> in ELM.
CASE_IO_MODULES = ("ReadWriteMod.F90", "FUTConstantsMod.F90")
MANIFEST = "spel_instrument.json"
# Original text of every line the accessibility edits changed (relative to
# srcroot, 0-based line of the uninstrumented file), so `--undo` can revert
# them. Lives next to the copied IO modules in components/elm/src/main.
ACCESS_LEDGER = ".spel-access-edits.json"
DEFAULT_FREQ = 9
DEFAULT_MAX_TPF = 720

regex_module = re.compile(r"^\s*module\s+(?!procedure\b)(\w+)\s*(?:!.*)?$", re.I)
regex_type_def = re.compile(r"^\s*type\b\s*(?:,[^:]*)?(?:::)?\s*(\w+)\s*$", re.I)


class InstrumentError(Exception):
    pass


# --------------------------------------------------------------------------
# Fortran source helpers (free form)
# --------------------------------------------------------------------------
def strip_comment(line: str) -> str:
    """Remove a trailing `!` comment, ignoring `!` inside string literals."""
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "!":
            return line[:i]
    return line


def statement_end(lines: list[str], start: int) -> int:
    """Last physical line of the statement starting at `start` (`&` continuation)."""
    i = start
    while i + 1 < len(lines) and strip_comment(lines[i]).rstrip().endswith("&"):
        i += 1
    return i


def statement_text(lines: list[str], start: int) -> str:
    """Joined, comment-free, lowercased text of the statement at `start`."""
    parts = []
    for line in lines[start : statement_end(lines, start) + 1]:
        code = strip_comment(line).strip()
        parts.append(code.removeprefix("&").removesuffix("&").strip())
    return " ".join(p for p in parts if p).lower()


def strip_instrumentation(lines: list[str]) -> list[str]:
    return [ln for ln in lines if not ln.rstrip().endswith(MARKER)]


def elm_source_files(srcroot: Path) -> list[Path]:
    return sorted((srcroot / "components/elm/src").rglob("*.F90"))


# --------------------------------------------------------------------------
# Call site (from driver_callsites)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptureNames:
    """
    ELM-side names for one case. Every case gets its own capture module and
    copies of its IO modules, so one ELM build can capture many cases.
    """

    tag: str
    # elm_drv scalar dummies the unit test needs: ((name, type), ...); see
    # FunctionalUnitTest.driver_args
    drv_args: tuple[tuple[str, str], ...] = ()
    # time-step scalars captured every step (spel.scripts.io.time_scalars)
    time: tuple = ()

    @classmethod
    def for_case(
        cls, case_name: str, drv_args: tuple[tuple[str, str], ...] = (), time: tuple = ()
    ) -> "CaptureNames":
        tag = re.sub(r"\W+", "_", case_name.lower()).strip("_")
        if tag and tag[0].isdigit():
            tag = f"c{tag}"
        if not tag or len(tag) > MAX_TAG:
            raise InstrumentError(
                f"case name {case_name!r} can't be used in Fortran names "
                f"(1-{MAX_TAG} letters, digits or _)"
            )
        return cls(tag, tuple(drv_args), tuple(time))

    @property
    def module(self) -> str:
        return f"{CAPTURE_PREFIX}{self.tag}"

    @property
    def inputs(self) -> str:
        return f"spel_capture_inputs_{self.tag}"

    @property
    def outputs(self) -> str:
        return f"spel_capture_outputs_{self.tag}"

    @property
    def file_prefix(self) -> str:
        """ELM's capture files are <tag>.spel-*.nc in the run directory."""
        return f"{self.tag}."

    def io_module(self, name: str) -> str:
        """ELM-side name of a per-case IO module (ReadWriteMod, FUTConstantsMod)."""
        return f"{name}_{self.tag}"


def _check_call_sites(lines: list[str], sites: DriverSites) -> tuple[int, int, str]:
    """(first call line, last line of the last call statement, bounds actual)."""
    if sites.capture_lns:
        raise InstrumentError(
            f"{sites.path} has hand-written capture calls at lines "
            f"{[ln + 1 for ln in sites.capture_lns]}; remove them and re-run `spel create`"
        )
    calls = [c for found in sites.calls.values() for c in found]
    if not calls:
        raise InstrumentError(f"No unit-test routine is called from {sites.routine}")
    for call in calls:
        name = call.callee.split("::")[-1]
        line = lines[call.ln] if call.ln < len(lines) else ""
        # a type-bound call (`call col_ns%Summary`) names its binding, not the routine
        if not re.match(rf"^\s*call\s+({name}\b|[\w%]+%\w+\s*\()", strip_comment(line), re.I):
            raise InstrumentError(
                f"{sites.path}:{call.ln + 1} is no longer `call {name}`; "
                "it changed since `spel create` -- re-run create"
            )
    for callee, found in sites.calls.items():
        if len(found) > 1:
            print(f"Warning: {callee} is called {len(found)} times in {sites.routine}; capturing around all")
    # the captures need a bounds_type in scope: the first call that passes one (routines
    # such as VegStructUpdate take only filters)
    with_bounds = sorted((c for c in calls if c.bounds is not None), key=lambda c: c.ln)
    if not with_bounds:
        raise InstrumentError(f"No bounds_type actual argument at the call(s) to {[c.callee for c in calls]}")
    first = min(calls, key=lambda c: c.ln)
    return first.ln, max(statement_end(lines, c.ln) for c in calls), with_bounds[0].bounds


def _first_calls_end(lines: list[str], sites: DriverSites) -> int:
    """Last line of the first call to each routine (main.F90 makes those calls)."""
    return max(statement_end(lines, min(c.ln for c in found)) for found in sites.calls.values())


_BLOCK_OPEN = re.compile(
    r"^(\w+\s*:\s*)?(if\s*\(.*\)\s*then|do\b|select\s*(case|type)\b|associate\s*\(|block$|where\s*\(.*\)$)"
)
_BLOCK_CLOSE = re.compile(r"^end\s*(if|do|select|associate|block|where)\b")
_BLOCK_MID = re.compile(r"^(else\b|elseif\b|case\b|elsewhere\b)")


def _same_block(lines: list[str], start: int, end: int) -> bool:
    """Whether lines start..end stay in one block: no else/case of, or exit from,
    the construct around `start`, and no construct left open."""
    depth = 0
    ln = start
    while ln <= end:
        text = statement_text(lines, ln)
        if _BLOCK_CLOSE.match(text):
            depth -= 1
            if depth < 0:
                return False
        elif _BLOCK_MID.match(text):
            if depth == 0:
                return False
        elif _BLOCK_OPEN.match(text):
            depth += 1
        ln = statement_end(lines, ln) + 1
    return depth == 0


def _construct_spans(lines: list[str]) -> tuple[dict[int, tuple[int, ...]], dict[int, int], dict[int, list[int]]]:
    """
    For every statement line, the opening lines of the constructs around it (outermost
    first); each opener's closing line; each opener's else/case lines.
    """
    enclosing: dict[int, tuple[int, ...]] = {}
    closer: dict[int, int] = {}
    mids: dict[int, list[int]] = {}
    stack: list[int] = []
    ln = 0
    while ln < len(lines):
        text = statement_text(lines, ln)
        if _BLOCK_CLOSE.match(text) and stack:
            closer[stack.pop()] = ln
            enclosing[ln] = tuple(stack)
        elif _BLOCK_MID.match(text) and stack:
            mids.setdefault(stack[-1], []).append(ln)
            enclosing[ln] = tuple(stack)
        else:
            enclosing[ln] = tuple(stack)
            if _BLOCK_OPEN.match(text):
                stack.append(ln)
        ln = statement_end(lines, ln) + 1
    return enclosing, closer, mids


def _widened_span(lines: list[str], start: int, end: int) -> Optional[tuple[int, int]]:
    """
    The smallest span around start..end made of whole statements and constructs of the
    innermost construct that contains both (EcosystemDynLeaching inside `if (use_cn)`, then
    balance checks at the clump-loop level, then SurfaceAlbedo inside `if (doalb)`): its
    first line and the line closing it. None when it would cross an else/case line of that
    construct, i.e. when the calls are in different branches.
    """
    enclosing, closer, mids = _construct_spans(lines)
    # `end` may be a continuation line: the statement it belongs to
    end = max((ln for ln in enclosing if ln <= end), default=-1)
    if start not in enclosing or end < start:
        return None
    sa, ea = enclosing[start], enclosing[end]
    k = 0
    while k < min(len(sa), len(ea)) and sa[k] == ea[k]:
        k += 1
    new_start = sa[k] if k < len(sa) else start
    new_end = closer.get(ea[k], end) if k < len(ea) else end
    if k < len(ea) and ea[k] not in closer:
        return None
    common = sa[k - 1] if k > 0 else None
    if common is not None and any(new_start < m <= new_end for m in mids.get(common, [])):
        return None
    return new_start, statement_end(lines, new_end)


def instrument_lines(
    lines: list[str], groups: list[tuple[DriverSites, CaptureNames]]
) -> tuple[list[str], dict[str, str]]:
    """
    For each case (sites, names) in this file: insert its capture calls around
    its recorded call sites (inputs before the first, outputs after the last)
    and a `use` of its capture module after the module statement.
    Idempotent: previously inserted lines are removed first, and recorded
    line numbers refer to the file without them.
    Returns (new_lines, {tag: bounds_actual}).
    """
    lines = strip_instrumentation(lines)
    before: dict[int, list[str]] = {}
    after: dict[int, list[str]] = {}
    uses: dict[str, list[str]] = {}
    bounds: dict[str, str] = {}
    for sites, names in groups:
        first, end, bnd = _check_call_sites(lines, sites)
        bounds[names.tag] = bnd
        if _same_block(lines, first, end):
            # a routine called again later in the block (AerosolMasses for snow,
            # then lake columns) is captured around its first call only: the
            # code in between isn't part of the unit test
            spans = [(first, _first_calls_end(lines, sites))]
        elif all(len(found) == 1 for found in sites.calls.values()) and (
            widened := _widened_span(lines, first, end)
        ) is not None:
            # several routines called once each, in sibling constructs of one block: one
            # capture around the whole stretch
            spans = [widened]
        else:
            # e.g. HydrologyDrainage in both branches of an if: capture around
            # each call, or the inputs capture sits in a branch that never runs
            calls = sorted(c.ln for found in sites.calls.values() for c in found)
            spans = [(ln, statement_end(lines, ln)) for ln in calls]
        for span_first, span_end in spans:
            indent = re.match(r"\s*", lines[span_first]).group()
            actuals = "".join(f", {name}" for name, _ in names.drv_args)
            before.setdefault(span_first, []).append(
                f"{indent}call {names.inputs}({bnd}{actuals}) {MARKER}\n"
            )
            after.setdefault(span_end, []).append(
                f"{indent}call {names.outputs}({bnd}) {MARKER}\n"
            )
        uses.setdefault(sites.module, []).append(
            f"  use {names.module}, only : {names.inputs}, {names.outputs} {MARKER}\n"
        )

    out: list[str] = []
    for i, line in enumerate(lines):
        out.extend(before.get(i, ()))
        out.append(line)
        out.extend(after.get(i, ()))
        if (m := regex_module.match(line)) and m.group(1).lower() in uses:
            out.extend(uses.pop(m.group(1).lower()))
    if uses:
        path = groups[0][0].path
        raise InstrumentError(f"`module {', '.join(uses)}` not found in {path}")
    return out, bounds


# --------------------------------------------------------------------------
# Accessibility fixes
# --------------------------------------------------------------------------
# module -> {name: 0-based declaration line, or None for derived types}
IOSymbols = dict[str, dict[str, Optional[int]]]


def io_symbols(fut: FunctionalUnitTest) -> IOSymbols:
    """
    ELM module entities imported by the generated IO modules: the active
    instances (or, for elm_instMod instances, their types) used by
    ReadWriteMod and the global variables used by FUTConstantsMod.
    """
    active, _, elminst = hio.get_var_usage_and_elm_inst_vars(fut.type_dict)
    symbols: IOSymbols = {}
    for var in [*active.values(), *fut.non_parameter_global_vars.values()]:
        if var.name == "bounds" or var.declaration == "elm_instmod":
            continue
        symbols.setdefault(var.declaration, {})[var.name] = var.ln
    for type_mod, var in elminst:
        symbols.setdefault(type_mod, {})[var.type] = None
    return symbols


def capture_instances(fut: FunctionalUnitTest) -> list[str]:
    """elm_instMod instances passed to write_elmtypes (same order as ReadWriteMod)."""
    _, _, elminst = hio.get_var_usage_and_elm_inst_vars(fut.type_dict)
    return [var.name for _, var in elminst]


def _remove_from_access_stmt(lines: list[str], start: int, name: str) -> bool:
    """Drop `name` from the private/protected statement at `start`."""
    end = statement_end(lines, start)
    word = re.compile(rf"\b{name}\b", re.I)
    for i in range(start, end + 1):
        line = lines[i]
        code = strip_comment(line)
        if not word.search(code.split("::", 1)[-1] if i == start else code):
            continue
        new = re.sub(rf"\b{name}\b\s*,\s*", "", code, count=1, flags=re.I)
        if new == code:
            new = re.sub(rf",\s*\b{name}\b", "", code, count=1, flags=re.I)
        if new == code:  # only entity left
            if start != end:
                raise InstrumentError(f"Can't remove {name} from statement at line {start + 1}")
            indent = re.match(r"\s*", line).group()
            lines[i] = f"{indent}! {line.strip()}\n"
        else:
            lines[i] = new + line[len(code) :]
        return True
    return False


def access_ledger_path(srcroot: Path) -> Path:
    return Path(srcroot) / "components/elm/src/main" / ACCESS_LEDGER


def load_access_ledger(srcroot: Path) -> dict[str, list[dict]]:
    path = access_ledger_path(srcroot)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except ValueError as err:
        raise InstrumentError(f"Corrupt access ledger {path}: {err}")


def write_access_ledger(srcroot: Path, ledger: dict[str, list[dict]]) -> None:
    path = access_ledger_path(srcroot)
    if ledger:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    elif path.exists():
        path.unlink()


def record_access_edits(
    ledger: dict[str, list[dict]], rel: str, before: list[str], after: list[str]
) -> None:
    """Add the lines that differ between `before` and `after` to `ledger[rel]`.
    A line already recorded keeps its first original text."""
    if len(before) != len(after):
        raise InstrumentError(f"Accessibility edit changed the line count of {rel}")
    entries = {e["ln"]: e for e in ledger.get(rel, [])}
    for ln, (old, new) in enumerate(zip(before, after)):
        if old == new:
            continue
        if ln in entries:
            entries[ln]["edited"] = new
        else:
            entries[ln] = {"ln": ln, "original": old, "edited": new}
    if entries:
        ledger[rel] = [entries[ln] for ln in sorted(entries)]


def revert_access_edits(srcroot: Path) -> tuple[list[Path], dict[str, list[dict]]]:
    """
    Restore the lines recorded in the access ledger. A recorded line is put
    back only if the file still holds the edited text (at its line, or at a
    unique line if the file shifted); anything else is left in place and
    returned as unresolved (it stays in the ledger).
    """
    srcroot = Path(srcroot)
    ledger = load_access_ledger(srcroot)
    restored: list[Path] = []
    unresolved: dict[str, list[dict]] = {}
    for rel, entries in sorted(ledger.items()):
        path = srcroot / rel
        if not path.is_file():
            unresolved[rel] = entries
            continue
        lines = path.read_text().splitlines(keepends=True)
        code_lns = [i for i, ln in enumerate(lines) if not ln.rstrip().endswith(MARKER)]
        code = [lines[i] for i in code_lns]
        changed = False
        for e in entries:
            ln = e["ln"]
            if not (ln < len(code) and code[ln] == e["edited"]):
                hits = [i for i, t in enumerate(code) if t == e["edited"]]
                if len(hits) != 1:
                    unresolved.setdefault(rel, []).append(e)
                    continue
                ln = hits[0]
            code[ln] = e["original"]
            lines[code_lns[ln]] = e["original"]
            changed = True
        if changed:
            path.write_text("".join(lines))
            restored.append(path)
    write_access_ledger(srcroot, unresolved)
    return restored, unresolved


def make_accessible(
    lines: list[str], head: ModuleHead, symbols: dict[str, Optional[int]], path: str
) -> tuple[list[str], list[str]]:
    """
    Make `symbols` public and not protected. `head` and the declaration lines
    come from `spel create`; each edited line is checked against the current
    text so a changed file is an error and re-running is a no-op.
    """
    lines = list(lines)
    changed: set[str] = set()
    for name, decl_ln in sorted(symbols.items()):
        if head.is_public(name) and not head.is_protected(name):
            continue
        for ln in head.stmt_ln.get(name, []):
            kind = statement_text(lines, ln).split(None, 1)[0] if lines[ln].strip() else ""
            if kind.startswith(("private", "protected")) and _remove_from_access_stmt(lines, ln, name):
                changed.add(name)
        if decl_ln is None:
            if not head.is_public(name) and not head.stmt_ln.get(name):
                raise InstrumentError(f"Can't make type {name} public in {path}")
            continue

        text = statement_text(lines, decl_ln) if decl_ln < len(lines) else ""
        spec, sep, ents = text.partition("::")
        if not sep or name not in entity_names(ents):
            raise InstrumentError(
                f"{path}:{decl_ln + 1} no longer declares {name}; re-run `spel create`"
            )
        code = strip_comment(lines[decl_ln])
        if "::" not in code:
            raise InstrumentError(f"Can't edit declaration split before '::' at {path}:{decl_ln + 1}")
        attrs, rest = code.split("::", 1)
        new = re.sub(r",\s*protected\b", "", attrs, flags=re.I)
        new = re.sub(r",\s*private\b", ", public", new, flags=re.I)
        if head.default_private and not head.access.get(name) and not re.search(r",\s*public\b", new, re.I):
            new = new.rstrip() + ", public "
        if new != attrs:
            lines[decl_ln] = new + "::" + rest + lines[decl_ln][len(code) :]
            changed.add(name)
    return lines, sorted(changed)


# {module: {type name: {component: 0-based declaration line}}}
ComponentSymbols = dict[str, dict[str, dict[str, int]]]


def component_symbols(fut: FunctionalUnitTest) -> ComponentSymbols:
    """Active components of active derived types (read/written by ReadWriteMod)."""
    symbols: ComponentSymbols = {}
    for name, dtype in getattr(fut, "type_dict", {}).items():
        if not dtype.active or not dtype.declaration:
            continue
        comps = {}
        for c, v in dtype.components.items():
            if not v.active or v.ln is None:
                continue
            if getattr(v, "nested", False):
                # core%leaf is declared in core's type: make `core` accessible here
                holder = dtype.components.get(c.split("%")[0])
                if holder is not None and holder.ln is not None:
                    comps[holder.name] = holder.ln
                continue
            comps[c] = v.ln
        if comps:
            symbols.setdefault(dtype.declaration, {})[name] = comps
    return symbols


def make_components_public(
    lines: list[str], type_name: str, comps: dict[str, int], path: str
) -> tuple[list[str], list[str]]:
    """
    Make the components `comps` of `type_name` accessible: comment out the
    type's default `private` statement and turn `, private` into `, public`
    on their declarations. Line counts are kept; re-running is a no-op.
    """
    lines = list(lines)
    changed: list[str] = []
    first = min(comps.values())
    start = next(
        (
            i for i in range(min(first, len(lines) - 1), -1, -1)
            if (m := regex_type_def.match(strip_comment(lines[i])))
            and m.group(1).lower() == type_name.lower()
        ),
        None,
    )
    if start is None:
        raise InstrumentError(f"{path}: no `type {type_name}` above line {first + 1}; re-run `spel create`")
    for i in range(start + 1, len(lines)):
        code = strip_comment(lines[i]).strip().lower()
        if code == "contains" or re.match(r"end\s*type\b", code):
            break
        if code == "private":
            indent = re.match(r"\s*", lines[i]).group()
            lines[i] = f"{indent}! {lines[i].strip()}\n"
            changed.append(f"{type_name} (default private)")
    for comp, ln in sorted(comps.items()):
        text = statement_text(lines, ln) if ln < len(lines) else ""
        spec, sep, ents = text.partition("::")
        if not sep or comp not in entity_names(ents):
            raise InstrumentError(f"{path}:{ln + 1} no longer declares {type_name}%{comp}; re-run `spel create`")
        code = strip_comment(lines[ln])
        attrs, rest = code.split("::", 1)
        new = re.sub(r",\s*private\b", ", public", attrs, flags=re.I)
        if new != attrs:
            lines[ln] = new + "::" + rest + lines[ln][len(code):]
            changed.append(f"{type_name}%{comp}")
    return lines, changed


def components_public(
    lines: list[str], types: dict[str, dict[str, int]], path: str
) -> tuple[list[str], list[str]]:
    changed: list[str] = []
    for type_name, comps in sorted(types.items()):
        lines, done = make_components_public(lines, type_name, comps, path)
        changed.extend(done)
    return lines, changed


def make_case_sources_accessible(fut: FunctionalUnitTest, case_dir: Path) -> dict[str, list[str]]:
    """
    Make the entities the generated IO modules import public in the unit-test
    copies of their modules (the same edit `spel instrument` makes in ELM).
    Must run before `insert_header_for_unittest` shifts the copies' lines.
    """
    made: dict[str, list[str]] = {}
    symbols, comps = io_symbols(fut), component_symbols(fut)
    for mod in sorted({*symbols, *comps}):
        fort_mod = fut.module_dict.get(mod)
        if fort_mod is None:
            continue
        path = Path(case_dir) / Path(fort_mod.filepath).name
        if not path.exists():
            continue
        src, changed = path.read_text().splitlines(keepends=True), []
        if mod in symbols:
            src, changed = make_accessible(src, module_head(fort_mod), symbols[mod], str(path))
        if mod in comps:
            src, more = components_public(src, comps[mod], str(path))
            changed = [*changed, *more]
        if changed:
            path.write_text("".join(src))
            made[mod] = changed
    return made


# --------------------------------------------------------------------------
# Capture module
# --------------------------------------------------------------------------
def generate_capture_module(
    names: CaptureNames,
    instances: list[str],
    freq: int = DEFAULT_FREQ,
    max_tpf: int = DEFAULT_MAX_TPF,
    subset: Optional[SubsetSpec] = None,
    compact: Sequence[tuple[str, str, str]] = (),
) -> str:
    """
    Wrappers so the ELM call site only needs one call before/after. Each case
    owns its spel_io_type handles and writes <tag>.spel-*.nc, so several
    cases can be captured by one run.

    With a `subset`, the first capture selects the gridcells (see
    spel.scripts.subset) and every write runs with that subset active in
    nc_io and with the subset's bounds; ranks without the subset write nothing.

    `compact` lists the (instance, subgrid, declaring module) of the types a
    `spel compact rewrite` gave compact storage (see compact_capture_instances):
    at a capture the components point at the compact storage of the clump's
    filter, so every capture first unpacks the clump into the full layout and
    switches the instance to it, and switches back after. The files then hold
    what a build without the compact layout holds, and unit tests of either
    layout can read them. Without compact instances the module is unchanged.
    """
    parts = _capture_parts(names, instances)
    compact_use, compact_decl, compact_pre, compact_post, compact_procs = _compact_capture_lines(compact)
    if subset is not None:
        return _subset_capture_module(
            names, parts, freq, max_tpf, subset, compact_use, compact_pre, compact_post, compact_decl, compact_procs
        )
    inst_use, inst_args, rw, fc, pre, drv_dummies, drv_use, drv_decl, drv_copy = parts
    inst_use += compact_use
    return f"""module {names.module}
  !!! Auto-generated by SPEL: dump reference data for functional unit tests.
  !!! Inputs/outputs are captured every capture_freq model steps.
  use shr_kind_mod, only : r8 => shr_kind_r8
  use decompMod, only : bounds_type, get_proc_bounds
  use elm_time_manager, only : get_nstep
  use nc_io, only : spel_io_type
  use {rw}, only : write_elmtypes{drv_use}
  use {fc}, only : write_constants
{inst_use}  implicit none
  private
  public :: {names.inputs}, {names.outputs}
  integer, parameter :: capture_freq = {freq}
  integer, parameter :: max_tpf = {max_tpf}
  type(spel_io_type) :: io_constants, io_inputs, io_outputs
  logical :: capturing = .false.
{compact_decl}contains
  subroutine {names.inputs}(bounds{drv_dummies})
    type(bounds_type), intent(in) :: bounds
{drv_decl}{drv_copy}    if (.not. io_constants%created) then
      call io_constants%init(base_fn='{pre}spel-constants', max_tpf=max_tpf, read_io=.false.)
      call io_inputs%init(base_fn='{pre}spel-inputs', max_tpf=max_tpf, read_io=.false.)
      call io_outputs%init(base_fn='{pre}spel-outputs', max_tpf=max_tpf, read_io=.false.)
      call write_constants(io_constants)
    end if
    capturing = mod(get_nstep(), capture_freq) == 0
    if (capturing) then
{compact_pre}      call write_elmtypes(io_inputs, bounds{inst_args})
{compact_post}    end if
  end subroutine {names.inputs}

  subroutine {names.outputs}(bounds)
    type(bounds_type), intent(in) :: bounds
    if (capturing) then
{compact_pre}      call write_elmtypes(io_outputs, bounds{inst_args})
{compact_post}    end if
  end subroutine {names.outputs}
{compact_procs}end module {names.module}
"""


def _compact_capture_lines(compact: Sequence) -> tuple[str, str, str, str, str]:
    """
    (use lines, declarations, lines before/after a capture, contained
    procedures) for the compact fields (CompactField records of
    spel.transforms.compact.unit_test): the capture gathers each field's full
    layout through the inverse map and rebinds afterwards.
    """
    if not compact:
        return "", "", "", "", ""
    from spel.transforms.compact.unit_test import capture_pieces

    return capture_pieces(list(compact))


def compact_capture_instances(fut: FunctionalUnitTest) -> list:
    """
    The case's compact fields (CompactField records): the fields of the
    sources' `.spel-compact.json` groups that belong to an active instance and
    a component the case uses.
    """
    from spel.scripts.config import E3SM_SRCROOT
    from spel.transforms.compact.unit_test import compact_fields, load_manifest

    type_dict = getattr(fut, "type_dict", None)
    if not type_dict:
        return []
    manifest = load_manifest(getattr(fut, "e3sm_srcroot", E3SM_SRCROOT))
    if not manifest:
        return []
    active_instances, _, _ = hio.get_var_usage_and_elm_inst_vars(type_dict)
    return compact_fields(type_dict, active_instances, manifest)


def _capture_parts(names: CaptureNames, instances: list[str]) -> tuple[str, ...]:
    """Pieces of the capture module that depend on the case's instances and elm_drv dummies."""
    cont = " &\n         "
    inst_use = (
        f"  use elm_instMod, only :{cont}{(','+cont).join(instances)}\n" if instances else ""
    )
    inst_args = "".join(f",{cont}{n}={n}" for n in instances)
    rw, fc = names.io_module("ReadWriteMod"), names.io_module("FUTConstantsMod")
    pre = names.file_prefix
    ftype = {"real": "real(r8)", "integer": "integer", "logical": "logical"}
    drv_dummies = "".join(f", {n}" for n, _ in names.drv_args)
    drv_use = "".join(f", spel_drv_{n}" for n, _ in names.drv_args)
    drv_decl = "".join(f"    {ftype[t]}, intent(in) :: {n}\n" for n, t in names.drv_args)
    # logicals are stored as integers (0/1), like the time scalars
    drv_copy = "".join(
        f"    spel_drv_{n}(1) = merge(1, 0, {n})\n" if t == "logical" else f"    spel_drv_{n}(1) = {n}\n"
        for n, t in names.drv_args
    )
    if names.time:
        from spel.scripts.io.time_scalars import capture_lines

        t_uses, t_sets = capture_lines(list(names.time))
        inst_use += t_uses
        drv_use += "".join(f", {ts.array}" for ts in names.time)
        drv_copy += t_sets
    return inst_use, inst_args, rw, fc, pre, drv_dummies, drv_use, drv_decl, drv_copy


def _subset_capture_module(
    names: CaptureNames,
    parts: tuple[str, ...],
    freq: int,
    max_tpf: int,
    subset: SubsetSpec,
    compact_use: str = "",
    compact_pre: str = "",
    compact_post: str = "",
    compact_decl: str = "",
    compact_procs: str = "",
) -> str:
    from spel.scripts.subset import capture_select_routine

    inst_use, inst_args, rw, fc, pre, drv_dummies, drv_use, drv_decl, drv_copy = parts
    mask_uses = "".join(f"  use {u}\n" for u in subset.uses())
    return f"""module {names.module}
  !!! Auto-generated by SPEL: dump reference data for functional unit tests.
  !!! Inputs/outputs are captured every capture_freq model steps, for a
  !!! gridcell subset (spel_select_subset)
  use shr_kind_mod, only : r8 => shr_kind_r8
  use decompMod, only : bounds_type, get_proc_bounds, get_proc_clumps, get_clump_bounds
  use elm_time_manager, only : get_nstep
  use spmdMod, only : masterproc, iam, npes, mpicom, MPI_INTEGER, MPI_MAX, MPI_MIN
  use nc_io, only : spel_io_type, spel_subset_type, nc_subset_set_level, nc_subset_count, &
       nc_subset_activate, nc_subset_deactivate
  use {rw}, only : write_elmtypes{drv_use}
  use {fc}, only : write_constants
{inst_use}{compact_use}{mask_uses}  implicit none
  private
  public :: {names.inputs}, {names.outputs}
  integer, parameter :: capture_freq = {freq}
  integer, parameter :: max_tpf = {max_tpf}
  integer, parameter :: spel_subset_max = {subset.max_cells}
  type(spel_io_type) :: io_constants, io_inputs, io_outputs
  type(spel_subset_type), target :: spel_sub
  type(bounds_type) :: spel_bounds_sub
  integer :: spel_clump_begg = -1
  logical :: spel_writer = .false.
  logical :: capturing = .false.
{compact_decl}contains
  subroutine {names.inputs}(bounds{drv_dummies})
    type(bounds_type), intent(in) :: bounds
{drv_decl}{drv_copy}    if (.not. io_constants%created) then
      call spel_select_subset(bounds)
      call io_constants%init(base_fn='{pre}spel-constants', max_tpf=max_tpf, read_io=.false.)
      call io_inputs%init(base_fn='{pre}spel-inputs', max_tpf=max_tpf, read_io=.false.)
      call io_outputs%init(base_fn='{pre}spel-outputs', max_tpf=max_tpf, read_io=.false.)
      if (spel_writer) then
        call nc_subset_activate(spel_sub)
        call write_constants(io_constants)
        call nc_subset_deactivate()
      end if
    end if
    capturing = spel_writer .and. bounds%begg == spel_clump_begg .and. mod(get_nstep(), capture_freq) == 0
    if (capturing) then
{compact_pre}      call nc_subset_activate(spel_sub)
      call write_elmtypes(io_inputs, spel_bounds_sub{inst_args})
      call nc_subset_deactivate()
{compact_post}    end if
  end subroutine {names.inputs}

  subroutine {names.outputs}(bounds)
    type(bounds_type), intent(in) :: bounds
    if (capturing) then
{compact_pre}      call nc_subset_activate(spel_sub)
      call write_elmtypes(io_outputs, spel_bounds_sub{inst_args})
      call nc_subset_deactivate()
{compact_post}    end if
  end subroutine {names.outputs}
{capture_select_routine(subset, names.tag)}
{compact_procs}end module {names.module}
"""


def rename_io_modules(text: str, names: CaptureNames) -> str:
    """ELM copy of a per-case IO module: module and use names get the case tag."""
    for mod in CASE_IO_MODULES:
        stem = mod.removesuffix(".F90")
        text = re.sub(rf"\b{stem}\b", names.io_module(stem), text, flags=re.I)
    return text


def remove_io_modules(dest: Path) -> list[Path]:
    """Delete IO/capture modules a previous instrumentation copied to `dest`."""
    patterns = [
        *SHARED_IO_MODULES,
        *(f"{m.removesuffix('.F90')}_*.F90" for m in CASE_IO_MODULES),
        f"{CAPTURE_PREFIX}*.F90",
        *CASE_IO_MODULES,
        "SpelCaptureMod.F90",  # pre-tagging layout
    ]
    removed = sorted({p for pat in patterns for p in Path(dest).glob(pat)})
    for p in removed:
        p.unlink()
    return removed


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
@dataclass
class InstrumentReport:
    call_site: Path
    module: str
    bounds: str
    tag: str = ""
    case_dir: Optional[Path] = None
    copied: list[Path] = field(default_factory=list)
    made_public: dict[Path, list[str]] = field(default_factory=dict)
    subset: Optional[SubsetSpec] = None

    def to_json(self) -> dict:
        return {
            "call_site": str(self.call_site),
            "module": self.module,
            "bounds": self.bounds,
            "tag": self.tag,
            "copied": [str(p) for p in self.copied],
            "made_public": {str(k): v for k, v in self.made_public.items()},
            "subset": self.subset.to_json() if self.subset else None,
        }


@dataclass
class MultiInstrumentReport:
    reports: dict[str, InstrumentReport] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    made_public: dict[Path, list[str]] = field(default_factory=dict)
    removed: list[Path] = field(default_factory=list)


def load_case(case: str) -> FunctionalUnitTest:
    """The FunctionalUnitTest `spel create` stored for `case` (name or path)."""
    from spel.scripts.config import unittests_dir
    from spel.scripts.export_objects import CASE_PICKLE, unpickle_case

    case_dir = Path(case) if Path(case).is_dir() else Path(unittests_dir) / case
    if not (case_dir / CASE_PICKLE).exists():
        raise InstrumentError(f"{case_dir / CASE_PICKLE} not found; run `spel create` first")
    fut = unpickle_case(case_dir)
    if getattr(fut, "driver_sites", None) is None:
        raise InstrumentError(
            f"{case_dir / CASE_PICKLE} has no elm_drv call sites; re-run `spel create`"
        )
    fut.case_dir = str(case_dir.resolve())
    return fut


def _io_source(case_dir: Path, f: str, mods_dir: Optional[Path]) -> Path:
    if mods_dir is not None and f in SHARED_IO_MODULES and (Path(mods_dir) / f).exists():
        return Path(mods_dir) / f
    return case_dir / f


def _check_case(fut: FunctionalUnitTest, mods_dir: Optional[Path]) -> dict:
    case_dir, sites = Path(fut.case_dir), fut.driver_sites
    needed = [*SHARED_IO_MODULES, *CASE_IO_MODULES]
    if missing := [f for f in needed if not _io_source(case_dir, f, mods_dir).exists()]:
        raise InstrumentError(f"{case_dir} is missing {missing}; run `spel create` first")
    if not_called := [s for s in fut.primary_subroutines if s not in sites.calls]:
        raise InstrumentError(
            f"{not_called} not called from {sites.routine}; "
            f"only call sites in one routine per case are supported"
        )
    symbols = io_symbols(fut)
    if sites.module in symbols:
        raise InstrumentError(f"{sites.module} is used by the IO modules (circular dependency)")
    return symbols


def instrument_cases(
    futs: list[FunctionalUnitTest],
    srcroot: Path,
    freq: int = DEFAULT_FREQ,
    dry_run: bool = False,
    mods_dir: Optional[Path] = None,
    subsets: Optional[dict[str, SubsetSpec]] = None,
) -> MultiInstrumentReport:
    """
    Instrument ELM so one run captures reference data for every case in
    `futs`. A case that can't be instrumented is recorded in `failures` and
    skipped; the others are still instrumented. `subsets` maps case names to
    the gridcell subset their capture writes (default: every gridcell).
    """
    subsets = subsets or {}
    srcroot = Path(srcroot)
    dest = srcroot / "components/elm/src/main"
    result = MultiInstrumentReport()
    accepted: list[tuple[FunctionalUnitTest, CaptureNames]] = []
    symbols: dict[str, dict[str, Optional[int]]] = {}
    fort_mods: dict = {}
    tags: dict[str, str] = {}
    texts: dict[Path, list[str]] = {}
    case_symbols_of: dict[str, dict] = {}
    comp_syms: ComponentSymbols = {}

    for fut in futs:
        name = fut.case_name
        try:
            from spel.scripts.io.time_scalars import unit_test_time_scalars

            names = CaptureNames.for_case(
                name,
                tuple(getattr(fut, "driver_args", {}).items()),
                tuple(getattr(fut, "time_scalars", None) or unit_test_time_scalars(fut)),
            )
            if names.tag in tags:
                raise InstrumentError(f"case name clashes with {tags[names.tag]} (tag {names.tag})")
            case_symbols = _check_case(fut, mods_dir)
            path = srcroot / fut.driver_sites.path
            if path not in texts:
                texts[path] = path.read_text().splitlines(keepends=True)
            _, bounds = instrument_lines(texts[path], [(fut.driver_sites, names)])
            for mod in case_symbols:
                if mod not in fut.module_dict:
                    raise InstrumentError(f"module {mod} is not part of the unit test analysis")
        except InstrumentError as err:
            result.failures[name] = str(err)
            continue
        tags[names.tag] = name
        accepted.append((fut, names))
        case_symbols_of[name] = case_symbols
        for mod, names_ in case_symbols.items():
            symbols.setdefault(mod, {}).update(names_)
            fort_mods.setdefault(mod, fut.module_dict[mod])
        for mod, types in component_symbols(fut).items():
            if mod not in fut.module_dict:
                continue
            fort_mods.setdefault(mod, fut.module_dict[mod])
            for type_name, comps_ in types.items():
                comp_syms.setdefault(mod, {}).setdefault(type_name, {}).update(comps_)
        result.reports[name] = InstrumentReport(
            path, fut.driver_sites.module, bounds[names.tag], tag=names.tag,
            case_dir=Path(fut.case_dir), subset=subsets.get(name),
        )

    # Accessibility edits keep line counts, so recorded call lines stay valid.
    edits: dict[Path, list[str]] = {}
    for mod, names_ in sorted(symbols.items()):
        fort_mod = fort_mods[mod]
        mod_path = srcroot / Path(fort_mod.filepath).resolve().relative_to(srcroot.resolve())
        src = texts.get(mod_path) or mod_path.read_text().splitlines(keepends=True)
        src, changed = make_accessible(
            strip_instrumentation(src), module_head(fort_mod), names_, str(mod_path)
        )
        if changed:
            edits[mod_path] = src
            result.made_public[mod_path] = changed
    for mod, types in sorted(comp_syms.items()):
        mod_path = srcroot / Path(fort_mods[mod].filepath).resolve().relative_to(srcroot.resolve())
        src = edits.get(mod_path) or strip_instrumentation(
            texts.get(mod_path) or mod_path.read_text().splitlines(keepends=True)
        )
        src, changed = components_public(src, types, str(mod_path))
        if changed:
            edits[mod_path] = src
            result.made_public.setdefault(mod_path, []).extend(changed)
    access_ledger = load_access_ledger(srcroot)
    for mod_path in result.made_public:
        before = strip_instrumentation(
            texts.get(mod_path) or mod_path.read_text().splitlines(keepends=True)
        )
        rel = str(mod_path.resolve().relative_to(srcroot.resolve()))
        record_access_edits(access_ledger, rel, before, edits[mod_path])
    by_file: dict[Path, list] = {}
    for fut, names in accepted:
        by_file.setdefault(srcroot / fut.driver_sites.path, []).append((fut.driver_sites, names))
    for path, groups in by_file.items():
        edits[path], _ = instrument_lines(edits.get(path, texts[path]), groups)

    for fut, names in accepted:
        report = result.reports[fut.case_name]
        for mod, case_names in case_symbols_of[fut.case_name].items():
            mod_path = srcroot / Path(fort_mods[mod].filepath).resolve().relative_to(srcroot.resolve())
            if mine := [n for n in result.made_public.get(mod_path, []) if n in case_names]:
                report.made_public[mod_path] = mine
        for mod, types in component_symbols(fut).items():
            if mod not in fort_mods:
                continue
            mod_path = srcroot / Path(fort_mods[mod].filepath).resolve().relative_to(srcroot.resolve())
            mine = [
                n for n in result.made_public.get(mod_path, [])
                if n.split("%")[0].split(" ")[0] in types
            ]
            if mine:
                report.made_public.setdefault(mod_path, []).extend(mine)
        report.copied = [
            *(dest / f for f in sorted(SHARED_IO_MODULES)),
            *(dest / f"{names.io_module(f.removesuffix('.F90'))}.F90" for f in CASE_IO_MODULES),
            dest / f"{names.module}.F90",
        ]
    if dry_run:
        return result

    result.removed = remove_io_modules(dest)
    for edit_path, edit_lines in edits.items():
        edit_path.write_text("".join(edit_lines))
    write_access_ledger(srcroot, access_ledger)
    if accepted:
        for f in sorted(SHARED_IO_MODULES):
            shutil.copy2(_io_source(Path(accepted[0][0].case_dir), f, mods_dir), dest / f)
    for fut, names in accepted:
        case_dir = Path(fut.case_dir)
        for f in CASE_IO_MODULES:
            out = dest / f"{names.io_module(f.removesuffix('.F90'))}.F90"
            out.write_text(rename_io_modules((case_dir / f).read_text(), names))
        capture = generate_capture_module(
            names, capture_instances(fut), freq=freq, subset=subsets.get(fut.case_name),
            compact=compact_capture_instances(fut),
        )
        (dest / f"{names.module}.F90").write_text(capture)
        (case_dir / CAPTURE_DIR).mkdir(exist_ok=True)
        (case_dir / CAPTURE_DIR / f"{names.module}.F90").write_text(capture)
        report = result.reports[fut.case_name]
        (case_dir / MANIFEST).write_text(json.dumps(report.to_json(), indent=2) + "\n")
    return result


def instrument_elm(
    fut: FunctionalUnitTest,
    srcroot: Path,
    freq: int = DEFAULT_FREQ,
    dry_run: bool = False,
    mods_dir: Optional[Path] = None,
    subset: Optional[SubsetSpec] = None,
) -> InstrumentReport:
    """Instrument ELM for a single case."""
    result = instrument_cases(
        [fut], srcroot, freq=freq, dry_run=dry_run, mods_dir=mods_dir,
        subsets={fut.case_name: subset} if subset else None,
    )
    if fut.case_name in result.failures:
        raise InstrumentError(result.failures[fut.case_name])
    return result.reports[fut.case_name]


def uninstrument_elm(srcroot: Path) -> list[Path]:
    """
    Remove inserted capture lines and the copied IO modules, and revert the
    accessibility edits recorded in the access ledger. Returns the restored
    source files. Ledger entries that couldn't be reverted (the edited line
    changed since) stay in the ledger; see `load_access_ledger`.
    """
    srcroot = Path(srcroot)
    restored, _ = revert_access_edits(srcroot)
    for path in elm_source_files(srcroot):
        text = path.read_text(errors="replace")
        if MARKER not in text:
            continue
        lines = text.splitlines(keepends=True)
        stripped = strip_instrumentation(lines)
        if stripped != lines:
            path.write_text("".join(stripped))
            restored.append(path)
    remove_io_modules(srcroot / "components/elm/src/main")
    return sorted(set(restored))


def undo_instrumentation(srcroot: Path) -> list[Path]:
    """`uninstrument_elm` and report what was restored / left behind."""
    restored = uninstrument_elm(srcroot)
    for path in restored:
        print(f"Restored {path} (SPEL capture calls / accessibility edits)")
    for rel, entries in load_access_ledger(srcroot).items():
        lns = ", ".join(str(e["ln"] + 1) for e in entries)
        print(f"Warning: couldn't revert accessibility edits in {rel} (lines {lns}); "
              f"they changed since `spel instrument`. See {access_ledger_path(srcroot)}")
    return restored


# --------------------------------------------------------------------------
# CIME case
# --------------------------------------------------------------------------
class CaseRunError(InstrumentError):
    """The casegen script failed; `build_failed` tells a build from a run failure."""

    def __init__(self, msg: str, casedir: Optional[Path] = None, build_failed: bool = False):
        super().__init__(msg)
        self.casedir = casedir
        self.build_failed = build_failed


def build_errors(casedir: Path, max_lines: int = 20) -> Optional[str]:
    """
    Compiler errors of the last failed case.build (None if the last build
    didn't fail), taken from the build log CaseStatus points to.
    """
    status = Path(casedir) / "CaseStatus"
    if not status.is_file():
        return None
    text = status.read_text(errors="replace")
    builds = re.findall(r"case\.build (success|error)(.*?)(?=\n\s*-{5,}|\Z)", text, re.S)
    if not builds or builds[-1][0] != "error":
        return None
    detail = builds[-1][1].strip()
    if m := re.search(r"cat (\S+)", detail):
        log = Path(m.group(1))
        lines = log.read_text(errors="replace").splitlines() if log.is_file() else []
        errs = []
        for i, line in enumerate(lines):
            if re.match(r"\s*(Error|Fatal Error):", line):
                loc = next((lines[j] for j in range(i - 1, max(i - 6, -1), -1)
                            if re.match(r"\S+\.F90:\d+", lines[j])), "")
                errs.extend([loc, line] if loc else [line])
        if errs:
            return f"{log}:\n" + "\n".join(errs[:max_lines])
    return detail or "case.build failed"


def _rundir(casedir: Path) -> Path:
    return Path(subprocess.run(
        ["./xmlquery", "RUNDIR", "--value"],
        cwd=casedir, check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip())


def _stream(cmd: list[str], cwd: Optional[Path] = None) -> tuple[int, Optional[str]]:
    """Run cmd echoing its output; (exit code, CASEDIR it printed)."""
    print("Running:", " ".join(cmd))
    casedir = None
    # no stdin: mpiexec (hydra) forwards it and aborts if it is e.g. `yes | spel ...`
    with subprocess.Popen(
        cmd, cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as proc:
        for line in proc.stdout:
            print(line, end="")
            if line.startswith("CASEDIR="):
                casedir = line.removeprefix("CASEDIR=").strip()
    return proc.returncode, casedir


def run_case(
    casegen: Path,
    case_name: str,
    srcroot: Path,
    extra_args: Optional[list[str]] = None,
) -> Path:
    """
    Create/build/submit a case with the casegen script; return its RUNDIR.
    With SPEL_CASE_RUNNER set (e.g. ~/scripts/run_e3sm.sh), the case is only
    built and then run in place by that script from the case directory (no
    batch job: small cases run on the login or an interactive node).
    """
    runner = os.environ.get("SPEL_CASE_RUNNER")
    cmd = [
        "bash", str(casegen),
        "--case", case_name,
        "--srcroot", str(srcroot),
        "--build", *([] if runner else ["--submit"]),
        *(extra_args or []),
    ]
    code, casedir = _stream(cmd)
    if code:
        if casedir and (errors := build_errors(Path(casedir))):
            raise CaseRunError(f"ELM build failed: {errors}", Path(casedir), build_failed=True)
        raise CaseRunError(
            f"{casegen.name} failed with exit code {code}",
            Path(casedir) if casedir else None,
        )
    if casedir is None:
        raise InstrumentError("casegen script didn't report CASEDIR")
    if runner:
        code, _ = _stream(["bash", str(Path(runner).expanduser())], cwd=casedir)
        if code:
            raise CaseRunError(f"{runner} failed with exit code {code}", Path(casedir))
    else:
        wait_for_case_jobs(Path(casedir))
    return _rundir(Path(casedir))


def resubmit_case(casedir: Path) -> Path:
    """Re-run an already built case (`./case.submit`); return its RUNDIR."""
    code, _ = _stream(["./case.submit"], cwd=casedir)
    if code:
        raise CaseRunError(f"case.submit failed with exit code {code}", casedir)
    wait_for_case_jobs(casedir)
    return _rundir(casedir)


_JOB_DONE = {"COMPLETED"}
_JOB_FAILED = {"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "DEADLINE", "PREEMPTED", "BOOT_FAIL"}


def case_job_ids(job_ids: str) -> list[str]:
    """Slurm job ids in a case's JOB_IDS value (`case.run:123,case.st_archive:124`)."""
    return re.findall(r"(?<=:)\d+", job_ids)


def job_state(sacct_output: str) -> str:
    """State of a job from `sacct -X -n -o State%20` ('CANCELLED by 1234' -> 'CANCELLED')."""
    first = sacct_output.strip().splitlines()[0].strip() if sacct_output.strip() else ""
    return first.split()[0] if first else ""


def wait_for_case_jobs(
    casedir: Path,
    timeout_s: float = 4 * 3600,
    poll_s: float = 30.0,
    run=None,
    sleep=None,
) -> None:
    """
    On a batch machine `case.submit` only queues the run: wait for the case's
    Slurm jobs (JOB_IDS) to finish, so that the RUNDIR holds the outputs when
    they are collected. A case run on the login node has no JOB_IDS and returns
    at once. Raises CaseRunError if a job ends in a failed state or the wait
    times out.
    """
    import time as _time
    run = run or (lambda cmd, cwd=None: subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout)
    sleep = sleep or _time.sleep
    try:
        ids = case_job_ids(run(["./xmlquery", "JOB_IDS", "--value"], cwd=casedir))
    except (OSError, subprocess.SubprocessError):
        return
    if not ids or not shutil.which("sacct"):
        return
    print(f"waiting for Slurm job(s) {', '.join(ids)} of {casedir.name}", flush=True)
    waited = 0.0
    pending = list(ids)
    while pending:
        for job in list(pending):
            state = job_state(run(["sacct", "-j", job, "-X", "-n", "-o", "State%20"]))
            if state in _JOB_DONE:
                pending.remove(job)
            elif state in _JOB_FAILED:
                raise CaseRunError(f"Slurm job {job} ended in state {state}", casedir)
        if pending:
            if waited >= timeout_s:
                raise CaseRunError(f"Slurm job(s) {', '.join(pending)} still not finished after {int(waited)} s", casedir)
            sleep(poll_s)
            waited += poll_s


def collect_outputs(rundir: Path, dest: Path, prefix: str = "") -> list[Path]:
    """Copy <prefix>spel-*.nc from the run directory to dest/spel-*.nc."""
    copied = []
    for f in sorted(Path(rundir).glob(f"{prefix}spel-*.nc")):
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / f.name.removeprefix(prefix)
        shutil.copy2(f, out)
        copied.append(out)
    return copied


def write_reference_meta(dest: Path, case_name: str, subset: Optional[SubsetSpec] = None) -> Path:
    """Record which E3SM checkout (and gridcell subset) produced the reference data in dest."""
    from spel.scripts.analysis_cache import current_meta
    from spel.scripts.config import REFERENCE_META_FILENAME

    meta = {
        "case_name": case_name,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "subset": subset.to_json() if subset else None,
    } | current_meta()
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / REFERENCE_META_FILENAME
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def instrument_case(
    case: str,
    freq: int = DEFAULT_FREQ,
    run: bool = False,
    case_args: Optional[list[str]] = None,
    dry_run: bool = False,
    subset: Optional[SubsetSpec] = None,
) -> InstrumentReport:
    """`spel instrument`: instrument ELM for a unit-test case, optionally run it."""
    from spel.scripts.config import (
        CASEGEN_SCRIPT,
        E3SM_SRCROOT,
        input_data_dir,
        spel_mods_dir,
    )

    fut = load_case(case)
    report = instrument_elm(
        fut, E3SM_SRCROOT, freq=freq, dry_run=dry_run, mods_dir=spel_mods_dir, subset=subset
    )

    verb = "Would instrument" if dry_run else "Instrumented"
    print(f"{verb} {report.call_site} ({report.module}, bounds={report.bounds})")
    if subset:
        print(f"  gridcell subset: {subset.description}")
    for path, names in report.made_public.items():
        print(f"  public/unprotected in {path.name}: {', '.join(names)}")
    print(f"  IO modules -> {report.copied[0].parent}")
    if dry_run or not run:
        return report

    if not CASEGEN_SCRIPT.exists():
        raise InstrumentError(f"casegen script {CASEGEN_SCRIPT} not found (set SPEL_CASEGEN)")
    rundir = run_case(CASEGEN_SCRIPT, fut.case_name, E3SM_SRCROOT, case_args)
    prefix = CaptureNames.for_case(fut.case_name).file_prefix
    copied = collect_outputs(rundir, Path(input_data_dir) / fut.case_name, prefix)
    if not copied:
        raise InstrumentError(f"No {prefix}spel-*.nc files were written to {rundir}")
    write_reference_meta(copied[0].parent, fut.case_name, subset)
    print(f"Copied {len(copied)} files from {rundir} to {copied[0].parent}")
    return report
