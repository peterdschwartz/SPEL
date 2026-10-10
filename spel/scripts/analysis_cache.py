"""
Cached elm_drv analysis.

Every unit test is extracted from one analysis whose root is elm_drv: its
call tree (and that of any routine whose call the edited driver comments
out) is analyzed once, and the result is pickled with the edited module
sources. A routine that fails to analyze is recorded (with the routines
whose call trees contain it) instead of aborting the build. `spel create` then only selects
the requested routines and redoes the root-dependent steps (bindings to
elm_drv's actuals, active variables, generated files), unless the user asks
for a re-analysis (e.g. after switching E3SM or SPEL branches).

Each E3SM checkout (E3SM_SRCROOT, `--srcroot`) has its own cache, so a
reference checkout and a development checkout can be used side by side:

    <unittests_dir>/.spel-cache/<srcroot name>-<hash of its path>/elm_drv/
        analysis.pkl   FunctionalUnitTest of the whole elm_drv analysis
        meta.json      SPEL / E3SM commits it was built from
        src/           edited module sources
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import spel.scripts.config as cfg
from spel.scripts.config import E3SM_SRCROOT, default_mods, unittests_dir
from spel.scripts.driver_callsites import (
    DRIVER_MODULE,
    DRIVER_ROUTINE,
    MARKER,
    CallerRoutine,
    DriverSites,
    caller_sites,
    call_path,
    callers_of,
    compose_bindings,
    load_module,
    routine_lines,
)
from spel.scripts.fortran_modules import (
    DECLARATION_ONLY_MODULES,
    FortranModule,
    declaration_source,
    get_filename_from_module,
    imported_from,
)
from spel.scripts.fortran_parser.scope_walk import CallEvent
from spel.scripts.fortran_parser.spel_ast import SemanticError
from spel.scripts.record_access import ArgBinding
from spel.scripts.functional_unit_test import FunctionalUnitTest
from spel.scripts.logging_configs import get_logger

ANALYSIS_PICKLE = "analysis.pkl"
META_FILE = "meta.json"
SRC_DIR = "src"
DRIVER_ID = f"{DRIVER_MODULE}::{DRIVER_ROUTINE}"
SPEL_ROOT = Path(__file__).resolve().parents[2]


def cache_root() -> Path:
    return Path(unittests_dir) / ".spel-cache"


def cache_key(srcroot: Path) -> str:
    """Readable and unique per checkout: <dir name>-<8 hex digits of its path>."""
    root = Path(srcroot).expanduser().resolve()
    return f"{root.name}-{hashlib.sha1(str(root).encode()).hexdigest()[:8]}"


def cache_dir(srcroot: Optional[Path] = None) -> Path:
    return cache_root() / cache_key(srcroot or E3SM_SRCROOT) / DRIVER_ROUTINE


def legacy_cache_dir() -> Path:
    """Where the single, srcroot-independent cache used to live."""
    return cache_root() / DRIVER_ROUTINE


def migrate_legacy_cache(logger: Optional[logging.Logger] = None) -> bool:
    """Move a pre-per-srcroot cache to cache_dir() if it was built from E3SM_SRCROOT."""
    old, new = legacy_cache_dir(), cache_dir()
    meta_path = old / META_FILE
    if new.exists() or not meta_path.is_file():
        return False
    try:
        saved = json.loads(meta_path.read_text()).get("e3sm_srcroot")
    except (OSError, ValueError):
        return False
    if saved is None or Path(saved).resolve() != Path(E3SM_SRCROOT).resolve():
        return False
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(new))
    if logger:
        logger.info(f"Moved the elm_drv analysis cache of {E3SM_SRCROOT} to {new}")
    return True


def list_caches() -> list[tuple[Path, dict]]:
    """(cache dir, meta.json contents) of every per-srcroot cache."""
    out = []
    for meta_path in sorted(cache_root().glob(f"*/{DRIVER_ROUTINE}/{META_FILE}")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            meta = {}
        out.append((meta_path.parent, meta))
    return out


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------
def git_state(repo: Path) -> dict[str, Optional[str]]:
    def git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True, text=True, check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def current_meta() -> dict:
    return {
        "e3sm_srcroot": str(E3SM_SRCROOT),
        "e3sm": git_state(Path(E3SM_SRCROOT)),
        "spel": git_state(SPEL_ROOT),
    }


def stale_reasons(saved: dict, now: dict) -> list[str]:
    """Why a cache built with `saved` metadata may not match the current sources."""
    reasons = []
    if saved.get("e3sm_srcroot") != now["e3sm_srcroot"]:
        reasons.append(f"E3SM_SRCROOT {saved.get('e3sm_srcroot')} -> {now['e3sm_srcroot']}")
    for repo in ("e3sm", "spel"):
        old, new = saved.get(repo, {}), now[repo]
        if old.get("commit") != new["commit"]:
            reasons.append(
                f"{repo.upper()} {old.get('branch')}@{(old.get('commit') or '?')[:10]}"
                f" -> {new['branch']}@{(new['commit'] or '?')[:10]}"
            )
    return reasons


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def instrumented_files(srcroot: Path) -> list[Path]:
    from spel.scripts.instrument_elm import elm_source_files

    return [
        p for p in elm_source_files(srcroot)
        if MARKER in p.read_text(errors="replace")
    ]


def driver_callees(unit_test: FunctionalUnitTest) -> dict:
    """
    The routines elm_drv calls, resolved in elm_driver's (unedited) scope.
    elm_driver's own routines are left out: its edited copy cannot be part of
    a unit test (elm_instMod is replaced by SPEL's).
    """
    from spel.scripts.module_resolver import ModuleScopes
    from spel.scripts.record_access import AccessMapper

    sub_dict = unit_test.subroutine_dict
    drv_mod, lines = load_module(DRIVER_MODULE)
    drv = CallerRoutine(DRIVER_MODULE, DRIVER_ROUTINE, routine_lines(lines, DRIVER_ROUTINE))
    scopes = ModuleScopes({**unit_test.module_dict, DRIVER_MODULE: drv_mod}, sub_dict)
    mapper = AccessMapper(sub_dict, scopes)
    rec = drv.walk_syntax_tree(scopes)
    callees = {}
    for e in rec.events:
        if not isinstance(e, CallEvent):
            continue
        sub = mapper.callee(drv, rec, e)
        if sub is None or sub.library or sub.module == DRIVER_MODULE:
            continue
        callees[sub.id] = sub
    return dict(sorted(callees.items()))


def driver_roots(unit_test: FunctionalUnitTest) -> dict:
    """
    elm_drv first (its call tree covers every routine it reaches, at any
    depth), then the routines elm_drv calls: only those the edited driver
    doesn't reach (calls commented out) get analyzed as roots of their own.
    """
    return {DRIVER_ID: unit_test.subroutine_dict[DRIVER_ID]} | driver_callees(unit_test)


def build_analysis(logger: Optional[logging.Logger] = None) -> Path:
    """Analyze everything elm_drv calls; write the cache. Returns the cache dir."""
    from spel.scripts.export_objects import dump_unit_test
    from spel.scripts.UnitTestforELM import analyze_unit_test

    logger = logger or get_logger("SPEL", level=logging.INFO)
    if marked := instrumented_files(Path(E3SM_SRCROOT)):
        sys.exit(
            "Error- E3SM is instrumented by SPEL (" + ", ".join(p.name for p in marked)
            + "); run `spel instrument --undo` before analyzing elm_drv"
        )
    out = cache_dir()
    if out.exists():
        shutil.rmtree(out)
    src = out / SRC_DIR
    src.mkdir(parents=True)

    logger.info(f"Analyzing {DRIVER_ID} in {E3SM_SRCROOT} (cache: {out})")
    unit_test = FunctionalUnitTest(casedir=src, cfg=cfg.options, logger=logger)
    analyze_unit_test(unit_test, [DRIVER_ID], roots_from=driver_roots)
    if unit_test.subroutine_dict[DRIVER_ID].record_access is None:
        logger.error(f"The {DRIVER_ID} analysis failed: only its callees were analyzed")

    dump_unit_test(unit_test, out / ANALYSIS_PICKLE)
    meta = current_meta() | {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "roots": list(unit_test.primary_subroutines),
        "mapped": sum(s.record_access is not None for s in unit_test.subroutine_dict.values()),
        "failures": unit_test.analysis_failures,
        "incomplete": unit_test.analysis_incomplete,
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    logger.info(
        f"Cached {DRIVER_ID} analysis: {meta['mapped']} routines mapped, "
        f"{len(unit_test.analysis_failures)} failed, "
        f"{len(unit_test.analysis_incomplete)} incomplete (see {out / META_FILE})"
    )
    return out


def load_analysis(
    reanalyze: bool = False, logger: Optional[logging.Logger] = None
) -> FunctionalUnitTest:
    """
    A fresh copy of the cached elm_drv analysis, built first if missing or if
    `reanalyze`. A cache built from other commits is only warned about.
    """
    from spel.scripts.export_objects import _load_pickle

    logger = logger or get_logger("SPEL", level=logging.INFO)
    migrate_legacy_cache(logger)
    path = cache_dir() / ANALYSIS_PICKLE
    if reanalyze or not path.is_file():
        build_analysis(logger)
    else:
        meta_path = cache_dir() / META_FILE
        saved = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        for reason in stale_reasons(saved, current_meta()):
            logger.warning(
                f"Cached elm_drv analysis may be stale ({reason}); "
                "use --reanalyze to rebuild it"
            )
    logger.info(f"Loading cached {DRIVER_ID} analysis from {path}")
    return _load_pickle(path)


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------
def module_closure(
    mod_dict: dict[str, FortranModule], seeds: Iterable[str], type_dict: Optional[dict] = None
) -> set[str]:
    """
    `seeds` and every module they use (transitively), within mod_dict.
    Declaration-only modules (elm_instMod) are replaced by the modules that
    provide the names actually imported from them.
    """
    out: set[str] = set()
    stack = [m for m in seeds if m in mod_dict]
    while stack:
        mod = stack.pop()
        if mod in out or mod in DECLARATION_ONLY_MODULES:
            continue
        out.add(mod)
        stack.extend(m for m in mod_dict[mod].modules if m in mod_dict and m not in out)
        for provider in DECLARATION_ONLY_MODULES & set(mod_dict[mod].modules):
            for name in imported_from(mod_dict, [mod], provider):
                source = declaration_source(mod_dict[provider], name, type_dict or {})
                if source is not None and source[0] in mod_dict:
                    stack.append(source[0])
    return out


def reset_active(unit_test: FunctionalUnitTest) -> None:
    for dtype in unit_test.type_dict.values():
        dtype.active = False
        for comp in dtype.components.values():
            comp.active = False
        for inst in dtype.instances.values():
            inst.active = False


def call_sites(
    fut: FunctionalUnitTest, selected: dict, logger: logging.Logger
) -> tuple[DriverSites, list[ArgBinding]]:
    """
    Where to capture the selected routines, and their dummies bound to what
    elm_drv passes. Routines elm_drv calls directly are captured in elm_drv;
    others in their immediate caller (on the shortest call chain from
    elm_drv), with arguments composed through the chain.
    """
    sites, sub_dict = fut.driver_sites, fut.subroutine_dict
    direct = set(sites.calls) if sites is not None else set()
    if sites is None:
        sys.exit(f"Error- the cached analysis has no {DRIVER_ROUTINE} call sites; rebuild it")
    if all(s in direct for s in selected):
        return (
            DriverSites(
                module=sites.module,
                routine=sites.routine,
                path=sites.path,
                calls={k: v for k, v in sites.calls.items() if k in selected},
                capture_lns=list(sites.capture_lns),
            ),
            [b for b in fut.driver_bindings if b.callee in selected],
        )
    if any(s in direct for s in selected):
        sys.exit(
            f"Error- can't mix routines called directly by {DRIVER_ROUTINE} with nested ones: "
            f"{sorted(selected)}"
        )
    paths = {}
    for s in selected:
        path = call_path(sub_dict, direct, s)
        if path is None:
            sys.exit(f"Error- {s} is not reached from any routine {DRIVER_ROUTINE} calls")
        paths[s] = path
    callers = {path[-2] for path in paths.values()}
    if len(callers) > 1:
        sys.exit(
            "Error- the selected routines must share their caller: "
            + ", ".join(f"{s} <- {p[-2]}" for s, p in paths.items())
        )
    caller = sub_dict[callers.pop()]
    for s, path in paths.items():
        logger.info(f"Call chain: {DRIVER_ROUTINE} -> {' -> '.join(path)}")
        if others := [c for c in callers_of(sub_dict, s) if c != caller.id]:
            logger.warning(f"{s} is also called by {others}; capturing only in {caller.id}")
    bindings = [b for p in paths.values() for b in compose_bindings(sub_dict, fut.driver_bindings, p)]
    try:
        nested_sites = caller_sites(caller, [sub_dict[s] for s in selected])
    except SemanticError as err:
        sys.exit(f"Error- {err}")
    return nested_sites, bindings


def extract_unit_test(
    analysis: FunctionalUnitTest,
    sub_name_list: list[str],
    case_dir: Path,
    logger: logging.Logger,
) -> FunctionalUnitTest:
    """
    Narrow (in place) the elm_drv analysis to the unit test of the routines
    in sub_name_list: their modules, the root-dependent access views, and the
    edited sources copied into case_dir. Generation is left to the caller.
    """
    from spel.scripts.UnitTestforELM import adopt_record_access, select_subroutines

    fut = analysis
    src = cache_dir() / SRC_DIR
    for sub in fut.subroutine_dict.values():
        sub.unit_test_function = False
    selected = select_subroutines(fut.subroutine_dict, sub_name_list, logger)
    if failed := {s: fut.analysis_failures[s] for s in selected if s in fut.analysis_failures}:
        sys.exit(f"Error- the elm_drv analysis failed for: {json.dumps(failed, indent=2)}")
    incomplete = getattr(fut, "analysis_incomplete", {})
    if partial := {s: incomplete[s] for s in selected if s in incomplete}:
        sys.exit(
            "Error- the analysis failed for routines called by the selection "
            f"(see {cache_dir() / META_FILE}): {json.dumps(partial, indent=2)}"
        )
    unreached = [s for s, sub in selected.items() if sub.record_access is None]
    if unreached:
        sys.exit(f"Error- not reached from {DRIVER_ROUTINE} in the cached analysis: {unreached}")
    sites, bindings = call_sites(fut, selected, logger)

    mods = module_closure(
        fut.module_dict,
        {s.module for s in selected.values()} | set(default_mods),
        fut.type_dict,
    )
    fut.ordered_mods = [m for m in fut.ordered_mods if m in mods]
    fut.module_dict = {
        m: mod
        for m, mod in fut.module_dict.items()
        if m in mods or m in DECLARATION_ONLY_MODULES
    }
    fut.subroutine_dict = {k: s for k, s in fut.subroutine_dict.items() if s.module in mods}
    fut.type_dict = {
        k: t for k, t in fut.type_dict.items() if t.declaration in mods or k == "bounds_type"
    }
    reset_active(fut)

    fut.primary_subroutines = selected
    setfilters = fut.subroutine_dict["filtermod::setfilters"]
    setfilters.unit_test_function = True
    fut.driver_sites = sites
    fut.driver_bindings = bindings
    fut.case_dir = case_dir
    fut.case_name = case_dir.name
    fut.logger = logger
    fut.active_global_vars = {}
    fut.guarded_usage_dict = {}

    roots = {s.id for s in fut.subroutine_dict.values() if s.unit_test_function}
    adopt_record_access(fut.subroutine_dict, fut.type_dict, roots, fut.driver_bindings)

    for mod in fut.ordered_mods:
        fn = get_filename_from_module(mod)
        if fn is None:
            sys.exit(f"Error- no source file for module {mod}")
        shutil.copy2(src / Path(fn).name, case_dir / Path(fn).name)
    logger.info(
        f"Extracted {list(selected)} from the {DRIVER_ROUTINE} analysis: "
        f"{len(fut.ordered_mods)} modules"
    )
    return fut
