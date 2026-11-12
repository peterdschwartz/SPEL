"""
Step coupling of a case: which of its variables other routines of elm_drv write within one step.

A multi-step run calls the case's routines again and again. Between two calls ELM runs the rest of
the step, and some of those routines write variables the case reads (CanopyHydrology writes
`h2osoi_liq`, ColWaterBalanceCheck drains `top_ws%from_uphill`, ...). Those are the *boundary*
variables of the case: a multi-step scenario must say for each what stands in for the missing
writers (`forcing`, `between` or `keep`, SPEC.md "Multi-step runs").

They come from the static trace of the whole step (db_export.trace.StaticTrace from elm_drv on the
full cached analysis, as the database's call-tree view uses): every write to a case variable
outside the subtrees of the case's routines is a `Writer`, with the elm_drv routine it is under,
the routine and line that write, whether it runs before, between or after the case's routines,
its filter and its namelist gates. `boundary(inv, values)` keeps the writers live under namelist
values (gates not decided false) and on filters that can share an element with the case's own
accesses (db_export.filters).

The trace takes about half a minute; the writers are cached per case under `<case>/proptest/`,
keyed by the analysis and the case's `fut.pkl`.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from spel.proptest.errors import ProptestError
from spel.proptest.inventory import Inventory

log = logging.getLogger(__name__)

DRIVER_ID = "elm_driver::elm_drv"
CACHE_VERSION = 1


@dataclass(frozen=True)
class Writer:
    """A write of a case variable by a routine outside the case, within one step."""

    routine: str       # the elm_drv callee it is under (`soilhydrologymod::...`, or elm_drv itself)
    sub: str           # the routine whose line writes
    file: str          # source file of `sub` (as the analysis records it)
    line: int          # 1-based line in `file`
    status: str        # w | rw
    conditional: bool  # inside a non-namelist if (or a routine called from one)
    filter: str        # filter members of the index ('' = any element)
    when: str          # before | between | after: relative to the case's routines in the step
    gates: tuple = ()  # namelist conditions it runs under (ConditionExpectation)

    def where(self) -> str:
        name = self.sub.split("::")[-1]
        return f"{name} ({Path(self.file).name}:{self.line})"


@dataclass
class Coupling:
    """Writers of every case variable outside the case, and the case's own filters per variable."""

    writers: dict[str, list[Writer]]
    case_filters: dict[str, set[str]]  # inst%field -> filter keys of the case's accesses
    disjoint: set[tuple[str, str]]


def _cache_path(inv: Inventory, analysis: Path) -> Path:
    fut = Path(inv.case_dir) / "fut.pkl"
    key = "|".join(str(x) for x in (CACHE_VERSION, analysis, analysis.stat().st_mtime_ns,
                                    fut.stat().st_mtime_ns if fut.is_file() else 0))
    return Path(inv.case_dir) / "proptest" / f"coupling-{hashlib.sha1(key.encode()).hexdigest()[:12]}.pkl"


def _analysis_path(inv: Inventory) -> tuple[Path, Path]:
    from spel.scripts.analysis_cache import ANALYSIS_PICKLE, cache_dir

    ut = inv.unit_test
    srcroot = getattr(ut, "e3sm_srcroot", None) if ut is not None else None
    if not srcroot:
        raise ProptestError(f"{inv.case_dir}: the case does not record its E3SM checkout (e3sm_srcroot)")
    path = cache_dir(Path(srcroot)) / ANALYSIS_PICKLE
    if not path.is_file():
        raise ProptestError(
            f"no elm_drv analysis for {srcroot} ({path}): the step coupling needs it; "
            f"run `spel analyze --srcroot {srcroot}`"
        )
    return path, Path(srcroot)


def coupling(inv: Inventory, rebuild: bool = False) -> Coupling:
    """The case's coupling to the rest of the step (cached)."""
    path, srcroot = _analysis_path(inv)
    cache = _cache_path(inv, path)
    if cache.is_file() and not rebuild:
        try:
            return pickle.loads(cache.read_bytes())
        except Exception as exc:  # an old or truncated cache: rebuild it
            log.warning("ignoring the coupling cache %s (%s)", cache, exc)
    out = _compute(inv, path, srcroot)
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache.write_bytes(pickle.dumps(out))
    except Exception as exc:
        log.warning("could not cache the coupling (%s)", exc)
    return out


def _compute(inv: Inventory, analysis: Path, srcroot: Path) -> Coupling:
    from spel.scripts.export_objects import _load_pickle

    fut = _load_pickle(analysis, srcroot=srcroot)
    if DRIVER_ID not in fut.subroutine_dict:
        raise ProptestError(f"{analysis} is not an analysis of {DRIVER_ID}")
    wanted = {k for k, v in inv.vars.items() if v.kind in ("input", "inout", "output")}
    return trace_coupling(fut, srcroot, inv.routines, wanted)


def trace_coupling(fut, srcroot: Path, routines, wanted: set[str]) -> Coupling:
    """The coupling of routines `routines` (elm_drv callees, ids) for the variables `wanted`
    (inst%field) in the static trace of elm_drv over the full analysis `fut`."""
    from spel.scripts.db_export import filters as flt
    from spel.scripts.db_export.context import ExportContext
    from spel.scripts.db_export.trace import StaticTrace
    from spel.transforms.core.config import gates

    trace = StaticTrace(ExportContext(fut, {}, Path(srcroot)), [DRIVER_ID]).run()
    nodes = trace.nodes
    routines = {r.lower() for r in routines}
    spans = [(n["order"], n["order"] + n["size"]) for n in nodes
             if n["depth"] == 1 and n["subroutine"].lower() in routines]
    if not spans:
        raise ProptestError(f"none of {sorted(routines)} is called by {DRIVER_ID} in the analysis")
    first, last = min(a for a, _ in spans), max(b for _, b in spans)

    def in_case(order: int) -> bool:
        return any(a <= order < b for a, b in spans)

    def top(order: int) -> str:
        while nodes[order]["depth"] > 1:
            order = nodes[order]["parent_order"]
        return nodes[order]["subroutine"]

    subs = {s.id: s for s in fut.subroutine_dict.values()}
    chain: dict[int, tuple] = {}

    def chain_gates(order: int) -> tuple:
        if order not in chain:
            node = nodes[order]
            parent = node["parent_order"]
            if parent == "" or parent is None:
                chain[order] = ()
            else:
                caller = subs.get(nodes[parent]["subroutine"])
                here = tuple(gates(caller, node["call_ln"])) if caller is not None and node["call_ln"] != "" else ()
                chain[order] = chain_gates(parent) + here
        return chain[order]

    writers: dict[str, list[Writer]] = {}
    case_filters: dict[str, set[str]] = {}
    for a in trace.accesses:
        if not a["instance"] or not a["member"]:
            continue
        key = f"{a['instance']}%{a['member']}".lower()
        if key not in wanted:
            continue
        order = a["node_order"]
        if in_case(order):
            case_filters.setdefault(key, set()).add(flt.filter_members(a["filter"]))
            continue
        if "w" not in a["status"]:
            continue
        sub_id = nodes[order]["subroutine"]
        sub = subs.get(sub_id)
        when = "before" if order < first else ("after" if order >= last else "between")
        own = tuple(gates(sub, a["ln"])) if sub is not None else ()
        writers.setdefault(key, []).append(Writer(
            routine=top(order), sub=sub_id, file=str(getattr(sub, "filepath", "")), line=int(a["ln"]) + 1,
            status=a["status"], conditional=bool(a["conditional"]), filter=a["filter"] or "", when=when,
            gates=chain_gates(order) + own))
    disjoint = flt.disjoint_pairs(flt.derive(trace.filter_sites)[1])
    for key in writers:  # one entry per routine and line (a line may write several times)
        seen, uniq = set(), []
        for w in writers[key]:
            k = (w.sub, w.line, w.filter)
            if k not in seen:
                seen.add(k)
                uniq.append(w)
        writers[key] = uniq
    return Coupling(writers, case_filters, disjoint)


def boundary(inv: Inventory, values: Optional[dict[str, Any]] = None,
             cpl: Optional[Coupling] = None) -> dict[str, list[Writer]]:
    """inst%field -> its writers outside the case that can run under namelist `values` (gates not
    decided false) on elements the case also touches (filters not known disjoint)."""
    from spel.scripts.db_export.filters import filter_members, filters_overlap
    from spel.scripts.validate_access import evaluate_guard

    cpl = cpl or coupling(inv)
    values = values or {}
    out: dict[str, list[Writer]] = {}
    for key, ws in cpl.writers.items():
        mine = cpl.case_filters.get(key, {"*"})
        live = [w for w in ws
                if not any(evaluate_guard(g, values) is False for g in w.gates)
                and filters_overlap({filter_members(w.filter)}, mine, cpl.disjoint)]
        if live:
            out[key] = live
    return out


def describe(writers: list[Writer], n: int = 3) -> str:
    """`phasechangeh2osfc (SoilTemperatureMod.F90:986) in soiltemperature, before; ...` for messages."""
    def one(w: Writer) -> str:
        top = w.routine.split("::")[-1]
        inside = "" if top == w.sub.split("::")[-1] else f" in {top}"
        cond = ", conditional" if w.conditional else ""
        return f"{w.where()}{inside}, {w.when}{cond}"

    parts = [one(w) for w in writers[:n]]
    more = len(writers) - n
    return "; ".join(parts) + (f"; and {more} more" if more > 0 else "")
