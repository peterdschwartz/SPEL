"""
Targeted host/device transitions for elm_drv.

The whole-state transitions (`elm_device_update` before every elm_drv call that changes
sides) copy ~13 GB per step on the stride-8 case. Here the transitions only switch
`elm_device_ready`, and acc-sync moves the data each host segment of elm_drv uses:

- `update self` of what a host segment reads or writes that device code writes, before it;
  `update device` of what it writes, after it (acc_sync.plan rooted at elm_drv, with the
  call kinds of the transitions: `generate.call_kind`);
- at elm_drv's first statement, `update device` of what the coupler code around elm_drv
  writes (lnd_run_mct: lnd_import, the forcing downscaling); before its end, `update self`
  of what that code reads and device code writes (lnd_export, the next step's import).
  These routines are outside the analysis; they are scope-walked here (`step_boundary`);
- the clump filters (arrays of derived types, which acc-sync cannot update member by
  member) through `elm_device_filters_update` around the segments that touch them;
- the whole state before `restFile_write` (`elm_device_pull_all`): it reads the state
  through the pointers registered by restartvar, which the analysis does not follow.
  History needs nothing extra: hist_htapes_wrapup writes the hbuf (pulled in
  hist_compute_update_host) and ldomain/level constants the device never writes.

Every inserted update carries `if(.not. elm_device_fullsync)`: with SPEL_DEVICE_FULLSYNC=1
the whole-state transitions run instead, so both modes are in one build for A/B runs.
Everything placed by the analysis's lines: the driver must be the file `spel analyze` read
(`strip` removes an earlier application first; `apply` refuses a mismatch).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spel.transforms.device.generate import MODULE_NAME, DeviceDataError, call_kinds

TAG = "! spel targeted"
COND = ".not. elm_device_fullsync"
DRV = "elm_driver::elm_drv"
# the coupler routine that calls elm_drv, and the modules of the code it runs around it
RUN_ROUTINE = ("lnd_comp_mct", "lnd_run_mct")
CPL_MODULES = ("lnd_import_export", "lnd_downscale_atm_forcing", "lnd_disagg_forc", "lnd_comp_mct")
FILTER_INSTANCES = ("filter", "filter_inactive_and_active")
PULL_ALL_BEFORE = {"restfile_write"}
USE_NAMES = ("elm_device_fullsync", "elm_device_pull_all", "elm_device_filters_update")


@dataclass
class Targeted:
    plan: object  # acc_sync.SyncPlan, elm_drv only
    before: list[tuple[str, str]] = field(default_factory=list)  # coupler accesses before elm_drv
    after: list[tuple[str, str]] = field(default_factory=list)
    report: list[str] = field(default_factory=list)


def step_boundary(fut) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    """
    (path, status) of the global accesses of lnd_run_mct outside its elm_drv call: what the
    coupler code writes before a step and reads after it (it runs both around every call).
    The cpl routines are scope-walked like elm_drv's callers (driver_callsites), callees
    outside the analysis included, transitively.
    """
    from spel.scripts.driver_callsites import CallerRoutine, load_module, routine_lines
    from spel.scripts.fortran_parser.symbols import SymbolKind
    from spel.scripts.module_resolver import ModuleScopes
    from spel.scripts.record_access import AccessMapper, callsite_globals

    mods: dict[str, tuple] = {}

    def module(m: str):
        if m not in mods:
            mods[m] = load_module(m)
        return mods[m]

    scopes = ModuleScopes(dict(fut.module_dict), fut.subroutine_dict)
    for m in CPL_MODULES:
        scopes.mod_dict[m] = module(m)[0]
    subs = {k: s for k, s in fut.subroutine_dict.items() if s.record_access is not None}
    extra: dict[str, CallerRoutine] = {}
    notes: list[str] = []

    def add(m: str, r: str) -> None:
        sid = f"{m}::{r}"
        if sid in subs or sid in extra:
            return
        fm, lines = module(m)
        scopes.mod_dict.setdefault(m, fm)
        cr = CallerRoutine(m, r, routine_lines(lines, r))
        extra[sid] = cr
        rec = cr.walk_syntax_tree(scopes)
        for e in rec.calls:
            sym = rec.lookup(e.name)
            if sym is None or sym.kind is not SymbolKind.PROCEDURE or not sym.target or "::" not in sym.target:
                continue
            if sym.target not in subs:
                try:
                    add(*sym.target.split("::"))
                except Exception as err:  # a callee that does not walk: its effects are unknown
                    notes.append(f"step boundary: {sym.target} not walked ({type(err).__name__}: {err})")

    add(*RUN_ROUTINE)
    subs.update(extra)
    run = extra["::".join(RUN_ROUTINE)]
    maps = AccessMapper(subs, scopes).maps(run)
    rec = run.record
    drv_lines = {e.ln for e in rec.calls if e.name.lower() == "elm_drv"}
    if not drv_lines:
        raise DeviceDataError(f"{'::'.join(RUN_ROUTINE)} does not call elm_drv")
    lines = {lt.ln for lt in run.sub_lines} - drv_lines
    acc = callsite_globals(maps, lines)
    before = sorted({(p, "w") for p, rws in acc.items() if any("w" in rw.status for rw in rws)})
    after = sorted({(p, "r") for p, rws in acc.items() if any("r" in rw.status for rw in rws)})
    notes.append(f"step boundary: {len(extra)} cpl routines walked, {len(before)} paths written, {len(after)} read")
    return before, after, notes


def _continuation_end(lines: list[str], ln: int) -> int:
    while ln + 1 < len(lines) and lines[ln].split("!")[0].rstrip().endswith("&"):
        ln += 1
    return ln


def device_spans(fut, lines: list[str]) -> list[tuple[int, int]]:
    """elm_drv's lines that run on the device: its kernels and the calls the transitions put there
    (generate.call_kinds, the kinds _transition_edits gives them)."""
    from spel.transforms.acc_directives import find_directives
    from spel.transforms.acc_sync import _spans

    sub = fut.subroutine_dict[DRV]
    spans = _spans(find_directives(lines, sub.startline, sub.endline), sub.record)
    for ln, _, kind in call_kinds(fut):
        if kind == "device":
            spans.append((ln, _continuation_end(lines, ln)))
    return sorted(spans)


def check_lines(fut, drv: Path) -> None:
    """The analysis's elm_drv calls are on the same lines in the driver file."""
    from spel.scripts.fortran_parser.module_spec import callee_name, parse_modules, statements_with_next

    mod = next(m for m in parse_modules(drv) if m.name.lower() == "elm_driver")
    body = next(p for p in mod.subprograms if p.name.lower() == "elm_drv")
    found = {(st.lineno, name) for st, _ in statements_with_next(body.body, body.end_ln)
             if (name := callee_name(st)) is not None}
    rec = fut.subroutine_dict[DRV].record
    want = {(e.ln, e.name.split("%")[-1].lower()) for e in rec.calls}
    missing = sorted(want - found)
    if missing:
        raise DeviceDataError(
            f"{drv.name}: {len(missing)} elm_drv calls of the analysis are not on its lines "
            f"(first: line {missing[0][0] + 1} {missing[0][1]}); strip, then `spel analyze` again")


def plan(fut, drv: Path) -> Targeted:
    from spel.transforms import acc_sync

    lines = drv.read_text().split("\n")
    before, after, notes = step_boundary(fut)
    sp = acc_sync.plan(fut, [DRV], device_spans={DRV: device_spans(fut, lines)}, boundary={DRV: (before, after)})
    sp.routines = [r for r in sp.routines if r.routine == DRV]
    t = Targeted(sp, before, after, notes)
    for rs in sp.routines:
        for seg in rs.segments:
            t.report.append(
                f"  {seg.first + 1}-{seg.after}: self {len(seg.stale)} device {0 if seg.header_only else len(seg.writes)}"
                + (f" skipped {sorted(seg.skipped)}" if seg.skipped else "") + f"  | {lines[seg.first].strip()[:60]}")
        other = sorted(p for p in rs.skipped if p.split("%")[0] not in FILTER_INSTANCES)
        if other:
            t.report.append(f"not updatable (left to the whole-state pulls): {other}")
    n_self = sum(len(s.stale) for r in sp.routines for s in r.segments)
    n_dev = sum(len(s.writes) for r in sp.routines for s in r.segments if not s.header_only)
    t.report.insert(0, f"elm_drv: {sum(len(r.segments) for r in sp.routines)} segments, "
                       f"{n_self} update self / {n_dev} update device names")
    return t


def strip(lines: list[str]) -> list[str]:
    """Remove an earlier application (TAG lines and the COND updates with their continuations)."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        s = lines[i].lstrip().lower()
        if TAG in lines[i]:
            i += 1
            continue
        if s.startswith("!$acc update"):
            j = i
            while lines[j].rstrip().endswith("&") and j + 1 < len(lines) and lines[j + 1].lstrip().startswith("!$acc&"):
                j += 1
            if f"if({COND})" in lines[j]:
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    return out


def apply(fut, srcroot: Path, dry_run: bool = False) -> list[str]:
    from spel.transforms import acc_sync

    drv = srcroot / "components" / "elm" / "src" / "main" / "elm_driver.F90"
    text = drv.read_text().split("\n")
    stripped = strip(text)
    if stripped != text:
        if not dry_run:
            drv.write_text("\n".join(stripped))
        else:
            raise DeviceDataError(f"{drv.name} carries an earlier application: strip it (apply without --dry-run)")
    check_lines(fut, drv)
    t = plan(fut, drv)
    lines = drv.read_text().split("\n")
    ins = acc_sync.edits(t.plan, {drv: lines}, cond=COND).get(drv, [])
    # acc-sync's own lines (its comment, use statements for callees' globals) carry its TAG
    ins = [(ln, order, [x.replace(acc_sync.TAG, TAG) for x in new]) for ln, order, new in ins]
    from spel.transforms.acc_directives import find_directives
    from spel.transforms.acc_locals import _above_directives

    directives = find_directives(lines, 0, len(lines) - 1)
    for rs in t.plan.routines:
        for seg in rs.segments:
            if any(p.split("%")[0] in FILTER_INSTANCES for p in seg.skipped):
                ind = acc_sync._indent(lines[seg.first])
                ins.append((_above_directives(seg.first, directives), 0,
                            [f"{ind}call elm_device_filters_update(to_device=.false.)  {TAG}"]))
                if not seg.header_only:
                    ins.append((seg.after, -2, [f"{ind}call elm_device_filters_update(to_device=.true.)  {TAG}"]))
    sub = fut.subroutine_dict[DRV]
    for c in sub.record.calls:
        if c.name.split("%")[-1].lower() in PULL_ALL_BEFORE:
            ind = acc_sync._indent(lines[c.ln])
            ins.append((c.ln, 5, [f"{ind}call elm_device_pull_all()  {TAG}: restart writes read the whole state"]))
    use_ln = next((i for i, x in enumerate(lines) if x.lstrip().lower().startswith(f"use {MODULE_NAME.lower()}")), None)
    if use_ln is None:
        raise DeviceDataError(f"{drv.name}: no `use {MODULE_NAME}`")
    ins.append((use_ln + 1, 0, [f"  use {MODULE_NAME}, only : {', '.join(USE_NAMES)}  {TAG}"]))
    for ln, _, new in sorted(ins, key=lambda x: (-x[0], -x[1])):
        lines[ln:ln] = new
    if not dry_run:
        drv.write_text("\n".join(lines))
    return t.report + [f"{drv.name}: {len(ins)} insertions{' (dry run)' if dry_run else ''}"]
