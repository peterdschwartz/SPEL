"""
Targeted host/device updates for elm_drv.

The state lives on the device for the whole run (device-data copies it at the end of
initialization; there are no host/device transitions and `elm_device_ready` stays on, so
compute constructs always run on the device). acc-sync moves the data each host segment of
elm_drv uses:

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
  through the pointers registered by restartvar, which the analysis does not follow;
- elm_drv's calls into modules the analysis excludes (ndep_interp, VOCEmission: their calls
  are removed before the analysis, so what they read and write is unknown) get no copies:
  they are reported, and a configuration that runs one aborts in elm_drv's GPU guard (or the
  module gets ported). Before 2026-11-01 every derived-type instance they were passed was
  pulled whole before and pushed whole after.
  History needs nothing extra: hist_htapes_wrapup writes the hbuf (pulled in
  hist_compute_update_host) and ldomain/level constants the device never writes.

Every inserted update block has a TAG comment line above it, so `strip` finds exactly what an
earlier application inserted (it also removes the older form, whose updates carried
`if(.not. elm_device_fullsync)`). Pulls (`update self`) carry acc-sync's `if(elm_device_ready)`.
Everything placed by the analysis's lines: the driver must be the file `spel analyze` read
(`strip` removes an earlier application first; `apply` refuses a mismatch).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spel.transforms.core.edits import EditSet, stacked
from spel.transforms.core.registry import transform
from spel.transforms.core.source import indent
from spel.transforms.core.transform import Change, Plan as TPlan, Refusal, Transform
from spel.transforms.device.generate import MODULE_NAME, DeviceDataError, call_kinds

TAG = "! spel targeted"
# the guard of the updates of applications before 2026-11-01 (whole-state transitions as a
# fallback); `strip` still removes them
LEGACY_COND = ".not. elm_device_fullsync"
DRV = "elm_driver::elm_drv"
# the coupler routine that calls elm_drv, and the modules of the code it runs around it
RUN_ROUTINE = ("lnd_comp_mct", "lnd_run_mct")
CPL_MODULES = ("lnd_import_export", "lnd_downscale_atm_forcing", "lnd_disagg_forc", "lnd_comp_mct")
FILTER_INSTANCES = ("filter", "filter_inactive_and_active")
PULL_ALL_BEFORE = {"restfile_write"}
USE_NAMES = ("elm_device_pull_all", "elm_device_filters_update")


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
    from spel.transforms.core.query import device_spans as _spans

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


def excluded_calls(fut, drv: Path) -> list[tuple[int, int, list[tuple[str, str]]]]:
    """
    (first line, last line, [(instance, type)]) of elm_drv's calls of routines of modules the
    analysis excludes (edit_files.bad_modules, through the use statements of elm_driver and
    elm_drv), with the derived-type instances among their actual arguments.
    """
    from spel.scripts.edit_files import bad_modules
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.scripts.fortran_parser.module_spec import callee_name, parse_modules, statements_with_next
    from spel.scripts.registrations import _designator_parts
    from spel.transforms.acc_sync import _Types
    from spel.transforms.core.query import construct_end as _end_of

    mod = next(m for m in parse_modules(drv) if m.name.lower() == "elm_driver")
    body = next(p for p in mod.subprograms if p.name.lower() == "elm_drv")
    names: set[str] = set()
    for st in list(getattr(mod.spec, "statements", [])) + list(body.body.statements):
        if isinstance(st, A.UseStatement) and st.module.lower() in bad_modules and st.has_only:
            names |= {str(o).split("=>")[0].strip().lower() for o in st.objs}
    for skip in ("t_startf", "t_stopf", "endrun", "shr_sys_flush", "shr_sys_abort", "get_curr_date",
                 "get_curr_time_string", "restfile_write", "restfile_filename", "hist_htapes_wrapup",
                 "hist_update_hbuf", "elm_device_update", "elm_device_copyin"):
        names.discard(skip)
    types = _Types(fut)
    sub = fut.subroutine_dict[DRV]
    lines = drv.read_text().split("\n")
    out = []
    for st, _ in statements_with_next(body.body, body.end_ln):
        name = callee_name(st)
        if name is None or name not in names:
            continue
        insts = []
        for arg in getattr(st.function, "args", []) or []:
            parts = _designator_parts(arg)
            if parts is None or parts[1]:
                continue
            b = types.base(sub, parts[0])
            if b is not None and b[0] in types.types and b[1] == 0 and b[0] != "bounds_type":
                insts.append((parts[0], b[0]))
        out.append((st.lineno, _end_of(st, lines), insts))
    return out


def plan(fut, drv: Path, config=None) -> Targeted:
    from spel.transforms import acc_sync

    lines = drv.read_text().split("\n")
    before, after, notes = step_boundary(fut)
    sp = acc_sync.plan(fut, [DRV], device_spans={DRV: device_spans(fut, lines)}, boundary={DRV: (before, after)},
                       config=config)
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
    """Remove an earlier application: TAG lines, the update statement (with its continuations)
    right below a TAG line, and the older applications' updates guarded by LEGACY_COND."""
    out: list[str] = []
    i = 0
    after_tag = False  # the previous line was a TAG comment line (it heads one update statement)
    while i < len(lines):
        s = lines[i].lstrip().lower()
        if TAG in lines[i]:
            after_tag = s.startswith("!") and not s.startswith("!$")
            i += 1
            continue
        if s.startswith("!$acc update"):
            j = i
            while lines[j].rstrip().endswith("&") and j + 1 < len(lines) and lines[j + 1].lstrip().startswith("!$acc&"):
                j += 1
            if after_tag or f"if({LEGACY_COND})" in lines[j]:
                after_tag = False
                i = j + 1
                continue
        after_tag = False
        out.append(lines[i])
        i += 1
    return out


def driver_path(srcroot: Path) -> Path:
    return Path(srcroot) / "components" / "elm" / "src" / "main" / "elm_driver.F90"


def edit_set(fut, srcroot: Path, lines: list[str], config=None) -> tuple[Targeted, EditSet]:
    """The plan and the insertions into the driver `lines`, which must be the analyzed file on
    disk without an earlier application (`check_lines`)."""
    from spel.transforms import acc_sync

    drv = driver_path(srcroot)
    check_lines(fut, drv)
    t = plan(fut, drv, config)
    ins = acc_sync.edits(t.plan, {drv: lines}).get(drv, [])
    # acc-sync's own lines (its comment, use statements for callees' globals) carry its TAG;
    # every update block gets a TAG line above it (what `strip` keys on)
    ins = [(ln, order, [x.replace(acc_sync.TAG, TAG) for x in new]) for ln, order, new in ins]
    ins = [(ln, order, ([f"{indent(new[0])}{TAG}"] + new) if new and new[0].lstrip().lower().startswith("!$acc update")
            else new) for ln, order, new in ins]
    from spel.transforms.acc_directives import find_directives
    from spel.transforms.core.query import above_directives as _above_directives

    directives = find_directives(lines, 0, len(lines) - 1)
    for rs in t.plan.routines:
        for seg in rs.segments:
            if any(p.split("%")[0] in FILTER_INSTANCES for p in seg.skipped):
                ind = indent(lines[seg.first])
                ins.append((_above_directives(seg.first, directives), 0,
                            [f"{ind}call elm_device_filters_update(to_device=.false.)  {TAG}"]))
                if not seg.header_only:
                    ins.append((seg.after, -2, [f"{ind}call elm_device_filters_update(to_device=.true.)  {TAG}"]))
    # restFile_write is in a module the analysis excludes (no call event): its call from the parse tree
    from spel.scripts.fortran_parser.module_spec import callee_name, parse_modules, statements_with_next

    mod = next(m for m in parse_modules(drv) if m.name.lower() == "elm_driver")
    body = next(p for p in mod.subprograms if p.name.lower() == "elm_drv")
    pulls = [st.lineno for st, _ in statements_with_next(body.body, body.end_ln) if callee_name(st) in PULL_ALL_BEFORE]
    if not pulls:
        raise DeviceDataError(f"{drv.name}: no call of {sorted(PULL_ALL_BEFORE)} in elm_drv")
    for ln in pulls:
        ind = indent(lines[ln])
        ins.append((ln, 5, [f"{ind}call elm_device_pull_all()  {TAG}: restart writes read the whole state"]))
    # calls the analysis cannot see into get no copies: host-only code, so their configurations
    # must abort in elm_drv's GPU guard (or the modules get ported)
    sub = fut.subroutine_dict[DRV]
    for first, last, insts in excluded_calls(fut, drv):
        if config is not None and not config.span_active(sub, first, last):
            continue  # in a branch that cannot run in the configuration
        t.report.append(f"excluded call at line {first + 1}: host-only code, not synced "
                        f"(instances {', '.join(i for i, _ in insts) or 'none'}): the GPU guard must abort its configuration")
    use_ln = next((i for i, x in enumerate(lines) if x.lstrip().lower().startswith(f"use {MODULE_NAME.lower()}")), None)
    if use_ln is None:
        raise DeviceDataError(f"{drv.name}: no `use {MODULE_NAME}`")
    ins.append((use_ln + 1, 0, [f"  use {MODULE_NAME}, only : {', '.join(USE_NAMES)}  {TAG}"]))
    return t, EditSet(drv).extend(stacked((ln, new, order) for ln, order, new in ins))


def apply(fut, srcroot: Path, dry_run: bool = False) -> list[str]:
    drv = driver_path(srcroot)
    text = drv.read_text().split("\n")
    stripped = strip(text)
    if stripped != text:
        if not dry_run:
            drv.write_text("\n".join(stripped))
        else:
            raise DeviceDataError(f"{drv.name} carries an earlier application: strip it (apply without --dry-run)")
    lines = drv.read_text().split("\n")
    t, es = edit_set(fut, srcroot, lines)
    if not dry_run:
        drv.write_text("\n".join(es.apply(lines)))
    return t.report + [f"{drv.name}: {len(es)} insertions{' (dry run)' if dry_run else ''}"]


@dataclass
class Options:
    pass


@transform
class DeviceTargeted(Transform):
    """
    Matches: elm_drv's host segments (acc-sync rooted at elm_drv), the clump filters, the
             restart write, calls into modules the analysis excludes.
    Changes: acc-sync updates move the data each host segment of elm_drv uses (the state stays
             on the device: no transitions); calls into excluded modules are reported, not
             synced (their configurations must abort in elm_drv's GPU guard).
    Refuses: a driver that carries an earlier application (`targeted.strip` it and re-analyze).
    Design: device/targeted.py.
    """

    name = "device-targeted"
    summary = "targeted host/device updates for the host code of elm_drv (no transitions)"
    Options = Options

    def build(self, ctx, opts) -> TPlan:
        out = TPlan(self.name)
        drv = driver_path(ctx.srcroot)
        lines = ctx.lines(drv)
        if strip(lines) != lines:
            return out.add(Refusal(drv.name, "carries an earlier application: strip it, then `spel analyze` again"))
        try:
            t, es = edit_set(ctx.fut, ctx.srcroot, lines, ctx.config)
        except DeviceDataError as err:
            return out.add(Refusal(drv.name, str(err)))
        out.notes += t.report + [f"{drv.name}: {len(es)} insertions"]
        return out.add(Change(drv, es.edits, where="elm_driver::elm_drv"))
