"""
`!$acc routine seq` where the device needs it, and nowhere else (`spel transform acc-routines`).

A routine needs `!$acc routine` exactly when a kernel calls it, directly or not. Upstream ELM marks
many routines that no kernel reaches (host-level BGC, phenology, state updates, dynamic subgrid);
nvfortran then compiles a device version of each, which fails for routines that call timers,
`endrun`, `p2c` or `acc update`. The GPU branch removed 130 such directives by hand (16f5e04706)
and added the ones the device link missed (8b55a9f8ed). This pass derives both from the analysis:

- device routines = every routine called inside a compute construct (`parallel`/`kernels`/
  `serial` regions and loops, parsed by acc_directives) of any analyzed routine, and everything
  they call (acc_apply.DeviceFacts / _closure). The whole analysis counts, not only the code of a
  configuration: a kernel in a branch that does not run still has to compile.
- prune (default): `!$acc routine` / `!$acc routine seq` (no name argument) is removed from analyzed
  routines that are not device routines.
- add (default): `!$acc routine seq` goes into device routines without one (after the declarations,
  as acc-apply places it).

A kernel in code the analysis does not cover (initialization) can call a pruned routine: the GPU
build then fails at compile or link time (never silently); `--keep` names routines to leave alone.
Kernels with calls the analysis cannot resolve are reported, and their callees are not added.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spel.transforms.acc_directives import DirectiveError, find_directives, next_code_line
from spel.transforms.core.edits import delete, insert
from spel.transforms.core.registry import transform
from spel.transforms.core.transform import Change, Plan, Refusal, Transform

TAG = "! spel acc-routines"


def compute_ranges(sub, lines: list[str]) -> list[tuple[int, int]]:
    """(first, last) lines of the compute constructs in `sub` (0-based, inclusive)."""
    out = []
    ds = find_directives(lines, sub.startline, min(sub.endline, len(lines) - 1))
    loops = sub.record.all_loops() if sub.record is not None else []
    for k, d in enumerate(ds):
        if d.is_compute_loop:
            ln = next_code_line(lines, d.end_ln)
            lp = next((lp for lp in loops if lp.start_ln == ln), None)
            if lp is not None:
                out.append((d.start_ln, lp.end_ln))
            else:
                out.append((d.start_ln, sub.endline))  # unknown extent: be generous
        elif d.is_compute_region:
            end = next((e.start_ln for e in ds[k + 1:] if e.construct == f"end {d.construct}"), sub.endline)
            out.append((d.start_ln, end))
    return out


def device_routines(fut) -> tuple[set[str], list[str]]:
    """Ids of the routines a kernel reaches, and the calls on the way that do not resolve."""
    from spel.transforms.acc_apply import DeviceFacts, _closure

    facts = DeviceFacts(fut)
    direct, missing = set(), []
    for sub in fut.subroutine_dict.values():
        if sub.record is None:
            continue
        lines = facts.lines(sub)
        try:
            ranges = compute_ranges(sub, lines)
        except DirectiveError as err:
            missing.append(f"{sub.id}: directives do not parse ({err})")
            continue
        for lo, hi in ranges:
            ids, miss = facts.callees_in(sub, lo, hi)
            direct |= ids
            missing += [f"{m} (kernel in {sub.id})" for m in miss]
    reached, deep = _closure(facts, direct)
    return reached, missing + deep


@dataclass
class Options:
    prune: bool = True  # remove `routine` directives from routines no kernel reaches
    add: bool = True  # add `routine seq` to device routines without one
    keep: list[str] = field(default_factory=list)  # routine ids (module::name) left alone


@transform
class AccRoutines(Transform):
    """
    Matches: every analyzed routine with or without an `!$acc routine` directive.
    Changes: removes `!$acc routine [seq]` from routines no kernel reaches; adds `!$acc routine seq`
             to routines a kernel reaches (directly or not) that lack one.
    Refuses: `routine(name)` directives (interfaces) are not touched; routines in `--keep`.
    """

    name = "acc-routines"
    summary = "`!$acc routine seq` exactly on the routines kernels reach (prune the rest, add missing)"
    Options = Options

    def build(self, ctx, opts) -> Plan:
        from spel.transforms.acc_apply import _routine_seq_edit

        out = Plan(self.name)
        device, missing = device_routines(ctx.fut)
        out.notes.append(f"{len(device)} routines reached from kernels")
        out.notes += [f"unresolved: {m}" for m in sorted(set(missing))]
        keep = {k.lower() for k in opts.keep}
        for sub in sorted((s for s in ctx.fut.subroutine_dict.values() if s.record is not None), key=lambda s: s.id):
            if sub.id.lower() in keep:
                continue
            path = ctx.path(sub)
            lines = ctx.lines(path)
            try:
                ds = [d for d in find_directives(lines, sub.startline, min(sub.endline, len(lines) - 1))
                      if d.construct == "routine"]
            except DirectiveError as err:
                out.add(Refusal(sub.id, f"directives do not parse: {err}"))
                continue
            if sub.id in device:
                if opts.add and not ds:
                    try:
                        ins = _routine_seq_edit(sub, lines)
                    except (Exception, SystemExit) as err:  # noqa: BLE001 - the parser exits on errors
                        out.add(Refusal(sub.id, f"no place for `routine seq`: {err}"))
                        continue
                    if ins is not None:
                        new = [l.replace("! spel acc-apply", TAG) for l in ins.lines]
                        out.add(Change(path, [insert(ins.ln, new, order=ins.order)], where=sub.id,
                                       why="called from a kernel"))
                continue
            if opts.prune:
                drop = [d for d in ds if not d.construct_args]
                if drop:
                    out.add(Change(path, [delete(d.start_ln, d.end_ln - d.start_ln + 1, why="no kernel reaches it")
                                          for d in drop], where=sub.id, why="no kernel reaches it"))
        return out


@dataclass
class DeclareOptions:
    pass


@transform
class AccDeclares(Transform):
    """
    Matches: module variables (not parameters, not derived-type instances: the device-data layer
             copies those) read or written inside compute constructs or anywhere in a routine a kernel
             reaches (acc-routines' device routines), without an `acc declare` in their module.
    Changes: `!$acc declare create(...)` in each such module (acc_apply._declare_edit, after the
             declarations); the device-data layer then copies their values.
    Refuses: modules whose specification part does not parse (module_spec).
    """

    name = "acc-declares"
    summary = "`!$acc declare create` for module variables device code uses"
    Options = DeclareOptions

    def build(self, ctx, opts) -> Plan:
        from collections import defaultdict
        from pathlib import Path

        from spel.scripts.fortran_parser.module_spec import ModuleSpecError, module_spec
        from spel.scripts.fortran_parser.symbols import Origin
        from spel.transforms.acc_apply import _declare_edit, _path

        out = Plan(self.name)
        fut = ctx.fut
        device, _ = device_routines(fut)
        specs: dict[Path, object] = {}

        def spec_of(path: Path):
            if path not in specs:
                try:
                    specs[path] = module_spec(path)
                except ModuleSpecError as err:
                    specs[path] = None
                    out.add(Refusal(path.name, f"specification part does not parse: {err}"))
            return specs[path]

        wanted: dict[Path, set[str]] = defaultdict(set)  # module file -> names
        for sub in fut.subroutine_dict.values():
            if sub.record is None:
                continue
            lines = ctx.lines(ctx.path(sub))
            # a device routine needs every module variable it uses on the device; a kernel of a
            # host routine only the arrays (scalars in a compute construct are firstprivate: the
            # host value is copied in at launch)
            in_routine = sub.id in device
            if in_routine:
                ranges = [(sub.startline, sub.endline)]
            else:
                try:
                    ranges = compute_ranges(sub, lines)
                except DirectiveError:
                    continue
            if not ranges:
                continue
            agv = {k.lower(): v for k, v in (getattr(sub, "active_global_vars", None) or {}).items()}
            for a in sub.record.accesses(origin=Origin.GLOBAL):
                if not any(lo <= a.ln <= hi for lo, hi in ranges):
                    continue
                name = a.base.lower()
                v = agv.get(name)
                if v is not None:
                    if getattr(v, "parameter", False) or str(v.type).lower() in fut.type_dict:
                        continue  # constants; derived-type instances (the device-data layer copies them)
                    m = getattr(fut, "module_dict", {}).get(str(v.declaration).lower())
                    if m is None:
                        continue
                    path = _path(m)
                else:
                    path = ctx.path(sub)  # no global table (snippets): the routine's own module
                spec = spec_of(path)
                info = spec.variables.get(name) if spec is not None else None
                if info is None or "parameter" in info.attrs:
                    continue
                if not in_routine and info.rank == 0:
                    continue
                wanted[path].add(name)
        for path, names in sorted(wanted.items()):
            spec = spec_of(path)
            mlines = ctx.lines(path)
            # arrays the module puts on the device itself (`enter data copyin/create`) are present already
            entered = {v for d in find_directives(mlines) if d.construct == "enter data"
                       for c in d.clauses for v in c.variables()}
            need = sorted(n for n in names if n not in spec.declared and n not in entered)
            if need:
                ins = _declare_edit(path, set(need), mlines)
                new = [l.replace("! spel acc-apply", TAG) for l in ins.lines]
                out.add(Change(path, [insert(ins.ln, new, order=ins.order)], where=spec.name, why=", ".join(need)))
        return out
