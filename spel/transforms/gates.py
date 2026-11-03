"""
Gates for a GPU-ported tree (`spel gates`): checks that must pass before building and validating.

They replace the scripts in spelval/tools (sync_drift.py, stale_updates.py, collapse_inquiry_scan.py,
missing_declares.py), and use the parser and the analysis instead of regexes. Most are "this
transform has nothing left to do" (planned in memory, nothing written):

- collapse-bounds: no `collapse(n)` nest takes its bounds from lbound/ubound/size (nvfortran 25.9
  runs such kernels empty on the GPU).
- acc-declares: every module variable device code uses has an `acc declare`.
- acc-routines: `!$acc routine seq` exactly on the routines kernels reach.
- sync-drift: every `update self/device` acc-sync plans for the host segments of the call tree is
  in the sources (syncs go stale when a later pass changes what is device-resident).
- stale-updates: no push of never-set host data after a kernel wrote it (acc_sync.stale_updates).

With a run configuration (`--config`, core/config.py) the gates look only at the code that runs
in it; this replaces spelval/tools/scope.py's hand-kept exclusions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

GATES = ("collapse-bounds", "acc-declares", "acc-routines", "sync-drift", "stale-updates")


@dataclass
class GateResult:
    gate: str
    findings: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.findings and not self.error

    def __str__(self) -> str:
        head = f"{'PASS' if self.ok else 'FAIL'} {self.gate}"
        if self.error:
            return f"{head}: {self.error}"
        return "\n".join([f"{head}" + (f": {len(self.findings)} findings" if self.findings else "")]
                         + [f"    {f}" for f in self.findings[:40]]
                         + ([f"    ... {len(self.findings) - 40} more"] if len(self.findings) > 40 else []))


def _transform_gate(name: str, ctx) -> GateResult:
    """A transform that has nothing left to do passes; its planned changes are the findings."""
    from spel.transforms.core import driver
    from spel.transforms.core.registry import get

    t = get(name)()
    res = driver.run(t, ctx, t.Options(), sink=driver.Memory(), verify=False)
    return GateResult(name, [f"{c.where}: {c.why}" for c in res.plan.changes])


def _sync_drift(ctx, roots: list[str]) -> GateResult:
    from spel.transforms import acc_sync
    from spel.transforms.acc_directives import find_directives, split_top_level

    fut = ctx.fut
    pl = acc_sync.plan(fut, roots, config=ctx.config)
    scope = set(ctx.call_tree(roots)) if ctx.config is not None else None
    out = GateResult("sync-drift")

    def norm(x):
        return re.sub(r"\(.*?\)", "", x).strip().lower()

    for rs in pl.routines:
        if rs.routine.endswith("::elm_drv") or (scope is not None and rs.routine not in scope):
            continue
        sub = fut.subroutine_dict[rs.routine]
        lines = ctx.lines(Path(rs.path))
        have_dev, have_self = set(), set()
        for d in find_directives(lines, sub.startline, min(sub.endline, len(lines) - 1)):
            if d.construct == "update":
                for c in d.clauses:
                    tgt = have_dev if c.name == "device" else have_self if c.name in ("self", "host") else None
                    if tgt is not None and c.args:
                        tgt |= {norm(a) for a in split_top_level(c.args) if a.strip()}
        miss_dev, miss_self = set(), set()
        for seg in rs.segments:
            if not seg.header_only:
                miss_dev |= {norm(w) for w in seg.writes} - have_dev
            miss_self |= {norm(w) for w in seg.stale} - have_self
        if miss_dev or miss_self:
            out.findings.append(f"{rs.routine}: missing update device {sorted(miss_dev)[:8]}, "
                                f"update self {sorted(miss_self)[:8]}")
    return out


def _stale_updates(ctx, roots: list[str]) -> GateResult:
    from spel.transforms.acc_sync import stale_updates

    out = GateResult("stale-updates")
    subs = ctx.call_tree(roots).values() if ctx.config is not None else ctx.fut.subroutine_dict.values()
    for sub in sorted(subs, key=lambda s: s.id):
        if sub.record is None:
            continue
        for st in stale_updates(sub, ctx.lines(ctx.path(sub))):
            if st.kind == "push":
                out.findings.append(str(st))
    return out


def run(ctx, roots: Iterable[str] = ("elm_driver::elm_drv",), only: Optional[Iterable[str]] = None) -> list[GateResult]:
    roots = list(roots)
    out = []
    for gate in only or GATES:
        try:
            if gate == "sync-drift":
                out.append(_sync_drift(ctx, roots))
            elif gate == "stale-updates":
                out.append(_stale_updates(ctx, roots))
            else:
                out.append(_transform_gate(gate, ctx))
        except Exception as err:  # noqa: BLE001 - a gate that cannot run fails, with its reason
            out.append(GateResult(gate, error=f"{type(err).__name__}: {err}"))
    return out
