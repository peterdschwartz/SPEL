"""
Remove OpenACC directives (`spel transform acc-strip`).

Two uses:

- Re-planning acc-sync on code that changed: `--sync-only` removes what acc-sync wrote (its tag
  comment lines and `!$acc update ... if_present` statements) in the given routines, so a new
  acc-sync run plans on clean host code (replaces spelval/tools/strip_sync.py).
- Separating generated from hand-written directives when a GPU branch is split into PRs: stripping
  the compute and data directives of a ported file gives its code without the generated layer; a
  rerun of the pipeline on that text, diffed against the ported file, leaves only the directives
  written by hand. `routine` and `declare` are kept by default: upstream ELM carries several hundred
  of them, and they are not what acc-apply/acc-locals/acc-sync generate in bulk.

Directives are found with `acc_directives.find_directives` (continuation lines and the comment
lines between them belong to the directive). A full-line `! spel acc-...` tag comment right before
a removed directive goes with it. Nothing else changes: code that passes such as section-loops or
collapse-bounds rewrote stays (it is tagged on its lines).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from spel.transforms.acc_directives import DirectiveError, find_directives
from spel.transforms.core.edits import delete
from spel.transforms.core.registry import transform
from spel.transforms.core.transform import Change, Plan, Refusal, Transform

KEEP_DEFAULT = ["routine", "declare"]
TAG_RE = re.compile(r"^\s*!\s*spel acc-", re.IGNORECASE)


def removable(d, keep: set[str], only: set[str], sync_only: bool) -> bool:
    """Whether directive `d` is removed under the options."""
    if sync_only:
        return d.construct == "update" and d.clause("if_present") is not None
    base = d.construct[4:] if d.construct.startswith("end ") else d.construct
    if only:
        return d.construct in only or base in only
    return d.construct not in keep and base not in keep


def strip_ranges(lines: list[str], lo: int = 0, hi: int | None = None, keep=KEEP_DEFAULT, only=(),
                 sync_only: bool = False) -> list[tuple[int, int]]:
    """(first, count) of the line ranges to delete among lines lo..hi (0-based, inclusive)."""
    hi = len(lines) - 1 if hi is None else hi
    keep_s, only_s = {k.lower() for k in keep}, {k.lower() for k in only}
    out: list[tuple[int, int]] = []
    for d in find_directives(lines, lo, hi):
        if not removable(d, keep_s, only_s, sync_only):
            continue
        start = d.start_ln
        # tag comment lines directly above the directive (acc-sync writes one per block)
        while start - 1 >= lo and TAG_RE.match(lines[start - 1]) and not (out and out[-1][0] + out[-1][1] > start - 1):
            start -= 1
        out.append((start, d.end_ln - start + 1))
    return out


def stripped(lines: list[str], **kw) -> list[str]:
    """`lines` without the removed directives (for tests and scripts)."""
    drop = {i for s, n in strip_ranges(lines, **kw) for i in range(s, s + n)}
    return [l for i, l in enumerate(lines) if i not in drop]


@dataclass
class Options:
    files: list[str] = field(default_factory=list)  # source files (absolute, or relative to the srcroot)
    routines: list[str] = field(default_factory=list)  # only inside these routines (module::name)
    keep: list[str] = field(default_factory=lambda: list(KEEP_DEFAULT))  # constructs kept
    only: list[str] = field(default_factory=list)  # strip only these constructs (overrides keep)
    sync_only: bool = False  # only acc-sync's `update ... if_present` lines and tags


@transform
class AccStrip(Transform):
    """
    Matches: the OpenACC directives of the given files (`--files`) or routines (`--routines`).
    Changes: deletes them with their continuation lines and the `! spel acc-...` tag comments
             right above them. Default: every construct except `routine` and `declare` (`--keep`);
             `--only update` strips one kind; `--sync-only` strips just acc-sync's
             `update ... if_present` lines (to re-plan acc-sync on changed code).
    Refuses: nothing; a file whose directives do not parse is reported and left alone.
    """

    name = "acc-strip"
    summary = "remove OpenACC directives (all compute/data ones, or only acc-sync's updates)"
    Options = Options

    def build(self, ctx, opts) -> Plan:
        out = Plan(self.name)
        if not opts.files and not opts.routines:
            return out.add(Refusal("acc-strip", "give --files or --routines"))
        spans: dict[Path, list[tuple[int, int, str]]] = {}
        for f in opts.files:
            p = Path(f)
            p = p if p.is_absolute() else ctx.srcroot / p
            spans.setdefault(p, []).append((0, len(ctx.lines(p)) - 1, p.name))
        for sub in ctx.routines(opts.routines) if opts.routines else []:
            spans.setdefault(ctx.path(sub), []).append((sub.startline, sub.endline, sub.id))
        for path, ranges in sorted(spans.items()):
            lines = ctx.lines(path)
            edits, seen = [], set()
            for lo, hi, where in ranges:
                try:
                    rr = strip_ranges(lines, lo, hi, opts.keep, opts.only, opts.sync_only)
                except DirectiveError as err:
                    out.add(Refusal(where, f"directive does not parse: {err}"))
                    continue
                for s, n in rr:
                    if s not in seen:
                        seen.add(s)
                        edits.append(delete(s, n, why="acc-strip"))
            if edits:
                out.add(Change(path, edits, where=path.name, why=f"{len(edits)} directives"))
        return out
