"""
The transform protocol.

A transform says *where* it applies (`select`) and *what it does there* (`plan`); the driver
(`core/driver.py`) does the rest: reading sources, applying the edits together (conflicts are
errors), reporting, diffs, writing through a sink, the manifest.

    @transform
    class CollapseBounds(Transform):
        name = "collapse-bounds"
        summary = "hoist inquiry-function loop bounds out of collapsed nests"

        def select(self, ctx, opts):
            yield from ctx.routines(opts.routines)          # the matches

        def plan(self, ctx, opts, sub):                     # one match -> Change / Refusal / None
            ...
            return Change(path, [insert(ln, text), replace(ln, text)], where=sub.id, why="...")

A transform whose matches interact (a whole call tree planned at once) overrides `build` instead
and returns the Plan itself. Options are a dataclass (`Options`): its fields become CLI flags of
`spel transform <name>` and keys of `--spec` files.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Iterable, Optional, Union

from spel.transforms.core.edits import Edit, EditSet
from spel.transforms.core.source import Sources


@dataclass
class Change:
    """Edits of one file, made for one match."""

    path: Path
    edits: list[Edit]
    where: str = ""
    why: str = ""


@dataclass
class Refusal:
    """A match the transform leaves alone, and why."""

    where: str
    reason: str

    def __str__(self) -> str:
        return f"{self.where}: {self.reason}"


Outcome = Union[Change, Refusal, str, None]


@dataclass
class Plan:
    """Everything a transform would do: changes, refusals, notes, new files."""

    transform: str
    changes: list[Change] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    new_files: dict[Path, str] = field(default_factory=dict)

    def add(self, outcome) -> "Plan":
        """Add a Change, Refusal, note (str), None, or an iterable of them."""
        if outcome is None:
            return self
        if isinstance(outcome, Change):
            self.changes.append(outcome)
        elif isinstance(outcome, Refusal):
            self.refusals.append(outcome)
        elif isinstance(outcome, str):
            self.notes.append(outcome)
        else:
            for o in outcome:
                self.add(o)
        return self

    def __bool__(self) -> bool:
        return bool(self.changes or self.new_files)

    def edit_sets(self) -> dict[Path, EditSet]:
        """The changes grouped per file (conflicting edits raise EditConflict)."""
        out: dict[Path, EditSet] = {}
        for ch in self.changes:
            es = out.setdefault(Path(ch.path), EditSet(Path(ch.path)))
            es.extend(ch.edits)
        return out

    def outputs(self, sources: Sources) -> dict[Path, str]:
        """New text of every file the plan touches or creates."""
        out = {p: "\n".join(es.apply(sources.lines(p))) for p, es in self.edit_sets().items() if es}
        out.update(self.new_files)
        return out

    def diff(self, sources: Sources) -> str:
        parts = [es.diff(sources.lines(p)) for p, es in sorted(self.edit_sets().items())]
        for p, text in sorted(self.new_files.items()):
            old = sources.lines(p) if Path(p).exists() else []
            parts.append(_new_file_diff(p, old, text))
        return "".join(parts)

    def report(self) -> str:
        files = {Path(c.path) for c in self.changes} | set(self.new_files)
        lines = [f"{self.transform}: {len(self.changes)} changes in {len(files)} files, "
                 f"{len(self.refusals)} refused"]
        for c in self.changes:
            lines.append(f"  {c.where or Path(c.path).name}: {len(c.edits)} edits" + (f"  ({c.why})" if c.why else ""))
        for p in sorted(self.new_files):
            lines.append(f"  {'rewritten' if Path(p).exists() else 'new file'} {p}")
        lines += [f"  refused {r}" for r in self.refusals]
        said = {f"{c.where}: {c.why}" for c in self.changes}
        lines += [f"  note: {n}" for n in self.notes if n not in said]
        return "\n".join(lines)

    def explain(self, sources: Sources, where: str, context: int = 2) -> str:
        """Before/after of the changes whose location contains `where`."""
        out = []
        for c in self.changes:
            if where.lower() not in (c.where or "").lower():
                continue
            out.append(f"== {c.where}" + (f": {c.why}" if c.why else ""))
            es = EditSet(Path(c.path)).extend(c.edits)
            out.append(es.diff(sources.lines(c.path), context=context))
        out += [f"refused {r}" for r in self.refusals if where.lower() in r.where.lower()]
        return "\n".join(out) if out else f"nothing at {where}"

    def to_json(self) -> dict:
        return {
            "transform": self.transform,
            "changes": [{"path": str(c.path), "where": c.where, "why": c.why, "edits": len(c.edits)}
                        for c in self.changes],
            "refusals": [{"where": r.where, "reason": r.reason} for r in self.refusals],
            "notes": self.notes,
            "new_files": sorted(map(str, self.new_files)),
        }


def _new_file_diff(path, old: list[str], text: str) -> str:
    import difflib

    return "".join(difflib.unified_diff([l + "\n" for l in old], [l + "\n" for l in text.split("\n")],
                                        f"a/{str(path).lstrip('/')}", f"b/{str(path).lstrip('/')}"))


class Context:
    """What a transform reads: the analysis, the sources (read once), the srcroot."""

    def __init__(self, fut, srcroot: Optional[Path] = None, sources: Optional[Sources] = None, config=None):
        self.fut = fut
        if srcroot is None:
            from spel.scripts.config import E3SM_SRCROOT

            srcroot = Path(E3SM_SRCROOT)
        self.srcroot = Path(srcroot)
        self.sources = sources or Sources()
        # a run configuration (core/config.py): only the code active in it is transformed
        self.config = config

    def path(self, sub) -> Path:
        """Absolute source file of a routine."""
        p = Path(sub.filepath)
        return p if p.is_absolute() else self.srcroot / p

    def lines(self, sub_or_path) -> list[str]:
        p = sub_or_path if isinstance(sub_or_path, (str, Path)) else self.path(sub_or_path)
        return self.sources.lines(p)

    def routine(self, rid: str):
        from spel.transforms.core.query import routine

        return routine(self.fut, rid)

    def routines(self, ids: Optional[Iterable[str]] = None, analyzed: bool = True) -> list:
        """The routines `ids` (default: all analyzed ones), sorted by id. With a configuration,
        the default is the routines the time step runs in it (`ACTIVE_ROOTS`' active tree)."""
        subs = self.fut.subroutine_dict
        if ids is None and self.config is not None:
            keys = sorted(self.config.call_tree(self.fut, ACTIVE_ROOTS))
        else:
            keys = sorted(set(ids)) if ids is not None else sorted(subs)
        return [subs[k] for k in keys if k in subs and (not analyzed or subs[k].record is not None)]

    def call_tree(self, roots: Iterable[str], exclude=None) -> dict:
        """id -> Subroutine of the routines reachable from `roots` (the roots included); with a
        configuration, only through the branches that run in it."""
        if self.config is not None:
            return self.config.call_tree(self.fut, roots, exclude)
        from spel.transforms.core.query import call_tree

        return call_tree(self.fut, roots, exclude)

    def active(self, sub, lo: int, hi: Optional[int] = None) -> bool:
        """Whether lines lo..hi of `sub` can run (always, without a configuration)."""
        if self.config is None:
            return True
        return self.config.span_active(sub, lo, lo if hi is None else hi)

    @cached_property
    def child_loops(self) -> dict[str, list]:
        """
        id -> the routine's loops over the patches of a filtered column (`do p = pfti(c), pftf(c)`,
        `p = pfti(c) + pi - 1`), each with the filter map that replaces it and whether that is exact
        (core/filter_maps.py); with a configuration, only the loops that run in it.
        """
        from spel.transforms.core import filter_maps

        routines = None
        if self.config is not None:
            routines = set(self.config.call_tree(self.fut, (filter_maps.DRIVER_ID,)))
        out: dict[str, list] = {}
        for cl in filter_maps.scan(self.fut, routines=routines):
            sub = self.fut.subroutine_dict[cl.sub_id]
            if self.active(sub, cl.loop.start_ln, cl.loop.end_ln):
                out.setdefault(cl.sub_id, []).append(cl)
        return out

    @cached_property
    def facts(self):
        """Per-routine facts of the device passes (compute/device routine, unsafe lines, trees)."""
        from spel.transforms.acc_apply import DeviceFacts

        return DeviceFacts(self.fut)


# what a time step runs: elm_drv's tree and the lnd2atm averaging lnd_run_mct calls after it
# (the device roots of device/generate.py)
ACTIVE_ROOTS = ("elm_driver::elm_drv", "lnd2atmmod::lnd2atm", "lnd2atmmod::lnd2atm_minimal")


@dataclass
class NoOptions:
    pass


class Transform:
    """Base class of a transform; see the module docstring."""

    name: ClassVar[str] = ""
    summary: ClassVar[str] = ""
    Options: ClassVar[type] = NoOptions

    def select(self, ctx: Context, opts) -> Iterable[Any]:
        return ()

    def plan(self, ctx: Context, opts, match) -> Outcome | Iterable[Outcome]:
        raise NotImplementedError

    def build(self, ctx: Context, opts) -> Plan:
        """The plan: `plan` of every `select`ed match. Override for whole-program transforms."""
        p = Plan(self.name)
        for m in self.select(ctx, opts):
            p.add(self.plan(ctx, opts, m))
        return p

    @classmethod
    def options(cls, **values):
        """An Options instance from keyword values (unknown keys are an error)."""
        return cls.Options(**values)

    @classmethod
    def describe(cls) -> str:
        """The module docstring's summary plus the class docstring (Matches/Changes/Refuses)."""
        import inspect

        doc = inspect.cleandoc(cls.__doc__ or "")
        opts = ""
        if is_dataclass(cls.Options) and fields(cls.Options):
            def default(f):
                return f.default_factory() if f.default is MISSING and f.default_factory is not MISSING else f.default

            opts = "\nOptions:\n" + "\n".join(
                f"  --{f.name.replace('_', '-')}  {f.metadata.get('help', '')} (default {default(f)!r})"
                for f in fields(cls.Options)
            )
        return f"{cls.name}: {cls.summary}\n\n{doc}{opts}"


def options_json(opts) -> dict:
    if is_dataclass(opts):
        return json.loads(json.dumps(asdict(opts), default=str))
    return {}
