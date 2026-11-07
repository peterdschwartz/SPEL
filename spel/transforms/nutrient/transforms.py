"""
The nutrient passes as registered transforms (`spel transform nutrient-*`). Each runs its
pass's builder, which edits the sources in memory, and returns the result as edits:
merge's own `FileEdits` map onto core edits one to one; the others post-process whole texts
(renames over every file, helpers and imports added afterwards), so their files are taken as
line diffs (`core.edits.diff_edits`). extract and groups also record what they moved in the
`.spel-nutrient.json` manifest the later passes read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spel.transforms.core.edits import diff_edits, insert, replace
from spel.transforms.core.registry import transform
from spel.transforms.core.transform import Change, Plan, Refusal, Transform
from spel.transforms.nutrient.merge import DEFAULT_BINDINGS

SPEL_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class NutrientOptions:
    family: list[str] = field(default_factory=list, metadata={"help": "only these families (e.g. vegetation_nutrient_flux)"})


@dataclass
class MergeOptions(NutrientOptions):
    procedure: list[str] = field(default_factory=list, metadata={
        "help": f"the bindings to merge (default: {' '.join(DEFAULT_BINDINGS)})"})


@dataclass
class GroupsOptions(NutrientOptions):
    groups: Optional[Path] = field(default=None, metadata={
        "help": "TOML file of role tables (family -> base names) replacing the built-in GROUPS"})


def _elm_src(ctx) -> Path:
    return ctx.srcroot / "components" / "elm" / "src"


def _families(ctx, opts):
    from spel.transforms.nutrient.common import find_families

    families = find_families(ctx.fut.type_dict)
    return [f for f in families if f.name in opts.family] if opts.family else families


def _whole_files(out: Plan, ctx, files: dict[Path, list[str]], why: str) -> Plan:
    """The changed files of a pass that rebuilt them whole, as line-diff edits."""
    for path, lines in files.items():
        old = ctx.lines(path)
        if lines != old:
            out.add(Change(Path(path), diff_edits(old, lines, why=why), where=Path(path).name))
    return out


def core_edits(fe, lines: list[str], why: str = "") -> list:
    """merge's FileEdits as core edits: the same text as `fe.apply(lines)`."""
    # FileEdits.apply puts the insertions at a line before a replacement starting there, and
    # drops insertions at the end of the file
    edits = [insert(ln, new, order=-1, why=why) for ln, new in fe.insert.items() if ln < len(lines)]
    return edits + [replace(s, new, n=e - s + 1, expect=lines[s:e + 1], why=why) for s, (e, new) in fe.replace.items()]


class NutrientTransform(Transform):
    Options = NutrientOptions

    def build(self, ctx, opts) -> Plan:
        from spel.transforms.nutrient.extract import ExtractError

        out = Plan(self.name)
        try:
            return self.plan(ctx, opts, _families(ctx, opts), out)
        except ExtractError as err:
            return Plan(self.name).add(Refusal(self.name, str(err)))

    def plan(self, ctx, opts, families, out: Plan) -> Plan:
        raise NotImplementedError


class _Extracting(NutrientTransform):
    key = "families"

    def edits(self, families, opts):
        raise NotImplementedError

    def plan(self, ctx, opts, families, out):
        from spel.scripts.analysis_cache import git_state
        from spel.transforms.nutrient.extract import MANIFEST, Extractor, manifest

        edits, skipped = self.edits(families, opts)
        ex = Extractor(edits, ctx.fut.type_dict, _elm_src(ctx), skipped)
        ex.build()
        out.notes += ex.report().split("\n")
        _whole_files(out, ctx, ex.files, self.name)
        mpath = _elm_src(ctx) / "data_types" / MANIFEST
        data = json.loads("\n".join(ctx.lines(mpath))) if mpath.exists() else {}
        data.update(manifest(ex, git_state(SPEL_ROOT).get("commit"), git_state(ctx.srcroot), self.key))
        out.new_files[mpath] = json.dumps(data, indent=1) + "\n"
        return out


@transform
class NutrientExtract(_Extracting):
    """
    Matches: the C/N/P type families (`spel nutrient` lists them).
    Changes: each family's shared components move to a base type; the C/N/P types extend it.
    """

    name = "nutrient-extract"
    summary = "move the fields the C/N/P types share to base types they extend"

    def edits(self, families, opts):
        from spel.transforms.nutrient.extract import core_edits

        return core_edits(families)


@transform
class NutrientGroups(_Extracting):
    """
    Changes: N/P-only components move to role holders (nutrient/groups.py, or the `groups` TOML file).
    """

    name = "nutrient-groups"
    summary = "move N/P-only fields to role components (mineral, uptake, retrans)"
    Options = GroupsOptions
    key = "groups"

    def edits(self, families, opts):
        from spel.transforms.nutrient.groups import group_edits, load_groups

        return group_edits(families, load_groups(opts.groups) if opts.groups else None), {}


@transform
class NutrientSpecies(NutrientTransform):
    """
    Changes: adds `species` to the family cores and sets it in each Init.
    """

    name = "nutrient-species"
    summary = "give each core its element (core%species, set in Init); after extract"

    def plan(self, ctx, opts, families, out):
        from spel.transforms.nutrient.species import SpeciesEditor

        ed = SpeciesEditor(families, ctx.fut, _elm_src(ctx))
        ed.build()
        out.notes += ed.notes
        return _whole_files(out, ctx, ed.files, self.name)


@transform
class NutrientMerge(NutrientTransform):
    """
    Changes: statements the C/N/P procedures share move to a base-type procedure they call.
    """

    name = "nutrient-merge"
    summary = "move the procedures' shared statements to base-type procedures; after extract"
    Options = MergeOptions

    def plan(self, ctx, opts, families, out):
        from spel.transforms.nutrient.merge import Merger

        m = Merger(families, ctx.fut, bindings=opts.procedure or DEFAULT_BINDINGS)
        m.build()
        out.notes += [l for l in m.report().split("\n") if l and not l.startswith("not merged: ")]
        out.refusals += [Refusal("nutrient-merge", msg) for msg in m.refused]
        for path, fe in m.edits.items():
            edits = core_edits(fe, m.lines(path), why=self.name)
            if edits:
                out.add(Change(Path(path), edits, where=Path(path).name))
        return out


class _History(NutrientTransform):
    spec_name = "HISTORY"

    def plan(self, ctx, opts, families, out):
        from spel.transforms.nutrient import history

        hm = history.HistoryMerger(families, ctx.fut, getattr(history, self.spec_name))
        hm.build()
        out.notes += [l for l in hm.report().split("\n") if l and not l.startswith("not merged: ")]
        out.refusals += [Refusal(self.name, msg) for msg in hm.refused]
        return _whole_files(out, ctx, hm.apply(), self.name)


@transform
class NutrientHistory(_History):
    """
    Changes: the per-species hist_addfld calls of core fields become one loop over species.
    """

    name = "nutrient-history"
    summary = "one history registration per core field for every species"


@transform
class NutrientRestart(_History):
    """
    Changes: the per-species restartvar calls of core fields become one loop over species.
    """

    name = "nutrient-restart"
    summary = "one restartvar per core field for every species"
    spec_name = "RESTART"
