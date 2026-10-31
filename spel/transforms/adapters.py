"""
The large passes (compact layout, nutrient types, module split, device data) as registered
transforms.

These passes still build their output with their own machinery (whole generated modules, many
coordinated files, manifests). An adapter runs the pass's `spel` command on the context's
analysis with every file write captured (`core.capture.VirtualFS`), and returns what it would
have written as the whole-file outputs of a Plan. They then work like any other transform:
`spel transform NAME --dry-run --diff`, `-o DIR`, `--verify`, multi-step `--spec` files, the
manifest. Their options are the command's flags.

To port one onto core edits later, replace its adapter with a Transform whose `build` returns
the edits; its name and options stay.
"""

from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from spel.transforms.core.capture import VirtualFS
from spel.transforms.core.registry import transform
from spel.transforms.core.transform import Plan, Refusal, Transform


class CliAdapter(Transform):
    """A legacy `spel` command run with its writes captured. Subclasses give `command` (the
    argv after `spel`) and an Options dataclass; each option becomes the flag of its
    `metadata["flag"]` (default `--name-with-dashes`)."""

    command: list[str] = []

    def argv(self, opts) -> list[str]:
        out = list(self.command)
        for f in fields(opts):
            value = getattr(opts, f.name)
            flag = f.metadata.get("flag", "--" + f.name.replace("_", "-"))
            if value is None or value is False or value == [] or value == "":
                continue
            if value is True:
                out.append(flag)
            elif isinstance(value, list):
                out += [flag, *map(str, value)]
            else:
                out += [flag, str(value)]
        return out

    def run(self, ctx, opts) -> None:
        """The legacy pass; what it prints becomes the plan's notes, what it writes its outputs."""
        import spel.scripts.analysis_cache as ac
        from spel import cli

        saved_argv, loader = sys.argv, ac.load_analysis
        sys.argv = ["spel", *self.argv(opts)]
        ac.load_analysis = lambda *a, **k: ctx.fut
        try:
            cli.main()
        finally:
            sys.argv, ac.load_analysis = saved_argv, loader

    def build(self, ctx, opts) -> Plan:
        plan = Plan(self.name)
        vfs = VirtualFS()
        stdout = io.StringIO()
        try:
            with vfs.active(), contextlib.redirect_stdout(stdout):
                self.run(ctx, opts)
        except SystemExit as err:
            if err.code not in (None, 0):
                plan.refusals.append(Refusal(self.name, str(err.code)))
        except Exception as err:  # noqa: BLE001  (the legacy passes' own errors)
            plan.refusals.append(Refusal(self.name, f"{type(err).__name__}: {err}"))
        plan.notes += [l for l in stdout.getvalue().splitlines() if l.strip()]
        if not plan.refusals:
            for path, text in sorted(vfs.files.items()):
                plan.new_files[Path(path)] = text
        return plan


@dataclass
class CompactOptions:
    type: list[str] = field(default_factory=list, metadata={
        "help": "the derived types to rewrite (default: the patch/column C/N/P flux types, or their family base types on nutrient-transformed sources)"})
    roles: bool = field(default=False, metadata={"help": "also the role holders of the default types (nutrient-transformed sources)"})
    loops_only: bool = field(default=False, metadata={"help": "only the SetValues loops, no compact layout"})
    setvalues_loops: bool = field(default=False, metadata={"help": "also turn the flux types' blocked SetValues loops into do concurrent"})
    exclude_fields: str = field(default="", metadata={"help": "comma-separated components to drop from every selection"})
    fields: str = field(default="", metadata={"help": "comma-separated components to keep (only these)"})
    no_map_sites: bool = field(default=False, metadata={"help": "only filter-gathered sites"})


@transform
class CompactRewrite(CliAdapter):
    """
    Matches: the per-step zeroed fields of the selected flux types that every access site
             reaches through a soil filter (`spel compact inventory` explains each verdict).
    Changes: those fields move to per-group compact buffers; access sites, history, restart and
             the SetValues loops are rewritten; CompactBindMod and the manifest are generated.
    Adapter: runs `spel compact rewrite`. Design: compact/DESIGN.md.
    """

    name = "compact-rewrite"
    summary = "compact (filter-packed) storage for the per-step zeroed flux fields"
    Options = CompactOptions
    command = ["compact", "rewrite"]


@dataclass
class DeviceOptions:
    pass


@transform
class DeviceData(CliAdapter):
    """
    Changes: generates ElmDeviceDataMod and the DeepCopy*Mod group modules (device init,
             first-step deep copy, whole-state updates) and edits the driver to call them.
    Adapter: runs `spel device-data`.
    """

    name = "device-data"
    summary = "generate the device-data modules for a full OpenACC run"
    Options = DeviceOptions
    command = ["device-data"]


@dataclass
class NutrientOptions:
    family: list[str] = field(default_factory=list, metadata={"help": "only these families (e.g. vegetation_nutrient_flux)"})


@dataclass
class MergeOptions(NutrientOptions):
    procedure: list[str] = field(default_factory=list, metadata={
        "help": "the bindings to merge (default: setvalues zerodwt clean)"})


def _nutrient(action: str, summary_: str, doc: str, options=NutrientOptions):
    cls = type(f"Nutrient{action.title()}", (CliAdapter,), {
        "__doc__": doc + f"\nAdapter: runs `spel nutrient {action}`.",
        "name": f"nutrient-{action}",
        "summary": summary_,
        "Options": options,
        "command": ["nutrient", action],
        "__module__": __name__,
    })
    return transform(cls)


NutrientExtract = _nutrient(
    "extract", "move the fields the C/N/P types share to base types they extend",
    "Changes: each family's shared components move to a base type; the C/N/P types extend it.")
NutrientSpecies = _nutrient(
    "species", "give each core its element (core%species, set in Init); after extract",
    "Changes: adds `species` to the family cores and sets it in each Init.")
NutrientMerge = _nutrient(
    "merge", "move the procedures' shared statements to base-type procedures; after extract",
    "Changes: statements the C/N/P procedures share move to a base-type procedure they call.",
    MergeOptions)
@dataclass
class GroupsOptions(NutrientOptions):
    groups: Optional[Path] = field(default=None, metadata={
        "help": "TOML file of role tables (family -> base names) replacing the built-in GROUPS"})


NutrientGroups = _nutrient(
    "groups", "move N/P-only fields to role components (mineral, uptake, retrans)",
    "Changes: N/P-only components move to role holders (nutrient/groups.py, or the `groups` TOML file).",
    GroupsOptions)
NutrientHistory = _nutrient(
    "history", "one history registration per core field for every species",
    "Changes: the per-species hist_addfld calls of core fields become one loop over species.")
NutrientRestart = _nutrient(
    "restart", "one restartvar per core field for every species",
    "Changes: the per-species restartvar calls of core fields become one loop over species.")
