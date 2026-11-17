"""Saved failing examples (`FailureStore`): written in the capture format the case reads, with
`example.json` (and the drawn domain, the forcing, a relation's follow-up), and read back for
`replay`."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from spel.proptest import strategies as strat
from spel.proptest.domain import CONSTANTS_FILE, INPUTS_FILE, DomainSpec, SynthDomain, build_domain
from spel.proptest.errors import SpecError
from spel.proptest.inventory import Inventory
from spel.proptest.paths import slug
from spel.proptest.runner.drawing import Example
from spel.proptest.runner.elements import ElementSpace
from spel.proptest.runner.reporting import Failure
from spel.proptest.spec import Spec

FAILURES_DIR = "failures"
EXAMPLE_FILE = "example.json"
DOMAIN_FILE = "domain.toml"  # the gridcells of a failure on a drawn domain
FORCING_FILE = "forcing.npz"  # a multi-step example's forcing values (step axis first)


class FailureStore:
    """Failures of one session under `<data>/failures/`."""

    def __init__(self, space: ElementSpace, spec: Spec, inv: Inventory, domain: SynthDomain, domain_spec: DomainSpec,
                 case_dir: Path, data: Path, options: dict[str, Any]):
        self.space, self.spec, self.inv, self.domain, self.domain_spec = space, spec, inv, domain, domain_spec
        self.case_dir, self.data, self.options = case_dir, data, options
        self._capture_dir: Optional[Path] = None

    def capture_dir(self) -> Path:
        """A directory with the domain in capture format (both files): the domain's own, or for
        `init = "direct"`, which writes no inputs file, the same domain written once as files
        (saved failures must stay readable by the case and by `replay`)."""
        if (self.domain.directory / INPUTS_FILE).is_file():
            return self.domain.directory
        if self._capture_dir is None:
            spec = dataclasses.replace(self.domain_spec, init="files")
            self._capture_dir = build_domain(
                self.inv,
                spec,
                self.domain.directory.with_name(self.domain.directory.name + "-files"),
            ).directory
        return self._capture_dir

    def write_example(self, out: Path, example: Example) -> None:
        import netCDF4

        out.mkdir(parents=True, exist_ok=True)
        for name in (INPUTS_FILE, CONSTANTS_FILE):
            shutil.copyfile(self.capture_dir() / name, out / name)
        with netCDF4.Dataset(out / INPUTS_FILE, "a") as ds:
            for name, arr in example.inputs.items():
                nc = self.inv.vars[name].nc_name
                if (
                    nc not in ds.variables
                ):  # a dormant input the domain of this configuration does not hold
                    continue
                a = np.asarray(arr)
                ds.variables[nc][...] = (a.astype(np.int32) if a.dtype == bool else a).T

    def read_example(self, directory: Path, names: list[str]) -> dict[str, np.ndarray]:
        import netCDF4

        out: dict[str, np.ndarray] = {}
        with netCDF4.Dataset(directory / INPUTS_FILE) as ds:
            ds.set_auto_mask(False)
            for name in names:
                var = self.inv.vars.get(name)
                if var is None or name not in self.domain.extents:
                    raise SpecError(
                        f"{directory}: {name} is not a variable of {self.case_dir.name} any more"
                    )
                if var.nc_name not in ds.variables:
                    continue
                a = np.asarray(ds.variables[var.nc_name][...]).T
                if a.size != int(np.prod(self.space.shape(name), dtype=int)):
                    raise SpecError(
                        f"{directory}: {name} has shape {a.shape}, the spec's domain gives {self.space.shape(name)}"
                    )
                a = a.reshape(self.space.shape(name))
                out[name] = a != 0 if var.type == "logical" else a
        return out

    def save(self, f: Failure, drawn: str, suffix: str, sibling: Callable[[dict], "FailureStore"]) -> Path:
        """Write a failing example to `<data>/failures/<scenario>__<check>/`: the domain's two files with the
        example's inputs written in (the format of a capture, readable by the case), and
        `example.json`. The follow-up of a relation under other namelist options goes to
        `follow/` the same way. drawn: the drawn domain's description ("" = the spec's own domain), suffix: of the
        directory name, sibling: the store of this case under further namelist options."""
        out = (
            self.data
            / FAILURES_DIR
            / (slug(f"{f.scenario}__{f.check}") + suffix)
        )
        self.write_example(out, f.example)
        if drawn:  # replay needs this domain, not the spec's sample
            from spel.proptest import tomlw

            (out / DOMAIN_FILE).write_text(
                tomlw.dumps(
                    {"gridcell": [g.to_table() for g in self.domain_spec.gridcells]},
                    {(): f"the drawn domain of this failure: {drawn}"},
                )
            )
        meta = {
            "spec": str(self.spec.path),
            "case": self.case_dir.name,
            "scenario": f.scenario,
            "check": f.check,
            "element": f.element,
            "message": f.message,
            "seed": f.example.seed,
            "signature": f.signature.to_json() if f.signature is not None else None,
            "modes": {k: strat.MODES[v] for k, v in sorted(f.example.modes.items())},
            "extra": f.example.extra,
            "inputs": sorted(f.example.inputs),
            "namelist": self.options,
        }
        if f.example.forcing:
            np.savez(
                out / FORCING_FILE,
                **{k.replace("%", "__"): v for k, v in f.example.forcing.items()},
            )
            meta["forcing"] = sorted(f.example.forcing)
        if f.example.follow is not None:
            sibling(f.example.extra.get("namelist", {})).write_example(
                out / "follow", f.example.follow
            )
            meta["follow_inputs"] = sorted(f.example.follow.inputs)
        (out / EXAMPLE_FILE).write_text(json.dumps(meta, indent=1, default=str))
        return out

    def load(self, directory: Path, sibling: Callable[[dict], "FailureStore"]) -> tuple[dict, Example]:
        """A saved failure: its `example.json` and the example with the saved inputs."""
        directory = Path(directory)
        try:
            meta = json.loads((directory / EXAMPLE_FILE).read_text())
        except OSError as exc:
            raise SpecError(f"{directory / EXAMPLE_FILE}: {exc.strerror}") from None
        ex = Example(
            inputs=self.read_example(directory, meta["inputs"]),
            seed=int(meta.get("seed", 0)),
            extra=dict(meta.get("extra", {})),
        )
        if meta.get("forcing"):
            with np.load(directory / FORCING_FILE) as z:
                ex.forcing = {
                    k: np.asarray(z[k.replace("%", "__")]) for k in meta["forcing"]
                }
        if meta.get("follow_inputs"):
            store = sibling(ex.extra.get("namelist", {}))
            ex.follow = Example(
                inputs=store.read_example(directory / "follow", meta["follow_inputs"]),
                seed=ex.seed,
            )
        return meta, ex
