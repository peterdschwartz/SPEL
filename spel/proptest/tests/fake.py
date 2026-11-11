"""A synthetic case for tests of the runner: an inventory without Fortran behind it and a
backend whose "routine" is numpy.

    y(p)    = 2 x(p) + state(p)          output
    ysum(p) = sum(prof(p, :))            output
    state(p) = state(p) + 1              inout
    never(p)                             output, never written

and under the namelist option `use_extra` (off by default), as the analysis would record it:

    y(p)     = y(p) + boost(p)           boost: an input only read under use_extra
    extra(p) = 2 boost(p)                extra: an output only written under use_extra
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from spel.proptest.fexpr import FArray
from spel.proptest.inventory import (
    ConstInfo,
    Inventory,
    TracedAccess,
    VarInfo,
    parse_dims,
)
from spel.proptest.worker import FortranAbort
from spel.scripts.fortran_parser.boolen_expression import Expectation

# the gate of the code under `if (use_extra)`
USE_EXTRA = Expectation("use_extra", "True")

NLEV = 4


def var(name: str, bounds: str, kind: str, ftype="real", status="r"):
    inst, _, comp = name.partition("%")
    rank = len(bounds.split(",")) if bounds else 0
    return VarInfo(
        name=name,
        instance=inst,
        field=comp,
        type=ftype,
        rank=rank,
        dims=parse_dims(bounds, rank, name),
        status=status,
        kind=kind,
    )


def inventory(case_dir: Path, extra: Optional[list[VarInfo]] = None) -> Inventory:
    case_dir.mkdir(parents=True, exist_ok=True)
    vs = [
        var("veg_pp%itype", "begp:endp", "topology", "integer"),
        var("a%x", "begp:endp", "input"),
        var("a%prof", "begp:endp,1:nlev", "input"),
        var("a%state", "begp:endp", "inout", status="rw"),
        var("b%y", "begp:endp", "output", status="w"),
        var("b%ysum", "begp:endp", "output", status="w"),
        var("b%never", "begp:endp", "output", status="w"),
        var("a%boost", "begp:endp", "input"),
        var("b%extra", "begp:endp", "output", status="w"),
        *(extra or []),
    ]
    return Inventory(
        case_dir=case_dir,
        routines=["m::r"],
        vars={v.name: v for v in vs},
        constants={"use_extra": ConstInfo("use_extra", "m", "logical", 0, [], None)},
        parameters={"nlev": NLEV},
        aliases={
            "x": "a%x",
            "prof": "a%prof",
            "state": "a%state",
            "y": "b%y",
            "ysum": "b%ysum",
            "never": "b%never",
            "boost": "a%boost",
            "extra": "b%extra",
        },
        filters={},
        instances={},
        namelist={"use_extra": False},
        accesses={
            "a%boost": [TracedAccess("r", False, (USE_EXTRA,))],
            "b%extra": [TracedAccess("w", False, (USE_EXTRA,))],
        },
    )


def spec_data(case_dir: Path, **tables) -> dict:
    data = {
        "case": {"dir": str(case_dir)},
        "settings": {"max_examples": 25, "database": None},
        "domain": {"pfts": [1, 13], "per_pft": 2, "nsets": 2},
        "scenario": {
            "s": {
                "inputs": {
                    "a%x": {"range": [0.0, 1.0]},
                    "a%prof": {"range": [0.0, 2.0]},
                    "a%state": {"range": [-1.0, 1.0]},
                }
            }
        },
    }
    data.update(tables)
    return data


class FakeBackend:
    """bug: "" (correct); "neighbour" (y also takes 0.1 x of the previous element of its half);
    "offset" (y is 1e-6 larger); "abort" (the routine stops when any x > 0.9)."""

    def __init__(
        self,
        bug: str = "",
        statics: Optional[dict[str, np.ndarray]] = None,
        extra: bool = False,
    ):
        self.bug = bug
        self.statics = statics or {}
        self.extra = extra  # the value of use_extra this "process" runs under
        self.calls = 0
        self.stopped = False
        self.seen: set[str] = set()  # every input a run was given

    def get(self, names):
        return {
            n: (
                FArray(np.asarray(self.statics[n]), (1,) * np.ndim(self.statics[n]), n)
                if n in self.statics
                else None
            )
            for n in names
        }

    def run(self, inputs, outputs, trap=False):
        self.calls += 1
        self.seen.update(inputs)
        x = np.asarray(inputs["a%x"], dtype=float)
        if self.bug == "abort" and (x > 0.9).any():
            raise FortranAbort(
                "compute: the Fortran process ended (exit code 2)\nERROR STOP x too large"
            )
        state = np.asarray(inputs["a%state"], dtype=float)
        y = 2.0 * x + state
        if self.bug == "neighbour":
            half = x.size // 2
            for h in (0, 1):
                seg = x[h * half : (h + 1) * half]
                y[h * half : (h + 1) * half] += 0.1 * np.roll(seg, 1)
        if self.bug == "offset":
            y = y + 1.0e-6
        extra = np.full(x.shape, np.nan)
        if self.extra:
            boost = np.asarray(
                inputs["a%boost"], dtype=float
            )  # read only under the option: must have been set
            y = y + boost
            extra = 2.0 * boost
        values = {
            "b%y": y,
            "b%ysum": np.asarray(inputs["a%prof"], dtype=float).sum(axis=1),
            "a%state": state + 1.0,
            "b%never": np.full(x.shape, np.nan),
            "b%extra": extra,
            **{k: np.asarray(v) for k, v in inputs.items() if k != "a%state"},
        }
        return {n: FArray(values[n], (1,) * values[n].ndim, n) for n in outputs}

    def stop(self):
        self.stopped = True


def backends(bug: str = "", made: Optional[list] = None):
    """A backend factory for `Session(backend=...)`: one FakeBackend per set of namelist options."""

    def make(options):
        b = FakeBackend(bug, extra=bool(options.get("use_extra", False)))
        if made is not None:
            made.append(b)
        return b

    return make
