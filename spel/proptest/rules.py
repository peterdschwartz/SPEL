"""Input rules of a scenario (`"inst%field" = { range = [0.1, 6.0] }`) and the registry of rule kinds.

A rule kind says how its table is checked and how u in [0, 1] maps to a value (its quantile
function); everything else (drawing u, per-pft and per-level overrides, clipping, the active
subspace's coordinates) works on that map and is the same for every kind. A new kind is one
class:

    @register
    class Triangular(RuleKind):
        name = "triangular"                       # the key: { triangular = [lo, mode, hi] }

        def args(self, value, where):
            lo, mode, hi = (float(x) for x in value["triangular"])
            if not lo <= mode <= hi:
                raise SpecError(f"{where}: bad triangular {value['triangular']}")
            return {"triangular": [lo, mode, hi]}

        def quantile(self, args, base):
            lo, mode, hi = args["triangular"]
            ...
            return lambda u: ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Callable, Optional

import numpy as np

from spel.proptest.errors import SpecError

DISTS = ("uniform", "loguniform", "normal", "lognormal")
COMMON_KEYS = {"clip", "units", "pft", "levels", "note"}
_NORMAL = NormalDist()


@dataclass
class Rule:
    name: str  # inst%field
    kind: str
    args: dict[str, Any]
    clip: Optional[tuple[float, float]] = None
    by_pft: dict[int, "Rule"] = field(default_factory=dict)
    # overrides for sections of the first non-subgrid dimension: (section text, rule)
    by_level: list[tuple[str, "Rule"]] = field(default_factory=list)

    def describe(self) -> str:
        a = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.kind}({a})"


class RuleKind:
    name = ""
    keys: tuple[str, ...] = ()  # keys of the table besides `name` and the common ones
    parameters = True  # a parameter (no subgrid dimension) may take it
    relative = False  # a factor on the variable's own value (the quantile map gets it as `base`)
    constant = False  # nothing to draw: the same value for every u
    fixed = ""  # why the active-subspace analysis cannot step it ("" = a continuous coordinate)

    def args(self, value: dict, where: str) -> dict[str, Any]:
        """The checked arguments of a rule table (raise SpecError)."""
        raise NotImplementedError

    def quantile(self, args: dict[str, Any], base: Optional[np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
        """u in [0, 1] -> value."""
        raise NotImplementedError

    def bounded(self, args: dict[str, Any]) -> bool:
        """Whether u = 0 and u = 1 are finite values (the edges mode uses them)."""
        return True

    def limits(self, args: dict[str, Any]) -> Optional[tuple[float, float]]:
        """Bounds the kind itself puts on its values (before `clip`)."""
        return None


KINDS: dict[str, RuleKind] = {}


def register(cls: type[RuleKind]) -> type[RuleKind]:
    if not cls.name:
        raise ValueError(f"{cls.__name__}: a rule kind needs a name")
    KINDS[cls.name] = cls()
    return cls


def _pair(value: dict, key: str, where: str) -> tuple[float, float]:
    try:
        lo, hi = (float(x) for x in value[key])
    except (TypeError, ValueError):
        raise SpecError(f"{where}: {key} takes [low, high]") from None
    if not lo <= hi:
        raise SpecError(f"{where}: bad {key} {value[key]}")
    return lo, hi


@register
class Range(RuleKind):
    name = "range"

    def args(self, value, where):
        _pair(value, "range", where)
        return {"range": value["range"]}

    def quantile(self, args, base):
        lo, hi = map(float, args["range"])
        return lambda u: lo + u * (hi - lo)


@register
class LogRange(RuleKind):
    name = "log_range"
    parameters = False

    def args(self, value, where):
        lo, _ = _pair(value, "log_range", where)
        if lo <= 0:
            raise SpecError(f"{where}: bad log_range {value['log_range']}")
        return {"log_range": value["log_range"]}

    def quantile(self, args, base):
        lo, hi = (math.log(float(x)) for x in args["log_range"])
        return lambda u: np.exp(lo + u * (hi - lo))


@register
class Values(RuleKind):
    name = "values"
    fixed = "discrete values"

    def args(self, value, where):
        if not value["values"]:
            raise SpecError(f"{where}: empty values")
        return {"values": value["values"]}

    def quantile(self, args, base):
        vals = np.asarray(args["values"])
        return lambda u: vals[np.minimum((np.asarray(u) * len(vals)).astype(int), len(vals) - 1)]


@register
class Value(RuleKind):
    name = "value"
    constant = True
    fixed = "fixed value"

    def args(self, value, where):
        return {"value": value["value"]}

    def quantile(self, args, base):
        return lambda u: np.full(np.shape(u), args["value"])


@register
class Scale(RuleKind):
    name = "scale"
    relative = True

    def args(self, value, where):
        _pair(value, "scale", where)
        return {"scale": value["scale"]}

    def quantile(self, args, base):
        if base is None:
            raise SpecError("scale needs the variable's value")
        lo, hi = map(float, args["scale"])
        return lambda u: base * (lo + np.asarray(u) * (hi - lo))


@register
class Dist(RuleKind):
    name = "dist"
    keys = ("mean", "sigma", "min", "max")
    parameters = False

    def args(self, value, where):
        d = value["dist"]
        if d not in DISTS:
            raise SpecError(f"{where}: dist must be one of {DISTS}")
        args = {"dist": d}
        for k in self.keys:
            if k in value:
                args[k] = float(value[k])
        if d in ("normal", "lognormal") and not {"mean", "sigma"} <= set(args):
            raise SpecError(f"{where}: {d} needs mean and sigma")
        if d in ("uniform", "loguniform") and not {"min", "max"} <= set(args):
            raise SpecError(f"{where}: {d} needs min and max")
        return args

    def quantile(self, args, base):
        d = args["dist"]
        if d == "uniform":
            return lambda u: args["min"] + u * (args["max"] - args["min"])
        if d == "loguniform":
            lo, hi = math.log(args["min"]), math.log(args["max"])
            return lambda u: np.exp(lo + u * (hi - lo))
        ppf = np.vectorize(lambda x: _NORMAL.inv_cdf(min(max(x, 1e-12), 1 - 1e-12)))
        if d == "normal":
            return lambda u: args["mean"] + args["sigma"] * ppf(u)
        return lambda u: np.exp(args["mean"] + args["sigma"] * ppf(u))

    def bounded(self, args):
        return args["dist"] not in ("normal", "lognormal")

    def limits(self, args):
        if not self.bounded(args) and {"min", "max"} <= set(args):
            return args["min"], args["max"]
        return None


def parse(name: str, value: Any, where: str) -> Rule:
    """A rule from its TOML value: a bare value (a constant) or a table with exactly one kind."""
    if not isinstance(value, dict):
        return Rule(name, "value", {"value": value})
    kinds = [k for k in KINDS if k in value]
    if len(kinds) != 1:
        raise SpecError(f"{where}: give exactly one of {tuple(KINDS)}")
    kind = KINDS[kinds[0]]
    args = kind.args(value, where)
    extra = set(value) - {kind.name, *kind.keys} - COMMON_KEYS
    if extra:
        raise SpecError(f"{where}: unknown keys {sorted(extra)}")
    clip = None
    if "clip" in value:
        clip = _pair(value, "clip", where)
    by_pft = {int(p): parse(name, sub, f"{where}.pft.{p}") for p, sub in value.get("pft", {}).items()}
    by_level = [(str(sec), parse(name, sub, f"{where}.levels.{sec}")) for sec, sub in value.get("levels", {}).items()]
    for sub in [*by_pft.values(), *(r for _, r in by_level)]:
        if sub.by_pft or sub.by_level:
            raise SpecError(f"{where}: an override (pft, levels) cannot have overrides of its own")
    return Rule(name, kind.name, args, clip, by_pft, by_level)
