"""Shared helpers of the declarations' TOML readers and of weight resolution."""

from __future__ import annotations

from typing import Optional, Sequence

from spel.proptest.hierarchy.codes import WEIGHT_TOL, HierarchyError


def resolve_weights(weights: Sequence[Optional[float]], where: str) -> list[float]:
    """Given weights, the missing ones sharing the rest equally; the result sums to 1."""
    given = [w for w in weights if w is not None]
    if any(w < 0 for w in given):
        raise HierarchyError(f"{where}: negative weight in {list(weights)}")
    rest = 1.0 - sum(given)
    missing = len(weights) - len(given)
    if rest < -WEIGHT_TOL or (missing == 0 and abs(rest) > WEIGHT_TOL) or (missing and rest <= WEIGHT_TOL):
        raise HierarchyError(f"{where}: weights {list(weights)} must sum to 1 (missing ones share the rest)")
    return [w if w is not None else rest / missing for w in weights]


def weight_of(value: dict, where: str) -> Optional[float]:
    """The optional `weight` key of a table."""
    w = value.get("weight")
    if w is None:
        return None
    if isinstance(w, bool) or not isinstance(w, (int, float)):
        raise HierarchyError(f"{where}: weight must be a number, not {w!r}")
    return float(w)


def check_keys(value: dict, known: set[str], where: str) -> None:
    extra = set(value) - known
    if extra:
        raise HierarchyError(f"{where}: unknown keys {sorted(extra)} (known: {sorted(known)})")
