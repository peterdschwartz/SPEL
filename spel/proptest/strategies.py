"""Input rules -> values, drawn through Hypothesis.

Hypothesis' per-example data budget (a few thousand choices) can't hold an independent float for
every element of every input, so a draw is structured:

- one numpy seed per example;
- per input a mode: 0 = one value for all elements (Hypothesis draws it), 1 = independent values
  per entry (numpy, from the seed), 2 = edges (each entry at the low or high end), and for inputs
  with other dimensions than the subgrid one (layers, bands) 3 = independent per element but one
  value along those dimensions (a uniform column).

Values are drawn as u in [0, 1] and mapped through the rule's quantile function (rules.py), so
per-pft and per-level rules (`pft = {13 = {...}}`, `levels = {"1:3" = {...}}`) reuse the same u,
and the active-subspace analysis has its normalized coordinates. Shrinking moves modes towards 0,
so a minimal failing example has as many constant inputs as possible.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from spel.proptest.errors import SpecError
from spel.proptest.rules import KINDS, Rule

MODES = ("constant", "per-element", "edges", "per-column")
EDGE = 1e-3  # u of the edges for unbounded distributions


def quantile(rule: Rule, base: Optional[np.ndarray] = None) -> Callable[[np.ndarray], np.ndarray]:
    """u in [0, 1] -> value."""
    kind = KINDS.get(rule.kind)
    if kind is None:
        raise SpecError(f"{rule.name}: unsupported rule {rule.describe()}")
    try:
        return kind.quantile(rule.args, base)
    except SpecError as exc:
        raise SpecError(f"{rule.name}: {exc}") from None


def _bounded(rule: Rule) -> bool:
    return KINDS[rule.kind].bounded(rule.args)


def _limited(rule: Rule, out: np.ndarray, inherited_clip: Optional[tuple[float, float]]) -> np.ndarray:
    """`out` within the rule's own limits: the kind's (min/max of a normal), then its `clip`."""
    limits = KINDS[rule.kind].limits(rule.args)
    if limits is not None:
        out = np.clip(out, *limits)
    clip = rule.clip if rule.clip is not None else inherited_clip
    return out if clip is None else np.clip(out, *clip)


def values_of(rule: Rule, u: np.ndarray, pft: Optional[np.ndarray], axis: Optional[int], ftype: str,
              base: Optional[np.ndarray] = None, levels: Optional[list[tuple[tuple, Rule]]] = None) -> np.ndarray:
    """Map u (shape of the variable) through the rule and its overrides. levels: (index of the
    section in the variable's array, rule) for each `levels` override (runner.Session.level_rules)."""
    out = _limited(rule, np.asarray(quantile(rule, base)(u), dtype=float), None)
    if rule.by_pft and pft is not None and axis is not None:
        for p, sub in rule.by_pft.items():
            mask = pft == p
            if not mask.any():
                continue
            # a pft's rule has its own limits; without a clip of its own it keeps the base rule's
            v = _limited(sub, np.asarray(quantile(sub, base)(u), dtype=float), rule.clip)
            sel = [slice(None)] * u.ndim
            sel[axis] = mask
            out[tuple(sel)] = v[tuple(sel)]
    for sel, sub in levels or []:
        b = None if base is None else base[sel]
        out[sel] = _limited(sub, np.asarray(quantile(sub, b)(u[sel]), dtype=float), rule.clip)
    if ftype == "integer":
        return np.rint(out).astype(np.int32)
    if ftype == "logical":
        return out != 0
    return out


def drawn(rule: Rule) -> bool:
    """Whether the rule (or one of its overrides) has anything to draw."""
    return any(not KINDS[r.kind].constant for r in [rule, *rule.by_pft.values(), *(s for _, s in rule.by_level)])


def draw_u(draw: Callable, name: str, rule: Rule, shape: tuple[int, ...], rng: np.random.Generator,
           axis: Optional[int] = None) -> tuple[np.ndarray, int]:
    """u for one input and its mode, the mode drawn by Hypothesis (`draw` = data.draw). axis: the
    variable's subgrid axis (None: no subgrid dimension)."""
    from hypothesis import strategies as st

    if not drawn(rule):
        return np.zeros(shape), 0
    if KINDS[rule.kind].relative or not shape:
        return np.full(shape, draw(st.floats(0.0, 1.0), label=f"u[{name}]")), 0
    columns = axis is not None and len(shape) > 1
    mode = draw(st.integers(0, 3 if columns else 2), label=f"mode[{name}]")
    if mode == 0:
        return np.full(shape, draw(st.floats(0.0, 1.0), label=f"u[{name}]")), 0
    if mode == 1:
        return rng.random(shape), 1
    if mode == 3:
        per_element = [1] * len(shape)
        per_element[axis] = shape[axis]
        return np.broadcast_to(rng.random(per_element), shape).copy(), 3
    edge = 0.0 if _bounded(rule) else EDGE
    return np.where(rng.random(shape) < 0.5, edge, 1.0 - edge), 2


def draw_seed(draw: Callable) -> np.random.Generator:
    from hypothesis import strategies as st

    return np.random.default_rng(draw(st.integers(0, 2**32 - 1), label="seed"))


def describe_modes(modes: dict[str, int]) -> str:
    """The inputs not drawn as one constant."""
    return ", ".join(f"{k}={MODES[v]}" for k, v in sorted(modes.items()) if v) or "all constant"
