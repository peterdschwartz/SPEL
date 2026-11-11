"""Functions property expressions may call besides the Fortran intrinsics of fexpr.

Arguments arrive evaluated: per-element vectors (or (elements, levels) arrays for `x(p, :)`),
with numpy semantics. A spec adds its own from Python files next to it:

    # my.proptest.toml                      # helpers.py
    [functions]                             import numpy as np
    modules = ["helpers.py"]                from spel.proptest.functions import register

                                            @register
                                            def qsat(t, p):
                                                es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
                                                return 0.622 * es / (p - 0.378 * es)

after which `qsat(forc_t(t), forc_pbot(t))` works in `derive`, `assume` and the checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from spel.proptest.errors import SpecError

_MARK = "_spel_proptest_name"


def close(a, b, rtol=1e-12, atol=0.0):
    """|a - b| <= atol + rtol * |b|, element-wise (NaN is never close)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.abs(a - b) <= atol + rtol * np.abs(b)


def agree(a, b, rtol=0.0, atol=0.0):
    """Equal within the tolerances (identical by default), or both NaN: for comparing two runs,
    where a value neither run wrote is not a difference."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    with np.errstate(invalid="ignore"):
        return (a == b) | (np.abs(a - b) <= atol + rtol * np.abs(b)) | (np.isnan(a) & np.isnan(b))


def between(x, lo, hi):
    """lo <= x <= hi (both ends included)."""
    x = np.asarray(x)
    return (x >= lo) & (x <= hi)


def finite(x):
    """Every value finite and not ELM's spval (1e36); over trailing dimensions too."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x) & (np.abs(x) < 1.0e35)
    return ok.reshape(ok.shape[0], -1).all(axis=1) if ok.ndim > 1 else ok


def integral(x, dz):
    """Layer integral sum(x * dz) over the trailing dimension, per element."""
    return np.sum(np.asarray(x, dtype=float) * np.asarray(dz, dtype=float), axis=-1)


def cumsum(x):
    """Running sum along the trailing dimension (layer interfaces from thicknesses: `cumsum(dz(c, :))`)."""
    return np.cumsum(np.asarray(x), axis=-1)


def rel_change(new, old, floor=1e-30):
    """(new - old) / max(|old|, floor)."""
    new, old = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
    return (new - old) / np.maximum(np.abs(old), floor)


def implies(a, b):
    """.not. a .or. b."""
    return np.logical_or(np.logical_not(a), b)


def active_dir(*args, **kwargs):
    """`active_dir(p, "t_veg(p)", k)`: the coordinate of each element's drawn inputs along the
    k-th direction (default 1) of the active subspace `spel proptest subspace` computed for that
    output and this scenario; inputs normalized to [-1, 1] as there. Bound to the example by
    the runner (`Session.context_functions`), so it only works inside a run."""
    raise SpecError("active_dir is only available in the checks of a run")


FUNCTIONS: dict[str, Callable] = {
    "close": close, "agree": agree, "same": agree, "between": between, "finite": finite, "integral": integral,
    "cumsum": cumsum, "rel_change": rel_change, "implies": implies, "active_dir": active_dir,
}


def register(fn: Optional[Callable] = None, *, name: Optional[str] = None):
    """Mark a function of a `[functions] modules` file as callable from the spec's expressions,
    under its own name or `name` (lower case, as Fortran names are)."""

    def mark(f: Callable) -> Callable:
        setattr(f, _MARK, (name or f.__name__).lower())
        return f

    return mark(fn) if fn is not None else mark


def load(table: Any, base: Path) -> dict[str, Callable]:
    """The built-in functions plus those the spec's `[functions] modules` register."""
    from spel.proptest.fexpr import ELEMENTAL, REDUCTIONS

    out = dict(FUNCTIONS)
    if not table:
        return out
    if not isinstance(table, dict) or set(table) - {"modules"}:
        raise SpecError('[functions]: modules = ["file.py", ...]')
    modules = table.get("modules", [])
    for entry in [modules] if isinstance(modules, str) else modules:
        path = Path(entry) if Path(entry).is_absolute() else Path(base) / entry
        if not path.is_file():
            raise SpecError(f"[functions]: no file {path}")
        digest = hashlib.sha1(str(path).encode()).hexdigest()[:8]
        mspec = importlib.util.spec_from_file_location(f"spel_proptest_functions_{path.stem}_{digest}", path)
        module = importlib.util.module_from_spec(mspec)
        try:
            mspec.loader.exec_module(module)
        except Exception as exc:
            raise SpecError(f"[functions]: {path}: {type(exc).__name__}: {exc}") from None
        found = {getattr(f, _MARK): f for f in vars(module).values() if callable(f) and hasattr(f, _MARK)}
        if not found:
            raise SpecError(f"[functions]: {path} registers no function (decorate them with "
                            "spel.proptest.functions.register)")
        for fname in found:
            if fname in FUNCTIONS or fname in ELEMENTAL or fname in REDUCTIONS:
                raise SpecError(f"[functions]: {path}: {fname} is already a function of the expression language")
        out.update(found)
    return out
