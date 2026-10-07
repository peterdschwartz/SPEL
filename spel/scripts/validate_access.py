"""
Validate SPEL's static read/write classification against model data.

For every timestep, the state captured before the call (spel-inputs) is
compared with the state after it (spel-outputs, or a FUT's outputs):
  * a variable classified read-only ("r") must never change -> failure
  * a variable SPEL never classified must never change      -> failure
  * an output ("w"/"rw") that never changes on any step     -> warning only
    (e.g. it is reset to the value it already had)

Fields SPEL found to be accessed only inside namelist-gated ifs carry that
guard. The guard is evaluated with the namelist values in spel-constants:
  * guard inactive: unchanged / absent is expected; any change -> failure
  * guard active or undecidable: the rules above apply

The classification is written to the case directory by `spel create`
(ACCESS_MANIFEST_FILENAME).
"""

from __future__ import annotations

import json
import operator
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import xarray as xr
from tabulate import tabulate

from spel.scripts.fortran_parser.boolen_expression import (
    AllOf,
    AnyOf,
    ConditionExpectation,
    Expectation,
    simplify,
)

ACCESS_MANIFEST_FILENAME = "spel_access.json"
OUTPUT_STATUSES = {"w", "rw"}


def fortran_to_nc(name: str) -> str:
    return name.replace("%", "__")


def access_manifest(subroutines: Iterable) -> dict[str, str]:
    """
    Merge `elmtype_access_summary` of the unit-test subroutines:
    read-only only if read-only everywhere; written-only if written-only everywhere.
    """
    merged: dict[str, set[str]] = {}
    for sub in subroutines:
        for name, rw in sub.elmtype_access_summary.items():
            merged.setdefault(name, set()).add(getattr(rw, "status", rw))
    return {name: s.pop() if len(s) == 1 else "rw" for name, s in merged.items()}


def merge_guards(
    per_sub: Iterable[tuple[dict[str, Any], dict[str, ConditionExpectation]]],
) -> dict[str, ConditionExpectation]:
    """
    per_sub: (accessed fields, namelist guards) of each unit-test subroutine.
    A field stays guarded only if every subroutine accessing it guards it;
    the guards are then OR'ed.
    """
    guards: dict[str, list[ConditionExpectation]] = {}
    unguarded: set[str] = set()
    for accessed, sub_guards in per_sub:
        for name in accessed:
            if name in sub_guards:
                guards.setdefault(name, []).append(sub_guards[name])
            else:
                unguarded.add(name)
    merged = {}
    for name, conds in guards.items():
        if name in unguarded:
            continue
        unique = tuple(dict.fromkeys(conds))
        merged[name] = unique[0] if len(unique) == 1 else simplify(AnyOf(unique))
    return merged


def guard_to_json(cond: ConditionExpectation) -> dict:
    match cond:
        case Expectation(variable, constraint):
            return {"var": variable, "constraint": constraint}
        case AllOf(items):
            return {"all": [guard_to_json(c) for c in items]}
        case AnyOf(items):
            return {"any": [guard_to_json(c) for c in items]}
    raise TypeError(f"Unknown condition type: {type(cond)}")


def guard_from_json(obj: dict) -> ConditionExpectation:
    if "all" in obj:
        return AllOf(tuple(guard_from_json(c) for c in obj["all"]))
    if "any" in obj:
        return AnyOf(tuple(guard_from_json(c) for c in obj["any"]))
    return Expectation(obj["var"], obj["constraint"])


def write_access_manifest(
    case_dir: Path,
    subroutines: list[str],
    access: dict[str, str],
    guards: Optional[dict[str, ConditionExpectation]] = None,
) -> Path:
    path = Path(case_dir) / ACCESS_MANIFEST_FILENAME
    payload = {
        "subroutines": sorted(subroutines),
        "variables": dict(sorted(access.items())),
        "guards": {
            name: {"fortran": cond.to_fortran(), "expr": guard_to_json(cond)}
            for name, cond in sorted((guards or {}).items())
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _read_manifest(case_dir: Path) -> dict:
    path = Path(case_dir) / ACCESS_MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No {ACCESS_MANIFEST_FILENAME} in {case_dir}; re-run `spel create` for this case"
        )
    return json.loads(path.read_text())


def load_access_manifest(case_dir: Path) -> dict[str, str]:
    return _read_manifest(case_dir)["variables"]


def load_access_guards(case_dir: Path) -> dict[str, ConditionExpectation]:
    guards = _read_manifest(case_dir).get("guards", {})
    return {name: guard_from_json(g["expr"]) for name, g in guards.items()}


# ---------------- namelist guard evaluation ----------------

_COMPARE = {
    "==": operator.eq, "=": operator.eq, ".eq.": operator.eq,
    "/=": operator.ne, "!=": operator.ne, ".ne.": operator.ne,
    ">": operator.gt, ".gt.": operator.gt,
    ">=": operator.ge, ".ge.": operator.ge,
    "<": operator.lt, ".lt.": operator.lt,
    "<=": operator.le, ".le.": operator.le,
}
_KIND_SUFFIX = re.compile(r"_\w+$")


def namelist_values(ds: xr.Dataset) -> dict[str, Any]:
    """Scalar variables of a spel-constants file (namelist options, parameters)"""
    return {
        str(name): da.values.item()
        for name, da in ds.data_vars.items()
        if da.ndim == 0 and _is_numeric(da)
    }


def _fortran_value(text: str, values: dict[str, Any]) -> Optional[Any]:
    text = text.strip()
    low = text.lower()
    if low in (".true.", ".false."):
        return low == ".true."
    if text in values:
        return values[text]
    if low in values:
        return values[low]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    try:
        return float(_KIND_SUFFIX.sub("", low).replace("d", "e"))
    except ValueError:
        return None


def evaluate_guard(cond: ConditionExpectation, values: dict[str, Any]) -> Optional[bool]:
    """True/False if decidable from `values`, None otherwise (three-valued logic)"""
    match cond:
        case Expectation(variable, constraint):
            lhs = values.get(variable, values.get(variable.lower()))
            if lhs is None:
                return None
            if constraint in ("True", "False"):
                return bool(lhs) == (constraint == "True")
            op, _, rhs_text = constraint.partition(" ")
            fn = _COMPARE.get(op.lower())
            rhs = _fortran_value(rhs_text, values)
            if fn is None or rhs is None:
                return None
            if isinstance(rhs, bool):
                lhs = bool(lhs)
            try:
                return bool(fn(lhs, rhs))
            except TypeError:
                return None
        case AllOf(items):
            results = [evaluate_guard(c, values) for c in items]
            if False in results:
                return False
            return True if all(r is True for r in results) else None
        case AnyOf(items):
            results = [evaluate_guard(c, values) for c in items]
            if True in results:
                return True
            return False if all(r is False for r in results) else None
    raise TypeError(f"Unknown condition type: {type(cond)}")


@dataclass
class VarChange:
    name: str
    status: str  # SPEL classification ("-" if SPEL never saw it)
    steps_changed: int
    first_step: int  # 1-based
    max_abs_diff: float
    guard: str = ""  # namelist guard (fortran), set if it is inactive for this run


@dataclass
class AccessReport:
    nsteps: int
    checked: int = 0
    violations: list[VarChange] = field(default_factory=list)
    unchanged_outputs: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # classified, not in the files
    inactive: list[str] = field(default_factory=list)  # guard off: unchanged/absent as expected
    guard_of: dict[str, str] = field(default_factory=dict)  # nc name -> fortran guard

    @property
    def ok(self) -> bool:
        return not self.violations


def _per_step(arr: np.ndarray, dims: tuple[str, ...]) -> np.ndarray:
    """Array with time as the leading axis (a single step if there is none)."""
    if "time" in dims:
        return np.moveaxis(arr, dims.index("time"), 0)
    return arr[np.newaxis, ...]


def step_changes(pre: xr.DataArray, post: xr.DataArray, nsteps: int):
    """(changed mask per step, max |diff| over changed elements)"""
    a = _per_step(pre.values, pre.dims)[:nsteps]
    b = _per_step(post.values, post.dims)[:nsteps]
    if a.shape != b.shape:
        raise ValueError(f"{pre.name}: shapes differ {a.shape} vs {b.shape}")
    changed = a != b
    if np.issubdtype(a.dtype, np.floating):
        changed &= ~(np.isnan(a) & np.isnan(b))
    per_step = changed.reshape(changed.shape[0], -1).any(axis=1)
    max_diff = float(np.abs(a[changed] - b[changed]).max()) if changed.any() else 0.0
    return per_step, max_diff


def _is_numeric(da: xr.DataArray) -> bool:
    return np.issubdtype(da.dtype, np.number) or np.issubdtype(da.dtype, np.bool_)


def validate_access(
    pre: xr.Dataset,
    post: xr.Dataset,
    access: dict[str, str],
    guards: Optional[dict[str, ConditionExpectation]] = None,
    nml: Optional[dict[str, Any]] = None,
) -> AccessReport:
    status_of = {fortran_to_nc(k): v for k, v in access.items()}
    guards = {fortran_to_nc(k): v for k, v in (guards or {}).items()}
    nml = nml or {}
    inactive = {n for n, cond in guards.items() if evaluate_guard(cond, nml) is False}

    nsteps = min(pre.sizes.get("time", 1), post.sizes.get("time", 1))
    report = AccessReport(nsteps=nsteps)
    report.guard_of = {n: cond.to_fortran() for n, cond in guards.items()}
    absent = {n for n in status_of if n not in pre or n not in post}
    report.missing = sorted(absent - inactive)
    expected_quiet = set(absent & inactive)

    for name in pre.data_vars:
        if name not in post or not _is_numeric(pre[name]):
            continue
        report.checked += 1
        status = status_of.get(name, "-")
        per_step, max_diff = step_changes(pre[name], post[name], nsteps)
        changed = bool(per_step.any())
        if name in inactive:
            if not changed:
                expected_quiet.add(name)
                continue
        elif status in OUTPUT_STATUSES:
            if not changed:
                report.unchanged_outputs.append(name)
            continue
        if changed:
            report.violations.append(
                VarChange(
                    name=name,
                    status=status,
                    steps_changed=int(per_step.sum()),
                    first_step=int(np.argmax(per_step)) + 1,
                    max_abs_diff=max_diff,
                    guard=report.guard_of[name] if name in inactive else "",
                )
            )
    report.unchanged_outputs.sort()
    report.inactive = sorted(expected_quiet)
    return report


def _with_guard(report: AccessReport, name: str) -> str:
    guard = report.guard_of.get(name)
    return f"  {name}  [if {guard}]" if guard else f"  {name}"


def format_report(report: AccessReport) -> str:
    out = [
        f"Access validation: {report.checked} variables over {report.nsteps} timesteps",
    ]
    if report.violations:
        out.append(
            "FAIL: variables that changed although SPEL classified them as inputs (r), "
            "did not detect them (-), or only accesses them under an inactive namelist guard:"
        )
        rows = [
            (v.name, v.status, f"{v.steps_changed}/{report.nsteps}", v.first_step, v.max_abs_diff, v.guard)
            for v in report.violations
        ]
        out.append(
            tabulate(
                rows,
                headers=["Variable", "SPEL", "Steps changed", "First step", "Max |diff|", "Inactive guard"],
                tablefmt="psql",
            )
        )
    if report.unchanged_outputs:
        out.append(
            f"WARNING: {len(report.unchanged_outputs)} outputs (w/rw) never changed on any timestep:"
        )
        out.extend(_with_guard(report, n) for n in report.unchanged_outputs)
    if report.missing:
        out.append(f"NOTE: {len(report.missing)} classified variables not in the files:")
        out.extend(_with_guard(report, n) for n in report.missing)
    if report.inactive:
        out.append(
            f"OK: {len(report.inactive)} variables unchanged/absent as expected "
            "(namelist guard inactive for this run):"
        )
        out.extend(_with_guard(report, n) for n in report.inactive)
    out.append("PASS" if report.ok else "FAIL")
    return "\n".join(out) + "\n"


def constants_file_for(inputs_fn: Path) -> Optional[Path]:
    """spel-inputsNNNN.nc -> sibling spel-constantsNNNN.nc, if it exists"""
    inputs_fn = Path(inputs_fn)
    if "spel-inputs" not in inputs_fn.name:
        return None
    candidate = inputs_fn.with_name(inputs_fn.name.replace("spel-inputs", "spel-constants"))
    return candidate if candidate.exists() else None


def run_validation(
    inputs_fn: Path,
    outputs_fn: Path,
    case_dir: Path,
    ostream=None,
    constants_fn: Optional[Path] = None,
) -> int:
    """Returns the exit code: 1 if any input changed, else 0"""
    ostream = ostream or sys.stdout
    access = load_access_manifest(case_dir)
    guards = load_access_guards(case_dir)
    constants_fn = constants_fn or constants_file_for(inputs_fn)
    nml: dict[str, Any] = {}
    if constants_fn:
        with xr.open_dataset(constants_fn) as consts:
            nml = namelist_values(consts)
    with xr.open_dataset(inputs_fn) as pre, xr.open_dataset(outputs_fn) as post:
        report = validate_access(pre, post, access, guards, nml)
    ostream.write(f"Inputs (pre-call): {inputs_fn}\nOutputs (post-call): {outputs_fn}\n")
    if guards:
        if constants_fn:
            ostream.write(f"Namelist values: {constants_fn}\n")
        else:
            ostream.write("WARNING: no constants file; namelist guards can't be evaluated\n")
        unknown = sorted(
            {v for c in guards.values() for v in c.variable_names()} - nml.keys()
        )
        if unknown:
            ostream.write(f"WARNING: guard variables not in constants: {', '.join(unknown)}\n")
    ostream.write(format_report(report))
    return 0 if report.ok else 1


def case_dir_for(case: str) -> Path:
    from spel.scripts.config import unittests_dir

    path = Path(case)
    return path if path.is_dir() else Path(unittests_dir) / case
