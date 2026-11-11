"""`spel proptest coverage`: the branches of the routines under test a scenario never enters.

A scenario may pass every check because it never reaches the code a check is about. This runs
examples of a scenario on a gfortran debug build of the case (`-fprofile-arcs -ftest-coverage`,
the build variant `build-py-gfortran-off-dbg`) and reads the line counts back with gcov:

- line counts come from gcov's JSON, as the difference between before and after the run (the
  counters accumulate in the build directory; nothing is deleted);
- a branch whose condition the scenario's namelist options decide is reported apart ("off"):
  it is not something the scenario failed to reach. The decision is the analysis' own
  (`infer_condition_expectations` on the parsed condition, `evaluate_guard` on the values);
- which lines form a branch comes from SPEL's parser: every routine of the call tree of the
  routines under test (`Subroutine.child_subroutines`) is parsed from the case's copy of its file
  (`registrations.parse_routine`, the line range gcov reports for the function), and an `if`,
  `else if` or `else` branch is "never entered" when none of its lines ran. Only the outermost
  such branch is reported.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from spel.proptest import build
from spel.proptest.errors import SpecError
from spel.proptest.inventory import Inventory
from spel.proptest.spec import Spec
from spel.proptest.worker import FortranAbort


@dataclass
class Branch:
    file: str
    line: int  # 1-based line of the `if` / `else if` / `else`
    kind: str
    condition: str
    routine: str
    lines: int  # executable lines in the branch
    off: bool = False  # the scenario's namelist options decide that it cannot run


@dataclass
class RoutineCoverage:
    routine: str
    file: str
    calls: int
    executable: int
    executed: int


@dataclass
class CoverageResult:
    scenario: str
    examples: int
    aborted: int
    routines: list[RoutineCoverage] = field(default_factory=list)
    branches: list[Branch] = field(default_factory=list)
    unparsed: dict[str, str] = field(default_factory=dict)  # routine -> why its branches are not listed


def call_tree(inv: Inventory) -> dict[tuple[str, str], str]:
    """(module, routine) -> source file name, for the routines under test and everything they call."""
    out: dict[tuple[str, str], str] = {}

    def visit(sub) -> None:
        key = (str(sub.module).lower(), str(sub.name).lower())
        if key in out:
            return
        out[key] = Path(str(sub.filepath)).name
        for child in (getattr(sub, "child_subroutines", None) or {}).values():
            visit(child)

    for sub in inv.unit_test.primary_subroutines.values():
        visit(sub)
    return out


_TOOL: list[str] = []


def gcov_tool() -> str:
    """The gcov of the gfortran in use: `gcov-<major>` when it exists (the plain `gcov` of a
    system with several GCCs may read another version's files), else `gcov`."""
    if not _TOOL:
        tool = "gcov"
        try:
            major = subprocess.run(["gfortran", "-dumpversion"], capture_output=True, text=True).stdout.strip().split(".")[0]
            if major and shutil.which(f"gcov-{major}"):
                tool = f"gcov-{major}"
        except OSError:
            pass
        if shutil.which(tool) is None:
            raise build.BuildError("gcov not found (coverage needs gfortran and its gcov)")
        _TOOL.append(tool)
    return _TOOL[0]


def _gcov(bdir: Path, fname: str) -> Optional[dict[str, Any]]:
    """gcov's JSON entry for one source of the case (None before the file's first run)."""
    gcda = sorted(bdir.rglob(f"{fname}.gcda"))
    if not gcda:
        return None
    res = subprocess.run([gcov_tool(), "--json-format", "--stdout", str(gcda[0])], cwd=bdir, capture_output=True,
                         text=True)
    if res.returncode != 0 or not res.stdout.strip():
        raise build.BuildError(f"gcov failed on {gcda[0].name}: {res.stderr.strip()[-300:]}")
    for entry in json.loads(res.stdout).get("files", []):
        if Path(entry.get("file", "")).name == fname:
            return entry
    return None


def _counts(entry: Optional[dict[str, Any]]) -> tuple[dict[int, int], dict[str, int]]:
    """(line -> count, function -> calls) of a gcov entry."""
    if entry is None:
        return {}, {}
    lines: dict[int, int] = {}
    for ln in entry.get("lines", []):
        lines[ln["line_number"]] = lines.get(ln["line_number"], 0) + int(ln["count"])
    return lines, {f["name"]: int(f["execution_count"]) for f in entry.get("functions", [])}


def measure(spec: Spec, inv: Inventory, scenario: str, examples: int = 20, seed: int = 0,
            all_files: bool = False) -> CoverageResult:
    from spel.proptest.runner import Session

    if scenario not in spec.scenarios:
        raise SpecError(f"no scenario {scenario!r} (have {sorted(spec.scenarios)})")
    gcov_tool()
    from spel.proptest import spec as specmod

    scen = spec.scenarios[scenario]
    values = {k: v for k, v in specmod.constants(spec, inv, scen).items() if not getattr(v, "ndim", 0)}
    decide = decider(set(inv.namelist), values)
    variant = build.Variant(fc="gfortran", acc="OFF", dbg=True)
    bdir = build.build_dir(spec.case_dir, variant)
    routines = call_tree(inv)
    files = sorted(set(routines.values())) if not all_files else sorted(p.name for p in spec.case_dir.glob("*.F90"))
    files = [f for f in files if (spec.case_dir / f).is_file()]

    before = {f: _counts(_gcov(bdir, f)) for f in files} if bdir.is_dir() else {}
    aborted = 0
    rng = np.random.default_rng(seed)
    with Session(spec, inv, fc=variant.fc, acc=variant.acc, dbg=True, namelist=scen.namelist) as s:
        s.preflight(scen, [])
        for k in range(max(1, examples)):
            ex = s.inputs_at(scen, 0.5 if k == 0 else rng.random)
            try:
                s.backend.run(s.live_inputs(ex), [], trap=scen.trap_fpe)
            except FortranAbort:
                aborted += 1  # the counters of that worker are lost with it
    # the worker has ended: its counters are in the .gcda files now
    res = CoverageResult(scenario, max(1, examples), aborted)
    wanted = {f"__{m}_mod_{r}": (m, r) for (m, r) in routines}  # gfortran's names of module procedures, lower case
    for fname in files:
        entry = _gcov(bdir, fname)
        if entry is None:
            continue
        lines, calls = _counts(entry)
        lines0, calls0 = before.get(fname, ({}, {}))
        delta = {ln: c - lines0.get(ln, 0) for ln, c in lines.items()}
        for fn in entry.get("functions", []):
            key = wanted.get(fn["name"].lower())
            if key is None and not all_files:
                continue
            label = "::".join(key) if key else fn.get("demangled_name", fn["name"])
            lo, hi = int(fn["start_line"]), int(fn["end_line"])
            own = {ln: c for ln, c in delta.items() if lo <= ln <= hi}
            ncalls = calls.get(fn["name"], 0) - calls0.get(fn["name"], 0)
            res.routines.append(RoutineCoverage(label, fname, ncalls, len(own), sum(c > 0 for c in own.values())))
            if ncalls <= 0:
                continue
            try:
                res.branches += list(_unentered(spec.case_dir / fname, lo, hi, own, label, decide))
            except (SystemExit, Exception) as exc:  # a routine the parser cannot read back from the case copy
                res.unparsed[label] = f"{type(exc).__name__}: {str(exc)[:120]}"
    res.routines.sort(key=lambda r: (r.file, r.routine))
    res.branches.sort(key=lambda b: (b.file, b.line))
    return res


def decider(namelist: set[str], values: dict[str, Any]):
    """condition (a parsed expression) -> True/False when the namelist values decide it, else
    None. Only the namelist variables of the condition (and the cascade variables that follow
    from them) are looked at; a condition that reads none is not a namelist branch."""
    from spel.proptest.fexpr import uses
    from spel.scripts.fortran_parser.boolen_expression import infer_condition_expectations
    from spel.scripts.nml.analyze_namelist import cascade_to_triggers
    from spel.scripts.nml.namelist_cascade import NML_CASCADES
    from spel.scripts.validate_access import evaluate_guard

    gate = namelist | set(NML_CASCADES)

    def decide(condition) -> Optional[bool]:
        used = {u.path[0] for u in uses(condition)}
        names = used & gate
        if not names:
            return None
        try:
            verdict = evaluate_guard(cascade_to_triggers(infer_condition_expectations(expr=condition, variables=names)), values)
        except Exception:  # a condition the expectation inference does not handle: undecided
            return None
        # The expectation is what the namelist variables must be for the condition to hold: when
        # it fails the condition is false, but when it holds a condition that also reads other
        # things is still open.
        if verdict is True and not used <= gate:
            return None
        return verdict

    return decide


def _unentered(path: Path, lo: int, hi: int, counts: dict[int, int], routine: str, decide=None) -> Iterator[Branch]:
    """The outermost if/else-if/else branches of the routine at lines lo..hi (1-based) none of
    whose lines ran. decide: see `decider`; marks the branches the configuration turns off."""
    import io
    from contextlib import redirect_stdout

    from spel.scripts.fortran_parser.spel_ast import IfConstruct
    from spel.scripts.registrations import iter_statements, parse_routine

    with redirect_stdout(io.StringIO()):
        node = parse_routine(path, lo - 1, hi - 1, skip_io=True)
    body = getattr(node, "body", None)
    if body is None:
        return

    def lines_of(block) -> set[int]:
        return {stmt.lineno + 1 for stmt, _ in iter_statements(block)} & set(counts)

    dead: list[tuple[int, int]] = []  # line spans of branches already reported
    for stmt, _ in iter_statements(body):
        if not isinstance(stmt, IfConstruct):
            continue
        head = stmt.lineno + 1
        if any(a <= head <= b for a, b in dead):
            continue
        # each branch with whether the namelist options rule it out: its own condition is false,
        # or an earlier branch of the construct is certainly taken
        verdict = decide(stmt.condition) if decide is not None else None
        taken = verdict is True
        branches = [("if", str(stmt.condition), stmt.consequence, head, verdict is False)]
        for e in stmt.else_ifs:
            verdict = decide(e.condition) if decide is not None else None
            branches.append(("else if", str(e.condition), e.consequence, getattr(e, "lineno", stmt.lineno) + 1,
                             taken or verdict is False))
            taken = taken or verdict is True
        if stmt.else_ is not None:
            branches.append(("else", f"not ({stmt.condition})" if not stmt.else_ifs else "none of the above",
                             stmt.else_.alternative, getattr(stmt.else_, "lineno", stmt.lineno) + 1, taken))
        for kind, cond, block, at, off in branches:
            own = lines_of(block) - {head}
            if own and all(counts[ln] <= 0 for ln in own):
                dead.append((min(own), max(own)))
                yield Branch(path.name, at, kind, cond, routine, len(own), off)


def report(res: CoverageResult, max_condition: int = 90, show_off: bool = False) -> str:
    out = [f"scenario {res.scenario}: {res.examples} examples" + (f", {res.aborted} aborted" if res.aborted else "")]
    called = [r for r in res.routines if r.calls > 0]
    never = [r for r in res.routines if r.calls <= 0]
    out.append(f"  {len(called)} routines ran; lines run / executable:")
    for r in called:
        out.append(f"    {r.routine} ({r.file}): {r.executed}/{r.executable}")
    if never:
        out.append("  routines of the call tree that never ran: " + ", ".join(r.routine for r in never))
    reachable = [b for b in res.branches if not b.off]
    off = [b for b in res.branches if b.off]

    def line(b: Branch) -> str:
        cond = b.condition if len(b.condition) <= max_condition else b.condition[:max_condition - 3] + "..."
        return f"    {b.file}:{b.line} {b.kind} ({cond})  [{b.lines} lines, {b.routine.partition('::')[2] or b.routine}]"

    out.append(f"  branches the scenario could reach and never entered: {len(reachable)}")
    out += [line(b) for b in reachable]
    out.append(f"  branches the namelist options turn off: {len(off)}" + ("" if show_off or not off else " (--all lists them)"))
    if show_off:
        out += [line(b) for b in off]
    for name, why in sorted(res.unparsed.items()):
        out.append(f"  {name}: branches not listed ({why})")
    return "\n".join(out)
