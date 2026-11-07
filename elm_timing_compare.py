#!/usr/bin/env python3
"""
Compare the ELM (land) timers and throughput of two E3SM cases.

Each case can be given as its run directory, case directory or timing
directory. CIME writes timing/ in the run directory and moves it to the case
directory when the run is archived; both places are searched (the run
directory's CASEROOT file points to the case). The newest
e3sm_timing_stats* (GPTL global statistics) and e3sm_timing.* (summary) are
used unless --job picks a job id.

Sections:
  1. per case: land timers (GPTL "l:" timers) ordered by time spent
  2. the land timers of both cases side by side, with the speedup factor
  3. effective SYPD (model throughput) and land-only throughput, with the
     speedup factor of case B over case A

Timers are inclusive (a timer contains the timers nested in it), so their
times don't add up. max is the slowest process (the critical path); total is
summed over all processes (process-seconds), so compare totals between cases
with the same number of processes. --format markdown prints Markdown tables.

Example:
  ./elm_timing_compare.py $SCRATCH/E3SM_testings/<nocrop-case> \\
      $SCRATCH/E3SM_testings/<crop-case> --labels nocrop crop --top 25
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Timer:
    name: str
    processes: int
    count: float
    walltotal: float  # summed over processes and threads
    wallmax: float  # slowest process

    @property
    def wallavg(self) -> float:
        return self.walltotal / max(self.processes, 1)


@dataclass
class CaseTiming:
    label: str
    stats_file: Path
    summary_file: Path | None
    timers: dict[str, Timer]
    sypd: float | None  # Model Throughput, simulated years / wallclock day
    lnd_sypd: float | None  # LND Run Time, simulated years / wallclock day
    run_time: float | None  # seconds
    sim_days: float | None


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def timing_dirs(path: Path) -> list[Path]:
    """Directories that may hold this case's timing files."""
    path = path.resolve()
    dirs = [path, path / "timing"]
    caseroot = path / "CASEROOT"  # run directory -> case directory
    if caseroot.is_file():
        case = Path(caseroot.read_text().strip())
        dirs += [case / "timing"]
    elif (path / "env_run.xml").is_file():  # case directory -> run directory
        m = re.search(r'id="RUNDIR"[^>]*value="([^"]+)"', (path / "env_run.xml").read_text())
        if m and "$" not in m.group(1):
            dirs += [Path(m.group(1)) / "timing"]
    return [d for d in dict.fromkeys(dirs) if d.is_dir()]


def find_files(path: Path, job: str | None) -> tuple[Path, Path | None]:
    if path.is_file():  # a saved stats file, e.g. one of several repeats of a run
        return path, None

    def newest(pattern: str) -> list[Path]:
        found = [f for d in timing_dirs(path) for f in d.glob(pattern)]
        if job:
            found = [f for f in found if f".{job}." in f.name]
        return sorted(found, key=lambda f: f.stat().st_mtime, reverse=True)

    # model_timing_stats: runs started directly (run_e3sm.sh), which skip CIME's timing archive
    stats = newest("e3sm_timing_stats*") or newest("model_timing_stats")
    if not stats:
        sys.exit(f"no e3sm_timing_stats* or model_timing_stats for {path} (looked in {[str(d) for d in timing_dirs(path)]})")
    summaries = [f for f in newest("e3sm_timing.*") if "stats" not in f.name]
    # prefer the summary of the same job as the stats file
    job_id = re.search(r"e3sm_timing_stats\.(\d+)\.", stats[0].name)
    if job_id:
        same = [f for f in summaries if f".{job_id.group(1)}." in f.name]
        summaries = same or summaries
    return stats[0], (summaries[0] if summaries else None)


_stats_line = re.compile(
    r'^"(?P<name>[^"]+)"\s+\S+\s+(?P<procs>\d+)\s+\d+\s+(?P<count>\S+)\s+(?P<total>\S+)\s+(?P<max>\S+)'
)


def parse_stats(path: Path) -> dict[str, Timer]:
    timers = {}
    with _open(path) as f:
        for line in f:
            if m := _stats_line.match(line):
                timers[m["name"]] = Timer(
                    name=m["name"],
                    processes=int(m["procs"]),
                    count=float(m["count"]),
                    walltotal=float(m["total"]),
                    wallmax=float(m["max"]),
                )
    return timers


def parse_summary(path: Path | None) -> tuple[float | None, ...]:
    if path is None:
        return None, None, None, None

    def grab(pattern: str, text: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    with _open(path) as f:
        text = f.read()
    return (
        grab(r"Model Throughput:\s*([\d.]+)\s*simulated_years/day", text),
        grab(r"LND Run Time:.*?([\d.]+)\s*myears/wday", text),
        grab(r"Run Time\s*:\s*([\d.]+)\s*seconds", text),
        grab(r"Init Time\s*:.*\n.*?(?:simulated|Run Length)\s*:\s*([\d.]+)\s*days", text)
        or grab(r"Run Length\s*:\s*([\d.]+)\s*days", text),
    )


def load(path: Path, label: str, job: str | None) -> CaseTiming:
    stats, summary = find_files(path, job)
    sypd, lnd_sypd, run_time, sim_days = parse_summary(summary)
    return CaseTiming(label, stats, summary, parse_stats(stats), sypd, lnd_sypd, run_time, sim_days)


def land_timers(case: CaseTiming) -> dict[str, Timer]:
    return {n: t for n, t in case.timers.items() if n.startswith("l:")}


SORT_KEYS = {
    "max": lambda t: t.wallmax,
    "avg": lambda t: t.wallavg,
    "total": lambda t: t.walltotal,
}


def factor(a: float | None, b: float | None, higher_is_better: bool) -> str:
    """'1.85x faster' / '0.62x (1.61x slower)' for B relative to A."""
    if not a or not b:
        return "n/a"
    speedup = (b / a) if higher_is_better else (a / b)
    if speedup >= 1:
        return f"{speedup:.2f}x faster"
    return f"{speedup:.2f}x ({1 / speedup:.2f}x slower)"


def render(title: str, notes: list[str], headers: list[str], rows: list[list[str]], fmt: str) -> None:
    """Print one table: aligned text (numbers right-aligned) or a Markdown table."""
    if fmt == "markdown":
        print(f"\n### {title}\n")
        for note in notes:
            print(f"{note}  ")
        if notes:
            print()
        esc = lambda c: c.replace("|", "\\|")  # noqa: E731
        print("| " + " | ".join(esc(h) for h in headers) + " |")
        print("|" + "|".join(":---" if i == 0 else "---:" for i in range(len(headers))) + "|")
        for row in rows:
            print("| " + " | ".join(esc(c) for c in row) + " |")
        return
    print(f"\n=== {title} ===")
    for note in notes:
        print(note)
    widths = [max(len(r[i]) for r in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, *rows]:
        cells = [c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths))]
        print("  ".join(cells).rstrip())


def num(v: float | None, missing: str = "-") -> str:
    return f"{v:.3f}" if v is not None else missing


def print_case(case: CaseTiming, sort: str, top: int, fmt: str) -> None:
    timers = sorted(land_timers(case).values(), key=SORT_KEYS[sort], reverse=True)[:top]
    rows = [
        [t.name, num(t.wallmax), num(t.wallavg), num(t.walltotal), f"{t.count / max(t.processes, 1):.0f}"]
        for t in timers
    ]
    render(
        f"{case.label}: land timers by {sort} time",
        [f"stats: `{case.stats_file}`" if fmt == "markdown" else f"stats: {case.stats_file}"],
        ["timer", "max (s)", "avg (s)", "total (s)", "calls/proc"],
        rows,
        fmt,
    )


def print_comparison(a: CaseTiming, b: CaseTiming, sort: str, top: int, fmt: str) -> None:
    ta, tb = land_timers(a), land_timers(b)
    key = SORT_KEYS[sort]
    names = sorted(
        set(ta) | set(tb),
        key=lambda n: max(key(ta[n]) if n in ta else 0.0, key(tb[n]) if n in tb else 0.0),
        reverse=True,
    )[:top]
    rows = []
    for n in names:
        row = [n]
        for metric in ("max", "total"):  # slowest process, and summed over processes
            va = SORT_KEYS[metric](ta[n]) if n in ta else None
            vb = SORT_KEYS[metric](tb[n]) if n in tb else None
            row += [num(va), num(vb), factor(va, vb, higher_is_better=False)]
        rows.append(row)
    render(
        f"land timers, {a.label} vs {b.label} (ordered by {sort} time)",
        [
            f"factor: {b.label} relative to {a.label}; "
            "max = slowest process, total = summed over processes"
        ],
        [
            "timer",
            f"{a.label} max (s)", f"{b.label} max (s)", "max factor",
            f"{a.label} total (s)", f"{b.label} total (s)", "total factor",
        ],
        rows,
        fmt,
    )


def print_throughput(a: CaseTiming, b: CaseTiming, fmt: str) -> None:
    quote = (lambda p: f"`{p}`") if fmt == "markdown" else str
    notes = [f"{c.label}: summary {quote(c.summary_file) if c.summary_file else 'not found'}" for c in (a, b)]
    stats = [
        ("effective SYPD (model throughput)", a.sypd, b.sypd, True),
        ("land SYPD (LND run time)", a.lnd_sypd, b.lnd_sypd, True),
        ("run time (s)", a.run_time, b.run_time, False),
    ]
    elm_a = a.timers.get("l:elm_run")
    elm_b = b.timers.get("l:elm_run")
    if elm_a and elm_b:
        stats.append(("l:elm_run max (s)", elm_a.wallmax, elm_b.wallmax, False))
        stats.append(("l:elm_run total (s)", elm_a.walltotal, elm_b.walltotal, False))
    if a.sim_days and b.sim_days and a.sim_days != b.sim_days:
        notes.append(f"note: simulated lengths differ ({a.sim_days} vs {b.sim_days} days)")
    render(
        f"throughput ({b.label} relative to {a.label})",
        notes,
        ["", a.label, b.label, "factor"],
        [[name, num(va, "n/a"), num(vb, "n/a"), factor(va, vb, higher)] for name, va, vb, higher in stats],
        fmt,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case_a", type=Path, help="run, case or timing directory of case A (baseline)")
    p.add_argument("case_b", type=Path, help="run, case or timing directory of case B")
    p.add_argument("--labels", nargs=2, metavar=("A", "B"), help="names for the two cases")
    p.add_argument("--sort", choices=SORT_KEYS, default="max",
                   help="max: slowest process (default); avg: mean over processes; total: summed over processes")
    p.add_argument("--top", type=int, default=30, help="timers to show per table (default 30)")
    p.add_argument("--job-a", help="job id of case A's timing files (default: newest)")
    p.add_argument("--job-b", help="job id of case B's timing files (default: newest)")
    p.add_argument("--format", choices=("text", "markdown"), default="text",
                   help="text: aligned columns (default); markdown: Markdown tables")
    args = p.parse_args()

    def default_label(path: Path) -> str:
        path = path.resolve()
        return path.parent.name if path.name in ("timing", "run") else path.name

    labels = args.labels or [default_label(args.case_a), default_label(args.case_b)]
    a = load(args.case_a, labels[0], args.job_a)
    b = load(args.case_b, labels[1], args.job_b)
    if args.format == "markdown":
        print(f"## ELM timing: {a.label} vs {b.label}")
    print_case(a, args.sort, args.top, args.format)
    print_case(b, args.sort, args.top, args.format)
    print_comparison(a, b, args.sort, args.top, args.format)
    print_throughput(a, b, args.format)


if __name__ == "__main__":
    main()
