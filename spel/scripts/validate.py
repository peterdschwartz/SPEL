"""
`spel validate`: create, capture and validate many unit tests with one ELM run.

    1. Make sure the cached elm_drv analysis exists (`spel analyze`)
    2. `spel create` every case in parallel (each in its own process)
    3. Instrument ELM for all cases at once (instrument_elm.instrument_cases)
    4. Build and run one CIME case (a failed run, not build, is resubmitted
       once); collect each case's <tag>.spel-*.nc
       (and the run's lnd_in) into unit-tests/input-data/<case>
    5. Remove the instrumentation
    6. `spel run` every case in parallel (build, run, bit-for-bit diff and
       access validation)
    7. Print a summary and write unit-tests/validate-report.json
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from spel.scripts.instrument_elm import (
    DEFAULT_FREQ,
    CaptureNames,
    CaseRunError,
    InstrumentError,
    collect_outputs,
    instrument_cases,
    load_case,
    resubmit_case,
    run_case,
    uninstrument_elm,
    write_reference_meta,
)

REPORT_FILE = "validate-report.json"
DEFAULT_RUN_NAME = "spel-validate"
LOG_TAIL = 15

PASS = "PASS"
FAIL = "FAIL"
CREATE_FAILED = "CREATE_FAILED"
INSTRUMENT_FAILED = "INSTRUMENT_FAILED"
NOT_CALLED = "NOT_CALLED"
CASE_RUN_FAILED = "ELM_RUN_FAILED"
READY = "READY"  # --dry-run: would be instrumented


@dataclass
class CaseResult:
    case: str
    subs: list[str]
    status: str = ""
    detail: str = ""
    log: Optional[str] = None
    files: int = 0


@dataclass
class ValidateReport:
    cases: dict[str, CaseResult] = field(default_factory=dict)
    rundir: Optional[str] = None
    lnd_in: Optional[str] = None
    made_public: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(r.status == PASS for r in self.cases.values())

    def to_json(self) -> dict:
        return {
            "rundir": self.rundir,
            "lnd_in": self.lnd_in,
            "made_public": self.made_public,
            "cases": {k: asdict(v) for k, v in self.cases.items()},
        }


# --------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------
def parse_request(subs: Optional[list[str]] = None, list_file: Optional[Path] = None) -> dict[str, list[str]]:
    """
    {case name: routines}. `-s a b` gives one case per routine (named after
    it); a list file has one case per line, `case: sub1 sub2` or just `sub`,
    with `#` comments.
    """
    request: dict[str, list[str]] = {}

    def add(case: str, names: list[str]) -> None:
        case = case.strip()
        if not re.fullmatch(r"[\w.-]+", case):
            raise ValueError(f"invalid case name {case!r}")
        if case in request:
            raise ValueError(f"case {case!r} listed twice")
        if not names:
            raise ValueError(f"case {case!r} has no routines")
        request[case] = [n.lower() for n in names]

    for sub in subs or []:
        add(sub.lower(), [sub])
    if list_file is not None:
        for raw in Path(list_file).read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" in line:
                case, rest = line.split(":", 1)
                add(case, rest.split())
            elif len(line.split()) == 1:
                add(line.lower(), [line])
            else:
                raise ValueError(f"{list_file}: expected `case: sub1 sub2` or `sub`, got {raw!r}")
    if not request:
        raise ValueError("no cases requested (use -s or --list)")
    return request


# --------------------------------------------------------------------------
# Parallel subprocess helpers
# --------------------------------------------------------------------------
SPEL_ROOT = Path(__file__).resolve().parents[2]


def spel_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "spel.cli", *args]


def run_logged(cmd: list[str], log: Path, env: Optional[dict] = None, cwd: Optional[Path] = None) -> int:
    """Run `cmd` with stdout/stderr in `log` and no stdin (prompts get EOF)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as out:
        out.write(f"$ {shlex.join(cmd)}\n")
        out.flush()
        return subprocess.run(
            cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            env=env, cwd=cwd or SPEL_ROOT,
        ).returncode


def log_tail(log: Path, n: int = LOG_TAIL) -> str:
    try:
        return "".join(log.read_text(errors="replace").splitlines(keepends=True)[-n:])
    except OSError:
        return ""


def parallel(func: Callable[[str], None], cases: list[str], jobs: int) -> None:
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        list(pool.map(func, cases))


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------
def create_cases(report: ValidateReport, jobs: int, log_dir: Path) -> None:
    from spel.scripts.config import spel_output_dir, unittests_dir

    def create(case: str) -> None:
        result = report.cases[case]
        log = log_dir / f"{case}.create.log"
        env = {**os.environ, "SPEL_OUTPUT_DIR": str(Path(spel_output_dir) / "validate" / case)}
        print(f"  create {case}: {' '.join(result.subs)}", flush=True)
        code = run_logged(spel_command("create", "-s", *result.subs, "-c", case), log, env=env)
        result.log = str(log)
        if code or not (Path(unittests_dir) / case / "fut.pkl").is_file():
            result.status = CREATE_FAILED
            result.detail = log_tail(log)

    parallel(create, list(report.cases), jobs)


def run_cases(report: ValidateReport, cases: list[str], jobs: int) -> None:
    from spel.scripts.config import unittests_dir

    def run(case: str) -> None:
        result = report.cases[case]
        log = Path(unittests_dir) / case / "validate.log"
        print(f"  run {case}", flush=True)
        code = run_logged(spel_command("run", case), log)
        result.log = str(log)
        result.status = PASS if code == 0 else FAIL
        result.detail = log_tail(log, 3 if code == 0 else LOG_TAIL)

    parallel(run, cases, jobs)


def write_report(report: ValidateReport, path: Path) -> None:
    path.write_text(json.dumps(report.to_json(), indent=2) + "\n")


def print_summary(report: ValidateReport, path: Path) -> None:
    width = max(len(c) for c in report.cases)
    print("\n==================== spel validate ====================")
    for case, r in report.cases.items():
        line = r.detail.strip().splitlines()[-1] if r.detail.strip() else ""
        print(f"  {case:<{width}}  {r.status:<17} {line[:100]}")
    failed = [r for r in report.cases.values() if r.status not in (PASS, READY)]
    for r in failed:
        if r.log:
            print(f"  {r.case}: see {r.log}")
    print(f"Report: {path}")
    print(f"{len(report.cases) - len(failed)}/{len(report.cases)} passed")


def validate(
    request: dict[str, list[str]],
    jobs: int = 4,
    freq: int = DEFAULT_FREQ,
    case_args: Optional[list[str]] = None,
    run_name: str = DEFAULT_RUN_NAME,
    reanalyze: bool = False,
    skip_create: bool = False,
    keep_instrumentation: bool = False,
    dry_run: bool = False,
) -> ValidateReport:
    from spel.scripts.analysis_cache import ANALYSIS_PICKLE, build_analysis, cache_dir
    from spel.scripts.config import (
        CASEGEN_SCRIPT,
        E3SM_SRCROOT,
        input_data_dir,
        spel_mods_dir,
        unittests_dir,
    )

    report = ValidateReport(cases={c: CaseResult(c, subs) for c, subs in request.items()})
    report_path = Path(unittests_dir) / REPORT_FILE
    log_dir = Path(unittests_dir) / ".validate-logs"
    for case in request:
        CaptureNames.for_case(case)  # fail early on unusable names
    if not dry_run and not CASEGEN_SCRIPT.exists():
        raise InstrumentError(f"casegen script {CASEGEN_SCRIPT} not found (set SPEL_CASEGEN)")

    # 1-2. analysis + unit tests
    if reanalyze or not (cache_dir() / ANALYSIS_PICKLE).is_file():
        print("Building the elm_drv analysis cache", flush=True)
        build_analysis()
    if skip_create:
        for case, r in report.cases.items():
            if not (Path(unittests_dir) / case / "fut.pkl").is_file():
                r.status, r.detail = CREATE_FAILED, "--skip-create but no fut.pkl; run without it"
    else:
        print(f"Creating {len(request)} unit test(s) with {jobs} job(s)", flush=True)
        create_cases(report, jobs, log_dir)

    # 3. instrument
    futs = []
    for case, r in report.cases.items():
        if r.status:
            continue
        try:
            futs.append(load_case(case))
        except InstrumentError as err:
            r.status, r.detail = INSTRUMENT_FAILED, str(err)
    multi = instrument_cases(futs, E3SM_SRCROOT, freq=freq, dry_run=dry_run, mods_dir=spel_mods_dir)
    for case, msg in multi.failures.items():
        report.cases[case].status, report.cases[case].detail = INSTRUMENT_FAILED, msg
    report.made_public = {str(p): v for p, v in multi.made_public.items()}
    verb = "Would instrument" if dry_run else "Instrumented"
    for case, rep in multi.reports.items():
        print(f"{verb} {case}: {rep.call_site.name} ({rep.module}, bounds={rep.bounds})")
    for path, names in multi.made_public.items():
        print(f"  public/unprotected in {path.name}: {', '.join(names)}")
    if dry_run or not multi.reports:
        if dry_run:
            for case in multi.reports:
                report.cases[case].status = READY
        write_report(report, report_path)
        print_summary(report, report_path)
        return report

    # 4. one ELM run for all cases
    try:
        try:
            rundir = run_case(CASEGEN_SCRIPT, run_name, E3SM_SRCROOT, case_args)
        except CaseRunError as err:
            # Runs occasionally fail transiently; builds don't.
            if err.build_failed or err.casedir is None:
                raise
            print(f"ELM run failed ({err}); resubmitting once", flush=True)
            rundir = resubmit_case(err.casedir)
    except InstrumentError as err:
        for case in multi.reports:
            report.cases[case].status, report.cases[case].detail = CASE_RUN_FAILED, str(err)
        rundir = None
    finally:
        # 5. leave the E3SM tree clean
        if not keep_instrumentation:
            for path in uninstrument_elm(E3SM_SRCROOT):
                print(f"Removed SPEL capture calls from {path}")

    if rundir is not None:
        report.rundir = str(rundir)
        lnd_in = Path(rundir) / "lnd_in"
        if lnd_in.is_file():
            report.lnd_in = str(lnd_in)
        for case, rep in multi.reports.items():
            dest = Path(input_data_dir) / case
            for old in dest.glob("spel-*.nc"):
                old.unlink()
            copied = collect_outputs(rundir, dest, CaptureNames(rep.tag).file_prefix)
            if copied:
                write_reference_meta(dest, case)
            r = report.cases[case]
            r.files = len(copied)
            if lnd_in.is_file():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lnd_in, dest / "lnd_in")
            if not copied:
                r.status = NOT_CALLED
                r.detail = (
                    f"no {rep.tag}.spel-*.nc in {rundir}: {rep.call_site.name} never called "
                    "it at a capture step (inactive for this configuration? see lnd_in)"
                )

    # 6. unit tests
    ready = [c for c, r in report.cases.items() if not r.status]
    if ready:
        print(f"Running {len(ready)} unit test(s) with {jobs} job(s)", flush=True)
        run_cases(report, ready, jobs)

    write_report(report, report_path)
    print_summary(report, report_path)
    return report
