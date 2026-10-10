import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from spel.srcroot_override import OPTION as SRCROOT_OPTION, apply_srcroot

# must run before anything imports spel.scripts.config (E3SM_SRCROOT is fixed at import)
apply_srcroot(sys.argv[1:])

from spel.scripts.config import unittests_dir  # noqa: E402
from spel.scripts.export_objects import unpickle_unit_test  # noqa: E402
from spel.scripts.ml_training.dataset_analysis import summarize_data  # noqa: E402
from spel.scripts.ml_training.prepare_dataset import separate_inputs_outputs  # noqa: E402
from spel.scripts.ml_training.sample_spel_output import sample  # noqa: E402
from spel.scripts.ml_training.train import train  # noqa: E402
from spel.scripts.profiler_context import profile_ctx  # noqa: E402

SPEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def create(args):
    from spel.scripts.UnitTestforELM import create_unit_test

    with profile_ctx(enabled=True, section="create") as pr:
        unit_test = create_unit_test(
            sub_names=args.subs,
            casename=args.case,
            keep=args.keep,
            db_mode=args.db_mode,
            reanalyze=args.reanalyze,
        )
    if (args.instrument or args.run_case) and not args.db_mode:
        from spel.scripts.instrument_elm import instrument_case

        instrument_case(
            str(unit_test.case_dir),
            freq=args.freq,
            run=args.run_case,
            case_args=shlex.split(args.case_args),
        )
        if args.run_case:
            _run_and_validate(unit_test.case_dir)


def analyze(args):
    from spel.scripts.analysis_cache import build_analysis

    with profile_ctx(enabled=False, section="analyze"):
        build_analysis()


def _run_and_validate(case_dir) -> None:
    """After --run-case collected ELM's data: `spel run` the unit test + validate."""
    from spel.scripts.run_fut import run_unit_test

    if code := run_unit_test(case_dir):
        raise SystemExit(code)


def instrument(args):
    from spel.scripts.instrument_elm import instrument_case, undo_instrumentation

    if args.undo:
        from spel.scripts.config import E3SM_SRCROOT

        undo_instrumentation(E3SM_SRCROOT)
        return
    report = instrument_case(
        args.case,
        freq=args.freq,
        run=args.run_case,
        case_args=shlex.split(args.case_args),
        dry_run=args.dry_run,
    )
    if args.run_case and not args.dry_run:
        _run_and_validate(report.case_dir)


def validate(args):
    from spel.scripts.validate import parse_request, validate as run_validate

    try:
        request = parse_request(args.subs, args.list_file)
    except (ValueError, OSError) as err:
        raise SystemExit(f"spel validate: error: {err}")
    report = run_validate(
        request,
        jobs=args.jobs,
        freq=args.freq,
        case_args=shlex.split(args.case_args),
        run_name=args.run_name,
        reanalyze=args.reanalyze,
        skip_create=args.skip_create,
        keep_instrumentation=args.keep_instrumentation,
        dry_run=args.dry_run,
    )
    if not report.ok and not args.dry_run:
        raise SystemExit(1)


def export(args):
    from spel.scripts.export_objects import export_table_csv

    export_table_csv(args.commit)


def diff(args):
    from spel.scripts.relerror import find_diffs
    from spel.scripts.validate_access import case_dir_for, run_validation

    if not args.test and not args.inputs:
        raise SystemExit("spel diff: give --test and/or --inputs")
    if args.inputs and not args.case:
        raise SystemExit("spel diff: --inputs requires --case")
    code = 0
    if args.test and find_diffs(refn=args.ref, compfn=args.test, var=args.var):
        code = 1
    if args.inputs:
        # the reference is the post-call state of the model itself
        code = run_validation(
            inputs_fn=args.inputs,
            outputs_fn=args.ref,
            case_dir=case_dir_for(args.case),
            constants_fn=args.constants,
        ) or code
    if code:
        raise SystemExit(code)


def sample_training(args):
    n = int(args.num_samples)
    unit_test = unpickle_unit_test(casename=args.case_name)
    input_set: set[str] = set()
    output_set: set[str] = set()
    separate_inputs_outputs(
        unit_test.subroutine_dict, inputs=input_set, outputs=output_set
    )
    sample(args.case_name, "spel-inputs", samples_per_file=n, var_name_set=input_set)
    sample(args.case_name, "spel-outputs", samples_per_file=n, var_name_set=output_set)
    summarize_data(args.case_name)
    return


def _train(args):
    data_dir = Path(unittests_dir) / f"input-data/{args.case_name}"
    train(data_dir)
    return


def run(args):
    from spel.scripts.run_fut import run_unit_test

    if args.case:
        case_dir = Path(f"{SPEL_ROOT}/unit-tests/{args.case}")
    else:
        # assume cwd
        case_dir = Path.cwd()
    # REMAINDER swallows options given after the case name
    exe_args, skip = [], False
    for a in args.exe_args:
        if skip or a == "--no-diff" or a.startswith(SRCROOT_OPTION + "="):
            skip = False
            continue
        if a == SRCROOT_OPTION:
            skip = True
            continue
        exe_args.append(a)
    validate = not args.no_diff and "--no-diff" not in args.exe_args
    if code := run_unit_test(case_dir, exe_args, validate=validate):
        raise SystemExit(code)


def repl(args):
    from IPython.terminal.embed import InteractiveShellEmbed

    banner = "SPEL (IPython) — autoreload is on"
    exit_msg = "bye"

    shell = InteractiveShellEmbed(banner1=banner, exit_msg=exit_msg)

    # Enable magics programmatically
    shell.run_line_magic("load_ext", "autoreload")
    shell.run_line_magic("autoreload", "2")
    shell.run_line_magic("xmode", "Minimal")
    shell.run_line_magic("config", "TerminalInteractiveShell.confirm_exit=False")

    from spel.scripts.fortran_parser.spel_repl import parse_line

    shell.push({"parse_line": parse_line})

    shell()  # drop into IPython loop
    # start_repl()
    return


def upload(args):
    from spel.scripts.config import scripts_dir

    SPEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mach = args.machine
    dest = args.dest
    upload_script = Path(__file__).parent / "scripts" / "upload.sh"
    subprocess.run(
        [f"{scripts_dir}/upload.sh", mach, dest],
        check=True,
        cwd=".",
    )
    return


def restore(args):
    from spel.scripts.restore_files import restore_case

    restore_case(args.case, dry_run=args.dry_run, yes=args.yes)


def config(args):
    import textwrap
    from pathlib import Path

    import spel.scripts.config as cfg

    if args.set_srcroot:
        from dotenv import set_key

        new_root = Path(args.set_srcroot).expanduser().resolve()
        cfg.LOCAL_ENV_FILE.touch(exist_ok=True)
        set_key(str(cfg.LOCAL_ENV_FILE), "SPEL_E3SM_SRCROOT", str(new_root))
        print(
            f"Set SPEL_E3SM_SRCROOT={new_root} in {cfg.LOCAL_ENV_FILE}\n"
            "(untracked -- re-run spel for this to take effect)"
        )
        return

    from spel.scripts.analysis_cache import cache_dir, list_caches, migrate_legacy_cache

    migrate_legacy_cache()

    print(textwrap.dedent(f"""
    E3SM SRCROOT: {cfg.E3SM_SRCROOT}
    ELM SRC     : {cfg.ELM_SRC}
    SHR SRC     : {cfg.SHR_SRC}
    Cache       : {cache_dir()}

    Local override file: {cfg.LOCAL_ENV_FILE}
    (change with: spel config --set-srcroot <path>, or per command with --srcroot <path>)
    """))
    caches = list_caches()
    print("elm_drv analysis caches:" if caches else "No elm_drv analysis caches yet")
    for path, meta in caches:
        e3sm = meta.get("e3sm") or {}
        print(
            f"  {meta.get('e3sm_srcroot', '?')}  "
            f"{e3sm.get('branch')}@{(e3sm.get('commit') or '?')[:10]}  "
            f"built {meta.get('created_at', '?')[:19]}  -> {path.parent.name}"
        )


def add_capture_args(parser: argparse.ArgumentParser, standalone: bool) -> None:
    if not standalone:
        parser.add_argument(
            "--instrument",
            action="store_true",
            help="Instrument ELM's call site and copy the IO modules into elm/src/main",
        )
    parser.add_argument(
        "--run-case",
        action="store_true",
        help="Create/build/run the CIME case (casegen script) and copy spel-*.nc "
        "into unit-tests/input-data/<case>" + ("" if standalone else " (implies --instrument)"),
    )
    parser.add_argument(
        "--freq", type=int, default=9, help="Capture every N model steps (default: 9)"
    )
    parser.add_argument(
        "--case-args",
        default="",
        help="Extra casegen script options, e.g. \"--stop-n 10 --stop-option ndays --keep\"",
    )


def main():
    desc = (
        "spel create: "
        "   Given input of subroutine names,"
        "   SPEL analyzes all dependencies related"
        "   to the subroutines"
        "spel export: "
        "   Given a casename, take pkl files and create database csvs"
        "spel diff: "
        "   Input two netcdf files to compare with scripts.relerror"
    )
    parser = argparse.ArgumentParser(prog="spel", description=desc)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Arg parser for spel create
    create_parser = subparsers.add_parser("create", help="Run the create command")
    create_parser.add_argument(
        "-s",
        nargs="+",
        required=True,
        dest="subs",
        help="Specify subroutines",
    )
    create_parser.add_argument(
        "-c",
        required=False,
        dest="case",
        default="fut",
        help="Specify case name",
    )
    create_parser.add_argument(
        "-u",
        required=False,
        dest="keep",
        action="store_true",
        help="Re-use existing case",
    )
    create_parser.add_argument(
        "--db",
        required=False,
        dest="db_mode",
        action="store_true",
        help="Don't make Unit Test",
    )
    create_parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="Rebuild the cached elm_drv analysis first (e.g. after switching "
        "E3SM/SPEL branches or editing ELM); otherwise it is reused",
    )
    add_capture_args(create_parser, standalone=False)
    create_parser.set_defaults(func=create)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="(Re)build the cached elm_drv analysis that `spel create` extracts from",
    )
    analyze_parser.set_defaults(func=analyze)

    # Parser for 'spel instrument'
    instrument_parser = subparsers.add_parser(
        "instrument",
        help="Instrument ELM to dump reference data for a unit-test case",
    )
    instrument_parser.add_argument(
        "case", nargs="?", default="fut",
        help="unit-test case (name or path); uses the analysis `spel create` saved in <case>/fut.pkl",
    )
    instrument_parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change"
    )
    instrument_parser.add_argument(
        "--undo", action="store_true", help="Remove SPEL capture calls from ELM and revert its public/unprotected edits"
    )
    add_capture_args(instrument_parser, standalone=True)
    instrument_parser.set_defaults(func=instrument)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Create unit tests for many routines, capture their reference data "
        "with one instrumented ELM run, then run and validate them all",
    )
    validate_parser.add_argument(
        "-s", nargs="+", dest="subs", default=[],
        help="routines to test, one case each (named after the routine)",
    )
    validate_parser.add_argument(
        "--list", dest="list_file", type=Path,
        help="file with one case per line: `case: sub1 sub2` or `sub` (# comments)",
    )
    validate_parser.add_argument(
        "-j", "--jobs", type=int, default=4,
        help="parallel `spel create`/`spel run` processes (default: 4)",
    )
    validate_parser.add_argument(
        "--run-name", default="spel-validate", help="CIME case name (default: spel-validate)"
    )
    validate_parser.add_argument(
        "--reanalyze", action="store_true", help="rebuild the cached elm_drv analysis first"
    )
    validate_parser.add_argument(
        "--skip-create", action="store_true", help="reuse existing unit-test cases"
    )
    validate_parser.add_argument(
        "--keep-instrumentation", action="store_true",
        help="leave the capture calls and IO modules in ELM after the run",
    )
    validate_parser.add_argument(
        "--dry-run", action="store_true",
        help="create the cases and report the instrumentation, but don't change ELM",
    )
    validate_parser.add_argument(
        "--freq", type=int, default=9, help="Capture every N model steps (default: 9)"
    )
    validate_parser.add_argument(
        "--case-args", default="",
        help="Extra casegen script options, e.g. \"--stop-n 10 --stop-option ndays\"",
    )
    validate_parser.set_defaults(func=validate)

    # Parser for 'spel export'
    export_parser = subparsers.add_parser("export", help="Run the export command")
    export_parser.add_argument(
        "-c",
        required=True,
        dest="commit",
        help="Casename of the pickled unit test (spel/scripts/fut_<casename>.pkl)",
    )
    export_parser.set_defaults(func=export)

    # Parser for 'spel diff'
    diff_parser = subparsers.add_parser("diff", help="Run diff command")
    diff_parser.add_argument(
        "--ref",
        required=True,
        dest="ref",
        help="reference netcdf file (post-call state, e.g. spel-outputs0001.nc)",
    )
    diff_parser.add_argument(
        "--test",
        required=False,
        dest="test",
        help="test netcdf file",
    )
    diff_parser.add_argument(
        "-v",
        required=False,
        dest="var",
        help="Optional: only report variable var",
    )
    diff_parser.add_argument(
        "--inputs",
        required=False,
        dest="inputs",
        help=(
            "pre-call netcdf file (e.g. spel-inputs0001.nc). Validates SPEL's "
            "static analysis against --ref: inputs must not change (fails), "
            "outputs that never change are flagged"
        ),
    )
    diff_parser.add_argument(
        "--case",
        required=False,
        dest="case",
        help="unit-test case name or directory holding spel_access.json (with --inputs)",
    )
    diff_parser.add_argument(
        "--constants",
        required=False,
        dest="constants",
        help=(
            "netcdf file with the run's namelist values used to evaluate namelist "
            "guards (default: spel-constants file next to --inputs)"
        ),
    )
    diff_parser.set_defaults(func=diff)

    # Parser for 'spel run'
    run_parser = subparsers.add_parser(
        "run",
        help="compile and run FUT, then diff against ELM's outputs and validate the access analysis",
    )
    run_parser.add_argument(
        "--no-diff",
        action="store_true",
        help="skip the bit-for-bit diff and access validation",
    )
    run_parser.add_argument("case", nargs="?", help="Unit test executable name")
    # run_parser.add_argument("-c", required=False, dest="case", help="FUT name")
    run_parser.add_argument(
        "exe_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the unit test executable",
    )
    run_parser.set_defaults(func=run)

    upload_parser = subparsers.add_parser(
        "upload", help="rsync netCDF-Interface files <mach> <dest>"
    )
    upload_parser.add_argument("machine", help="remote machine")
    upload_parser.add_argument("dest", help="path")
    upload_parser.set_defaults(func=upload)

    repl_parser = subparsers.add_parser("repl", help="Start repl ")
    repl_parser.set_defaults(func=repl)

    # Parser for 'spel sample'
    sample_parser = subparsers.add_parser(
        "sample",
        help="Randomly sample nsteps from spel input/outputs files for training",
    )
    sample_parser.add_argument(
        "-n",
        required=True,
        dest="num_samples",
        help="number of samples per file",
    )
    sample_parser.add_argument(
        "-c",
        required=True,
        dest="case_name",
        help="case name matching existing Pickled Functional Unit Test",
    )
    sample_parser.set_defaults(func=sample_training)

    train_parser = subparsers.add_parser("train", help="train nn")
    train_parser.add_argument(
        "-c",
        required=True,
        dest="case_name",
        help="case name matching existing Pickled Functional Unit Test",
    )
    train_parser.set_defaults(func=_train)

    cfg_parser = subparsers.add_parser(
        "config", help="Display or adjust config for SPEL"
    )
    cfg_parser.add_argument(
        "--set-srcroot",
        dest="set_srcroot",
        default=None,
        help="Persist E3SM_SRCROOT to the untracked local config file "
        "(.spel.env) instead of editing config.py",
    )
    cfg_parser.set_defaults(func=config)

    # Parser for 'spel restore'
    restore_parser = subparsers.add_parser(
        "restore",
        help="Remove the '!#py ' token from files changed in a unit-test "
        "case and copy them back into E3SM_SRCROOT",
    )
    restore_parser.add_argument(
        "-c",
        required=True,
        dest="case",
        help="Unit-test case name (directory under unit-tests/)",
    )
    restore_parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Only show which files would be restored, don't write anything",
    )
    restore_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        dest="yes",
        help="Don't prompt for confirmation before copying files",
    )
    restore_parser.set_defaults(func=restore)

    # parsed for --help/validation only: apply_srcroot() already applied it
    srcroot_help = (
        "E3SM checkout to use for this command (default: SPEL_E3SM_SRCROOT / "
        ".spel.env, see `spel config`). Each checkout has its own elm_drv analysis cache"
    )
    for p in (parser, *subparsers.choices.values()):
        p.add_argument(
            SRCROOT_OPTION, metavar="DIR", default=argparse.SUPPRESS, help=srcroot_help
        )

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as err:
        from spel.scripts.instrument_elm import InstrumentError

        if isinstance(err, InstrumentError):
            parser.exit(1, f"spel {args.command}: error: {err}\n")
        raise


if __name__ == "__main__":
    main()
