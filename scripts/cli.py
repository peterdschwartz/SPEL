import argparse
import os
import subprocess

from scripts.fortran_parser.spel_repl import parse_line
from scripts.profiler_context import profile_ctx

SPEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def create(args):
    from scripts.UnitTestforELM import create_unit_test

    with profile_ctx(enabled=True, section="create") as pr:
        create_unit_test(
            sub_names=args.subs,
            casename=args.case,
            keep=args.keep,
            db_mode=args.db_mode,
        )


def export(args):
    from scripts.export_objects import export_table_csv

    export_table_csv(args.commit)


def diff(args):
    from scripts.relerror import find_diffs

    find_diffs(refn=args.ref, compfn=args.test, var=args.var)
    return


def run(args):
    if args.case:
        unit_test = f"{SPEL_ROOT}/unit-tests/{args.case}"
    else:
        # assme cwd
        unit_test = "."
    # cmake_path = f"{unit_test}/check_config.sh"
    # test_exe = f"{unit_test}/build/elmtest"

    # Run config check
    subprocess.run(["./check_config.sh"], check=True, cwd=unit_test)

    # Run the test executable with extra args
    subprocess.run(["./build/elmtest", *args.exe_args], check=True, cwd=unit_test)


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

    from scripts.fortran_parser.spel_repl import parse_line

    shell.push({"parse_line": parse_line})

    shell()  # drop into IPython loop
    # start_repl()
    return


def upload(args):
    SPEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mach = args.machine
    dest = args.dest
    subprocess.run(
        [f"{SPEL_ROOT}/scripts/upload.sh", mach, dest],
        check=True,
        cwd=".",
    )
    return


def main():
    desc = (
        "spel create: "
        "   Given input of subroutine names,"
        "   SPEL analyzes all dependencies related"
        "   to the subroutines"
        "spel export: "
        "   Given commit number, take pkl files and create database csvs"
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
    create_parser.set_defaults(func=create)

    # Parser for 'spel export'
    export_parser = subparsers.add_parser("export", help="Run the export command")
    export_parser.add_argument(
        "-c",
        required=True,
        dest="commit",
        help="Specify commit value ",
    )
    export_parser.set_defaults(func=export)

    # Parser for 'spel diff'
    diff_parser = subparsers.add_parser("diff", help="Run diff command")
    diff_parser.add_argument(
        "--ref",
        required=True,
        dest="ref",
        help="reference netcdf file",
    )
    diff_parser.add_argument(
        "--test",
        required=True,
        dest="test",
        help="test netcdf file",
    )
    diff_parser.add_argument(
        "-v",
        required=False,
        dest="var",
        help="Optional: only report variable var",
    )
    diff_parser.set_defaults(func=diff)

    # Parser for 'spel run'
    run_parser = subparsers.add_parser("run", help="compile and run FUT")
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
