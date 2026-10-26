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
from spel.scripts.profiler_context import profile_ctx  # noqa: E402

SPEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def create(args):
    from spel.scripts.UnitTestforELM import create_unit_test

    import spel.scripts.config as cfg

    cfg.options.device_data = args.device_data
    with profile_ctx(enabled=True, section="create") as pr:
        unit_test = create_unit_test(
            sub_names=args.subs,
            casename=args.case,
            keep=args.keep,
            db_mode=args.db_mode,
            reanalyze=args.reanalyze,
            capture_for=args.capture_for,
        )
    if (args.instrument or args.run_case) and not args.db_mode:
        from spel.scripts.instrument_elm import instrument_case

        instrument_case(
            str(unit_test.case_dir),
            freq=args.freq,
            run=args.run_case,
            case_args=shlex.split(args.case_args),
            subset=_subset_spec(args),
        )
        if args.run_case:
            _run_and_validate(unit_test.case_dir)


def device_data(args):
    from pathlib import Path

    from spel.scripts.analysis_cache import load_analysis
    from spel.scripts.config import E3SM_SRCROOT
    from spel.transforms.device.generate import DeviceDataError, apply, generate

    fut = load_analysis(reanalyze=args.reanalyze)
    srcroot = Path(E3SM_SRCROOT)
    if args.print_module:
        texts, p = generate(fut, srcroot)
        for name, text in texts.items():
            print(f"! ===== {name}")
            print(text)
        for s in p.skipped:
            print("! skipped:", s)
        return
    try:
        p, report = apply(fut, srcroot, out_dir=Path(args.out_dir) if args.out_dir else None, dry_run=args.dry_run)
    except DeviceDataError as exc:
        sys.exit(f"device-data: {exc}")
    for line in report:
        print(("would write " if args.dry_run else "") + line)
    for s in p.skipped:
        print("skipped:", s)


def compact(args):
    args.type_given = "--type" in sys.argv
    from spel.scripts.analysis_cache import load_analysis
    from spel.transforms.compact.inventory import (
        FLUX_TYPES,
        build_inventory,
        detail,
        summary,
        write_json,
    )

    fut = load_analysis(reanalyze=args.reanalyze)
    if args.action == "rewrite":
        return compact_rewrite(fut, args)
    instances = args.instances.split(",") if args.instances else None
    reports, tracer = build_inventory(fut, types=FLUX_TYPES, instances=instances)
    print(summary(reports))
    for r in reports:
        if r.field in (args.field or []) or r.verdict in (args.verdict or []):
            print()
            print(detail(r))
    if args.output:
        write_json(reports, Path(args.output))
        print(f"\nwrote {args.output}")
    if tracer.unwalked:
        print(f"\nnot followed (no analysis): {', '.join(sorted(tracer.unwalked))}")


def compact_rewrite(fut, args):
    from spel.scripts.analysis_cache import git_state
    from spel.scripts.config import E3SM_SRCROOT
    from spel.transforms.compact.rewrite import RewriteError, Rewriter, manifest
    from spel.transforms.compact.select import select_fields

    from spel.transforms.compact.holders import family_base, nutrient_manifest, role_types

    types = list(args.type)
    nm = nutrient_manifest(fut)
    if nm is not None and not getattr(args, "type_given", False):
        # nutrient-transformed sources: the shared fields live on the family base types
        mapped = []
        for dtype in types:
            base = family_base(nm, dtype)
            if base and base not in mapped:
                mapped.append(base)
            elif not base and dtype not in mapped:
                mapped.append(dtype)
        if getattr(args, "roles", False):
            mapped += [r for r in role_types(nm, types) if r not in mapped]
        print(f"types: {', '.join(mapped)} (from .spel-nutrient.json)")
        types = mapped
    if getattr(args, "no_map_sites", False):
        import spel.transforms.compact.select as _sel

        _sel.MAP_SITES = False
    sels = []
    for dtype in [] if args.loops_only else types:
        sel = select_fields(fut, dtype)
        if getattr(args, "exclude_fields", None):
            for comp in args.exclude_fields.split(","):
                if comp in sel.components:
                    sel.components.pop(comp)
                    sel.history.pop(comp, None)
                    sel.levels.pop(comp, None)
                    sel.restart.pop(comp, None)
                    sel.excluded[comp] = "--exclude-fields"
            drop = set(args.exclude_fields.split(","))
            sel.sites = [st for st in sel.sites if st.components - drop]
        if getattr(args, "fields", None):
            keep = set(args.fields.split(","))
            for comp in list(sel.components):
                if comp not in keep:
                    sel.components.pop(comp)
                    sel.history.pop(comp, None)
                    sel.levels.pop(comp, None)
                    sel.excluded[comp] = "not in --fields"
            sel.sites = [st for st in sel.sites if st.components & keep]
            if not sel.components:
                print(f"{dtype}: no selected field in --fields; skipped")
                continue
        sels.append(sel)
        print(f"{dtype}: {len(sel.components)} fields, {len(sel.sites)} access sites")
        nested = [c for c, why in sel.excluded.items() if why.startswith("lives on")]
        if nested:
            print(f"  {len(nested)} shared fields live on an embedded holder; pass --type <that type> (see --excluded)")
        for comp, zc in sel.components.items():
            print(f"  {comp:32s} zeroed: {zc:20s} history: {sel.history[comp]}")
        if args.excluded:
            for comp, why in sorted(sel.excluded.items()):
                print(f"  excluded {comp:32s} {why}")
        else:
            print(f"  ({len(sel.excluded)} fields excluded; --excluded lists them)")
        if sel.untraced:
            print("  uses outside the elm_drv trace (text search; fields with a use of this type are excluded):")
            for u in sel.untraced:
                print(f"    {u}")
    rewriter = Rewriter(fut, sels)
    from spel.transforms.compact.inventory import FLUX_TYPES

    try:
        rewriter.build(loop_types=FLUX_TYPES if args.setvalues_loops else ())
    except RewriteError as err:
        raise SystemExit(f"compact rewrite: {err}")
    for note in rewriter.notes:
        print(note)
    if args.diff:
        print(rewriter.diff())
    if args.dry_run:
        return
    spel_commit = git_state(Path(SPEL_ROOT)).get("commit")
    e3sm = git_state(Path(E3SM_SRCROOT))
    try:
        written = rewriter.apply(
            manifest(rewriter, spel_commit, e3sm),
            output=Path(args.output) if args.output else None,
            replace=args.replace,
        )
    except RewriteError as err:
        raise SystemExit(f"compact rewrite: {err}")
    print("wrote:")
    for path in written:
        print(f"  {path}")


def nutrient(args):
    from spel.scripts.analysis_cache import load_analysis
    from spel.transforms.nutrient.common import detail, find_families, summary, to_json

    fut = load_analysis(reanalyze=args.reanalyze)
    families = find_families(fut.type_dict)
    if args.action in ("extract", "groups"):
        return nutrient_extract(fut, families, args)
    if args.action == "merge":
        return nutrient_merge(fut, families, args)
    if args.action == "species":
        return nutrient_species(fut, families, args)
    if args.action in ("history", "restart"):
        return nutrient_history(fut, families, args)
    if args.action == "procedures":
        from spel.transforms.nutrient import procedures

        reports = [
            r
            for fam in families
            if not args.family or fam.name in args.family
            for r in procedures.compare_family(fam, fut.subroutine_dict, fut.type_dict)
        ]
        print(procedures.summary(reports))
        for r in reports:
            if r.binding in (args.procedure or []):
                print()
                print(procedures.detail(r, kinds=tuple(args.kinds)))
        return
    print(summary(families))
    for fam in families:
        if fam.name in (args.family or []):
            print()
            print(detail(fam))
    if args.output:
        to_json(families, Path(args.output))
        print(f"\nwrote {args.output}")


def split(args):
    from spel.scripts.analysis_cache import load_analysis
    from spel.scripts.config import E3SM_SRCROOT
    from spel.transforms.split.modules import PLANS, SplitError, Splitter

    mods, plan = PLANS[args.plan]
    fut = load_analysis(reanalyze=args.reanalyze)
    try:
        sp = Splitter(fut, mods, plan, Path(E3SM_SRCROOT))
        sp.build()
    except SplitError as err:
        raise SystemExit(f"split: {err}")
    for note in sp.notes:
        print(note)
    print(sp.report())
    if args.diff:
        print(sp.diff())
    if args.dry_run:
        return
    for path in sp.apply():
        print(f"  wrote {path}")


def nutrient_extract(fut, families, args):
    import difflib

    from spel.scripts.analysis_cache import git_state
    from spel.scripts.config import E3SM_SRCROOT, ELM_SRC
    from spel.transforms.nutrient.extract import ExtractError, Extractor, core_edits, manifest

    if args.family:
        families = [f for f in families if f.name in args.family]
    if args.action == "groups":
        from spel.transforms.nutrient.groups import group_edits

        try:
            edits, skipped = group_edits(families), {}
        except ExtractError as err:
            raise SystemExit(f"nutrient groups: {err}")
        key = "groups"
    else:
        edits, skipped = core_edits(families)
        key = "families"
    ex = Extractor(edits, fut.type_dict, Path(ELM_SRC), skipped)
    try:
        ex.build()
    except ExtractError as err:
        raise SystemExit(f"nutrient extract: {err}")
    print(ex.report())
    if args.diff:
        for path, lines in ex.files.items():
            old = Path(path).read_text(errors="replace").split("\n")
            sys.stdout.writelines(
                l + "\n" for l in difflib.unified_diff(old, lines, str(path), str(path), lineterm="")
            )
    if args.dry_run:
        return
    spel_commit = git_state(Path(SPEL_ROOT)).get("commit")
    for path in ex.write(manifest(ex, spel_commit, git_state(Path(E3SM_SRCROOT)), key)):
        print(f"  wrote {path}")


def nutrient_merge(fut, families, args):
    import difflib

    from spel.transforms.nutrient.merge import DEFAULT_BINDINGS, Merger

    if args.family:
        families = [f for f in families if f.name in args.family]
    m = Merger(families, fut, bindings=args.procedure or DEFAULT_BINDINGS)
    m.build()
    files = m.apply()
    print(m.report())
    if args.diff:
        for path, lines in files.items():
            old = m.lines(path)
            sys.stdout.writelines(
                l + "\n" for l in difflib.unified_diff(old, lines, str(path), str(path), lineterm="")
            )
    if args.dry_run:
        return
    for path, lines in files.items():
        Path(path).write_text("\n".join(lines))
        print(f"  wrote {path}")


def nutrient_history(fut, families, args):
    import difflib

    from spel.transforms.nutrient.history import HISTORY, RESTART, HistoryMerger

    if args.family:
        families = [f for f in families if f.name in args.family]
    hm = HistoryMerger(families, fut, RESTART if args.action == "restart" else HISTORY)
    hm.build()
    files = hm.apply()
    print(hm.report())
    if args.diff:
        for path, lines in files.items():
            old = Path(path).read_text(errors="replace").split("\n")
            sys.stdout.writelines(
                l + "\n" for l in difflib.unified_diff(old, lines, str(path), str(path), lineterm="")
            )
    if args.dry_run:
        return
    for path, lines in files.items():
        Path(path).write_text("\n".join(lines))
        print(f"  wrote {path}")


def nutrient_species(fut, families, args):
    import difflib

    from spel.scripts.config import ELM_SRC
    from spel.transforms.nutrient.extract import ExtractError
    from spel.transforms.nutrient.species import SpeciesEditor

    ed = SpeciesEditor(families, fut, Path(ELM_SRC))
    try:
        ed.build()
    except ExtractError as err:
        raise SystemExit(f"nutrient species: {err}")
    for note in ed.notes:
        print(note)
    changed = {p: l for p, l in ed.files.items() if "\n".join(l) != p.read_text(errors="replace")}
    if args.diff:
        for path, lines in changed.items():
            old = path.read_text(errors="replace").split("\n")
            sys.stdout.writelines(
                l + "\n" for l in difflib.unified_diff(old, lines, str(path), str(path), lineterm="")
            )
    if args.dry_run:
        return
    for path, lines in changed.items():
        path.write_text("\n".join(lines))
        print(f"  wrote {path}")


def acc2omp(args):
    from spel.scripts.analysis_cache import load_analysis
    from spel.transforms.acc_to_omp import TranslationError, case_modules, translate_modules

    case_dir = Path(f"{SPEL_ROOT}/unit-tests/{args.case}") if args.case else None
    if case_dir is not None and not case_dir.is_dir():
        raise SystemExit(f"no unit-test case {case_dir}")
    fut = load_analysis(reanalyze=args.reanalyze)
    modules = args.modules or (case_modules(fut, case_dir) if case_dir else [])
    if not modules:
        raise SystemExit("acc2omp: name the modules to translate, or give --case")

    failed = False
    for tr in translate_modules(fut, modules):
        print(tr.report())
        failed |= any(n.level == "error" for n in tr.notes)
        if case_dir is not None:
            dest = case_dir / tr.path.name
            target = dest.read_text().splitlines(keepends=True)
        else:
            dest = Path(args.output) / tr.path.name if args.output else None
            target = None
        try:
            new_lines = tr.apply(target)
        except TranslationError as err:
            if target is not None and not any("!$acc" in l.lower() for l in target):
                print(f"  {dest} has no OpenACC directives left (already translated?)")
            else:
                print(f"  ERROR {err}")
            failed = True
            continue
        if args.diff:
            print(tr.diff(target))
        if dest is not None and not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("".join(new_lines))
            print(f"  wrote {dest}")
    if case_dir is not None and not args.dry_run:
        print(f"Build with OpenMP: SPEL_ACC=OPENMP OMP_NUM_THREADS=<n> spel run {args.case}")
    if failed:
        raise SystemExit(1)


def accplan(args):
    from spel.scripts.analysis_cache import load_analysis
    from spel.scripts.record_access import AccessMapper
    from spel.transforms.acc_plan import report

    fut = load_analysis(reanalyze=args.reanalyze)
    mapper = AccessMapper(fut.subroutine_dict)
    mods = {m.lower() for m in args.modules}
    subs = [s for s in fut.subroutine_dict.values() if s.module in mods and (not args.routine or s.name in args.routine)]
    if not subs:
        raise SystemExit(f"accplan: no analyzed routines in {', '.join(sorted(mods))}")
    for p in report(subs, mapper, min_lines=args.min_lines):
        print(p)
        print()


def analyze(args):
    from spel.scripts.analysis_cache import build_analysis

    with profile_ctx(enabled=False, section="analyze"):
        build_analysis()


def _run_and_validate(case_dir) -> None:
    """After --run-case collected ELM's data: `spel run` the unit test + validate."""
    from spel.scripts.run_fut import run_unit_test

    if code := run_unit_test(case_dir):
        raise SystemExit(code)


def _subset_spec(args):
    """The --subset/--subset-max gridcell subset, or None."""
    from spel.scripts.subset import SubsetError, SubsetSpec

    if not args.subset:
        if args.subset_max is not None:
            raise SystemExit("--subset-max needs --subset")
        return None
    try:
        return SubsetSpec.parse(args.subset, args.subset_max)
    except SubsetError as err:
        raise SystemExit(f"--subset: {err}")


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
        subset=_subset_spec(args),
    )
    if args.run_case and not args.dry_run:
        _run_and_validate(report.case_dir)


def validate(args):
    from spel.scripts.validate import parse_request, parse_subsets, validate as run_validate

    try:
        request = parse_request(args.subs, args.list_file)
        subsets = parse_subsets(request, args.list_file, args.subset, args.subset_max)
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
        subsets=subsets,
        test_env=args.test_env,
    )
    if not report.ok and not args.dry_run:
        raise SystemExit(1)


def subset(args):
    from spel.scripts.config import REFERENCE_META_FILENAME, input_data_dir
    from spel.scripts.subset import SubsetError, subset_files

    src, dst = Path(input_data_dir) / args.src, Path(input_data_dir) / args.dst
    if not src.is_dir():
        raise SystemExit(f"spel subset: no reference data in {src}")
    if dst.resolve() == src.resolve():
        raise SystemExit("spel subset: the destination must differ from the source")
    try:
        gridcells = sorted({int(g) for g in args.gridcells.split(",")})
        files = subset_files(src, dst, gridcells)
    except (SubsetError, ValueError) as err:
        raise SystemExit(f"spel subset: {err}")
    meta_src = src / REFERENCE_META_FILENAME
    if meta_src.is_file():
        import json

        meta = json.loads(meta_src.read_text())
        meta["subset"] = {"gridcells": gridcells, "from": args.src}
        (dst / REFERENCE_META_FILENAME).write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(files)} files with gridcells {gridcells} of {src} to {dst}")


def export(args):
    """Write the analysis (elm_drv cache, or a case pickle) as database CSVs."""
    import json
    import logging
    from pathlib import Path

    from spel.scripts.analysis_cache import ANALYSIS_PICKLE, META_FILE, cache_dir, current_meta
    from spel.scripts.config import spel_output_dir, unittests_dir
    from spel.scripts.db_export.snapshot import default_label, export_snapshot
    from spel.scripts.export_objects import _load_pickle, unpickle_case
    from spel.scripts.logging_configs import get_logger

    logger = get_logger("SPEL", level=logging.INFO)
    roots = None
    if args.case:
        case_dir = unittests_dir / args.case
        fut = unpickle_case(case_dir)
        meta = current_meta() | {"roots": list(fut.primary_subroutines)}
        roots = list(fut.primary_subroutines)
    else:
        cache = cache_dir()
        if not (cache / ANALYSIS_PICKLE).is_file():
            raise SystemExit(f"No elm_drv analysis cache at {cache}; run `spel analyze` first")
        meta = json.loads((cache / META_FILE).read_text())
        fut = _load_pickle(cache / ANALYSIS_PICKLE, srcroot=Path(meta["e3sm_srcroot"]))
    label = args.label or default_label(meta) + (f"+{args.case}" if args.case else "")
    out_dir = Path(args.output) if args.output else spel_output_dir / "db-export" / label
    export_snapshot(fut, meta, out_dir, label=label, case=args.case, roots=roots, logger=logger)
    print(f"Snapshot '{label}' written to {out_dir}")
    print(f"Load it with: python spel/db/manage.py update_all_data {out_dir}")


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
    from spel.scripts.ml_training.dataset_analysis import summarize_data
    from spel.scripts.ml_training.prepare_dataset import separate_inputs_outputs
    from spel.scripts.ml_training.sample_spel_output import sample

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
    from spel.scripts.ml_training.train import train

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


def profile(args):
    import shlex

    from spel.scripts.run_fut import profile_unit_test

    case_dir = Path(f"{SPEL_ROOT}/unit-tests/{args.case}") if args.case else Path.cwd()
    exe_args = [args.nsets] if args.nsets else []
    profile_unit_test(
        case_dir,
        exe_args,
        tool=args.tool,
        steps=args.steps,
        output=Path(args.output) if args.output else None,
        stats=args.stats,
        tool_args=shlex.split(args.tool_args),
    )


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

    from spel.machines import ENV_VAR as MACHINE_VAR, MachineError, detect_machine

    try:
        machine = detect_machine()
        machine_line = (
            f"{machine.name} (toolchains: {', '.join(machine.toolchains)}; default {machine.default_fc})"
            if machine
            else f"none (current shell environment; {MACHINE_VAR}=<name> or `spel machine init`)"
        )
    except MachineError as err:
        machine_line = f"ERROR {err}"
    print(textwrap.dedent(f"""
    E3SM SRCROOT: {cfg.E3SM_SRCROOT}
    ELM SRC     : {cfg.ELM_SRC}
    SHR SRC     : {cfg.SHR_SRC}
    Cache       : {cache_dir()}
    Machine     : {machine_line}
                  (see `spel machine`)

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
    add_subset_args(parser)


def add_subset_args(parser: argparse.ArgumentParser) -> None:
    from spel.scripts.subset import DEFAULT_MAX_CELLS, PRESETS

    parser.add_argument(
        "--subset",
        metavar="MASK",
        help="capture only the gridcells matching MASK: "
        f"{', '.join(PRESETS)} (comma-separated: any of them), or "
        "'<level>: <Fortran expression>' with level gridcell/topounit/landunit/column/patch "
        "and index g/t/l/c/p, e.g. 'landunit: lun_pp%%itype(l) == istsoil'",
    )
    parser.add_argument(
        "--subset-max",
        type=int,
        metavar="N",
        help=f"keep at most N matching gridcells, evenly spaced (default: {DEFAULT_MAX_CELLS})",
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
        "--capture-for",
        action="append",
        default=[],
        metavar="CASE",
        help="also capture the fields and constants unit-test CASE reads (e.g. the same "
        "routine created from a development branch), so CASE can run on this case's data",
    )
    create_parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="Rebuild the cached elm_drv analysis first (e.g. after switching "
        "E3SM/SPEL branches or editing ELM); otherwise it is reused",
    )
    create_parser.add_argument(
        "--device-data",
        action="store_true",
        help="move the data with the generated device data layer (DeepCopy*Mod, ElmDeviceDataMod, as "
        "`spel device-data` writes for the full model) instead of the harness's SpelAccDataMod deep copy",
    )
    add_capture_args(create_parser, standalone=False)
    create_parser.set_defaults(func=create)

    acc2omp_parser = subparsers.add_parser(
        "acc2omp",
        help="translate modules' OpenACC directives to OpenMP (clauses from SPEL's analysis)",
    )
    acc2omp_parser.add_argument(
        "modules", nargs="*", help="modules to translate (default with --case: those with compute loops)"
    )
    acc2omp_parser.add_argument(
        "--case", help="rewrite the module copies in unit-tests/<case> in place"
    )
    acc2omp_parser.add_argument(
        "-o", "--output", help="write translated copies of the analyzed sources to this directory"
    )
    acc2omp_parser.add_argument("--diff", action="store_true", help="print the changes")
    acc2omp_parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true", help="report only, write nothing"
    )
    acc2omp_parser.add_argument(
        "--reanalyze", action="store_true", help="rebuild the cached elm_drv analysis first"
    )
    acc2omp_parser.set_defaults(func=acc2omp)

    accplan_parser = subparsers.add_parser(
        "accplan",
        help="propose an OpenACC loop structure (collapse / column-parallel / interchange / scatter) per loop, from the analysis",
    )
    accplan_parser.add_argument("modules", nargs="+", help="modules to report")
    accplan_parser.add_argument("--routine", nargs="*", help="only these routines")
    accplan_parser.add_argument("--min-lines", dest="min_lines", type=int, default=2, help="skip loops shorter than this")
    accplan_parser.add_argument("--reanalyze", action="store_true", help="rebuild the cached elm_drv analysis first")
    accplan_parser.set_defaults(func=accplan)

    compact_parser = subparsers.add_parser(
        "compact",
        help="inventory of the per-step zeroed flux fields for a compact subgrid layout",
    )
    compact_parser.add_argument(
        "action",
        choices=["inventory", "rewrite"],
        help="inventory: report the fields; rewrite: give the selected fields of --type compact storage in the E3SM sources",
    )
    compact_parser.add_argument(
        "--type",
        nargs="+",
        default=["vegetation_carbon_flux", "vegetation_nitrogen_flux", "vegetation_phosphorus_flux",
                 "column_carbon_flux", "column_nitrogen_flux", "column_phosphorus_flux"],
        help="rewrite: the derived types (default: the patch- and column-level C/N/P flux types; on "
        "nutrient-transformed sources their family base types, vegetation_nutrient_flux and column_nutrient_flux)",
    )
    compact_parser.add_argument(
        "--roles",
        action="store_true",
        help="rewrite: on nutrient-transformed sources also the role holders (mineral/uptake/retrans) of the default types",
    )
    compact_parser.add_argument("--diff", action="store_true", help="rewrite: print the changes")
    compact_parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true", help="rewrite: report only, write nothing"
    )
    compact_parser.add_argument("--excluded", action="store_true", help="rewrite: list the excluded fields")
    compact_parser.add_argument(
        "--loops-only",
        dest="loops_only",
        action="store_true",
        help="rewrite: only the SetValues loops, no compact layout (to measure the two separately)",
    )
    compact_parser.add_argument(
        "--setvalues-loops",
        dest="setvalues_loops",
        action="store_true",
        default=False,
        help="rewrite: also turn the flux types' blocked SetValues loops (associate idx => filter(fi:...)) "
        "into do concurrent loops, for all six types. Off by default: 7%% slower CNZero with Intel on the "
        "stride-8 case (2026-10-24); it helped nvfortran",
    )
    compact_parser.add_argument(
        "--no-setvalues-loops", dest="setvalues_loops", action="store_false", help="(the default)"
    )
    compact_parser.add_argument(
        "--exclude-fields", dest="exclude_fields", default="",
        help="rewrite: drop these components (comma-separated) from every selection, for bisecting",
    )
    compact_parser.add_argument(
        "--fields", default="",
        help="rewrite: keep only these components (comma-separated) of every selection, for bisecting",
    )
    compact_parser.add_argument(
        "--no-map-sites", dest="no_map_sites", action="store_true",
        help="rewrite: only filter-gathered sites (no x(compact_cidx_p(p)) for sub-filter or pfti-mapped members)",
    )
    compact_parser.add_argument(
        "--replace",
        action="store_true",
        help="rewrite -o: remove the files of the earlier rewrite in the output directory (by its manifest) first",
    )
    compact_parser.add_argument(
        "--instances",
        default="veg_cf,veg_nf,veg_pf,col_cf,col_nf,col_pf",
        help="comma-separated instances (default: the C12 flux instances)",
    )
    compact_parser.add_argument("--field", nargs="*", help="print the details of these instance%%fields")
    compact_parser.add_argument(
        "--verdict", nargs="*", help="print the details of fields with these verdicts (e.g. blocked)"
    )
    compact_parser.add_argument(
        "-o",
        "--output",
        help="inventory: write the full inventory as JSON; rewrite: write the rewritten files "
        "(by basename, e.g. a case's SourceMods/src.elm) here instead of in place",
    )
    compact_parser.add_argument(
        "--reanalyze", action="store_true", help="rebuild the cached elm_drv analysis first"
    )
    compact_parser.set_defaults(func=compact)

    device_parser = subparsers.add_parser(
        "device-data",
        help="generate ElmDeviceDataMod and the DeepCopy*Mod group modules (device init, first-step deep copy, whole-state updates) for a full OpenACC run",
    )
    device_parser.add_argument("--dry-run", action="store_true", help="report the plan and the edits, write nothing")
    device_parser.add_argument("--print", dest="print_module", action="store_true", help="print the generated modules")
    device_parser.add_argument(
        "-o", "--out-dir", default=None,
        help="write the modules and the edited driver files here (a case's SourceMods/src.elm) instead of in place",
    )
    device_parser.add_argument("--reanalyze", action="store_true", help="rebuild the cached elm_drv analysis first")
    device_parser.set_defaults(func=device_data)

    nutrient_parser = subparsers.add_parser(
        "nutrient",
        help="fields the C/N/P variants of a derived type share (candidates for a common base type)",
    )
    nutrient_parser.add_argument(
        "action",
        choices=["common", "procedures", "extract", "merge", "species", "groups", "history", "restart"],
        help="common: match the C/N/P types' components by name; "
        "procedures: compare the types' type-bound procedures statement by statement; "
        "extract: move the shared fields to base types the C/N/P types extend (edits the E3SM sources); "
        "merge: move the procedures' shared statements to base-type procedures (after extract); "
        "species: give each core its element (core%species, set in Init; after extract); "
        "groups: move N/P-only fields to role components (mineral, uptake, retrans; groups.py); "
        "history: one history registration per core field for every species (after species, merge --procedure init); "
        "restart: likewise one restartvar per core field",
    )
    nutrient_parser.add_argument(
        "--family",
        nargs="*",
        help="common: print the fields of these families (e.g. vegetation_nutrient_flux); "
        "procedures: only these families",
    )
    nutrient_parser.add_argument(
        "--procedure",
        nargs="*",
        help="procedures: print the statements of these bindings (e.g. init); "
        "merge: the bindings to merge (default: setvalues zerodwt clean)",
    )
    nutrient_parser.add_argument(
        "--kinds",
        nargs="*",
        default=["differs"],
        help="procedures: statement kinds to print (generic guards longname differs own other)",
    )
    nutrient_parser.add_argument("-o", "--output", help="common: write all families as JSON")
    nutrient_parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="extract, merge, species, groups, history, restart: report only")
    nutrient_parser.add_argument("--diff", action="store_true", help="extract, merge, species, groups, history, restart: print the changes")
    nutrient_parser.add_argument(
        "--reanalyze", action="store_true", help="rebuild the cached elm_drv analysis first"
    )
    nutrient_parser.set_defaults(func=nutrient)

    split_parser = subparsers.add_parser(
        "split",
        help="move type definitions and their procedures out of large modules into new ones "
        "(instances stay; every `use` of a moved name is rewritten)",
    )
    split_parser.add_argument(
        "--plan",
        default="elm-data-types",
        choices=["elm-data-types"],
        help="elm-data-types: Column/Vegetation/GridcellDataType -> Nutrient{State,Flux}Type, "
        "{Vegetation,Column}{State,Flux}Types; instances stay in the *DataType modules",
    )
    split_parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="report only")
    split_parser.add_argument("--diff", action="store_true", help="print the changes")
    split_parser.add_argument(
        "--reanalyze", action="store_true", help="rebuild the cached elm_drv analysis first"
    )
    split_parser.set_defaults(func=split)

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
        "--test-env", default=None, metavar="FILE",
        help="shell file sourced before each `spel run` (the unit-test toolchain env; the ELM build keeps the clean env)",
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
    add_subset_args(validate_parser)
    validate_parser.set_defaults(func=validate)

    subset_parser = subparsers.add_parser(
        "subset",
        help="Write some gridcells of a case's reference data (spel-*.nc) as a new case's",
    )
    subset_parser.add_argument("src", help="case in unit-tests/input-data to read")
    subset_parser.add_argument("dst", help="case in unit-tests/input-data to write")
    subset_parser.add_argument(
        "--gridcells", required=True, help="1-based gridcells to keep, e.g. 3,5,7"
    )
    subset_parser.set_defaults(func=subset)

    # Parser for 'spel export'
    export_parser = subparsers.add_parser(
        "export", help="Write the elm_drv analysis as CSVs for the database (spel/db)"
    )
    export_parser.add_argument(
        "-o", "--output", default=None,
        help="Output directory (default script-output/db-export/<label>)",
    )
    export_parser.add_argument(
        "--label", default=None,
        help="Snapshot label (default <e3sm branch>@<commit>)",
    )
    export_parser.add_argument(
        "--case", default=None,
        help="Export a case's pickle (unit-tests/<case>/fut.pkl) instead of the elm_drv cache",
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

    profile_parser = subparsers.add_parser(
        "profile",
        help="build and run FUT without outputs (SPEL_PROFILE=1) under Nsight Systems/Compute",
    )
    profile_parser.add_argument("case", nargs="?", help="Unit test name (default: cwd)")
    profile_parser.add_argument("nsets", nargs="?", help="Number of sets (default 1)")
    profile_parser.add_argument(
        "--tool",
        choices=["nsys", "ncu", "none"],
        default="nsys",
        help="nsys (timeline), ncu (kernels in the 'compute' NVTX range) or none",
    )
    profile_parser.add_argument(
        "--steps", type=int, default=None, help="stop after this many time steps (SPEL_MAX_STEPS)"
    )
    profile_parser.add_argument(
        "-o", "--output", default=None,
        help="report path (default unit-tests/profiles/<case>/<case>-nsets<N>-<tool>)",
    )
    profile_parser.add_argument("--stats", action="store_true", help="nsys: print summary stats")
    profile_parser.add_argument(
        "--tool-args", default="", help="extra profiler options, e.g. \"--launch-count 20\""
    )
    profile_parser.set_defaults(func=profile)

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

    # Parser for 'spel machine'
    from spel.machines.cli import add_parser as add_machine_parser

    add_machine_parser(subparsers)

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
