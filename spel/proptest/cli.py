"""`spel proptest`: property-based tests of extracted ELM routines (README, `spel proptest`;
the spec format is in spel/proptest/SPEC.md)."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any, Optional

# `--landunits` values: hierarchy.LandunitKind's (not imported here: the CLI loads lazily)
LANDUNIT_CHOICES = ("soil", "crop", "urban", "lake")


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "proptest",
        help="Property-based tests of a unit-test case: Hypothesis draws scenario inputs, "
        "properties/relations in a <name>.proptest.toml (Fortran syntax)",
    )
    sub = p.add_subparsers(dest="proptest_cmd", required=True)

    def build_options(q, rebuild: bool = True) -> None:
        q.add_argument(
            "--fc",
            default=None,
            help="compiler (default: the spec's [case] build, $SPEL_FC, the machine's)",
        )
        q.add_argument(
            "--acc",
            default=None,
            choices=["OFF", "MULTICORE", "GPU", "OPENMP"],
            help="OpenACC target of the build (default: the spec's, else OFF)",
        )
        if rebuild:
            q.add_argument(
                "--rebuild",
                action="store_true",
                help="rebuild the shared library first (the default "
                "when the case's sources changed since the last build)",
            )
            q.add_argument(
                "--no-build",
                action="store_true",
                help="use the library as it is, even when sources changed",
            )

    b = sub.add_parser(
        "build",
        help="generate the Python shim (SpelPyMod.F90) and build the case as a shared library",
    )
    b.add_argument("case", help="unit-test case name (unit-tests/<case>) or directory")
    build_options(b, rebuild=False)
    b.add_argument(
        "--dbg",
        action="store_true",
        help="debug build (-O0 + checks + coverage), in its own directory",
    )
    b.set_defaults(func=_reported(_build))

    i = sub.add_parser(
        "init",
        help="write a starter <case>.proptest.toml (rules from a reference capture, "
        "two first checks, TODO lines for what needs a decision)",
    )
    i.add_argument("case", help="unit-test case name or directory")
    i.add_argument(
        "--reference",
        default=None,
        help="capture directory, or its name under unit-tests/input-data "
        "(default: the case's own, when it has one)",
    )
    i.add_argument(
        "--where",
        default=None,
        help='patch-level condition for the statistics, e.g. "frac_veg_nosno(p) > 0"',
    )
    i.add_argument(
        "--stride",
        type=int,
        default=1,
        help="use every N-th time step of the reference",
    )
    i.add_argument(
        "--landunits",
        nargs="+",
        default=None,
        choices=LANDUNIT_CHOICES,
        metavar="KIND",
        help="the landunit kinds of the domain and the statistics (soil crop urban lake; default: "
        "from the filters the routines take, e.g. urbanl -> urban)",
    )
    i.add_argument(
        "--namelist",
        "--constant",
        dest="constant",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a namelist option or constant for the spec's [namelist] table, e.g. use_c13=true or one "
        "the reference lacks (soil_stress_method=0)",
    )
    i.add_argument(
        "-o",
        "--output",
        default=None,
        help="the spec to write (default: ./<case>.proptest.toml)",
    )
    i.add_argument("--force", action="store_true", help="replace an existing file")
    i.set_defaults(func=_reported(_init))

    r = sub.add_parser("run", help="run the checks of a spec")
    r.add_argument("spec", help="<name>.proptest.toml")
    r.add_argument(
        "-s", "--scenario", action="append", help="scenario name or glob (repeatable)"
    )
    r.add_argument(
        "-k", "--check", action="append", help="check name or glob (repeatable)"
    )
    r.add_argument(
        "-n",
        "--max-examples",
        type=int,
        default=None,
        help="Hypothesis examples per check",
    )
    r.add_argument("--seed", type=int, default=None, help="Hypothesis seed")
    r.add_argument(
        "--json", default=None, metavar="FILE", help="also write the outcomes as JSON"
    )
    r.add_argument(
        "--junit",
        default=None,
        metavar="FILE",
        help="also write a JUnit XML report (CI)",
    )
    build_options(r)
    r.set_defaults(func=_reported(_run))

    pr = sub.add_parser(
        "probe",
        help="run a few examples of a scenario and say what the routine did: outputs "
        "never written, elements selected, aborts (do this before writing checks)",
    )
    pr.add_argument("spec", help="<name>.proptest.toml")
    pr.add_argument(
        "-s", "--scenario", action="append", help="scenario name or glob (default: all)"
    )
    pr.add_argument(
        "-n",
        "--examples",
        type=int,
        default=5,
        help="examples per scenario (the first at the midpoints)",
    )
    pr.add_argument("--seed", type=int, default=0)
    build_options(pr)
    pr.set_defaults(func=_reported(_probe))

    rp = sub.add_parser(
        "replay", help="run a saved failing example again, without Hypothesis"
    )
    rp.add_argument(
        "directory", help="unit-tests/proptest-data/<case>/failures/<scenario>__<check>"
    )
    build_options(rp, rebuild=False)
    rp.set_defaults(func=_reported(_replay))

    cv = sub.add_parser(
        "coverage",
        help="the branches of the routines a scenario never enters (gfortran debug build + gcov)",
    )
    cv.add_argument("spec", help="<name>.proptest.toml")
    cv.add_argument(
        "-s", "--scenario", action="append", help="scenario name or glob (default: all)"
    )
    cv.add_argument(
        "-n", "--examples", type=int, default=20, help="examples per scenario"
    )
    cv.add_argument("--seed", type=int, default=0)
    cv.add_argument(
        "--all-files",
        action="store_true",
        help="every source of the case, not only the routines under test",
    )
    cv.add_argument(
        "--all",
        action="store_true",
        help="also list the branches the namelist options turn off",
    )
    cv.set_defaults(func=_reported(_coverage))

    s = sub.add_parser(
        "show", help="the case's variable inventory and the spec's scenarios and checks"
    )
    s.add_argument("target", help="<name>.proptest.toml or a case name/directory")
    s.add_argument(
        "--kind",
        action="append",
        help="list the variables of this kind (input, inout, output, ...)",
    )
    s.set_defaults(func=_reported(_show))

    st = sub.add_parser(
        "stats",
        help="reference statistics -> an input rule file and a markdown report "
        "(design aid; values never feed a test)",
    )
    st.add_argument("case", help="unit-test case name or directory")
    st.add_argument(
        "--reference",
        required=True,
        help="capture directory (or its name under unit-tests/input-data)",
    )
    st.add_argument(
        "--where",
        default=None,
        help='patch-level Fortran condition, e.g. "frac_veg_nosno(p) > 0"',
    )
    st.add_argument("--stride", type=int, default=1, help="use every N-th time step")
    st.add_argument(
        "--landunits",
        nargs="+",
        default=None,
        choices=LANDUNIT_CHOICES,
        metavar="KIND",
        help="only patches on landunits of these kinds (soil crop urban lake), and their coarser "
        "elements",
    )
    st.add_argument(
        "--namelist",
        "--constant",
        dest="constant",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a namelist option or constant over the reference's (e.g. soil_stress_method=0): the "
        "statistics cover the inputs that are read under these options",
    )
    st.set_defaults(func=_reported(_stats))

    a = sub.add_parser(
        "subspace",
        help="active subspace of an output over a scenario (batched finite differences)",
    )
    a.add_argument("spec", help="<name>.proptest.toml")
    a.add_argument("--scenario", required=True)
    a.add_argument(
        "--output",
        required=True,
        help='Fortran expression per element, e.g. "t_veg(p)"',
    )
    a.add_argument(
        "--input", action="append", help="restrict to these inputs (repeatable)"
    )
    a.add_argument(
        "--parameters",
        action="store_true",
        help="also step the inputs without a subgrid dimension "
        "(ruled parameters): one more call per direction and base draw",
    )
    a.add_argument(
        "--samples",
        type=int,
        default=4,
        help="base draws (each gives one point per element)",
    )
    a.add_argument(
        "--h",
        type=float,
        default=1.0e-3,
        help="step in the inputs' [0, 1] quantile coordinate",
    )
    a.add_argument(
        "--max-sets",
        type=int,
        default=33,
        help="domain copies per call (directions per call + 1)",
    )
    a.add_argument("--bootstrap", type=int, default=200)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=_reported(_subspace))


def _reported(command):
    """Run a command; a problem the user can fix (ProptestError) is one line and exit status 2."""

    def run(args) -> None:
        from spel.proptest.errors import ProptestError

        try:
            command(args)
        except ProptestError as err:
            print(f"spel proptest {args.proptest_cmd}: error: {err}", file=sys.stderr)
            sys.exit(2)

    return run


def _case_dir(case: str) -> Path:
    p = Path(case)
    if p.is_dir():
        return p.resolve()
    from spel.scripts.config import unittests_dir

    return Path(unittests_dir) / case


def _build(args) -> None:
    from spel.proptest import build

    variant = build.Variant.of(fc=args.fc, acc=args.acc, dbg=args.dbg)
    print(build.build(_case_dir(args.case), quiet=False, variant=variant))


def _landunits(names: Optional[list[str]]):
    from spel.proptest.hierarchy import LandunitKind

    return {LandunitKind(n) for n in names} if names else None


def _init(args) -> None:
    from spel.proptest import scaffold
    from spel.proptest.inventory import Inventory

    case_dir = _case_dir(args.case)
    inv = Inventory.load(case_dir)
    out = (
        Path(args.output)
        if args.output
        else Path.cwd() / f"{case_dir.name}.proptest.toml"
    )
    path, todo = scaffold.write(
        case_dir,
        inv,
        out,
        reference=scaffold.reference_dir(case_dir, args.reference),
        where=args.where,
        stride=args.stride,
        constants=dict(_parse_constant(c) for c in args.constant),
        force=args.force,
        landunits=_landunits(args.landunits),
    )
    print(path)
    for line in todo:
        print(f"  to do: {line}")
    print(
        f"next: spel proptest show {path}; spel proptest probe {path}; spel proptest run {path}"
    )


def _run(args) -> None:
    from spel.proptest import runner, spec

    sp, inv = spec.load(Path(args.spec))
    settings: dict[str, Any] = {}
    if args.max_examples is not None:
        settings["max_examples"] = args.max_examples
    if args.seed is not None:
        settings["seed"] = args.seed
    outcomes = runner.run(
        sp,
        inv,
        scenarios=args.scenario,
        checks=args.check,
        settings=settings,
        fc=args.fc,
        acc=args.acc,
        rebuild=args.rebuild,
        no_build=args.no_build,
    )
    if args.json:
        runner.write_json(outcomes, sp, settings, Path(args.json))
    if args.junit:
        runner.write_junit(outcomes, sp, Path(args.junit))
    failed = [o for o in outcomes if not o.passed]
    if not outcomes:
        print("no check selected", file=sys.stderr)
        sys.exit(2)
    print(f"\n{len(outcomes) - len(failed)} passed, {len(failed)} failed")
    if failed:
        sys.exit(1)


def _selected_scenarios(sp, patterns):
    from spel.proptest.errors import SpecError
    from spel.proptest.runner import _selected

    names = [n for n in sp.scenarios if not patterns or _selected(n, patterns)]
    if not names:
        raise SpecError(f"no scenario matches {patterns} (have {sorted(sp.scenarios)})")
    return names


def _probe(args) -> None:
    from spel.proptest import runner, spec

    sp, inv = spec.load(Path(args.spec))
    names = _selected_scenarios(sp, args.scenario)
    with runner.Session(
        sp, inv, fc=args.fc, acc=args.acc, rebuild=args.rebuild, no_build=args.no_build
    ) as s:
        for name in names:
            scen = sp.scenarios[name]
            print(
                s.for_scenario(scen).probe(scen, n=args.examples, seed=args.seed).text()
            )


def _replay(args) -> None:
    from spel.proptest import runner

    outcome = runner.replay(Path(args.directory), fc=args.fc, acc=args.acc)
    if not outcome.passed:
        sys.exit(1)


def _coverage(args) -> None:
    from spel.proptest import coverage, spec

    sp, inv = spec.load(Path(args.spec))
    for name in _selected_scenarios(sp, args.scenario):
        res = coverage.measure(
            sp,
            inv,
            name,
            examples=args.examples,
            seed=args.seed,
            all_files=args.all_files,
        )
        print(coverage.report(res, show_off=args.all))


def _show(args) -> None:
    from collections import Counter

    from spel.proptest import spec
    from spel.proptest.inventory import Inventory

    target = Path(args.target)
    sp = None
    if target.suffix == ".toml":
        sp, inv = spec.load(target)
    else:
        inv = Inventory.load(_case_dir(args.target))
    print(f"case {inv.case_dir}: routines {', '.join(inv.routines)}")
    counts = Counter(v.kind for v in inv.vars.values())
    print("variables: " + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    kinds = set(counts)
    for kind in args.kind or []:
        if kind not in kinds:
            print(f"\n{kind}: no such kind here (have {', '.join(sorted(kinds))})")
            continue
        print(f"\n{kind}:")
        for v in inv.of_kind(kind):
            dims = ", ".join(d.text for d in v.dims)
            units = f" [{v.units}]" if getattr(v, "units", "") else ""
            print(f"  {v.name}({dims}) {v.type}{units}")
    if sp is None:
        if inv.options():
            print(
                "namelist options the routines test (declared defaults): "
                + ", ".join(
                    f"{n} = {inv.fortran_value(n, d) if d is not None else '?'}"
                    for n, d in sorted(inv.options().items())
                )
            )
        return
    from spel.proptest.domain import CONSTANTS_FILE, DOMAIN_CONSTANTS, read_constants, time_values
    from spel.proptest.shim import direct_init

    lay = sp.domain.layout(inv.parameters)
    cells = Counter(g.label for g in lay.set_gridcells)
    print(
        f"domain: {', '.join(f'{n} x {label}' for label, n in cells.items())} per set, {sp.domain.nsets} sets; "
        "elements " + ", ".join(f"{lev} {n}" for lev, n in lay.sizes().items())
    )
    decided = {k: f(lay) for k, f in DOMAIN_CONSTANTS.items() if k in inv.constants}
    if decided:
        print("decided by the domain: " + ", ".join(f"{k} = {v}" for k, v in decided.items()))
    if sp.domain.init:
        print(f"init: {sp.domain.init} ([domain] init)")
    elif direct_init(inv):
        print(
            "init: direct (the case allocates from descriptors: no inputs file is written)"
        )
    else:
        print(
            "init: files (the case predates direct init; `spel create` it again for that)"
        )
    reference = (
        read_constants(sp.domain.reference / CONSTANTS_FILE)
        if sp.domain.reference is not None
        else {}
    )
    for name, sc in sp.scenarios.items():
        consts = spec.constants(sp, inv, sc)
        live = inv.configured(consts)
        over = f"{sc.level} in {sc.filter}" if sc.filter else sc.level
        dormant = spec.dormant_rules(sc, live)
        print(
            f"\nscenario {name}: over {over}, {len(sc.inputs)} ruled inputs, {len(sc.derive)} derived, "
            f"{len(sc.assume)} assumptions"
        )
        print("  namelist options: " + _options(sp, inv, sc, consts, reference))
        dead = Counter(
            ", ".join(v.dead_under)
            for v in live.vars.values()
            if v.kind == "unused" and v.dead_under
        )
        if dead:
            print(
                f"  {sum(dead.values())} variables are not used under these options ("
                + "; ".join(f"{k}: {n}" for k, n in dead.most_common())
                + ")"
            )
        if dormant:
            print(
                "  rules for inputs that are not read under these options (dormant): "
                + ", ".join(dormant)
            )
        missing = spec.missing_rules(sc, live, consts)
        if missing:
            print("  inputs without a rule: " + ", ".join(missing))
        if sc.steps > 1:
            _show_steps(sc, inv, live, consts)
    consts = spec.constants(sp, inv)
    defaulted = time_values(inv, sp.domain, consts)[1]
    if defaulted:
        print(
            "\ntime scalars not set in [domain.time]: "
            + ", ".join(f"{k} = {v}" for k, v in sorted(defaulted.items()))
        )
    if sp.compare is not None:
        print(
            f"compare: {sp.compare.case_dir.name}"
            + (f" build {sp.compare.build}" if sp.compare.build else "")
        )
    for c in sp.checks:
        extra = f" [namelist {c.namelist}]" if getattr(c, "namelist", None) else ""
        print(f"{c.table} {c.name!r}{extra}: {c.judged.text}")


def _show_steps(sc, inv, live, consts) -> None:
    """A multi-step scenario: its step tables and the boundary variables (coupling.boundary) with
    their writers and how the scenario declares them."""
    from spel.proptest import coupling
    from spel.proptest.errors import ProptestError

    print(f"  {sc.steps} steps; forcing: {len(sc.forcing) + len(sc.forcing_exprs)}, between: {len(sc.between)}, "
          f"keep: {', '.join(sc.keep) or '-'}" + (f"; time: {sc.time}" if sc.time else ""))
    try:
        found = coupling.boundary(inv, consts)
    except ProptestError as exc:
        print(f"  boundary variables: unknown ({exc})")
        return
    rows = [(n, w) for n, w in sorted(found.items()) if live.vars[n].kind in ("input", "inout")]
    if not rows:
        print("  boundary variables: none (no other elm_drv routine writes what the routines read)")
        return
    print(f"  boundary variables (read by the routines, written by other elm_drv routines within a step): {len(rows)}")
    for n, w in rows:
        how = ("forcing" if n in sc.forcing or any(v == n for v, _ in sc.forcing_exprs)
               else "between" if any(v == n for v, _ in sc.between)
               else "keep" if sc.boundary_declared(n) else "UNDECLARED")
        print(f"    {how:10s} {n} ({live.vars[n].kind}): {coupling.describe(w, 2)}")


def _options(sp, inv, scenario, consts, reference) -> str:
    """`option = value (where it comes from)` for the options the routines test and those the spec sets."""
    from spel.proptest import spec

    mine = spec.options(sp, scenario)
    out = []
    for name in sorted(set(inv.options()) | set(mine)):
        if name in scenario.namelist:
            source = "scenario"
        elif name in sp.namelist:
            source = "[namelist]"
        elif name in {k.lower() for k in sp.domain.constants}:
            source = "[domain.constants]"
        elif name in reference:
            source = "reference"
        else:
            source = "not given: 0"
        out.append(
            f"{name} = {inv.fortran_value(name, consts.get(name, 0))} ({source})"
        )
    return ", ".join(out)


def _parse_constant(text: str) -> tuple[str, Any]:
    from spel.proptest.errors import SpecError

    name, sep, value = text.partition("=")
    if not sep or not name.strip():
        raise SpecError(f"--constant {text!r}: expected NAME=VALUE")
    try:
        return name.strip().lower(), tomllib.loads(f"v = {value.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        raise SpecError(
            f"--constant {text!r}: VALUE must be a number or a quoted string"
        ) from None


def _stats(args) -> None:
    from spel.proptest import scaffold, stats
    from spel.proptest.domain import CONSTANTS_FILE, read_constants
    from spel.proptest.inventory import Inventory

    case_dir = _case_dir(args.case)
    ref = scaffold.reference_dir(case_dir, args.reference)
    assert ref, f"Reference directory {args.reference} for {case_dir} not found!!"
    inv = Inventory.load(case_dir)
    consts = read_constants(ref / CONSTANTS_FILE)
    consts.update(dict(_parse_constant(c) for c in args.constant))
    live = inv.configured(consts)
    res = stats.compute(
        live,
        ref,
        where=args.where,
        stride=args.stride,
        constants=consts,
        landunits=_landunits(args.landunits),
    )
    for path in stats.write(live, res, case_dir):
        print(path)


def _subspace(args) -> None:
    from spel.proptest import spec, subspace

    sp, inv = spec.load(Path(args.spec))
    res = subspace.analyze(
        sp,
        inv,
        args.scenario,
        args.output,
        samples=args.samples,
        h=args.h,
        max_sets=args.max_sets,
        seed=args.seed,
        bootstrap=args.bootstrap,
        inputs=args.input,
        parameters=args.parameters,
    )
    j, md = subspace.write(res, sp.case_dir)
    print(subspace.report_md(res))
    print(j)
    print(md)
