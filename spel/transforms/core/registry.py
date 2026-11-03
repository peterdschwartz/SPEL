"""
The transform registry and the `spel transform` command.

    spel transform list
    spel transform explain NAME
    spel transform NAME [options] [--dry-run] [--diff] [--report] [--where TEXT] [--verify] [-o DIR]
                   [--config PRESET|lnd_in]
    spel transform --spec FILE.toml [--dry-run] [--diff] ...

`--config` (or `config = "..."` at the top of a spec) gives the run configuration: a preset of
spel/db (presets/<NAME>.cfg) or a case's lnd_in. The transforms then touch only the code that
runs in it: namelist-gated branches that cannot run, and the routines reached only through
them, are left alone (core/config.py).

A transform registers itself with `@transform`; the built-in ones are imported on first use
(`BUILTINS`). Its `Options` dataclass fields become flags (`routines` -> `--routines`, a bool
field -> a switch, a list field -> several values) and keys of the spec file:

    # one transform
    transform = "collapse-bounds"
    [options]
    roots = ["elm_driver::elm_drv"]

    # or a pipeline: the steps plan in order, each on the text the previous ones produced, and
    # the net result is written once (core/pipeline.py: the analysis of the edited files is
    # refreshed in memory; --reanalyze-steps writes each step and reanalyzes instead)
    [[step]]
    transform = "section-loops"
    [[step]]
    transform = "acc-apply"
    options = { roots = ["elm_driver::elm_drv"] }
"""

from __future__ import annotations

import argparse
import importlib
import sys
import typing
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Optional

from spel.transforms.core.transform import Transform

REGISTRY: dict[str, type[Transform]] = {}

# modules that define built-in transforms (imported lazily: some import the whole analysis stack)
BUILTINS = [
    "spel.transforms.collapse_bounds",
    "spel.transforms.blocked_loops",
    "spel.transforms.section_loops",
    "spel.transforms.loop_fusion",
    "spel.transforms.filter_map",
    "spel.transforms.filter_locals",
    "spel.transforms.scalarize_locals",
    "spel.transforms.drop_dummies",
    "spel.transforms.defer_checks",
    "spel.transforms.acc_locals",
    "spel.transforms.acc_apply",
    "spel.transforms.acc_sync",
    "spel.transforms.acc_strip",
    "spel.transforms.acc_stale",
    "spel.transforms.cpp_resolve",
    "spel.transforms.acc_routines",
    "spel.transforms.acc_to_omp",
    "spel.transforms.device.generate",
    "spel.transforms.device.targeted",
    "spel.transforms.split.modules",
    "spel.transforms.nutrient.transforms",
    "spel.transforms.compact.rewrite",
    "spel.transforms.prune",
]
_loaded = False


def transform(cls: type[Transform]) -> type[Transform]:
    """Class decorator: register a Transform under its `name`."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} has no name")
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise ValueError(f"transform {cls.name!r} is registered twice ({REGISTRY[cls.name].__name__}, {cls.__name__})")
    REGISTRY[cls.name] = cls
    return cls


def load_builtins() -> dict[str, type[Transform]]:
    global _loaded
    if not _loaded:
        for mod in BUILTINS:
            importlib.import_module(mod)
        _loaded = True
    return REGISTRY


def get(name: str) -> type[Transform]:
    reg = load_builtins()
    if name not in reg:
        raise KeyError(f"unknown transform {name!r}; known: {', '.join(sorted(reg))}")
    return reg[name]


# --- options <-> flags / spec keys -----------------------------------------------------------


def _base_type(tp):
    """(scalar type, is_list) of an Options field annotation."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or (origin is not None and str(origin) == "types.UnionType"):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return _base_type(args[0]) if len(args) == 1 else (str, False)
    if origin in (list, tuple, set, frozenset):
        args = typing.get_args(tp)
        return (args[0] if args else str), True
    return tp, False


def _resolved_fields(opts_cls):
    hints = typing.get_type_hints(opts_cls)
    return [(f, hints.get(f.name, str)) for f in fields(opts_cls)]


def add_option_flags(parser: argparse.ArgumentParser, opts_cls) -> None:
    if not is_dataclass(opts_cls):
        return
    group = parser.add_argument_group("transform options")
    for f, tp in _resolved_fields(opts_cls):
        flag = "--" + f.name.replace("_", "-")
        base, many = _base_type(tp)
        default = None if f.default is MISSING and f.default_factory is MISSING else (
            f.default if f.default is not MISSING else f.default_factory())
        helptext = f.metadata.get("help", "") + (f" (default {default!r})" if default not in (None, [], ()) else "")
        if base is bool and not many:
            group.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction,
                               default=argparse.SUPPRESS, help=helptext)
        else:
            group.add_argument(flag, dest=f.name, nargs="+" if many else None,
                               type=Path if base is Path else (int if base is int else (float if base is float else str)),
                               default=argparse.SUPPRESS, help=helptext)


def make_options(t: type[Transform], values: dict):
    """An Options instance from flag/spec values (unknown keys are an error)."""
    if not is_dataclass(t.Options):
        if values:
            raise ValueError(f"{t.name} takes no options, got {sorted(values)}")
        return t.Options()
    known = {f.name: tp for f, tp in _resolved_fields(t.Options)}
    vals = {}
    for k, v in values.items():
        key = k.replace("-", "_")
        if key not in known:
            raise ValueError(f"{t.name}: unknown option {k!r}; known: {', '.join(sorted(known))}")
        base, many = _base_type(known[key])
        if base is Path:
            v = [Path(x) for x in v] if many else Path(v)
        if many and typing.get_origin(known[key]) is tuple:
            v = tuple(v)
        vals[key] = v
    return t.Options(**vals)


# --- specs -----------------------------------------------------------------------------------


def spec_config(path: Path) -> Optional[str]:
    """The `config` of a TOML spec file (a preset name or a namelist file), if any."""
    import tomllib

    return tomllib.loads(Path(path).read_text()).get("config")


def load_spec(path: Path) -> list[tuple[type[Transform], object]]:
    """The (transform, options) steps of a TOML spec file."""
    import tomllib

    data = tomllib.loads(Path(path).read_text())
    raw = data.get("step") or [data]
    steps = []
    for i, st in enumerate(raw):
        if "transform" not in st:
            raise ValueError(f"{path}: step {i + 1} has no `transform`")
        t = get(st["transform"])
        steps.append((t, make_options(t, dict(st.get("options", {})))))
    return steps


# --- command ---------------------------------------------------------------------------------


def _run_parser(prog: str, t: Optional[type[Transform]]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=t.describe() if t else None,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="plan and report, write nothing")
    p.add_argument("--diff", action="store_true", help="print the unified diff of the changes")
    p.add_argument("--report", action="store_true", help="print what changed where, and what was refused")
    p.add_argument("--where", help="print before/after of the changes at locations containing TEXT")
    p.add_argument("--json", action="store_true", help="print the plan as JSON")
    p.add_argument("--verify", action="store_true", help="re-parse the files written (problems: exit 1)")
    p.add_argument("-o", "--out", type=Path, help="write the files flat into DIR instead of in place")
    p.add_argument("--reanalyze", action="store_true", help="rebuild the cached elm_drv analysis first")
    p.add_argument("--config", help="run configuration: a preset (spel/db presets/NAME.cfg) or a case's lnd_in; "
                                    "only the code that runs in it is transformed")
    if t is None:
        p.add_argument("--reanalyze-steps", action="store_true",
                       help="pipeline: write each step in place and reanalyze before the next (no -o)")
    if t is not None:
        add_option_flags(p, t.Options)
    return p


def _steps_from_args(argv: list[str]):
    if argv and argv[0] == "--spec":
        if len(argv) < 2:
            sys.exit("spel transform: --spec needs a file")
        spec = Path(argv[1])
        args = _run_parser("spel transform --spec FILE", None).parse_args(argv[2:])
        if args.config is None:
            args.config = spec_config(spec)
        return args, load_spec(spec)
    t = get(argv[0])
    p = _run_parser(f"spel transform {t.name}", t)
    args = p.parse_args(argv[1:])
    base = {a.dest for a in _run_parser("x", None)._actions}
    vals = {k: v for k, v in vars(args).items() if k not in base}
    return args, [(t, make_options(t, vals))]


def main(argv: list[str]) -> int:
    """`spel transform ...`: returns the exit status."""
    import json

    from spel.transforms.core import driver

    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    if argv[0] == "list":
        for name, t in sorted(load_builtins().items()):
            print(f"{name:20s} {t.summary}")
        return 0
    if argv[0] == "explain":
        if len(argv) < 2:
            sys.exit("spel transform explain: give a transform name")
        print(get(argv[1]).describe())
        return 0

    args, steps = _steps_from_args(argv)
    from spel.scripts.config import E3SM_SRCROOT

    srcroot = Path(E3SM_SRCROOT)
    sink = driver.OutDir(args.out) if args.out else driver.InPlace()

    def show(plan, sources) -> None:
        if args.json:
            print(json.dumps(plan.to_json(), indent=1))
        if args.report or args.dry_run or not (args.diff or args.where or args.json):
            print(plan.report())
        if args.diff:
            sys.stdout.write(plan.diff(sources))
        if args.where:
            print(plan.explain(sources, args.where))

    def done(written, problems) -> int:
        for w in written:
            print(f"wrote {w}")
        for prob in problems:
            print(f"verify: {prob}", file=sys.stderr)
        return 1 if problems else 0

    if len(steps) > 1 and not getattr(args, "reanalyze_steps", False):
        from spel.transforms.core.pipeline import Pipeline, PipelineError

        ctx0 = driver.context(srcroot=srcroot, reanalyze=args.reanalyze, config=args.config)
        if ctx0.config is not None:
            print(ctx0.config.describe())
        try:
            res = Pipeline([(t(), opts) for t, opts in steps]).run(
                ctx0.fut, srcroot, sink=sink, dry_run=args.dry_run, verify=args.verify, manifest=True,
                on_step=lambda i, st: show(st.result.plan, st.ctx.sources), config=ctx0.config)
        except PipelineError as err:
            print(f"spel transform: {err}\n(nothing written; --reanalyze-steps runs the steps one "
                  "at a time with a reanalysis in between)", file=sys.stderr)
            return 2
        return done(res.written, res.problems)

    if len(steps) > 1 and args.out and not args.dry_run:
        sys.exit("spel transform: --reanalyze-steps runs in place (each step analyzes what the "
                 "earlier ones wrote); use a scratch worktree instead of -o")
    status = 0
    for i, (t, opts) in enumerate(steps):
        # a later step needs the analysis of what the earlier ones wrote
        ctx = driver.context(srcroot=srcroot, reanalyze=args.reanalyze or (i > 0 and not args.dry_run),
                             config=args.config)
        if ctx.config is not None and i == 0:
            print(ctx.config.describe())
        res = driver.run(t(), ctx, opts, sink=sink, dry_run=args.dry_run, verify=args.verify,
                         manifest=not args.dry_run)
        show(res.plan, ctx.sources)
        status = max(status, done(res.written, res.problems))
        if args.dry_run and i < len(steps) - 1:
            print(f"(dry run: steps after {t.name} plan on the unmodified sources)")
    return status
