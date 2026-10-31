"""
Running a transform: plan → report / diff / dry run → apply (EditSet) → write (sink) → verify →
manifest. Transforms don't do any of this themselves.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spel.transforms.core.source import Sources
from spel.transforms.core.transform import Context, Plan, Transform, options_json

MANIFEST = ".spel-transforms.json"


# --- sinks: where the output goes ------------------------------------------------------------


class Sink:
    root: Optional[Path] = None

    def target(self, path: Path) -> Path:
        raise NotImplementedError

    def write(self, files: dict[Path, str]) -> list[Path]:
        out = []
        for path, text in sorted(files.items()):
            t = self.target(Path(path))
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text(text, newline=newline_of(Path(path)))
            out.append(t)
        return out


def newline_of(path: Path) -> str:
    """The line ending of an existing source (sources are read with universal newlines; some ELM
    files are CRLF and must stay so); "\\n" for new files."""
    try:
        with open(path, "rb") as f:
            return "\r\n" if b"\r\n" in f.read() else "\n"
    except OSError:
        return "\n"


class InPlace(Sink):
    """Edit the sources where they are (the srcroot's git keeps the originals)."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else None

    def target(self, path: Path) -> Path:
        return path


class OutDir(Sink):
    """Write every file flat into one directory (a case's SourceMods/src.elm)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def target(self, path: Path) -> Path:
        return self.root / Path(path).name


class Memory(Sink):
    """Keep the output (tests, golden runs, chaining in memory)."""

    def __init__(self):
        self.files: dict[Path, str] = {}

    def target(self, path: Path) -> Path:
        return path

    def write(self, files: dict[Path, str]) -> list[Path]:
        self.files.update({Path(p): t for p, t in files.items()})
        return sorted(map(Path, files))


# --- the run ---------------------------------------------------------------------------------


@dataclass
class Result:
    plan: Plan
    outputs: dict[Path, str] = field(default_factory=dict)
    written: list[Path] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def run(t: Transform, ctx: Context, opts=None, sink: Optional[Sink] = None, dry_run: bool = False,
        verify: bool = False, manifest: bool = False) -> Result:
    """Plan `t` and, unless `dry_run`, write the result through `sink` (default: in place)."""
    opts = opts if opts is not None else t.Options()
    res = apply_plan(t.build(ctx, opts), ctx, sink, dry_run, verify)
    if manifest and res.written:
        write_manifest(sink or InPlace(), t, opts, res.plan, res.written, ctx.srcroot)
    return res


def apply_plan(plan: Plan, ctx: Context, sink: Optional[Sink] = None, dry_run: bool = False,
               verify: bool = False) -> Result:
    """Apply a plan built already: edits -> new texts -> (unless `dry_run`) the sink.
    ctx.sources keep the original texts (plan.diff/explain use them); a later transform needs
    a new Context, and the analysis of the written files."""
    res = Result(plan, plan.outputs(ctx.sources))
    if verify:
        res.problems = verify_parse(res.outputs)
    if not dry_run and res.outputs:
        res.written = (sink or InPlace()).write(res.outputs)
    return res


def verify_parse(outputs: dict[Path, str]) -> list[str]:
    """Problems found re-parsing the Fortran files a transform produced (empty: all parse)."""
    import tempfile

    from spel.scripts.fortran_parser.module_spec import ModuleSpecError, parse_modules

    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        for path, text in outputs.items():
            if Path(path).suffix.lower() not in (".f90", ".f"):
                continue
            copy = Path(tmp) / Path(path).name  # same name: the preprocessor keys on it
            copy.write_text(text)
            try:
                parse_modules(copy)
            except ModuleSpecError:
                problems.append(f"{path}: does not parse")
    return problems


def _git_commit(root: Path) -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(sink: Sink, t: Transform, opts, plan: Plan, written: list[Path], srcroot: Path) -> Path:
    """Append this run to `.spel-transforms.json` in the sink's root (the srcroot when in place)."""
    root = sink.root or srcroot
    path = Path(root) / MANIFEST
    runs = json.loads(path.read_text()).get("runs", []) if path.exists() else []
    runs.append({
        "transform": t.name,
        "options": options_json(opts),
        "source_commit": _git_commit(srcroot),
        "files": sorted(str(p) for p in written),
        "refused": [str(r) for r in plan.refusals],
    })
    path.write_text(json.dumps({"runs": runs}, indent=1) + "\n")
    return path


def context(fut=None, srcroot: Optional[Path] = None, reanalyze: bool = False) -> Context:
    """A Context on the cached analysis of the srcroot (loaded unless `fut` is given)."""
    if fut is None:
        from spel.scripts.analysis_cache import load_analysis

        fut = load_analysis(reanalyze=reanalyze)
    return Context(fut, srcroot, Sources())
