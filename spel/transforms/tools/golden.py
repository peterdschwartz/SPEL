"""
Golden outputs of every source transform, for refactoring SPEL's transforms safely.

    python -m spel.transforms.tools.golden OUT_DIR [--srcroot E3SM] [--only PASS ...]
    python -m spel.transforms.tools.golden OUT_DIR --srcroot E3SM --chain nutrient
    diff -r OLD_OUT NEW_OUT

Each pass runs in this process on the cached analysis of `--srcroot`, against pristine sources,
behind a virtual file system: every file a pass writes is captured (never written to the
checkout) and dumped under OUT_DIR/<pass>/ (absolute paths flattened), with its log and stdout
in OUT_DIR/<pass>.log. Two runs on the same analysis must give identical trees unless a change
is intended.

`--chain` replays a sequence of transforms the way they are applied for real, each on the output
of the previous one, re-analyzing in between: in a scratch git worktree of the srcroot's HEAD
(removed afterwards), with the spel CLI as a subprocess. OUT_DIR/chain-<name>/NN-<step>.diff is
what each step changed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
import traceback
from pathlib import Path


from spel.transforms.core.capture import VirtualFS  # noqa: E402  (re-exported)


ROOTS = ["elm_driver::elm_drv"]


def _ids(fut):
    return sorted(sid for sid, s in fut.subroutine_dict.items() if s.record is not None)


def p_blocked_loops(fut, srcroot, out):
    from spel.transforms import blocked_loops
    return blocked_loops.apply(fut, _ids(fut))


def p_collapse_bounds(fut, srcroot, out):
    from spel.transforms import collapse_bounds
    return collapse_bounds.apply(fut, _ids(fut))


def p_section_loops(fut, srcroot, out):
    from spel.transforms import section_loops
    return section_loops.apply(fut, _ids(fut))


def p_defer_checks(fut, srcroot, out):
    from spel.transforms import defer_checks
    p = defer_checks.plan(fut, ROOTS)
    return [str(r) for r in p.refused] + defer_checks.apply(fut, p)


def p_acc_locals(fut, srcroot, out):
    from spel.transforms import acc_locals
    return acc_locals.apply_fix(fut, acc_locals.check(fut))


def p_acc_apply(fut, srcroot, out):
    from spel.transforms import acc_apply
    pp = acc_apply.plan(fut, ROOTS)
    return pp.notes + acc_apply.apply(fut, pp)


def p_acc_sync(fut, srcroot, out):
    from spel.transforms import acc_sync
    sp = acc_sync.plan(fut, ROOTS)
    return list(sp.notes) + acc_sync.apply(fut, sp)


def p_acc_to_omp(fut, srcroot, out):
    from spel.transforms.acc_to_omp import translate_modules
    mods = sorted({s.module for s in fut.subroutine_dict.values()
                   if s.record is not None and "!$acc" in Path(s.filepath).read_text().lower()})
    log = []
    for tr in translate_modules(fut, mods):
        log.append(tr.report())
        try:
            Path(out, tr.path.name).write_text("".join(tr.apply()))
        except Exception as err:  # noqa: BLE001
            log.append(f"ERROR {err}")
    return log


def p_device_data(fut, srcroot, out):
    from spel.transforms.device import generate
    _, log = generate.apply(fut, Path(srcroot), out_dir=Path(out))
    return log


def p_device_targeted(fut, srcroot, out):
    from spel.transforms.device import targeted
    return targeted.apply(fut, Path(srcroot))


def _cli(argv):
    def run(fut, srcroot, out):
        import spel.scripts.analysis_cache as ac
        from spel import cli
        saved, sys.argv = sys.argv, ["spel", *[a.replace("{out}", str(out)) for a in argv]]
        loader = ac.load_analysis
        ac.load_analysis = lambda *a, **k: fut  # the pass gets the already loaded analysis
        try:
            cli.main()
        except SystemExit as err:
            print(f"SystemExit: {err.code}")
        finally:
            sys.argv, ac.load_analysis = saved, loader
        return []
    return run


PASSES = {
    "blocked_loops": p_blocked_loops,
    "collapse_bounds": p_collapse_bounds,
    "section_loops": p_section_loops,
    "defer_checks": p_defer_checks,
    "acc_locals": p_acc_locals,
    "acc_apply": p_acc_apply,
    "acc_sync": p_acc_sync,
    "acc_to_omp": p_acc_to_omp,
    "device_data": p_device_data,
    "device_targeted": p_device_targeted,
    "compact_rewrite": _cli(["transform", "compact-rewrite", "-o", "{out}"]),
    "nutrient_extract": _cli(["transform", "nutrient-extract"]),
    "nutrient_groups": _cli(["transform", "nutrient-groups"]),
    "nutrient_merge": _cli(["transform", "nutrient-merge"]),
    "nutrient_history": _cli(["transform", "nutrient-history"]),
    "nutrient_restart": _cli(["transform", "nutrient-restart"]),
    "nutrient_species": _cli(["transform", "nutrient-species"]),
    "split": _cli(["transform", "split"]),
    "prune": _cli(["transform", "prune"]),
}


# the nutrient refactor of the E3SM branch (cc1ed91d5f..97423d0173), run on master plus the
# removal of the unused legacy C/N/P types (34887d2ab6)
CHAINS = {
    "nutrient": [
        ["transform", "nutrient-extract"],
        ["transform", "nutrient-species"],
        ["transform", "nutrient-merge"],
        ["transform", "nutrient-groups"],
        ["transform", "nutrient-merge", "--procedure", "init"],
        ["transform", "nutrient-history"],
        ["transform", "nutrient-restart"],
        ["transform", "nutrient-merge", "--procedure", "summary"],
        ["transform", "split"],
    ],
}


def run_chain(name: str, srcroot: Path, out_root: Path) -> None:
    import shutil
    import subprocess

    def git(*a, cwd):
        return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout

    work = Path(tempfile.gettempdir()) / f"golden-chain-{name}"
    if work.exists():
        git("worktree", "remove", "--force", str(work), cwd=srcroot)
    git("worktree", "add", "--detach", str(work), "HEAD", cwd=srcroot)
    d = out_root / f"chain-{name}"
    d.mkdir(parents=True, exist_ok=True)
    spel = shutil.which("spel") or "spel"
    try:
        for k, step in enumerate(CHAINS[name], 1):
            label = f"{k:02d}-" + "-".join(a.lstrip("-") for a in step)
            cmd = [spel, *step, "--reanalyze", "--srcroot", str(work)]
            res = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True)
            git("add", "-A", cwd=work)
            # the transform driver's manifest is not part of a pass's output
            diff = git("diff", "--cached", "--no-color", "--", ".", ":(exclude,glob)**/.spel-transforms.json", cwd=work)
            git("-c", "user.name=golden", "-c", "user.email=golden@local", "commit", "-q", "--allow-empty",
                "-m", label, cwd=work)
            (d / f"{label}.diff").write_text(diff.replace(str(work), "$SRC"))
            log = res.stdout + "\n--- stderr\n" + "\n".join(
                l for l in res.stderr.splitlines() if "Error: bc_in not known" not in l
            )
            (d / f"{label}.log").write_text(log.replace(str(work), "$SRC"))
            print(f"chain {name} {label}: exit {res.returncode}, {diff.count(chr(10))} diff lines", file=sys.stderr)
    finally:
        git("worktree", "remove", "--force", str(work), cwd=srcroot)


def main(argv=None):
    ap = argparse.ArgumentParser(description="golden outputs of every source transform")
    ap.add_argument("out_dir")
    ap.add_argument("--srcroot", default=os.environ.get("SPEL_E3SM_SRCROOT"))
    ap.add_argument("--only", nargs="+", choices=sorted(PASSES))
    ap.add_argument("--chain", choices=sorted(CHAINS))
    args = ap.parse_args(argv)
    if args.chain:
        return run_chain(args.chain, Path(args.srcroot).resolve(), Path(args.out_dir).resolve())
    if args.srcroot:
        os.environ["SPEL_E3SM_SRCROOT"] = str(Path(args.srcroot).resolve())
    from spel.scripts.analysis_cache import load_analysis
    from spel.scripts.config import E3SM_SRCROOT

    src = str(E3SM_SRCROOT)
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    for name in args.only or PASSES:
        fut = load_analysis()
        vfs = VirtualFS()
        scratch = tempfile.mkdtemp(prefix=f"golden-{name}-")
        buf = io.StringIO()
        with vfs.active(), contextlib.redirect_stdout(buf):
            try:
                log = PASSES[name](fut, E3SM_SRCROOT, scratch) or []
            except Exception:  # noqa: BLE001
                log = ["EXCEPTION", traceback.format_exc()]
        d = out_root / name
        d.mkdir(exist_ok=True)
        for path, text in sorted(vfs.files.items()):
            rel = path.replace(src, "SRC").replace(scratch, "OUT").strip("/").replace("/", "__")
            (d / rel).write_text(text.replace(scratch, "$OUT"))  # the transform manifest lists the -o paths
        text = "\n".join(map(str, log)) + "\n--- stdout\n" + buf.getvalue()
        (out_root / f"{name}.log").write_text(text.replace(scratch, "$OUT").replace(src, "$SRC"))
        print(f"{name}: {len(vfs.files)} files", file=sys.stderr)


if __name__ == "__main__":
    main()
