"""Build and run a functional unit test, then validate it against ELM's data."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence


def run_unit_test(case_dir: Path, exe_args: Sequence[str] = (), validate: bool = True) -> int:
    """
    `spel run`: configure/build (check_config.sh), run build/elmtest, then
    (validate=True) compare its fut-outputs with ELM's spel-outputs and check
    the static access analysis (validate_access.validate_case).
    Returns the validation exit code (0 = pass).
    """
    from spel.scripts.config import input_data_dir
    from spel.scripts.validate_access import validate_case

    case_dir = Path(case_dir).resolve()
    subprocess.run(["./check_config.sh"], check=True, cwd=case_dir)
    subprocess.run(["./build/elmtest", *exe_args], check=True, cwd=case_dir)
    if not validate:
        return 0
    return validate_case(case_dir, Path(input_data_dir) / case_dir.name)


PROFILE_TOOLS = ("nsys", "ncu", "none")


def find_nvidia_tool(name: str) -> str:
    """`name` (nsys/ncu) from PATH, else from the NVHPC install next to nvfortran."""
    import shutil

    if found := shutil.which(name):
        return found
    if nvf := shutil.which("nvfortran"):
        cand = Path(nvf).resolve().parent / name
        if cand.exists():
            return str(cand)
    raise SystemExit(f"Error: {name} not found on PATH or next to nvfortran")


def profile_command(
    tool: str,
    exe_args: Sequence[str],
    output: Path,
    stats: bool = False,
    tool_args: Sequence[str] = (),
) -> list[str]:
    """Command line that runs build/elmtest under `tool` (nsys, ncu or none)."""
    exe = ["./build/elmtest", *exe_args]
    if tool == "none":
        return exe
    if tool == "nsys":
        cmd = [
            find_nvidia_tool("nsys"), "profile",
            "--trace=cuda,nvtx,openacc,osrt",
            "--force-overwrite=true",
            f"--output={output}",
        ]
        if stats:
            cmd.append("--stats=true")
    elif tool == "ncu":
        # Kernels inside the routines under test only (not IO/duplicate_sets)
        cmd = [
            find_nvidia_tool("ncu"),
            "--nvtx", "--nvtx-include", "compute/",
            "--force-overwrite",
            "--export", str(output),
        ]
    else:
        raise ValueError(f"unknown profiler {tool!r}; use one of {PROFILE_TOOLS}")
    return [*cmd, *tool_args, *exe]


def profile_unit_test(
    case_dir: Path,
    exe_args: Sequence[str] = (),
    tool: str = "nsys",
    steps: int | None = None,
    output: Path | None = None,
    stats: bool = False,
    tool_args: Sequence[str] = (),
) -> Path | None:
    """
    `spel profile`: build (check_config.sh), then run build/elmtest with
    SPEL_PROFILE=1 (no device->host output copies, no output files) and, for
    steps, SPEL_MAX_STEPS, under Nsight Systems/Compute. Returns the report path.
    """
    import os
    import shlex

    case_dir = Path(case_dir).resolve()
    if output is None:
        nsets = exe_args[0] if exe_args else "1"
        output = case_dir.parent / "profiles" / case_dir.name / f"{case_dir.name}-nsets{nsets}-{tool}"
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(["./check_config.sh"], check=True, cwd=case_dir)
    env = dict(os.environ, SPEL_PROFILE="1")
    if steps is not None:
        env["SPEL_MAX_STEPS"] = str(steps)
    cmd = profile_command(tool, exe_args, output, stats=stats, tool_args=tool_args)
    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=case_dir, env=env)
    if tool == "none":
        return None
    ext = ".nsys-rep" if tool == "nsys" else ".ncu-rep"
    report = output if output.suffix == ext else output.with_name(output.name + ext)
    print(f"Profile report: {report}")
    return report
