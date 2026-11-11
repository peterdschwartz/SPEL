"""
Build a unit-test case as the shared library `libspelcase.so` for spel proptest.

The case's CMakeLists.txt (from `functional_unit_test.generate_cmake`) has the `SPEL_PYLIB`
option; the library is configured in `<case>/build-py` next to the case's own `build/`, so the
executable build is left alone. Other builds of the same case (another compiler, an OpenACC or
OpenMP target, a debug build with coverage) go to `build-py-<fc>-<acc>[-dbg]`, so they can be
used side by side (`[case] build`, `[compare] build`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from spel.proptest import shim
from spel.proptest.errors import ProptestError
from spel.proptest.inventory import Inventory


class BuildError(ProptestError):
    pass


BUILD_DIR = "build-py"
LIB_NAME = "libspelcase.so"
# Written after every successful build: sources newer than it are not in the library. It holds the
# directory of the netcdf-fortran the library was linked with (`runtime_environment`).
STAMP = ".spel-built"
ACC_TARGETS = ("OFF", "MULTICORE", "GPU", "OPENMP")


@dataclass(frozen=True)
class Variant:
    """One build of a case: compiler ("" = $SPEL_FC or gfortran), OpenACC target, debug."""

    fc: str = ""
    acc: str = "OFF"
    dbg: bool = False

    @classmethod
    def of(cls, table: Optional[dict[str, Any]] = None, fc: Optional[str] = None, acc: Optional[str] = None,
           dbg: Optional[bool] = None) -> "Variant":
        """From a `build = { fc, acc, dbg }` table, with the given values (command line) first."""
        table = table or {}
        v = cls(fc=str(fc or table.get("fc", "")), acc=str(acc or table.get("acc", "OFF")).upper(),
                dbg=bool(table.get("dbg", False) if dbg is None else dbg))
        if v.acc not in ACC_TARGETS:
            raise BuildError(f"acc = {v.acc!r}: one of {', '.join(ACC_TARGETS)}")
        return v

    @property
    def compiler(self) -> str:
        return self.fc or default_compiler()

    @property
    def dirname(self) -> str:
        if self.compiler == default_compiler() and self.acc == "OFF" and not self.dbg:
            return BUILD_DIR
        return f"{BUILD_DIR}-{Path(self.compiler).name}-{self.acc.lower()}{'-dbg' if self.dbg else ''}"

    def describe(self) -> str:
        return f"{self.compiler}, ACC={self.acc}{', debug' if self.dbg else ''}"


def default_compiler() -> str:
    """`$SPEL_FC`, else the default toolchain of the machine file that applies, else gfortran."""
    if os.environ.get("SPEL_FC"):
        return os.environ["SPEL_FC"]
    try:
        from spel.machines.machine import detect_machine

        machine = detect_machine(dict(os.environ))
        if machine is not None and machine.default_fc:
            return machine.default_fc
    except Exception:  # no usable machine file: the shell's compiler
        pass
    return "gfortran"


def build_dir(case_dir: Path, variant: Optional[Variant] = None) -> Path:
    return Path(case_dir) / (variant or Variant()).dirname


def library_path(case_dir: Path, variant: Optional[Variant] = None) -> Path:
    return build_dir(case_dir, variant) / LIB_NAME


def build(case_dir: Path, inv: Optional[Inventory] = None, fc: Optional[str] = None, dbg: bool = False,
          jobs: int = 0, quiet: bool = True, acc: str = "OFF", variant: Optional[Variant] = None) -> Path:
    """Generate the shim, (re)configure and make; returns the library path."""
    from spel.scripts.functional_unit_test import generate_cmake

    case_dir = Path(case_dir).resolve()
    variant = variant or Variant.of(fc=fc, acc=acc, dbg=dbg)
    inv = inv or Inventory.load(case_dir)
    shim.write(inv)
    cmakelists = case_dir / "CMakeLists.txt"
    if not cmakelists.is_file() or "SPEL_PYLIB" not in cmakelists.read_text():  # a case from before the option
        generate_cmake(sorted(p.name for p in case_dir.glob("*.F90")), case_dir)

    env = _environment(variant.compiler)
    compiler = shutil.which(variant.compiler, path=env.get("PATH"))
    if compiler is None:
        raise BuildError(f"Fortran compiler {variant.compiler} not found")
    bdir = build_dir(case_dir, variant)
    args = [
        "cmake", "-S", str(case_dir), "-B", str(bdir), "-DSPEL_PYLIB=ON", f"-DDBG={'ON' if variant.dbg else 'OFF'}",
        f"-DACC={variant.acc}", f"-DCMAKE_Fortran_COMPILER={compiler}",
    ]
    _run(args, env, quiet)
    _run(["cmake", "--build", str(bdir), "-j", str(jobs or os.cpu_count() or 4)], env, quiet)
    lib = library_path(case_dir, variant)
    if not lib.is_file():
        raise BuildError(f"build finished without {lib}")
    (bdir / STAMP).write_text(json.dumps({"compiler": compiler, "libdirs": _libdirs(env)}))
    return lib


def _libdirs(env: dict[str, str]) -> list[str]:
    """Where the netcdf-fortran of this build environment is (its `.so` is compiler specific)."""
    try:
        res = subprocess.run(["pkg-config", "--variable=libdir", "netcdf-fortran"], env=env, capture_output=True, text=True)
    except OSError:
        return []
    return [res.stdout.strip()] if res.returncode == 0 and res.stdout.strip() else []


def runtime_environment(case_dir: Path, variant: Optional[Variant] = None) -> dict[str, str]:
    """The environment a process loading this build's library starts with: the toolchain of
    its compiler when a machine file applies, and the library directories recorded at build
    time ahead of `LD_LIBRARY_PATH`. Two builds of a case (gfortran and nvfortran) can then run
    side by side, whatever the shell is set up for."""
    variant = variant or Variant()
    env = _environment(variant.compiler, quiet=True)
    try:
        dirs = json.loads((build_dir(case_dir, variant) / STAMP).read_text()).get("libdirs", [])
    except (OSError, ValueError):
        dirs = []
    if dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*dirs, *filter(None, [env.get("LD_LIBRARY_PATH", "")])])
    return env


def stale(case_dir: Path, inv: Optional[Inventory] = None, variant: Optional[Variant] = None) -> str:
    """Why the library has to be (re)built, "" when it is up to date: it is missing, a source of
    the case (or its analysis, fut.pkl) is newer than the last build, or the shim the inventory
    generates is not the one on disk."""
    case_dir = Path(case_dir)
    lib = library_path(case_dir, variant)
    if not lib.is_file():
        return "no library yet"
    stamp = build_dir(case_dir, variant) / STAMP
    built = (stamp if stamp.is_file() else lib).stat().st_mtime
    newer = sorted(p.name for p in [*case_dir.glob("*.F90"), case_dir / "fut.pkl"]
                   if p.is_file() and p.stat().st_mtime > built)
    if newer:
        return f"{', '.join(newer[:3])}{' ...' if len(newer) > 3 else ''} changed since the last build"
    if inv is not None and not shim.current(inv):
        return "the generated shim changed"
    return ""


def _environment(fc: str, quiet: bool = False) -> dict[str, str]:
    """The machine's toolchain for `fc` (`spel machine`), else the current environment."""
    import contextlib
    import io

    from spel.scripts.run_fut import toolchain_environment

    before = os.environ.get("SPEL_FC")
    os.environ["SPEL_FC"] = fc
    try:
        with contextlib.redirect_stdout(io.StringIO() if quiet else None) if quiet else contextlib.nullcontext():
            env = toolchain_environment()
    finally:
        if before is None:
            os.environ.pop("SPEL_FC", None)
        else:
            os.environ["SPEL_FC"] = before
    return dict(env) if env is not None else dict(os.environ)


def _run(args: list[str], env: dict[str, str], quiet: bool) -> None:
    res = subprocess.run(args, env=env, capture_output=quiet, text=True)
    if res.returncode != 0:
        tail = "\n".join(((res.stdout or "") + (res.stderr or "")).splitlines()[-40:])
        raise BuildError(f"{' '.join(args[:2])} failed ({res.returncode}):\n{tail}")
