"""
Per-machine configuration of SPEL.

A machine file (TOML) says what SPEL needs to know about a computer that E3SM's
``config_machines.xml`` does not: where the unit tests should live (scratch), which Fortran
compiler to use by default, and the *toolchains*: for each compiler, how to set up a shell so
that the compiler, ``cmake`` and a matching ``netcdf-fortran`` (through ``pkg-config``) are
found. A toolchain usually just names the E3SM compiler (``e3sm_compiler = "gnu"``): the modules
and environment variables then come from the E3SM checkout's ``config_machines.xml``, plus a
little glue (``pkg_config_path``, ``env``). It can also be explicit shell lines (``commands``) or
a file to source (``env_file``).

Machine files are looked up in this order (later wins on the same name):

1. ``spel/machines/*.toml`` -- bundled (perlmutter, chrysalis)
2. ``$SPEL_MACHINES_DIR`` or ``~/.config/spel/machines/*.toml`` -- the user's own

The current machine is ``$SPEL_MACHINE`` (name, or ``none`` to disable), otherwise the first
machine whose ``[detect]`` rules (or, failing that, the E3SM machine's ``NODENAME_REGEX``)
match this host. ``spel machine`` shows and checks all of it.
"""

from __future__ import annotations

import os
import re
import shlex
import socket
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

from spel.machines.e3sm_machines import (
    E3SMMachine,
    E3SMMachines,
    E3SMMachinesError,
    local_hostnames,
)

ENV_VAR = "SPEL_MACHINE"
MACHINES_DIR_VAR = "SPEL_MACHINES_DIR"
BUNDLED_DIR = Path(__file__).parent
TEMPLATE = BUNDLED_DIR / "template.toml"
UNIT_TESTS_DIRNAME = "spel-unit-tests"
INPUT_DATA_DIRNAME = "spel-input-data"


class MachineError(Exception):
    pass


def user_machines_dir() -> Path:
    return Path(os.environ.get(MACHINES_DIR_VAR, "~/.config/spel/machines")).expanduser()


def expand(value: str) -> str:
    """`$VAR`, `${VAR}` and `~` in a machine-file value."""
    return os.path.expanduser(os.path.expandvars(value))


@dataclass
class Toolchain:
    fc: str
    e3sm_compiler: Optional[str] = None
    mpilib: Optional[str] = None
    commands: list[str] = field(default_factory=list)
    env_file: Optional[str] = None
    pkg_config_path: list[str] = field(default_factory=list)
    ld_library_path: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""

    @classmethod
    def from_dict(cls, fc: str, data: Mapping) -> "Toolchain":
        known = {
            "e3sm_compiler", "mpilib", "commands", "env_file", "pkg_config_path",
            "ld_library_path", "env", "description",
        }
        unknown = set(data) - known
        if unknown:
            raise MachineError(f"toolchain {fc}: unknown keys {sorted(unknown)}")
        return cls(
            fc=fc,
            e3sm_compiler=data.get("e3sm_compiler"),
            mpilib=data.get("mpilib"),
            commands=list(data.get("commands", [])),
            env_file=data.get("env_file"),
            pkg_config_path=list(data.get("pkg_config_path", [])),
            ld_library_path=list(data.get("ld_library_path", [])),
            env={k: str(v) for k, v in data.get("env", {}).items()},
            description=data.get("description", ""),
        )

    def source(self) -> str:
        if self.e3sm_compiler:
            return f"E3SM compiler '{self.e3sm_compiler}'" + (f" ({self.mpilib})" if self.mpilib else "")
        if self.env_file:
            return f"env_file {self.env_file}"
        if self.commands:
            return f"{len(self.commands)} shell command(s)"
        return "current environment"


@dataclass
class Machine:
    name: str
    path: Optional[Path] = None
    description: str = ""
    e3sm_machine: Optional[str] = None
    config_machines: Optional[str] = None  # override of <srcroot>/cime_config/machines/config_machines.xml
    scratch: Optional[str] = None
    default_fc: Optional[str] = None
    cprnc: Optional[str] = None
    make_jobs: Optional[int] = None
    detect: dict = field(default_factory=dict)
    casegen: dict = field(default_factory=dict)
    toolchains: dict[str, Toolchain] = field(default_factory=dict)
    _e3sm: Optional[E3SMMachine] = field(default=None, repr=False)

    # --- loading --------------------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: Path) -> "Machine":
        try:
            data = tomllib.loads(Path(path).read_text())
        except (OSError, tomllib.TOMLDecodeError) as err:
            raise MachineError(f"{path}: {err}") from err
        return cls.from_dict(data, path=Path(path))

    @classmethod
    def from_dict(cls, data: Mapping, path: Optional[Path] = None) -> "Machine":
        known = {
            "name", "description", "e3sm_machine", "config_machines", "scratch", "default_fc",
            "cprnc", "make_jobs", "detect", "casegen", "toolchains",
        }
        unknown = set(data) - known
        if unknown:
            raise MachineError(f"{path or 'machine'}: unknown keys {sorted(unknown)}")
        name = data.get("name") or (path.stem if path else None)
        if not name:
            raise MachineError(f"{path}: a machine needs a name")
        toolchains = {
            fc: Toolchain.from_dict(fc, tc) for fc, tc in data.get("toolchains", {}).items()
        }
        default_fc = data.get("default_fc")
        if default_fc and default_fc not in toolchains:
            raise MachineError(f"{path}: default_fc '{default_fc}' is not a toolchain")
        return cls(
            name=name,
            path=path,
            description=data.get("description", ""),
            e3sm_machine=data.get("e3sm_machine"),
            config_machines=data.get("config_machines"),
            scratch=data.get("scratch"),
            default_fc=default_fc or (next(iter(toolchains)) if toolchains else None),
            cprnc=data.get("cprnc"),
            make_jobs=data.get("make_jobs"),
            detect=dict(data.get("detect", {})),
            casegen=dict(data.get("casegen", {})),
            toolchains=toolchains,
        )

    # --- derived values -------------------------------------------------------------------

    def scratch_dir(self) -> Optional[Path]:
        return Path(expand(self.scratch)) if self.scratch else None

    def unit_tests_dir(self) -> Optional[Path]:
        s = self.scratch_dir()
        return s / UNIT_TESTS_DIRNAME if s else None

    def input_data_dir(self) -> Optional[Path]:
        s = self.scratch_dir()
        return s / INPUT_DATA_DIRNAME if s else None

    def config_machines_path(self, srcroot: Optional[Path] = None) -> Optional[Path]:
        if self.config_machines:
            return Path(expand(self.config_machines))
        if srcroot is None:
            srcroot = _srcroot()
        if srcroot is None:
            return None
        from spel.machines.e3sm_machines import config_machines_path

        return config_machines_path(srcroot)

    def e3sm(self, srcroot: Optional[Path] = None) -> Optional[E3SMMachine]:
        """The E3SM machine entry (None when the machine file names none or the XML is missing)."""
        if self._e3sm is not None:
            return self._e3sm
        if not self.e3sm_machine:
            return None
        path = self.config_machines_path(srcroot)
        if path is None:
            return None
        self._e3sm = E3SMMachines(path).get(self.e3sm_machine)
        return self._e3sm

    def cprnc_path(self, srcroot: Optional[Path] = None) -> Optional[str]:
        if self.cprnc:
            return expand(self.cprnc)
        try:
            e3sm = self.e3sm(srcroot)
        except E3SMMachinesError:
            return None
        return e3sm.cprnc() if e3sm else None

    def toolchain(self, fc: Optional[str] = None) -> Toolchain:
        fc = fc or self.default_fc
        if not fc:
            raise MachineError(f"machine {self.name} has no toolchains")
        try:
            return self.toolchains[fc]
        except KeyError:
            raise MachineError(
                f"machine {self.name} has no toolchain '{fc}' (has: {', '.join(self.toolchains) or 'none'})"
            ) from None

    # --- detection ------------------------------------------------------------------------

    def matches_host(
        self,
        hostnames: Optional[Iterable[str]] = None,
        environ: Optional[Mapping[str, str]] = None,
        srcroot: Optional[Path] = None,
    ) -> bool:
        hosts = list(hostnames) if hostnames is not None else local_hostnames()
        env = environ if environ is not None else os.environ
        if self.detect:
            regex = self.detect.get("hostname")
            if regex and any(re.match(regex, h) for h in hosts if h):
                return True
            for var, pattern in (self.detect.get("env") or {}).items():
                value = env.get(var)
                if value is not None and re.match(str(pattern) + "$", value):
                    return True
            return False
        try:
            e3sm = self.e3sm(srcroot)
        except E3SMMachinesError:
            return False
        return bool(e3sm and e3sm.matches_host(hosts, env))

    # --- toolchain environment ------------------------------------------------------------

    def toolchain_script(self, fc: Optional[str] = None, srcroot: Optional[Path] = None) -> str:
        """A `source`-able sh script that sets up the toolchain's compiler, netcdf and tools."""
        tc = self.toolchain(fc)
        lines = [
            f"# SPEL toolchain '{tc.fc}' of machine '{self.name}'"
            + (f" ({self.path})" if self.path else ""),
            f"# {tc.description}" if tc.description else "",
        ]
        if tc.e3sm_compiler:
            try:
                e3sm = self.e3sm(srcroot)
            except E3SMMachinesError as err:
                e3sm, reason = None, str(err)
            else:
                reason = "the machine file sets no e3sm_machine"
            if e3sm is None:
                raise MachineError(
                    f"toolchain {tc.fc} takes its modules from E3SM compiler '{tc.e3sm_compiler}', "
                    f"but {reason}: point SPEL at an E3SM checkout (`spel config --set-srcroot`, "
                    "or `config_machines` in the machine file) or give the toolchain explicit "
                    "`commands`/`env_file`"
                )
            shell_env = e3sm.shell_env(tc.e3sm_compiler, tc.mpilib)
            lines.append(f"# modules of E3SM machine '{e3sm.name}', compiler '{tc.e3sm_compiler}' ({e3sm.path})")
            lines += shell_env.lines()
        if tc.env_file:
            lines.append(f"source {shlex.quote(expand(tc.env_file))}")
        lines += tc.commands
        if tc.pkg_config_path:
            joined = ":".join(tc.pkg_config_path)
            lines.append(f'export PKG_CONFIG_PATH="{joined}${{PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}}"')
        if tc.ld_library_path:
            joined = ":".join(tc.ld_library_path)
            lines.append(f'export LD_LIBRARY_PATH="{joined}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"')
        for key, value in tc.env.items():
            # defaults: a value already set wins (e.g. ACC_DEVICE_TYPE=nvidia for GPU runs
            # over the toolchain's multicore default)
            if re.fullmatch(r"[\w./:+-]*", value):
                lines.append(f'export {key}="${{{key}:-{value}}}"')
            else:
                lines.append(f"export {key}={shlex.quote(value)}")
        lines.append(f"export SPEL_FC={shlex.quote(tc.fc)}")
        if self.make_jobs:
            lines.append(f'export MAKEFLAGS="${{MAKEFLAGS:--j{int(self.make_jobs)}}}"')
        return "\n".join(line for line in lines if line is not None and line != "") + "\n"

    def toolchain_environ(
        self,
        fc: Optional[str] = None,
        srcroot: Optional[Path] = None,
        base: Optional[Mapping[str, str]] = None,
    ) -> dict[str, str]:
        """The environment after sourcing `toolchain_script` in bash (for subprocess `env=`)."""
        script = self.toolchain_script(fc, srcroot)
        return environ_after(script, base)


def environ_after(script: str, base: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Run `script` in bash and return the resulting environment."""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", prefix="spel-toolchain-", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        # the script's own output (module messages) goes to stderr; stdout is `env -0` only
        proc = subprocess.run(
            ["bash", "-c", '{ . "$1"; } 1>&2 && env -0', "bash", path],
            capture_output=True,
            env=dict(base) if base is not None else None,
        )
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        raise MachineError(
            "toolchain setup failed:\n" + proc.stderr.decode(errors="replace").strip()
        )
    env: dict[str, str] = {}
    for item in proc.stdout.split(b"\0"):
        if not item:
            continue
        key, _, value = item.decode(errors="replace").partition("=")
        env[key] = value
    return env


# --- registry ---------------------------------------------------------------------------------


def _srcroot() -> Optional[Path]:
    root = os.environ.get("SPEL_E3SM_SRCROOT")
    if root:
        return Path(root).expanduser()
    try:
        from spel.scripts.config import E3SM_SRCROOT

        return Path(E3SM_SRCROOT)
    except Exception:  # pragma: no cover - config import failure
        return None


def machine_files(user_dir: Optional[Path] = None) -> list[Path]:
    """Bundled machine files, then the user's (sorted by name within each)."""
    files = sorted(p for p in BUNDLED_DIR.glob("*.toml") if p.name != TEMPLATE.name)
    udir = user_dir if user_dir is not None else user_machines_dir()
    if udir.is_dir():
        files += sorted(udir.glob("*.toml"))
    return files


def load_machines(user_dir: Optional[Path] = None) -> dict[str, Machine]:
    machines: dict[str, Machine] = {}
    for path in machine_files(user_dir):
        m = Machine.from_toml(path)
        machines[m.name] = m  # a user file replaces a bundled one of the same name
    return machines


def detect_machine(
    environ: Optional[Mapping[str, str]] = None,
    hostnames: Optional[Iterable[str]] = None,
    user_dir: Optional[Path] = None,
    srcroot: Optional[Path] = None,
) -> Optional[Machine]:
    """`$SPEL_MACHINE` (a name, or `none`), else the first machine file matching this host."""
    env = environ if environ is not None else os.environ
    machines = load_machines(user_dir)
    forced = env.get(ENV_VAR)
    if forced is not None:
        forced = forced.strip()
        if forced.lower() in ("", "none", "off", "0"):
            return None
        if forced not in machines:
            raise MachineError(
                f"{ENV_VAR}={forced}: no such machine file (known: {', '.join(machines) or 'none'}; "
                f"`spel machine init {forced}` creates one)"
            )
        return machines[forced]
    hosts = list(hostnames) if hostnames is not None else local_hostnames()
    for m in machines.values():
        if m.matches_host(hosts, env, srcroot):
            return m
    return None


def current_toolchain_environ(
    environ: Optional[Mapping[str, str]] = None, fc: Optional[str] = None
) -> tuple[Optional[Machine], Optional[str], Optional[dict[str, str]]]:
    """(machine, fc, environment) for building a unit test here, or (None, None, None)
    when no machine file applies. `fc` defaults to `$SPEL_FC`, then the machine's default."""
    env = dict(environ if environ is not None else os.environ)
    machine = detect_machine(env)
    if machine is None:
        return None, None, None
    fc = fc or env.get("SPEL_FC") or machine.default_fc
    if not fc or fc not in machine.toolchains:
        return machine, fc, None
    return machine, fc, machine.toolchain_environ(fc, base=env)
