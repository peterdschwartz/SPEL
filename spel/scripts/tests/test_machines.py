"""Machine files, E3SM config_machines.xml reading and toolchain environments (spel/machines)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from spel.machines import cli as mcli
from spel.machines.e3sm_machines import (
    E3SMMachines,
    E3SMMachinesError,
    expand_cime_value,
    match_attrib,
)
from spel.machines.machine import (
    ENV_VAR,
    Machine,
    MachineError,
    current_toolchain_environ,
    detect_machine,
    environ_after,
    load_machines,
)

CONFIG_MACHINES = """<?xml version="1.0"?>
<config_machines version="2.0">
  <machine MACH="pm-cpu">
    <DESC>fake perlmutter</DESC>
    <NODENAME_REGEX>$ENV{NERSC_HOST}:perlmutter</NODENAME_REGEX>
    <COMPILERS>intel,gnu,nvidia</COMPILERS>
    <MPILIBS>mpich</MPILIBS>
    <MPILIBS compiler="gnu">mpich,openmpi</MPILIBS>
    <CIME_OUTPUT_ROOT>$ENV{PSCRATCH}/e3sm_scratch/pm-cpu</CIME_OUTPUT_ROOT>
    <DIN_LOC_ROOT>/inputdata</DIN_LOC_ROOT>
    <CCSM_CPRNC>/tools/cprnc/cprnc</CCSM_CPRNC>
    <module_system type="module" allow_error="true">
      <init_path lang="sh">/opt/lmod/init/sh</init_path>
      <cmd_path lang="sh">module</cmd_path>
      <modules>
        <command name="unload">PrgEnv-gnu</command>
        <command name="unload">PrgEnv-nvidia</command>
      </modules>
      <modules compiler="gnu" mpilib="mpich">
        <command name="load">PrgEnv-gnu/8.6.0</command>
        <command name="load">gcc-native/14</command>
      </modules>
      <modules compiler="gnu" mpilib="openmpi">
        <command name="load">openmpi/5</command>
      </modules>
      <modules compiler="nvidia">
        <command name="load">PrgEnv-nvidia</command>
        <command name="load">nvidia/25.9</command>
      </modules>
      <modules compiler="!nvidia">
        <command name="load">craype-accel-host</command>
      </modules>
      <modules mpilib="mpich">
        <command name="load">cray-mpich/9.0.1</command>
        <command name="load">cray-netcdf-hdf5parallel/4.9.2.1</command>
        <command name="load">cmake/3.30.2</command>
      </modules>
      <modules DEBUG="TRUE">
        <command name="load">debug-only</command>
      </modules>
      <modules queue="debug">
        <command name="load">queue-only</command>
      </modules>
    </module_system>
    <environment_variables>
      <env name="HDF5_USE_FILE_LOCKING">FALSE</env>
      <env name="LD_LIBRARY_PATH">$ENV{CRAY_LD_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}</env>
      <env name="CASE_SPECIFIC">$CIME_OUTPUT_ROOT/x</env>
      <env name="UNSET_ME"></env>
    </environment_variables>
    <environment_variables compiler="gnu" mpilib="mpich">
      <env name="ADIOS2_ROOT">$SHELL{if [ -z "$ADIOS2_ROOT" ]; then echo /adios/gnu; else echo "$ADIOS2_ROOT"; fi}</env>
      <env name="WHO">$COMPILER-$MPILIB</env>
    </environment_variables>
    <environment_variables BUILD_THREADED="TRUE">
      <env name="OMP_STACKSIZE">128M</env>
    </environment_variables>
  </machine>
  <machine MACH="chrysalis">
    <NODENAME_REGEX>chr.*</NODENAME_REGEX>
    <COMPILERS>intel,gnu</COMPILERS>
    <MPILIBS>openmpi,impi</MPILIBS>
    <CCSM_CPRNC>/lcrc/cprnc</CCSM_CPRNC>
    <module_system type="module">
      <init_path lang="sh">/soft/lmod/init/sh</init_path>
      <modules compiler="intel">
        <command name="load">intel-oneapi-compilers/2025.2.0</command>
      </modules>
      <modules compiler="intel" mpilib="openmpi">
        <command name="load">openmpi/4.1.8</command>
        <command name="load">netcdf-fortran/4.6.2</command>
      </modules>
    </module_system>
  </machine>
</config_machines>
"""


@pytest.fixture
def srcroot(tmp_path: Path) -> Path:
    root = tmp_path / "E3SM"
    (root / "components/elm/src").mkdir(parents=True)
    xml = root / "cime_config/machines/config_machines.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text(CONFIG_MACHINES)
    return root


@pytest.fixture
def machines(srcroot: Path) -> E3SMMachines:
    return E3SMMachines.from_srcroot(srcroot)


# --- config_machines.xml ----------------------------------------------------------------------


def test_match_attrib_like_cime():
    assert match_attrib("gnu", "gnu")
    assert not match_attrib("gnu", "gn")  # whole value
    assert match_attrib("gnu", "!nvidia")
    assert not match_attrib("nvidia", "!nvidia")
    assert match_attrib(True, "TRUE") and not match_attrib(False, "TRUE")


def test_expand_cime_value():
    assert expand_cime_value("$ENV{X}:$ENV{Y}") == "${X}:${Y}"
    assert expand_cime_value("$SHELL{dirname $(which nf-config)}") == "$(dirname $(which nf-config))"
    assert expand_cime_value("a b") == '"a b"'


def test_machines_probe_env_and_hostname(machines: E3SMMachines):
    assert machines.names() == ["pm-cpu", "chrysalis"]
    assert machines.probe(["login36"], {"NERSC_HOST": "perlmutter"}) == "pm-cpu"
    assert machines.probe(["chrlogin1.lcrc.anl.gov"], {}) == "chrysalis"
    assert machines.probe(["laptop"], {"NERSC_HOST": "cori"}) is None


def test_machine_values(machines: E3SMMachines):
    pm = machines.get("pm-cpu")
    assert pm.cprnc() == "/tools/cprnc/cprnc"
    assert pm.default_compiler() == "intel"
    assert pm.mpilibs("gnu") == ["mpich", "openmpi"]
    assert pm.mpilibs("nvidia") == ["mpich"]
    assert pm.module_init_sh() == "/opt/lmod/init/sh"
    with pytest.raises(E3SMMachinesError):
        machines.get("nope")


def test_shell_env_matches_blocks_like_cime(machines: E3SMMachines):
    env = machines.get("pm-cpu").shell_env("gnu")
    assert env.init == ". /opt/lmod/init/sh"
    assert env.module_lines == [
        "module unload PrgEnv-gnu PrgEnv-nvidia",
        "module load PrgEnv-gnu/8.6.0 gcc-native/14",
        "module load craype-accel-host",  # compiler="!nvidia"
        "module load cray-mpich/9.0.1 cray-netcdf-hdf5parallel/4.9.2.1 cmake/3.30.2",
    ]  # DEBUG="TRUE" and queue="debug" blocks do not apply
    assert "export HDF5_USE_FILE_LOCKING=FALSE" in env.env_lines
    assert "export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}" in env.env_lines
    assert "unset UNSET_ME" in env.env_lines
    assert 'export ADIOS2_ROOT=$(if [ -z "$ADIOS2_ROOT" ]; then echo /adios/gnu; else echo "$ADIOS2_ROOT"; fi)' in env.env_lines
    assert "export WHO=gnu-mpich" in env.env_lines
    assert not any("OMP_STACKSIZE" in l for l in env.env_lines)  # BUILD_THREADED
    assert env.skipped == ["CASE_SPECIFIC=$CIME_OUTPUT_ROOT/x"]
    assert any(l.startswith("# skipped CASE_SPECIFIC") for l in env.env_lines)

    nv = machines.get("pm-cpu").shell_env("nvidia")
    assert "module load PrgEnv-nvidia nvidia/25.9" in nv.module_lines
    assert "module load craype-accel-host" not in nv.module_lines
    assert "export WHO=gnu-mpich" not in nv.env_lines

    openmpi = machines.get("pm-cpu").shell_env("gnu", "openmpi")
    assert "module load openmpi/5" in openmpi.module_lines
    assert not any("cray-mpich" in l for l in openmpi.module_lines)


# --- machine files --------------------------------------------------------------------------


def write_machine(dir: Path, name: str, text: str) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    p = dir / f"{name}.toml"
    p.write_text(text)
    return p


LAPTOP = """
name = "laptop"
scratch = "$HOME/scratch"
default_fc = "nvfortran"
make_jobs = 4
[detect]
hostname = "mylaptop.*"
[casegen]
compiler = "gnu"
[toolchains.gfortran]
commands = ["export MARK=gnu"]
pkg_config_path = ["$HOME/.local/gcc/netcdf-fortran/lib/pkgconfig"]
[toolchains.nvfortran]
env_file = "~/nvhpc-env.sh"
ld_library_path = ["/opt/nvhpc/lib"]
[toolchains.nvfortran.env]
ACC_DEVICE_TYPE = "multicore"
"""


def test_machine_from_toml_and_script(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = write_machine(tmp_path / "m", "laptop", LAPTOP)
    m = Machine.from_toml(path)
    assert m.name == "laptop" and m.default_fc == "nvfortran" and m.make_jobs == 4
    assert m.scratch_dir() == tmp_path / "scratch"
    assert m.unit_tests_dir() == tmp_path / "scratch/spel-unit-tests"
    assert m.cprnc_path() is None  # no E3SM machine
    assert m.matches_host(["mylaptop.home"], {})
    assert not m.matches_host(["desktop"], {})
    assert m.casegen == {"compiler": "gnu"}

    gnu = m.toolchain_script("gfortran")
    assert "export MARK=gnu" in gnu
    assert 'export PKG_CONFIG_PATH="$HOME/.local/gcc/netcdf-fortran/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"' in gnu
    assert "export SPEL_FC=gfortran" in gnu
    assert 'export MAKEFLAGS="${MAKEFLAGS:--j4}"' in gnu

    nv = m.toolchain_script()  # the default toolchain
    assert f"source {tmp_path}/nvhpc-env.sh" in nv
    assert 'export ACC_DEVICE_TYPE="${ACC_DEVICE_TYPE:-multicore}"' in nv  # a default: GPU runs override it
    assert 'export LD_LIBRARY_PATH="/opt/nvhpc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' in nv
    with pytest.raises(MachineError):
        m.toolchain("ifx")


def test_machine_file_validation(tmp_path: Path):
    with pytest.raises(MachineError, match="unknown keys"):
        Machine.from_dict({"name": "x", "scratchh": "/s"})
    with pytest.raises(MachineError, match="unknown keys"):
        Machine.from_dict({"name": "x", "toolchains": {"gfortran": {"modules": []}}})
    with pytest.raises(MachineError, match="default_fc"):
        Machine.from_dict({"name": "x", "default_fc": "ifx", "toolchains": {"gfortran": {}}})
    bad = write_machine(tmp_path, "bad", "name = \n")
    with pytest.raises(MachineError):
        Machine.from_toml(bad)
    # the file name is the default name
    assert Machine.from_toml(write_machine(tmp_path, "byname", "scratch = '/s'\n")).name == "byname"


def test_e3sm_backed_machine(srcroot: Path, tmp_path: Path):
    text = """
name = "pm"
e3sm_machine = "pm-cpu"
[toolchains.gfortran]
e3sm_compiler = "gnu"
pkg_config_path = ["$CRAY_MPICH_DIR/lib/pkgconfig"]
[toolchains.nvfortran]
e3sm_compiler = "nvidia"
[toolchains.ifx]
commands = ["module load intel"]
"""
    m = Machine.from_toml(write_machine(tmp_path / "m", "pm", text))
    assert m.cprnc_path(srcroot) == "/tools/cprnc/cprnc"
    # no [detect]: E3SM's NODENAME_REGEX decides
    assert m.matches_host(["login1"], {"NERSC_HOST": "perlmutter"}, srcroot)
    assert not m.matches_host(["login1"], {"NERSC_HOST": "other"}, srcroot)
    script = m.toolchain_script("gfortran", srcroot)
    assert ". /opt/lmod/init/sh" in script
    assert "module load PrgEnv-gnu/8.6.0 gcc-native/14" in script
    assert script.index("module load") < script.index("export PKG_CONFIG_PATH=")
    assert "export SPEL_FC=gfortran" in script
    assert "module load intel" in m.toolchain_script("ifx", srcroot)
    # an E3SM toolchain without a config_machines.xml is an error that says what to do
    m2 = Machine.from_dict({"name": "pm", "e3sm_machine": "pm-cpu", "toolchains": {"gfortran": {"e3sm_compiler": "gnu"}}})
    with pytest.raises(MachineError, match="srcroot"):
        m2.toolchain_script("gfortran", tmp_path / "no-e3sm")


def test_environ_after_runs_the_script(monkeypatch):
    env = environ_after("export SPEL_TEST_MARK=42\necho noise-to-stderr\n", base={"PATH": os.environ["PATH"]})
    assert env["SPEL_TEST_MARK"] == "42"
    with pytest.raises(MachineError, match="toolchain setup failed"):
        environ_after("false\n")


def test_detect_machine_precedence(tmp_path: Path, monkeypatch):
    user = tmp_path / "machines"
    write_machine(user, "laptop", LAPTOP)
    write_machine(user, "perlmutter", 'name = "perlmutter"\ndescription = "user override"\n[detect]\nenv = { NERSC_HOST = "perlmutter" }\n')
    monkeypatch.setenv("HOME", str(tmp_path))
    all_machines = load_machines(user)
    assert all_machines["perlmutter"].description == "user override"  # replaces the bundled file
    assert "chrysalis" in all_machines  # bundled
    assert detect_machine({}, ["mylaptop.home"], user).name == "laptop"
    assert detect_machine({"NERSC_HOST": "perlmutter"}, ["login1"], user).name == "perlmutter"
    assert detect_machine({}, ["unknown"], user) is None
    assert detect_machine({ENV_VAR: "laptop"}, ["unknown"], user).name == "laptop"
    for off in ("none", "", "0"):
        assert detect_machine({ENV_VAR: off}, ["mylaptop"], user) is None
    with pytest.raises(MachineError, match="no such machine"):
        detect_machine({ENV_VAR: "nope"}, [], user)


def test_current_toolchain_environ(tmp_path: Path, monkeypatch):
    user = tmp_path / "machines"
    write_machine(user, "laptop", LAPTOP)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SPEL_MACHINES_DIR", str(user))
    base = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "SPEL_MACHINES_DIR": str(user), ENV_VAR: "laptop", "SPEL_FC": "gfortran"}
    machine, fc, env = current_toolchain_environ(base)
    assert machine.name == "laptop" and fc == "gfortran"
    assert env["MARK"] == "gnu" and env["SPEL_FC"] == "gfortran" and env["MAKEFLAGS"] == "-j4"
    assert env["PKG_CONFIG_PATH"].startswith(f"{tmp_path}/.local/gcc/netcdf-fortran/lib/pkgconfig")
    # an unknown compiler: the machine is known but there is no environment for it
    machine, fc, env = current_toolchain_environ(dict(base, SPEL_FC="flang"))
    assert machine.name == "laptop" and fc == "flang" and env is None
    assert current_toolchain_environ(dict(base, **{ENV_VAR: "none"})) == (None, None, None)


# --- spel machine CLI -------------------------------------------------------------------------


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_cli_init_from_template_and_from_machine(tmp_path: Path, monkeypatch, capsys):
    user = tmp_path / "machines"
    monkeypatch.setenv("SPEL_MACHINES_DIR", str(user))
    mcli.cmd_init(Args(name="mybox", base=None, force=False))
    text = (user / "mybox.toml").read_text()
    assert 'name = "mybox"' in text and "[toolchains.gfortran]" in text
    m = Machine.from_toml(user / "mybox.toml")  # the template parses and loads
    assert set(m.toolchains) == {"gfortran", "nvfortran"}
    with pytest.raises(MachineError, match="exists"):
        mcli.cmd_init(Args(name="mybox", base=None, force=False))
    mcli.cmd_init(Args(name="pm2", base="perlmutter", force=False))
    pm2 = Machine.from_toml(user / "pm2.toml")
    assert pm2.name == "pm2" and pm2.e3sm_machine == "pm-cpu"
    with pytest.raises(MachineError, match="no such machine"):
        mcli.cmd_init(Args(name="x", base="nope", force=False))
    with pytest.raises(MachineError, match="name"):
        mcli.cmd_init(Args(name="bad name", base=None, force=False))


def test_cli_vars_and_show(tmp_path: Path, monkeypatch, capsys):
    user = tmp_path / "machines"
    write_machine(user, "laptop", LAPTOP)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SPEL_MACHINES_DIR", str(user))
    monkeypatch.setenv(ENV_VAR, "laptop")
    monkeypatch.setattr(mcli, "_spel_root", lambda: tmp_path / "SPEL")
    mcli.cmd_vars(Args())
    out = capsys.readouterr().out
    assert "SPEL_MACHINE=laptop" in out
    assert f"SPEL_SCRATCH={tmp_path}/scratch" in out
    assert "SPEL_CASEGEN_COMPILER=gnu" in out and "SPEL_MAKE_JOBS=4" in out
    mcli.cmd_show(Args())
    out = capsys.readouterr().out
    assert "laptop (SPEL_MACHINE=laptop)" in out and "nvfortran" in out
    monkeypatch.setenv(ENV_VAR, "none")
    mcli.cmd_show(Args())
    assert "Machine     : none" in capsys.readouterr().out


def test_cli_setup_moves_unit_tests_to_scratch(tmp_path: Path, monkeypatch, capsys):
    user = tmp_path / "machines"
    write_machine(user, "laptop", LAPTOP)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SPEL_MACHINES_DIR", str(user))
    monkeypatch.setenv(ENV_VAR, "laptop")
    root = tmp_path / "SPEL"
    monkeypatch.setattr(mcli, "_spel_root", lambda: root)
    (root / "unit-tests/input-data/canflux").mkdir(parents=True)
    (root / "unit-tests/canflux").mkdir()
    (root / "unit-tests/canflux/main.F90").write_text("x")
    scratch = tmp_path / "scratch"
    with pytest.raises(MachineError, match="does not exist"):
        mcli.cmd_setup(Args(move=False))
    scratch.mkdir()
    with pytest.raises(MachineError, match="--move"):
        mcli.cmd_setup(Args(move=False))
    mcli.cmd_setup(Args(move=True))
    ut = root / "unit-tests"
    assert ut.is_symlink() and ut.resolve() == scratch / "spel-unit-tests"
    assert (ut / "input-data").is_symlink() and (ut / "input-data").resolve() == scratch / "spel-input-data"
    assert (scratch / "spel-unit-tests/canflux/main.F90").read_text() == "x"
    assert (scratch / "spel-input-data/canflux").is_dir()
    mcli.cmd_setup(Args(move=False))  # idempotent
    assert "(already)" in capsys.readouterr().out
