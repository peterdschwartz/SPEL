"""
Scalars that change every time step and that a unit test must see per step.

Module variables are captured once, in the constants file; derived-type
components every step. The time-step state is module scalars, so it is
captured separately, per step, through `spel_t_<name>(1)` arrays in
ReadWriteMod (like elm_drv's scalar arguments, `spel_drv_<name>`):

  * spel_timeMod's `spel_nstep`, `spel_first_restart_step`, `spel_step_size`
    and `spel_rad_step_size`, behind the unit-test stand-ins of
    elm_time_manager's getters (TIME_SHIMS; the capture module sets them from
    get_nstep(), is_first_restart_step(), get_step_size(), get_rad_step_size())
  * every scalar of the branch's timeInfoMod (nstep_mod, secs_curr, doalb, ...)

The ELM capture module sets spel_t_<name>(1) before each capture; the unit
test copies them back after reading each step's inputs (main.F90 !#TIME_SYNC_INIT, !#TIME_SYNC_STEP).
Logicals are stored as integers (0/1).
"""

from __future__ import annotations

from dataclasses import dataclass

# elm_time_manager names a unit test gets from spel_timeMod instead
TIME_SHIMS = {
    name: "spel_timemod"
    for name in (
        "is_first_step",
        "is_first_restart_step",
        "get_nstep",
        "get_step_size",
        "get_step_size_real",
        "get_rad_step_size",
        "is_end_curr_day",
        "get_curr_date",
    )
}
SHIM_MODULE = "spel_timeMod"
TIME_MODULE = "timeinfomod"


@dataclass(frozen=True)
class TimeScalar:
    name: str  # the variable in the unit test
    module: str  # its module
    type: str  # real | integer | logical
    capture: str  # its value in ELM's capture module
    capture_module: str  # module the capture expression comes from
    # value when the capture lacks the array (captured before it was added): the unit test
    # reads it only if present, and a non-positive value means "use the fallback"
    fallback: str = ""

    @property
    def array(self) -> str:
        return f"spel_t_{self.name}"

    @property
    def io_type(self) -> str:
        return "real" if self.type == "real" else "integer"

    def to_array(self) -> str:
        """capture: spel_t_<name>(1) = <value>"""
        if self.type == "logical":
            return f"{self.array}(1) = merge(1, 0, {self.capture})"
        return f"{self.array}(1) = {self.capture}"

    def from_array(self) -> str:
        """unit test: <name> = spel_t_<name>(1)"""
        if self.type == "logical":
            return f"{self.name} = {self.array}(1) /= 0"
        return f"{self.name} = {self.array}(1)"


def unit_test_time_scalars(unit_test) -> list[TimeScalar]:
    """The case's per-step scalars (see time_scalars)."""
    return time_scalars(
        getattr(unit_test, "module_dict", {}),
        getattr(unit_test, "non_parameter_global_vars", {}),
        getattr(unit_test, "step_written_globals", ()),
    )


def time_scalars(module_dict: dict, globals_: dict | None = None, step_written=()) -> list[TimeScalar]:
    """
    spel_timeMod's four, every timeInfoMod scalar, and the scalars among the
    case's globals (`globals_`) that something in the time step writes
    (`step_written`, e.g. dynSubgridControlMod's subgrid_weights_updated).
    The analyzed driver doesn't show timeInfoMod's writes (they come from
    elm_time_manager, which SPEL removes), hence the module list.
    """
    out = [
        TimeScalar("spel_nstep", SHIM_MODULE, "integer", "get_nstep()", "elm_time_manager"),
        TimeScalar(
            "spel_first_restart_step", SHIM_MODULE, "logical", "is_first_restart_step()", "elm_time_manager"
        ),
        TimeScalar("spel_step_size", SHIM_MODULE, "integer", "get_step_size()", "elm_time_manager",
                   fallback="nint(dtime_mod)"),
        TimeScalar("spel_rad_step_size", SHIM_MODULE, "integer", "get_rad_step_size()", "elm_time_manager",
                   fallback="nint(dtime_mod)"),
    ]
    mod = module_dict.get(TIME_MODULE)
    for name, var in sorted(getattr(mod, "global_vars", {}).items()):
        if var.dim == 0 and not var.parameter and var.type in ("real", "integer", "logical"):
            out.append(TimeScalar(name, "timeinfoMod", var.type, name, "timeinfoMod"))
    have = {s.name for s in out}
    written = set(step_written)
    for name, var in sorted((globals_ or {}).items()):
        if (
            name in written
            and name not in have
            and var.dim == 0
            and var.type in ("real", "integer", "logical")
            and var.declaration
        ):
            mod = str(var.declaration)
            out.append(TimeScalar(name, mod, var.type, name, mod))
    return out


def capture_lines(scalars: list[TimeScalar]) -> tuple[str, str]:
    """(use statements, assignments) for the ELM capture module."""
    by_mod: dict[str, set[str]] = {}
    for s in scalars:
        names = by_mod.setdefault(s.capture_module, set())
        names.add(s.capture.removesuffix("()"))
    uses = "".join(f"  use {m}, only : {', '.join(sorted(n))}\n" for m, n in sorted(by_mod.items()))
    sets = "".join(f"    {s.to_array()}\n" for s in scalars)
    return uses, sets


def sync_lines(scalars: list[TimeScalar]) -> tuple[list[str], list[str]]:
    """(use statements, statements) for main.F90 after each read of the inputs."""
    uses = [f"use ReadWriteMod, only : {', '.join(s.array for s in scalars)}\n"]
    by_mod: dict[str, list[str]] = {}
    for s in scalars:
        by_mod.setdefault(s.module, []).append(s.name)
    uses += [f"use {m}, only : {', '.join(n)}\n" for m, n in sorted(by_mod.items())]
    body = [f"   {s.from_array()}\n" for s in scalars]
    # captures older than an array: fall back once the other scalars (dtime_mod) are set
    body += [f"   if ({s.array}(1) <= 0) {s.name} = {s.fallback}\n" for s in scalars if s.fallback]
    if any(s.fallback for s in scalars):
        uses.append("use timeinfoMod, only : dtime_mod\n")
    # if_present: a scalar without an `!$acc declare` (no device kernel reads it, e.g.
    # timeinfoMod::first) has no device copy, and a plain update of it aborts on a GPU
    body.append(f"   !$acc update device({', '.join(s.name for s in scalars)}) if_present\n")
    return uses, body
