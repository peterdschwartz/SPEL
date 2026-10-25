"""
Generate `ElmDeviceDataMod.F90` and the driver calls for a full-model OpenACC run.

The module gives ELM three entry points, all no-ops without `_OPENACC`:

- `elm_device_init()`: device selection and `acc_init`, called first in `initialize1`
  (before any module data is touched by ELM; data in `declare` directives was already
  placed on the default device at program start, so the run must give each rank one
  visible device, e.g. `srun --gpus-per-task=1`).
- `elm_device_copyin()`: the one-time deep copy of every type instance of the analysis
  to the device, called in `elm_drv` on the first time step (after initialization and
  restart reads), guarded by `elm_device_resident`. Instances with `!$acc declare
  create` get their components copied under the already present object; the others get
  `enter data copyin` of the object first. Pointer components the analysis records as
  aliases (`=>` to another component) are attached instead of copied. After the copy
  every array is checked with `acc_is_present` and the misses are reported.
- `elm_device_update(to_device)`: `update device`/`update self` of every component of
  every instance (brute force: the whole state), called at the step boundaries and
  around the host-side sections of `elm_drv`.

Everything is generated from `fut.type_dict` (instances, components, nesting, pointer
targets) so it follows the sources it is applied to.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MODULE_NAME = "ElmDeviceDataMod"
TAG = "! spel device-data"
INTRINSIC = {"real", "integer", "logical", "character", "double", "complex", "double precision"}
MAX_DEPTH = 5


class DeviceDataError(RuntimeError):
    pass


@dataclass
class Instance:
    name: str
    module: str
    type_name: str
    dim: int
    allocatable: bool
    pointer: bool
    declared: bool  # has an `!$acc declare` in its module


@dataclass
class Plan:
    instances: list[Instance]
    types: dict[str, object]  # type name -> DerivedType (only the ones reached)
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --- analysis side -------------------------------------------------------------------


def _acc_declared_names(path: Path) -> set[str]:
    """Names in `!$acc declare ...` directives of one source file (continuations joined)."""
    names: set[str] = set()
    cur: Optional[str] = None
    for raw in path.read_text(errors="ignore").splitlines():
        s = raw.strip()
        if s.lower().startswith("!$acc"):
            body = s[5:].strip()
            if cur is None:
                if not body.lower().startswith("declare"):
                    continue
                cur = body
            else:
                cur += " " + body
            if cur.rstrip().endswith("&"):
                cur = cur.rstrip("& ")
                continue
            inner = re.sub(r"\b(create|copyin|copyout|copy|present|device_resident|link)\s*\(", "(", cur, flags=re.I)
            names.update(n.lower() for n in re.findall(r"\b([A-Za-z]\w*)\b", inner) if n.lower() != "declare")
            cur = None
        elif cur is not None:
            cur = None
    return names


_TYPE_DEF_RE = re.compile(r"^\s*type\s*(?:,\s*([^:]*?))?\s*(?:::)?\s*(\w+)\s*$", re.I)


def type_is_public(dtype, fut) -> bool:
    """
    Whether the type can be `use`d from another module: its definition says
    `public`/`private`, else the module's default accessibility, else a `public ::`
    list naming it. (Accessibility is not in the analysis record.)
    """
    mod = fut.module_dict.get(str(dtype.declaration).lower())
    if mod is None:
        return False
    try:
        lines = Path(mod.filepath).read_text(errors="ignore").split("\n")
    except OSError:
        return False
    name = dtype.type_name.lower()
    default_private = False
    listed_public = False
    in_type = False
    for raw in lines:
        s = raw.split("!")[0].strip().lower()
        if not s:
            continue
        m = _TYPE_DEF_RE.match(s)
        if m and not s.startswith("type("):
            in_type = True
        elif re.match(r"^end\s*type\b", s):
            in_type = False
            continue
        if in_type and not m:
            continue  # inside a type definition: its `contains` is not the module's
        if s == "contains":
            break
        if s == "private":
            default_private = True
        if m and m.group(2) == name and not s.startswith("type("):
            attrs = {a.strip() for a in (m.group(1) or "").split(",")}
            if "private" in attrs:
                return False
            if "public" in attrs:
                return True
        if s.startswith("public") and re.search(rf"\b{name}\b", s):
            listed_public = True
    return listed_public or not default_private


def variable_is_public(name: str, mod) -> bool:
    """
    Whether module variable `name` can be `use`d: a `public`/`private` attribute on its
    declaration, else a `public ::`/`private ::` list naming it, else the module default.
    """
    try:
        lines = Path(mod.filepath).read_text(errors="ignore").split("\n")
    except OSError:
        return False
    name = name.lower()
    default_private = False
    listed: Optional[bool] = None
    decl_attr: Optional[bool] = None
    stmt = ""
    in_type = False
    for raw in lines:
        s = raw.split("!")[0].strip().lower()
        if not s:
            continue
        stmt = (stmt + " " + s.lstrip("&")).strip() if stmt else s
        if stmt.endswith("&"):
            stmt = stmt[:-1]
            continue
        full, stmt = stmt, ""
        if _TYPE_DEF_RE.match(full) and not full.startswith("type("):
            in_type = True
        elif re.match(r"^end\s*type\b", full):
            in_type = False
        if in_type:
            continue  # components and type-bound `contains` are not module entities
        if full == "contains":
            break
        if full == "private":
            default_private = True
        elif re.match(r"^(public|private)\s*(::)?\s", full) and re.search(rf"\b{name}\b", full.split("::")[-1]):
            listed = full.startswith("public")
        elif "::" in full and not full.startswith(("type ", "type,")) or (full.startswith("type(") and "::" in full):
            attrs, _, entities = full.partition("::")
            if re.search(rf"(^|[,\s]){name}\s*(\(|=|,|$)", entities.strip()):
                words = {a.strip() for a in attrs.split(",")}
                if "private" in words:
                    decl_attr = False
                elif "public" in words:
                    decl_attr = True
    if decl_attr is not None:
        return decl_attr
    if listed is not None:
        return listed
    return not default_private


def plan(fut) -> Plan:
    """Every module-level instance of an analyzed derived type, with what the copy needs."""
    private = {k for k, t in fut.type_dict.items() if not type_is_public(t, fut)}
    type_dict = {k: t for k, t in fut.type_dict.items() if k not in private}
    declared_cache: dict[str, set[str]] = {}
    instances: list[Instance] = []
    skipped: list[str] = []
    reached: dict[str, object] = {}

    def reach(tname: str, depth: int = 0) -> None:
        t = type_dict.get(tname.lower())
        if t is None or tname.lower() in reached or depth > MAX_DEPTH:
            return
        reached[tname.lower()] = t
        for c in t.components.values():
            if str(c.type).lower() not in INTRINSIC:
                reach(str(c.type), depth + 1)

    for k in sorted(private):
        skipped.append(f"type {k}: private in its module (and its instances)")
    for dtype in type_dict.values():
        for inst in dtype.instances.values():
            mod = (inst.declaration or "").lower()
            if not mod or mod not in fut.module_dict:
                skipped.append(f"{inst.name}: no declaring module ({inst.declaration!r})")
                continue
            if getattr(inst, "private", False):
                skipped.append(f"{mod}::{inst.name}: private")
                continue
            gv = fut.module_dict[mod].global_vars
            if inst.name not in gv and inst.name.lower() not in {g.lower() for g in gv}:
                # the analysis also registers nested-type components (veg_cf%core) as
                # instances of the component's type; only module variables are copied here
                skipped.append(f"{mod}::{inst.name}: not a module variable (nested component)")
                continue
            if getattr(inst, "parameter", False):
                skipped.append(f"{mod}::{inst.name}: parameter (constant, cannot be a data-clause item)")
                continue
            if not variable_is_public(inst.name, fut.module_dict[mod]):
                skipped.append(f"{mod}::{inst.name}: private module variable")
                continue
            if not dtype.components:
                skipped.append(f"{mod}::{inst.name}: type {dtype.type_name} has no components")
                continue
            fpath = Path(fut.module_dict[mod].filepath)
            if mod not in declared_cache:
                declared_cache[mod] = _acc_declared_names(fpath) if fpath.exists() else set()
            instances.append(
                Instance(
                    name=inst.name,
                    module=mod,
                    type_name=dtype.type_name,
                    dim=int(inst.dim or 0),
                    allocatable=bool(getattr(inst, "allocatable", False)),
                    pointer=bool(getattr(inst, "ptrscalar", False)),
                    declared=inst.name.lower() in declared_cache[mod],
                )
            )
            reach(dtype.type_name)
    instances.sort(key=lambda i: (i.module, i.name))
    return Plan(instances=instances, types=reached, skipped=skipped, notes=sorted(private))


# --- code generation -----------------------------------------------------------------


def _assoc(c) -> Optional[str]:
    """The Fortran test that `this%c` has storage, or None when it always has."""
    if getattr(c, "ptrscalar", False):
        return "associated"
    if getattr(c, "allocatable", False):
        return "allocated"
    return None


def _is_alias(c) -> bool:
    return bool(getattr(c, "ptrscalar", False) and getattr(c, "pointer", None))


_PRIVATE_COMPONENTS: dict[str, set[str]] = {}


def private_components(t, fut) -> set[str]:
    """Component names of type `t` that are private (a bare `private` in the definition, or
    a `private` attribute on the declaration); they cannot be named outside the module."""
    key = t.type_name.lower()
    if key in _PRIVATE_COMPONENTS:
        return _PRIVATE_COMPONENTS[key]
    out: set[str] = set()
    mod = fut.module_dict.get(str(t.declaration).lower())
    try:
        lines = Path(mod.filepath).read_text(errors="ignore").split("\n") if mod else []
    except OSError:
        lines = []
    in_type = False
    all_private = False
    stmt = ""
    for raw in lines:
        s = raw.split("!")[0].strip().lower()
        if not s:
            continue
        stmt = (stmt + " " + s.lstrip("&")).strip() if stmt else s
        if stmt.endswith("&"):
            stmt = stmt[:-1]
            continue
        full, stmt = stmt, ""
        m = _TYPE_DEF_RE.match(full)
        if m and not full.startswith("type(") and m.group(2) == key:
            in_type = True
            continue
        if in_type:
            if re.match(r"^end\s*type\b", full):
                break
            if full == "private":
                all_private = True
            elif full == "contains":
                continue
            elif "::" in full:
                attrs, _, entities = full.partition("::")
                words = {a.strip() for a in attrs.split(",")}
                names = [re.match(r"\s*(\w+)", e).group(1) for e in split_entities(entities) if re.match(r"\s*(\w+)", e)]
                if "private" in words:
                    out.update(names)
                elif "public" in words:
                    out.difference_update(names)
                    for n in names:
                        out.discard(n)
                    if all_private:
                        _PUBLIC_OVERRIDE.setdefault(key, set()).update(names)
    if all_private:
        for c in t.components:
            if c.lower() not in _PUBLIC_OVERRIDE.get(key, set()):
                out.add(c.lower())
    _PRIVATE_COMPONENTS[key] = out
    return out


_PUBLIC_OVERRIDE: dict[str, set[str]] = {}


def split_entities(text: str) -> list[str]:
    """Split an entity list on top-level commas (initializers and array specs may hold commas)."""
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def _components(t, fut=None):
    if fut is None:
        return list(t.components.values())
    hidden = private_components(t, fut)
    return [c for c in t.components.values() if c.name.lower() not in hidden]


def _copyin_routine(t, type_dict, fut) -> list[str]:
    """`dd_copyin_<type>(this)`: enter data for the arrays, recursion into nested types."""
    name = t.type_name
    out = [
        f"  subroutine dd_copyin_{name}(this)",
        f"    type({name}), intent(inout), target :: this",
        "    integer :: i_",
        "#ifdef _OPENACC",
    ]
    body: list[str] = []
    for c in _components(t, fut):
        ctype = str(c.type).lower()
        path = f"this%{c.name}"
        if ctype in INTRINSIC:
            if c.dim == 0 or ctype.startswith("character"):
                continue  # scalars travel inside the object; strings stay on the host
            if _is_alias(c):
                continue  # attached in dd_attach_<type>
            test = _assoc(c)
            if test:
                body += [f"    if ({test}({path})) then", f"       !$acc enter data copyin({path})", "    end if"]
            else:
                body += [f"    !$acc enter data copyin({path})"]
        elif ctype in type_dict:
            if c.dim == 0:
                body += [f"    call dd_copyin_{ctype}({path})"]
            else:
                test = _assoc(c)
                open_ = [f"    if ({test}({path})) then"] if test else []
                close = ["    end if"] if test else []
                body += open_ + [
                    f"    {'   ' if test else ''}!$acc enter data copyin({path})",
                    f"    {'   ' if test else ''}do i_ = lbound({path}, 1), ubound({path}, 1)",
                    f"    {'   ' if test else ''}   call dd_copyin_{ctype}({path}(i_))",
                    f"    {'   ' if test else ''}end do",
                ] + close
    if not body:
        body = ["    continue"]
    out += body + ["#endif", f"  end subroutine dd_copyin_{name}", ""]
    return out


def _attach_routine(t, type_dict, fut) -> list[str]:
    """`dd_attach_<type>(this)`: attach the alias pointer components once their targets are present."""
    name = t.type_name
    out = [
        f"  subroutine dd_attach_{name}(this)",
        f"    type({name}), intent(inout), target :: this",
        "    integer :: i_",
        "#ifdef _OPENACC",
    ]
    body: list[str] = []
    for c in _components(t, fut):
        ctype = str(c.type).lower()
        path = f"this%{c.name}"
        if ctype in INTRINSIC:
            if _is_alias(c) and c.dim > 0:
                body += [f"    if (associated({path})) then", f"       !$acc enter data attach({path})", "    end if"]
        elif ctype in type_dict:
            if c.dim == 0:
                body += [f"    call dd_attach_{ctype}({path})"]
            else:
                test = _assoc(c)
                open_ = [f"    if ({test}({path})) then"] if test else []
                close = ["    end if"] if test else []
                body += open_ + [
                    f"    {'   ' if test else ''}do i_ = lbound({path}, 1), ubound({path}, 1)",
                    f"    {'   ' if test else ''}   call dd_attach_{ctype}({path}(i_))",
                    f"    {'   ' if test else ''}end do",
                ] + close
    if not body:
        body = ["    continue"]
    out += body + ["#endif", f"  end subroutine dd_attach_{name}", ""]
    return out


def _update_routine(t, type_dict, fut) -> list[str]:
    """`dd_update_<type>(this, to_device)`: update device/self of every component."""
    name = t.type_name
    out = [
        f"  subroutine dd_update_{name}(this, to_device)",
        f"    type({name}), intent(inout), target :: this",
        "    logical, intent(in) :: to_device",
        "    integer :: i_",
        "#ifdef _OPENACC",
    ]
    body: list[str] = []

    def upd(path: str, indent: str) -> list[str]:
        return [
            f"{indent}if (to_device) then",
            f"{indent}   !$acc update device({path})",
            f"{indent}else",
            f"{indent}   !$acc update self({path})",
            f"{indent}end if",
        ]

    for c in _components(t, fut):
        ctype = str(c.type).lower()
        path = f"this%{c.name}"
        if ctype in INTRINSIC:
            if ctype.startswith("character") or _is_alias(c):
                continue
            test = _assoc(c) if c.dim > 0 else None
            if test:
                body += [f"    if ({test}({path})) then"] + upd(path, "       ") + ["    end if"]
            else:
                body += upd(path, "    ")
        elif ctype in type_dict:
            if c.dim == 0:
                body += [f"    call dd_update_{ctype}({path}, to_device)"]
            else:
                test = _assoc(c)
                open_ = [f"    if ({test}({path})) then"] if test else []
                close = ["    end if"] if test else []
                body += open_ + [
                    f"    {'   ' if test else ''}do i_ = lbound({path}, 1), ubound({path}, 1)",
                    f"    {'   ' if test else ''}   call dd_update_{ctype}({path}(i_), to_device)",
                    f"    {'   ' if test else ''}end do",
                ] + close
    if not body:
        body = ["    continue"]
    out += body + ["#endif", f"  end subroutine dd_update_{name}", ""]
    return out


def _check_routine(t, type_dict, fut) -> list[str]:
    """`dd_check_<type>(this, label, nmiss)`: count array components not present on the device."""
    name = t.type_name
    out = [
        f"  subroutine dd_check_{name}(this, label, nmiss)",
        f"    type({name}), intent(in), target :: this",
        "    character(len=*), intent(in) :: label",
        "    integer, intent(inout) :: nmiss",
        "    integer :: i_",
        "#ifdef _OPENACC",
    ]
    body: list[str] = []
    for c in _components(t, fut):
        ctype = str(c.type).lower()
        path = f"this%{c.name}"
        if ctype in INTRINSIC:
            if c.dim == 0 or ctype.startswith("character"):
                continue
            test = _assoc(c)
            cond = f"{test}({path})" if test else ".true."
            body += [
                f"    if ({cond}) then",
                f"       if (.not. acc_is_present({path})) then",
                f"          nmiss = nmiss + 1",
                f"          if (nmiss <= 50) write(iulog,*) 'elm_device: not present: ', trim(label)//'%{c.name}'",
                "       end if",
                "    end if",
            ]
        elif ctype in type_dict:
            if c.dim == 0:
                body += [f"    call dd_check_{ctype}({path}, trim(label)//'%{c.name}', nmiss)"]
            else:
                test = _assoc(c)
                open_ = [f"    if ({test}({path})) then"] if test else []
                close = ["    end if"] if test else []
                body += open_ + [
                    f"    {'   ' if test else ''}do i_ = lbound({path}, 1), ubound({path}, 1)",
                    f"    {'   ' if test else ''}   call dd_check_{ctype}({path}(i_), trim(label)//'%{c.name}(i)', nmiss)",
                    f"    {'   ' if test else ''}end do",
                ] + close
    if not body:
        body = ["    continue"]
    out += body + ["#endif", f"  end subroutine dd_check_{name}", ""]
    return out


def _instance_call(inst: Instance, what: str, extra: str = "") -> list[str]:
    """Statements applying `dd_<what>_<type>` to one instance (array instances element-wise)."""
    lines: list[str] = []
    if inst.dim == 0:
        return [f"    call dd_{what}_{inst.type_name}({inst.name}{extra})"]
    test = "associated" if inst.pointer else ("allocated" if inst.allocatable else None)
    ind = "       " if test else "    "
    if test:
        lines.append(f"    if ({test}({inst.name})) then")
    lines += [
        f"{ind}do i_ = lbound({inst.name}, 1), ubound({inst.name}, 1)",
        f"{ind}   call dd_{what}_{inst.type_name}({inst.name}(i_){extra})",
        f"{ind}end do",
    ]
    if test:
        lines.append("    end if")
    return lines


def generate(fut, srcroot: Path) -> tuple[str, Plan]:
    """The module text and the plan it was generated from."""
    p = plan(fut)
    # components of private types are opaque to the generated code (no `use` possible)
    type_dict = {k.lower(): v for k, v in fut.type_dict.items() if k.lower() not in set(p.notes)}
    try:
        commit = subprocess.run(
            ["git", "-C", str(srcroot), "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        commit = "?"
    by_module: dict[str, list[str]] = {}
    for inst in p.instances:
        by_module.setdefault(inst.module, []).append(inst.name)
    type_mods: dict[str, set[str]] = {}
    for t in p.types.values():
        type_mods.setdefault(str(t.declaration).lower(), set()).add(t.type_name)

    L: list[str] = []
    L += [
        f"module {MODULE_NAME}",
        "  !",
        "  ! GENERATED by `spel device-data` from the elm_drv analysis of the sources",
        f"  ! ({srcroot}, {commit}). Do not edit; regenerate.",
        "  !",
        "  ! Device data layer for full-model OpenACC runs: device initialization, the",
        "  ! one-time deep copy of every type instance to the device (first time step),",
        "  ! and whole-state updates around the host-side sections of the driver.",
        "  ! Everything is a no-op without _OPENACC.",
        "  !",
        "#ifdef _OPENACC",
        "  use openacc",
        "#endif",
        "  use elm_varctl, only : iulog",
        "  use spmdMod,    only : masterproc, iam",
    ]
    for mod in sorted(type_mods):
        names = ", ".join(sorted(type_mods[mod]))
        L.append(f"  use {mod}, only : {names}")
    for mod in sorted(by_module):
        L.append(f"  use {mod}, only : {', '.join(by_module[mod])}")
    L += [
        "  implicit none",
        "  private",
        "  public :: elm_device_init, elm_device_copyin, elm_device_update, elm_device_resident",
        "  logical, public, save :: elm_device_resident = .false.  ! the state lives on the device",
        "contains",
        "",
        "  subroutine elm_device_init()",
        "    ! Select the device and initialize the runtime; called first in initialize1.",
        "    ! Data in `declare` directives was placed on the default device when the",
        "    ! program started, so each rank must see exactly one device (srun",
        "    ! --gpus-per-task=1); with several visible devices the local rank picks one",
        "    ! and the declared module data of the other device is lost: warn loudly.",
        "#ifdef _OPENACC",
        "    integer :: ndev, dev, localid, ios",
        "    character(len=32) :: s",
        "    ndev = acc_get_num_devices(acc_device_nvidia)",
        "    if (ndev > 1) then",
        "       call get_environment_variable('SLURM_LOCALID', s, status=ios)",
        "       localid = 0",
        "       if (ios == 0) read(s, *, iostat=ios) localid",
        "       dev = mod(max(localid, 0), ndev)",
        "       call acc_set_device_num(dev, acc_device_nvidia)",
        "       write(iulog,*) 'elm_device_init: WARNING rank', iam, 'sees', ndev, 'devices; picked', dev, &",
        "            ' - module data in acc declare directives may be on another device; bind one GPU per rank'",
        "    end if",
        "    call acc_init(acc_device_nvidia)",
        "    if (masterproc .or. ndev > 1) write(iulog,*) 'elm_device_init: rank', iam, 'device', &",
        "         acc_get_device_num(acc_device_nvidia), 'of', ndev",
        "#endif",
        "  end subroutine elm_device_init",
        "",
        "  subroutine elm_device_copyin()",
        "    ! One-time deep copy of every type instance (first time step).",
        "#ifdef _OPENACC",
        "    integer :: i_, nmiss",
        "    if (elm_device_resident) return",
    ]
    for inst in p.instances:
        L.append(f"    ! --- {inst.module}::{inst.name} ({inst.type_name}{', declare' if inst.declared else ''})")
        if not inst.declared:
            L.append(f"    !$acc enter data copyin({inst.name})")
        L += _instance_call(inst, "copyin")
    L += ["    ! pointer components that alias other components: attach once everything is present"]
    for inst in p.instances:
        L += _instance_call(inst, "attach")
    L += ["    nmiss = 0"]
    for inst in p.instances:
        L += _instance_call(inst, "check", f", '{inst.name}', nmiss")
    L += [
        "    if (nmiss > 0) then",
        "       write(iulog,*) 'elm_device_copyin: rank', iam, ':', nmiss, 'arrays NOT present on the device'",
        "    else if (masterproc) then",
        "       write(iulog,*) 'elm_device_copyin: every array of every instance is present on the device'",
        "    end if",
        "    elm_device_resident = .true.",
        "#endif",
        "  end subroutine elm_device_copyin",
        "",
        "  subroutine elm_device_update(to_device)",
        "    ! update device (host -> device) or update self (device -> host) of the whole state.",
        "    logical, intent(in) :: to_device",
        "#ifdef _OPENACC",
        "    integer :: i_",
        "    if (.not. elm_device_resident) return",
    ]
    for inst in p.instances:
        L += _instance_call(inst, "update", ", to_device")
    L += ["#endif", "  end subroutine elm_device_update", ""]
    for tname in sorted(p.types):
        t = p.types[tname]
        L += _copyin_routine(t, type_dict, fut)
        L += _attach_routine(t, type_dict, fut)
        L += _update_routine(t, type_dict, fut)
        L += _check_routine(t, type_dict, fut)
    L.append(f"end module {MODULE_NAME}")
    return "\n".join(L) + "\n", p


# --- applying to the sources -----------------------------------------------------------


@dataclass
class Edit:
    path: Path
    anchor: Optional[str]  # regex of the line to insert before/after (must match exactly once, or first=True)
    lines: list[str]
    after: bool = False
    first: bool = False  # several matches allowed: use the first
    at_line: Optional[int] = None  # insert before this 0-based line instead of matching an anchor


COMPUTE_RE = re.compile(r"^\s*!\$acc\s+(parallel|kernels|serial)\b", re.I)
# routines the analysis does not see (their modules are excluded from it) but which run kernels
DEVICE_EXTRA = {"hist_update_hbuf"}
# calls that touch no model state: they never start a host or a device section
NEUTRAL = {
    "t_startf", "t_stopf", "get_clump_bounds", "get_proc_bounds", "get_proc_clumps", "shr_sys_flush",
    "endrun", "get_curr_date", "get_nstep", "is_end_curr_day", "is_end_curr_month", "get_step_size",
    "elm_device_update", "elm_device_copyin", "mpi_barrier", "shr_sys_abort",
}


def device_routines(fut) -> set[str]:
    """
    Basenames of the analyzed routines whose call tree contains an OpenACC compute
    construct (`parallel`, `kernels`, `serial`): calling one runs device kernels.
    The directives are comments to the parser, so they are read from the source lines.
    """
    subs = fut.subroutine_dict
    has_region: set[str] = set()
    callees: dict[str, set[str]] = {}
    text_cache: dict[str, list[str]] = {}
    for sid, sub in subs.items():
        base = sid.split("::")[-1].lower()
        try:
            if sub.filepath not in text_cache:
                text_cache[sub.filepath] = Path(sub.filepath).read_text(errors="ignore").split("\n")
            lines = text_cache[sub.filepath][sub.startline : sub.endline + 1]
        except Exception:
            lines = []
        if any(COMPUTE_RE.match(l) for l in lines):
            has_region.add(base)
        rec = getattr(sub, "record", None)
        if rec is not None:
            callees.setdefault(base, set()).update(str(c.name).split("%")[-1].lower() for c in rec.calls)
    device = set(has_region) | DEVICE_EXTRA
    changed = True
    while changed:
        changed = False
        for base, cs in callees.items():
            if base not in device and cs & device:
                device.add(base)
                changed = True
    return device


def _transition_edits(fut, drv: Path) -> tuple[list[Edit], list[str]]:
    """
    Whole-state syncs at every host/device transition of elm_drv's call sequence:
    `update self` before the first host routine after device routines, `update device`
    before the first device routine after host routines. At the start of the step the
    state counts as host-modified (coupler import, IO). Brute force; the refinement is
    per-routine variable lists from the access analysis.
    """
    sub = fut.subroutine_dict.get("elm_driver::elm_drv")
    if sub is None:
        raise DeviceDataError("elm_driver::elm_drv not in the analysis")
    device = device_routines(fut)
    lines = drv.read_text().split("\n")
    edits: list[Edit] = []
    report: list[str] = []
    state = "host"
    for call in sorted(sub.record.calls, key=lambda c: c.ln):
        base = str(call.name).split("%")[-1].lower()
        if base in NEUTRAL:
            continue
        kind = "device" if base in device else "host"
        if kind == state:
            continue
        ln = call.ln
        if ln >= len(lines) or not re.match(r"^\s*(if\s*\(.*\)\s*)?call\b", lines[ln], re.I):
            report.append(f"  skipped transition at {ln + 1} ({call.name}): statement does not start there")
            continue
        to_device = kind == "device"
        edits.append(
            Edit(
                drv,
                None,
                [
                    "#ifdef _OPENACC",
                    f"    call elm_device_update(to_device={'.true.' if to_device else '.false.'})  {TAG}: {'host -> device' if to_device else 'device -> host'} before {call.name}",
                    "#endif",
                ],
                at_line=ln,
            )
        )
        report.append(f"  {ln + 1}: {'update device' if to_device else 'update self  '} before {call.name}")
        state = kind
    return edits, report


def _driver_edits(fut, src: Path) -> tuple[list[Edit], list[str]]:
    drv = src / "main" / "elm_driver.F90"
    ini = src / "main" / "elm_initializeMod.F90"
    use_line = f"  use {MODULE_NAME}, only : elm_device_init, elm_device_copyin, elm_device_update, elm_device_resident  {TAG}"
    edits = [
        Edit(ini, r"^\s*use shr_kind_mod\s*,", [f"  use {MODULE_NAME}, only : elm_device_init  {TAG}"], first=True),
        Edit(ini, r"^\s*call t_startf\('elm_init1'\)", ["#ifdef _OPENACC", f"    call elm_device_init()  {TAG}", "#endif"], after=True),
        Edit(drv, r"^\s*use shr_kind_mod\s*,", [use_line], first=True),
        Edit(
            drv,
            r"^\s*call get_proc_bounds\(bounds_proc\)",
            [
                "#ifdef _OPENACC",
                f"    {TAG}: first time step: the whole state to the device (after initialization and restart reads)",
                "    if (.not. elm_device_resident) then",
                "       call t_startf('device_copyin')",
                "       call elm_device_copyin()",
                "       call t_stopf('device_copyin')",
                "    end if",
                "#endif",
            ],
            after=True,
            first=True,
        ),
    ]
    trans, report = _transition_edits(fut, drv)
    return edits + trans, report


def apply(fut, srcroot: Path, out_dir: Optional[Path] = None, dry_run: bool = False) -> tuple[Plan, list[str]]:
    """
    Write `main/ElmDeviceDataMod.F90` and insert the calls into elm_initializeMod and
    elm_driver. With `out_dir`, the edited files and the module go there (a case's
    SourceMods/src.elm); otherwise the sources are edited in place. Refuses sources
    that already carry the module or the tag.
    """
    src = srcroot / "components" / "elm" / "src"
    text, p = generate(fut, srcroot)
    edits, report = _driver_edits(fut, src)
    files: dict[Path, list[str]] = {}
    for e in edits:
        if e.path not in files:
            if not e.path.exists():
                raise DeviceDataError(f"{e.path}: not found")
            content = e.path.read_text()
            if TAG in content or f"use {MODULE_NAME}" in content:
                raise DeviceDataError(f"{e.path}: already carries the device data calls")
            files[e.path] = content.split("\n")
    # resolve every insertion point on the original text, then insert bottom-up
    placed: list[tuple[Path, int, list[str], str]] = []
    for e in edits:
        lines = files[e.path]
        if e.at_line is not None:
            i = e.at_line
            where = f"line {i + 1}"
        else:
            hits = [k for k, l in enumerate(lines) if re.match(e.anchor, l, re.I)]
            if not hits or (len(hits) != 1 and not e.first):
                raise DeviceDataError(f"{e.path.name}: anchor {e.anchor!r} matched {len(hits)} lines, need 1")
            i = hits[0] + (1 if e.after else 0)
            where = f"{'after' if e.after else 'before'} {e.anchor} ({hits[0] + 1})"
        placed.append((e.path, i, e.lines, where))
    for path, i, new, where in sorted(placed, key=lambda t: -t[1]):
        files[path][i:i] = new
        report.append(f"{path.name}: {where}")
    dest_mod = (out_dir or src / "main") / f"{MODULE_NAME}.F90"
    if dest_mod.exists() and out_dir is None and TAG not in dest_mod.read_text():
        raise DeviceDataError(f"{dest_mod} exists and was not generated here")
    if not dry_run:
        dest_mod.parent.mkdir(parents=True, exist_ok=True)
        dest_mod.write_text(text)
        for path, lines in files.items():
            dest = (out_dir / path.name) if out_dir else path
            dest.write_text("\n".join(lines))
    report.insert(0, f"{dest_mod}: {len(p.instances)} instances of {len(p.types)} types; {len(p.skipped)} instances skipped")
    return p, report
