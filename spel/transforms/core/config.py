"""
A run configuration for the transforms: the namelist values of a compset or a case, so a
transform touches only the parts of ELM that run in it.

    spel transform acc-apply --config CNPCROP              # a preset of spel/db (presets/*.cfg)
    spel transform acc-sync --config CASE/run/lnd_in       # a case's namelist file
    spel transform --spec steps.toml                       # config = "..." at the top of the spec

A namelist-gated branch is decided from the values (the analysis' `FlatIfs`: the condition of
every if/else-if/else branch whose condition reads namelist variables, with cascades such as
`num_pcropp` <- `use_crop` resolved to their triggers). A variable the configuration does not
set has the default its declaration gives (`use_hydrstress = .false.`); a branch whose
condition stays undecided counts as active. What follows:

- `line_active(sub, ln)` is False on the lines of a branch that cannot run;
- `call_tree(fut, roots)` leaves out the routines reached only through such lines
  (hybrid_phs under `use_hydrstress`, the land-cover change bookkeeping under
  `active_dynamic_subgrid`), and everything reached only through them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from spel.scripts.analyze_subroutines import Subroutine
from spel.scripts.functional_unit_test import FunctionalUnitTest
from spel.scripts.types import FlatIfs

PRESETS = (
    Path(__file__).resolve().parents[2]
    / "db"
    / "app"
    / "management"
    / "commands"
    / "presets"
)


def parse_namelist_values(path: Path) -> dict[str, str]:
    """{variable: Fortran literal} of the `name = value` lines of a .cfg preset or an lnd_in file."""
    data = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = _strip_comment(line).strip()
        if not line or line.startswith(("#", "&", "/")):
            continue
        key, sep, value = line.partition("=")
        if sep and key.strip():
            data[key.strip().lower()] = value.strip().rstrip(",").strip()
    return data


def _strip_comment(line: str) -> str:
    """The line without a `!` comment (a `!` inside a quoted string is kept)."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "!":
            return line[:i]
    return line


def _python_value(text: str) -> Any:
    from spel.scripts.validate_access import _fortran_value

    t = str(text).strip()
    if t in ("True", "False"):  # a declaration default as the analysis records it
        return t == "True"
    return _fortran_value(t, {})


@dataclass
class Configuration:
    """Namelist values (lower-case names -> Python values) and where they came from."""

    values: dict[str, Any]
    source: str = ""
    # names whose value is the declaration's default
    defaults: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, spec: str, fut=None) -> "Configuration":
        """A preset name (spel/db presets/<NAME>.cfg), or the path of a .cfg/lnd_in file. With
        `fut`, the namelist variables it does not set get their declared defaults."""
        p = Path(spec)
        if not p.exists():
            cand = PRESETS / f"{spec}.cfg"
            if not cand.exists():
                known = ", ".join(sorted(x.stem for x in PRESETS.glob("*.cfg")))
                raise FileNotFoundError(
                    f"no configuration {spec!r}: not a file, nor a preset ({known})"
                )
            p = cand
        raw = parse_namelist_values(p)
        cfg = cls({k: _python_value(v) for k, v in raw.items()}, str(p))
        if fut is not None:
            cfg.add_defaults(fut)
        return cfg

    def fresh(self) -> "Configuration":
        """The same values without the per-routine caches (a pipeline step's refreshed analysis)."""
        return Configuration(dict(self.values), self.source, set(self.defaults))

    def add_defaults(self, fut) -> None:
        for name, default in namelist_defaults(fut).items():
            if name not in self.values and default is not None:
                self.values[name] = default
                self.defaults.add(name)

    # --- deciding branches -----------------------------------------------------------------

    def branch(self, flat_if: FlatIfs) -> Optional[bool]:
        """Whether a namelist-gated branch runs: True/False when the values decide it."""
        from spel.scripts.nml.analyze_namelist import cascade_to_triggers
        from spel.scripts.validate_access import evaluate_guard

        cond = flat_if.condtional_expectation
        if cond is None:
            return None
        try:
            return evaluate_guard(cascade_to_triggers(cond), self.values)
        except Exception:
            return None

    def dead_branches(self, sub: Subroutine) -> list[tuple[int, int]]:
        """(first, last) body lines (0-based) of the routine's branches that cannot run."""
        cache = self.__dict__.setdefault("_dead", {})
        if sub.id not in cache:
            out = []
            for f in sub.flat_ifs:
                if not (f.nml_vars or f.nml_cascades) or self.branch(f) is not False:
                    continue
                span = branch_lines(f)
                if span is not None:
                    out.append(span)
            cache[sub.id] = sorted(out)
        return cache[sub.id]

    def line_active(self, sub, ln: int) -> bool:
        """False on a line of a branch that cannot run in this configuration."""
        return not any(a <= ln <= b for a, b in self.dead_branches(sub))

    def span_active(self, sub, lo: int, hi: int) -> bool:
        """False when every line lo..hi lies in branches that cannot run."""
        dead = self.dead_branches(sub)
        return not any(a <= lo and hi <= b for a, b in dead)

    def call_tree(
        self,
        fut: FunctionalUnitTest,
        roots: Iterable[str],
        exclude=None,
    ) -> dict[str, object]:
        """`core.query.call_tree`, without the routines reached only through dead branches."""
        out: dict[str, object] = {}
        stack = list(roots)
        while stack:
            sid = stack.pop()
            if sid in out or sid not in fut.subroutine_dict:
                continue
            if exclude is not None and exclude(sid):
                continue
            sub = fut.subroutine_dict[sid]
            out[sid] = sub
            children = sub.child_subroutines
            lines: dict[str, list[int]] = {}
            for ln, desc in sub.sub_call_desc.items():
                ch = children.get(desc.fn)
                if ch is not None:
                    lines.setdefault(ch.id, []).append(ln)
            for ch in children.values():
                lns = lines.get(ch.id)
                # no recorded call line (a function reference): kept
                if lns is None or any(self.line_active(sub, ln) for ln in lns):
                    stack.append(ch.id)
        return out

    def describe(self) -> str:
        set_ = len(self.values) - len(self.defaults)
        return f"configuration {self.source}: {set_} namelist values set, {len(self.defaults)} from declarations"


def branch_lines(flat_if: FlatIfs) -> Optional[tuple[int, int]]:
    """(first, last) lines (0-based) a branch's condition decides: its body, or for a one-line
    `if (use_fates) call x` the if's own line. None for a branch without body lines."""
    if flat_if.start_ln == flat_if.end_ln:
        return flat_if.start_ln, flat_if.start_ln
    if flat_if.end_ln - flat_if.start_ln > 1:
        return flat_if.start_ln + 1, flat_if.end_ln - 1
    return None


def gates(sub: Subroutine, ln: int) -> list:
    """
    The namelist conditions under which line `ln` of the routine runs:
    `ConditionExpectation` (cascades as their triggers) of every namelist-gated branch whose
    lines include it. `validate_access.evaluate_guard(gate, values)` decides one;
    the line is dead in a configuration when a gate is False (what `Configuration.line_active` says for the
    whole configuration).
    """
    from spel.scripts.nml.analyze_namelist import cascade_to_triggers

    out = []
    for f in sub.flat_ifs:
        if not (f.nml_vars or f.nml_cascades):
            continue
        cond = f.condtional_expectation
        span = branch_lines(f)
        if cond is not None and span is not None and span[0] <= ln <= span[1]:
            out.append(cascade_to_triggers(cond))
    return out


def namelist_names(fut: FunctionalUnitTest) -> set[str]:
    """Every namelist variable the analysis' namelist ifs test, directly or as a cascade's trigger."""
    out: set[str] = set()
    for sub in fut.subroutine_dict.values():
        for f in sub.flat_ifs:
            out.update(k.lower() for k in (f.nml_vars or {}))
            for dep in (f.nml_cascades or {}).values():
                out.update(n.lower() for n in dep.trigger_names)
    return out


def namelist_defaults(fut: FunctionalUnitTest) -> dict[str, Any]:
    """Declared defaults of the namelist variables the analysis' namelist ifs test."""
    out: dict[str, Any] = {}
    for sub in fut.subroutine_dict.values():
        for f in sub.flat_ifs:
            for name, nml in f.nml_vars.items():
                v = nml.variable
                default_val = v.default_value
                if name.lower() not in out and default_val not in (None, ""):
                    out[name.lower()] = _python_value(default_val)
            for dep in f.nml_cascades.values():
                defaults = dep.trigger_defaults
                for k, name in enumerate(dep.trigger_names):
                    default_val = defaults[k] if k < len(defaults) else None
                    if out.get(name.lower()) is None and default_val is not None:
                        out[name.lower()] = _python_value(default_val)
                    else:
                        out.setdefault(name.lower(), None)
    return {k: v for k, v in out.items() if v is not None}
