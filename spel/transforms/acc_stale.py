"""
Names in OpenACC data clauses that no longer exist (`spel transform acc-stale`).

Upstream ELM's directives have rotted: they name variables that were removed or renamed long ago
(elm_varctl `declare copyin(pf_clmnstep0)`, PhenologyInit `update device(crit_dayl, ...)` of what
are now PhenolParamsInst components, AllocationMod `present(col_plant_ndemand(:,:))` of what became
`col_plant_ndemand_vr`, `enter data create(sum, ...)`). A CPU build never sees them (directives are
comments there); every `-acc` build fails on them. This pass removes such names, judged by what the
routine can see, without the analysis (so initialization routines count too):

- the module's variables and its module-level `use` imports;
- the routine's dummies, declarations and `use` imports;
- an `only:` list is taken as written; a whole-module `use` imports that module's variables (from an
  index of the ELM and share/ modules); a whole-module `use` of a module not in the index makes the
  routine opaque and its directives are left alone.

Clauses judged: present, copy, copyin, copyout, create, delete, no_create, deviceptr, attach, detach,
link, device_resident, private, firstprivate, reduction, and the update lists (device, self, host).
A data directive left without names (update, enter/exit data, declare) is deleted; a compute
construct only loses the names. Directives with array sections in a judged clause are rebuilt with
their sections as written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spel.transforms.acc_directives import DirectiveError, find_directives, split_top_level
from spel.transforms.core.edits import delete, replace
from spel.transforms.core.fortran import directive_clauses
from spel.transforms.core.registry import transform
from spel.transforms.core.transform import Change, Plan, Refusal, Transform

JUDGED = {"present", "copy", "copyin", "copyout", "create", "delete", "no_create", "deviceptr", "attach",
          "detach", "link", "device_resident", "private", "firstprivate", "reduction", "device", "self", "host"}
DATA_ONLY = {"update", "enter data", "exit data", "declare"}
MODULE_RE = re.compile(r"^\s*module\s+(\w+)\s*(?:!.*)?$", re.I)


def module_index(roots: list[Path]) -> dict[str, Path]:
    """module name -> file, from `module NAME` lines (a text index used only to find files)."""
    out: dict[str, Path] = {}
    for root in roots:
        for f in sorted(root.rglob("*.F90")):
            if "external_models" in f.parts:
                continue
            for line in f.read_text(errors="replace").split("\n"):
                m = MODULE_RE.match(line)
                if m and m.group(1).lower() != "procedure":
                    out.setdefault(m.group(1).lower(), f)
    return out


def _base(item: str) -> str:
    m = re.match(r"\s*([A-Za-z_]\w*)", item)
    return m.group(1).lower() if m else ""


class Scopes:
    def __init__(self, index: dict[str, Path]):
        self.index = index
        self._vars: dict[str, Optional[set[str]]] = {}

    def module_vars(self, name: str) -> Optional[set[str]]:
        """Variables a whole-module `use` imports (None: the module is unknown)."""
        from spel.scripts.fortran_parser.module_spec import ModuleSpecError, module_spec

        name = name.lower()
        if name not in self._vars:
            path = self.index.get(name)
            if path is None:
                self._vars[name] = None
            else:
                try:
                    spec = module_spec(path)
                    self._vars[name] = set(spec.variables)
                except ModuleSpecError:
                    self._vars[name] = None
        return self._vars[name]

    def imports(self, statements) -> tuple[set[str], bool]:
        """(names imported by the use statements among `statements`, opaque)."""
        from spel.scripts.fortran_parser import spel_ast as A

        names, opaque = set(), False
        for st in statements:
            if not isinstance(st, A.UseStatement):
                continue
            if st.has_only:
                for o in st.objs:
                    text = str(o)
                    names.add(text.split("=>")[0].strip().lower() if "=>" in text else _base(text))
            else:
                mv = self.module_vars(st.module)
                if mv is None:
                    opaque = True
                else:
                    names |= mv
                for r in st.renames or []:
                    names.add(str(r).split("=>")[0].strip().lower())
        return names, opaque


def _declared(statements) -> set[str]:
    """Declared names, plus the associate names of every associate construct inside (kernels name
    them in present/private clauses: `present(x)` with `associate(x => this%x)`)."""
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.scripts.registrations import iter_statements

    out = set()
    for st in statements:
        if isinstance(st, A.VariableDecl):
            out |= {e.token.literal.lower() for e in st.entities}
    block = A.BlockStatement.__new__(A.BlockStatement)
    block.statements = list(statements)
    for st, _ in iter_statements(block):
        if isinstance(st, A.AssociateConstruct):
            out |= {k.lower() for k in st.associations}
    return out


def stale_edits(lines: list[str], lo: int, hi: int, visible: set[str], where: str):
    """(edits, stale names) for the directives on lines lo..hi."""
    edits, gone = [], []
    for d in find_directives(lines, lo, hi):
        new_clauses, stale_here, changed = [], [], False
        for c in d.clauses:
            if c.name not in JUDGED or not c.args:
                new_clauses.append(str(c))
                continue
            head, items = "", split_top_level(c.args)
            if c.name == "reduction" and ":" in c.args:
                head, rest = c.args.split(":", 1)
                items, head = split_top_level(rest), head + ":"
            keep = [i for i in items if _base(i) in visible or not _base(i)]
            drop = [i for i in items if i not in keep]
            if drop:
                changed = True
                stale_here += [_base(i) for i in drop]
            if keep:
                new_clauses.append(f"{c.name}({head}{', '.join(k.strip() for k in keep)})")
        if not changed:
            continue
        gone += stale_here
        n = d.end_ln - d.start_ln + 1
        moves = [c for c in new_clauses if c.split("(")[0] in JUDGED]
        head = d.construct + (f"({d.construct_args})" if d.construct_args is not None else "")
        if d.construct in DATA_ONLY and not moves:
            edits.append(delete(d.start_ln, n, why=f"acc-stale {where}"))
        else:
            edits.append(replace(d.start_ln, directive_clauses(d.indent, head, new_clauses, 110, sentinel="!$acc"),
                                 n=n, why=f"acc-stale {where}"))
    return edits, gone


@dataclass
class Options:
    files: list[str] = field(default_factory=list)  # default: every ELM source file


@transform
class AccStale(Transform):
    """
    Matches: names in OpenACC data clauses (present/copy*/create/delete/update/private/reduction/...)
             that no variable visible in the routine (or module) has.
    Changes: removes them; a data directive left empty (update, enter/exit data, declare) is deleted.
    Refuses: files that do not parse; routines with a whole-module use of a module not in the
             index (their visible names are unknown) are left alone.
    """

    name = "acc-stale"
    summary = "remove names of variables that no longer exist from OpenACC data clauses"
    Options = Options

    def build(self, ctx, opts) -> Plan:
        from spel.scripts.fortran_parser.module_spec import ModuleSpecError, parse_modules

        out = Plan(self.name)
        elm = ctx.srcroot / "components" / "elm" / "src"
        roots = [r for r in (elm, ctx.srcroot / "share") if r.is_dir()]
        if opts.files:
            files = [Path(f) if Path(f).is_absolute() else ctx.srcroot / f for f in opts.files]
        elif elm.is_dir():
            files = [f for f in sorted(elm.rglob("*.F90")) if "external_models" not in f.parts]
        else:
            files = sorted({ctx.path(s) for s in ctx.fut.subroutine_dict.values()})
            roots = sorted({f.parent for f in files})
        scopes = Scopes(module_index(roots))
        for path in files:
            try:
                mods = parse_modules(path)
            except ModuleSpecError as err:
                out.add(Refusal(path.name, str(err)))
                continue
            lines = ctx.lines(path)
            edits, gone = [], []
            for m in mods:
                spec_stmts = m.spec.statements if hasattr(m.spec, "statements") else []
                mod_names, mod_opaque = scopes.imports(spec_stmts)
                mod_names |= _declared(spec_stmts) | {p.name.lower() for p in m.subprograms}
                first_sub = min((p.lineno for p in m.subprograms), default=len(lines))
                if not mod_opaque:
                    e, g = stale_edits(lines, m.lineno, first_sub - 1, mod_names, m.name)
                    edits += e
                    gone += g
                for p in m.subprograms:
                    sub_names, opaque = scopes.imports(p.body.statements)
                    if opaque or mod_opaque:
                        continue
                    visible = mod_names | sub_names | _declared(p.body.statements) | {a.lower() for a in p.args}
                    for c in getattr(p, "contains", None) or []:
                        visible |= {c.name.lower()}
                        visible |= _declared(c.body.statements) | {a.lower() for a in c.args}
                    e, g = stale_edits(lines, p.lineno, p.end_ln if p.end_ln > 0 else len(lines) - 1, visible, p.name)
                    edits += e
                    gone += g
            if edits:
                out.add(Change(path, edits, where=path.name, why="stale names: " + ", ".join(sorted(set(gone)))))
        return out
