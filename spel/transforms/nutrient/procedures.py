"""
How much of the C/N/P types' procedures is the same code.

For a family (common.py) and a type-bound procedure the types have in common
(`init`, `setvalues`, ...), every executable statement of each type's routine
is normalized and compared with the other types' statements:

  * a component, written ``this%leafn`` or through an associate name, becomes
    its field's base name (``this%{leaf}``),
  * inside string literals, the history/restart name derived from the
    component (``'LEAFN'``) becomes the base name, and the element's own
    words become placeholders: ``N`` / ``gN`` / ``nitrogen`` -> ``#`` /
    ``g#`` / ``#element``,
  * carbon isotope labels are dropped (``'C13_LEAFC'`` -> ``'{leaf}'``,
    ``gC13`` -> ``g#``, restart ``'leafc_13'`` -> ``'{leaf}'``), and guards on the `carbon_type` dummy are ignored: the
    c12/c13/c14 copies of a registration are one base statement whose names
    the base code builds from the isotope prefix,
  * dummy arguments keep their names (they correspond by name, not by
    position), and a whole-array section ``x(begp:endp)`` of the bounds
    dummies is ``x(:)``,
  * the statement's guards (enclosing ifs) are normalized the same way.

A statement is

  generic   the same normalized text, under the same guards, in every type
            that has all the fields it touches: base-type code
  guards    the same code as the other types', under different conditions
            (carbon's `if (.not. use_fates)`)
  longname  the same except for the `long_name=` text (history metadata only)
  differs   it touches only fields the types share, but some type's version
            differs (history `default=`, ratios to other instances...):
            a parameter of the base code, or stays in the concrete types
  own       it touches a field only its type has, or `this` through something
            the matching doesn't know (`call this%SetValues(...)`): stays in the
            concrete type
  other     it touches no component of the type (loops over filters, locals)

The routine's structure (which statements are plain statements, their
guards, associate names, dummies) comes from SPEL's parse tree (tree.py);
statements are compared as their source text, which keeps string case.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from spel.transforms.nutrient.common import ELEMENTS, Family
from spel.transforms.nutrient.tree import Leaf, Routine, associations, read_routine, split_code

ELEMENT_WORD = {e: w for w, e in ELEMENTS.items()}
_STRING_RE = re.compile(r"('(?:[^']|'')*'|\"(?:[^\"]|\"\")*\")")
# kept for callers outside the nutrient package
_split_code = split_code
Stmt = Leaf


@dataclass
class ProcReport:
    family: str
    binding: str
    routines: dict[str, str]  # element -> routine id
    stmts: dict[str, list[Leaf]] = field(default_factory=dict)
    trees: dict[str, Routine] = field(default_factory=dict)

    def counts(self, element: str) -> Counter:
        return Counter(s.kind for s in self.stmts[element])


class Normalizer:
    def __init__(self, element: str, names: dict[str, str], dummies: list[str] = (), old_names: dict = None):
        self.e = element  # c | n | p
        self.names = names  # component -> base name
        # components moved by extract/groups (manifest): old name -> new path; strings
        # (history and restart names) still spell the old one ('leafn', 'LEAFN')
        self.old_names = old_names or {}
        self.aliases: dict[str, str] = {}
        # dummies correspond by name between the C/N/P routines, never by position
        # (veg_cf_summary's 2nd dummy is num_soilp, veg_nf_summary's num_soilc)
        self.dummies = {d.lower(): d.lower() for d in dummies if d.lower() != "this"}
        lo = [d for d in self.dummies if d.startswith("beg")]
        hi = [d for d in self.dummies if d.startswith("end")]
        self.whole = f"({lo[0]}:{hi[0]})" if lo and hi else None
        word = ELEMENT_WORD[element]
        E = element.upper()
        self._string_subs = []
        if element == "c":
            self._string_subs += [
                (re.compile(r"(?<![A-Za-z0-9])C1[34]_"), ""),
                (re.compile(r"\bgC1[34](?=\b|/)"), "gC"),
                (re.compile(r"\bC1[34] "), ""),
                (re.compile(r"_1[34]$"), ""),
            ]
        self._string_subs += [
            (re.compile(rf"\b{word}\b", re.I), "#element"),
            (re.compile(rf"\bg{E}(?=\b|/)"), "g#"),
            (re.compile(rf"\b{E}\b"), "#"),
        ]

    def string(self, lit: str) -> str:
        body = lit[1:-1]
        for comp in sorted(self.old_names, key=len, reverse=True):
            rx = rf"(?<![A-Za-z0-9{{]){re.escape(comp)}(?![A-Za-z0-9}}])"
            body = re.sub(rx, "{" + self.old_names[comp] + "}", body, flags=re.I)
        for comp in sorted(self.names, key=len, reverse=True):
            # history names embed the component between underscores (PFT_FIRE_NLOSS)
            rx = rf"(?<![A-Za-z0-9{{]){re.escape(comp)}(?![A-Za-z0-9}}])"
            body = re.sub(rx, "{" + self.names[comp] + "}", body, flags=re.I)
        for rx, rep in self._string_subs:
            body = rx.sub(rep, body)
        return "'" + body + "'"

    def code(self, text: str) -> tuple[str, frozenset[str]]:
        touched: set[str] = set()

        def comp(m: re.Match) -> str:
            name = m.group(1).lower()
            if name not in self.names and "%" in name:
                # this%x%y where x isn't a component holding fields: this%x, then %y as is
                head, _, tail = name.partition("%")
                if head in self.names:
                    touched.add(self.names[head])
                    return "this%{" + self.names[head] + "}%" + tail
                name = head
            base = self.names.get(name)
            if base is None:
                touched.add("?" + name)  # a component the family's matching doesn't know
                return m.group(0)
            touched.add(base)
            return "this%{" + base + "}"

        def alias(m: re.Match) -> str:
            name = m.group(0).lower()
            if name in self.dummies:
                return self.dummies[name]
            comp_name = self.aliases.get(name)
            if comp_name is None or comp_name not in self.names:
                return m.group(0)
            touched.add(self.names[comp_name])
            return "{" + self.names[comp_name] + "}"

        out = []
        for i, piece in enumerate(_STRING_RE.split(text)):
            if i % 2:
                out.append(self.string(piece))
                continue
            piece = re.sub(r"\s+", "", piece).lower()
            piece = re.sub(r"this%(\w+(?:%\w+)?)", comp, piece)
            if self.aliases or self.dummies:
                piece = re.sub(r"(?<![%\w])[a-z_]\w*", alias, piece)
            if self.whole:
                piece = piece.replace("}" + self.whole, "}(:)")
            out.append(piece)
        return "".join(out), frozenset(touched)


def moved_names(type_dict: dict, tname: str) -> dict[str, str]:
    """Old component -> new path for `tname`, from the manifest extract/groups wrote, if any."""
    import json

    path = Path(type_dict[tname].filepath).parent / ".spel-nutrient.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for key in ("families", "groups"):
        for entry in data.get(key, {}).values():
            out.update(entry.get("types", {}).get(tname, {}))
    return out


def routine_statements(sub, element: str, names: dict[str, str], old_names=None) -> tuple[Routine, list[Leaf]]:
    """The routine's tree, and its statements with their normalized keys."""
    r = read_routine(sub)
    norm = Normalizer(element, names, r.dummies, old_names)
    for alias, path in associations(r).items():
        if path[0] == "this" and len(path) > 1:
            norm.aliases[alias] = "%".join(path[1:])
    leaves = list(r.leaves())
    for s in leaves:
        g = " && ".join(norm.code(c)[0] for c in s.guards if "carbon_type" not in c)
        s.iso = any("carbon_type" in c for c in s.guards)
        s.body, s.fields = norm.code(s.text)
        s.key = f"[{g}] {s.body}"
        s.plain = f"[{g}] " + re.sub(r",?long_name='(?:[^']|'')*'", "", s.body)
    return r, leaves


def compare_procedure(fam: Family, binding: str, sub_dict: dict, type_dict: dict) -> ProcReport | None:
    routines = {}
    for e, tname in fam.types.items():
        sid = type_dict[tname].procedures.get(binding)
        if isinstance(sid, str) and sid in sub_dict:
            routines[e] = sid
    if len(routines) < 2:
        return None
    # component -> base, per element; which elements have each base field
    names = {e: {} for e in fam.types}
    has: dict[str, set[str]] = {}
    for f in fam.fields:
        for e, v in f.variants.items():
            names[e][v.name] = f.base
            has.setdefault(f.base, set()).add(e)
    # fields held in components (core, mineral, ...: the analysis flattens them to
    # core%leaf): the same path in each type that has it
    for e, tname in fam.types.items():
        for path in type_dict[tname].components:
            if "%" in path:
                names[e][path] = path
                has.setdefault(path, set()).add(e)
    rep = ProcReport(fam.name, binding, routines)
    for e, sid in routines.items():
        old = moved_names(type_dict, fam.types[e])
        rep.trees[e], rep.stmts[e] = routine_statements(sub_dict[sid], e, names[e], old)
    keys = {e: Counter(s.key for s in ss) for e, ss in rep.stmts.items()}
    bodies = {e: Counter(s.body for s in ss) for e, ss in rep.stmts.items()}
    plains = {e: Counter(s.plain for s in ss) for e, ss in rep.stmts.items()}
    for e, ss in rep.stmts.items():
        for s in ss:
            real = {b for b in s.fields if not b.startswith("?")}
            if not s.fields:
                s.kind = "other"
                continue
            if len(real) < len(s.fields):
                s.kind = "own"  # touches a component the matching left out (not in this family's fields)
                continue
            # the elements whose routines could hold this statement
            sharing = set.intersection(*(has[b] for b in real)) & set(routines)
            if len(sharing) < 2:
                s.kind = "own"
            elif all(keys[o][s.key] for o in sharing):
                s.kind = "generic"
            elif all(bodies[o][s.body] for o in sharing):
                s.kind = "guards"
            elif all(plains[o][s.plain] for o in sharing):
                s.kind = "longname"
            else:
                s.kind = "differs"
    return rep


def compare_family(fam: Family, sub_dict: dict, type_dict: dict) -> list[ProcReport]:
    bindings = [b for b, es in fam.procedures.items() if len(es) > 1]
    reps = [compare_procedure(fam, b, sub_dict, type_dict) for b in bindings]
    return [r for r in reps if r is not None]


def summary(reports: list[ProcReport]) -> str:
    out = [
        f"{'family':26s} {'procedure':12s} {'el':2s} {'stmts':>6s} {'generic':>8s} "
        f"{'guards':>7s} {'lname':>6s} {'differs':>8s} {'own':>5s} {'other':>6s}"
    ]
    for r in reports:
        for e in r.stmts:
            c = r.counts(e)
            out.append(
                f"{r.family:26s} {r.binding:12s} {e:2s} {len(r.stmts[e]):6d} {c['generic']:8d} "
                f"{c['guards']:7d} {c['longname']:6d} {c['differs']:8d} {c['own']:5d} {c['other']:6d}"
            )
    out.append(
        "\ngeneric: same code in every type with its fields (base type); guards: same code,\n"
        "different enclosing conditions; lname: differs in long_name only; differs: shared fields, different code; own: a\n"
        "field only this type has; other: no component"
    )
    return "\n".join(out)


def detail(r: ProcReport, kinds=("differs",)) -> str:
    out = [f"== {r.family} {r.binding}: " + ", ".join(f"{e}={s}" for e, s in r.routines.items())]
    for e, ss in r.stmts.items():
        sel = [s for s in ss if s.kind in kinds]
        if sel:
            out.append(f"\n {e} ({len(sel)}):")
            out.extend(f"  {s.ln + 1:6d} {s.kind:8s} {s.text[:150]}" for s in sel)
    return "\n".join(out)
