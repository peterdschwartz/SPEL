"""
Give each nutrient base type its element: ``core%species``.

After `spel transform nutrient-extract`, base-type routines (merge.py) need to know
which element (and carbon isotope) the core they get belongs to, e.g. to
keep carbon's ``if (.not. use_fates)`` behaviour as
``.not. (is_carbon(this%species) .and. use_fates)``. This:

  * declares ``integer :: species`` first in each base type,
  * sets ``this%core%species`` as the first statement of each concrete
    type's Init: ``species_from_string(carbon_type)`` for the carbon types
    (their Init takes `carbon_type`), CN_SPECIES_N / CN_SPECIES_P otherwise,
  * removes the carbon types' own ``species`` (an allocated integer
    pointer, c12/c13/c14): its declaration, ``allocate(this%species)`` and
    ``this%species = ...`` statements, and renames its other uses
    (``veg_cs%species`` -> ``veg_cs%core%species``),
  * adds ``is_carbon(species)`` (pure, `!$acc routine seq`) to SpeciesMod,
  * extends each data-type module's ``use SpeciesMod, only :`` list.

The value of species is the same as before wherever the old component was
set, so results don't change.
"""

from __future__ import annotations

import re
from pathlib import Path

from spel.transforms.nutrient.common import Family
from spel.transforms.nutrient.extract import CORE, EXCLUDE, ExtractError, rename_in_lines
from spel.scripts.fortran_parser.spel_ast import AllocateStatement
from spel.transforms.nutrient.tree import Leaf, assigned_path, designator_path, read_routine, split_code as _split_code

SPECIES = "species"
NAMES = {"n": "CN_SPECIES_N", "p": "CN_SPECIES_P"}
IS_CARBON = """\
  pure logical function is_carbon(species)
    !$acc routine seq
    ! Whether species is one of the carbon isotopes
    integer, intent(in) :: species
    is_carbon = species == CN_SPECIES_C12 .or. species == CN_SPECIES_C13 .or. species == CN_SPECIES_C14
  end function is_carbon
"""


class SpeciesEditor:
    def __init__(self, families: list[Family], fut, elm_src: Path):
        self.fut = fut
        self.elm_src = Path(elm_src)
        self.families = [f for f in families if f.name in fut.type_dict]
        self.files: dict[Path, list[str]] = {}
        self.notes: list[str] = []

    def lines(self, path: Path) -> list[str]:
        path = Path(path)
        if path not in self.files:
            self.files[path] = path.read_text(errors="replace").split("\n")
        return self.files[path]

    def build(self) -> None:
        td, sd = self.fut.type_dict, self.fut.subroutine_dict
        # edits by original line per file: inserts before, deletions
        inserts: dict[Path, dict[int, list[str]]] = {}
        drops: dict[Path, set[int]] = {}
        old_species: dict[str, dict[str, str]] = {}  # type -> {species: species} for the rename
        modules: set[Path] = set()
        for fam in self.families:
            base = td[fam.name]
            if SPECIES in base.components:
                raise ExtractError(f"{fam.name} already has {SPECIES}")
            path = Path(base.filepath)
            lines = self.lines(path)
            modules.add(path)
            start = _type_line(lines, fam.name)
            indent = re.match(r"\s*", lines[start + 1]).group(0) or "    "
            inserts.setdefault(path, {}).setdefault(start + 1, []).append(
                f"{indent}integer :: {SPECIES}  ! CN_SPECIES_* (SpeciesMod) of the type holding this core"
            )
            for e, tname in fam.types.items():
                dtype = td[tname]
                if CORE not in dtype.components:
                    raise ExtractError(f"{tname}: run `spel transform nutrient-extract` first")
                init_id = dtype.procedures.get("init")
                if init_id not in sd:
                    raise ExtractError(f"{tname}: no analyzed Init")
                init = sd[init_id]
                try:
                    r = read_routine(init)
                except Exception as err:
                    raise ExtractError(f"{init_id}: {err}; run `spel analyze` if the sources changed")
                value = NAMES.get(e)
                if e == "c":
                    if "carbon_type" not in [d.lower() for d in r.dummies]:
                        raise ExtractError(f"{init_id} has no carbon_type dummy")
                    value = "species_from_string(carbon_type)"
                at, ind = first_executable_line(lines, r)
                inserts[path].setdefault(at, []).append(f"{ind}this%{CORE}%{SPECIES} = {value}")
                # the carbon types' own species
                own = dtype.components.get(SPECIES)
                if own is not None:
                    code = _split_code(lines[own.ln])[0]
                    if not re.search(rf"::\s*{SPECIES}\b", code, re.I):
                        raise ExtractError(f"{tname}%{SPECIES}: line {own.ln + 1} is not its declaration")
                    drops.setdefault(path, set()).add(own.ln)
                    for sid in dtype.procedures.values():
                        sub = sd.get(sid) if isinstance(sid, str) else None
                        if sub is None:
                            continue
                        for leaf in read_routine(sub).leaves():
                            if not sets_own_species(leaf):
                                continue
                            if leaf.span.first != leaf.span.last or ";" in _split_code(lines[leaf.span.first])[0]:
                                raise ExtractError(f"{sid}: line {leaf.span.first + 1} has more than the species statement")
                            drops[path].add(leaf.span.first)
                            self.notes.append(f"removed {tname} line {leaf.span.first + 1}: {leaf.text}")
                    old_species[tname] = {SPECIES: f"{CORE}%{SPECIES}"}
        for path in list(self.files):
            lines = self.files[path]
            out = []
            for i, l in enumerate(lines):
                out.extend(inserts.get(path, {}).get(i, []))
                if i not in drops.get(path, set()):
                    out.append(l)
            self.files[path] = out
        for path in modules:
            self.add_imports(path)
        self.rename_old_species(old_species)
        self.add_is_carbon()

    def add_imports(self, path: Path) -> None:
        lines = self.files[path]
        want = ["species_from_string", "CN_SPECIES_N", "CN_SPECIES_P", "is_carbon"]
        rx = re.compile(r"^(\s*use\s+SpeciesMod\s*,\s*only\s*:)(.*)$", re.I)
        for i, l in enumerate(lines):
            m = rx.match(_split_code(l)[0])
            if m:
                have = [n.strip() for n in m.group(2).split(",") if n.strip()]
                missing = [n for n in want if n.lower() not in {h.lower() for h in have}]
                if missing:
                    lines[i] = m.group(1) + " " + ", ".join(have + missing)
                return
        first_use = next(i for i, l in enumerate(lines) if re.match(r"^\s*use\b", l, re.I))
        indent = re.match(r"\s*", lines[first_use]).group(0)
        lines.insert(first_use, f"{indent}use SpeciesMod, only : " + ", ".join(want))

    def rename_old_species(self, by_type: dict[str, dict[str, str]]) -> None:
        if not by_type:
            return
        instances = {i.lower(): t for t in by_type for i in self.fut.type_dict[t].instances}
        decl_rx = re.compile(r"^\s*(?:type|class)\s*\(\s*(\w+)\s*\)[^:]*::(.*)$", re.I)
        for path in sorted(self.elm_src.rglob("*.F90")):
            if any(x in str(path) for x in EXCLUDE):
                continue
            text = self.files[path] if path in self.files else path.read_text(errors="replace").split("\n")
            if not any(re.search(rf"%\s*{SPECIES}\b", l, re.I) for l in text):
                continue
            # this%core%species is already right: only rename obj%species of the carbon types
            n = rename_in_lines(text, instances, by_type, decl_rx)
            if n:
                self.files[path] = text
                self.notes.append(f"renamed {n} uses of %{SPECIES} in {path.name}")

    def add_is_carbon(self) -> None:
        path = next(self.elm_src.rglob("SpeciesMod.F90"))
        lines = self.lines(path)
        if any(re.search(r"\bfunction\s+is_carbon\b", l, re.I) for l in lines):
            return
        pub = next(i for i, l in enumerate(lines) if re.match(r"^\s*public\s*::\s*species_from_string", l, re.I))
        lines.insert(pub + 1, "  public :: is_carbon            ! whether a species is a carbon isotope")
        end = max(i for i, l in enumerate(lines) if re.match(r"^\s*end\s+module\b", l, re.I))
        lines[end:end] = IS_CARBON.split("\n")
        self.notes.append("added is_carbon to SpeciesMod")


def _type_line(lines: list[str], name: str) -> int:
    rx = re.compile(rf"^\s*type\s*(,[^:]*)?::\s*{name}\b", re.I)
    return next(i for i, l in enumerate(lines) if rx.match(_split_code(l)[0]))


def first_executable_line(lines: list[str], r) -> tuple[int, str]:
    """(line of the routine's first executable statement, its indentation)."""
    if not r.body:
        raise ExtractError(f"{r.name} has no executable statement")
    n = r.body[0]
    ln = n.span.first if isinstance(n, Leaf) else n.head.first
    return ln, lines[ln][: len(lines[ln]) - len(lines[ln].lstrip())]


def sets_own_species(leaf: Leaf) -> bool:
    """`allocate(this%species)` or `this%species = ...`: the carbon types' old species."""
    if isinstance(leaf.node, AllocateStatement):
        return [designator_path(o) for o in leaf.node.objects] == [["this", SPECIES]]
    return assigned_path(leaf.node) == ["this", SPECIES]
