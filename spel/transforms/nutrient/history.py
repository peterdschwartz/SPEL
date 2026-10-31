"""
One history registration per `core` field, for every C/N/P variant at once.

Each concrete Init registers its `core` fields itself, the carbon one three
times (c12, c13, c14 under `carbon_type` branches, and none under FATES):

    this%core%leaf(begp:endp) = spval
    call hist_addfld1d (fname='LEAFN', units='gN/m^2', avgflag='A', &
         long_name='leaf N', ptr_patch=this%core%leaf)

`<base>_init_history(this, ...)` registers each such field once, its name,
units and long name made from `this%species` by SpeciesMod helpers
(`nutrient_name(this%species, 'LEAF#')` -> LEAFC, C13_LEAFC, LEAFN, ...;
`nutrient_units(..., 'g#/m^2')` -> gC13/m^2; `nutrient_text(..., 'leaf #')`
-> 'C13 leaf C'). A value no template reproduces for every variant is kept as
it was, per species (`by_species`); a `default=` that differs between
variants becomes `hist_default(<the species it's active for>)`. Every
variant's history output is unchanged: the same fields, names, units, long
names, defaults and pointers.

A field merges when every variant registers it with the same call, the same
other arguments and guards (other than `carbon_type` and FATES), the same
`spval` statement before it (or none), and moving it is safe: the concrete
Init calls the base routine where its first moved registration was, so a
moved `spval` assignment runs earlier than before; no kept statement on that
field and no other call may sit in between. A field registered for only some
variants is registered under a species condition; one only carbon skipped
under FATES, under `.not. (is_carbon(this%species) .and. use_fates)`.

The same generator does restart (RESTART): one `restartvar` per field in
`<base>_restart(this, ...)`, its varname from `nutrient_restart_name`
('leaf#' -> leafc, leafn, leafc_13). A variant whose call is followed by a
use of `readvar` (carbon isotopes initialized from C12 when missing from the
file) keeps its own call, and the base one is conditioned on the other
species. On read, restartvar writes the field, so no kept statement on it may
run between the base call and the call's old place.

Everything comes from the parse tree (tree.py, parsed keeping string case):
the calls, their keyword arguments and values, pointer targets, guards.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spel.scripts.fortran_parser.spel_ast import Identifier, StringLiteral
from spel.transforms.nutrient.common import ORDER, Family
from spel.transforms.nutrient.extract import CORE, ExtractError
from spel.transforms.nutrient.merge import FileEdits, emit
from spel.transforms.nutrient.tree import (
    Block,
    Leaf,
    Routine,
    assigned_path,
    call_keywords,
    declared_names,
    declaration_lines,
    designator_path,
    names_in,
    read_routine,
    statement_names,
    use_lines,
    written_paths,
)

HIST_CALLS = ("hist_addfld1d", "hist_addfld2d", "hist_addfld_decomp")


@dataclass(frozen=True)
class Spec:
    name: str  # history | restart
    binding: str  # the concrete procedure: init | restart
    calls: tuple[str, ...]
    suffix: str  # of the base routine's name
    templates: tuple[tuple[str, str], ...]  # keyword -> SpeciesMod helper
    ptr: str = ""  # the pointer keyword; "" = any ptr_*
    defaults: bool = False  # default= by species (hist_default)
    spval: bool = False  # an `x = spval` before the call goes with it
    readvar: bool = False  # restart: a variant whose readvar is used afterwards stays
    writes_field: bool = False  # the call writes the field (restart reads)

    @property
    def special(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self.templates) + (("default",) if self.defaults else ())


HISTORY = Spec(
    "history", "init", HIST_CALLS, "_init_history",
    (("fname", "nutrient_name"), ("units", "nutrient_units"), ("long_name", "nutrient_text")),
    defaults=True, spval=True,
)
RESTART = Spec(
    "restart", "restart", ("restartvar",), "_restart",
    (("varname", "nutrient_restart_name"), ("long_name", "nutrient_text"), ("units", "nutrient_units")),
    ptr="data", readvar=True, writes_field=True,
)
SPECIAL = HISTORY.special
VARIANTS = ("c12", "c13", "c14", "n", "p")
SPECIES = {v: f"CN_SPECIES_{v.upper()}" for v in VARIANTS}
FATES_TERMS = ("not(use_fates)", "(.not.use_fates)")
ISO_TERM = "(carbon_type=="


@dataclass
class Reg:
    variant: str
    call: Leaf
    spval: Optional[Leaf]
    field: str  # core%leaf
    name: str  # the called routine
    args: dict[str, object]  # keyword -> value expression
    ptr: str  # the pointer keyword (ptr_patch, ptr_col, ...)
    others: tuple[str, ...]  # guards other than carbon_type / FATES
    fates: bool  # under `.not. use_fates`


@dataclass
class FieldPlan:
    field: str
    regs: dict[str, Reg]  # variant -> registration
    statements: list[str] = field(default_factory=list)  # the base routine's lines for it
    guard: tuple[str, ...] = ()  # conditions of the base routine's if around it


def variants_of(element: str, guards: tuple[str, ...]):
    """(variants a statement under `guards` runs for, under FATES guard?, other guards)."""
    isos, fates, others = [], False, []
    for g in guards:
        if g in FATES_TERMS:
            fates = True
        elif "carbon_type" in g:
            if g.startswith(ISO_TERM) and g.endswith(")"):
                isos.append(g[len(ISO_TERM) : -1])
        else:
            others.append(g)
    if element != "c":
        return [element], fates, tuple(others)
    return isos or ["c12", "c13", "c14"], fates, tuple(others)


def sibling_lists(nodes: list):
    """Every list of sibling nodes in a routine body (top level, each block segment)."""
    yield nodes
    for n in nodes:
        if isinstance(n, Block):
            for _, kids in n.segments:
                yield from sibling_lists(kids)


def collect(r: Routine, element: str, spec: Spec = HISTORY) -> list[Reg]:
    out = []
    for nodes in sibling_lists(r.body):
        for i, n in enumerate(nodes):
            if not isinstance(n, Leaf):
                continue
            ck = call_keywords(n)
            if ck is None or ck[0] not in spec.calls:
                continue
            name, parsed = ck
            ptrs = [k for k in parsed if (k == spec.ptr if spec.ptr else k.startswith("ptr_"))]
            if len(ptrs) != 1 or not isinstance(parsed[ptrs[0]], Identifier):
                continue  # no pointer, or a section of a field: left as it is
            path = designator_path(parsed[ptrs[0]])
            if not path or path[:2] != ["this", CORE] or len(path) != 3:
                continue
            prev = nodes[i - 1] if i > 0 else None
            spval = None
            if spec.spval and isinstance(prev, Leaf) and assigned_path(prev.node) == path:
                if str(prev.node.expression.right_expr).lower() == "spval":
                    spval = prev
            variants, fates, others = variants_of(element, n.guards)
            for v in variants:
                out.append(Reg(v, n, spval, f"{CORE}%{path[2]}", name, parsed, ptrs[0], others, fates))
    return out


def unquote(value) -> Optional[str]:
    """The text of a string literal argument, else None."""
    return value.value if isinstance(value, StringLiteral) else None


def fortran(value) -> str:
    """An argument value back as Fortran: string literals quoted, anything else as the parser prints it."""
    if isinstance(value, StringLiteral):
        q = "'" if "'" not in value.value else '"'
        return f"{q}{value.value}{q}"
    return str(value)


def render(kind: str, variant: str, template: str) -> str:
    letter = "C" if variant.startswith("c") else variant.upper()
    iso = variant[1:] if variant in ("c13", "c14") else ""
    if kind == "fname":
        return (f"C{iso}_" if iso else "") + template.replace("#", letter)
    if kind == "units":
        return template.replace("#", letter + iso)
    if kind == "varname":
        return template.replace("#", letter.lower()) + (f"_{iso}" if iso else "")
    return (f"C{iso} " if iso else "") + template.replace("#", letter)


def template_for(kind: str, values: dict[str, str]) -> Optional[str]:
    """A template `render` turns into every variant's value, else None."""
    base_v = next((v for v in ("c12", "n", "p") if v in values), None)
    if base_v is None:
        return None
    text = values[base_v]
    letter = render(kind, base_v, "#")[0]
    sites = [i for i, ch in enumerate(text) if ch == letter][-6:]
    for k in range(0, len(sites) + 1):
        for subset in itertools.combinations(sites, k):
            chars = list(text)
            for i in subset:
                chars[i] = "#"
            t = "".join(chars)
            if all(render(kind, v, t) == values[v] for v in values):
                return t
    return None


def species_condition(variants: set[str]) -> str:
    parts = []
    rest = set(variants)
    if {"c12", "c13", "c14"} <= rest:
        parts.append("is_carbon(this%species)")
        rest -= {"c12", "c13", "c14"}
    parts += [f"this%species == {SPECIES[v]}" for v in VARIANTS if v in rest]
    return " .or. ".join(parts)


def normalized(text: str, dummies: list[str] = ()) -> str:
    """Statement text with `this%core%` dropped: compares variants (dummies by name)."""
    return re.sub(r"\s+", "", text).lower().replace(f"this%{CORE}%", "this%")


HELPERS = """\
  pure function species_label(species) result(label)
    ! C, C13, C14, N or P
    integer, intent(in) :: species
    character(len=:), allocatable :: label
    select case (species)
    case (CN_SPECIES_C13)
       label = 'C13'
    case (CN_SPECIES_C14)
       label = 'C14'
    case (CN_SPECIES_N)
       label = 'N'
    case (CN_SPECIES_P)
       label = 'P'
    case default
       label = 'C'
    end select
  end function species_label

  pure function fill_template(template, with) result(text)
    ! template with each '#' replaced by `with`
    character(len=*), intent(in) :: template, with
    character(len=:), allocatable :: text
    integer :: i
    text = ''
    do i = 1, len(template)
       if (template(i:i) == '#') then
          text = text // with
       else
          text = text // template(i:i)
       end if
    end do
  end function fill_template

  pure function nutrient_name(species, template) result(name)
    ! history field name: '#' is the element letter, carbon isotopes prefixed (C13_LEAF#)
    integer, intent(in) :: species
    character(len=*), intent(in) :: template
    character(len=:), allocatable :: name
    if (species == CN_SPECIES_C13 .or. species == CN_SPECIES_C14) then
       name = species_label(species) // '_' // fill_template(template, 'C')
    else
       name = fill_template(template, species_label(species))
    end if
  end function nutrient_name

  pure function nutrient_units(species, template) result(units)
    ! units: '#' is the element (or isotope) label (g#/m^2 -> gC13/m^2)
    integer, intent(in) :: species
    character(len=*), intent(in) :: template
    character(len=:), allocatable :: units
    units = fill_template(template, species_label(species))
  end function nutrient_units

  pure function nutrient_text(species, template) result(text)
    ! long name: '#' is the element letter, carbon isotopes prefixed (C13 leaf C)
    integer, intent(in) :: species
    character(len=*), intent(in) :: template
    character(len=:), allocatable :: text
    if (species == CN_SPECIES_C13 .or. species == CN_SPECIES_C14) then
       text = species_label(species) // ' ' // fill_template(template, 'C')
    else
       text = fill_template(template, species_label(species))
    end if
  end function nutrient_text

  pure function nutrient_restart_name(species, template) result(name)
    ! restart variable name: '#' is the element letter in lowercase, carbon isotopes suffixed (leaf#_storage -> leafc_storage_13)
    integer, intent(in) :: species
    character(len=*), intent(in) :: template
    character(len=:), allocatable :: name
    select case (species)
    case (CN_SPECIES_C13)
       name = fill_template(template, 'c') // '_13'
    case (CN_SPECIES_C14)
       name = fill_template(template, 'c') // '_14'
    case (CN_SPECIES_N)
       name = fill_template(template, 'n')
    case (CN_SPECIES_P)
       name = fill_template(template, 'p')
    case default
       name = fill_template(template, 'c')
    end select
  end function nutrient_restart_name

  pure function by_species(species, c12, c13, c14, n, p) result(text)
    ! the text given for species
    integer, intent(in) :: species
    character(len=*), intent(in) :: c12, c13, c14, n, p
    character(len=:), allocatable :: text
    select case (species)
    case (CN_SPECIES_C13)
       text = c13
    case (CN_SPECIES_C14)
       text = c14
    case (CN_SPECIES_N)
       text = n
    case (CN_SPECIES_P)
       text = p
    case default
       text = c12
    end select
  end function by_species

  pure function hist_default(active) result(default)
    ! a history field's default= argument
    logical, intent(in) :: active
    character(len=:), allocatable :: default
    if (active) then
       default = 'active'
    else
       default = 'inactive'
    end if
  end function hist_default
"""
HELPER_NAMES = ("nutrient_name", "nutrient_units", "nutrient_text", "nutrient_restart_name", "by_species", "hist_default")
SPECIES_NAMES = ("CN_SPECIES_C12", "CN_SPECIES_C13", "CN_SPECIES_C14", "CN_SPECIES_N", "CN_SPECIES_P")


def helper_blocks() -> dict[str, list[str]]:
    """Each helper function's lines, by name."""
    blocks, cur, name = {}, [], None
    for l in HELPERS.split("\n"):
        words = l.split()
        if "function" in words and not l.strip().lower().startswith("end"):
            name = words[words.index("function") + 1].split("(")[0]
            cur = []
        cur.append(l)
        if l.strip().lower().startswith("end function"):
            blocks[name] = cur + [""]
    return blocks


def add_helpers(lines: list[str]) -> list[str]:
    """SpeciesMod with the helpers it doesn't have yet (and their public statements)."""
    out = list(lines)
    have = set()
    for l in out:
        words = l.split("!")[0].lower().split()
        if "function" in words and words[0] != "end":
            have.add(words[words.index("function") + 1].split("(")[0])
    missing = {n: b for n, b in helper_blocks().items() if n not in have}
    if not missing:
        return out
    pub = next(i for i, l in enumerate(out) if l.strip().lower().startswith("public :: species_from_string"))
    out[pub + 1 : pub + 1] = [f"  public :: {n}" for n in HELPER_NAMES if n in missing]
    end = max(i for i, l in enumerate(out) if l.strip().lower().startswith("end module"))
    out[end:end] = [l for b in missing.values() for l in b]
    return out


class HistoryMerger:
    def __init__(self, families: list[Family], fut, spec: Spec = HISTORY):
        self.spec = spec
        self.fut = fut
        self.families = [f for f in families if f.name in fut.type_dict]
        self.edits: dict[Path, FileEdits] = {}
        self.results: list[str] = []
        self.refused: list[str] = []
        self.kept: dict[str, dict[str, str]] = {}  # family -> field -> why it stays
        self._lines: dict[Path, list[str]] = {}

    def lines(self, path: Path) -> list[str]:
        path = Path(path)
        if path not in self._lines:
            self._lines[path] = path.read_text(errors="replace").split("\n")
        return self._lines[path]

    def build(self) -> None:
        for fam in self.families:
            try:
                self.merge(fam)
            except ExtractError as err:
                self.refused.append(f"{fam.name}: {err}")

    def merge(self, fam: Family) -> None:
        td, sd = self.fut.type_dict, self.fut.subroutine_dict
        routines: dict[str, Routine] = {}
        for e, tname in fam.types.items():
            sid = td[tname].procedures.get(self.spec.binding)
            if sid not in sd:
                raise ExtractError(f"{tname}: no analyzed {self.spec.binding}")
            routines[e] = read_routine(sd[sid])
            if routines[e].unsupported:
                raise ExtractError(f"{sid}: {'; '.join(routines[e].unsupported)}")
        paths = {r.path for r in routines.values()}
        if len(paths) != 1:
            raise ExtractError("routines in several files")
        path = paths.pop()
        lines = self.lines(path)
        universe = {v for e in routines for v in (("c12", "c13", "c14") if e == "c" else (e,))}
        regs = {e: collect(r, e, self.spec) for e, r in routines.items()}
        by_field: dict[str, dict[str, Reg]] = {}
        why: dict[str, str] = {}
        for e, rs in regs.items():
            for g in rs:
                if g.variant in by_field.setdefault(g.field, {}):
                    why[g.field] = f"registered twice for {g.variant}"
                by_field[g.field][g.variant] = g
        if self.spec.readvar:
            # a variant whose readvar is used after its call keeps the call
            for fld, vs in by_field.items():
                for v in [v for v, g in vs.items() if self.readvar_used(g, routines[self._element(g)])]:
                    vs.pop(v)
                if len(vs) < 2 and fld not in why:
                    why[fld] = "readvar used after the call"
        plans: dict[str, FieldPlan] = {}
        for fld, vs in by_field.items():
            if fld in why:
                continue
            reason = self.check(vs, routines)
            if reason:
                why[fld] = reason
                continue
            plans[fld] = FieldPlan(fld, vs)
        # order: a moved spval assignment runs at the base call; nothing kept in between may touch it
        while True:
            blocked = self.order_blocked(plans, routines)
            if not blocked:
                break
            for fld, reason in blocked.items():
                why[fld] = reason
                plans.pop(fld)
        self.kept[fam.name] = why
        if not plans:
            return
        base_name = f"{fam.name}{self.spec.suffix}"
        if any(k.split("::")[-1] == base_name for k in sd):
            raise ExtractError(f"{base_name} exists already (merged before)")
        ref_e = next(e for e in ORDER if e in routines)
        body = self.base_body(fam, plans, routines, universe, ref_e)
        dummies, spec_lines = self.base_signature(self.needed_names(plans), routines, ref_e, lines)
        out = [
            f"  subroutine {base_name}({', '.join(['this'] + dummies)})",
            "    !",
            "    ! !DESCRIPTION:",
            f"    ! {'History' if self.spec.name == 'history' else 'Restart'} fields of the core of {', '.join(fam.types.values())}:",
            f"    ! one {self.spec.calls[0]} per field for every species (generated by `spel nutrient {self.spec.name}`)",
            "    !",
            *spec_lines[: spec_lines.index("    ! !ARGUMENTS:") + 1],
            f"    type({fam.name}), intent(inout) :: this",
            *spec_lines[spec_lines.index("    ! !ARGUMENTS:") + 1 :],
            "    !-----------------------------------------------------------------------",
            "",
            *body,
            "",
            f"  end subroutine {base_name}",
            "",
        ]
        first = min(r.start for r in routines.values())
        fe = self.edits.setdefault(path, FileEdits())
        fe.insert.setdefault(first, []).extend(out)
        moved_ids = {id(g.call) for p in plans.values() for g in p.regs.values()}
        moved_ids |= {id(g.spval) for p in plans.values() for g in p.regs.values() if g.spval}
        for e, r in routines.items():
            keep = lambda l: id(l) not in moved_ids
            local = r.local_names

            def scaffold(l: Leaf, local=local) -> bool:
                ps = written_paths(l.node)
                return bool(ps) and all(p[0] in local and len(p) == 1 for p in ps) and not l.is_call

            essential = lambda l: keep(l) and not scaffold(l)
            untouched = lambda b: not any(id(l) in moved_ids for l in b.leaves())
            pairs = emit(lines, r, keep=keep, essential=essential, scaffold=scaffold, untouched=untouched)
            at_line = self.call_line(r, moved_ids)
            at = next(i for i, (ln, _) in enumerate(pairs) if ln >= at_line)
            ref_line = pairs[at][1]
            indent = ref_line[: len(ref_line) - len(ref_line.lstrip())] or "    "
            args = ", ".join([f"this%{CORE}"] + self._dummy_names)
            new = [t for _, t in pairs]
            new.insert(at, f"{indent}call {base_name}({args})")
            fe.add_replace(r.start, r.end, new)
        self.results.append(
            f"{fam.name}: {base_name}, {len(plans)} fields ({len(why)} kept: "
            + ", ".join(f"{k}" for k in list(why)[:6])
            + (" ..." if len(why) > 6 else "")
            + ")"
        )

    def check(self, vs: dict[str, Reg], routines) -> str:
        regs = list(vs.values())
        if len({g.name for g in regs}) > 1:
            return "different calls"
        if len({g.others for g in regs}) > 1:
            return "different guards"
        if len({g.ptr for g in regs}) > 1:
            return "different pointer kinds"
        carbon_fates = {g.fates for g in regs if g.variant.startswith("c")}
        other_fates = {g.fates for g in regs if not g.variant.startswith("c")}
        if len(carbon_fates) > 1 or len(other_fates) > 1 or (other_fates == {True} and carbon_fates == {False}):
            return "FATES guards that don't fit"
        keys = {k for g in regs for k in g.args}
        for k in keys - set(self.spec.special) - {regs[0].ptr}:
            vals = {fortran(g.args[k]) if k in g.args else None for g in regs}
            if len(vals) > 1:
                return f"different {k}="
            if not isinstance(regs[0].args[k], (StringLiteral, Identifier)):
                return f"{k}= is an expression"
        for k, _ in self.spec.templates:
            vals = [g.args.get(k) for g in regs]
            if any(v is None for v in vals) and not all(v is None for v in vals):
                return f"{k}= missing in some"
            if all(v is not None for v in vals) and any(unquote(v) is None for v in vals):
                return f"{k}= not a string literal"
        if len({g.spval is None for g in regs}) > 1:
            return "spval before it in some"
        spvals = {normalized(g.spval.text, routines[self._element(g)].dummies) for g in regs if g.spval}
        if len(spvals) > 1:
            return "different spval statements"
        return ""

    def readvar_used(self, g: Reg, r: Routine) -> bool:
        """Whether `readvar` (the call's out-argument) is read before the next such call."""
        leaves = list(r.leaves())
        for l in leaves[leaves.index(g.call) + 1 :]:
            if l.is_call and (call_keywords(l) or ("",))[0] in self.spec.calls:
                return False
            mentions = re.search(r"\breadvar\b", l.text.lower()) or any("readvar" in x for x in l.guards)
            if mentions:
                return True
        return False

    @staticmethod
    def _element(g: Reg) -> str:
        return "c" if g.variant.startswith("c") else g.variant

    def order_blocked(self, plans: dict[str, FieldPlan], routines) -> dict[str, str]:
        moved = {id(g.call) for p in plans.values() for g in p.regs.values()}
        moved |= {id(g.spval) for p in plans.values() for g in p.regs.values() if g.spval}
        out = {}
        for e, r in routines.items():
            leaves = list(r.leaves())
            idx = [i for i, l in enumerate(leaves) if id(l) in moved]
            if not idx:
                continue
            start = self.call_index(r, leaves, moved)
            for fld, p in plans.items():
                g = next((g for v, g in p.regs.items() if self._element(g) == e), None)
                if g is None:
                    continue
                first = g.spval if g.spval is not None else g.call if self.spec.writes_field else None
                if first is None:
                    continue
                j = leaves.index(first)
                name = fld.split("%")[1]
                for l in leaves[start:j]:
                    if id(l) in moved:
                        continue
                    if (l.is_call and (call_keywords(l) or ("",))[0] not in self.spec.calls) or any(
                        p_[:3] == ["this", CORE, name] for p_ in (written_paths(l.node) or [])
                    ) or f"%{CORE}%{name}" in re.sub(r"\s+", "", l.text).lower():
                        out[fld] = f"{e}: line {l.ln + 1} runs between the base call and the statement moved"
                        break
        return out

    @staticmethod
    def call_index(r: Routine, leaves: list[Leaf], moved: set[int]) -> int:
        """Index (in `leaves`) of the first leaf of the top-level node holding the first moved leaf."""
        for n in r.body:
            own = [n] if isinstance(n, Leaf) else list(n.leaves())
            if any(id(l) in moved for l in own):
                return leaves.index(own[0]) if own else 0
        return len(leaves)

    @staticmethod
    def call_line(r: Routine, moved: set[int]) -> int:
        for i, n in enumerate(r.body):
            own = [n] if isinstance(n, Leaf) else list(n.leaves())
            if any(id(l) in moved for l in own):
                prev = r.body[i - 1] if i > 0 else None
                if isinstance(prev, Leaf) and prev.is_directive and not prev.closes:
                    return prev.span.first
                return n.span.first if isinstance(n, Leaf) else n.head.first
        return r.tail.first

    def needed_names(self, plans: dict[str, FieldPlan]) -> set[str]:
        """Names the base statements refer to (from the parse tree): call arguments, spval statements, guards."""
        names: set[str] = set()
        for p in plans.values():
            g0 = next(iter(p.regs.values()))
            for k, v in g0.args.items():
                if k != g0.ptr:
                    names |= names_in(v)
            if g0.spval is not None:
                names |= statement_names(g0.spval.node)
            for gs, ge in zip(g0.call.guards, g0.call.guard_exprs):
                if gs in g0.others:
                    names |= names_in(ge)
        return names - {"this"}

    def base_signature(self, names: set[str], routines, ref_e, lines) -> tuple[list[str], list[str]]:
        """
        The base routine's dummies (the reference's, which every routine has by name)
        and its specification lines: `use` of the names a routine imports itself, then the
        declarations of the dummies and locals it needs. Names are resolved in the reference
        routine, then in the others; the rest are visible from the module.
        """
        ref = routines[ref_e]
        positions, decls, uses = set(), [], {}
        rest = set(names)
        for e in [ref_e] + [e for e in routines if e != ref_e]:
            if not rest:
                break
            r = routines[e]
            d, dc, us = r.requirements(rest)
            positions |= set(d)
            decls += dc
            for module, ns in us:
                uses.setdefault(module, [])
                uses[module] += [n for n in ns if n not in uses[module]]
            rest -= {r.dummies[i].lower() for i in d} | {n for _, _, ns in dc for n in ns}
            rest -= {n for _, ns in us for n in ns}
        dummies = [ref.dummies[i] for i in sorted(positions)]
        for r in routines.values():
            missing = [d for d in dummies if d.lower() not in {x.lower() for x in r.dummies}]
            if missing:
                raise ExtractError(f"{r.name} has no dummy {missing}: dummies correspond by name")
        self._dummy_names = dummies
        # declarations of the dummies as the reference names them
        _, dummy_decls, _ = ref.requirements({d.lower() for d in dummies})
        seen, spec = set(), []
        for item in dummy_decls + decls:
            if id(item[1]) not in seen:
                seen.add(id(item[1]))
                spec.append(item)
        return dummies, use_lines(list(uses.items())) + ["    ! !ARGUMENTS:"] + declaration_lines(lines, spec)

    def base_body(self, fam, plans, routines, universe, ref_e) -> list[str]:
        out: list[str] = []
        open_guards: list[str] = []
        ref = routines[ref_e]
        # the order of the first variant that has the field
        order = []
        seen = set()
        for e in ORDER:
            for g in sorted((g for p in plans.values() for g in p.regs.values() if self._element(g) == e), key=lambda g: g.call.ln):
                if g.field not in seen:
                    seen.add(g.field)
                    order.append(g.field)
        for fld in order:
            p = plans[fld]
            regs = p.regs
            g0 = next(iter(regs.values()))
            guards = []
            carbon_fates = any(g.fates for g in regs.values() if g.variant.startswith("c"))
            other_fates = any(g.fates for g in regs.values() if not g.variant.startswith("c"))
            if carbon_fates and other_fates:
                guards.append(".not. use_fates")
            elif carbon_fates:
                guards.append(".not. (is_carbon(this%species) .and. use_fates)")
            guards += [self.fortran_guard(x) for x in g0.others]
            present = set(regs)
            if present != universe:
                guards.append(species_condition(present))
            # close/open ifs to match
            k = 0
            while k < len(open_guards) and k < len(guards) and open_guards[k] == guards[k]:
                k += 1
            while len(open_guards) > k:
                open_guards.pop()
                out.append("    " + "   " * len(open_guards) + "end if")
            while len(open_guards) < len(guards):
                out.append("    " + "   " * len(open_guards) + f"if ({guards[len(open_guards)]}) then")
                open_guards.append(guards[len(open_guards)])
            ind = "    " + "   " * len(open_guards)
            if g0.spval:
                text = re.sub(rf"\bthis(\s*)%(\s*){CORE}\s*%", r"this\1%\2", g0.spval.text, flags=re.I)
                out.append(ind + text)
            out.extend(self.registration(fld, regs, ind))
            out.append("")
        while open_guards:
            open_guards.pop()
            out.append("    " + "   " * len(open_guards) + "end if")
        return out

    @staticmethod
    def fortran_guard(norm: str) -> str:
        """A guard back as Fortran (guards are kept normalized: `(crop_prog)` -> `crop_prog`)."""
        return norm[1:-1] if norm.startswith("(") and norm.endswith(")") else norm

    def registration(self, fld: str, regs: dict[str, Reg], ind: str) -> list[str]:
        g0 = next(iter(regs.values()))
        args = []
        for k, v in g0.args.items():
            if k == g0.ptr:
                args.append(f"{k}=this%{fld.split('%')[1]}")
            elif k in dict(self.spec.templates):
                values = {var: unquote(g.args[k]) for var, g in regs.items()}
                if len(set(values.values())) == 1:
                    args.append(f"{k}={fortran(v)}")  # the same for every species
                    continue
                t = template_for(k, values)
                fn = dict(self.spec.templates)[k]
                if t is not None:
                    args.append(f"{k}={fn}(this%species, '{t}')")
                else:
                    lits = ", ".join(fortran(StringLiteral(None, values.get(var, ""))) for var in VARIANTS)
                    args.append(f"{k}=by_species(this%species, {lits})")
            elif k == "default" and self.spec.defaults:
                continue
            else:
                args.append(f"{k}={fortran(v)}")
        defaults = {var: (unquote(g.args["default"]) or "").lower() if "default" in g.args else "active" for var, g in regs.items()}
        if not self.spec.defaults:
            pass
        elif len(set(defaults.values())) == 1:
            if "default" in g0.args:
                args.append(f"default={fortran(g0.args['default'])}")
        else:
            active = {var for var, d in defaults.items() if d == "active"}
            args.append(f"default=hist_default({species_condition(active)})")
        lines, cur = [], f"{ind}call {g0.name} ("
        for i, a in enumerate(args):
            piece = a + (")" if i == len(args) - 1 else ", ")
            if len(cur) + len(piece) > 100 and not cur.endswith("("):
                lines.append(cur.rstrip() + " &")
                cur = ind + "     "
            cur += piece
        lines.append(cur)
        return lines

    def apply(self) -> dict[Path, list[str]]:
        files = {p: fe.apply(self.lines(p)) for p, fe in self.edits.items()}
        if files:
            species = next(Path(next(iter(files))).parents[1].rglob("SpeciesMod.F90"))
            files[species] = add_helpers(self.lines(species))
            for p in list(files):
                if p != species:
                    files[p] = self.add_imports(files[p])
        return files

    @staticmethod
    def add_imports(lines: list[str]) -> list[str]:
        """Each edited module uses the helpers from SpeciesMod."""
        for i, l in enumerate(lines):
            code = l.split("!")[0]
            if code.strip().lower().startswith("use speciesmod") and "only" in code.lower():
                have = {x.strip().lower() for x in code.split(":", 1)[1].split(",")}
                missing = [n for n in HELPER_NAMES + SPECIES_NAMES if n.lower() not in have]
                if missing:
                    lines[i] = code.rstrip() + ", " + ", ".join(missing)
                return lines
        raise ExtractError("no `use SpeciesMod, only :` line (run `spel transform nutrient-species` first)")

    def report(self) -> str:
        out = list(self.results)
        for fam, why in self.kept.items():
            for fld, reason in why.items():
                out.append(f"  {fam} kept {fld}: {reason}")
        out += [f"not merged: {m}" for m in self.refused]
        return "\n".join(out)
