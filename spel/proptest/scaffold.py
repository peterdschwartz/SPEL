"""`spel proptest init`: a starter `<case>.proptest.toml` for a case.

It writes what can be known without judgement: the case, a domain, one scenario that takes its
rules from the `stats` rule file, and two checks every spec should start with (finite outputs,
identical copies). What needs a decision is left as `# TODO` lines: inputs the reference gives no
rule for, and constants it lacks.

The domain's landunit kinds follow the filters elm_drv passes the routines (`routine_landunits`):
urban filters (`urbanl`, ...) give urban landunits, lake filters a lake, `icemecc` glacier-mec,
any other filter (or none) vegetation; the complement of a filter the routine also takes
(`nourbanl` next to `urbanl`) adds nothing. `--landunits` names the kinds instead. Each kind gets
a gridcell (`[[domain.gridcell]]`, two copies): vegetation one soil column with the pfts the
reference has most of, a town the urban densities of the reference's active urban landunits, a
lake, a wetland, a glacier, an ice sheet (two glacier-mec classes in the glcmask), a tundra (soil
and two polygons). The statistics cover the same landunit kinds. Without vegetation
the scenario runs over the kind's patch filter (`p in urbanp`), and an urban domain derives the
canyon geometry the way ELM does (`URBAN_DERIVE`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from spel.proptest import spec as specmod
from spel.proptest import stats
from spel.proptest import tomlw
from spel.proptest.domain import (
    CONSTANTS_FILE,
    FILTER_COMPLEMENTS,
    FILTER_KINDS,
    INPUTS_FILE,
    KIND_PATCH_FILTER,
    DomainSpec,
    density_of,
    read_constants,
)
from spel.proptest.hierarchy import (Glacier, GlacierMec, Gridcell, Lake, LandunitKind, Polygon, Soil, Urban,
                                     UrbanDensity, Wetland)
from spel.proptest.errors import SpecError
from spel.proptest.fexpr import ExprError, designator, parse_expr
from spel.proptest.inventory import Inventory

MAX_PFTS = 3
SCENARIO = "default"
VEGETATED = {LandunitKind.SOIL, LandunitKind.CROP}
COPIES = 2
DEFAULT_DENSITIES = [UrbanDensity.HD, UrbanDensity.MD]

# The urban canyon geometry is not independent: UrbanParamsType::Init derives the displacement
# height and roughness of the town from canyon_hwr and ht_roof (Macdonald 1998, Grimmond and Oke
# 1999: alpha = 4.43, beta = 1, C_d = 1.2, vkc = 0.4, plan area index hwr/(hwr+1)), the wind height
# in the canyon is half the roof height, and CanopyTemperatureMod puts the forcing heights of urban
# patches above them. Drawn independently, about 30% of the drawn landunits have ht_roof - z_d <= z_0
# and UrbanFluxes takes logs of negative numbers. Written for the variables the routines read.
_PLAN = "(lun_pp%canyon_hwr(l) / (lun_pp%canyon_hwr(l) + 1.0))"
_ZD_H = "(1.0 - lun_pp%z_d_town(l) / lun_pp%ht_roof(l))"
URBAN_DERIVE = [
    (
        "lun_pp%z_d_town",
        f"(1.0 + 4.43**(-{_PLAN}) * ({_PLAN} - 1.0)) * lun_pp%ht_roof(l)",
    ),
    (
        "lun_pp%z_0_town",
        f"lun_pp%ht_roof(l) * {_ZD_H} * exp(-1.0 * (0.5 * 1.2 / 0.4**2 * {_ZD_H} "
        f"* (1.0 - {_PLAN}) * lun_pp%canyon_hwr(l))**(-0.5))",
    ),
    ("urbanparams_vars%wind_hgt_canyon", "0.5 * lun_pp%ht_roof(l)"),
    # 30 m: the atmosphere's forcing height (atm2lnd forc_hgt) under the town's roughness layer
    (
        "frictionvel_vars%forc_hgt_u_patch",
        "30.0 + lun_pp%z_0_town(l) + lun_pp%z_d_town(l)",
    ),
    (
        "frictionvel_vars%forc_hgt_t_patch",
        "30.0 + lun_pp%z_0_town(l) + lun_pp%z_d_town(l)",
    ),
    (
        "frictionvel_vars%forc_hgt_q_patch",
        "30.0 + lun_pp%z_0_town(l) + lun_pp%z_d_town(l)",
    ),
]


def routine_landunits(inv: Inventory) -> set[LandunitKind]:
    """The landunit kinds the routines work on, from the filters elm_drv passes them (module doc)."""
    members = set()
    for b in getattr(inv.unit_test, "driver_bindings", []) or []:
        if not any(b.callee == r for r in inv.routines):
            continue
        try:
            path = designator(parse_expr(b.actual))
        except ExprError:
            continue
        if path and len(path) == 2 and path[0] == "filter":
            members.add(path[1].removeprefix("num_"))
    kinds = {FILTER_KINDS[m] for m in members if m in FILTER_KINDS}
    others = {
        m
        for m in members
        if m not in FILTER_KINDS and FILTER_COMPLEMENTS.get(m) not in members
    }
    if others or not kinds:
        kinds |= VEGETATED
    return kinds


def reference_dir(case_dir: Path, reference: Optional[str]) -> Optional[Path]:
    """The capture to use: the named one (a directory, or a name under unit-tests/input-data),
    else the case's own input-data when it has a capture."""
    input_data = Path(case_dir).parent / "input-data"
    if reference:
        ref = Path(reference)
        if not ref.is_dir():
            ref = input_data / reference
        if not ref.is_dir():
            raise SpecError(f"--reference {reference!r}: no such capture directory")
        return ref.resolve()
    own = input_data / Path(case_dir).name
    return own if (own / INPUTS_FILE).is_file() else None


def declared_default(inv: Inventory, name: str) -> Any:
    """The value a constant's declaration gives it (None when it has none)."""
    if inv.namelist.get(name) is not None:
        return inv.namelist[name]
    var = (
        getattr(inv.unit_test, "non_parameter_global_vars", {}).get(name)
        if inv.unit_test is not None
        else None
    )
    text = str(getattr(var, "default_value", "") or "").strip()
    if text in ("", "None"):
        return None
    from spel.transforms.core.config import _python_value

    return _python_value(text)


def write(
    case_dir: Path,
    inv: Inventory,
    out: Path,
    reference: Optional[Path] = None,
    where: Optional[str] = None,
    stride: int = 1,
    constants: Optional[dict[str, Any]] = None,
    force: bool = False,
    landunits: Optional[set[LandunitKind]] = None,
) -> tuple[Path, list[str]]:
    """Write the starter spec to `out`; returns it and what is left to do (one line each).
    constants: options and constants given on the command line (`--namelist NAME=VALUE`).
    landunits: the landunit kinds of the domain (default: `routine_landunits`)."""
    out = Path(out).resolve()
    if out.exists() and not force:
        raise SpecError(f"{out} exists (use --force to replace it)")
    case = Path(case_dir).name
    given = {k.lower(): v for k, v in (constants or {}).items()}
    todo: list[str] = []

    consts = read_constants(reference / CONSTANTS_FILE) if reference is not None else {}
    in_reference = set(consts)
    consts.update(given)
    # scalars the routines read that nobody gave: the declaration's default, said in the file
    defaulted: dict[str, Any] = {}
    lacking: list[str] = []
    for c in inv.constants.values():
        if c.rank == 0 and c.status and c.name not in consts:
            d = declared_default(inv, c.name)
            if d is None:
                lacking.append(c.name)
            else:
                defaulted[c.name] = d
    consts.update(defaulted)
    for (
        c
    ) in (
        inv.constants.values()
    ):  # scalars nobody reads and the reference lacks are 0 in the domain too
        if c.rank == 0:
            consts.setdefault(c.name, 0)
    live = inv.configured(
        consts
    )  # the variables that matter under these namelist options

    kinds = set(landunits) if landunits else routine_landunits(inv)
    vegetated = bool(kinds & VEGETATED)
    rule_file = None
    pfts: list[int] = []
    densities: list[UrbanDensity] = []
    why: dict[str, str] = {}
    if reference is not None:
        # the statistics of the landunit kinds the domain has
        restrict = None if kinds == set(LandunitKind) else kinds
        res = stats.compute(
            live,
            reference,
            where=where,
            stride=stride,
            constants=consts,
            landunits=restrict,
        )
        rule_file, _ = stats.write(live, res, case_dir)
        pfts = [
            p for p, _ in sorted(res.pfts.items(), key=lambda kv: -kv[1])[:MAX_PFTS]
        ]
        densities = sorted(
            {d for t in res.landunits if (d := density_of(t)) is not None},
            key=lambda d: d.value,
        )
        why = {s.name: s.missing for s in res.inputs if s.missing}
    else:
        todo.append(
            "no reference capture: every input needs a rule and every parameter a value (--reference)"
        )

    lines = [
        f"# Property-based tests of {', '.join(r.partition('::')[2] or r for r in inv.routines)} (case {case}).",
        "# Written by `spel proptest init`; the format is described in spel/proptest/SPEC.md.",
        "#   spel proptest show  FILE     what is still missing, the namelist options in use",
        "#   spel proptest probe FILE     run a few examples: outputs never written, elements selected",
        "#   spel proptest run   FILE",
        "",
        "[case]",
        f'name = "{case}"',
        "",
        "[settings]",
        "max_examples = 50",
        "",
    ]
    takes = ", ".join(sorted(k.value for k in kinds))
    gridcells: list[Gridcell] = []
    notes: list[str] = []
    if vegetated:
        if pfts:
            notes.append(f"vegetation: one soil column with the reference's {len(pfts)} most common pfts")
        else:
            pfts = [1]
            notes.append("vegetation: TODO choose the pfts")
            todo.append("[domain] the vegetated gridcell's pfts: choose them")
        gridcells.append(Gridcell.of(Soil(*sorted(pfts)), copies=COPIES, name="vegetation"))
    if LandunitKind.URBAN in kinds:
        if not densities:
            densities = DEFAULT_DENSITIES
            todo.append("[domain] the town's urban densities: choose them (the reference has no active urban landunit)")
        gridcells.append(Gridcell.of(*(Urban(d) for d in densities), copies=COPIES, name="town"))
        notes.append("town: the densities of the reference's active urban landunits, five columns each")
    if LandunitKind.LAKE in kinds:
        gridcells.append(Gridcell.of(Lake(), copies=COPIES, name="lake"))
    if LandunitKind.WETLAND in kinds:
        gridcells.append(Gridcell.of(Wetland(), copies=COPIES, name="wetland"))
    if LandunitKind.GLACIER in kinds:
        gridcells.append(Gridcell.of(Glacier(), copies=COPIES, name="glacier"))
    if LandunitKind.GLACIER_MEC in kinds:
        gridcells.append(Gridcell.of(GlacierMec({1: 0.5, 2: 0.5}), copies=COPIES, name="ice sheet", glcmask=True))
        notes.append("ice sheet: glacier-mec with two elevation classes in the glcmask (give each a `topo` if the "
                     "routines read glc_topo)")
    if LandunitKind.POLYGON in kinds:
        tundra = sorted(pfts) if pfts else [1]
        gridcells.append(Gridcell.of(Soil(*tundra, weight=0.4), Polygon("low", *tundra, weight=0.3),
                                     Polygon("high", *tundra, weight=0.3), copies=COPIES, name="tundra"))
    dom = DomainSpec(gridcells, nsets=2, reference=reference)
    head = [f"landunit kinds: {takes} ({'--landunits' if landunits else 'from the filters the routines take'})",
            "nsets = 2: relations use the two halves of the sets", *notes]
    lines.append(tomlw.dumps({"domain": dom.to_table(Path(case_dir).parent / "input-data")},
                             {("domain",): "\n".join(head)}).rstrip("\n"))

    # the namelist options: what the spec sets, and what the routines test with the reference's values
    options = live.options()
    settable = {
        k: v
        for k, v in {**defaulted, **given}.items()
        if k in inv.constants and inv.constants[k].rank == 0
    }
    lines += [
        "",
        "[namelist]",
        "# Options of this spec, over the reference's. Variables only used under options that are off need",
        "# no rule and cannot be named in a check. A scenario or a relation can change options:",
        "#   namelist = { option = value }",
    ]
    for name in sorted(settable):
        note = (
            "   # the declaration's default; the reference does not record it"
            if name in defaulted and name not in given
            else ""
        )
        lines.append(f"{name} = {_toml(_typed(inv, name, settable[name]))}{note}")
    for name in lacking:
        lines.append(
            f"# TODO {name} = ?   (the routines read it; the reference does not record it)"
        )
        todo.append(
            f"[namelist] {name}: the routines read it and nobody gave its value"
        )
    shown = [
        f"{n} = {inv.fortran_value(n, consts[n])}"
        for n in sorted(options)
        if n in in_reference and n not in settable
    ]
    if shown:
        lines.append(
            "# The routines also test (the reference's values): " + ", ".join(shown)
        )

    over = "p"
    if (
        not vegetated
        and len(kinds) == 1
        and next(iter(kinds)) in KIND_PATCH_FILTER
        and inv.filters.get(KIND_PATCH_FILTER[next(iter(kinds))]) == 1
    ):
        over = f"p in {KIND_PATCH_FILTER[next(iter(kinds))]}"
    lines += ["", f"[scenario.{SCENARIO}]", f'over = "{over}"']
    if rule_file is not None:
        rel = os.path.relpath(rule_file, out.parent)
        if rel.count("..") > 2:  # a spec far from the case: the path as it is
            rel = str(rule_file)
        lines.append(
            f'extends = "{rel}"   # from `spel proptest stats`: a starting point'
        )
    lines += [
        "",
        f"[scenario.{SCENARIO}.inputs]",
        "# Override the rule file here: widen what the reference under-samples, pin what the scenario excludes.",
    ]
    body_at = len(lines)
    lines += [
        "",
        f"[scenario.{SCENARIO}.derive]",
        "# Inputs that must be consistent with others, as Fortran expressions:",
        '# "top_as%thbot" = "top_as%tbot(t)"',
    ]
    if LandunitKind.URBAN in kinds:
        derived = [
            (n, e)
            for n, e in URBAN_DERIVE
            if n in live.vars and live.vars[n].kind in specmod.INPUT_KINDS
        ]
        if derived:
            lines += [
                "# The canyon geometry as ELM derives it (UrbanParamsType, CanopyTemperatureMod): drawn",
                "# independently, the roughness length can exceed the roof height (NaN outputs).",
            ]
            lines += [f'"{n}" = "{e}"' for n, e in derived]

    outputs = [
        v
        for v in live.of_kind("output")
        if v.level == "p"
        and v.rank == 1
        and v.type == "real"
        and _guard(live, v, consts)
    ]
    if outputs:
        expr = "\n          .and. ".join(f"finite({v.name}(p))" for v in outputs)
        same = "\n            .and. ".join(
            f"same(follow%{v.name}(p), base%{v.name}(p))" for v in outputs
        )
        lines += [
            "",
            "[[property]]",
            'name = "outputs are finite"',
            "# Every real patch output. `spel proptest probe` lists the ones this scenario never writes: remove them.",
            f'expr = """{expr}"""',
            "",
            "[[relation]]",
            'name = "copies of the inputs give identical outputs"',
            "# no `follow`: the second half of the sets is a copy of the first",
            f'expect = """{same}"""',
        ]
    else:
        lines += [
            "",
            "# TODO [[property]]: the routines have no real patch-level output to start from",
        ]
        todo.append("[[property]]: write the first check")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    # the inputs still without a rule, as TODO lines in the scenario's table
    sp, _ = specmod.load(out, inv=inv)
    scen = sp.scenarios[SCENARIO]
    values = specmod.constants(sp, inv, scen)
    missing = specmod.missing_rules(scen, inv.configured(values), values)
    if missing:
        extra = []
        for name in missing:
            var = inv.vars[name]
            dims = ", ".join(d.text for d in var.dims)
            note = (
                f"   # {var.type}({dims})"
                + (f" [{var.units}]" if var.units else "")
                + (f": {why[name]}" if name in why else "")
            )
            extra.append(f'# TODO "{name}" = {{ range = [?, ?] }}{note}')
        lines[body_at:body_at] = extra
        out.write_text("\n".join(lines) + "\n")
        todo.append(
            f"[scenario.{SCENARIO}.inputs]: {len(missing)} inputs have no rule (TODO lines)"
        )
    return out, todo


def _typed(inv: Inventory, name: str, value: Any) -> Any:
    """A constant's value in its own type (a logical given as 0/1 is written as false/true)."""
    c = inv.constants.get(name)
    if c is not None and c.type == "logical":
        return bool(value)
    if c is not None and c.type == "integer" and float(value) == int(value):
        return int(value)
    return value


def _guard(inv: Inventory, var, consts: dict[str, Any]) -> bool:
    try:
        return inv.guard_active(var, consts)
    except Exception:  # a guard on a constant nobody gave: keep the variable
        return True


def _toml(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    return repr(v)
