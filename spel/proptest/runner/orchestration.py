"""Running a spec (`run`): every selected check of every selected scenario as a Hypothesis test,
one `Outcome` each; on drawn domains (`[domain.draw]`, domains.py) on each drawn domain, one
Session each, shrinking a failing domain (`_shrink`). `replay` runs a saved failure again;
`write_json`/`write_junit` report the outcomes."""

from __future__ import annotations

import dataclasses
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from spel.proptest.errors import SpecError
from spel.proptest.inventory import Inventory
from spel.proptest.runner.failures import DOMAIN_FILE, EXAMPLE_FILE
from spel.proptest.runner.reporting import FailureSignature, PropertyFailure, _indent
from spel.proptest.runner.session import Session
from spel.proptest.spec import Spec


@dataclass
class Outcome:
    scenario: str
    check: str
    passed: bool
    message: str = ""
    kind: str = ""  # property | relation | differential
    seconds: float = 0.0
    failure: str = ""  # directory of the saved failing example
    domain: str = ""  # drawn domains: which one
    signature: Optional[FailureSignature] = None  # how it failed (PropertyFailure only)


def run(
    spec: Spec,
    inv: Inventory,
    scenarios: Optional[list[str]] = None,
    checks: Optional[list[str]] = None,
    settings: Optional[dict] = None,
    fc: Optional[str] = None,
    rebuild: bool = False,
    out=sys.stdout,
    no_build: bool = False,
    acc: Optional[str] = None,
    session: Optional[Session] = None,
    backend: Any = None,
) -> list[Outcome]:
    """Every selected check of every selected scenario as a Hypothesis test produces one Outcome.
    Session: use this one (it stays open) instead of making one from the spec.
    With drawn domains (`[domain.draw]`), every check runs on each of the draw's domains (from
    the run's seed), one session each.
    A failure is shrunk to a smaller domain and saved with it.
    backend: Session's ."""
    if spec.domain.draw is None or session is not None:
        return _run_domain(
            spec,
            inv,
            scenarios,
            checks,
            settings,
            fc,
            rebuild,
            out,
            no_build,
            acc,
            session,
            backend=backend,
        )
    from spel.proptest import domains as dmod

    cfg = {**spec.settings, **(settings or {})}
    seed = int(cfg.get("seed") or 0)
    budget = int(cfg.get("shrink", 40))
    settings = {k: v for k, v in (settings or {}).items() if k != "shrink"}
    spec = dataclasses.replace(
        spec, settings={k: v for k, v in spec.settings.items() if k != "shrink"}
    )
    draw = spec.domain.draw
    kw = dict(fc=fc, no_build=no_build, acc=acc, backend=backend)
    outcomes: list[Outcome] = []
    for k, cells in enumerate(draw.domains(seed)):
        tag = f"domain {k + 1}/{draw.count} (seed {seed})"
        print(f"== {tag}: {_domain_summary(cells)}", file=out, flush=True)
        # with shrinking, inputs are shrunk only on the final domain (_shrink's full run): not here
        first = {**settings, "phases": _NO_INPUT_SHRINK()} if budget else settings
        res = _run_domain(
            _drawn(spec, cells),
            inv,
            scenarios,
            checks,
            first,
            rebuild=rebuild and k == 0,
            out=out,
            tag=(tag, f"__d{k + 1}"),
            **kw,
        )
        for o in res:
            o.domain = tag
        failed = [o for o in res if not o.passed and o.failure]
        if failed and budget:
            _shrink(
                spec, inv, cells, failed, settings, budget, tag, f"__d{k + 1}", out, kw
            )
        outcomes += res
    return outcomes


def _shrink(
    spec: Spec,
    inv: Inventory,
    cells,
    failed: list[Outcome],
    settings: dict,
    budget: int,
    tag: str,
    suffix: str,
    out,
    kw: dict,
) -> None:
    """Shrink a drawn domain on which `failed` checks failed (their outcomes are updated).

    First the domain is shrunk for all of them together: a candidate is kept when every one
    still fails on it, and fails the same way (its FailureSignature: aborted, or which terms are
    non-finite), so the shrinking does not slip to another bug that trips the same check; a
    candidate whose quick run happens to fail differently is passed over, which may leave the
    domain less reduced (one session per candidate runs them all). Then each check alone, from
    there, on what is left of the budget (different bugs may need different domains). Candidate
    runs are quick: at most 2 examples and no input shrinking, so a failure that needs rare inputs
    may stop the shrinking early (the domain is then only less reduced). Each check then runs once more, fully, on its final
    domain, which saves its failure with shrunk inputs; when that failure's signature differs from
    the drawn domain's (input shrinking slipped), the outcome's message says so."""
    from spel.proptest import domains as dmod

    quick = {
        **settings,
        "max_examples": min(
            2, int({**spec.settings, **settings}.get("max_examples", 100))
        ),
        "phases": _NO_INPUT_SHRINK(),
    }
    tried = 0
    t0 = time.monotonic()

    want = {(o.scenario, o.check): o.signature for o in failed}

    def same(o: Outcome) -> bool:
        w = want.get((o.scenario, o.check))
        return w is None or o.signature == w

    def fails(cand, keys: list[tuple[str, str]]) -> set[tuple[str, str]]:
        nonlocal tried
        tried += 1
        r = _run_domain(
            _drawn(spec, cand),
            inv,
            sorted({sc for sc, _ in keys}),
            sorted({c for _, c in keys}),
            quick,
            out=io.StringIO(),
            tag=(tag + ", shrinking", suffix),
            **kw,
        )
        return {
            (o.scenario, o.check) for o in r if not o.passed and o.failure and same(o)
        } & set(keys)

    def descend(start, keys):
        best = start
        while tried < budget:
            for cand in dmod.shrink_candidates(best):
                if tried >= budget:
                    return best
                if fails(cand, keys) == set(keys):
                    best = cand
                    break
            else:
                return best
        return best

    keys = [(o.scenario, o.check) for o in failed]
    joint = descend(cells, keys)
    final = {key: descend(joint, [key]) if len(keys) > 1 else joint for key in keys}
    # the full run (input shrinking, the saved failure) on each final domain, checks sharing one together
    groups: dict[int, tuple[Any, list]] = {}
    for key, dom in final.items():
        groups.setdefault(id(dom), (dom, []))[1].append(key)
    by_key = {(o.scenario, o.check): o for o in failed}
    searching = time.monotonic() - t0
    print(
        f"  shrinking: {tried} candidate domains in {searching:.0f} s"
        f" ({searching / max(tried, 1):.1f} s each)",
        file=out,
        flush=True,
    )
    for dom, ks in groups.values():
        t1 = (
            time.monotonic()
        )  # (on the drawn domain itself too, when nothing smaller fails: shrunk inputs)
        r = _run_domain(
            _drawn(spec, dom),
            inv,
            sorted({sc for sc, _ in ks}),
            sorted({c for _, c in ks}),
            settings,
            out=io.StringIO(),
            tag=(tag + ", shrunk", suffix),
            **kw,
        )
        notes = []
        for o in r:
            orig = by_key.get((o.scenario, o.check))
            if orig is None or o.passed or not o.failure:
                continue
            w = want.get((o.scenario, o.check))
            orig.message, orig.failure = o.message, o.failure
            if w is not None and o.signature is not None and o.signature != w:
                orig.message += (
                    f"\n  note: the drawn domain failed differently ({w.text()}); "
                    f"this shrunk example: {o.signature.text()}"
                )
                notes.append(
                    f"  note: {o.check!r} failed differently after shrinking: {w.text()} -> "
                    f"{o.signature.text()}"
                )
            orig.signature = o.signature
            if dom is not cells:
                orig.domain = f"{tag}, shrunk from {dmod.size(cells)} to {dmod.size(dom)} elements"
        print(
            f"  shrunk ({tried} domains tried) for {', '.join(repr(c) for _, c in ks)}: {_domain_summary(dom)}"
            f" (full run there: {time.monotonic() - t1:.0f} s)",
            file=out,
            flush=True,
        )
        for line in notes:
            print(line, file=out, flush=True)
        for _, c in ks:
            o = next((x for x in failed if x.check == c), None)
            if o is not None and o.failure:
                print(f"    saved: {o.failure}", file=out, flush=True)


def _NO_INPUT_SHRINK() -> tuple:
    """Hypothesis phases without input shrinking (saved failures replayed first)."""
    from hypothesis import Phase

    return (Phase.explicit, Phase.reuse, Phase.generate)


def _drawn(spec: Spec, cells) -> Spec:
    return dataclasses.replace(spec, domain=spec.domain.with_gridcells(cells))


def _domain_summary(cells) -> str:
    def topo(t):
        return "+".join(
            lu.label + ("(0)" if lu.weight == 0.0 else "") for lu in t.landunits
        ) + (" [w0]" if t.weight == 0.0 else "")

    return "; ".join(
        f"{g.name}{' x' + str(g.copies) if g.copies > 1 else ''}"
        f"{' glcmask' if g.glcmask else ''}: "
        + " | ".join(topo(t) for t in g.topounits)
        for g in cells
    )


def _run_domain(
    spec: Spec,
    inv: Inventory,
    scenarios: Optional[list[str]] = None,
    checks: Optional[list[str]] = None,
    settings: Optional[dict] = None,
    fc: Optional[str] = None,
    rebuild: bool = False,
    out=sys.stdout,
    no_build: bool = False,
    acc: Optional[str] = None,
    session: Optional[Session] = None,
    tag: Optional[tuple[str, str]] = None,
    backend: Any = None,
) -> list[Outcome]:
    """run() on the spec's one domain. tag: (description, failure-directory suffix) of a drawn domain."""
    outcomes = []
    root = session or Session(
        spec, inv, fc=fc, rebuild=rebuild, no_build=no_build, acc=acc, backend=backend
    )
    try:
        for name, scen in spec.scenarios.items():
            if scenarios and not _selected(name, scenarios):
                continue
            selected = [
                c for c in root.checks(scen) if not checks or _selected(c.name, checks)
            ]
            if not selected:
                continue
            s = root.for_scenario(
                scen
            )  # the scenario's namelist options: its own session
            if tag is not None:
                s.drawn, s.failure_suffix = tag
            s.preflight(scen, selected)
            for c in selected:
                print(f"[{name}] {c.table} {c.name!r} ...", file=out, flush=True)
                s.last_failure = None
                start = time.monotonic()
                try:
                    s.hypothesis_test(scen, c, settings)()
                except PropertyFailure as exc:
                    saved = s.save_failure()
                    outcomes.append(
                        Outcome(
                            name,
                            c.name,
                            False,
                            str(exc),
                            c.table,
                            time.monotonic() - start,
                            str(saved or ""),
                            signature=getattr(s.last_failure, "signature", None),
                        )
                    )
                    print(f"  FAIL\n{_indent(str(exc))}", file=out, flush=True)
                    if saved:
                        print(
                            f"  saved: {saved}  (spel proptest replay {saved})",
                            file=out,
                            flush=True,
                        )
                    continue
                except SpecError as exc:
                    outcomes.append(
                        Outcome(
                            name,
                            c.name,
                            False,
                            str(exc),
                            c.table,
                            time.monotonic() - start,
                        )
                    )
                    print(f"  FAIL\n{_indent(str(exc))}", file=out, flush=True)
                    continue
                except Exception as exc:  # Hypothesis' own errors (e.g. Unsatisfiable)
                    outcomes.append(
                        Outcome(
                            name,
                            c.name,
                            False,
                            f"{type(exc).__name__}: {exc}",
                            c.table,
                            time.monotonic() - start,
                        )
                    )
                    print(f"  ERROR {type(exc).__name__}: {exc}", file=out, flush=True)
                    continue
                outcomes.append(
                    Outcome(name, c.name, True, "", c.table, time.monotonic() - start)
                )
                print("  ok", file=out, flush=True)
    finally:
        if session is None:
            root.close()
    return outcomes


def replay(
    directory: Path,
    fc: Optional[str] = None,
    acc: Optional[str] = None,
    no_build: bool = False,
    out=sys.stdout,
) -> Outcome:
    """Run a saved failing example again (no Hypothesis): the same report when it still fails."""
    from spel.proptest import spec as specmod

    directory = Path(directory)
    try:
        meta = json.loads((directory / EXAMPLE_FILE).read_text())
    except OSError as exc:
        raise SpecError(f"{directory / EXAMPLE_FILE}: {exc.strerror}") from None
    sp, inv = specmod.load(Path(meta["spec"]))
    if (directory / DOMAIN_FILE).is_file():  # a failure on a drawn domain: that domain
        import tomllib

        from spel.proptest.hierarchy import Gridcell

        table = tomllib.loads((directory / DOMAIN_FILE).read_text())
        cells = [
            Gridcell.from_table(g, f"{DOMAIN_FILE} gridcell {k}")
            for k, g in enumerate(table["gridcell"])
        ]
        sp = _drawn(sp, cells)
    scen = sp.scenarios.get(meta["scenario"])
    check = next((c for c in sp.checks if c.name == meta["check"]), None)
    if scen is None or check is None:
        raise SpecError(
            f"{directory}: {meta['spec']} has no scenario {meta['scenario']!r} with check {meta['check']!r} any more"
        )
    start = time.monotonic()
    with Session(sp, inv, fc=fc, acc=acc, no_build=no_build) as root:
        s = root.for_scenario(scen)
        _, ex = s.load_failure(directory)
        try:
            s.judge(scen, check, ex, hypothesis=False)
        except PropertyFailure as exc:
            print(
                f"[{scen.name}] {check.table} {check.name!r}: FAIL\n{_indent(str(exc))}",
                file=out,
            )
            return Outcome(
                scen.name,
                check.name,
                False,
                str(exc),
                check.table,
                time.monotonic() - start,
                str(directory),
            )
    print(
        f"[{scen.name}] {check.table} {check.name!r}: ok (the saved example passes now)",
        file=out,
    )
    return Outcome(
        scen.name,
        check.name,
        True,
        "",
        check.table,
        time.monotonic() - start,
        str(directory),
    )


def write_json(outcomes: list[Outcome], spec: Spec, settings: dict, path: Path) -> None:
    data = {
        "spec": str(spec.path),
        "case": spec.case_dir.name,
        "settings": {**spec.settings, **settings},
        "passed": sum(o.passed for o in outcomes),
        "failed": sum(not o.passed for o in outcomes),
        "outcomes": [dataclasses.asdict(o) for o in outcomes],
    }
    Path(path).write_text(json.dumps(data, indent=1))


def write_junit(outcomes: list[Outcome], spec: Spec, path: Path) -> None:
    """A JUnit XML report (one testcase per check, classname = the scenario), for CI."""
    from xml.etree import ElementTree as ET

    suite = ET.Element(
        "testsuite",
        name=f"spel proptest {spec.path.name}",
        tests=str(len(outcomes)),
        failures=str(sum(not o.passed for o in outcomes)),
        time=f"{sum(o.seconds for o in outcomes):.3f}",
    )
    for o in outcomes:
        case = ET.SubElement(
            suite,
            "testcase",
            classname=f"{spec.case_dir.name}.{o.scenario}",
            name=o.check,
            time=f"{o.seconds:.3f}",
        )
        if not o.passed:
            ET.SubElement(
                case,
                "failure",
                message=o.message.splitlines()[0] if o.message else "failed",
            ).text = o.message
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def _selected(name: str, patterns: list[str]) -> bool:
    """An exact name or a glob (names may contain `[`, e.g. "btran in [0, 1]")."""
    import fnmatch

    return name in patterns or any(fnmatch.fnmatchcase(name, p) for p in patterns)
