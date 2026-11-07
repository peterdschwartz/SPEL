"""
Snippet harness for transforms (DESIGN.md §3): run a transform on a few lines of Fortran,
without an ELM analysis.

    from spel.transforms.core.testing import check_transform

    def test_hoist(tmp_path):
        check_transform(CollapseBounds, BEFORE, AFTER, tmp_path)

`snippet_context` writes the snippet to a file under `tmp_path` and analyzes it as `spel
analyze` would analyze a source, as far as one file allows: logical lines from the
preprocessor, a parse of the whole file, a scope walk of every routine (internal ones with
their host's record) and the access maps (`record_access`). Names a routine does not declare
resolve to module globals (the snippet's own routines and `procedures` to procedures), so a
snippet needs no stub modules. `check_transform` runs the transform in memory, compares the
result with the expected text, re-parses it and plans again on it (a second run must change
nothing).

Line numbers are 0-based indices into the snippet, as everywhere: a leading newline makes the
first statement line 1.
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

from spel.scripts.fortran_parser import spel_ast as A
from spel.scripts.fortran_parser.symbols import DictResolver, Origin, Symbol, SymbolKind, apply_use
from spel.transforms.core import driver
from spel.transforms.core.transform import Context, Transform

log = logging.getLogger("spel.transforms.testing")


class SnippetResolver(DictResolver):
    """Unknown names are module globals; `use`d modules export whatever the routine asks for."""

    def __init__(self, procedures: Iterable[str] = (), globals: Iterable[str] = (),
                 targets: Optional[dict[str, str]] = None):
        # targets: routine ids of the snippet's own module procedures, so calls between them resolve
        targets = {k.lower(): v for k, v in (targets or {}).items()}
        syms = {n.lower(): Symbol(n.lower(), SymbolKind.VARIABLE, Origin.GLOBAL) for n in globals}
        syms.update({n.lower(): Symbol(n.lower(), SymbolKind.PROCEDURE, Origin.GLOBAL, targets.get(n.lower()))
                     for n in procedures})
        super().__init__(globals=syms)

    def fallback(self, name: str) -> Optional[Symbol]:
        return Symbol(name, SymbolKind.VARIABLE, Origin.GLOBAL)

    def use_module(self, stmt) -> dict[str, Symbol]:
        def missing(remote: str) -> Symbol:
            return self.globals.get(remote) or Symbol(remote, SymbolKind.VARIABLE, Origin.GLOBAL)

        return apply_use(stmt, self.modules.get(stmt.module, {}), missing)


class SnippetSub:
    """The parts of `Subroutine` that transforms read, for a routine of a snippet."""

    library = False

    def __init__(self, tree, module: str, path: Path, logical: list, host: Optional["SnippetSub"]):
        self.syntax_tree = tree
        self.name = tree.name
        self.module = module
        self.id = f"{module}::{tree.name}"
        self.filepath = path
        self.func = tree.is_function
        self.startline = tree.lineno
        self.end_stmt_ln = tree.end_ln
        self.contains_ln = tree.contains_ln if tree.contains_ln >= 0 else None
        self.endline = self.contains_ln if self.contains_ln is not None else self.end_stmt_ln
        self.is_internal = host is not None
        self.host = host
        self.host_id = host.id if host else None
        self.internal_subs: dict[str, SnippetSub] = {}
        self.sub_lines = [lt for lt in logical if self.startline <= lt.ln <= self.endline]
        self.record = None
        self.record_access = None
        self.registrations: list = []
        self.sub_call_desc: dict = {}
        self.child_subroutines: dict = {}
        self.flat_ifs: list = []
        self.elmtype_access_by_ln: dict = {}
        self.active_global_vars: dict = {}
        self.associate_vars: dict = {}
        self.associate_start = self.associate_end = -1

    @property
    def last_decl_ln(self) -> int:
        return self.record.last_decl_ln if self.record else -1

    @property
    def dummy_args_list(self) -> list[str]:
        return list(self.syntax_tree.args)

    def walk_syntax_tree(self, scopes=None):
        return self.record

    def __repr__(self) -> str:
        return f"SnippetSub({self.id})"


def analyze_snippet(path: Path, procedures: Iterable[str] = (), globals: Iterable[str] = (),
                    access: bool = True) -> dict[str, SnippetSub]:
    """id -> SnippetSub of every routine of the Fortran file `path` (modules and bare routines)."""
    from spel.scripts.fortran_parser.scope_walk import walk_subroutine
    from spel.scripts.fortran_parser.spel_parser import Parser
    from spel.transforms.core.pipeline import preprocessed_lines

    path = Path(path).resolve()
    n = len(path.read_text().splitlines())
    logical = preprocessed_lines(path, n)
    prog = Parser(lines=logical, logger="Parser-snippet").parse_program()

    tops: list[tuple[object, str]] = []
    for st in prog.statements:
        if isinstance(st, A.ModuleConstruct):
            tops += [(sp, st.name) for sp in st.subprograms]
        elif isinstance(st, A.SubroutineDefinitionConstruct):
            tops.append((st, path.stem.lower()))

    def names(tree):
        yield tree.name
        for c in tree.contains:
            yield from names(c)

    targets = {t.name.lower(): f"{m}::{t.name.lower()}" for t, m in tops}
    resolver = SnippetResolver([*procedures, *(x for t, _ in tops for x in names(t))], globals, targets)
    subs: dict[str, SnippetSub] = {}

    def add(tree, module: str, host: Optional[SnippetSub]):
        sub = SnippetSub(tree, module, path, logical, host)
        sub.record = walk_subroutine(tree, resolver=resolver, host=host.record if host else None)
        subs[sub.id] = sub
        if host is not None:
            host.internal_subs[sub.name] = sub
        for c in tree.contains:
            add(c, module, sub)

    for tree, module in tops:
        add(tree, module, None)

    if access:
        from spel.scripts.record_access import AccessMapper

        mapper = AccessMapper(subs)
        for sub in subs.values():
            sub.record_access = mapper.maps(sub)
    return subs


def snippet_context(src: str, tmp_path: Path, name: str = "snippet_mod.F90", **analysis) -> Context:
    """A Context over the snippet `src`, written to `tmp_path/name` (keyword arguments: see
    `analyze_snippet`). `ctx.fut.subroutine_dict` holds its routines."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = (tmp_path / name).resolve()
    path.write_text(src)
    subs = analyze_snippet(path, **analysis)
    fut = SimpleNamespace(subroutine_dict=subs, module_dict={}, type_dict={}, snippet=path,
                          snippet_analysis=analysis)
    return Context(fut, tmp_path.resolve())


class SnippetOverlay:
    """The pipeline overlay of a snippet context: the edited snippet is analyzed again whole."""

    def __init__(self, fut, srcroot: Path):
        self.fut = SimpleNamespace(**vars(fut))

    def update(self, old_texts: dict, new_texts: dict, step: str) -> None:
        from spel.transforms.core.pipeline import PipelineError

        path = self.fut.snippet
        if set(map(Path, new_texts)) - {path}:
            raise PipelineError(f"{step} writes files other than the snippet")
        try:
            self.fut.subroutine_dict = analyze_snippet(path, **self.fut.snippet_analysis)
        except BaseException as err:  # noqa: BLE001 (the parser exits on errors)
            raise PipelineError(f"after {step}: the snippet no longer parses or walks: {err}") from None


def _diff(want: str, got: str) -> str:
    return "".join(difflib.unified_diff(want.splitlines(True), got.splitlines(True), "expected", "got"))


def check_transform(t, before: str, after: str, tmp_path: Path, opts=None, name: str = "snippet_mod.F90",
                    idempotent: bool = True, **analysis):
    """Run transform `t` (class or instance) on the snippet `before` and assert that it gives
    `after`, that the result parses, and (`idempotent`) that planning again on it changes
    nothing. Returns the driver Result."""
    t = t() if isinstance(t, type) and issubclass(t, Transform) else t
    opts = opts if opts is not None else t.Options()
    ctx = snippet_context(before, tmp_path, name, **analysis)
    path = ctx.fut.snippet
    res = driver.run(t, ctx, opts, sink=driver.Memory(), verify=True)
    got = res.outputs.get(path, before)
    assert got == after, f"{t.name} gave:\n{_diff(after, got)}\n{res.plan.report()}"
    assert not res.problems, res.problems
    if idempotent and got != before:
        # a fresh directory: the parse caches key on path and mtime
        again = snippet_context(got, Path(tmp_path) / f"again-{t.name}", name, **analysis)
        res2 = driver.run(t, again, opts, sink=driver.Memory(), dry_run=True)
        out2 = res2.outputs.get(again.fut.snippet, got)
        assert out2 == got, f"{t.name} is not idempotent:\n{_diff(got, out2)}"
    return res


def check_pipeline(steps, before: str, after: str, tmp_path: Path, name: str = "snippet_mod.F90", **analysis):
    """`check_transform` for a `Pipeline` of `steps` ((transform, opts) pairs or transforms):
    every step plans on the previous one's output. Returns the PipelineResult."""
    from spel.transforms.core.pipeline import Pipeline

    ctx = snippet_context(before, tmp_path, name, **analysis)
    path = ctx.fut.snippet
    res = Pipeline(steps).run(ctx.fut, ctx.srcroot, sink=driver.Memory(), verify=True, overlay=SnippetOverlay)
    got = res.outputs.get(path, before)
    report = "\n".join(s.result.plan.report() for s in res.steps)
    assert got == after, f"the pipeline gave:\n{_diff(after, got)}\n{report}"
    assert not res.problems, res.problems
    return res
