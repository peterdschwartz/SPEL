"""ELM's plant functional types by name: `pftvarcon::pftconrd`'s `expected_pftnames(i) = '...'`,
the names (and order) ELM requires of the parameter file. Read with SPEL's parser from a
srcroot's `pftvarcon.F90` once and cached per srcroot (`.spel-cache/<srcroot>/pft_names.json`,
keyed by the file's hash, so an edited source is read again).

`pft_enum(srcroot)` is an IntEnum of them, so `Soil(Pft.C3_NON_ARCTIC_GRASS) == Soil(13)`:
`Pft` (from `spel.proptest.hierarchy`) for the configured srcroot (`SPEL_E3SM_SRCROOT`),
`api.Case.pfts` for the srcroot a case was made from. `Pft.X.elm_name` is ELM's name."""

from __future__ import annotations

import hashlib
import json
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from spel.proptest.hierarchy.codes import HierarchyError

PFTVARCON = Path("components/elm/src/main/pftvarcon.F90")
MODULE, ROUTINE, ARRAY = "pftvarcon", "pftconrd", "expected_pftnames"
CACHE_FILE = "pft_names.json"


class PftBase(IntEnum):
    """The base of every `Pft` enum: members are pft indices (pft_pp%itype, pftcon rows)."""

    @property
    def elm_name(self) -> str:
        """The name in pftvarcon (`c3_non-arctic_grass`)."""
        return type(self)._elm_names[self.value]  # type: ignore[attr-defined]


def _srcroot(srcroot: Optional[str | Path]) -> Path:
    if srcroot is None:
        from spel.scripts.config import E3SM_SRCROOT

        srcroot = E3SM_SRCROOT
    return Path(srcroot).expanduser().resolve()


def pft_names(srcroot: Optional[str | Path] = None) -> tuple[str, ...]:
    """ELM's pft names, index 0 (not_vegetated) to mxpft, of a srcroot (default: the configured one)."""
    root = _srcroot(srcroot)
    src = root / PFTVARCON
    if not src.is_file():
        raise HierarchyError(f"no {PFTVARCON} under {root}: pft names come from the E3SM source")
    digest = hashlib.sha1(src.read_bytes()).hexdigest()
    from spel.scripts.analysis_cache import cache_key, cache_root

    cache = cache_root() / cache_key(root) / CACHE_FILE
    try:
        saved = json.loads(cache.read_text())
        if saved.get("sha1") == digest:
            return tuple(saved["names"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    names = _read(src)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"source": str(src), "sha1": digest, "names": list(names)}, indent=1))
    except OSError:
        pass  # a read-only cache: parse again next time
    return names


def _read(src: Path) -> tuple[str, ...]:
    """The `expected_pftnames(<int>) = '<name>'` assignments of pftconrd, from the parse tree."""
    import spel.scripts.fortran_parser.spel_ast as A
    from spel.scripts.fortran_parser.module_spec import ModuleSpecError, parse_modules
    from spel.scripts.registrations import iter_statements

    try:
        mods = parse_modules(src)
    except ModuleSpecError as exc:
        raise HierarchyError(f"pft names: {exc}") from None
    mod = next((m for m in mods if str(m.name).lower() == MODULE), None)
    sub = next((s for s in mod.subprograms if str(s.name).lower() == ROUTINE), None) if mod else None
    if sub is None:
        raise HierarchyError(f"pft names: no {MODULE}::{ROUTINE} in {src}")
    found: dict[int, str] = {}
    for stmt, _ in iter_statements(sub.body):
        node = stmt.expression if isinstance(stmt, A.ExpressionStatement) else stmt
        if isinstance(node, A.AssignmentStatement):
            lhs, rhs = node.left, node.right
        elif isinstance(node, A.InfixExpression) and node.operator == "=":
            lhs, rhs = node.left_expr, node.right_expr
        else:
            continue
        if not (isinstance(lhs, A.FuncExpression) and str(lhs.function).lower() == ARRAY and len(lhs.args) == 1):
            continue
        index, value = lhs.args[0], rhs
        if not (isinstance(index, A.IntegerLiteral) and isinstance(value, A.StringLiteral)):
            raise HierarchyError(f"pft names: {src}:{stmt.lineno + 1}: expected {ARRAY}(<integer>) = '<name>'")
        if index.value in found:
            raise HierarchyError(f"pft names: {src}: {ARRAY}({index.value}) is set twice")
        found[index.value] = value.value.strip()
    if not found or sorted(found) != list(range(len(found))):
        raise HierarchyError(f"pft names: {src}: {ARRAY} is not set for 0..n (have {sorted(found)})")
    return tuple(found[k] for k in range(len(found)))


def member_name(elm_name: str) -> str:
    """`c3_non-arctic_grass` -> `C3_NON_ARCTIC_GRASS`."""
    out = "".join(ch if ch.isalnum() else "_" for ch in elm_name.strip()).upper()
    return out if out[:1].isalpha() else f"PFT_{out}"


@lru_cache(maxsize=None)
def _enum(root: Path) -> type[PftBase]:
    names = pft_names(root)
    members = [member_name(n) for n in names]
    if len(set(members)) != len(members):
        raise HierarchyError(f"pft names of {root} are not distinct as identifiers: {members}")
    cls = PftBase("Pft", [(m, k) for k, m in enumerate(members)])  # type: ignore[call-overload]
    cls._elm_names = names  # type: ignore[attr-defined]
    cls.__doc__ = f"ELM's pfts in {root} (pftvarcon::expected_pftnames)."
    return cls


def pft_enum(srcroot: Optional[str | Path] = None) -> type[PftBase]:
    """The `Pft` IntEnum of a srcroot (default: the configured one), built once per srcroot."""
    return _enum(_srcroot(srcroot))
