"""Pickling of a FunctionalUnitTest (the CSV export lives in spel.scripts.db_export)."""

import pickle
import shutil
from itertools import chain
from typing import Optional
from pathlib import Path

from spel.scripts.analyze_subroutines import Subroutine
from spel.scripts.config import E3SM_SRCROOT, scripts_dir
from spel.scripts.DerivedType import DerivedType
from spel.scripts.fortran_modules import FortranModule
from spel.scripts.functional_unit_test import FunctionalUnitTest


def _fut_objects(fut: FunctionalUnitTest):
    return chain(
        fut.module_dict.values(),
        fut.subroutine_dict.values(),
        fut.type_dict.values(),
    )


CASE_PICKLE = "fut.pkl"


def pickle_unit_test(fut: FunctionalUnitTest) -> None:
    """
    Serialize a FunctionalUnitTest to spel/scripts/fut_<case>.pkl (export,
    database) and <case_dir>/fut.pkl (`spel instrument`, survives other creates).
    """
    output_path = Path(scripts_dir) / f"fut_{fut.case_name}.pkl"
    # Case copy first: a concurrent `spel create` may clear scripts_dir/*.pkl
    if Path(fut.case_dir).is_dir():
        dump_unit_test(fut, Path(fut.case_dir) / CASE_PICKLE)
        shutil.copyfile(Path(fut.case_dir) / CASE_PICKLE, output_path)
    else:
        dump_unit_test(fut, output_path)


def dump_unit_test(fut: FunctionalUnitTest, output_path: Path) -> None:
    """Pickle `fut` with source paths relative to E3SM_SRCROOT (see _load_pickle)."""
    objects = list(_fut_objects(fut))
    original_paths = [obj.filepath for obj in objects]

    try:
        for obj in objects:
            path = Path(obj.filepath)
            # paths outside the source tree (e.g. libraries) stay absolute;
            # _load_pickle's `E3SM_SRCROOT / path` leaves them unchanged.
            if path.is_relative_to(E3SM_SRCROOT):
                path = path.relative_to(E3SM_SRCROOT)
            obj.filepath = path

        with output_path.open("wb") as dbfile:
            pickle.dump(fut, dbfile)
    finally:
        for obj, filepath in zip(objects, original_paths):
            obj.filepath = filepath


def unpickle_unit_test(casename: str) -> FunctionalUnitTest:
    """
    Function to load SPEL's output from pickled files.
    """
    return _load_pickle(Path(scripts_dir) / f"fut_{casename}.pkl")


def case_srcroot(case_dir: Path) -> Optional[Path]:
    """The E3SM checkout a case's spel_meta.json names (`spel create`; `spel archive extract
    --set-srcroot` rewrites it) when it is on this machine; None without a meta file or a
    recorded checkout, or (with a warning) when the checkout is not here."""
    import json
    import warnings

    from spel.scripts.config import CASE_META_FILENAME

    try:
        meta = json.loads((Path(case_dir) / CASE_META_FILENAME).read_text())
    except (OSError, ValueError):
        return None
    recorded = meta.get("e3sm_srcroot") if isinstance(meta, dict) else None
    if not recorded:
        return None
    root = Path(recorded).expanduser()
    if root.is_dir():
        return root.resolve()
    warnings.warn(
        f"{Path(case_dir).name} was made from {recorded}, which is not on this machine: its sources "
        f"are read from {E3SM_SRCROOT} (`spel archive extract --set-srcroot DIR` records another checkout)",
        stacklevel=3,
    )
    return None


def unpickle_case(case_dir: Path) -> FunctionalUnitTest:
    """Load the FunctionalUnitTest `spel create` stored in a case directory. The case's
    spel_meta.json decides its checkout when that is on this machine: source paths are anchored
    there and `e3sm_srcroot` (which selects the analysis cache: step coupling, units) names it.
    Otherwise sources come from the configured E3SM_SRCROOT and `e3sm_srcroot` stays as pickled."""
    root = case_srcroot(case_dir)
    fut = _load_pickle(Path(case_dir) / CASE_PICKLE, srcroot=root)
    if root is not None:
        fut.e3sm_srcroot = root
    return fut


def _load_pickle(input_path: Path, srcroot: Optional[Path] = None) -> FunctionalUnitTest:
    """
    Load a pickle written by dump_unit_test, re-anchoring source paths to
    `srcroot` (default: the configured E3SM_SRCROOT).
    """
    root = Path(srcroot) if srcroot is not None else E3SM_SRCROOT
    with input_path.open("rb") as dbfile:
        fut: FunctionalUnitTest = pickle.load(dbfile)

    for obj in _fut_objects(fut):
        obj.filepath = root / obj.filepath

    return fut
