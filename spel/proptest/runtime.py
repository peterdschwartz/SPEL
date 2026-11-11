"""ctypes access to a case built as `libspelcase.so` (see shim.py for the exported routines).

Fortran module state lives once per process, so one process hosts one initialized case; the
worker (worker.py) runs a Case in a subprocess so that an `endrun`/`stop`/signal in Fortran
becomes a test failure instead of killing the test runner.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from spel.proptest.errors import ProptestError
from spel.proptest.fexpr import FArray

# the values every element of a new allocation gets (as the domain files hold them: domain.py)
FILL = {"real": 1.0e36, "integer": -9999, "logical": 0}
C_TYPES = {"real": (ctypes.c_double, np.float64), "integer": (ctypes.c_int32, np.int32),
           "logical": (ctypes.c_int32, np.int32)}


class CaseError(ProptestError):
    pass


class Case:
    def __init__(self, lib_path: Path, manifest_path: Path):
        self.lib = ctypes.CDLL(str(lib_path))
        self.manifest = json.loads(Path(manifest_path).read_text())
        self.entries = {e["name"]: e for e in self.manifest["entries"]}
        self.lib.spel_py_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.spel_py_compute.argtypes = []
        self.lib.spel_py_trap.argtypes = [ctypes.c_int]
        self.lib.spel_py_true.restype = ctypes.c_int
        self.lib.spel_py_view.restype = ctypes.c_int
        self.lib.spel_py_view.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                                          ctypes.POINTER(ctypes.c_int * 7), ctypes.POINTER(ctypes.c_int * 7)]
        self.true = int(self.lib.spel_py_true())
        self._views: dict[str, Optional[FArray]] = {}
        self._initial: dict[str, np.ndarray] = {}
        self.initialized = False

    # -- lifecycle
    def init(self, path_arg: str, direct: Optional[dict] = None) -> None:
        """Initialize from the domain files under `path_arg` (elm_init), or with `direct`
        (domain.Domain.direct) from the constants file there and the given bounds and values."""
        if self.initialized:
            raise CaseError("a case initializes once per process")
        b = path_arg.encode()
        if direct is None:
            self.lib.spel_py_init(b, len(b))
        else:
            self._init_direct(b, direct)
        self.initialized = True
        self._views.clear()
        # the state elm_init left (the domain files): reset() returns to it
        self._initial = {n: v.data.copy() for n in self.entries if (v := self.view(n)) is not None
                         and self.entries[n]["kind"] not in ("constant",)}

    def _init_direct(self, path: bytes, direct: dict) -> None:
        """spel_py_begin (constants), a descriptor per variable to allocate, spel_py_allocate,
        the values through views, then spel_py_setup (filters, clumps, time sync)."""
        if not self.manifest.get("direct"):
            raise CaseError(f"case {self.manifest['case']} predates direct init (no spel_varMod): "
                            "recreate it with `spel create`, or use [domain] init = \"files\"")
        lib = self.lib
        lib.spel_py_set_bounds.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                           ctypes.c_int, ctypes.c_double, ctypes.c_int]
        lib.spel_py_begin(path, len(path))
        for name, (ext, ftype) in direct["alloc"].items():
            e = self.entries.get(name)
            if e is None or not e.get("alloc"):
                raise CaseError(f"{name}: the shim does not allocate it (rebuild the library)")
            rank = len(ext)
            lb = (ctypes.c_int * max(rank, 1))(*[lo for lo, _ in ext])
            ub = (ctypes.c_int * max(rank, 1))(*[hi for _, hi in ext])
            fill_r8, fill_int = (FILL[ftype], 0) if ftype == "real" else (0.0, int(FILL[ftype]))
            bname = name.encode()
            lib.spel_py_set_bounds(e["id"], bname, len(bname), rank, lb, ub, 1, fill_r8, fill_int)
        lib.spel_py_allocate()
        for name, value in direct["values"].items():
            self.set(name, value)
        self._views.clear()  # spel_py_setup allocates the filters
        sizes = direct["sizes"]
        lib.spel_py_setup(*(int(sizes[lev]) for lev in ("g", "t", "l", "c", "p")))

    def compute(self) -> None:
        self.lib.spel_py_compute()

    def trap(self, on: bool) -> None:
        self.lib.spel_py_trap(1 if on else 0)

    def reset(self, poison_outputs: bool = True) -> None:
        """Every variable back to its state after init; written-only real outputs set to NaN."""
        for name, data in self._initial.items():
            view = self.view(name)
            e = self.entries[name]
            if poison_outputs and e["kind"] == "output" and e["type"] == "real":
                view.data[...] = np.nan
            else:
                view.data[...] = data

    # -- data
    def view(self, name: str) -> Optional[FArray]:
        """The variable's memory as an FArray (Fortran order, Fortran lower bounds); None if unallocated."""
        if name in self._views:
            return self._views[name]
        e = self.entries.get(name)
        if e is None:
            raise CaseError(f"{name} is not exposed by the shim")
        addr = ctypes.c_void_p()
        lb, ub = (ctypes.c_int * 7)(), (ctypes.c_int * 7)()
        rank = self.lib.spel_py_view(e["id"], ctypes.byref(addr), ctypes.byref(lb), ctypes.byref(ub))
        if rank == -2:
            raise CaseError(f"{name}: unknown id {e['id']} (rebuild the library)")
        if rank < 0 or not addr.value:
            self._views[name] = None
            return None
        ctype, dtype = C_TYPES[e["type"]]
        lbs = tuple(lb[k] for k in range(rank))
        shape = tuple(ub[k] - lb[k] + 1 for k in range(rank))
        n = int(np.prod(shape)) if shape else 1
        buf = (ctype * n).from_address(addr.value)
        arr = np.ctypeslib.as_array(buf).view(dtype)
        arr = arr.reshape(shape, order="F") if shape else arr.reshape(())
        fa = FArray(arr, lbs, name)
        self._views[name] = fa
        return fa

    def get(self, name: str) -> Any:
        """A copy as numpy (logicals as bool)."""
        v = self.view(name)
        if v is None:
            return None
        data = v.data.copy()
        if self.entries[name]["type"] == "logical":
            data = data != 0
        return FArray(data, v.lb, name)

    def set(self, name: str, value: Any) -> None:
        v = self.view(name)
        if v is None:
            raise CaseError(f"{name} is not allocated")
        val = np.asarray(value)
        if self.entries[name]["type"] == "logical":
            val = np.where(val.astype(bool), self.true, 0).astype(np.int32)
        v.data[...] = val
