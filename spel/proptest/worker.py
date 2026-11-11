"""A subprocess that owns an initialized case.

Fortran `stop`/`error stop`, `endrun` and signals (SIGSEGV, SIGFPE with trapping on) end the
process; the parent sees that as `FortranAbort` with the tail of the Fortran output, and the next
request starts a fresh worker. Hypothesis treats the abort like any other failure and shrinks it.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from spel.proptest.fexpr import FArray


class FortranAbort(Exception):
    """The Fortran code ended the worker process."""


def _serve(conn, lib: str, manifest: str, path_arg: str, log: str, direct: Optional[dict] = None) -> None:
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.environ["GFORTRAN_UNBUFFERED_ALL"] = "1"  # keep the output up to a crash
    from spel.proptest.runtime import Case

    case = Case(Path(lib), Path(manifest))
    case.init(path_arg, direct)
    conn.send(("ready", None))
    while True:
        try:
            req = conn.recv()
        except EOFError:
            return
        op = req.get("op")
        try:
            if op == "stop":
                return
            if op == "get":
                conn.send(("ok", {n: _pack(case.get(n)) for n in req["names"]}))
            elif op == "run":
                case.trap(bool(req.get("trap", False)))
                case.reset(poison_outputs=req.get("poison", True))
                for name, value in req["inputs"].items():
                    case.set(name, value)
                case.compute()
                case.trap(False)
                conn.send(("ok", {n: _pack(case.get(n)) for n in req["outputs"]}))
            else:
                conn.send(("error", f"unknown op {op!r}"))
        except Exception as exc:  # Python-side errors (bad names, shapes): report, keep serving
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


def _pack(fa: Optional[FArray]):
    return None if fa is None else (fa.data, fa.lb)


class Worker:
    def __init__(self, lib: Path, manifest: Path, path_arg: str, timeout: float = 600.0,
                 log_dir: Optional[Path] = None, env: Optional[dict[str, str]] = None,
                 direct: Optional[dict] = None):
        """log_dir: where the Fortran output goes (default: the system's temporary directory).
        The log is removed when the worker stops, unless the Fortran code ended a process.
        env: the environment the process starts with (default: this one's); the dynamic loader
        reads `LD_LIBRARY_PATH` when the process starts, so it cannot be set later.
        direct: initialize without an inputs file (domain.Domain.direct, runtime.Case.init)."""
        self.lib, self.manifest, self.path_arg = str(lib), str(manifest), path_arg
        self.env = env
        self.direct = direct
        self.timeout = timeout
        self.proc: Optional[mp.Process] = None
        self.conn = None
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="worker-", suffix=".log", dir=log_dir)
        os.close(fd)
        self.log = Path(name)
        self.starts = 0
        self.aborts = 0

    # -- process management
    def start(self) -> None:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(child, self.lib, self.manifest, self.path_arg, str(self.log),
                                                     self.direct), daemon=True)
        saved = dict(os.environ) if self.env is not None else None
        if saved is not None:  # a spawned process starts with the parent's environment of this moment
            os.environ.clear()
            os.environ.update(self.env)
        try:
            self.proc.start()
        finally:
            if saved is not None:
                os.environ.clear()
                os.environ.update(saved)
        child.close()
        self.conn = parent
        self.starts += 1
        self._receive("initialization")

    def stop(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            try:
                self.conn.send({"op": "stop"})
                self.proc.join(5)
            except (BrokenPipeError, OSError):
                pass
            if self.proc.is_alive():
                self.proc.kill()
        self.proc = None
        self.conn = None
        if not self.aborts:
            self.log.unlink(missing_ok=True)

    def __enter__(self) -> "Worker":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _ensure(self) -> None:
        if self.proc is None or not self.proc.is_alive():
            self.start()

    def _receive(self, what: str) -> Any:
        try:
            if not self.conn.poll(self.timeout):
                self.proc.kill()
                self.proc.join(5)
                self.proc = None
                self.aborts += 1
                raise FortranAbort(f"{what}: no answer within {self.timeout} s\n{self.output_tail()}")
            status, payload = self.conn.recv()
        except (EOFError, ConnectionResetError, BrokenPipeError):
            self.proc.join(5)
            code = self.proc.exitcode
            self.proc = None
            self.aborts += 1
            raise FortranAbort(f"{what}: the Fortran process ended (exit code {code})\n{self.output_tail()}") from None
        if status == "error":
            raise RuntimeError(payload)
        return payload

    def output_tail(self, n: int = 30) -> str:
        try:
            lines = self.log.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])

    # -- requests
    def get(self, names: list[str]) -> dict[str, Optional[FArray]]:
        self._ensure()
        self.conn.send({"op": "get", "names": list(names)})
        return _unpack(self._receive("get"))

    def run(self, inputs: dict[str, np.ndarray], outputs: list[str], trap: bool = False,
            poison: bool = True) -> dict[str, Optional[FArray]]:
        self._ensure()
        try:
            self.conn.send({"op": "run", "inputs": inputs, "outputs": list(outputs), "trap": trap, "poison": poison})
        except (BrokenPipeError, OSError):
            self.proc = None
            self._ensure()
            self.conn.send({"op": "run", "inputs": inputs, "outputs": list(outputs), "trap": trap, "poison": poison})
        return _unpack(self._receive("compute"))


def _unpack(payload: dict) -> dict[str, Optional[FArray]]:
    return {n: (None if v is None else FArray(v[0], tuple(v[1]), n)) for n, v in payload.items()}
