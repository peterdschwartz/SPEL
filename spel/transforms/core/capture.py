"""
Capturing the files a legacy pass writes, so that its output becomes a Plan (`adapters.py`) or
a golden snapshot (`tools/golden.py`) instead of edits to the checkout.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import os
import pathlib


class VirtualFS:
    """Captures writes through Path.write_text/write_bytes and open(..., 'w'/'a'); later reads
    of a captured path see the captured text."""

    def __init__(self):
        self.files: dict[str, str] = {}

    @staticmethod
    def key(p) -> str:
        return os.path.abspath(os.fspath(p))

    @contextlib.contextmanager
    def active(self):
        P = pathlib.Path
        saved = (P.write_text, P.write_bytes, P.read_text, P.exists, P.is_file, builtins.open, P.mkdir)
        vfs = self
        real_open, real_exists, real_is_file, real_read = builtins.open, P.exists, P.is_file, P.read_text

        def write_text(self, data, encoding=None, errors=None, newline=None):
            vfs.files[vfs.key(self)] = data
            return len(data)

        def write_bytes(self, data):
            vfs.files[vfs.key(self)] = data.decode(errors="replace")
            return len(data)

        def read_text(self, encoding=None, errors=None, *a, **k):
            k_ = vfs.key(self)
            if k_ in vfs.files:
                return vfs.files[k_]
            return real_read(self, encoding=encoding, errors=errors)

        def exists(self, *a, **k):
            return vfs.key(self) in vfs.files or real_exists(self, *a, **k)

        def is_file(self, *a, **k):
            return vfs.key(self) in vfs.files or real_is_file(self, *a, **k)

        def mkdir(self, *a, **k):
            return None

        class _Capture(io.StringIO):
            def __init__(self, path, initial=""):
                super().__init__(initial)
                self.seek(0, io.SEEK_END)
                self._path = path

            def close(self):
                vfs.files[self._path] = self.getvalue()
                super().close()

        def open_(file, mode="r", *a, **k):
            if isinstance(file, int):
                return real_open(file, mode, *a, **k)
            path = vfs.key(file)
            if any(c in mode for c in "wax+") and "b" not in mode and not path.startswith("/dev/"):
                initial = vfs.files.get(path, "") if "a" in mode else ""
                return _Capture(path, initial)
            if "b" not in mode and path in vfs.files:
                return io.StringIO(vfs.files[path])
            return real_open(file, mode, *a, **k)

        P.write_text, P.write_bytes, P.read_text = write_text, write_bytes, read_text
        P.exists, P.is_file, builtins.open, P.mkdir = exists, is_file, open_, mkdir
        try:
            yield self
        finally:
            (P.write_text, P.write_bytes, P.read_text, P.exists, P.is_file, builtins.open, P.mkdir) = saved
