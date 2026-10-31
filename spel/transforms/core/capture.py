"""
Virtual files: capturing the files a pass writes, so that its output becomes a golden snapshot
(`tools/golden.py`) instead of edits to the checkout, and serving edited text that isn't written
yet to everything that reads sources (a `Pipeline` step plans on its predecessors' output).
"""

from __future__ import annotations

import builtins
import contextlib
import io
import os
import pathlib


class VirtualFS:
    """Captures writes through Path.write_text/write_bytes and open(..., 'w'/'a'); later reads
    of a captured path see the captured text. Reads cover Path.read_text, open(..., 'r') and the
    C preprocessor of the analysis (`edit_files.apply_preprocessor`).

    `files` seeds the virtual text; `capture=False` makes the overlay read-only (writes go to
    disk as usual)."""

    def __init__(self, files: dict | None = None, capture: bool = True):
        self.files: dict[str, str] = {self.key(p): t for p, t in (files or {}).items()}
        self.capture = capture

    @staticmethod
    def key(p) -> str:
        return os.path.abspath(os.fspath(p))

    @contextlib.contextmanager
    def active(self):
        from spel.scripts import edit_files

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
            if vfs.capture and any(c in mode for c in "wax+") and "b" not in mode and not path.startswith("/dev/"):
                initial = vfs.files.get(path, "") if "a" in mode else ""
                return _Capture(path, initial)
            if "b" not in mode and path in vfs.files:
                return io.StringIO(vfs.files[path])
            return real_open(file, mode, *a, **k)

        P.read_text, P.exists, P.is_file, builtins.open = read_text, exists, is_file, open_
        if self.capture:
            P.write_text, P.write_bytes, P.mkdir = write_text, write_bytes, mkdir
        saved_overrides = edit_files.SOURCE_OVERRIDES
        edit_files.SOURCE_OVERRIDES = self.files  # the same dict: files captured later count too
        try:
            yield self
        finally:
            (P.write_text, P.write_bytes, P.read_text, P.exists, P.is_file, builtins.open, P.mkdir) = saved
            edit_files.SOURCE_OVERRIDES = saved_overrides
