"""The errors `spel proptest` reports to the user as one line (cli.py) instead of a traceback:
a spec, domain, expression, shim, build or case problem the user can fix."""


class ProptestError(Exception):
    pass


class SpecError(ProptestError):
    """A problem in a `<name>.proptest.toml` (also raised while running it)."""
