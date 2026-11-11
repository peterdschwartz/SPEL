"""
Property-based testing of extracted ELM routines (`spel proptest`).

A SPEL unit-test case is built as a shared library with a generated `bind(C)` shim
(`shim.py`). Hypothesis draws synthetic inputs over a synthetic subgrid domain
(`domain.py`, `strategies.py`) and checks invariants and metamorphic relations written
in Fortran syntax in a TOML file (`spec.py`). The expressions are parsed with SPEL's parser and evaluated
with numpy (`fexpr.py`). Every fact about the routine's variables comes from SPEL's
analysis (`inventory.py`), never from scanning source text.
"""
