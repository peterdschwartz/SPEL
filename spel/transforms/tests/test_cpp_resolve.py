"""Preprocessor conditionals on settled macros (cpp_resolve)."""
import pytest

from spel.transforms.cpp_resolve import CppError, resolve_lines

SRC = """\
a
#ifdef MODAL_AER
b
#ifdef _OPENACC
c
#else
d
#endif
#else
e
#ifdef _OPENACC
f
#endif
#endif
#ifndef MODAL_AER
g
#endif
#ifdef _OPENACC
#ifndef MODAL_AER
h
#else
i
#endif
#endif
z""".split("\n")


def kept(src, values):
    drop = set(resolve_lines(src, values))
    return [l for k, l in enumerate(src) if k not in drop]


def test_modal_aer_defined():
    assert kept(SRC, {"MODAL_AER": True}) == ["a", "b", "#ifdef _OPENACC", "c", "#else", "d", "#endif",
                                               "#ifdef _OPENACC", "i", "#endif", "z"]


def test_undefined():
    assert kept(SRC, {"MODAL_AER": False}) == ["a", "e", "#ifdef _OPENACC", "f", "#endif", "g",
                                                "#ifdef _OPENACC", "h", "#endif", "z"]


def test_refuses_mixed_if():
    with pytest.raises(CppError):
        resolve_lines(["#if defined(MODAL_AER) && defined(X)", "a", "#endif"], {"MODAL_AER": True})
