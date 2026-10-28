"""DeepCopyGlobalsMod: module variables of the same name in two modules get module-qualified local names."""
from pathlib import Path

from spel.transforms.device.generate import GlobalVar, _globals_module


def test_same_name_in_two_modules():
    gvars = [GlobalVar("perchroot", "canopystatetype", 0, False), GlobalVar("perchroot", "soilmoiststressmod", 0, False),
             GlobalVar("oldfflag", "canopyhydrologymod", 0, False)]
    text = _globals_module(gvars, Path("/src"), "abc")
    assert "use canopystatetype, only : perchroot_canopystatetype => perchroot" in text
    assert "use soilmoiststressmod, only : perchroot_soilmoiststressmod => perchroot" in text
    assert "use canopyhydrologymod, only : oldfflag" in text
    assert "!$acc update device(perchroot_soilmoiststressmod)" in text
    assert "update device(perchroot)" not in text


def test_driver_module_declares_each_public_once():
    # Intel rejects a symbol in a `public ::` list that also has the public attribute
    import inspect

    from spel.transforms.device import generate

    src = inspect.getsource(generate)
    assert "logical, public, save :: elm_device_resident" not in src
