"""
edit_files: objects of a type that unittest_defs.h #defines to an intrinsic
(hlm_fates_interface_type -> integer) are kept, so they can still be passed as
arguments; only statements that reference their components are commented out.
"""

import pytest

import spel.scripts.edit_files as ef
from spel.scripts.tests.test_edit_files import edit

INST = """
module elm_instmod
  use elmfatesinterfacemod, only : hlm_fates_interface_type
  implicit none
  type(hlm_fates_interface_type) :: alm_fates
contains
  subroutine init()
    call alm_fates%init(1)
  end subroutine init
end module elm_instmod
"""

CALLER = """
module soildecompmod
  use elm_instmod, only : fates => alm_fates
  implicit none
contains
  subroutine decomp(bounds)
    integer :: bounds
    call resolve(bounds, fates)
    if (fates%n > 0) then
       bounds = 1
    end if
  end subroutine decomp
  subroutine resolve(bounds, elm_fates)
    use elmfatesinterfacemod, only : hlm_fates_interface_type
    integer :: bounds, n
    type(hlm_fates_interface_type), intent(inout) :: elm_fates
    n = elm_fates%fates(1)%n
    bounds = n
  end subroutine resolve
end module soildecompmod
"""


@pytest.fixture(autouse=True)
def fresh_exports(monkeypatch):
    monkeypatch.setattr(ef, "opaque_exports", {})
    monkeypatch.setattr(ef, "unavailable_exports", {})
    # other tests' analyses may have marked these modules unavailable
    monkeypatch.setattr(ef, "bad_modules", ef.bad_modules - {"elm_instmod", "soildecompmod"})


def active(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip() and "!#py" not in line]


def test_header_types_read_from_unittest_defs():
    assert "hlm_fates_interface_type" in ef.header_types
    assert "alm_fates" not in ef.bad_symbols


def test_header_type_objects_kept_component_refs_removed():
    inst = active(edit(INST, "elm_instmod"))
    assert "type(hlm_fates_interface_type) :: alm_fates" in inst
    assert not any("use elmfatesinterfacemod" in line for line in inst)
    assert "call alm_fates%init(1)" not in inst
    assert "alm_fates" not in ef.unavailable_exports.get("elm_instmod", set())

    caller = active(edit(CALLER, "soildecompmod"))
    assert "use elm_instmod, only : fates => alm_fates" in caller
    assert "call resolve(bounds, fates)" in caller
    assert "type(hlm_fates_interface_type), intent(inout) :: elm_fates" in caller
    # component references go, with the whole if block around them
    assert not any("%" in line for line in caller)
    assert "bounds = 1" not in caller
    assert "bounds = n" in caller
