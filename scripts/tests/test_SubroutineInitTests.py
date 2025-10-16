import os
import re
import unittest

import scripts.export_objects as eo
from scripts.analyze_subroutines import Subroutine
from scripts.nml.analyze_elm import elmtypes, generate_noifs, insert_default
from scripts.nml.analyze_ifs import flatten, run
from scripts.nml.analyze_namelist import find_all_namelist, generate_namelist_dict
from scripts.types import ReadWrite


class SubroutineTests(unittest.TestCase):
    """
    Tests to exercise the Subroutine class methods.
    NOTE: Hardcoding answers may be difficult if E3SM version is changed.
          Storing copy of files, may not allow testing of E3SM's directory structure.
          Sol'n:  store the commit hash the tests were validated against?
    """

    def test_get_subroutine_file(self):
        from scripts.analyze_subroutines import Subroutine

        sname = "SoilTemperature"
        # Initialize Subroutine
        s = Subroutine(sname, mod_name="soiltemperaturemod", calltree=["elm_drv"])
        correct_filepath = "E3SM/components/elm/src/biogeophys/SoilTemperatureMod.F90"
        regex_fp = re.compile(f"({correct_filepath})$")
        self.assertRegex(s.filepath, regex_fp)

    def test_parse_modules_simple(self):
        """
        Test to verfiy that modules are parsed correctly
        """
        from scripts.config import E3SM_SRCROOT
        from scripts.fortran_modules import get_filename_from_module

        module_name = "shr_kind_mod"
        file_path = get_filename_from_module(module_name, verbose=True)
        correct_filepath = "/share/util/shr_kind_mod.F90"
        file_path = file_path.replace(E3SM_SRCROOT, "")
        self.assertEqual(file_path, correct_filepath)

    def test_find_elm_variables(self):

        mod_dict, sub_dict, type_dict = eo.unpickle_unit_test("-352536b")
        subroutine_calls = {}
        for s in sub_dict.keys():
            for calls in sub_dict[s].child_Subroutine.keys():
                subroutine_calls.setdefault(calls, []).extend(
                    sub_dict[s].child_Subroutine[calls].subroutine_call
                )

        file = "analysis/t.txt"
        parent_ifs = run(file)
        insert_default(mod_dict)

        namelist_dict = find_all_namelist()
        flat = flatten(parent_ifs)
        generate_namelist_dict(mod_dict, namelist_dict, flat, subroutine_calls)
        subroutine = "test"
        sub_dict = {
            subroutine: Subroutine(
                name=subroutine,
                calltree=[],
                file="test",
                start=-999,
                end=-999,
                lib_func=True,
            )
        }
        sub_dict[subroutine].elmtype_access_by_ln = {
            "elmtype2": [ReadWrite(status="w", ln=7), ReadWrite(status="r", ln=20)],
            "elmtype_to_add": [ReadWrite(status="w", ln=14)],
            "blah": [ReadWrite(status="w", ln=4)],
            "y": [ReadWrite(status="w", ln=9)],
            "elm_type": [ReadWrite(status="w", ln=23), ReadWrite(status="w", ln=12)],
            "elmtype_to_not_not_add": [ReadWrite(status="w", ln=1)],
            "z": [ReadWrite(status="w", ln=20)],
        }
        noif = generate_noifs(sub_dict, subroutine, parent_ifs)
        alive, dead = elmtypes(sub_dict, parent_ifs, "test", namelist_dict, noif)
        print(alive)

        self.assertIn("elmtype2", alive)
        self.assertIn("elm_type", alive)
        self.assertIn("z", alive)
        self.assertEqual(len(alive), 3)
        self.assertIn("elmtype_to_not_not_add", dead)
        self.assertIn("blah", dead)
        self.assertIn("elmtype2", dead)
        self.assertIn("y", dead)
        self.assertIn("elm_type", dead)
        self.assertIn("elmtype_to_add", dead)


if __name__ == "__main__":
    import os

    print(os.getcwd())
    unittest.main()
