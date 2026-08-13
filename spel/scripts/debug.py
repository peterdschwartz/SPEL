#!/bin/python3

from spel.scripts.fortran_modules import build_module_tree, print_spel_module_dependencies
from spel.scripts.utilityFunctions import Variable
from spel.scripts.export_objects import unpickle_unit_test
from pprint import pprint, pformat

from spel.scripts.io.helper import get_var_usage_and_elm_inst_vars
from spel.scripts.types import Scope

mod_dict, sub_dict, type_dict = unpickle_unit_test('c76c282')

trees = build_module_tree(mod_dict)
for tree in trees:
    print(tree)
    print(tree.find_dependency("elm_instmod"))

