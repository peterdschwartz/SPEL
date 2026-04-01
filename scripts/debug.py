#!/bin/python3

from scripts.fortran_modules import build_module_tree, print_spel_module_dependencies
from scripts.utilityFunctions import Variable
from scripts.export_objects import unpickle_unit_test
from pprint import pprint, pformat

from scripts.io.helper import get_var_usage_and_elm_inst_vars
from scripts.types import Scope

mod_dict, sub_dict, type_dict = unpickle_unit_test('c76c282')

trees = build_module_tree(mod_dict)
for tree in trees:
    print(tree)
    print(tree.find_dependency("elm_instmod"))

