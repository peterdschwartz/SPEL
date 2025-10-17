#!/bin/python3

from scripts.export_objects import unpickle_unit_test
from pprint import pprint, pformat

from scripts.types import Scope

mod_dict, sub_dict, type_dict = unpickle_unit_test('98f61cc')

# debug_sub = sub_dict['phenologymod::cropplantdate']
sub_name = 'cropmod::plant_month'
debug_sub  = sub_dict[sub_name]

print("Arguments")
pprint(debug_sub.arguments)
print("dtype_vars")
pprint(debug_sub.dtype_vars)
print("propagated")
pprint(debug_sub.propagated_access_by_ln)

