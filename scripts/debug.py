# #!/bin/python3
#
# from scripts.utilityFunctions import Variable
# from scripts.export_objects import unpickle_unit_test
# from pprint import pprint, pformat
#
# from scripts.io.helper import get_var_usage_and_elm_inst_vars
# from scripts.types import Scope
#
# mod_dict, sub_dict, type_dict = unpickle_unit_test('c76c282')
#
# sub_name = 'soillittverttranspmod::soillittverttransp'
# debug_sub  = sub_dict[sub_name]
#
# active_instances, use_statements, elminst_vars = get_var_usage_and_elm_inst_vars(type_dict)
#
# instance_to_user_type: dict[str, str] = {}
# for type_name, dtype in type_dict.items():
#     if "bounds" in type_name:
#         continue
#     for instance in dtype.instances.values():
#         instance_to_user_type[instance.name] = type_name
#
# dtype_vars: dict[str, Variable] = {}
#
# for inst_var in active_instances.values():
#     type_name = instance_to_user_type[inst_var.name]
#     dtype = type_dict[type_name]
#     for field_var in dtype.components.values():
#         if field_var.active and not field_var.pointer:
#             new_var = field_var.copy()
#             new_var.name = f"{inst_var.name}%{field_var.name.split('%')[-1]}"
#             dtype_vars[new_var.name] = new_var
#
# for var in dtype_vars.values():
#     print(var.name, var.bounds)
import warnings

warnings.filterwarnings("error")

import numpy as np  # noqa: F401

print("Imported numpy OK")

import xarray as xr  # noqa: F401

print("Imported xarray OK")

# add more suspects one by one:
import pandas  # noqa: F401

print("Imported pandas OK")
