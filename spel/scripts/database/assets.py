modules = [
    "module_id",
    "module_name"
]

module_dependency = [
    "dependency_id",
    "module_id",
    "dep_module_id"
    "object_used",
]

subroutine_active_global_vars = [
    "variable_id",
    "subroutine_id",
    "instance_id",
    "member_id",
    "status",
]

subroutine_args = [
    "arg_id",
    "subroutine_id",
    "arg_type",
    "arg_name",
    "dim",
]

subroutine_calltree = [
    "parent_id",
    "parent_subroutine_id",
    "child_subroutine_id",
]

subroutine_local_arrays = [
    "local_arry_id",
    "subroutine_id",
    "array_name",
    "dim",
]

subroutines = [
    "subroutine_id",
    "subroutine_name",
    "module_id",
]

type_definitions = [
    "define_id",
    "module_id",
    "user_type_id",
    "member_type",
    "member_name",
    "dim",
    "bounds",
    "active",
]

user_type_instances = [
    "instance_id",
    "instance_type_id",
    "instance_name",
]

user_types = [
    "user_type_id",
    "module_id",
    "user_type_name",
]






def query(cur,
          cols="*", # cols you want to see. default is all (*)
          conditions = []
          ):
    statement = "SELECT "
    
    if cols == "*":
        statement += cols
    else:
        c = []
        for el in cols:
            for col in cols[el]:
                c.append(f"{el}.{col}")
        statement += f"{','.join(c)}"
    statement += f" FROM {' JOIN '.join(cols.keys())}"
    statement += f" WHERE {' AND '.join(conditions)}"
    print(statement)

dict = {"subroutines":["subroutine_name"],
        "user_type_instances": ["instance_name"],
        "subroutine_active_global_vars": ["status"],
        "type_definitions": ["member_name"]
        }
    
query(None,
      cols=dict,
      conditions=[
          "subroutine_active_global_vars.subroutine_id = subroutines.subroutine_id",
          "user_type_instances.instance_id = subroutine_active_global_vars.instance_id",
          "subroutine_active_global_vars.member_id = type_definitions.define_id"
      ]
      )

"""
SELECT 
	m1.subroutine_name, 
	m2.instance_name, 
	m4.member_name, 
	m3.status 
FROM 
	subroutines m1 
JOIN 
	user_type_instances m2 
JOIN 
	subroutine_active_global_vars m3 
JOIN 
	type_definitions m4 
WHERE 
	m3.subroutine_id = m1.subroutine_id 
and 
	m3.instance_id = m2.instance_id 
and 
	m3.member_id=m4.define_id;

"""