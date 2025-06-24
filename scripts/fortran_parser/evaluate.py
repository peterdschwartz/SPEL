from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pprint import pprint
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from scripts.DerivedType import DerivedType, expand_dtype
from scripts.interfaces import resolve_interface2

if TYPE_CHECKING:
    from scripts.analyze_subroutines import Subroutine
    from scripts.utilityFunctions import Variable

from scripts.fortran_parser.environment import Environment, add_ptr_vars
from scripts.fortran_parser.lexer import Lexer
from scripts.fortran_parser.spel_parser import Parser
from scripts.fortran_parser.tracing import Trace
from scripts.types import (ArgDesc, ArgNode, ArgType, ArgVar, CallDesc,
                           IdentKind, LineTuple)

REAL = "real"
INT = "integer"
CHAR = "character"
BOOL = "logical"
INHERITED = "inherited"

map_py_to_f_types = {int: INT, float: REAL,  str : CHAR, bool: BOOL ,}

@dataclass
class SymbolTable:
    vars: List[ArgNode]
    intrinsics: List[ArgNode]
    fns: List[ArgNode]

class ArgTree:
    """
    node: ArgNode
    children: List[ArgTree]
    parent
    """
    def __init__(self, node):
        self.node: ArgNode = node
        self.children: List[ArgTree] = []
        self.parent: Optional[ArgTree] = None

    def add_child(self, child: ArgTree):
        child.parent = self
        self.children.append(child)

    def traverse_preorder(self):
        """Pre-order traversal (node -> children)."""
        yield self
        for child in self.children:
            yield from child.traverse_preorder()

    def traverse_postorder(self):
        """Post-order traversal (children -> node)."""
        for child in self.children:
            yield from child.traverse_postorder()
        yield self

    def __repr__(self):
        return f"ArgTree({self.node.ident}, children={len(self.children)})"

    def inc_argn(self):
        """
        Increment argn for each node in ArgTree. Used for Class Methods
        """
        for node in self.traverse_preorder():
            node.node.argn += 1

    def print_tree(self, level: int = 0):
        """Recursively prints the tree in a hierarchical format."""
        indent = "|--" * level
        print(
            f"{indent}-{self.node.argn} {self.node.node}(kind: {self.node.kind.name})"
        )

        for child in self.children:
            child.print_tree(level + 1)

    def remove(self):
        """
        Removes this node from the tree by reattaching its children to its parent.
        Raises a ValueError if attempting to remove the root node.
        -- self must be a child node!
        """
        if self.parent is None:
            raise ValueError("Cannot remove the root node.")

        parent = self.parent
        index = parent.children.index(self)

        # Update parent references for this node's children.
        for child in self.children:
            child.parent = parent

        # Short-hand slice syntax to replace child node with it's children
        parent.children[index:index+1] = self.children

        self.children = []
        self.parent = None


# Dictionary to hold intrinsic Fortran functions.
intrinsic_fns = {
    "min": ArgType(datatype=REAL, dim=0),
    "max": ArgType(datatype=REAL, dim=0),
    "sin": ArgType(datatype=REAL, dim=0),
    "cos": ArgType(datatype=REAL, dim=0),
    "tan": ArgType(datatype=REAL, dim=0),
    "acos": ArgType(datatype=REAL, dim=0),
    "asin": ArgType(datatype=REAL, dim=0),
    "atan": ArgType(datatype=REAL, dim=0),
    "abs": ArgType(datatype=REAL, dim=0),
    "exp": ArgType(datatype=REAL, dim=0),
    "sqrt": ArgType(datatype=REAL, dim=0),
    "dim": ArgType(datatype=INT, dim=0),
    "mod": ArgType(datatype=INT, dim=0),
    "modulo": ArgType(datatype=INT, dim=0),
    # string
    "trim": ArgType(datatype=CHAR, dim=0),
    "len": ArgType(datatype=INT, dim=0),
    # array
    "size": ArgType(datatype=INT, dim=0),
    "shape": ArgType(datatype=INT, dim=1),
    # "any",
    # "all",
    # math
    "sum": ArgType(datatype=REAL, dim=0),
    "product": ArgType(datatype=REAL, dim=0),
    # "matmul",
    "maxval": ArgType(datatype=REAL, dim=0),
    "minval": ArgType(datatype=REAL, dim=0),
    "maxloc": ArgType(datatype=INT, dim=0),
    "minloc": ArgType(datatype=INT, dim=0),
    # conversion functions
    "int": ArgType(datatype=INT, dim=0),
    "real": ArgType(datatype=REAL, dim=0),
    "dble": ArgType(datatype=REAL, dim=0),
    "epsilon": ArgType(datatype=REAL, dim=0),
}


_ignore_subs = [
    r'\bcpu_time\b',
    r"\bget_curr_date\b",
    r"\bt_start",
    r"\bt_stop",
    r"\bendrun\b",
    r"\bupdate_vars",
]
ignore_str = "|".join(_ignore_subs)
regex_ignore = re.compile(r"({})".format(ignore_str))


def construct_arg_tree(
    args,
    env: Environment,
) -> List[ArgTree]:
    """
    Takes a list of expressions from subroutine call, returns
    a List with the ith element corresponding the ith argument expression.
    """
    flat_tree: List[List[ArgNode]] = [[] for i in range(0, len(args))]
    for arg_num, expr in enumerate(args):
        evaluate(arg_num, expr, env, flat_tree, nested=0)

    arg_trees : List[ArgTree] = []
    for nodes in flat_tree:
        arg_trees.append(make_tree(nodes))

    return arg_trees


def make_tree(flat_arg_nodes: List[ArgNode]) -> ArgTree:
    """
    Generates a linked tree structure for each argument node based on nested_level
    """

    root = ArgTree(flat_arg_nodes[0])
    stack = [(flat_arg_nodes[0].nested_level, root)]  # Stack to track parents

    for node in flat_arg_nodes[1:]:
        node_tree = ArgTree(node)

        # Find the correct parent by checking the stack
        while stack and stack[-1][0] >= node.nested_level:
            stack.pop()

        if stack:
            parent_tree = stack[-1][1]
            parent_tree.add_child(node_tree)

        stack.append((node.nested_level, node_tree))

    return root


@Trace.trace_decorator("parse_subroutine_call")
def parse_subroutine_call(
    sub: Subroutine,
    sub_dict: dict[str, Subroutine],
    input: LineTuple,
    ilist: list[str],
    instance_dict: dict[str, DerivedType],
    verbose: bool = False,
) -> Optional[CallDesc]:
    """
    Function to generate and parse an AST for a subroutine call string
    """
    lex = Lexer(input=input.line)
    parser = Parser(lex=lex)
    program = parser.parse_program()
    if len(program.statements) > 1:
        print("Error -- multiple calls in input")
        sys.exit(1)

    stmt = program.statements[0]
    ast: Dict[str, Any] = stmt.to_dict()

    node_type = ast["Node"]
    if node_type != "SubCallStatement":
        print("Error -- Not a SubCallStatement!\n{input}\n")
        sys.exit(1)
    sub_node: dict[str, Any] = ast["Sub"]
    sub_name = sub_node["Func"]
    arg_list = sub_node["Args"]
    if regex_ignore.search(sub_name):
        return None

    type_dict: dict[str,DerivedType] = {inst.type_name : inst for inst in instance_dict.values()}

    if not sub.environment:
        env: Environment = create_environment(sub, sub_dict, type_dict)
        sub.environment = env
        if verbose:
            print("Environment: \n")
            pprint(env.to_dict(),sort_dicts=False)
    else:
        env = sub.environment


    arg_tree = construct_arg_tree(arg_list, env)

    arg_desc_list = create_arg_descr(arg_tree, env)


    if sub_name in ilist:
        fn = resolve_interface2(sub_name, arg_desc_list, sub_dict) 
    elif "%" in sub_name:
        fn = resolve_class_method(sub_name, instance_dict, arg_desc_list, arg_tree)
    else:
        fn = sub_name
    if not fn:
        print(f"Error - couldn't resolve interface or class method {sub_name}")
        sys.exit(1)

    sort_variables(sub, arg_tree, arg_desc_list)

    return CallDesc(alias=sub_name,
                    fn=fn,
                    args=arg_desc_list,
                    globals=[],
                    locals=[],
                    dummy_args=[],
                    lpair=input)

def resolve_class_method(
    subname: str,
    inst_dict: dict[str,DerivedType],
    arg_desc_list: list[ArgDesc],
    tree: list[ArgTree],
)->str:
    """
    Function that returns the name of the class method.
    Side-effects:
        New arg_desc is added for class var, and the argn's are incremented
    """
    inst_name, method = subname.split('%')
    dtype = inst_dict[inst_name]
    for arg in arg_desc_list:
        arg.increment_arg_number()

    class_arg_node = ArgNode(
        argn=0,
        ident=inst_name,
        kind=IdentKind.variable,
        nested_level=0,
        node={"Node":"Ident", "Val": inst_name},
    )
    for branch in tree:
        branch.inc_argn()

    cl_tree = ArgTree(class_arg_node)
    tree.insert(0, cl_tree)

    arg_type = ArgType(datatype=dtype.type_name, dim=0)
    class_arg_desc = ArgDesc(argn=0,
                       intent='',
                       keyword=False,
                       key_ident='',
                       argtype=arg_type,
                       locals=[],
                       globals=[],
                       dummy_args=[],
                       )

    arg_desc_list.insert(0,class_arg_desc)

    return dtype.procedures[method]

def create_environment(
    sub: Subroutine,
    sub_dict: dict[str, Subroutine],
    type_dict: dict[str, DerivedType],
) -> Environment:
    """
    Package relevant variables and functions for this subroutine.
    variables are categorized as local, dummy argument, or "global"
    """
    intrinsic_types = { "real", "integer", "character", "logical" }
    local_vars: dict[str,Variable] = sub.local_variables

    for ptr, gv_key in sub.associate_vars.items():
        if gv_key in sub.dtype_vars:
            sub.dtype_vars[ptr] = sub.dtype_vars[gv_key]

    variables: dict[str, Variable] = (
        sub.dtype_vars
        | sub.active_global_vars
        | local_vars
    )
    global_dict: dict[str,Variable] = sub.dtype_vars | sub.active_global_vars

    # Local Variables
    dtype_locals: list[Variable] = [ var for var in local_vars.values()
        if var.type not in intrinsic_types
    ]

    expanded_dtypes = expand_dtype(dtype_locals, type_dict)
    variables.update(expanded_dtypes)

    local_dict: dict[str,Variable] = expanded_dtypes | local_vars

    # Argument Variables
    dtype_args: list[Variable] = [
        var for var in sub.arguments.values() if var.type not in intrinsic_types
    ]
    expanded_dtypes = expand_dtype(dtype_args, type_dict)

    # Add associated names for argument derived types to expanded arg dict.
    for ptr, var in sub.associate_vars.items():
        gv_key = var[0] if isinstance(var,list) else var
        if gv_key in expanded_dtypes:
            expanded_dtypes[ptr] = expanded_dtypes[gv_key]

    variables.update(expanded_dtypes)
    variables.update(sub.arguments)

    dummy_dict: dict[str,Variable] = expanded_dtypes | sub.arguments

    instance_dict: Dict[str,DerivedType] = {}
    inst_var_dict: Dict[str, Variable] = {}

    for dtype in type_dict.values():
        for inst_name, inst_var in dtype.instances.items():
            if inst_name not in instance_dict:
                instance_dict[inst_name] = dtype
            inst_var_dict[inst_name] = inst_var

    global_dict.update(inst_var_dict)

    variables.update(inst_var_dict)

    for argname, arg in sub.arguments.items():
        if arg.type in type_dict.keys():
            instance_dict[argname] = type_dict[arg.type]

    return Environment(variables=variables,
                       locals=local_dict,
                       dummy_args=dummy_dict,
                       globals=global_dict,
                       fns=sub_dict,
                       inst_dict=instance_dict)

def evaluate(
    argn: int,
    expr:dict[str,Any],
    env: Environment,
    arg_tree_flat: List[List[ArgNode]],
    nested: int,
):
    """
    Function that is called recursively on an ast.expression to categorize
    all the identifiers.
    """
    node = expr["Node"]
    match node:
        case "Ident":
            var = expr["Val"]
            arg_tree_flat[argn].append(
                ArgNode(argn=argn, ident=var, kind=IdentKind.variable, nested_level=nested, node=expr)
            )

        case "FuncExpression":
            func_name = expr["Func"]

            if func_name in intrinsic_fns:
                arg_node = ArgNode(argn=argn, ident=func_name, kind=IdentKind.intrinsic, nested_level=nested, node=expr,)
            elif func_name in env.fns:
                arg_node = ArgNode(argn=argn, ident=func_name, kind=IdentKind.function, nested_level=nested, node=expr,)
            else:
                arg_node = ArgNode(argn=argn, ident=func_name, kind=IdentKind.variable, nested_level=nested, node=expr,)

            arg_list = expr["Args"]
            arg_tree_flat[argn].append(arg_node)
            for argexpr in arg_list:
                evaluate(argn, argexpr, env, arg_tree_flat, nested + 1)

        case "FieldAccessExpression":
            arg_tree_flat[argn].append(
                ArgNode(
                    argn=argn,
                    ident="FieldAccessExpression",
                    kind=IdentKind.field,
                    nested_level=nested,
                    node=expr,
                )
            )
            expr_list = [ expr["Left"], expr["Field"]]
            for opexpr in expr_list:
                evaluate(argn, opexpr, env, arg_tree_flat, nested + 1)

        case "InfixExpression":
            arg_tree_flat[argn].append(
                ArgNode(
                    argn=argn,
                    ident="InfixExpression",
                    kind=IdentKind.infix,
                    nested_level=nested,
                    node=expr,
                )
            )
            if expr["Op"] != "=":
                expr_list = [
                    expr["Left"],
                    expr["Right"],
                ]
            else:
                expr_list = [expr["Right"]]
            for argexpr in expr_list:
                evaluate(argn, argexpr, env, arg_tree_flat, nested + 1)

        case "PrefixExpression":
            arg_tree_flat[argn].append(
                ArgNode(
                    argn=argn,
                    ident="PrefixExpression",
                    kind=IdentKind.prefix,
                    nested_level=nested,
                    node=expr,
                )
            )
            evaluate(argn, expr["Right"], env, arg_tree_flat, nested + 1)

        case "BoundsExpression":
            arg_tree_flat[argn].append(
                ArgNode(
                    argn=argn,
                    ident=expr["Val"],
                    kind=IdentKind.slice,
                    nested_level=nested,
                    node=expr,
                )
            )
        case "IntegerLiteral":
            arg_tree_flat[argn].append(
                ArgNode(
                    argn=argn,
                    ident=expr["Val"],
                    kind=IdentKind.literal,
                    nested_level=nested,
                    node=expr,
                )
            )
        case "FloatLiteral":
            arg_tree_flat[argn].append(
                ArgNode(
                    argn=argn,
                    ident=expr["Val"],
                    kind=IdentKind.literal,
                    nested_level=nested,
                    node=expr,
                )
            )
        case "StringLiteral":
            arg_tree_flat[argn].append(
                ArgNode(
                    argn=argn,
                    ident=expr["Val"],
                    kind=IdentKind.literal,
                    nested_level=nested,
                    node=expr,
                )
            )
        case "LogicalLiteral":
            arg_tree_flat[argn].append(
                ArgNode(
                    argn=argn,
                    ident=expr["Val"],
                    kind=IdentKind.literal,
                    nested_level=nested,
                    node=expr,
                )
            )
        case _:
            print("Didn't account for Node type", node)
            sys.exit(1)
    return


def create_arg_descr(
    arg_tree: List[ArgTree],
    env: Environment,
) -> List[ArgDesc]:
    """
    Function that assigns an Variable type to each function argument and 
    returns an overall description of each argument expression.
    """
    args: List[ArgDesc] = []
    for argn, tree in enumerate(arg_tree):
        arg_type = check_arg_branch(tree, env)
        keyword = check_keyword(tree.node)
        key_ident = tree.node.node['Left'] if keyword else ''
        desc = ArgDesc(argn=argn,
                       intent='',
                       keyword=keyword,
                       key_ident=key_ident,
                       argtype=arg_type,
                       locals=[],
                       globals=[],
                       dummy_args=[],
                       )
        args.append(desc)


    return args

def sort_variables(sub: Subroutine, arg_tree: List[ArgTree], args: List[ArgDesc]) -> None:
    """
    Function to fill out the locals and globals field in ArgDesc
    """
    globals = sub.environment.globals
    locals = sub.environment.locals
    dummy = sub.environment.dummy_args
    for argn, branch in enumerate(arg_tree):
        for node in branch.traverse_preorder():
            if node.node.kind == IdentKind.variable:
                varname = node.node.ident
                if varname in globals:
                    args[argn].globals.append(ArgVar(node=node.node,var=globals[varname]))
                elif varname in locals:
                    args[argn].locals.append(ArgVar(node=node.node,var=locals[varname]))
                elif varname in dummy:
                    args[argn].dummy_args.append(ArgVar(node=node.node,var=dummy[varname]))
    return


def check_arg_branch(
    branch: ArgTree,
    env: Environment,
) -> ArgType:
    """
    Function that takes a list of ArgNodes representing the expression
    passed as an argument to a subroutine call.
    """
    return evaluate_arg_node(branch, env)


def evaluate_arg_node(branch: ArgTree, env: Environment) -> ArgType:
    node = branch.node
    match node.kind:
        case IdentKind.variable:
            var = env.variables[node.ident]
            dim = adjust_dim(var.dim, branch) if branch.children else var.dim
            return ArgType(datatype=var.type, dim=dim)
        case IdentKind.intrinsic:
            return intrinsic_fns[node.ident]
        case IdentKind.function:
            fn = env.fns[node.ident]
            res_var: Variable = fn.result
            return ArgType(datatype=res_var.type, dim=res_var.dim)
        case IdentKind.infix:
            return evaluate_infix_arg(branch, env)
        case IdentKind.literal:
            datatype = map_py_to_f_types[type(node.ident)]
            return ArgType(datatype=datatype, dim=0)
        case IdentKind.field:
            return evaluate_field_access(branch,env)
        case IdentKind.prefix:
            return evaluate_prefix_arg(branch,env)
        case _:
            return ArgType(datatype="default", dim=0)

def evaluate_field_access(branch: ArgTree, env: Environment) -> ArgType:
    """
    The return ArgType will be the type of the field being accessed.
    Side Effects:
        branch.node will be modified to be an identifier with ident
        value equal

    TODO: Handle case where field is an accessed by index and adjust dimensions
    """
    field_name= get_ident(branch.node.node["Field"])
    inst_node = branch.node.node["Left"]

    # FieldAccessExpression only gets used if the instance is AoS
    inst_name = get_ident(inst_node)
    inst = env.inst_dict[inst_name]
    field_var: Variable = inst.components[field_name]


    branch.node.ident = inst_name+"(index)%"+field_name
    branch.node.kind = IdentKind.variable

    for child in branch.children:
        child.remove()

    # dim = adjust_dim(var.dim, branch) if branch.children else var.dim
    arg_type = ArgType(datatype=field_var.type, dim=field_var.dim)

    return arg_type


def get_ident(node: dict[str,Any])->str:
    if node["Node"] == "FuncExpression":
        return node["Func"]
    elif node["Node"] == "Ident":
        return node["Val"]
    else:
        print("Unexpected Expression in FieldAccessExpression:\n",node)
        sys.exit(1)


def evaluate_prefix_arg(branch: ArgTree, env: Environment)-> ArgType:
    """
    Evaluate prefix expression by evaluating the it's child node.
    """
    if len(branch.children) != 1:
        print("Error - prefix expression has more than one child")
        branch.print_tree()

    child = branch.children[0]
    return evaluate_arg_node(child, env)


def evaluate_infix_arg(branch: ArgTree, env: Environment) -> ArgType:
    """
    Evaluate infix argument type by looking at the Left and Right
    expressions and taking the higher precision of the two.
    """
    child_types = [evaluate_arg_node(child,env) for child in branch.children]
    unique_types = set(child_types)
    if(len(unique_types) == 1):
        return next(iter(unique_types))
    else:
        # There is a mixed datatype expression.
        if(not arg_dim_equals(unique_types) or not valid_mixed_types(unique_types)):
            print("Error -- Infix Operation on non-equal dimension or compatible types")
            pprint(branch.node.node,sort_dicts=False)
            branch.print_tree()
            sys.exit(1)
        return next(filter(lambda arg: arg.datatype == REAL, unique_types))

def arg_dim_equals(arg_set: set[ArgType]) -> bool:
    return len({arg.dim for arg in arg_set}) <= 1

def valid_mixed_types(arg_set: set[ArgType]) -> bool:
    return {arg.datatype for arg in arg_set} <= {INT, REAL}

def adjust_dim(dim: int, branch: ArgTree) -> int:
    """
    Take into account indices and slices to get accurate number of dimensions
    """
    assert branch.children != [], "Adjusting dims but children empty"

    for child in branch.children:
        if child.node.kind != IdentKind.slice:
            dim -=1

    assert dim >= 0, f"can't have negative dimension:\n {branch.node}"
    return dim

def check_keyword(node: ArgNode)->bool:
    if(node.kind != IdentKind.infix):
        return False
    else:
        return bool(node.node['Op'] == '=')
