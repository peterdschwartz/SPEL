from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from enum import Enum, auto
from logging import Logger
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

from scripts.fortran_parser.spel_ast import Expression
from scripts.logging_configs import get_logger

if TYPE_CHECKING:
    from scripts.analyze_subroutines import Subroutine
    from scripts.utilityFunctions import Variable

regex_preand = re.compile(r'^\s*&')

class SubStart(NamedTuple):
    subname: str
    start_ln: int
    cpp_ln: Optional[int]


class ArgLabel(Enum):
    dummy = 1
    globals = 2
    locals = 3


class FortranTypes(Enum):
    CHAR = 1
    LOGICAL = 2
    INT = 3
    REAL = 4
    INHERITED = 5


class FileInfo(NamedTuple):
    fpath: str
    startln: int
    endln: int


@dataclass(frozen=True)
class ArgType:
    datatype: str
    dim: int


class IdentKind(Enum):
    intrinsic = 1
    variable = 2
    function = 3
    infix = 4
    prefix = 5
    slice = 6
    literal = 7
    field = 8


@dataclass
class GlobalVar:
    """
    Used to represent global non-derived type variables
        var: Variable
        init_sub_ptr: Subroutine
    """

    var: Variable
    init_sub_ptr: Subroutine


@dataclass
class ArgNode:
    argn: int
    ident: int | float | str
    kind: IdentKind
    nested_level: int
    node: dict

    def to_dict(self):
        return asdict(self)


@dataclass
class ArgVar:
    node: ArgNode
    var: Variable

    def to_dict(self):
        return asdict(self)


@dataclass
class ArgDesc:
    argn: int  # argument number
    intent: str  # in/out/inout => 'r', 'w', 'rw'
    keyword: bool  # passed as a keyword argument
    key_ident: str  # identifier of keyword
    argtype: ArgType  # overall type and dimension
    locals: list[ArgVar]  # local variables passed to this argument
    globals: list[ArgVar]  # global variables passed to this argument
    dummy_args: list[ArgVar]  # track dummy arguments

    def to_dict(self):
        return asdict(self)

    def increment_arg_number(self):
        """
        Function needed for class methods that have an implicit
        first argument
        """

        def inc_list(argvar_list: list[ArgVar]):
            for v in argvar_list:
                v.node.argn += 1

        self.argn += 1
        inc_list(self.locals)
        inc_list(self.globals)
        inc_list(self.dummy_args)


@dataclass
class CallDesc:
    alias: str  # Interface name, class method alias, or actual name
    fn: str  # real Name of called subroutine
    args: list[ArgDesc]
    lpair: LineTuple
    # summary of the ArgDesc fields
    globals: list[ArgVar]
    locals: list[ArgVar]
    dummy_args: list[ArgVar]

    def to_dict(self, long=False):
        if long:
            return asdict(self)
        else:
            return {k: v for k, v in asdict(self).items() if k not in ["args"]}

    def aggregate_vars(self) -> None:
        """
        Populates the globals, locals, and dummy_args field for the entire CallDesc
        """
        for arg in self.args:
            temp = [v for v in arg.globals if v not in self.globals]
            self.globals.extend(temp)

            temp = [v for v in arg.locals if v not in self.locals]
            self.locals.extend(temp)

            temp = [v for v in arg.dummy_args if v not in self.dummy_args]
            self.dummy_args.extend(temp)
        return


class ObjType(Enum):
    """
    enum for the kind of object being used:
        SUBROUTINE
        VARIABLE
        DTYPE
    """

    SUBROUTINE = auto()
    VARIABLE = auto()
    DTYPE = auto()


@dataclass(frozen=True)
class PointerAlias:
    """
    Create Class for used objects that may be aliased
         i.e.,    `ptr` => 'long_object_name'
    -------------------------------------------------
        ptr: Optional[str]
        obj: str
    """

    ptr: Optional[str]
    obj: str

    def __str__(self):
        if self.ptr:
            return f"{ self.ptr } => { self.obj }"
        else:
            return f"{ self.obj }"

    def __repr__(self):
        return f"{self.ptr} => {self.obj}"


@dataclass
class FunctionReturn:
    """
    Dataclass to package fortran function metadata
        return_type: str
        name: str
        result: str
        start_ln: int
        cpp_start: int
    """

    return_type: str
    name: str
    result: str
    start_ln: int
    cpp_start: Optional[int]


@dataclass
class SubInit:
    """
    Dataclass to Initialize Subroutine
        name: str
        mod_name: str
        file: str
        cpp_fn: str
        mod_lines: list[LineTuple]
        start: int
        end: int
        cpp_start: int
        cpp_end: int
        function: Optional[FunctionReturn]
    """

    name: str
    mod_name: str
    file: str
    cpp_fn: str
    mod_lines: list[LineTuple]
    start: int
    end: int
    cpp_start: Optional[int]
    cpp_end: Optional[int]
    function: Optional[FunctionReturn]


@dataclass
class ParseState:
    """
    Represent file for parsing
    """

    module_name: str  # Module in file
    cpp_file: bool  # File contains compiler preprocessor flags
    work_lines: list[LineTuple]  # Lines to parse -- may be equivalent to orig_lines
    orig_lines: list[LineTuple]  # original line number
    path: str  # path to original file
    curr_line: Optional[LineTuple]  # current LineTuple
    line_it: LogicalLineIterator  # Iterator for full fortran statements
    removed_subs: list[str]  # list of subroutines that have been completely removed
    sub_init_dict: dict[str, SubInit]  # Init objects for all subroutines in File
    sub_start: Optional[SubStart]  # Holds start of subroutine info
    func_init: Optional[FunctionReturn]  # holds start of function info
    in_sub: bool = False  # flag if parser is currently in a subroutine
    in_func: bool = False  # flag if parser is in a function

    def get_start_index(self) -> int:
        return self.line_it.start_index


class PreProcTuple(NamedTuple):
    """
    Holds line-numbers for original file and cpp file
        ln: int
        cpp_ln: Optional[int]
    """

    ln: int
    cpp_ln: Optional[int]


@dataclass
class ModUsage:
    all: bool
    clause_vars: set[PointerAlias]


@dataclass
class LineTuple:
    """
    line: str
    ln: int
    commented: bool
    """

    line: str
    ln: int
    # num_cont: int
    commented: bool = False


class ReadWrite(object):
    """
    - status
    - ln
    - line
    """

    def __init__(
        self,
        status: str,
        ln: int,
        line: LineTuple,
    ):
        self.status = status
        self.ln = ln
        self.ltuple: LineTuple = line

    def __eq__(self, other):
        return self.status == other.status and self.ln == other.ln

    def __repr__(self):
        return f"{self.status}@L{self.ln}"


class SubroutineCall:
    """
    namedtuple to log the subroutines called and their arguments
    to properly match read/write status of variables.
    """

    def __init__(self, subname, args, ln):
        self.subname: str = subname
        self.args: list[Any] = args
        self.ln: int = ln

    def __eq__(self, other):
        return (
            (self.subname == other.subname)
            and (self.args == other.args)
            and (self.ln == other.ln)
        )

    def __str__(self):
        return f"{self.subname}@{self.ln} ({self.args})"

    def __repr__(self):
        return str(self)


class CallTuple(NamedTuple):
    """
    Fields:
        nested
        subname
    """

    nested: int
    subname: str


class CallTree:
    """
    Represents node for subroutine and function calls in a call Tree
    Fields:
        node (CallTuple)
        children list[CallTree]
        parent CallTree|None
    """

    def __init__(self, node):
        self.node: CallTuple = node
        self.children: list[CallTree] = []
        self.parent: Optional[CallTree] = None

    def add_child(self, child: CallTree):
        child.parent = self
        self.children.append(child)

    def traverse_preorder(self):
        """Pre-order traversal (node -> children)."""
        yield self
        for child in self.children:
            yield from child.traverse_preorder()

    def traverse_postorder(self):
        """
        Post-order traversal (children -> node).
        """
        for child in self.children:
            yield from child.traverse_postorder()
        yield self

    def __repr__(self):
        return f"CallTree({self.node.subname}, children={len(self.children)})"
    def __str__(self):
        return self.node.subname

    def print_tree(self, level: int = 0):
        """Recursively prints the tree in a hierarchical format."""
        if level == 0:
            print("CallTree for ", self.node.subname)
        indent = "|--" * level
        print(f"{indent}>{self.node.subname}")

        for child in self.children:
            child.print_tree(level + 1)


class LogicalLineIterator:
    def __init__(self, lines: list[LineTuple], log_name: str = ""):
        self.lines = lines
        self.i = 0
        self.start_index = 0
        self.curr_line: LineTuple = lines[0]
        if not log_name:
            self.logger: Logger = get_logger("LineIter", level=logging.INFO)
        else:
            self.logger: Logger = get_logger(log_name, level=logging.INFO)

    def __iter__(self):
        return self

    def reset(self):
        self.i = 0
        self.start_index = 0

    def get_start_ln(self)->int:
        idx = self.start_index
        return self.lines[idx].ln

    def strip_comment(self) -> str:
        in_string = None  # None, "'", or '"'
        line = self.lines[self.i].line
        result = []
        i = 0
        while i < len(line):
            c = line[i]
            if c in ('"', "'"):
                if in_string is None:
                    in_string = c
                elif in_string == c:
                    # handle escaped quote inside string
                    if i + 1 < len(line) and line[i + 1] == c:
                        result.append(c)  # add one quote, skip next
                        i += 1
                    else:
                        in_string = None  # close string
                result.append(c)
            elif c == "!" and in_string is None:
                break  # comment starts here
            else:
                result.append(c)
            i += 1
        return "".join(result)

    def __next__(self):
        if self.i >= len(self.lines):
            raise StopIteration
        self.start_index = self.i

        cur_ln = self.lines[self.i].ln
        full_line = self.strip_comment()
        full_line = full_line.rstrip("\n").strip()
        num_continuations: int = 1
        while full_line.rstrip().endswith("&"):
            num_continuations += 1
            full_line = full_line.rstrip()[:-1].strip()
            self.i += 1
            if self.i >= len(self.lines):
                self.logger.error("Error-- line incomplete!")
                raise StopIteration
            # new_line = self.lines[self.i].line.split("!")[0].strip()
            new_line = self.strip_comment().strip()
            # test if line is just a comment or otherwise empty
            if not new_line:
                full_line += " &"  # re append & so loop goes to next line
            else:
                new_line = regex_preand.sub(' ',new_line)
                full_line += " " + new_line.rstrip("\n")

        result = (full_line.lower(), self.i)
        self.i += 1
        self.curr_line = LineTuple(line=full_line.lower(),ln=cur_ln)
        return result

    def next_n(self, n):
        """Get next n full logical lines."""
        results = []
        for _ in range(n):
            try:
                results.append(next(self))
            except StopIteration:
                break
        return results

    def insert_after(self,stmt: str):
        """
        Inserts after current line and increments subsequent lns
        """
        cur_ln = self.i
        self.lines.insert(cur_ln+1, LineTuple(line=stmt,ln=cur_ln+1))
        self.i += 1
        for i in range(cur_ln+2,len(self.lines)):
            self.lines[i].ln += 1

        return

    def get_curr_line(self):
        if self.i >= len(self.lines):
            return None
        return self.lines[self.i].line

    def has_next(self):
        return self.i < len(self.lines)

    def comment_cont_block(self, index: Optional[int] = None):
        old_index = index if index else self.start_index
        self.logger.debug(f"Commenting {old_index} -> {self.i}(excl)")
        for ln in range(old_index, self.i):
            self.lines[ln].commented = True

    def consume_until(
        self,
        end_pattern: re.Pattern,
        start_pattern: Optional[re.Pattern],
    ) -> tuple[list[LineTuple],int]:

        results: list[LineTuple] = [self.curr_line]
        ln: int = -1
        nesting = 0
        while self.has_next():
            full_line, ln = next(self)
            start_ln = self.get_start_ln()
            results.append(LineTuple(line=full_line,ln=start_ln))
            if start_pattern and start_pattern.match(full_line):
                nesting += 1
            if end_pattern.match(full_line):
                if nesting == 0:
                    break
                else:
                    nesting -= 1

        return results, ln

    def get_orig_ln(self):
        start_index = self.get_start_ln()
        return self.lines[start_index].ln

    def replace_in_line(
        self,
        lns: list[int],
        pattern: re.Pattern,
        repl_str: str,
        logger: Optional[Logger] = None,
    ):
        if not logger:
            logger = self.logger
        for ln in lns:
            self.i = ln
            full_line, adj_ln = next(self)
            delta_ln = adj_ln - ln
            m_ = pattern.search(full_line)
            if m_:
                for i in range(0, delta_ln + 1):
                    curr_line = self.lines[ln + i].line
                    self.lines[ln + i].line = pattern.sub(repl_str, curr_line)
            else:
                self.logger.error(
                    f"(replace_in_line) FAILED to match {pattern} in \n {full_line}"
                )

        return


@dataclass
class Pass:
    pattern: re.Pattern
    fn: Callable[[ParseState, logging.Logger], None]
    name: Optional[str] = None

class IfType(Enum):
    UNKNOWN = -1
    IF = 1
    ELSEIF = 2
    ELSE = 3
    SIMPLE = 4


class FlatIfs:
    def __init__(self, start, end, cond, kind):
        self.start_ln: int = start
        self.end_ln: int = end
        self.condition: Expression = cond
        self.kind: IfType = kind
        self.nml_vars: dict[str, NameList] = {}

    def __str__(self):
        return f"{self.kind.name} L{self.start_ln}-{self.end_ln} {self.condition}"

class PassManager:
    """
    Class for managing regex passes to modify_file
    """

    def __init__(self, logger):
        self.passes: list[Pass] = []
        self.logger: Logger = logger

    def add_pass(
        self,
        pattern: re.Pattern,
        fn: Callable[[ParseState, Logger], None],
        name: Optional[str] = None,
    ):
        self.passes.append(Pass(pattern, fn, name))

    def remove_pass(self, name: str):
        self.passes = [p for p in self.passes if p.name != name]

    def run(self, state: ParseState):
        for full_line, _ in state.line_it:
            # ln in LineTuple always points to original loc. line_it.i is cpp_ln if applicable
            # seems a little circuitous but makes state management easy
            start_index = state.line_it.start_index
            orig_ln = state.line_it.get_start_ln()
            status = state.line_it.lines[start_index].commented
            if not full_line or status:
                continue
            state.curr_line = LineTuple(line=full_line, ln=orig_ln)
            for p in self.passes:
                if p.pattern.search(full_line):
                    p.fn(state, self.logger)
                    break  # first match wins

class NameList:
    def __init__(self) -> None:
        """
        Class to hold infomation on namelist variables:
        * self.name : namelist name
        * self.group : the group the namelist variable belongs to
        * self.if_blocks : list of number lines that if statments where the namelist variable is present in
        * self.variable : a pointer to a Variable class
        * self.filepath : file where namelist variable was found
        """
        self.name: str = ""
        self.group: str = ""
        self.if_blocks: list[FlatIfs] = []
        self.variable: Optional[Variable] = None
        self.ln: int = -1
        self.filepath: str = ""

    def __str__(self) -> str:
        type_str = self.variable.type if self.variable else ""
        val = self.variable.default_value if self.variable else "N/A"
        return f"nml: {type_str} {self.name} {self.group} {val}"

    def __repr__(self) -> str:
        type_str = self.variable.type if self.variable else ""
        return f"Namelist(type={type_str},name={self.name},group={self.group})"

    def __eq__(self,other) -> bool:
        return self.name == other.name


class Precedence(Enum):
    _ = 0
    LOWEST = 1
    EQUALS = 2
    LESSGREATER = 3
    SUM = 4
    PRODUCT = 5
    PREFIX = 6
    BOUNDS = 7
    CALL = 8

