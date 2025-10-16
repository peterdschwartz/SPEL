import re
import sys

from scripts.analyze_subroutines import Subroutine
from scripts.config import spel_output_dir
from scripts.LoopConstructs import Kind, Loop, get_loops
from scripts.types import LineTuple, LogicalLineIterator

SUBGRID_MAP = {"p": "pft", "c": "col", "l": "lnd", "t": "top", "g": "grd"}
inv_subgrid_map = {val: key for key, val in SUBGRID_MAP.items()}


def create_map_name(loop_subgrid: str, var_subgrid: str) -> str:
    from_ = SUBGRID_MAP[loop_subgrid]
    to_ = SUBGRID_MAP[var_subgrid]
    return f"{from_}_to_{to_}_lookup"


def prune_associate_clause(sub: Subroutine):
    """
    Checks if any variables in associate clause are unused
    and removes them.
    """
    unused = [
        (ptr, target)
        for ptr, target in sub.associate_vars.items()
        if target not in sub.elmtype_access_by_ln
    ]
    if not unused:
        return

    new_lines: list[str] = []
    ifile = open(sub.filepath, "r")
    sub_start = sub.startline
    sub_end = sub.endline
    lines = ifile.readlines()

    # Store lines unrelated to this subroutine
    new_lines.extend(lines[0:sub_start])
    str_pattern_list = [rf"\s*{ptr}\s*(=>)\s*{target}\b" for ptr, target in unused]
    str_pattern = "|".join(str_pattern_list)
    regex_unused = re.compile(rf"({str_pattern})")

    for ln in range(sub_start, sub_end):
        line = lines[ln].strip().lower()
        if not regex_unused.match(line):
            new_lines.append(lines[ln])

    # add remaining lines:
    new_lines.extend(lines[sub_end:])
    with open(f"{spel_output_dir}/{sub.module}_pruned.F90", "w") as ofile:
        ofile.writelines(new_lines)
    return


def compress_on_filter(sub: Subroutine):
    """ """
    get_loops(sub)
    logger = sub.logger
    regex_bounds = re.compile(r"(bounds%beg|bounds%end)(g|l|t|c|p)")

    arrays = {
        key: val
        for key, val in sub.local_variables.items()
        if val.dim > 0 and regex_bounds.search(val.bounds)
    }
    loops_with_arrs: dict[str, list[Loop]] = {
        var: [loop for loop in sub.loops if var in loop.local_vars] for var in arrays
    }
    # for loop in sub.loops:
    #     logger.info(f"filter: {loop.filter}")

    compressible: dict[str, str] = {}
    for var, loops in loops_with_arrs.items():
        filter_set: set[str] = {f[:-1] for loop in loops for f in loop.filter}
        if len(filter_set) == 1:
            compressible[var] = filter_set.pop()

    adjust_array_access_and_allocation(sub, compressible)
    return


def adjust_array_access_and_allocation(
    sub: Subroutine,
    compressible: dict[str, str],
):
    ifile = open(sub.filepath, "r")
    lines = ifile.readlines()
    ifile.close()
    flines = [LineTuple(line=line, ln=i) for i, line in enumerate(lines)]
    regex_bounds = re.compile(r"(bounds%beg|bounds%end)(g|l|t|c|p)")
    local_vars = sub.local_variables

    line_it = LogicalLineIterator(flines)

    new_indices: set[str] = set()
    # new_mappings: dict[str, set[int]] = {}

    for varname, filt in compressible.items():
        var = local_vars[varname]
        decl = var.ln
        match_bounds = regex_bounds.search(var.bounds).groups()
        var_subgrid = match_bounds[1]

        # go through loops
        adjusted = False
        for loop in sub.loops:
            accesses = loop.local_vars.get(varname, None)
            if not accesses:
                continue
            lns = [rw.ltuple.ln for rw in accesses]
            loop_subgrid, filter_ = next(iter(loop.index_map.items()))
            # can generalize this
            if loop_subgrid not in ["p", "c", "l", "t", "g"]:
                sub.logger.error(f"UNEXPECTED INDEX!! {loop_subgrid}")

            # This checks if a variable from a different subgrid is being used
            # Requires introduction of new filter mapping
            filter_idx = loop.index
            if loop_subgrid != var_subgrid:
                filter_idx = f"f{var_subgrid}"
                map_name = create_map_name(loop_subgrid, var_subgrid)
                loop.new_mappings.add(map_name)
                new_indices.add(filter_idx)
            regex_sub = re.compile(rf"\b{varname}\s*\(\s*{var_subgrid}\s*\)")
            repl_str: str = f"{varname}({filter_idx})"
            line_it.replace_in_line(lns, regex_sub, repl_str, sub.logger)
            adjusted = True

        if not adjusted:
            continue

        # Substitute declaration (assumes automatic arrays)
        repl_bounds = f"bounds%beg{var_subgrid}:bounds%end{var_subgrid}"
        new_bounds = filt.replace("filter", "num")
        new_bounds = new_bounds + var_subgrid
        if new_bounds not in sub.arguments:
            sub.logger.error(f"{new_bounds} not in argument list!")
            sys.exit(1)
        new_bounds = f"1:{new_bounds}"
        new_line = line_it.lines[decl].line.replace(repl_bounds, new_bounds)
        line_it.lines[decl].line = new_line

        # check for vector expressions
        pattern = rf"{varname}\s*\(\s*beg\w\s*:\s*end\w\s*\)|{varname}\s*\(\s*bounds%(\w+)\s*:\s*bounds%(\w+)\s*\)"
        regex = re.compile(pattern)
        line_it.i = sub.last_decl_ln + 1

        lns = [line_it.start_index for line, _ in iter(line_it) if regex.search(line)]
        if lns:
            line_it.replace_in_line(lns, regex, f"{varname}({new_bounds})", sub.logger)

    insert_mappings(line_it, sub)
    insert_decls(line_it, sub)
    # Output file:
    base_fn = sub.filepath.split("/")[-1]
    with open(f"{spel_output_dir}{base_fn}", "w") as ofile:
        new_lines = [lpair.line for lpair in line_it.lines]
        ofile.writelines(new_lines)

    return


def insert_decls(
    line_it: LogicalLineIterator,
    sub: Subroutine,
):

    return


def insert_mappings(
    line_it: LogicalLineIterator,
    sub: Subroutine,
):
    def gen_index(map_name) -> str:
        test = map_name.split("_")
        indx = inv_subgrid_map[test[2]]
        return f"f{indx}"

    offset: int = 0
    for loop in sub.loops:
        if loop.kind == Kind.dowhile or not loop.new_mappings:
            continue
        mappings = loop.new_mappings
        loop.start += offset
        line_it.i = loop.start + 1
        for lt, _ in line_it:
            if lt:
                break
        cur_line = line_it.get_curr_line()
        assert cur_line
        spacing = re.match(r"^(\s*)", cur_line).group()
        for map in mappings:
            new_indx = gen_index(map)
            new_line = f"{spacing}{new_indx} = {map}({loop.index})\n"
            line_it.insert_after(new_line)
            offset += 1

    return
