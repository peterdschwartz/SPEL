import difflib
import linecache
import re
import sys
from os import name

from .analyze_ifs import flatten


def single_line_unwrapper(filename, ct, verbose=False):
    """
    Function that takes code segment that has line continuations
    and returns it all on one line.
    """

    full_line = ""
    while True:
        current_line = linecache.getline(filename, ct)
        if bool(re.search(r"^\s*!", current_line)):
            ct += 1
            continue

        current_line = current_line.split("!")[0]
        current_line = current_line.rstrip("\n")
        # continuation = current_line.endswith("&")
        continuation = bool(re.search(r"&", current_line))
        current_line = re.sub(r"&", " ", current_line)

        if current_line.isspace() or not current_line:
            ct += 1
            continue
        current_line = current_line.strip()
        full_line += current_line
        if not continuation:
            break
        ct += 1
    return full_line, ct


def binary_search(iterable, ln, r=0):
    """
    Search for if_block within based on a line number
    """
    left, right = 0, len(iterable) - 1

    while left <= right:
        mid = (left + right) // 2
        if r == 0:
            if iterable[mid].start <= ln and ln <= iterable[mid].end:
                return mid
        elif r == 1:
            if iterable[mid].start <= ln and ln <= iterable[mid].relative_end:
                return mid
        if iterable[mid].start < ln:
            left = mid + 1
        else:
            right = mid - 1
    else:
        return -1


def get_default_value(filename, line_number):
    line, _ = single_line_unwrapper(filename, line_number)
    default = re.findall(r".*=\s*(.*)", line)
    return default[0] if default else ""


def insert_default(mod_dict):
    """
    Insert default values of for each variable
    """
    for mod in mod_dict.values():
        filename = mod.filepath
        for gv in mod.global_vars:
            gv.default_value = get_default_value(filename, gv.ln)


def in_relative(node, ln):
    return node.start <= ln and ln <= node.relative_end


def find_elm(node, ln, namelist_dict):
    alive = node.default
    if not alive:
        ea = False

        for i in node.elseif:
            if i.default and i.start <= ln and ln <= i.relative_end:
                ea = True
                break
        index = binary_search(node.elseif, ln, r=0)

        if ea:
            elseif = node.elseif[index]
            if in_relative(elseif, ln):
                index = binary_search(elseif.children, ln, r=0)

                if index == -1:
                    return True
                else:
                    elseif_child = elseif.children[index]
                    return find_elm(elseif_child, ln, namelist_dict)
        else:
            if node.elses and node.elses.default:
                elses = node.elses
                if in_relative(elses, ln):
                    index = binary_search(elses.children, ln, r=0)
                    if index == -1:
                        return True
                    else:
                        else_child = elses.children[index]
                        return find_elm(else_child, ln, namelist_dict)

    else:
        if in_relative(node, ln):
            index = binary_search(node.children, ln, r=0)
            if index == -1:
                return True
            else:
                child = node.children[index]
                return find_elm(child, ln, namelist_dict)
    return False


def elmtypes(sub_dict, ifs, subroutine, namelist_dict, noif):
    """
    ["Allocation1_PlantNPDemand",
     "Allocation2_ResolveNPLimit",
     "Allocation3_PlantCNPAlloc",]:

     Separates the elm_variables by alive/dead branches

    """
    deadBranch = {}
    aliveBranch = {}

    for elm in sub_dict[subroutine].elmtype_access_by_ln.keys():
        aliveBranch[elm] = []
        deadBranch[elm] = []

        for e in sub_dict[subroutine].elmtype_access_by_ln[elm]:
            parent = ifs[binary_search(ifs, e.ln, r=0)]

            p = find_elm(parent, e.ln, namelist_dict)
            if p:
                aliveBranch[elm].append(e)

            else:
                deadBranch[elm].append(e)

        if not aliveBranch[elm]:
            aliveBranch.pop(elm)
        if not deadBranch[elm]:
            deadBranch.pop(elm)
    for i in noif:
        aliveBranch.setdefault(i, []).extend(noif[i])
    return aliveBranch, deadBranch


def subroutineCalls(sub_dict, parent_ifs, subroutine):
    flat = flatten(parent_ifs)
    aliveBranch = {}
    for calls in sub_dict[subroutine].child_Subroutine.keys():
        for sub in sub_dict[subroutine].child_Subroutine[calls].subroutine_call:
            if flat[binary_search(flat, sub.ln, r=1)].default:
                aliveBranch.setdefault(calls, []).append(sub)
    return aliveBranch


def generate_html_comparison(file1, file2, output_path="comparison.html"):
    html_content = """
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                vertical-align: top;
                white-space: pre-wrap;  
            }}
            th {{
                background-color: #f2f2f2;
                text-align: left;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .missing {{
                background-color: #ffdddd;  
                color: #a00;
            }}
        </style>
    </head>
    <body>
        <h2>File Comparison</h2>
        <table>
            <tr>
                <th>File1</th>
                <th>File2</th>
            </tr>
    """
    for i, j in zip(file1, file2):
        if j == "" and i != "":
            html_content += f"""
                <tr>
                    <td class="missing">{i}</td>
                    <td>{j}</td>
                </tr>
            """
        elif i == "" and j != "":
            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td class="missing">{j}</td>
                </tr>
            """
        else:
            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{j}</td>
                </tr>
            """

    html_content += """
        </table>
    </body>
    </html>
    """

    # Write to output file
    with open(output_path, "w") as output_file:
        output_file.write(html_content)

    print(f"HTML comparison saved to {output_path}")


def stagger_diff(lines1, lines2):
    d = difflib.ndiff(lines1, lines2)

    staggered_lines_1 = []
    staggered_lines_2 = []

    for line in d:
        if line.startswith(" "):
            staggered_lines_1.append(line[2:])
            staggered_lines_2.append(line[2:])
        elif line.startswith("-"):
            staggered_lines_1.append(line[2:])
            staggered_lines_2.append("")
        elif line.startswith("+"):
            staggered_lines_1.append("")
            staggered_lines_2.append(line[2:])

    # for l1, l2 in zip(staggered_lines_1, staggered_lines_2):
    #     print(f"{l1.strip():<40} | {l2.strip()}")

    return staggered_lines_1, staggered_lines_2


def test_equality(node1, node2):
    if node1.start != node2.start:
        print("Differs at start values")
        print(f"Node 1 start: {node1.start}")
        print(f"Node 2 start: {node2.start}")
        return False
    if node1.end != node2.end:
        print("Differs at end values")
        print(f"Node 1 end: {node1.end}")
        print(f"Node 2 end: {node2.end}")
        return False
    if node1.relative_end != node2.relative_end:
        print("Differs at relative_end values")
        print(f"Node 1 relative_end: {node1.relative_end}")
        print(f"Node 2 relative_end: {node2.relative_end}")
        return False
    if node1.default != node2.default:
        print("Differs at default values")
        print(f"Node 1 condition {node1.condition}: {node1.default}")
        print(f"Node 2 condition {node2.condition}: {node2.default}")
        return False
    if node1.kind != node2.kind:
        print("Differs at type values")
        print(f"Node 1 type {node1.kind}")
        print(f"Node 2 type {node2.kind}")

        return False
    if node1.depth != node2.depth:
        print("Differs at depth values")
        print(f"Node 1 depth {node1.depth}")
        print(f"Node 2 depth {node2.depth}")

        return False
    if node1.condition != node2.condition:
        return False
    for i, j in zip(node1.calls, node2.calls):
        if i != j:
            return False

    if node1.parent and node2.parent:
        if node1.parent.start != node2.parent.start:
            print("Differs at parent.start values")
            print(f"Node 1 parent.start {node1.parent.start}")
            print(f"Node 2 parent.start {node2.parent.start}")
            return False
    elif (not node1.parent and node2.parent) or (not node2.parent and node1.parent):
        return False

    return True


def cross_test(list1, list2):
    flat1 = flatten(list1)
    flat2 = flatten(list2)

    for i, j in zip(flat1, flat2):
        if not test_equality(i, j):
            print(f"different at list1: {i.start} list2: {j.start}")


def insert_stop(filename, parent_ifs):
    flat = flatten(parent_ifs)
    lines = []
    for block in flat:
        if not block.default:
            lines.append(block.start)

    with open(filename, "r") as file:
        content = file.readlines()

    count = 0
    for line in sorted(lines, reverse=True):
        if_statement = content[line]
        indent_level = len(if_statement) - len(if_statement.lstrip())
        tab = " " * indent_level
        content.insert(line + 1, f"{tab}STOP 'BRANCH SHOULD BE DEAD {line+1}'\n")

    with open(f"{filename.split('.')[0]}_COPY.F90", "w+") as file:
        content = "".join(content)
        file.write(content)

    return sorted(lines)


def generate_noifs(sub_dict, subroutine, parent_ifs):
    """
    Returns dictionary of namelist variables that are outisde
    if blocks and assumed to be active
    """
    noif = {}
    flat = flatten(parent_ifs)
    for key in sub_dict[subroutine].elmtype_access_by_ln.keys():
        for k in sub_dict[subroutine].elmtype_access_by_ln[key]:
            index = binary_search(flat, k.ln, r=1)

            if index == -1:
                noif.setdefault(key, []).append(k)
    return noif
