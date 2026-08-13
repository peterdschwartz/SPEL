from namelist import find_ifs_2, print_if_structure

color_map = [
    "#F0F8FF",
    "#FAEBD7",
    "#00FFFF",
    "#7FFFD4",
    "#F0FFFF",
    "#F5F5DC",
    "#FFE4C4",
    "#000000",
    "#FFEBCD",
    "#0000FF",
    "#8A2BE2",
    "#A52A2A",
    "#DEB887",
    "#5F9EA0",
    "#7FFF00",
    "#D2691E",
    "#FF7F50",
    "#6495ED",
    "#FFF8DC",
    "#DC143C",
    "#00FFFF",
    "#00008B",
    "#008B8B",
    "#B8860B",
    "#A9A9A9",
    "#006400",
    "#A9A9A9",
    "#BDB76B",
    "#8B008B",
    "#556B2F",
    "#FF8C00",
    "#9932CC",
    "#8B0000",
    "#E9967A",
    "#8FBC8F",
    "#483D8B",
    "#2F4F4F",
    "#2F4F4F",
    "#00CED1",
    "#9400D3",
    "#FF1493",
    "#00BFFF",
    "#696969",
    "#696969",
    "#1E90FF",
    "#B22222",
    "#FFFAF0",
    "#228B22",
    "#FF00FF",
    "#DCDCDC",
    "#F8F8FF",
    "#FFD700",
    "#DAA520",
    "#808080",
    "#008000",
    "#ADFF2F",
    "#808080",
    "#F0FFF0",
    "#FF69B4",
    "#CD5C5C",
    "#4B0082",
    "#FFFFF0",
    "#F0E68C",
    "#E6E6FA",
    "#FFF0F5",
    "#7CFC00",
    "#FFFACD",
    "#ADD8E6",
    "#F08080",
    "#E0FFFF",
    "#FAFAD2",
    "#D3D3D3",
    "#90EE90",
    "#D3D3D3",
    "#FFB6C1",
    "#FFA07A",
    "#20B2AA",
    "#87CEFA",
    "#778899",
    "#778899",
    "#B0C4DE",
    "#FFFFE0",
    "#00FF00",
    "#32CD32",
    "#FAF0E6",
    "#FF00FF",
    "#800000",
    "#66CDAA",
    "#0000CD",
    "#BA55D3",
    "#9370DB",
    "#3CB371",
    "#7B68EE",
    "#00FA9A",
    "#48D1CC",
    "#C71585",
    "#191970",
    "#F5FFFA",
    "#FFE4E1",
    "#FFE4B5",
    "#FFDEAD",
    "#000080",
    "#FDF5E6",
    "#808000",
    "#6B8E23",
    "#FFA500",
    "#FF4500",
    "#DA70D6",
    "#EEE8AA",
    "#98FB98",
    "#AFEEEE",
    "#DB7093",
    "#FFEFD5",
    "#FFDAB9",
    "#CD853F",
    "#FFC0CB",
    "#DDA0DD",
    "#B0E0E6",
    "#800080",
    "#663399",
    "#FF0000",
    "#BC8F8F",
    "#4169E1",
    "#8B4513",
    "#FA8072",
    "#F4A460",
    "#2E8B57",
    "#FFF5EE",
    "#A0522D",
    "#C0C0C0",
    "#87CEEB",
    "#6A5ACD",
    "#708090",
    "#708090",
    "#FFFAFA",
    "#00FF7F",
    "#4682B4",
    "#D2B48C",
    "#008080",
    "#D8BFD8",
    "#FF6347",
    "#40E0D0",
    "#EE82EE",
    "#F5DEB3",
    "#FFFFFF",
    "#F5F5F5",
    "#FFFF00",
    "#9ACD32",
]
color_map = [
    # "#ff4000",
    # "#ff8000",
    # "#ffbf00",
    "#ffff00",
    "#00ffff",
    "#ff00ff",
    "#80ff00",
    "#bfff00",
    "#40ff00",
    "#00ff00",
    "#00ff40",
    "#00ff80",
    "#00ffbf",
    "#00bfff",
    "#0080ff",
    "#0040ff",
    "#0000ff",
    "#4000ff",
    "#8000ff",
    "#bf00ff",
    "#ff00bf",
    "#ff0080",
    "#ff0040",
    "#ff0000",
]
color_dict = {}


def generate_html(nodes, filename):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; }
            .node { margin-left: 20px; }
        </style>
    </head>
    <body>
    """

    # Add each top-level node to the HTML content
    for node in nodes:
        html_content += format_node(node)

    html_content += format_code(filename)
    html_content += "</body></html>"

    with open("output.html", "w") as f:
        f.write(html_content)

    print("Generated HTML with depth-based highlighting.")


def format_node(node):
    # Get the node's depth and condition
    depth = node[0]
    block = node[1]

    prefix = ""
    match block.kind:
        case 1:
            prefix = "if"
        case 2:
            prefix = "else if"
        case 3:
            prefix = "else"
    condition = prefix + " " + block.condition
    color_dict[(block.start, block.relative_end)] = color_map[depth]
    # Build the HTML with a highlighted condition using the <mark> tag
    # Create a container div with indentation based on depth
    html = f'<div style="margin-left: {depth * 20}px;">'
    html += (
        f"    <mark style='background-color: {color_map[depth]};'>{condition}</mark>"
    )
    html += "</div>\n"  # Close the div

    # Create an unordered list with bullet points
    html += f'<div style="margin-left: {depth * 20 + 30}px;">'
    html += f'    <li><mark style="background-color: {color_map[depth]};">'
    html += f"Start: {block.start} end: {block.end} relative_end: {block.relative_end}</mark>"
    html += "</div>\n"
    # html += format_code(filename)
    return html
    # Start: {block.start} End: {block.end}


def format_code(filename):
    with open(filename, "r") as file:
        content = file.readlines()

    # Update the CSS for the vertical bar
    html = """
    <style>
        .vl {
            border-left: 2px solid black;
            height: 500px;
            position: absolute;
            left: 50%;
            margin-left: -1px; /* Adjusted to center the bar */
            top: 10px;
        }
    </style>
    <div class="vl">    """
    html += "<pre>"
    for ln in color_dict.keys():
        start, rel_end = ln[0], ln[1] + 1
        html += f"---{start} {rel_end-1}---\n"
        html += f'<mark style="background-color:{color_dict[ln]};">'
        html += "".join(content[start:rel_end])  # Removed unnecessary newline joining
        html += "</mark>"
    html += "</pre>"
    html += "</div> <!-- Vertical bar div -->"
    return html


def flatten(node, depth, visited, res):
    if node:
        if node.start not in visited:
            res.append((depth, node))
            visited.append(node.start)

        for i in node.elseif:
            flatten(i, depth, visited, res)
        for i in node.children:
            flatten(i, depth + 1, visited, res)

        if node.elses:
            flatten(node.elses, depth, visited, res)

    return res


def run(filename):
    file = open(filename, "r")
    r = file.readlines()
    """  FIX """
    blocks = find_ifs_2(r)
    # print_if_structure(blocks)
    total = []
    for block in blocks:
        total.extend(flatten(node=block, depth=0, visited=[], res=[]))
    flattened_list = sorted(total, key=lambda x: x[1].start)

    # for i in flattened_list:
    #     print("->" * i[0], i[1].start, i[1].condition, i[1].relative_end)
    # nodes = format_node(flattened_list)
    generate_html(flattened_list, filename)


