from .calltree import Node


def build_tree_html(tree: list[Node]):
    """Recursive function to build HTML for a tree."""
    html = '<ul id="SubTree">'
    for node in tree:
        html += process_node(node)
    html += "</ul>"
    return html


def process_node(node: Node):
    """
    Function to tranlate Node("name":name,"children":[])
    to html
    """
    html = ""

    if node.children:
        html += f'<li class="node"><span class="toggler checkbox" aria-expanded="false">{node.name}</span>'
        html += '<ul class="child">'
        for child in node.children:
            html += process_node(child)
        html += "</ul>"
        html += add_details_btn(node.name)
        html += "</li>"
    else:
        html += f'<li class="node"><span class="parent">{node.name}</span>'
        html += add_details_btn(node.name)
        html += "</li>"

    return html


def add_details_btn(name: str) -> str:
    """
    Adds html for button that sends requests via htmx
    """
    html = f'<button class="details-btn" aria-label="View Details" hx-get="/subroutine-details/{name}/" '
    html += 'hx-target="#right-panel" hx-trigger="click" hx-swap="innerHTML">'
    html += "</button>"
    return html
