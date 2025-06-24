import itertools


def ast_to_plantuml(ast: dict) -> str:
    lines = []
    counter = itertools.count()
    id_map = {}

    def next_id():
        return f"node_{next(counter)}"

    def visit(node, parent_id=None):
        if not isinstance(node, dict):
            return None

        this_id = next_id()
        id_map[id(node)] = this_id

        label = node.get("Node", "Unknown")
        content = ""
        if label == "FuncExpression":
            label = f'Func: {node["Func"]}'
            args = ", ".join(
                arg.get("Val", "?") if arg.get("Node") == "Ident" else "expr"
                for arg in node.get("Args", [])
            )
            content = f"Args: {args}"

        elif label == "InfixExpression":
            label = "InfixExpr:"
            content = f'Op: {node["Op"]}'
        elif label == "Ident":
            label = f"Identifier: {str(node["Val"])}"
        elif label == "FloatLiteral":
            label = f"Literal: {str(node["Val"])}"
        elif label == "IntLiteral":
            label = f"Literal: {str(node["Val"])}"
        elif label == "SubCallStatement":
            label = "SubCallStatement"
            content = ""

            # Combine label and content in a rectangle (no UML icon)
        if content:
            lines.append(f'rectangle "{label}\\n{content}" as {this_id}')
        else:
            lines.append(f'rectangle "{label}" as {this_id}')

        if parent_id:
            lines.append(f"{parent_id} --> {this_id}")

        # Handle children
        if node.get("Node") == "FuncExpression":
            for arg in node.get("Args", []):
                visit(arg, this_id)
        elif node.get("Node") == "InfixExpression":
            visit(node["Left"], this_id)
            visit(node["Right"], this_id)
        elif node.get("Node") == "SubCallStatement":
            visit(node["Sub"], this_id)

        return this_id

    # Start the traversal
    visit(ast)

    # Wrap in @startuml block
    return "\n".join(
        [
            "@startuml",
            "hide stereotype",
            "skinparam rectangle {",
            "  BackgroundColor white",
            "  BorderColor black",
            "  FontColor black",
            "  ShowStereotype false",
            "}",
            *lines,
            "@enduml",
        ]
    )
