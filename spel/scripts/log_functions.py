import pickle


def current_state(case):
    print(f"Current_state of CASE {case}")
    return


def center_print(name: str, pad_char="_") -> str:
    import shutil

    columns, rows = shutil.get_terminal_size()
    pad_length = int((columns - len(name)) / 2) if len(name) < columns else 0
    pad = pad_char * pad_length
    remainder = pad_char * int(columns % 2)
    print_string = f"{pad}{name}{pad}{remainder}"
    return print_string


def list_print(my_list, groups):
    print_string = "\n".join(
        " ".join(my_list[i : i + groups]) for i in range(0, len(my_list))
    )
    return print_string
