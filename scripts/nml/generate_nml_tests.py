import os
import random
import subprocess as sp
from string import ascii_lowercase

from analyze_subroutines import Subroutine
from helper_functions import ReadWrite

import scripts.nml.analyze_elm as ae
import scripts.nml.analyze_ifs as ai

# Declare ln as a global variable
ln = 0


def generate_condition():
    """Generate a random condition."""
    conditions = [
        (".true.", True),
        (".false.", False),
        (".not. .true.", False),
        (".not. .false.", True),
    ]
    return random.choice(conditions)


def insert_variables():
    variables = [i for i in ascii_lowercase]
    return random.choice(variables)


def generate_fortran_if(depth, max_depth, prev, stack, alive, dead):
    global ln
    if depth >= max_depth:
        return ""

    rando_condo = generate_condition()
    cond = rando_condo[1] and prev
    code = f"if ({rando_condo[0]}) then !{cond} \n\n"
    ln += 2

    rando_var = insert_variables()
    rando_var2 = insert_variables()
    choice = (
        f"{rando_var} = {str(rando_var2)}" if random.random() < 1 else "do_something"
    )
    if choice != "do_something":
        stack.append((rando_var, ln))
        stack.append((rando_var2, ln))
        if cond:
            alive.append(rando_var)
            alive.append(rando_var2)
        else:
            dead.append(rando_var)
            dead.append(rando_var)

    code += f"{choice}\n\n"
    ln += 2
    num_inner_blocks = random.randint(0, 2)
    for _ in range(num_inner_blocks):
        if random.random() < 0.7:  # 70% chance to add nested blocks
            code += f"{generate_fortran_if(depth + 1, max_depth, rando_condo[1] and prev, stack, alive,dead) }"

    elseif_present = False
    elseif_cond = False
    if random.random() < 0.5:
        condo_rando = generate_condition()
        elseif_present = True
        elseif_cond = condo_rando[1]
        cond = not (prev and rando_condo[1]) and condo_rando[1]
        code += f"else if ({condo_rando[0]}) then !{cond} \n\n"
        ln += 2
        rando_var = insert_variables()
        rando_var2 = insert_variables()
        choice = (
            f"{rando_var} = {str(rando_var2)}"
            if random.random() < 1
            else "do_something"
        )
        if choice != "do_something":
            stack.append((rando_var, ln))
            stack.append((rando_var2, ln))
            if cond:
                alive.append(rando_var)
                alive.append(rando_var2)
            else:
                dead.append(rando_var)
                dead.append(rando_var2)
        code += f"{choice}\n\n"
        ln += 2
    if random.random() < 0.5:
        if elseif_present:
            odnar_odanc = elseif_cond or rando_condo[1]
        else:
            odnar_odanc = rando_condo[1]

        code += f"else !{not odnar_odanc}\n\n"
        ln += 2
        rando_var = insert_variables()
        rando_var2 = insert_variables()
        choice = (
            f"{rando_var} = {str(rando_var2)}"
            if random.random() < 1
            else "do_something"
        )
        if choice != "do_something":
            stack.append((rando_var, ln))
            stack.append((rando_var2, ln))
            if not odnar_odanc:
                alive.append(rando_var)
                alive.append(rando_var2)
            else:
                dead.append(rando_var)
                dead.append(rando_var2)
        code += f"{choice}\n\n"
        ln += 2

    code += f"endif\n\n"
    ln += 2
    if random.random() < 0.5:
        rando_var = insert_variables()
        rando_var2 = insert_variables()
        choice = (
            f"{rando_var} = {str(rando_var2)}"
            if random.random() < 1
            else "do_something"
        )
        if choice != "do_something":
            stack.append((rando_var, ln))
            stack.append((rando_var2, ln))
            if prev:
                alive.append(rando_var)
                alive.append(rando_var2)
            else:
                dead.append(rando_var)
                dead.append(rando_var2)

        code += f"{choice}\n\n"
        ln += 2

    return code


def generate_fortran_program(num_blocks, max_depth):
    """Generate a complete Fortran program with randomly nested if blocks."""
    global ln
    program = "program test\nimplicit none\ninteger :: "
    for i in ascii_lowercase:
        program += f"{i}, "

    program = program[:-2]
    program += "\n"
    stack = []
    alive = []
    dead = []
    ln = 3

    for _ in range(num_blocks):
        program += generate_fortran_if(
            depth=0, max_depth=max_depth, prev=True, stack=stack, alive=alive, dead=dead
        )

        program += "\n"
        ln += 1

    program += "print '(A)', 'Done'\nend program test \n "
    subroutine = "test"
    sub_dict = {
        subroutine: Subroutine(
            name=subroutine,
            calltree=[],
            file="test",
            start=-999,
            end=-999,
            lib_func=True,
        )
    }
    """
    test_subroutine = "test"
    test_sub_dict = {
        test_subroutine: Subroutine(
            name=test_subroutine,
            calltree=[],
            file="test",
            start=-999,
            end=-999,
            lib_func=True,
        )
    }
    """
    prompt = """
### CODE TO REPLACE WITH ###
test_sub_dict[test_subroutine].elmtype_access_by_ln = {
    """
    sub_dict[subroutine].elmtype_access_by_ln = {}
    for i in stack:
        var = i[0]
        ln = i[1]
        sub_dict[subroutine].elmtype_access_by_ln.setdefault(var, []).append(
            ReadWrite(status="w", ln=ln)
        )

    for i in set([s[0] for s in stack]):
        prompt += f"'{i}' : \n{'':<8}["
        for j in sub_dict[subroutine].elmtype_access_by_ln[i]:
            prompt += f"ReadWrite(status='w',ln={j.ln}),\n{'':<8} "
        prompt += f"],\n{'':<4}"
    prompt += "}"

    return program, sub_dict, prompt


def main(num_blocks, max_depth):
    fortran_code, sub_dict, prompt = generate_fortran_program(
        num_blocks=num_blocks, max_depth=max_depth
    )
    with open("analysis/unitTest.F90", "w") as file:
        file.write(fortran_code)

    parent_ifs = ai.run("analysis/unitTest.F90")
    ai.set_default(parent_ifs, {})
    noif = ae.generate_noifs(sub_dict, "test", parent_ifs)

    _, _ = ae.elmtypes(sub_dict, parent_ifs, "test", {}, noif)

    ae.insert_stop("analysis/unitTest.F90", parent_ifs)

    output = sp.getoutput(
        "gfortran analysis/unitTest.F90 -o analysis/unitTest && ./analysis/unitTest"
    )
    errorFiles = 0
    if output != "Done":
        print(f"error: analysis/errors/errorFile_{errorFiles}")
        os.makedirs("analysis/errors", exist_ok=True)
        with open(f"analysis/errors/errorFile_{errorFiles}.F90", "w") as e:
            e.write("---AssertionError: 'STOP BRANCH SHOULD BE DEAD' != 'Done'---\n")
            with open("analysis/unitTest.F90", "r") as file:
                content = file.readlines()

            for i in content:
                e.write(i)

            e.write(prompt)
            errorFiles += 1

    else:
        print("OK")
        print(prompt)
    print("---------------------------------------------")


if __name__ == "__main__":
    main(1, 1)

    """
    for i in {1..100}; do py main.py  ;  done
    """
