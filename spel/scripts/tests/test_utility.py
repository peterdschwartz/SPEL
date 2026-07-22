import re
import textwrap

from spel.scripts.edit_files import apply_comments
from spel.scripts.types import LineTuple, LogicalLineIterator


def test_line_iterator():

    file_test = textwrap.dedent(
        """
        subroutine sub_name()
            integer :: x
            x = 1234
           call SUB(x &
                , y &
                ! optional third argument:
                , z) 
            ! use z for stuff!
            x = y + z ! stuf
        end subroutine sub_name
        """
    )
    test_lines: list[LineTuple] = [
        LineTuple(line=line, ln=i) for i, line in enumerate(file_test.splitlines())
    ]

    regex_sub = re.compile(r"^\s*(subroutine)\s+")
    it = LogicalLineIterator(test_lines)
    for unwrap, new_ln in it:
        start = new_ln
        if regex_sub.search(unwrap):
            _, _ = it.consume_until(re.compile(r"^(end\s+subroutine)"))
            it.comment_cont_block(start)

    print(test_lines)
    test_lines = apply_comments(test_lines)
    for l in test_lines:
        print(l.line)
