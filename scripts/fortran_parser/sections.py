import re
import sys
from typing import Optional

from scripts.fortran_parser.lexer import Lexer
from scripts.fortran_parser.spel_ast import Statement
from scripts.fortran_parser.spel_parser import Parser
from scripts.fortran_parser.tracing import Trace
from scripts.types import LineTuple, LogicalLineIterator


def parse_blocks(
    lines: list[LineTuple],
    regex_start: re.Pattern,
    regex_end: re.Pattern,
    regex_check: Optional[re.Pattern] = None,
    verbose: bool = False,
    tag: str = "",
) -> list[Statement]:
    """
    parses given lines for any blocks designated by the regex
    """
    line_it = LogicalLineIterator(lines, "parse_blocks")

    if verbose:
        Trace.enabled = True

    log_name = "Parser" if not tag else tag
    statements: list[Statement] = []
    for fline in line_it:
        full_line = fline.line
        start_ln = line_it.get_start_ln()
        m_start = regex_start.search(full_line)

        if m_start:
            check = regex_check.search(full_line) if regex_check else True
            if check:
                block, _ = line_it.consume_until(regex_end, regex_check)
            else:
                block = [LineTuple(full_line, ln=start_ln)]

            if verbose:
                str_ = "\n".join([l.line for l in block])
                line_it.logger.warning(str_)
                sys.exit(0)

            blk_iter = LogicalLineIterator(lines=block, log_name="parse_section")
            lexer = Lexer(blk_iter)
            parser = Parser(lexer, log_name)
            program = parser.parse_program()
            statements.extend(program.statements)

    return statements
