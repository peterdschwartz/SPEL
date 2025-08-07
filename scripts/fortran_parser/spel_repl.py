def parse_line(user_input: str):
    import scripts.fortran_parser.lexer as lexer
    import scripts.fortran_parser.spel_parser as sp
    import scripts.types as t_
    from scripts.fortran_parser.tracing import Trace

    Trace.enabled = True
    ln = 0
    ln += 1
    line = [t_.LineTuple(line=user_input, ln=ln)]

    lex = lexer.Lexer(t_.LogicalLineIterator(line))
    parser = sp.Parser(lex=lex)

    program = parser.parse_program()
    for stmt in program.statements:
        print(stmt)
        print(stmt.to_dict())
