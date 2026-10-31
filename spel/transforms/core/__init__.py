"""
Shared infrastructure of SPEL's source transforms (see ../DESIGN.md):

- `source`: logical statements, comments and continuations of fixed-up Fortran text
- `edits`: line edits collected per file and applied together against the original lines
- `fortran`: emitters for directives, wrapped statements and small constructs
- `query`: questions transforms ask the analysis (routines, statements, loops, accesses)
- `transform`, `driver`, `sink`, `registry`: the Transform protocol and how it is run
"""
