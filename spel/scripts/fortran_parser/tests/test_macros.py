"""
CPP conditional blocks inside routine bodies: #ifdef/#ifndef/#if, with
optional #elif/#else branches. #if/#elif conditions are kept as raw text
(they are C preprocessor expressions, not Fortran).
"""

from spel.scripts.fortran_parser.spel_ast import (
    IfConstruct,
    MacroIf,
    SubCallStatement,
)
from spel.scripts.fortran_parser.tests.test_Parser import parse_statements


def only_macro(txt: str) -> MacroIf:
    program = parse_statements(txt)
    stmts = [s for s in program.statements]
    assert len(stmts) == 1, [str(s) for s in stmts]
    assert isinstance(stmts[0], MacroIf)
    return stmts[0]


def test_if_defined():
    # lnd_comp_mct::lnd_run_mct
    stmt = only_macro(
        """#if (defined _memtrace)
    if(masterproc) then
       lbnum=1
       call memmon_dump_fort('memmon.out','lnd_run_mct:start::',lbnum)
    endif
#endif"""
    )
    assert stmt.condition == "(defined _memtrace)"
    assert stmt.symbol is None
    assert len(stmt.body.statements) == 1
    assert isinstance(stmt.body.statements[0], IfConstruct)
    assert stmt.branches == []


def test_ifdef_else():
    stmt = only_macro(
        """#ifdef have_moab
    call moab_init(x)
#else
    call mct_init(x)
    y = 2
#endif"""
    )
    assert stmt.symbol == "have_moab"
    assert stmt.condition is None
    assert isinstance(stmt.body.statements[0], SubCallStatement)
    assert len(stmt.branches) == 1
    else_branch = stmt.branches[0]
    assert else_branch.condition is None
    assert len(else_branch.body.statements) == 2


def test_if_elif_else():
    stmt = only_macro(
        """#if (dims==0)
    x = 0
#elif (dims==1)
    x = 1
#elif defined(two) && defined(three)
    x = 2
#else
    x = 3
#endif"""
    )
    assert stmt.condition == "(dims==0)"
    assert [b.condition for b in stmt.branches] == [
        "(dims==1)",
        "defined(two) && defined(three)",
        None,
    ]
    assert all(len(b.body.statements) == 1 for b in stmt.branches)


def test_nested_and_followed_by_statements():
    program = parse_statements(
        """#ifndef cpl_bypass
#ifdef have_moab
    call a(x)
#else
    call b(x)
#endif
#endif
    y = x"""
    )
    assert len(program.statements) == 2
    outer = program.statements[0]
    assert isinstance(outer, MacroIf) and outer.symbol == "cpl_bypass"
    inner = outer.body.statements[0]
    assert isinstance(inner, MacroIf) and len(inner.branches) == 1


def test_macro_inside_fortran_if():
    program = parse_statements(
        """if (x > 0) then
#if (defined _openmp)
    n = omp_get_num_threads()
#else
    n = 1
#endif
end if"""
    )
    assert len(program.statements) == 1
    ifc = program.statements[0]
    assert isinstance(ifc, IfConstruct)
    macro = ifc.consequence.statements[0]
    assert isinstance(macro, MacroIf) and len(macro.branches) == 1
