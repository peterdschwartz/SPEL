"""
Expected read/write accesses for subroutines in example_functions.F90.

Accesses are written as R/W/RW(<snippet>) where <snippet> is a piece of Fortran
source that appears on exactly one line of example_functions.F90. The snippet
is resolved to its (1-based) line number when this module is imported, so the
expectations survive lines being added/removed elsewhere in the file.
"""

from pathlib import Path

SOURCE_FILE = Path(__file__).with_name("example_functions.F90")
_SOURCE_LINES = SOURCE_FILE.read_text().splitlines()

elmtypes = "elmtypes"
args = "args"


def line_of(snippet: str) -> int:
    """Return the 1-based line number of the unique line containing `snippet`."""
    matches = [i + 1 for i, line in enumerate(_SOURCE_LINES) if snippet in line]
    if len(matches) != 1:
        raise ValueError(
            f"Snippet {snippet!r} must match exactly one line of "
            f"{SOURCE_FILE.name}, matched lines: {matches}"
        )
    return matches[0]


def R(snippet: str) -> str:
    return f"r@{line_of(snippet)}"


def W(snippet: str) -> str:
    return f"w@{line_of(snippet)}"


def RW(snippet: str) -> str:
    return f"rw@{line_of(snippet)}"


# --- call_sub anchors ---
PSU_DWT = "psu%dwt(:) = 19.8"
FT_INDEX = "ft_index_bigleaf(c) = field1(c)"
FILTER_NUM = "filter(i_type)%num_soilc = 10"
FILTER_SOILC = "filter(i_type)%soilc(:) = 4"
FIELD2_FROM_PTR = "mytype%field2(:) = test_ptr(:,1)*param2(:)"
FIELD1_FROM_FIELD2 = "field1(c) = shr_const_pi*mytype%field2(c)"
CALL_PARSING_SUB = "call test_parsing_sub(bounds,"
CALL_TRIDIAGONAL = "call Tridiagonal(bounds,"
CALL_COLNF_INIT = "call col_nf%Init(bounds%begc, bounds%endc)"
CALL_PTR_TEST = "call ptr_test_sub(filter(i_type)%num_soilc"
CALL_TRACE = "call trace_dtype_example(mytype, col_nf, .true.)"
CALL_TRACE_MIDDLE = "call trace_dtype_example(mytype,col_nf_middle,.true.)"
CALL_TRACE_LAST = "call trace_dtype_example(mytype,col_nf_last,.true.)"
PTR_FILL = "test_ptr(:,:) = SHR_CONST_SPVAL"
ONE_LINE_IF = "if(.true.) field1(c) = SHR_CONST_PI"
CALL_TRACE_ALL = (CALL_TRACE, CALL_TRACE_MIDDLE, CALL_TRACE_LAST)
CALL_PARENT = "call parent_sub()"
DECL_INPUT2 = "input2(bounds%begg:bounds%endg)"
DECL_TRI = tuple(f"real(r8) :: {v}_tri(bounds%begc" for v in "abcru")

# --- parent_sub anchors ---
CALL_NESTED = "call test_nested(mytype_inst)"

# --- trace_dtype_example anchors ---
IF_ACTIVE = "if ( mytype2%active .or. flag )then"
FIELD2_UPDATE = "field2(i) = field2(i)/field1(i) + field3(i)"
HRV_FROM_FIELD2 = "hrv(i) = field2(i)"
CALL_ADD = "call add(field2(i), field4(i))"

expected_access = {
    "test_sub_parse::call_sub": {
        elmtypes: {
            # ptr_test_sub's unused intent(in) dummies read them
            "filter%num_soilc": {W(FILTER_NUM), R(CALL_PTR_TEST)},
            "filter%soilc": {W(FILTER_SOILC), R(CALL_PTR_TEST)},
            # test_ptr may point at either *_fire field (select case), so
            # reads/writes through test_ptr count against both.
            "col_nf%m_n_to_litr_met_fire": {
                R(FIELD2_FROM_PTR),
                R(CALL_PARSING_SUB),
                W(CALL_COLNF_INIT),
                W(CALL_PTR_TEST),
                W(PTR_FILL),
            },
            "col_nf%m_n_to_litr_lig_fire": {
                R(FIELD2_FROM_PTR),
                W(CALL_COLNF_INIT),
                W(CALL_PTR_TEST),
                W(PTR_FILL),
            },
            "col_nf%hrv_deadstemn_to_prod10n": {W(CALL_COLNF_INIT), W(CALL_TRACE)},
            "col_nf%hrv_deadstemn_to_prod100n": {W(CALL_COLNF_INIT)},
            "col_nf_middle%hrv_deadstemn_to_prod10n": {W(CALL_TRACE_MIDDLE)},
            "col_nf_last%hrv_deadstemn_to_prod10n": {W(CALL_TRACE_LAST)},
            # bounds: the only bounds_type instance (dummy bound to it)
            "bounds%begc": {*(R(s) for s in DECL_TRI), R(CALL_TRIDIAGONAL), R(CALL_COLNF_INIT)},
            "bounds%endc": {*(R(s) for s in DECL_TRI), R(CALL_TRIDIAGONAL), R(CALL_COLNF_INIT)},
            "bounds%begg": {R(DECL_INPUT2), R(CALL_PARSING_SUB)},
            "bounds%endg": {R(DECL_INPUT2), R(CALL_PARSING_SUB)},
            # parent_sub -> test_nested -> photosynthesis, at the call line
            "mytype_inst%field1": {W(CALL_PARENT)},
            "mytype_inst%field2": {R(CALL_PARENT)},
            "mytype_inst%field3": {R(CALL_PARENT)},
            # elm_drv passes unused_inst as mytype
            "patch_state_updater%dwt": {W(PSU_DWT)},
            "unused_inst%field1": {
                R(FT_INDEX),
                W(FIELD1_FROM_FIELD2),
                RW(CALL_PARSING_SUB),
                *(R(s) for s in CALL_TRACE_ALL),
                W(ONE_LINE_IF),
            },
            "unused_inst%field2": {
                R(FIELD1_FROM_FIELD2),
                W(FIELD2_FROM_PTR),
                *(RW(s) for s in CALL_TRACE_ALL),
            },
            "unused_inst%field3": {R(s) for s in CALL_TRACE_ALL},
            "unused_inst%field4": {RW(s) for s in CALL_TRACE_ALL},
            "unused_inst%active": {R(s) for s in CALL_TRACE_ALL},
        },
        # NOTE: argument checks are currently disabled in test_ParseSubroutine.
        args: {
            "numf": {R(CALL_TRIDIAGONAL)},
            "bounds": {R(CALL_PARSING_SUB), R(CALL_TRIDIAGONAL), R(CALL_COLNF_INIT)},
            "bounds%begc": {R(CALL_TRIDIAGONAL), R(CALL_COLNF_INIT)},
            "bounds%endc": {R(CALL_TRIDIAGONAL), R(CALL_COLNF_INIT)},
            "bounds%begg": {R(CALL_PARSING_SUB)},
            "mytype": {
                R(FT_INDEX),
                RW(FIELD2_FROM_PTR),
                RW(FIELD1_FROM_FIELD2),
                RW(CALL_PARSING_SUB),
                R(CALL_TRACE),
                W(CALL_PTR_TEST),
                RW(CALL_TRACE_MIDDLE),
                RW(CALL_TRACE_LAST),
                W(ONE_LINE_IF),
            },
            "mytype%field1": {
                R(FT_INDEX),
                W(FIELD1_FROM_FIELD2),
                RW(CALL_PARSING_SUB),
                *(R(s) for s in CALL_TRACE_ALL),
                W(ONE_LINE_IF),
            },
            "mytype%field2": {
                R(FIELD1_FROM_FIELD2),
                W(FIELD2_FROM_PTR),
                *(RW(s) for s in CALL_TRACE_ALL),
            },
            "mytype%field3": {R(s) for s in CALL_TRACE_ALL},
            "mytype%field4": {RW(s) for s in CALL_TRACE_ALL},
            "mytype%active": {R(s) for s in CALL_TRACE_ALL},
            "patch_state_updater": {W(PSU_DWT)},
            "patch_state_updater%dwt": {W(PSU_DWT)},
        },
    },
    "test_sub_parse::trace_dtype_example": {
        elmtypes: {},
        args: {
            "mytype2": {R(IF_ACTIVE), RW(FIELD2_UPDATE), R(HRV_FROM_FIELD2), RW(CALL_ADD)},
            "mytype2%field1": {R(FIELD2_UPDATE)},
            "mytype2%field2": {RW(FIELD2_UPDATE), R(HRV_FROM_FIELD2), R(CALL_ADD)},
            "mytype2%field3": {R(FIELD2_UPDATE)},
            "mytype2%field4": {RW(CALL_ADD)},
            "mytype2%active": {R(IF_ACTIVE)},
            "col_nf_inst": {W(HRV_FROM_FIELD2)},
            "col_nf_inst%hrv_deadstemn_to_prod10n": {W(HRV_FROM_FIELD2)},
            "flag": {R(IF_ACTIVE)},
        },
    },
    "test_sub_parse::parent_sub": {
        elmtypes: {
            "mytype_inst%field1": {W(CALL_NESTED)},
            "mytype_inst%field2": {R(CALL_NESTED)},
            "mytype_inst%field3": {R(CALL_NESTED)},
        },
        args: {},
    },
}
