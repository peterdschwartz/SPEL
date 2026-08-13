import scripts.io.helper as hio
from spel.scripts.utilityFunctions import Variable


def create_text_io_routine(
    mode: str,
    sub_name: str,
    vars: dict[str, Variable],
    fn: str,
) -> list[str]:
    lines: list[str] = []

    max_var_dim = 5
    tabs = hio.indent("reset")
    lines.append(f"{tabs}subroutine {sub_name}(bounds,nsets)\n")
    tabs = hio.indent("shift")
    lines.extend(
        [
            f"{tabs}use fileio_mod, only : fio_open, fio_read, fio_close\n",
            # Dummy Variables
            f"{tabs}type(bounds_type), intent(in) :: bounds\n",
            f"{tabs}integer, intent(in) :: nsets\n",
            # Local Variables
            f"{tabs}character(len=256) :: filename = {sub_name}\n"
            f"{tabs}integer :: errcode = 0\n",
            f"{tabs}integer :: begp,  endp\n",
            f"{tabs}integer :: begc,  endc\n",
            f"{tabs}integer :: begg,  endg\n",
            f"{tabs}integer :: begl,  endt\n",
            f"{tabs}integer :: begt,  endl\n",
            f"{tabs}begp = bounds%begp; endp = bounds%endp/nsets\n",
            f"{tabs}begc = bounds%begc; endc = bounds%endc/nsets\n",
            f"{tabs}begl = bounds%begl; endl = bounds%endl/nsets\n",
            f"{tabs}begt = bounds%begt; endt = bounds%endt/nsets\n",
            f"{tabs}begg = bounds%begg; endg = bounds%endg/nsets\n",
        ]
    )

    return lines
