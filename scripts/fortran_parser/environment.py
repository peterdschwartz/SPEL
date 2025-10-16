from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.analyze_subroutines import Subroutine
    from scripts.DerivedType import DerivedType
    from scripts.utilityFunctions import Variable


@dataclass
class Environment:
    inst_dict: dict[str, DerivedType]
    variables: dict[str, Variable]
    locals: dict[str, Variable]
    globals: dict[str, Variable]
    dummy_args: dict[str, Variable]
    fns: dict[str, Subroutine]

    def to_dict(self):
        return asdict(self)


def add_ptr_vars(
    ptr_dict: dict[str, str],
    env_dict: dict[str, Variable],
) -> None:
    """
    env_dict is modified in place.
    ptr_dict: {'var_name':'ptr_name'}
    """
    for varname, ptrname in ptr_dict.items():
        if varname in env_dict:
            env_dict[ptrname] = env_dict[varname]
