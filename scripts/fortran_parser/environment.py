from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.analyze_subroutines import Subroutine
    from scripts.DerivedType import DerivedType
    from scripts.utilityFunctions import Variable


@dataclass
class Environment:
    inst_dict: dict[str, DerivedType] = field(default_factory=dict)
    variables: dict[str, Variable] = field(default_factory=dict)
    locals: dict[str, Variable] = field(default_factory=dict)
    globals: dict[str, Variable] = field(default_factory=dict)
    dummy_args: dict[str, Variable] = field(default_factory=dict)
    fns: dict[str, Subroutine] = field(default_factory=dict)

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
