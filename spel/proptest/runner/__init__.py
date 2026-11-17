"""Run the checks of a `<name>.proptest.toml` with Hypothesis.

- `session.py`: `Session`, one spec on one case: build, configuration, synthetic domain, backend,
  and how a check is judged on an example. It composes:
  - `elements.py`: `ElementSpace`, the domain's elements, shapes and fixed values;
  - `evaluation.py`: `Evaluation`, spec expressions and their namespaces;
  - `drawing.py`: `Example` and `Drawing`, inputs from rules, `derive`, relation follow-ups;
  - `stepping.py`: `Stepping`, one call or a multi-step run of the routines;
  - `failures.py`: `FailureStore`, saved failing examples;
  - `reporting.py`: `PropertyFailure`, `Failure`, `FailureSignature` and the failure report.
- `orchestration.py`: `run` (and drawn domains, `_shrink`), `replay`, `Outcome`, JSON/JUnit output.

What other modules use is re-exported here (`runner.Session`, `runner.run`, ...).
"""

from spel.proptest.runner.drawing import Example
from spel.proptest.runner.evaluation import _per_element
from spel.proptest.runner.failures import DOMAIN_FILE, EXAMPLE_FILE
from spel.proptest.runner.orchestration import (
    _NO_INPUT_SHRINK,
    Outcome,
    _run_domain,
    _selected,
    _shrink,
    replay,
    run,
    write_json,
    write_junit,
)
from spel.proptest.runner.reporting import Failure, FailureSignature, PropertyFailure
from spel.proptest.runner.session import ProbeResult, Session
from spel.proptest.runner.stepping import StepRun

__all__ = [
    "DOMAIN_FILE",
    "EXAMPLE_FILE",
    "Example",
    "Failure",
    "FailureSignature",
    "Outcome",
    "ProbeResult",
    "PropertyFailure",
    "Session",
    "StepRun",
    "_NO_INPUT_SHRINK",
    "_per_element",
    "_run_domain",
    "_selected",
    "_shrink",
    "replay",
    "run",
    "write_json",
    "write_junit",
]
