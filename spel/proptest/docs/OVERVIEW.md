# proptest in one page

`spel proptest` tests an ELM routine (a SPEL unit-test case) the way Hypothesis tests a Python
function: you state what must hold, it draws many inputs on a synthetic domain, runs the compiled
routine on them, and shrinks any failure to a small example it saves for replay.

Read this first; then [SPEC.md](SPEC.md) (the format), [Glossary.md](Glossary.md) (terms), the
tutorial notebook (`examples/canopyfluxes_tutorial.ipynb`) and a worked bug hunt
(`experiments/hillslope_im2/`).

## The pieces of a spec

| TOML | What it is | Python (reader) | Python (builder, `api`) |
|---|---|---|---|
| `[case]` | which unit-test case (and build) | `spec.Spec.case_dir`, `.build` | `api.Case` |
| `[domain]` | the gridcells to build, declared or drawn | `domain.DomainSpec` | `api.Domain`, `api.DomainDraw` |
| `[namelist]` | the configuration (which physics is on) | `spec.Spec.namelist` | `Spec(namelist=...)` |
| `[scenario.x]` | how inputs are drawn: one rule per input, `derive`, `assume`, steps | `spec.Scenario`, `rules.Rule` | `api.Scenario`, `api.Range`, ... |
| `[[property]]` | an expression that must hold at every element | `checks.Property` | `Spec.property` |
| `[[relation]]` | base run vs. changed follow-up run, compared by `expect` | `checks.Relation` | `Spec.relation` |
| `[[differential]]` | the same example on two builds/cases | `checks.Differential` | `Spec.differential` |

Every check runs in every scenario its `scenarios` patterns match (default: all).

The builder (`api.Spec`) produces the dict the reader (`spec.parse`) takes, and checks each piece
with the reader's code as it is added, so a spec written in Python, its TOML file and
`spel proptest` agree.

## From TOML to a verdict

![From TOML to a verdict: spec and inventory, a Session (build, domain, worker), then per example draw, prepare, run and judge; a failure is shrunk by Hypothesis, saved, and reported as an Outcome](pipeline.svg)

<!-- Source: pipeline.mmd (Mermaid). After editing it, regenerate the image with
     ./render_diagrams.sh (in this directory). -->

A relation or scenario with its own `namelist` runs in `Session.sibling`, a second worker on the
same library; a differential check runs the other case or build in `Session.other`.

`runner/` is a package: `Session` (session.py) composes `ElementSpace` (elements.py: elements,
shapes, fixed values), `Evaluation` (evaluation.py: expressions, namespaces), `Drawing`
(drawing.py: examples), `Stepping` (stepping.py: running the routines) and `FailureStore`
(failures.py); `run`, `_shrink` and `replay` are in orchestration.py, the failure report in
reporting.py. `Session` keeps the methods other modules call as one-line forwards.

Where to look for:

| You want to know... | Look at |
|---|---|
| which variables are inputs/outputs, and under which options | `inventory.py` (`Inventory.configured`) |
| how a TOML table is read and checked | `spec.py`, `rules.parse`, `checks.<Kind>.from_table` |
| how values are drawn | `runner/drawing.py` (`Drawing.draw_example`, `inputs_at`), `strategies.py`, `rules.py` |
| how ELM's hierarchy and filters are laid out | `hierarchy/` (`Layout`), `domain.py` |
| how the routine is called | `shim.py` (Fortran side), `worker.py`/`runtime.py` (Python side) |
| how a failure message is written | `runner/reporting.py` (`report`) |
| multi-step runs: what the rest of ELM's step does | `coupling.py`, `runner/stepping.py` (`Stepping.run_steps`), scenario `forcing`/`between`/`keep`/`carry` |
| drawn domains and their shrinking | `domains.py`, `runner/orchestration.py` (`run`, `_shrink`) |

## Names in expressions

Expressions are Fortran text, parsed with SPEL's parser and evaluated with numpy for all elements
at once. SPEC.md "Names and expressions" has the details and the full function list.

| Name | Meaning | Where |
|---|---|---|
| `x`, `inst%field(p)` | a variable after the run (an input: its drawn value) | everywhere |
| `p`, `c`, `l`, `t`, `g` | the element being judged, and its ancestors | everywhere |
| `input%x` | x before the run | checks |
| `prev%x`, `step` | x before the step being judged, and the step number (one call: `prev%x` is `input%x`) | checks, `forcing`, `between` |
| `filter%soilp`, `filter%num_soilp` | a filter of the case and its count | everywhere |
| `base%x`, `follow%x` | the base and follow-up runs | `[[relation]]` |
| `ref%x`, `dev%x` | this case and the `[compare]` one | `[[differential]]` |
| `close`, `implies`, `finite`, `child_sum`, `tend`, `acc`, ... | built-in functions | everywhere |

## Same names, different objects

| Name | Is |
|---|---|
| `api.Spec`, `api.Scenario` | builders you write in Python; their only output is the TOML dict |
| `spec.Spec`, `spec.Scenario` | that TOML, parsed and resolved: what the runner uses |
| `api.Domain` / `domain.DomainSpec` | the declared domain (builder / parsed) |
| `domain.SynthDomain` | the synthetic domain built on disk for one session |
| `api.Range`, ... / `rules.Range`, ... | a rule you write / how that rule kind draws values |
| `api.Case` / `runtime.Case` | the case (inventory + reference) / the loaded library |
