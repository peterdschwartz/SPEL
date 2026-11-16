# Glossary

Terms used by `spel proptest`, grouped by where they come from. The format itself is specified in
[`SPEC.md`](../SPEC.md)

## ELM Terms

- `gridcell`: a lat/lon point of the model grid and the top of the subgrid hierarchy. Every other
  element belongs to exactly one gridcell.
- `subgrid`: the structure inside a gridcell: `topounit` > `landunit` > `column` > `patch`. Each
  element has a weight in its parent (`wtgcell`, `wtlunit`, ... are weights in a coarser ancestor),
  and the weights of a parent's children sum to 1.
- `topounit`: a topographic division of a gridcell (e.g. an elevation band or a hillslope position)
- `landunit`: a land type within a topounit: soil (natural vegetation), crop, urban (by density),
      lake, wetland, glacier, glacier with elevation classes, or polygonal tundra. Its type code is
      `ltype` (`istsoil`, `isturb_md`, ...), checked against the case's parameters.
- `column`: the level that holds soil, snow and water state.
- `patch`: a plant functional type (`pft`) on a column, or a crop functional type (`cft`) on a crop column
- `active`: ELM's `is_active_l/c/p` (subgridWeightsMod). Weight-0 elements usually exist but are
  inactive, and most physics loops skip them.
- `filter`: a list of element indices a routine loops over (`filter%soilp`, `nolakeurbanp`),
          built from the hierarchy and the state by the case's filter builders.
- `ELM order`: the order in which ELM numbers elements within a clump. Landunits are created in one
              pass per type over all topounits, so a gridcell's elements are not contiguous.
- `clump`: a contiguous slice of a process's gridcells that one thread works on.
- `configuration`: the physics active in a run, chosen by the namelist options
- `namelist option`: a run-time switch read from `lnd_in`. A scenario or relation can set them
                  (`namelist = {...}`), and they decide which variables a routine reads at all.
- `constants`: module variables the routines read that are fixed for a run (physical constants,
  pft parameters, namelist values). They are captured once into the constants file; a scenario can
  `scale` them.

## SPEL Terms

- `case`: a unit test made by `spel create`: the routines under test with their ELM sources, built
        into a library (`libspelcase.so`) that proptest calls through a generated shim.
- `reference`: the data captured from a real ELM run for a case
- `inventory`: everything proptest knows about a case's variables: level, shape, kind
            (input/inout/output), units, and the namelist gates of each access
- `nsets`: the number of times the declared gridcells are repeated in the domain

## Property Testing Terms

- `property`: a logical statement about the outputs that must hold for every valid input.
- `metamorphic relation`: a property between two runs: a base input and a follow-up made from it by
        a known transformation (`follow`), with an expected relationship between 
        their outputs (`expect`, using `base%x` and `follow%x`)
- `differential test`: the same examples run on two implementations, a case and another case or
        build named in `[compare]` (`ref%x` vs `dev%x`), and the outputs compared. Typical uses are a
        refactor against the original, OpenACC against serial, or two compilers.
- `oracle`: whatever decides whether an output is right. Properties and relations are partial
        oracles: they reject wrong answers without computing the right one.
- `rule`: how one input's values are drawn: a constant, `range`, `log_range`, `values`, `dist`, or
        `scale` of the reference value, with optional per-pft and per-level overrides.
- `derive`: an input computed from other inputs instead of drawn, to keep inputs physically
        consistent (e.g. `laisha = elai - laisun`).
- `assume`: a precondition on the inputs. Elements where it is false are not judged, and an example
        with no element left is discarded.
- `scenario`: the input space a check is run on: a rule (or `derive` expression) for every input
        the routines read, namelist options, `assume` conditions, and which elements are judged.
- `check`: one property, relation or differential of a spec, with the scenarios it applies to and the elements it is judged `over`.
- `example`: one complete set of input values for the whole domain, drawn from a scenario. The
        routine runs once per example and every check judges the result.
- `draw`: picking values with Hypothesis. SPEL draws a small structure per example (a seed and one
        mode per input), and numpy expands it into arrays, which keeps Hypothesis' budget small.
- `mode`: how one input is spread over the elements in an example: one constant, independent per
        entry, edges (each entry at the low or high end of its rule), or independent per element and
        uniform down the layers.
- `domain`: the synthetic subgrid hierarchy the routine runs on, declared (`[[domain.gridcell]]`)
        or drawn (`[domain.draw]`). It plays the role of the grid of a real run.
- `drawn domain`: a domain generated from a seed: random landunit kinds, weights and topounits.
        Each of `count` domains gets its own session, and a failing domain is shrunk like an example.
- `layout`: a domain laid out as ELM would lay it out: indices, parents, weights, activity and
        flags per level, in ELM order. Topology inputs come from it, not from rules.
- `session`: one configured run environment: a case, a domain, a namelist configuration and a
        worker process. It turns examples into routine calls and returns the state.
- `worker`: the subprocess that loads the case library and calls the routines. It is separate so a
        Fortran crash or `stop` does not kill the runner, and because the library cannot be reloaded.
- `runner`: the driver of `spel proptest run`. For each check and scenario it asks Hypothesis for
        examples, runs them through a session, evaluates the check, and shrinks and saves failures.
- `verdict`: the outcome of a check on an example or run: pass, fail (with the failing elements), or
        an error such as NaN or a crash.
- `multi-step run`: a scenario with `steps = N`, where the state after step k is the input of step
        k+1. What the rest of the model does between steps is declared, not simulated: `forcing` (inputs
        set before each step), `between` (updates after each step), `keep` and `carry`.
- `dormant rule`: a rule for a variable the routines do not read under a configuration. It is kept
        for the configurations that do read it.

  ## Hypothesis Terms

- `Hypothesis`: the Python property-based testing library that drives the search for failing
        examples. SPEL uses its drawing, shrinking and example database, not its strategies for arrays.
- `strategy`: a Hypothesis description of how to generate values (`st.integers()`,
        `st.floats()`). SPEL draws through `st.data()` inside one test, so the choices depend on the scenario.
- `choice sequence`: the stream of primitive choices Hypothesis records for an example. Shrinking
        edits this stream, so an example's size is the number of choices, not the size of its arrays.
- `shrinking`: after a failure, Hypothesis searches for a simpler example that still fails. In SPEL
        that means fewer non-constant inputs and simpler values; drawn domains are shrunk separately
        (fewer gridcells, landunits, patches).
- `falsifying example`: the minimal failing example Hypothesis reports after shrinking. SPEL saves
        it under `failures/` as a capture that `spel proptest replay` reruns.
- `example database`: where Hypothesis stores failing examples so the next run tries them first.
        SPEL keeps one per scenario and check in `unit-tests/proptest-data/<case>/hypothesis/`.
- `seed`: makes a run reproducible (`--seed`, `[settings] seed`). With a seed, Hypothesis does not read the example database.
- `max_examples`: how many examples Hypothesis runs per check (default 100). Each example is one
        routine call, so this is the main cost control.
- `settings`: Hypothesis options (`max_examples`, `derandomize`, `phases`, `verbosity`, ...), given in `[settings]`.
- `assume` / `reject` (Hypothesis): discarding an example as invalid rather than failing it. SPEL's
        `assume` uses this when no element is left.
- `health check` / `deadline`: Hypothesis errors for slow or wasteful tests (too many discards, slow
        data generation, an example over its time limit). SPEL turns all of them off, since every example
        is a Fortran call that is slow by Hypothesis' standards; watch the discard count yourself.
