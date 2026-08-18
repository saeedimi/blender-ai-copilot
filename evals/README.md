# Blender AI Copilot Evaluation Harness — v0.8.0.2

This harness evaluates the Copilot from the **outside**, using the same `/chat`
and `/approve` API that the Blender UI uses. It does not bypass the controller.

## What it measures

- task / turn pass rate
- controller success vs harness semantic success
- semantic false-success rate
- deterministic verification completion
- explicit goal completion
- required and forbidden tool execution
- dynamic-discovery recall
- dynamic-discovery precision when an allowlist is supplied
- controller events such as direction normalization and reference resolution
- tool arguments and selected result fields
- repeated side-effect execution
- render-at-most-once violations
- LLM call count
- tool-step count
- prompt and generated tokens
- average and P95 request latency
- approval count

## Safety

The mesh suite contains high-risk Blender tools. The live runner will stop at
an approval boundary unless you explicitly pass `--auto-approve`.

Use `--auto-approve` only in a safe evaluation scene.

## Object placement

Every live evaluation run receives a generated run ID and a distant run-specific
origin. Object names also contain the run ID. This prevents new evaluation
objects from overlapping objects created by previous runs.

## Quick start

From the project root:

```bash
python -m evals.runner --suite evals/suites/smoke.json --dry-run
```

Inspect the generated prompts first.

Then run the smoke suite:

```bash
python -m evals.runner \
  --suite evals/suites/smoke.json \
  --backend http://127.0.0.1:8765 \
  --auto-approve
```

Run the memory suite:

```bash
python -m evals.runner \
  --suite evals/suites/memory.json \
  --backend http://127.0.0.1:8765
```

Run the render suite:

```bash
python -m evals.runner \
  --suite evals/suites/render.json \
  --backend http://127.0.0.1:8765
```

Run everything:

```bash
python -m evals.runner \
  --suite evals/suites/full.json \
  --backend http://127.0.0.1:8765 \
  --auto-approve
```

Reports are written under:

```text
eval_reports/<run_id>/
├── rendered_suite.json
├── run_variables.json
├── results.json
└── report.md
```

## Analyze existing traces

This is useful for historical operational metrics:

```bash
python -m evals.analyze --trace-dir traces
```

That reports success rate, verification-complete rate, latency, duplicate render
incidents, no-replay guard events, and unresolved reference events.

## Suites

### `smoke.json`

Fast regression coverage:

1. object creation
2. mesh direction normalization + semantic verification
3. read-only material inspection / mutation-pruning

### `memory.json`

Multi-turn referential memory:

1. create object
2. `Make it blue.`
3. `Shade it smooth.`

The harness verifies that `it` resolves to the exact created object and that a
dedicated material is created/assigned rather than recoloring an unrelated
shared material.

### `render.json`

Mixed multi-category workflow:

- mesh
- smooth shading
- material create + assignment
- camera create + aim + activation
- area light + energy + aim
- output filename
- exactly one terminal render
- saved-file verification

### `full.json`

All of the above.

## Exit codes

- `0`: every scored turn passed
- `1`: one or more evaluation checks failed
- `2`: backend connection failure

This makes the harness usable later in CI or a local regression script.

## v0.8.0.1 reliability regressions

The full suite now explicitly checks same-turn pronoun handling, cross-turn
`Shade it smooth` execution, mixed-workflow camera/light aim goals, and final-answer
claim grounding. The render test treats light energy supplied directly to `create_light`
as semantically valid instead of requiring a redundant `set_light_energy` call.


## v0.8.0.2 aim-goal regression

The mixed render case now asserts that camera/light aim goals are extracted and
executed with the exact generated source and target names. This specifically
regresses the v0.8.0.1 failure where decimal coordinates caused the aim clauses
to be missed by punctuation-sensitive matching.
