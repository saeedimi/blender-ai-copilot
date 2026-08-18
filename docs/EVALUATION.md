# Evaluation snapshot

The project includes a live evaluation harness that exercises the same backend
API used by the Blender UI.

## Core full-suite snapshot

A v0.8.0.2 full-suite run produced:

```text
Turn pass rate:            85.7%
Case pass rate:            80.0%
Semantic false-success:    14.3%
Average latency:           77.48 s
P95 latency:              231.13 s
Average LLM calls:          3.71
Average tool steps:         3.29
Duplicate render incidents: 0
Discovery precision:      100%
Discovery recall:         100%
```

This is intentionally presented as a portfolio/learning result rather than a
production benchmark.

## What the harness covers

- object creation
- mesh direction normalization
- read-only tool routing
- cross-turn references
- dedicated material assignment
- smooth shading
- camera creation / aiming / activation
- light creation / aiming
- render output handling
- render file verification
- duplicate side-effect detection

## Known limitations

The frozen milestone still has edge cases around:

- repeated already-completed semantic actions in long workflows
- some follow-up mutation goal extraction
- long-workflow latency with a small local LLM
- synthetic-evaluation assumptions that can depend on pre-existing Blender state

The project was intentionally stopped here because the purpose is to demonstrate
AI-agent architecture, safety, verification, RAG, memory, and evaluation rather
than production-grade Blender automation.
