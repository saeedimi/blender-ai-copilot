# Architecture

## High-level flow

```text
User
 |
 v
Blender Copilot UI
 |
 | POST /chat
 v
Agent / Controller
 |
 +-- bounded conversation context
 +-- structured referential memory
 +-- explicit task/goal ledger
 +-- dynamic semantic tool discovery
 +-- validation and risk classification
 +-- human-in-the-loop approval
 +-- deterministic semantic normalization
 +-- tool execution
 +-- deterministic verification
 +-- final-answer grounding
 |
 +--------------+
 |              |
 v              v
Local RAG    Blender file bridge
                |
                v
        Blender extension
                |
                v
           bpy / bmesh
```

## Separation of responsibilities

### LLM

The local language model is used for interpretation and planning where semantic
reasoning is useful.

### Controller

The controller manages:

- tool exposure
- goal tracking
- reference resolution
- argument validation
- risk classification
- approval boundaries
- deterministic normalization
- retries
- verification
- completed-state reasoning
- render safety

### Blender extension

The Blender extension performs the actual application-level operations using
`bpy` and `bmesh`.

### RAG

Blender documentation is retrieved locally using hybrid lexical/vector search
plus reranking.

## Tool grouping

Tools are organized along two dimensions.

Domain:

```text
objects
materials
modifiers
mesh
cameras
lights
rendering
knowledge
```

Behavior:

```text
read-only observation
safe state mutation
high-risk geometry mutation
terminal side effect
```

This allows the controller to gate tools and apply different safety policies
without exposing arbitrary application scripting.
