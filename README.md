# Blender AI Copilot

A local **agentic AI copilot for Blender** that translates natural-language instructions into safe, structured, and verifiable actions inside a real 3D application.


## Demo

![Blender AI Copilot Demo](assets/Demo.gif)

Example interaction:

```text
Create a red cube, a blue sphere, and a green cylinder.
Place them side by side and shade the curved objects smooth.
```

The Copilot translates the request into semantic Blender operations, executes them through the Blender extension, verifies the resulting state, and reports the result in the chat panel.

---


## What is Blender?

[Blender](https://www.blender.org/) is a free and open-source 3D creation suite used for modeling, animation, rendering, simulation, and other 3D workflows. Many Blender tasks require users to navigate complex menus, manage scene state, and execute multiple dependent operations in the correct order.

**Blender AI Copilot** explores how an AI agent can make that workflow more conversational. Instead of manually performing every operation, a user can describe a goal in natural language and the Copilot can interpret the request, select appropriate Blender tools, execute the actions, verify the resulting scene state, and continue the interaction through follow-up instructions.

For example:

```text
Create a red cube, a blue sphere, and a green cylinder.
Place them side by side and shade the curved objects smooth.
```

The goal of the project is not simply to connect an LLM to Blender. It is to build an **end-to-end agent architecture** around a stateful external application where actions can have real side effects and therefore need validation, memory, safety controls, and verification.

> **Portfolio / learning project.** This repository demonstrates agentic AI engineering patterns and is not intended as production-grade Blender automation.

---

## What I built

The project implements an end-to-end local agent system with the following components:

- **Local LLM planning and tool calling** using Ollama-compatible models
- **Semantic Blender tools** for structured operations instead of unrestricted Python execution
- **Agent/controller architecture** for multi-step task execution and goal tracking
- **Hybrid RAG** over Blender documentation using BM25, FAISS, embeddings, and cross-encoder reranking
- **Structured conversational memory** for follow-up references such as `Make it blue`
- **Human-in-the-loop approval** for higher-risk operations
- **Dynamic tool gating and argument validation** before execution
- **Deterministic state verification** after Blender actions
- **Retry and replay protection** for non-idempotent operations such as rendering
- **Live evaluation harness** for task completion, tool use, reference resolution, repeated mutations, latency, and other agent behaviors
- **Blender extension + local backend integration** so the same agent can be used directly from the Blender UI

Together, these components make Blender a practical test environment for broader agentic AI problems: planning, tool selection, state tracking, external-system interaction, safety, verification, and evaluation.

---

## Why this is an agentic AI project

A simple chatbot can generate instructions about how to use Blender. This project instead gives the model a constrained set of tools and places a deterministic controller between the model and Blender.

The model proposes **what should happen**. The controller decides **whether and how it can happen safely**, executes the corresponding semantic operations, checks the resulting state, and determines whether the task is complete.

That separation is important because Blender is stateful: creating, editing, moving, rendering, or deleting objects changes the environment. A useful agent therefore needs more than text generation—it needs memory, tool-use constraints, verification, failure handling, and protection against unsafe repeated side effects.

---


## What it can do

- Create and move Blender objects and mesh primitives
- Inspect scene objects, materials, cameras, lights, and render settings
- Create and assign materials
- Add and configure modifiers
- Perform selected mesh-editing operations
- Create, move, aim, and activate cameras
- Create and configure lights
- Configure render settings and render to file
- Understand follow-up references such as `Make it blue`
- Answer Blender questions using local RAG over Blender documentation
- Require approval for higher-risk operations
- Verify Blender state after semantic tool execution
- Run regression/evaluation suites against the live backend

---

## How it works

```text
User
 |
 v
Blender Copilot UI
 |
 | HTTP
 v
+---------------------------+
| Agent / Controller        |
|                           |
| - local LLM planning      |
| - goal tracking           |
| - structured memory       |
| - safety / validation     |
| - dynamic tool gating     |
| - deterministic checks    |
| - verification            |
+------------+--------------+
             |
       +-----+------+
       |            |
       v            v
   Local RAG    File Bridge
                    |
                    v
             Blender Extension
                    |
                    v
               bpy / bmesh
```

The **LLM proposes semantic actions**, but it does not directly control Blender. The controller owns validation, safety, reference handling, argument normalization, execution policy, verification, retry behavior, and task-completion logic.

This design keeps the model focused on interpretation and planning while deterministic code governs side effects. The model is **not given arbitrary Python execution**.

---

## Repository structure

```text
blender-ai-copilot/
├── blender_extension/       # Blender UI + semantic tool execution
├── src/                     # backend agent, controller, bridge, router, RAG
├── evals/                   # live evaluation harness + suites
│
├── rag/
│   ├── source_manifest.json
│   ├── builtin_corpus.json
│   └── README.md
│
├── scripts/
│   ├── build_rag_index.py
│   └── package_blender_extension.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EVALUATION.md
│   ├── INSTALLATION.md
│   └── MODELS.md
│
├── requirements.txt
├── .gitignore
├── run_evals.sh
└── README.md
```

---

# Installation

## Requirements

You need:

- **Blender 4.2+**
- **Python 3**
- **Ollama**
- A local Ollama model that supports reliable tool/function calling
- Enough RAM/VRAM for the model you select

The default model used during development is:

```text
qwen3:4b-instruct
```

The model weights are not stored in this repository.

---

## Option A — Recommended: clone the backend + install the prebuilt Blender extension

### 1. Clone the repository

```bash
git clone https://github.com/saeedimi/blender-ai-copilot.git
cd blender-ai-copilot
```

### 2. Create a Python environment

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Ollama and pull a model

For the default model:

```bash
ollama pull qwen3:4b-instruct
```

You may use another compatible Ollama model. See [Changing the model](#changing-the-model).

### 4. Build the local RAG index

```bash
python scripts/build_rag_index.py
```

This downloads the curated Blender documentation pages listed in the repository, chunks them, creates embeddings, and builds the local FAISS index used by the backend.

For a small offline smoke test:

```bash
python scripts/build_rag_index.py --builtin-only
```

### 5. Start the backend

```bash
python -m src.agent \
  --project-root . \
  --host 127.0.0.1 \
  --port 8765 \
  --ollama-url http://127.0.0.1:11434 \
  --model qwen3:4b-instruct
```

Keep this terminal running while using Blender.

### 6. Download the Blender extension

Download the installable extension ZIP from:

**[GitHub Releases](https://github.com/saeedimi/blender-ai-copilot/releases/latest)**

Do not unzip the extension package before installing it in Blender.

### 7. Install the extension in Blender

In Blender:

```text
Edit
→ Preferences
→ Add-ons / Extensions
→ Install from Disk
→ select the downloaded Blender AI Copilot ZIP
→ enable Blender AI Copilot
```

Then open a **3D Viewport** and press:

```text
N
```

Open the:

```text
Copilot
```

tab.

The default backend URL is:

```text
http://127.0.0.1:8765
```

Click **Check** before sending your first request.

---

## Option B — Build the Blender extension yourself

If you prefer to build the extension from the repository source:

```bash
python scripts/package_blender_extension.py
```

This creates:

```text
dist/blender_ai_copilot_extension.zip
```

Install that ZIP in Blender using:

```text
Edit
→ Preferences
→ Add-ons / Extensions
→ Install from Disk
```

---

# How to use the Copilot

After installation:

1. Start Ollama.
2. Start the Blender AI Copilot backend.
3. Open Blender.
4. Open `N → Copilot`.
5. Confirm the backend URL is `http://127.0.0.1:8765`.
6. Click **Check**.
7. Type a natural-language request in the Copilot panel.
8. Click **Send**.
9. If an operation is classified as higher risk, review and approve it before execution.
10. Continue the conversation with follow-up instructions.

### Simple example

```text
Create a UV sphere named Product at (0, 0, 1).
```

Then:

```text
Make it blue.
```

Then:

```text
Shade it smooth.
```

The controller maintains structured referential memory, so `it` can resolve to the previously created object.

### Multi-step example

```text
Create a UV sphere named Product at (0, 0, 1).
Shade it smooth, create a blue material and assign it to Product.
Create a camera at (0, -8, 4), aim it at Product, and make it active.
Create a white AREA light at (4, -3, 6) with energy 1000 and aim it at Product.
Set the render output to product.png and render to file.
```

### Multi-object example

```text
Create a red cube named RedCube at (-3, 0, 1),
a blue UV sphere named BlueSphere at (0, 0, 1),
and a green cylinder named GreenCylinder at (3, 0, 1).
Shade the sphere and cylinder smooth.
```

### Blender knowledge example

```text
What is the difference between a Bevel modifier and a Subdivision Surface modifier?
```

For Blender knowledge questions, the Copilot can retrieve relevant local documentation through the RAG pipeline before generating the answer.

---

# Changing the model

The Copilot is **not tied to one LLM**.

The backend accepts a model name through:

```text
--model
```

For example:

```bash
ollama pull qwen3:8b
```

Then:

```bash
python -m src.agent \
  --project-root . \
  --host 127.0.0.1 \
  --port 8765 \
  --ollama-url http://127.0.0.1:11434 \
  --model qwen3:8b
```

A stronger model may improve:

- long-request decomposition
- semantic tool selection
- argument extraction
- planning consistency
- natural-language responses

A larger model may also require more RAM/VRAM and may be slower.

The controller remains responsible for safety, verification, approvals, reference handling, and render replay protection regardless of the selected model.

### Using Ollama on another machine

You can also point the backend to a remote Ollama server:

```bash
python -m src.agent \
  --project-root . \
  --ollama-url http://192.168.1.50:11434 \
  --model YOUR_MODEL_NAME
```

The Blender extension talks to the Copilot backend. It does not communicate directly with Ollama.

The current backend uses Ollama's `/api/chat` tool-calling format. Other providers such as OpenAI, Anthropic, or Google would require a provider adapter.

See [`docs/MODELS.md`](docs/MODELS.md) for more detail.

---

# RAG

The project includes local Blender-documentation retrieval using:

- BM25
- FAISS
- sentence-transformer embeddings
- cross-encoder reranking

The retrieval pipeline is conceptually:

```text
Question
   ↓
BM25 + FAISS retrieval
   ↓
Embedding similarity
   ↓
Cross-encoder reranking
   ↓
Relevant Blender documentation
   ↓
Selected LLM
   ↓
Grounded answer
```

The GitHub repository stores:

```text
rag/source_manifest.json
rag/builtin_corpus.json
scripts/build_rag_index.py
```

It does **not** store copied documentation, generated vectors, or ML model weights.

Build the index locally with:

```bash
python scripts/build_rag_index.py
```

Changing the **main Ollama LLM** does not require rebuilding the RAG index.

Changing the **embedding model** does require rebuilding the FAISS index.

---

# Conversation memory

The Blender UI keeps the visible chat history while the local model receives a bounded context.

The backend also maintains structured referential memory for entities such as:

```text
last_object
last_material
last_camera
last_light
last_render_output
```

This supports interactions such as:

```text
Create a sphere named Product.
Make it blue.
Shade it smooth.
```

without sending the entire visible chat transcript to the local model.

---

# Safety and verification

The Copilot exposes semantic Blender tools instead of arbitrary Python execution.

Examples include:

```text
create_uv_sphere
create_material
assign_material
shade_smooth
aim_camera_at_object
aim_light_at_object
set_render_output
render_scene
```

The controller can:

- restrict which tools are visible for a request
- validate tool arguments
- classify operations by risk
- require user approval for higher-risk actions
- normalize deterministic semantics
- verify Blender state after execution
- prevent unsafe blind retries of non-idempotent actions

For terminal side effects such as rendering, a timeout is treated as an **unknown execution state** rather than automatically replaying the operation.

---

# Evaluation harness

The repository includes a live evaluation harness that exercises the same backend API used by the Blender UI.

Run the core suite:

```bash
python -m evals.runner \
  --suite evals/suites/full.json \
  --backend http://127.0.0.1:8765 \
  --auto-approve
```

The harness measures:

- task and turn success
- semantic false-success
- goal completion
- required/forbidden tool use
- verification
- reference resolution
- repeated mutations
- render-at-most-once behavior
- latency
- LLM calls
- tool steps
- discovery precision and recall

See [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

# Current status

This repository is frozen as a **learning and portfolio milestone**.

The project uses Blender as a concrete environment for demonstrating reusable agentic AI engineering patterns:

```text
Local LLM agents
Semantic tool calling
Multi-step task execution
RAG
Structured conversational memory
Reference resolution
Human-in-the-loop safety
Deterministic verification
Dynamic tool gating
Failure and replay handling
Agent evaluation
Observability
External-application integration
```

The goal is to demonstrate how an AI agent can interact with a stateful application through controlled tools, memory, safety checks, and verification rather than to provide complete Blender coverage or production-level reliability.

Known limitations and evaluation results are documented in [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

# Troubleshooting

### Blender cannot reach the backend

Make sure the backend is still running and that the Copilot panel points to:

```text
http://127.0.0.1:8765
```

Then click **Check**.

### Ollama cannot be reached

Make sure Ollama is running and that the `--ollama-url` value is correct.

### The model answers but does not operate Blender

The selected model must support reliable **tool/function calling** through Ollama. A model that only generates ordinary text is not sufficient.

### Blender tool calls time out

Keep Blender open and click **Check** in the Copilot panel before sending requests.

### RAG does not initialize

Build the local index:

```bash
python scripts/build_rag_index.py
```

If you changed the embedding model, rebuild the index.

For additional setup details, see [`docs/INSTALLATION.md`](docs/INSTALLATION.md).

---

# Release

Installable Blender extension packages are available from:

**[GitHub Releases](https://github.com/saeedimi/blender-ai-copilot/releases)**

Developers can also build the extension locally with:

```bash
python scripts/package_blender_extension.py
```

---

# License

Code in this repository is released under the [MIT License](LICENSE).
