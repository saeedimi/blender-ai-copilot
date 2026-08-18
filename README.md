# Blender AI Copilot

A local AI agent for controlling Blender through natural language.

The project combines a local LLM, semantic Blender tools, retrieval-augmented
generation (RAG), structured conversation memory, human-in-the-loop safety,
deterministic verification, and an evaluation harness.

> Portfolio / learning project. It is intentionally not positioned as a
> production-ready Blender automation system.

## What it can do

- Create and move Blender objects and mesh primitives
- Inspect scene objects, materials, cameras, lights, and render settings
- Create and assign materials
- Add and configure modifiers
- Perform selected mesh operations
- Create, position, aim, and activate cameras
- Create and configure lights
- Configure render settings and render to file
- Resolve follow-up references such as `Make it blue`
- Use Blender documentation through local RAG
- Require approval for high-risk operations
- Verify Blender state after semantic tool execution
- Run regression/evaluation suites against the live backend

## Architecture

```text
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
| - tool gating             |
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

The LLM proposes semantic actions. The controller owns validation, safety,
reference handling, verification, retry behavior, and task completion logic.
Arbitrary Python execution is not exposed to the model.

## Repository layout

```text
blender-ai-copilot/
├── blender_extension/   # Blender UI + semantic tool execution
├── src/                 # backend agent, tools, bridge, router, RAG
├── evals/               # live evaluation harness + suites
├── rag/
│   ├── source_manifest.json
│   ├── builtin_corpus.json
│   └── README.md
├── scripts/
│   ├── build_rag_index.py
│   └── package_blender_extension.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EVALUATION.md
│   ├── INSTALLATION.md
│   └── MODELS.md
├── requirements.txt
├── .gitignore
└── run_evals.sh
```

## Requirements

- Blender
- Python environment for the backend
- Ollama running locally
- A local chat model (the current default is `qwen3:4b-instruct`)

Python packages used by the backend are listed in `requirements.txt`.

Blender supplies `bpy`, `bmesh`, and `mathutils`; do not install those into the
backend environment from this requirements file.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/blender-ai-copilot.git
cd blender-ai-copilot
```

### 2. Install backend dependencies

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

### 3. Install Ollama and choose a model

The default model is:

```text
qwen3:4b-instruct
```

Pull it with:

```bash
ollama pull qwen3:4b-instruct
```

You are **not locked to this model**. Any Ollama model that supports reliable
tool/function calling can be selected with `--model`.

Example with a stronger model:

```bash
ollama pull qwen3:8b

python -m src.agent \
  --project-root . \
  --host 127.0.0.1 \
  --port 8765 \
  --ollama-url http://127.0.0.1:11434 \
  --model qwen3:8b
```

See [`docs/MODELS.md`](docs/MODELS.md).

### 4. Build the local RAG index

The generated documentation index is intentionally not committed to GitHub. Build it locally:

```bash
python scripts/build_rag_index.py
```

This fetches the curated official Blender Manual/Python API pages and creates the local
FAISS/chunk artifacts expected by the backend. For an offline installation smoke test:

```bash
python scripts/build_rag_index.py --builtin-only
```

See [`rag/README.md`](rag/README.md) for version pinning and rebuild rules.

### 5. Start the backend

```bash
python -m src.agent \
  --project-root . \
  --host 127.0.0.1 \
  --port 8765 \
  --ollama-url http://127.0.0.1:11434 \
  --model qwen3:4b-instruct
```

Leave this terminal running while Blender is open.

### 6. Build the Blender extension package

```bash
python scripts/package_blender_extension.py
```

This creates:

```text
dist/blender_ai_copilot_extension.zip
```

### 7. Install it in Blender

The extension requires Blender 4.2+.

In Blender:

```text
Edit -> Preferences -> Add-ons / Extensions
-> Install from Disk
-> select dist/blender_ai_copilot_extension.zip
-> enable Blender AI Copilot
```

Then open a 3D Viewport:

```text
N -> Copilot
```

The default backend URL is:

```text
http://127.0.0.1:8765
```

Click **Check** in the Copilot panel before using the agent.

For a full walkthrough and troubleshooting, see
[`docs/INSTALLATION.md`](docs/INSTALLATION.md).

### Using another Ollama server

You can point the backend at Ollama on another machine:

```bash
python -m src.agent \
  --project-root . \
  --ollama-url http://192.168.1.50:11434 \
  --model YOUR_MODEL_NAME
```

The current backend speaks Ollama's `/api/chat` tool-calling format directly.
Other model providers would require a small provider adapter.

## Example requests

```text
Create a UV sphere named Product at (0, 0, 1).

Make it blue.

Shade it smooth.

Create a camera named ProductCamera at (0, -8, 4),
aim it at Product, and make it active.

Create a white AREA light named KeyLight at (4, -3, 6)
with energy 1000 and aim it at Product.

Set the render output to product.png and render to file.
```

## Safety design

The model does not receive arbitrary Python execution.

Instead, it chooses from semantic tools such as:

```text
create_uv_sphere
assign_material
shade_smooth
aim_camera_at_object
aim_light_at_object
set_render_output
render_scene
```

The controller classifies operations by domain and behavior, validates tool
arguments, requests approval for higher-risk operations, and verifies resulting
Blender state.

For non-idempotent terminal actions such as rendering, a timeout is treated as
an unknown execution state rather than blindly replaying the action.

## Conversation memory

The Blender UI keeps the visible chat history, while the local model receives a
bounded context.

The backend also maintains structured referential memory for entities such as:

```text
last_object
last_material
last_camera
last_light
last_render_output
```

This supports follow-up requests like:

```text
Create a sphere named Product.
Make it blue.
Shade it smooth.
```

without sending the entire conversation to the local model.

## RAG

The project includes local Blender-documentation retrieval using:

- BM25
- FAISS
- sentence-transformer embeddings
- cross-encoder reranking

This is used for Blender knowledge questions without giving the model unrestricted
internet access. The GitHub repository stores the source manifest and index-builder code,
not copied documentation or generated vectors. Build the index with
`python scripts/build_rag_index.py`. Changing the main Ollama LLM does **not** require a
RAG rebuild; changing the embedding model does.

## Evaluation harness

The evaluation harness calls the same backend `/chat` and `/approve` API used by
the Blender UI.

Run the full suite:

```bash
python -m evals.runner   --suite evals/suites/full.json   --backend http://127.0.0.1:8765   --auto-approve
```

The harness measures, among other things:

- task and turn success
- semantic false-success
- goal completion
- tool execution
- verification
- reference resolution
- tool repetition
- render-at-most-once behavior
- latency and LLM calls

See `docs/EVALUATION.md`.

## Current status

This repository is frozen as a learning/portfolio milestone.

It demonstrates the architecture and engineering patterns rather than aiming
for complete Blender coverage or production reliability. Known edge cases and
the latest evaluation snapshot are documented in `docs/EVALUATION.md`.

Model weights, generated indexes, logs, renders, and local environments are not included in the repository. See .gitignore and the installation guide for details.

## License

Code in this repository is released under the [MIT License](LICENSE).

Original GAIA questions and attachments remain subject to their original terms.
