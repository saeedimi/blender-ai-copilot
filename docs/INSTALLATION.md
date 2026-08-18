# Installation

This guide is for someone cloning the project for the first time.

## Requirements

- Blender 4.2+
- Python 3
- Ollama
- Enough RAM/VRAM for the selected Ollama model

The repository does not contain model weights.

## Clone

```bash
git clone https://github.com/YOUR_USERNAME/blender-ai-copilot.git
cd blender-ai-copilot
```

## Backend environment

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

## Ollama model

Default:

```bash
ollama pull qwen3:4b-instruct
```

A different compatible tool-calling model may be used with `--model`.

## Build the RAG index

The repository intentionally excludes generated FAISS vectors and copied documentation.
Build the local index before starting the backend:

```bash
python scripts/build_rag_index.py
```

If you only want an offline smoke test of the installation:

```bash
python scripts/build_rag_index.py --builtin-only
```

The normal command uses the curated official Blender documentation manifest in `rag/`.

## Start backend

```bash
python -m src.agent \
  --project-root . \
  --host 127.0.0.1 \
  --port 8765 \
  --ollama-url http://127.0.0.1:11434 \
  --model qwen3:4b-instruct
```

Keep this terminal open.

## Package the Blender extension

```bash
python scripts/package_blender_extension.py
```

The generated file is:

```text
dist/blender_ai_copilot_extension.zip
```

Do not install the whole GitHub repository ZIP into Blender. Install the
extension ZIP generated above.

## Install in Blender

In Blender 4.2+:

```text
Edit
 -> Preferences
 -> Add-ons / Extensions
 -> Install from Disk
 -> dist/blender_ai_copilot_extension.zip
```

Enable **Blender AI Copilot** if necessary.

Then open a 3D Viewport and press:

```text
N -> Copilot
```

The panel defaults to:

```text
http://127.0.0.1:8765
```

Click **Check**. This verifies the backend and lets the Blender extension learn
the project root used by the controlled bridge.

## First test

```text
Create a UV sphere named Product at (0, 0, 1).
```

Then:

```text
Make it blue.
```

## Troubleshooting

**Backend unreachable**  
Confirm the backend terminal is running and the Copilot panel uses
`http://127.0.0.1:8765`.

**Ollama unreachable**  
Confirm Ollama is running and `--ollama-url` is correct.

**Model talks but does not operate Blender**  
The model must support/reliably produce Ollama tool calls. Plain text generation
alone is not sufficient for this agent.

**Blender command timeouts**  
Keep Blender open and click **Check** in the Copilot panel before sending tasks.

**RAG model downloads**  
Embedding and reranking model weights are intentionally not committed. Your ML
libraries may download/cache them on first use.

## Safety

The model is not given arbitrary Python execution. It receives semantic Blender
tools, and higher-risk operations can require approval.
