# Changing the model

The Copilot is not tied to `qwen3:4b-instruct`.

## Select another Ollama model

Make the model available in Ollama:

```bash
ollama pull MODEL_NAME
```

Start the backend with:

```bash
python -m src.agent \
  --project-root . \
  --ollama-url http://127.0.0.1:11434 \
  --model MODEL_NAME
```

Example:

```bash
ollama pull qwen3:8b

python -m src.agent \
  --project-root . \
  --ollama-url http://127.0.0.1:11434 \
  --model qwen3:8b
```

## Compatibility requirement

The backend sends semantic function schemas in the `tools` field of Ollama's
`/api/chat` endpoint and expects returned `tool_calls`.

Therefore, a replacement model should support tool/function calling through
Ollama and follow tool schemas reliably.

## What a better model may improve

A stronger model may improve:

- long-request decomposition
- tool selection
- argument extraction
- planning consistency
- final natural-language responses

It does not replace the controller. Validation, safety, approvals, deterministic
normalization, verification, goal tracking, and render replay protection remain
in the backend.

## Resource trade-off

A larger model generally needs more RAM/VRAM and may be slower.

The backend starts with an 8192-token Ollama context and can expand to 32768
when needed.

Optional environment overrides:

```bash
export BLENDER_COPILOT_NUM_CTX=8192
export BLENDER_COPILOT_MAX_NUM_CTX=32768
```

## Remote Ollama

Ollama may run on another machine:

```bash
python -m src.agent \
  --project-root . \
  --ollama-url http://192.168.1.50:11434 \
  --model MODEL_NAME
```

The Blender extension talks to the Copilot backend, not directly to Ollama.

## Other providers

The current code uses Ollama's API directly. OpenAI, Anthropic, Google, or
another provider would need a provider adapter.

## Main LLM vs RAG models

The `--model` option changes the **main Ollama reasoning/tool-calling model**. You can switch
that model without rebuilding the RAG index.

The RAG stack separately uses an embedding model and a reranker. If you change the embedding
model, rebuild the FAISS index:

```bash
python scripts/build_rag_index.py --embedding-model YOUR_EMBEDDING_MODEL
```

Changing only the reranker does not invalidate the stored FAISS vectors.
