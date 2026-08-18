# RAG data and reproducibility

The GitHub repository intentionally does **not** contain copied Blender documentation,
embedding-model weights, or generated FAISS artifacts.

Instead, the repository contains:

- `source_manifest.json` — the curated official Blender Manual/Python API pages
- `builtin_corpus.json` — a tiny educational/offline corpus authored for this project
- `../scripts/build_rag_index.py` — reproducible ingestion, chunking, embedding, and FAISS persistence

## Build the normal official-docs index

From the repository root:

```bash
python scripts/build_rag_index.py
```

This downloads/caches the configured official Blender documentation pages, chunks them,
embeds them with `BAAI/bge-small-en-v1.5`, and writes:

```text
rag/indexes/blender_chunks.faiss
rag/indexes/chunks.json
rag/indexes/index_meta.json
```

The embedding model is downloaded/cached by `sentence-transformers` if it is not already
available locally.

## Offline smoke-test index

If you want to verify the pipeline without downloading the Blender documentation corpus:

```bash
python scripts/build_rag_index.py --builtin-only
```

This produces a much smaller educational index. It is useful for installation checks,
but it is **not equivalent** to the official-docs RAG corpus used during development.

To validate the bundled corpus/chunking without loading any ML model:

```bash
python scripts/build_rag_index.py --builtin-only --dry-run
```

## Pin documentation versions

For more repeatable results, pin the documentation versions instead of using the defaults
`latest` and `current`:

```bash
python scripts/build_rag_index.py \
  --manual-version YOUR_MANUAL_VERSION \
  --api-version YOUR_API_VERSION
```

The selected versions, embedding model, source failures, document count, and chunk count
are stored in `rag/indexes/index_meta.json`.

## Changing models

Changing the **main Ollama LLM** does not require rebuilding this index.

Changing the **embedding model** does require rebuilding it:

```bash
python scripts/build_rag_index.py --embedding-model YOUR_EMBEDDING_MODEL
```

The reranker is loaded at runtime by `src/rag.py` and is independent of the stored FAISS
vectors.

## Why generated artifacts are not committed

The generated index, cached HTML, and model caches are machine-generated and can be rebuilt.
Keeping them out of Git makes the repository small and avoids redistributing a copied Blender
documentation corpus.
