"""Retrieval-only Blender documentation RAG used as an agent knowledge tool."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json
import re
import threading
import time

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any]


def bm25_tokenize(text):
    return re.findall(r"[A-Za-z0-9_\.]+", text.lower())


class BlenderRAGRetriever:
    """
    Loads the persisted artifacts produced by Notebook 02.

    This component performs retrieval only. It never calls the generation LLM.
    """

    def __init__(
        self,
        project_root,
        embedding_model_name="BAAI/bge-small-en-v1.5",
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        bm25_top_k=15,
        dense_top_k=15,
        hybrid_top_k=20,
    ):
        self.project_root = Path(project_root).expanduser().resolve()
        self.index_dir = self.project_root / "rag" / "indexes"

        self.faiss_path = self.index_dir / "blender_chunks.faiss"
        self.chunks_path = self.index_dir / "chunks.json"
        self.meta_path = self.index_dir / "index_meta.json"

        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name

        self.bm25_top_k = int(bm25_top_k)
        self.dense_top_k = int(dense_top_k)
        self.hybrid_top_k = int(hybrid_top_k)

        self._lock = threading.Lock()

        self._validate_artifacts()
        self.index_metadata = self._load_index_metadata()
        self._validate_index_metadata()
        self.chunks = self._load_chunks()
        self.faiss_index = faiss.read_index(str(self.faiss_path))

        if self.faiss_index.ntotal != len(self.chunks):
            raise RuntimeError(
                "FAISS index size does not match chunks.json. "
                "Re-run Notebook 02 persistence."
            )

        self.bm25_corpus = [
            bm25_tokenize(chunk.text)
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(self.bm25_corpus)

        print(f"[RAG] Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        print(f"[RAG] Loading reranker: {self.reranker_model_name}")
        self.reranker = CrossEncoder(self.reranker_model_name)

        print(
            f"[RAG] Ready: {len(self.chunks)} chunks, "
            f"{self.faiss_index.ntotal} FAISS vectors"
        )

    def _validate_artifacts(self):
        missing = [
            path
            for path in (self.faiss_path, self.chunks_path)
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing RAG artifacts:\n"
                + "\n".join(str(path) for path in missing)
                + "\nBuild them from the repository root with:"
                + "\n  python scripts/build_rag_index.py"
                + "\nFor an offline smoke-test index use:"
                + "\n  python scripts/build_rag_index.py --builtin-only"
            )

    def _load_index_metadata(self):
        if not self.meta_path.exists():
            return {}
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Could not read RAG index metadata: {self.meta_path} | {exc}"
            ) from exc

    def _validate_index_metadata(self):
        built_with = self.index_metadata.get("embedding_model")
        if built_with and built_with != self.embedding_model_name:
            raise RuntimeError(
                "RAG index embedding model mismatch. "
                f"Index was built with '{built_with}' but runtime is configured for "
                f"'{self.embedding_model_name}'. Rebuild with scripts/build_rag_index.py "
                "using the same embedding model."
            )

    def _load_chunks(self):
        raw = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        return [Chunk(**item) for item in raw]

    def bm25_search(self, query, top_k=None):
        top_k = top_k or self.bm25_top_k
        scores = self.bm25.get_scores(bm25_tokenize(query))
        indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "rank": rank,
                "index": int(idx),
                "chunk": self.chunks[int(idx)],
                "score": float(scores[idx]),
                "retriever": "bm25",
            }
            for rank, idx in enumerate(indices, start=1)
        ]

    def dense_search(self, query, top_k=None):
        top_k = top_k or self.dense_top_k
        top_k = min(int(top_k), self.faiss_index.ntotal)

        query_embedding = (
            self.embedding_model
            .encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            .astype("float32")
        )

        scores, indices = self.faiss_index.search(query_embedding, top_k)

        return [
            {
                "rank": rank,
                "index": int(idx),
                "chunk": self.chunks[int(idx)],
                "score": float(score),
                "retriever": "dense",
            }
            for rank, (score, idx) in enumerate(
                zip(scores[0], indices[0]),
                start=1,
            )
            if idx >= 0
        ]

    @staticmethod
    def reciprocal_rank_fusion(ranked_lists, rrf_k=60, top_k=20):
        fused_scores = defaultdict(float)
        chunks = {}
        source_ranks = defaultdict(dict)

        for retriever_name, results in ranked_lists:
            for item in results:
                idx = item["index"]
                chunks[idx] = item["chunk"]
                fused_scores[idx] += 1.0 / (rrf_k + item["rank"])
                source_ranks[idx][retriever_name] = item["rank"]

        ranked = sorted(
            fused_scores.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]

        return [
            {
                "rank": rank,
                "index": idx,
                "chunk": chunks[idx],
                "score": float(score),
                "source_ranks": source_ranks[idx],
                "retriever": "hybrid_rrf",
            }
            for rank, (idx, score) in enumerate(ranked, start=1)
        ]

    def hybrid_search(self, query):
        sparse = self.bm25_search(query, self.bm25_top_k)
        dense = self.dense_search(query, self.dense_top_k)

        return self.reciprocal_rank_fusion(
            [
                ("bm25", sparse),
                ("dense", dense),
            ],
            top_k=self.hybrid_top_k,
        )

    def score_cross_encoder(self, query, candidates):
        if not candidates:
            return []

        pairs = [
            [query, item["chunk"].text]
            for item in candidates
        ]

        scores = self.reranker.predict(pairs)
        scored = []

        for item, score in zip(candidates, scores):
            new_item = dict(item)
            new_item["hybrid_rank"] = item["rank"]
            new_item["reranker_score"] = float(score)
            scored.append(new_item)

        scored.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        for rank, item in enumerate(scored, start=1):
            item["rerank_rank"] = rank

        return scored

    @staticmethod
    def fuse_hybrid_and_reranker(
        scored_candidates,
        rrf_k=60,
        hybrid_weight=1.0,
        reranker_weight=1.0,
    ):
        fused = []

        for item in scored_candidates:
            hybrid_rank = item.get("hybrid_rank", 10**6)
            reranker_rank = item.get("rerank_rank", 10**6)

            score = (
                hybrid_weight / (rrf_k + hybrid_rank)
                + reranker_weight / (rrf_k + reranker_rank)
            )

            new_item = dict(item)
            new_item["hybrid_ce_fusion_score"] = float(score)
            fused.append(new_item)

        fused.sort(
            key=lambda item: item["hybrid_ce_fusion_score"],
            reverse=True,
        )

        return fused

    @staticmethod
    def diversify_by_document(results, top_k, max_chunks_per_doc=1):
        selected = []
        counts = defaultdict(int)

        for item in results:
            doc_id = item["chunk"].doc_id

            if counts[doc_id] >= max_chunks_per_doc:
                continue

            selected.append(item)
            counts[doc_id] += 1

            if len(selected) >= top_k:
                break

        return selected

    def retrieve(self, query, top_k=5):
        top_k = max(1, min(int(top_k), 8))

        with self._lock:
            candidates = self.hybrid_search(query)
            scored = self.score_cross_encoder(query, candidates)
            fused = self.fuse_hybrid_and_reranker(scored)

            return self.diversify_by_document(
                fused,
                top_k=top_k,
                max_chunks_per_doc=1,
            )

    def search(self, query, top_k=5):
        started = time.perf_counter()
        results = self.retrieve(query, top_k=top_k)

        passages = []

        for rank, item in enumerate(results, start=1):
            chunk = item["chunk"]

            passages.append(
                {
                    "citation": f"[DOC{rank}]",
                    "rank": rank,
                    "title": chunk.title,
                    "source": chunk.source,
                    "doc_id": chunk.doc_id,
                    "topic": chunk.metadata.get("topic"),
                    "source_type": chunk.metadata.get("source_type"),
                    "text": chunk.text,
                }
            )

        return {
            "success": True,
            "query": query,
            "results": passages,
            "retrieval_latency": time.perf_counter() - started,
        }
