#!/usr/bin/env python3
"""Build the Blender Copilot RAG artifacts from official docs or the bundled mini corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE_WORDS = 180
OVERLAP_WORDS = 40
REQUEST_HEADERS = {
    "User-Agent": "BlenderCopilot-RAG/0.8.0.2 (local documentation index builder)"
}


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str
    metadata: Dict[str, Any]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any]


def stable_id(*parts: Any) -> str:
    raw = "||".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_blender_page_text(raw_html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "beautifulsoup4 is required. Run: pip install -r requirements.txt"
        ) from exc
    soup = BeautifulSoup(raw_html, "html.parser")
    for element in soup.select(
        "script, style, nav, footer, .related, .sphinxsidebar, "
        ".wy-nav-side, .wy-side-nav-search, .headerlink"
    ):
        element.decompose()
    main = (
        soup.select_one("main")
        or soup.select_one("article")
        or soup.select_one("div.document")
        or soup.select_one("div.body")
        or soup.body
        or soup
    )
    return clean_text(main.get_text("\n"))


def load_manifest(project_root: Path, manual_version: str, api_version: str) -> List[dict]:
    manifest_path = project_root / "rag" / "source_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manual_base = f"https://docs.blender.org/manual/en/{manual_version}"
    api_base = f"https://docs.blender.org/api/{api_version}"
    sources: List[dict] = []
    for item in raw["manual_sources"]:
        spec = dict(item)
        spec.update(
            url=f"{manual_base}/{item['path']}",
            source_type="manual",
            manual_version=manual_version,
            api_version=None,
        )
        sources.append(spec)
    for item in raw["api_sources"]:
        spec = dict(item)
        spec.update(
            url=f"{api_base}/{item['path']}",
            source_type="python_api",
            manual_version=None,
            api_version=api_version,
        )
        sources.append(spec)
    return sources


def fetch_document(spec: dict, cache_dir: Path, force_refresh: bool, timeout: int = 30) -> Document:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests is required. Run: pip install -r requirements.txt"
        ) from exc
    cache_path = cache_dir / f"{spec['doc_id']}.html"
    from_cache = cache_path.exists() and not force_refresh
    if from_cache:
        raw_html = cache_path.read_text(encoding="utf-8", errors="ignore")
    else:
        last_error = None
        for attempt in range(1, 4):
            try:
                response = requests.get(
                    spec["url"], timeout=timeout, headers=REQUEST_HEADERS
                )
                response.raise_for_status()
                raw_html = response.text
                cache_path.write_text(raw_html, encoding="utf-8")
                break
            except Exception as exc:  # noqa: BLE001 - surface network failures clearly
                last_error = exc
                if attempt < 3:
                    time.sleep(float(attempt))
        else:
            raise RuntimeError(f"Failed to fetch {spec['url']}: {last_error}")

    text = extract_blender_page_text(raw_html)
    if len(text) < 200:
        raise ValueError(f"Extracted text is unexpectedly short for {spec['url']}")
    return Document(
        doc_id=spec["doc_id"],
        title=spec["title"],
        text=text,
        source=spec["url"],
        metadata={
            "source_type": spec["source_type"],
            "topic": spec["topic"],
            "official": True,
            "from_cache": from_cache,
            "manual_version": spec.get("manual_version"),
            "api_version": spec.get("api_version"),
        },
    )


def load_builtin_documents(project_root: Path) -> List[Document]:
    path = project_root / "rag" / "builtin_corpus.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(
            doc_id=item["doc_id"],
            title=item["title"],
            text=clean_text(item["text"]),
            source=item["source"],
            metadata={
                "source_type": "builtin",
                "topic": item["topic"],
                "official": False,
            },
        )
        for item in raw
    ]


def chunk_document(document: Document) -> List[Chunk]:
    words = document.text.split()
    chunks: List[Chunk] = []
    start = 0
    chunk_index = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE_WORDS, len(words))
        body = " ".join(words[start:end])
        topic = document.metadata.get("topic", "")
        chunk_text = f"Document: {document.title}\nTopic: {topic}\n{body}"
        metadata = dict(document.metadata)
        metadata.update(
            {
                "document_title": document.title,
                "document_source": document.source,
                "chunk_start_word": start,
                "chunk_end_word": end,
            }
        )
        chunks.append(
            Chunk(
                chunk_id=stable_id(document.doc_id, chunk_index, body[:100]),
                doc_id=document.doc_id,
                title=document.title,
                text=chunk_text,
                source=document.source,
                chunk_index=chunk_index,
                metadata=metadata,
            )
        )
        if end == len(words):
            break
        start = end - OVERLAP_WORDS
        chunk_index += 1
    return chunks


def build_chunks(documents: List[Document]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root (defaults to the parent of scripts/).",
    )
    parser.add_argument("--manual-version", default="latest")
    parser.add_argument("--api-version", default="current")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument(
        "--builtin-only",
        action="store_true",
        help="Build a small offline educational index without downloading Blender docs.",
    )
    parser.add_argument(
        "--allow-builtin-fallback",
        action="store_true",
        help="If every official page fails, build the bundled mini corpus instead.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate ingestion/chunking but do not load the embedding model or write an index.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    rag_root = project_root / "rag"
    cache_dir = rag_root / "cache" / "blender_official"
    index_dir = rag_root / "indexes"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    failures: List[dict] = []
    if args.builtin_only:
        documents = load_builtin_documents(project_root)
        build_mode = "builtin_only"
    else:
        sources = load_manifest(project_root, args.manual_version, args.api_version)
        documents = []
        print(f"Fetching {len(sources)} curated official Blender documentation pages...")
        for index, spec in enumerate(sources, start=1):
            print(f"[{index:02d}/{len(sources):02d}] {spec['doc_id']}")
            try:
                document = fetch_document(
                    spec,
                    cache_dir=cache_dir,
                    force_refresh=args.force_refresh,
                )
                documents.append(document)
                print(f"  OK: {len(document.text):,} characters")
            except Exception as exc:  # noqa: BLE001 - collect per-page failures
                failures.append(
                    {"doc_id": spec["doc_id"], "url": spec["url"], "error": repr(exc)}
                )
                print(f"  FAILED: {exc}")
            if args.request_delay and index < len(sources):
                time.sleep(args.request_delay)
        if not documents:
            if args.allow_builtin_fallback:
                print("No official pages loaded; using bundled educational corpus.")
                documents = load_builtin_documents(project_root)
                build_mode = "builtin_fallback"
            else:
                print(
                    "No official Blender pages could be loaded. Re-run with network access, "
                    "or use --builtin-only for the offline mini corpus.",
                    file=sys.stderr,
                )
                return 2
        else:
            build_mode = "official"

    chunks = build_chunks(documents)
    if not chunks:
        print("No chunks were produced.", file=sys.stderr)
        return 2

    if args.dry_run:
        print("\nRAG dry run passed")
        print(f"  Build mode: {build_mode}")
        print(f"  Documents:  {len(documents)}")
        print(f"  Chunks:     {len(chunks)}")
        return 0

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "RAG build dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss_path = index_dir / "blender_chunks.faiss"
    chunks_path = index_dir / "chunks.json"
    meta_path = index_dir / "index_meta.json"

    faiss.write_index(index, str(faiss_path))
    chunks_path.write_text(
        json.dumps([asdict(chunk) for chunk in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    meta = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_mode": build_mode,
        "embedding_model": args.embedding_model,
        "manual_version": None if args.builtin_only else args.manual_version,
        "api_version": None if args.builtin_only else args.api_version,
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "overlap_words": OVERLAP_WORDS,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "faiss_vectors": int(index.ntotal),
        "failed_sources": failures,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nRAG index ready")
    print(f"  Documents: {len(documents)}")
    print(f"  Chunks:    {len(chunks)}")
    print(f"  FAISS:     {faiss_path}")
    print(f"  Chunks:    {chunks_path}")
    print(f"  Metadata:  {meta_path}")
    if failures:
        print(f"  Source failures: {len(failures)} (recorded in index_meta.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
