import asyncio
import datetime

from src.infrastructure.database.child_chunks_db import ChildChunksDB
from src.infrastructure.database.parent_store_manager import ParentStoreManager
from src.services.embedding import EmbeddingService
from src.services.indexing import IndexingService
from src.services.retrieval import RetrievalService

from tavily import TavilyClient

from langchain_core.tools import tool

from src.core.config import config

SENSITIVE_TOOLS: set[str] = {"reindex_vault"}


class _Services:
    """Lazy singleton — connects to Qdrant/SQLite only on first access,
    off the event loop via asyncio.to_thread."""

    _instance: "_Services | None" = None

    def __init__(self) -> None:
        self.retrieval_service = None
        self.indexing_service = None
        self.parent_store = None
        self.child_db = None
        self._ready = False

    @classmethod
    async def get(cls) -> "_Services":
        if cls._instance is None:
            cls._instance = cls()
        if not cls._instance._ready:
            await asyncio.to_thread(cls._instance._init)
        return cls._instance

    def _init(self) -> None:
        self.child_db = ChildChunksDB(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            grpc_port=config.QDRANT_GRPC_PORT,
            prefer_grpc=config.QDRANT_PREFER_GRPC,
            collection_name=config.QDRANT_COLLECTION_NAME,
            vector_size=config.EMBED_DIMENSIONS,
        )
        self.parent_store = ParentStoreManager(db_path=config.SQLITE_DB_PATH)
        embedding_service = EmbeddingService()
        self.retrieval_service = RetrievalService(
            self.child_db, self.parent_store, embedding_service
        )
        self.indexing_service = IndexingService()
        self._ready = True


# --- Knowledge-base tools (lazy services, fully async) ---


@tool
async def search_knowledge(query: str) -> str:
    """Search the personal knowledge base for information relevant to
    the query. Returns grounding context and Obsidian source citations.
    Use this tool whenever the user asks a question that might be
    answered by their notes.
    Always cite the source number [i] in your answer."""

    svc = await _Services.get()
    result = await svc.retrieval_service.retrieve(query)

    if not result.parent_chunks:
        return "No relevant information found in the knowledge base."

    document_ids = list({c.document_id for c in result.parent_chunks})
    index_states = await asyncio.to_thread(
        svc.parent_store.get_index_states_batch, document_ids
    )
    doc_id_to_path = {s.document_id: s.source_path for s in index_states}

    lines: list[str] = []
    for i, chunk in enumerate(result.parent_chunks, start=1):
        source = doc_id_to_path.get(chunk.document_id, "unknown")
        heading = chunk.chunk_metadata.heading_string
        label = f"{source}"
        if heading:
            label += f" > {heading}"
        lines.append(f"[{i}] Source: {label}\n{chunk.content}")

    context_block = "\n\n---\n\n".join(lines)

    ref_lines: list[str] = []
    for i, c in enumerate(result.citations, start=1):
        heading_str = " > ".join(c.heading_path) if c.heading_path else ""
        display = c.source_path or "unknown"
        if heading_str:
            display += f" > {heading_str}"
        ref_lines.append(f"[{i}] {display} — {c.obsidian_link}")
    references = "\n\nReferences:\n" + "\n".join(ref_lines) if ref_lines else ""

    return f"{context_block}{references}"


@tool
async def reindex_vault(force: bool = False) -> str:
    """Re-index the Obsidian vault. Scans all markdown files, detects
    changes, embeds new/updated content, and removes deleted documents.
    Set force=True to re-index everything regardless of changes.

    WARNING: This calls the embedding API and may take several minutes
    for large vaults."""

    svc = await _Services.get()
    stats = await svc.indexing_service.run(force=force)
    return (
        f"Indexing complete: "
        f"{stats.indexed} indexed, "
        f"{stats.skipped} skipped (unchanged), "
        f"{stats.deleted} deleted (orphaned), "
        f"{stats.failed} failed "
        f"(total {stats.total} documents scanned)."
    )


@tool
async def get_document_info(source_path: str) -> str:
    """Get indexing metadata for a specific document in the vault.
    source_path is the relative path from the vault root, e.g.
    'Projects/My Note.md'. Use this to check if a document is indexed
    and when it was last processed."""

    svc = await _Services.get()
    match = await asyncio.to_thread(
        svc.parent_store.get_index_state_by_path, source_path
    )

    if match is None:
        return f"Document '{source_path}' is not indexed."

    last_indexed = datetime.datetime.fromtimestamp(match.mtime).strftime(
        "%Y-%m-%d %H:%M"
    )
    return (
        f"Document: {match.source_path}\n"
        f"Last indexed: {last_indexed}\n"
        f"Chunks: {len(match.chunk_ids)} total\n"
        f"Chunker version: {match.chunker_version}\n"
        f"Embed model: {match.embed_model}"
    )


@tool
async def get_vault_status() -> str:
    """Get a summary of the current state of the knowledge base:
    number of documents indexed, total chunks, and Qdrant collection
    health. Use this to answer questions like 'how many notes do you
    have indexed?'."""

    svc = await _Services.get()
    all_states = await asyncio.to_thread(svc.parent_store.get_all_index_states)
    total_docs = len(all_states)
    total_chunks = sum(len(s.chunk_ids) for s in all_states)

    try:
        collection = await asyncio.to_thread(svc.child_db.collection_info)
        qdrant_status = (
            f"{collection.get('points_count', 0)} vectors "
            f"(status: {collection.get('status', 'unknown')})"
        )
    except Exception:
        qdrant_status = "unavailable"

    if total_docs == 0:
        return "The knowledge base is empty. Run reindex_vault to index your vault."

    return (
        f"Knowledge base status:\n"
        f"  Documents indexed: {total_docs}\n"
        f"  Total chunks: {total_chunks}\n"
        f"  Qdrant vectors: {qdrant_status}"
    )


# --- Web search fallback ---


@tool
async def web_search(query: str) -> str:
    """Search the web for real-time information not found in the knowledge base.
    Use this only when the user explicitly asks for external information or
    the vault has no relevant results."""

    def _search(q: str) -> str:
        client = TavilyClient()
        results = client.search(q)
        output_lines: list[str] = []
        for r in results.get("results", []):
            output_lines.append(
                f"- {r['title']}: {r['content'][:300]}\n  URL: {r['url']}"
            )
        return "\n".join(output_lines) if output_lines else "No web results found."

    return await asyncio.to_thread(_search, query)


# All tools — safe to reference at import time (no DB connections made)
all_tools = [search_knowledge, reindex_vault, get_document_info, get_vault_status, web_search]
