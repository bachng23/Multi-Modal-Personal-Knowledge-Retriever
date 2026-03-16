import datetime

from langchain_core.tools import tool

from src.infrastructure.database.child_chunks_db import ChildChunksDB
from src.infrastructure.database.parent_store_manager import ParentStoreManager
from src.services.indexing import IndexingService
from src.services.retrieval import RetrievalService


def create_tools(
    retrieval_service: RetrievalService,
    indexing_service: IndexingService,
    parent_store: ParentStoreManager,
    child_db: ChildChunksDB,
) -> list:
    """Return the list of configured tools for the agent.

    All services are injected via closure so tools stay stateless
    and are easy to test with mock services.
    """

    @tool
    async def search_knowledge(query: str) -> str:
        """Search the personal knowledge base for information relevant to
        the query. Returns grounding context and Obsidian source citations.
        Use this tool whenever the user asks a question that might be
        answered by their notes.
        Always cite the source number [i] in your answer."""

        result = await retrieval_service.retrieve(query)

        if not result.parent_chunks:
            return "No relevant information found in the knowledge base."

        lines: list[str] = []
        for i, chunk in enumerate(result.parent_chunks, start=1):
            heading = chunk.chunk_metadata.heading_string
            label = heading if heading else f"Context {i}"
            lines.append(f"[{i}] {label}\n{chunk.content}")

        context_block = "\n\n---\n\n".join(lines)
        citation_links = ", ".join(c.obsidian_link for c in result.citations)
        sources = f"\nSources: {citation_links}" if citation_links else ""

        return f"{context_block}{sources}"

    @tool
    async def reindex_vault(force: bool = False) -> str:
        """
        Re-index the Obsidian vault. Scans all markdown files, detects
        changes, embeds new/updated content, and removes deleted documents.
        Set force=True to re-index everything regardless of changes.

        WARNING: This calls the embedding API and may take several minutes
        for large vaults. This tool requires user confirmation before running
        and is handled by the agent's human-in-the-loop mechanism.
        """

        stats = await indexing_service.run(force=force)
        return (
            f"Indexing complete: "
            f"{stats.indexed} indexed, "
            f"{stats.skipped} skipped (unchanged), "
            f"{stats.deleted} deleted (orphaned), "
            f"{stats.failed} failed "
            f"(total {stats.total} documents scanned)."
        )

    @tool
    def get_document_info(source_path: str) -> str:
        """Get indexing metadata for a specific document in the vault.
        source_path is the relative path from the vault root, e.g.
        'Projects/My Note.md'. Use this to check if a document is indexed
        and when it was last processed."""

        match = parent_store.get_index_state_by_path(source_path)

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
    def get_vault_status() -> str:
        """Get a summary of the current state of the knowledge base:
        number of documents indexed, total chunks, and Qdrant collection
        health. Use this to answer questions like 'how many notes do you
        have indexed?'."""

        all_states = parent_store.get_all_index_states()
        total_docs = len(all_states)
        total_chunks = sum(len(s.chunk_ids) for s in all_states)

        try:
            collection = child_db.collection_info()
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

    return [search_knowledge, reindex_vault, get_document_info, get_vault_status]
