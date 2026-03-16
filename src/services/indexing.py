import asyncio
from dataclasses import dataclass

from loguru import logger

from src.core.config import config
from src.core.models.document import Document
from src.core.models.indexState import IndexState
from src.core.utils import stable_hashing
from src.infrastructure.database.child_chunks_db import ChildChunksDB
from src.infrastructure.database.parent_store_manager import ParentStoreManager
from src.infrastructure.filesystem.obsidian_parser import parse_obsidian_document
from src.infrastructure.filesystem.vault_scanner import VaultScanner
from src.services.chunking import ParentChildChunker
from src.services.embedding import EmbeddingService
from src.services.obsidian_preprocessor import ObsidianPreprocessor


@dataclass
class IndexStats:
    """Summary of a single indexing run."""

    total: int = 0
    indexed: int = 0
    skipped: int = 0
    deleted: int = 0
    failed: int = 0

    def __str__(self) -> str:
        return (
            f"total={self.total}, indexed={self.indexed}, "
            f"skipped={self.skipped}, deleted={self.deleted}, "
            f"failed={self.failed}"
        )


class IndexingService:
    """Orchestrates the full indexing flow for an Obsidian vault.

    Flow for each document:
        scan vault -> parse -> preprocess -> chunk -> embed -> store

    Incremental: only re-indexes documents whose content hash, chunker
    version, or embed model has changed since the last run.

    Orphan cleanup: documents removed from the vault are deleted from
    both Qdrant and SQLite automatically.
    """

    def __init__(
        self,
        child_db: ChildChunksDB | None = None,
        parent_store: ParentStoreManager | None = None,
        embedding_service: EmbeddingService | None = None,
        vault_scanner: VaultScanner | None = None,
    ) -> None:
        self._child_db = child_db or ChildChunksDB(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            grpc_port=config.QDRANT_GRPC_PORT,
            prefer_grpc=config.QDRANT_PREFER_GRPC,
            collection_name=config.QDRANT_COLLECTION_NAME,
            vector_size=config.EMBED_DIMENSIONS,
        )
        self._parent_store = parent_store or ParentStoreManager(
            db_path=config.SQLITE_DB_PATH
        )
        self._embedding_service = embedding_service or EmbeddingService()
        self._vault_scanner = vault_scanner or VaultScanner()
        self._preprocessor = ObsidianPreprocessor(
            model_name=config.TIKTOKEN_ENCODING
        )
        self._chunker = ParentChildChunker()

    async def run(
        self,
        vault_dir: str | None = None,
        force: bool = False,
    ) -> IndexStats:
        """Index the vault incrementally.

        Args:
            vault_dir: Override vault path. Defaults to config.VAULT_PATH.
            force: Re-index all documents even if unchanged.

        Returns:
            IndexStats with counts for this run.
        """
        stats = IndexStats()

        # Scan vault
        documents = self._vault_scanner.load_vault(vault_dir)
        stats.total = len(documents)

        if not documents:
            logger.warning("No documents found in vault. Aborting.")
            return stats

        # Clean up orphaned documents (in DB but no longer in vault)
        stats.deleted = self._cleanup_orphans(
            vault_ids={doc.id for doc in documents}
        )

        # Index all documents concurrently, capped by semaphore to avoid overwhelming the OpenRouter rate limit.
        semaphore = asyncio.Semaphore(config.INDEX_CONCURRENCY)

        async def _process(doc: Document) -> None:
            async with semaphore:
                try:
                    content_hash = stable_hashing(doc.content)
                    needs = force or self._parent_store.needs_reindex(
                        doc.id,
                        content_hash,
                        config.CHUNKER_VERSION,
                        config.EMBED_MODEL,
                    )

                    if not needs:
                        stats.skipped += 1
                        return

                    await self._index_document(doc, content_hash)
                    stats.indexed += 1

                except Exception as e:
                    logger.error(f"Failed to index '{doc.source_path}': {e}")
                    stats.failed += 1

        await asyncio.gather(*[_process(doc) for doc in documents], return_exceptions=False)

        logger.info(f"Indexing complete: {stats}")
        return stats

    def _cleanup_orphans(self, vault_ids: set[str]) -> int:
        """Delete data for documents that no longer exist in the vault.

        Returns the number of documents deleted.
        """
        all_states = self._parent_store.get_all_index_states()
        orphan_ids = [
            s.document_id
            for s in all_states
            if s.document_id not in vault_ids
        ]

        for doc_id in orphan_ids:
            self._child_db.delete_by_document(doc_id)
            self._parent_store.delete_document_data(doc_id)
            logger.info(f"Deleted orphaned document: {doc_id}")

        if orphan_ids:
            logger.info(f"Cleaned up {len(orphan_ids)} orphaned document(s).")

        return len(orphan_ids)

    async def _index_document(self, doc: Document, content_hash: str) -> None:
        """Full pipeline for a single document: delete old data, re-chunk,
        embed, and persist everything."""
        logger.info(f"Indexing: {doc.source_path}")

        # Delete stale data before re-indexing
        self._child_db.delete_by_document(doc.id)
        self._parent_store.delete_document_data(doc.id)

        # parse -> preprocess -> chunk
        parsed = parse_obsidian_document(doc.content)
        sections = self._preprocessor.preprocess(parsed)
        chunks = self._chunker.chunk(sections, doc.id)

        if not chunks:
            logger.warning(f"No chunks produced for '{doc.source_path}', skipping.")
            return

        # embed (attaches vectors to child chunks in-place)
        chunks = await self._embedding_service.embed_chunks(chunks)

        # persist
        parents = [c for c in chunks if c.chunk_metadata.chunk_type == "parent"]
        children = [c for c in chunks if c.chunk_metadata.chunk_type == "child"]

        self._parent_store.save_parents(parents)
        self._child_db.add_chunks(children)

        # record index state for incremental re-index detection
        state = IndexState(
            document_id=doc.id,
            source_type=doc.source_type,
            source_path=doc.source_path,
            mtime=doc.mtime,
            content_hash=content_hash,
            chunk_ids=[c.id for c in chunks],
            chunker_version=config.CHUNKER_VERSION,
            embed_provider="OpenRouter",
            embed_model=config.EMBED_MODEL,
        )
        self._parent_store.save_index_state(state)

        logger.debug(
            f"'{doc.source_path}': "
            f"{len(parents)} parent(s), {len(children)} child(ren) indexed."
        )
