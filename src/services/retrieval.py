from loguru import logger

from src.core.config import config
from src.core.models.citation import Citation
from src.core.models.chunk import ChunkSearchResult
from src.core.models.retrieval import RetrievalResult
from src.infrastructure.database.child_chunks_db import ChildChunksDB
from src.infrastructure.database.parent_store_manager import ParentStoreManager
from src.services.embedding import EmbeddingService


class RetrievalService:
    """
    Query -> embed -> search children -> fetch parents -> build citations.

    Uses a parent-child retrieval strategy:
    - Search is done on small child chunks (precise vector match).
    - Context returned to the LLM is the larger parent chunks (full context).
    - One batch SQL query fetches all parents — no N+1 problem.
    """

    def __init__(
        self,
        child_db: ChildChunksDB | None = None,
        parent_store: ParentStoreManager | None = None,
        embedding_service: EmbeddingService | None = None,
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

    async def retrieve(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
    ) -> RetrievalResult:
        """
        Run a full retrieval query.

        Args:
            query: Natural-language question or search string.
            top_k: Number of child chunks to retrieve from Qdrant.
            score_threshold: Minimum cosine similarity score (0-1).
                             Pass None to return all results regardless of score.

        Returns:
            RetrievalResult with parent chunks for LLM context and citations.
        """

        # Embed the query
        query_embedding = await self._embedding_service.embed_query(query)

        # Vector search on child chunks
        child_results = self._child_db.search(
            query_embedding,
            limit=top_k,
        )

        if not child_results:
            logger.info(f"No results found for query: '{query}'")
            return RetrievalResult(
                query=query,
                parent_chunks=[],
                child_results=[],
                citations=[],
            )

        # Collect unique parent IDs (many children can share one parent)
        parent_ids = list({
            r.parent_chunk_id
            for r in child_results
            if r.parent_chunk_id
        })

        # Batch fetch parent chunks from SQLite (single query)
        parent_chunks = self._parent_store.get_parents_batch(parent_ids)


        # Resolve document_id -> source_path for citation building
        document_ids = list({r.document_id for r in child_results})
        index_states = self._parent_store.get_index_states_batch(document_ids)
        doc_id_to_path = {s.document_id: s.source_path for s in index_states}

        # Build deduplicated citations ordered by relevance score
        citations = self._build_citations(child_results, doc_id_to_path)

        logger.info(
            f"Query '{query}': "
            f"{len(child_results)} child hit(s), "
            f"{len(parent_chunks)} parent(s), "
            f"{len(citations)} citation(s)"
        )

        return RetrievalResult(
            query=query,
            parent_chunks=parent_chunks,
            child_results=child_results,
            citations=citations,
        )
    

    @staticmethod
    def _build_citations(
        child_results: list[ChunkSearchResult],
        doc_id_to_path: dict[str, str],
    ) -> list[Citation]:
        """
        Build a deduplicated list of Citations from child search results.

        - Deduplication key: (source_path, heading, block_id) — so two child
        chunks from the same section produce only one citation.
        - Results are processed in score order (highest first), so the most
        relevant section wins when deduplicating.
        """
        
        seen: set[tuple] = set()
        citations: list[Citation] = []

        for result in child_results:
            source_path = doc_id_to_path.get(result.document_id, "")
            key = (source_path, tuple(result.heading_path or []), result.block_id)

            if key in seen:
                continue
            seen.add(key)

            citations.append(Citation(
                source_path=source_path,
                heading_path=result.heading_path,
                block_id=result.block_id,
                quote=result.content[:200].strip(),
            ))

        return citations
