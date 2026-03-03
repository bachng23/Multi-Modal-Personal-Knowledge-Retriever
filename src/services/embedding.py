from langchain_openai import OpenAIEmbeddings
from loguru import logger

from src.core.config import config
from src.core.models.chunk import Chunk, EmbeddingMetadata


class EmbeddingService:
    """Embedding service via OpenRouter.

    Uses OpenAI-compatible API through OpenRouter to embed chunks and queries.
    """

    def __init__(
        self,
        model: str = config.EMBED_MODEL,
        dimensions: int = config.EMBED_DIMENSIONS,
        batch_size: int = 512,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size

        self._embedder = OpenAIEmbeddings(
            model=model,
            dimensions=dimensions,
            openai_api_key=config.OPENROUTER_API_KEY,
            openai_api_base=config.OPENROUTER_BASE_URL,
        )

        self._embed_metadata = EmbeddingMetadata(
            provider="OpenRouter",
            model=model,
            dimensions=dimensions,
        )

    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Embed a list of chunks in batches, returning the same objects
        with embeddings and embed_metadata populated.

        Only child chunks are embedded.
        """
        
        children = [c for c in chunks if c.chunk_metadata.chunk_type == "child"]
        if not children:
            return chunks

        texts = [c.content for c in children]
        all_vectors: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors = await self._embedder.aembed_documents(batch)
            all_vectors.extend(vectors)

        for chunk, vector in zip(children, all_vectors):
            chunk.embeddings = vector
            chunk.embed_metadata = self._embed_metadata

        logger.info(
            f"Embedded {len(children)} child chunk(s) "
            f"({len(texts) // self._batch_size + 1} batch(es), "
            f"model={self._model})"
        )
        return chunks

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for similarity search."""
        return await self._embedder.aembed_query(query)
