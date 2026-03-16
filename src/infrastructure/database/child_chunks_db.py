import uuid
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient, models

from src.core.models.chunk import Chunk, ChunkMetadata, ChunkSearchResult


def chunk_id_to_point_id(chunk_id: str) -> str:
    """blake2b 32-hex → UUID string cho Qdrant."""
    return str(uuid.UUID(chunk_id))


def point_id_to_chunk_id(point_id: str) -> str:
    """UUID string → blake2b 32-hex."""
    return uuid.UUID(point_id).hex


class ChildChunksDB:
    """
    Qdrant vector store for child chunks.

    Connecting to Qdrant Docker container through REST/gRPC.
    Fallback: return path instead of url for using embedded mode (test).
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
        collection_name: str = "child_chunks",
        vector_size: int = 1536,
        *,
        path: Path | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._vector_size = vector_size

        if path is not None:
            self._client = QdrantClient(path=str(path))
        else:
            self._client = QdrantClient(
                url=url,
                api_key=api_key,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
            )

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection_name):
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=self._vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
        )

        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="chunk_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        logger.info(f"Created Qdrant collection '{self._collection_name}'")

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> None:
        points: list[models.PointStruct] = []

        for chunk in chunks:
            if chunk.embeddings is None:
                raise ValueError(
                    f"Chunk {chunk.id} has no embeddings. Embed before adding."
                )

            points.append(
                models.PointStruct(
                    id=chunk_id_to_point_id(chunk.id),
                    vector=chunk.embeddings,
                    payload={
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "chunk_type": chunk.chunk_metadata.chunk_type,
                        "parent_chunk_id": chunk.chunk_metadata.parent_chunk_id,
                        "heading_path": chunk.chunk_metadata.heading_path,
                        "block_id": chunk.chunk_metadata.block_id,
                        "embed_provider": (
                            chunk.embed_metadata.provider
                            if chunk.embed_metadata
                            else None
                        ),
                        "embed_model": (
                            chunk.embed_metadata.model
                            if chunk.embed_metadata
                            else None
                        ),
                    },
                )
            )

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self._collection_name,
                points=batch,
            )

        logger.debug(f"Upserted {len(points)} child chunk(s) into Qdrant")

    def delete_by_document(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        query_filter: models.Filter | None = None,
        score_threshold: float | None = None,
    ) -> list[ChunkSearchResult]:
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        return [
            ChunkSearchResult(
                chunk_id=point.payload["chunk_id"],
                document_id=point.payload["document_id"],
                content=point.payload["content"],
                score=point.score,
                parent_chunk_id=point.payload.get("parent_chunk_id"),
                heading_path=point.payload.get("heading_path"),
                block_id=point.payload.get("block_id"),
            )
            for point in results.points
        ]

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        points = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[chunk_id_to_point_id(chunk_id)],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None

        p = points[0]
        return Chunk(
            id=p.payload["chunk_id"],
            document_id=p.payload["document_id"],
            chunk_index=p.payload["chunk_index"],
            content=p.payload["content"],
            chunk_metadata=ChunkMetadata(
                chunk_type=p.payload.get("chunk_type", "child"),
                parent_chunk_id=p.payload.get("parent_chunk_id"),
                heading_path=p.payload.get("heading_path"),
                block_id=p.payload.get("block_id"),
            ),
        )

    def count(self) -> int:
        return self._client.count(
            collection_name=self._collection_name
        ).count

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._ensure_collection()
        logger.warning(f"Reset Qdrant collection '{self._collection_name}'")

    def collection_info(self) -> dict:
        info = self._client.get_collection(self._collection_name)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.value,
        }