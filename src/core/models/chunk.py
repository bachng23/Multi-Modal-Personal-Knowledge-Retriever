from pydantic import BaseModel, Field
from typing import Dict, Any
from src.core.utils import stable_hashing

class EmbeddingMetadata(BaseModel):
    """Metadata for a chunk's embedding."""

    provider: str = Field(..., description="The provider of the embedding (e.g., 'OpenAI', 'HuggingFace')")
    model: str = Field(..., description="The model used to generate the embedding (e.g., 'text-embedding-3-small')")
    dimensions: int = Field(..., description="The dimensionality of the embedding vector")

class ChunkMetadata(BaseModel):
    """Metadata for a chunk."""

    heading_path: list[str] | None = Field(default=None, description="The heading path for the chunk, if any")
    block_id: str | None = Field(default=None, description="The block ID for the chunk, if any")
    parent_chunk_id: str | None = Field(default=None, description="The ID of the parent chunk, if this chunk is a child chunk")      
    chunk_type: str = Field(default="child", description="The type of chunk 'parent' or 'child'")               

    @property
    def heading_string(self) -> str:
        """Returns the heading path as a string, joined by ' > ' for prompt and display purposes."""

        if not self.heading_path:
            return ""
        return " > ".join(self.heading_path)

class Chunk(BaseModel):
    """Represents a chunk of content extracted from a document."""

    id: str = Field(..., description="Unique ID for the chunk")
    document_id: str = Field(..., description="ID of the source document")
    chunk_index: int = Field(..., description="Index of the chunk within the document")
    chunk_metadata: ChunkMetadata = Field(..., description="Metadata for the chunk")
    content: str = Field(..., description="The content of the chunk")
    embeddings: list[float] | None = Field(default=None, description="The embeddings for the chunk, if any")
    embed_metadata: EmbeddingMetadata | None = Field(default=None, description="Metadata for the chunk")
    
    @staticmethod
    def build_id(document_id: str, chunk_index: int) -> str:
        """Stable ID from document identity + chunk index."""
        
        return stable_hashing(f"{document_id}:{chunk_index}")


class ChunkSearchResult(BaseModel):
    """Result from a vector similarity search."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    parent_chunk_id: str | None = None
    heading_path: list[str] | None = None
    block_id: str | None = None