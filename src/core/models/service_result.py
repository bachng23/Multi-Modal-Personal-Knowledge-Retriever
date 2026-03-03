from pydantic import BaseModel, Field 
from src.core.models.chunk import Chunk
from src.core.models.citation import Citation


class ChunkSearchResult(BaseModel):
    """Result from a vector similarity search."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    parent_chunk_id: str | None = None
    heading_path: list[str] | None = None
    block_id: str | None = None

class RetrievalResult(BaseModel):
    """Structured result from a retrieval query."""

    query: str = Field(..., description="The original query string")
    parent_chunks: list[Chunk] = Field(default_factory=list, description="List of parent chunks with similarity scores")
    child_chunks: list[ChunkSearchResult] = Field(default_factory=list, description="List of child chunks with similarity scores")
    citation: list[Citation] = Field(default_factory=list, description="List of citations to source documents, if any")