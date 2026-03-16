from pydantic import BaseModel, Field 
from src.core.models.chunk import Chunk, ChunkSearchResult
from src.core.models.citation import Citation


class RetrievalResult(BaseModel):
    """Structured result from a retrieval query."""

    query: str = Field(..., description="The original query string")
    parent_chunks: list[Chunk] = Field(default_factory=list, description="List of parent chunks with similarity scores")
    child_results: list[ChunkSearchResult] = Field(default_factory=list, description="List of child chunks with similarity scores")
    citations: list[Citation] = Field(default_factory=list, description="List of citations to source documents, if any")