from pydantic import BaseModel, Field
from typing import List

class IndexState(BaseModel):
    """Represents the state of the document index for change detection and incremental updates."""
    
    document_id: str = Field(..., description="The ID of the document")
    source_type: str = Field(..., description="The type of source (e.g., 'Obsidian', 'Web', 'Database')")
    source_path: str = Field(..., description="The relative path of the document from the vault root or URL for web sources")
    mtime: float = Field(..., description="The last modified time of the document")
    content_hash: str = Field(..., description="The hash of the document content for change detection")
    chunk_ids: List[str] = Field(default_factory=list, description="List of chunk IDs associated with the document")
    chunker_version: str = Field(..., description="The version of the chunking algorithm used for this document")
    embed_provider: str = Field(..., description="The provider used for generating embeddings")
    embed_model: str = Field(..., description="The model used for generating embeddings")
