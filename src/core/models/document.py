from pydantic import BaseModel, Field
from typing import Dict, Any
from src.core.utils import stable_hashing

class Link(BaseModel):
    """Represents a wikilink or embed link."""

    target: str = Field(..., description="The name of the target document")
    target_heading: str | None = Field(default=None, description="The heading in the target document, if any")
    target_block_id: str | None = Field(default=None, description="The block ID in the target document, if any")
    is_embed: bool = Field(False, description="Whether this link is an embed link")

class Document(BaseModel):
    """Represents a document with metadata and content."""
    
    id: str = Field(..., description="Unique ID for a file")
    source_type: str = Field("Obsidian", description="The type of source (e.g., 'file', 'web', 'database')")
    source_path: str = Field(..., description="Rel path from Vault")     
    title: str = Field(..., description="Title of the document")
    aliases: list[str] = Field(default_factory=list, description="List of the aliases of the document")
    tags: list[str] = Field(default_factory=list, description="List of tags associated with the document")
    mtime: float = Field(..., description="Last modified time of the document")
    content: str = Field(..., description="The content of the document")
    links: list[Link] = Field(default_factory=list, description="List of links associated with the document")

    @staticmethod
    def build_id(source_type: str, source_path: str) -> str:
        """Builds a unique ID for the document based on its source type and path."""
        return stable_hashing(f"{source_type}:{source_path}")
    







