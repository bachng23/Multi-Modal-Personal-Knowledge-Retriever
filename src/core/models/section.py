from pydantic import BaseModel, Field

class Section(BaseModel):
    """Represents a section of a document after preprocessing, but before chunking."""

    heading_path: list[str] | None = Field(default_factory=list, description="The heading path for the section, if any")
    heading_level: int | None = Field(default=None, description="The heading level for the section, if any")
    content: str = Field(..., description="The content of section")
    block_ids: list[str] | None = Field(default_factory=list, description="The blocks of section, if any")
    tokens_count: int | None = Field(default=None, description="The token count for the section content, if available")