from pydantic import BaseModel, Field

class Citation(BaseModel):
    """Represents a citation for prompt and UI."""

    source_path: str = Field(..., description="The source path of the cited document")
    heading_path: list[str] | None = Field(default=None, description="The heading path for the citation, if any")
    block_id: str | None = Field(default=None, description="The block ID for the citation, if any")
    quote: str | None = Field(default=None, description="The quoted text for the citation")

    @property
    def obsidian_link(self) -> str:
        """Returns the Obsidian link format for the citation."""
        
        clean_path = self.source_path.replace(".md", "")
        link = clean_path

        if self.heading_path and len(self.heading_path) > 0:
            link += f"#{self.heading_path[-1]}"

        if self.block_id:
            clean_block = self.block_id.lstrip("^")
            link += f"^{clean_block}"
        
        return f"[[{link}]]"
    
