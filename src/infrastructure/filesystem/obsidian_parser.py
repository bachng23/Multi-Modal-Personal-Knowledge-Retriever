import re
import yaml
from typing import Any
from loguru import logger

from pydantic import BaseModel, Field

from src.core.models.document import Link


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")

_WIKILINK_RE = re.compile(r"(!?)\[\[([^\[\]]+?)\]\]")

_INLINE_TAG_RE = re.compile(r"(?<![/\w])#([a-zA-Z_][\w/-]*)")


class ParsedDocument(BaseModel):
    """Intermediate result of parsing a raw Obsidian markdown note."""

    frontmatter: dict[str, Any] = Field(default_factory=dict, description="Parsed YAML frontmatter")
    body: str = Field(..., description="Content with frontmatter stripped")
    tags: list[str] = Field(default_factory=list, description="Merged frontmatter + inline tags")
    links: list[Link] = Field(default_factory=list, description="List of wikilinks and embeds")


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """
    Extract YAML frontmatter from the top of a note.

    Returns (metadata_dict, body_without_frontmatter).
    If no frontmatter is found, returns ({}, content).
    """
    
    match = _FRONTMATTER_RE.match(content)
    if not match:
        logger.debug("No frontmatter found.")
        return {}, content

    raw_yaml = match.group(1)
    body = content[match.end() :]

    parsed = yaml.safe_load(raw_yaml)
    if not isinstance(parsed, dict):
        logger.debug("Frontmatter is not a valid YAML dictionary.")
        return {}, content
    logger.debug(f"Parsed frontmatter: {parsed}")
    return parsed, body


def _strip_code_blocks(text: str) -> str:
    """Remove fenced and inline code so that extraction regexes skip them."""
    text = _FENCED_CODE_RE.sub("", text)
    text = _INLINE_CODE_RE.sub("", text)
    logger.debug("Stripped code blocks and inline code for cleaner parsing.")
    return text


def parse_wikilinks(text: str) -> list[Link]:
    """Extract all wikilinks and embeds."""
    links: list[Link] = []

    for match in _WIKILINK_RE.finditer(text):
        is_embed = match.group(1) == "!"
        inner = match.group(2)

        # Separate alias: [[target|alias]] -> keep only target part
        target_part = inner.split("|", maxsplit=1)[0]

        target: str = target_part
        heading: str | None = None
        block_id: str | None = None

        if "#" in target_part:
            target, fragment = target_part.split("#", maxsplit=1)
            if fragment.startswith("^"):
                block_id = fragment[1:]
            else:
                heading = fragment

        links.append(
            Link(
                target=target.strip(),
                target_heading=heading,
                target_block_id=block_id,
                is_embed=is_embed,
            )
        )

    return links


def parse_inline_tags(text: str) -> list[str]:
    """Extract inline tags.

    Returns a deduplicated list of tag strings without the leading `#`.
    """
    seen: set[str] = set()
    tags: list[str] = []
    for match in _INLINE_TAG_RE.finditer(text):
        tag = match.group(1)
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def parse_obsidian_document(content: str) -> ParsedDocument:
    """Full parse of a raw Obsidian markdown note."""
    fm_dict, body = parse_frontmatter(content)

    clean = _strip_code_blocks(body)

    links = parse_wikilinks(clean)

    # Tags from frontmatter (handles both list and single-string forms)
    fm_tags_raw = fm_dict.get("tags", [])
    if isinstance(fm_tags_raw, str):
        fm_tags_raw = [fm_tags_raw]
    fm_tags: list[str] = [t.lstrip("#") for t in fm_tags_raw]

    inline_tags = parse_inline_tags(clean)

    # Merge and deduplicate while preserving order (frontmatter first)
    seen: set[str] = set()
    merged_tags: list[str] = []
    for tag in fm_tags + inline_tags:
        if tag not in seen:
            seen.add(tag)
            merged_tags.append(tag)

    return ParsedDocument(
        frontmatter=fm_dict,
        body=body,
        tags=merged_tags,
        links=links,
    )
