import re
import tiktoken
from loguru import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import config
from src.core.models.chunk import Chunk, ChunkMetadata
from src.core.models.section import Section

_BLOCK_ID_RE = re.compile(r"\s\^([A-Za-z0-9-]+)")

_MD_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]


class ParentChildChunker:
    """Two-tier chunker: large parent chunks for LLM context,
    small child chunks for precise vector search.

    Uses RecursiveCharacterTextSplitter (token-based via tiktoken)
    to split parent content into children.

    At retrieval time: search on children, fetch parent for full context.
    """

    def __init__(
        self,
        max_parent_size: int = config.MAX_CHUNK_SIZE,
        min_parent_size: int = config.MIN_PARENT_SIZE,
        child_chunk_size: int = config.CHILD_CHUNK_SIZE,
        child_chunk_overlap: int = config.CHILD_CHUNK_OVERLAP,
        parent_chunk_overlap: int = config.PARENT_CHUNK_OVERLAP,
        encoding_name: str = config.TIKTOKEN_ENCODING,
    ):
        self._max_parent_size = max_parent_size
        self._min_parent_size = min_parent_size
        self._tokenizer = tiktoken.get_encoding(encoding_name)

        self._parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=max_parent_size,
            chunk_overlap=parent_chunk_overlap,
            separators=_MD_SEPARATORS,
            strip_whitespace=True,
        )

        self._child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=_MD_SEPARATORS,
            strip_whitespace=True,
        )

    def chunk(
        self, sections: list[Section], document_id: str
    ) -> list[Chunk]:
        parent_groups = self._merge_into_parents(sections)

        chunks: list[Chunk] = []
        chunk_index = 0

        for group in parent_groups:
            parent_content = "\n\n".join(s.content for s in group)

            # Collect all block_ids across the group
            all_block_ids = [
                bid
                for s in group if s.block_ids
                for bid in s.block_ids
            ]

            # Build a heading_path that represents the full group, not just the first section
            parent_heading = self._resolve_group_heading(group)

            # ── Create parent chunk ──
            parent_id = Chunk.build_id(document_id, chunk_index)
            parent_meta = ChunkMetadata(
                heading_path=parent_heading,
                block_id=all_block_ids[0] if all_block_ids else None,
                chunk_type="parent",
            )
            chunks.append(Chunk(
                id=parent_id,
                document_id=document_id,
                chunk_index=chunk_index,
                chunk_metadata=parent_meta,
                content=parent_content,
            ))
            chunk_index += 1

            # ── Split parent into children ──
            child_fragments = self._child_splitter.split_text(parent_content)

            for fragment in child_fragments:
                child_heading = self._match_child_heading(fragment, group)
                child_block_id = self._scan_block_id(fragment)

                child_meta = ChunkMetadata(
                    heading_path=child_heading,
                    block_id=child_block_id,
                    chunk_type="child",
                    parent_chunk_id=parent_id,
                )
                chunks.append(Chunk(
                    id=Chunk.build_id(document_id, chunk_index),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    chunk_metadata=child_meta,
                    content=fragment,
                ))
                chunk_index += 1

        logger.info(
            f"Chunked document {document_id} into {len(chunks)} chunk(s) "
            f"({sum(1 for c in chunks if c.chunk_metadata.chunk_type == 'parent')} parents, "
            f"{sum(1 for c in chunks if c.chunk_metadata.chunk_type == 'child')} children)"
        )
        return chunks

    def _merge_into_parents(
        self, sections: list[Section]
    ) -> list[list[Section]]:
        """Greedily merge adjacent sections into parent-sized groups.

        Oversized sections (> max_parent_size) are split via _parent_splitter
        into synthetic sub-sections before grouping.
        """
        normalized = self._split_oversized_sections(sections)

        groups: list[list[Section]] = []
        current_group: list[Section] = []
        current_tokens = 0

        for section in normalized:
            section_tokens = section.tokens_count or 0

            if section_tokens >= self._min_parent_size:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_tokens = 0
                groups.append([section])
                continue

            if current_tokens + section_tokens > self._max_parent_size and current_group:
                groups.append(current_group)
                current_group = []
                current_tokens = 0

            current_group.append(section)
            current_tokens += section_tokens

            if current_tokens >= self._min_parent_size:
                groups.append(current_group)
                current_group = []
                current_tokens = 0

        if current_group:
            groups.append(current_group)

        return groups

    def _split_oversized_sections(
        self, sections: list[Section]
    ) -> list[Section]:
        """Split any section exceeding _max_parent_size into smaller
        sub-sections using _parent_splitter, preserving metadata."""
        result: list[Section] = []

        for section in sections:
            tokens = section.tokens_count or 0
            if tokens <= self._max_parent_size:
                result.append(section)
                continue

            fragments = self._parent_splitter.split_text(section.content)
            for fragment in fragments:
                result.append(Section(
                    heading_path=section.heading_path,
                    heading_level=section.heading_level,
                    content=fragment,
                    block_ids=[bid for bid in (section.block_ids or [])
                               if f"^{bid}" in fragment] or None,
                    tokens_count=len(self._tokenizer.encode(fragment)),
                ))

        return result

    @staticmethod
    def _resolve_group_heading(
        group: list[Section],
    ) -> list[str] | None:
        """Derive the heading_path for a parent chunk from its sections.

        - Single section → use its heading_path directly.
        - Multiple sections → find the longest common prefix of all
          heading_paths, so the parent represents the shared context.
          Falls back to the first section's path if no common prefix.
        """
        paths = [s.heading_path for s in group if s.heading_path]
        if not paths:
            return None
        if len(paths) == 1:
            return paths[0]

        # Longest common prefix across all heading paths
        prefix: list[str] = []
        for parts in zip(*paths):
            if len(set(parts)) == 1:
                prefix.append(parts[0])
            else:
                break

        return prefix if prefix else paths[0]

    @staticmethod
    def _match_child_heading(
        fragment: str, group: list[Section]
    ) -> list[str] | None:
        """Find the best matching section for a child fragment
        and return that section's heading_path.

        Uses substring containment: the section whose content
        contains the largest overlap with the fragment wins.
        Falls back to the first section if no match.
        """
        best_section = group[0]
        best_overlap = 0

        for section in group:
            # Check how much of the fragment appears in this section
            overlap = len(fragment) if fragment in section.content else 0
            if overlap > best_overlap:
                best_overlap = overlap
                best_section = section

        return best_section.heading_path

    @staticmethod
    def _scan_block_id(fragment: str) -> str | None:
        """Scan a child fragment for Obsidian block IDs (^some-id).
        Returns the first block ID found, or None."""
        match = _BLOCK_ID_RE.search(fragment)
        return match.group(1) if match else None