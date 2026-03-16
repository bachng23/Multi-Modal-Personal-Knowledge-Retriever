import re
import tiktoken

from loguru import logger

from src.core.models.section import Section
from src.infrastructure.filesystem.obsidian_parser import ParsedDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_BLOCK_ID_RE = re.compile(r"\s\^([A-Za-z0-9-]+)$")
_CODE_FENCE_RE = re.compile(r"^```")


class ObsidianPreprocessor:
    """Structural preprocessor for Obsidian markdown documents.

    Splits a parsed document into sections along heading boundaries while
    respecting code fences and collecting Obsidian block IDs.  
    
    The output is a flat list of Sections — raw text segments annotated with their
    heading ancestry — ready to be fed into chunking process.
    """

    def __init__(self, model_name: str):
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
            logger.info(f"Initialized tokenizer for model {model_name}")
        except KeyError:
            logger.warning(f"Model {model_name} not found. Falling back to cl100k_base encoding.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a given text using the initialized tokenizer."""
        return len(self.tokenizer.encode(text))

    def preprocess(self, parsed_doc: ParsedDocument) -> list[Section]:
        lines = parsed_doc.body.split("\n")

        in_code_block: bool = False
        heading_path: dict[int, str] = {}

        current_lines: list[str] = []
        current_heading_snapshot: dict[int, str] = {}
        current_level: int | None = None
        current_block_ids: list[str] = []

        sections: list[Section] = []

        for line in lines:
            if _CODE_FENCE_RE.match(line):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            if in_code_block:
                current_lines.append(line)
                continue

            heading_match = _HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                if current_lines:
                    section = self._build_section(
                        current_lines, current_heading_snapshot,
                        current_level, current_block_ids,
                    )
                    if section:
                        sections.append(section)
                    current_lines = []
                    current_block_ids = []

                heading_path[level] = title
                for k in [k for k in heading_path if k > level]:
                    del heading_path[k]

                current_heading_snapshot = dict(heading_path)
                current_level = level
                current_lines.append(line)

                block_id_match = _BLOCK_ID_RE.search(line)
                if block_id_match:
                    current_block_ids.append(block_id_match.group(1))

                continue

            block_id_match = _BLOCK_ID_RE.search(line)
            if block_id_match:
                current_block_ids.append(block_id_match.group(1))

            current_lines.append(line)

        if current_lines:
            section = self._build_section(
                current_lines, current_heading_snapshot,
                current_level, current_block_ids,
            )
            if section:
                sections.append(section)

        logger.info(
            f"Preprocessed document into {len(sections)} section(s)"
        )
        return sections

    def _build_section(
        self,
        lines: list[str],
        heading_path: dict[int, str],
        heading_level: int | None,
        block_ids: list[str],
    ) -> Section | None:
        content = "\n".join(lines).strip()
        if not content:
            return None

        sorted_path = [heading_path[k] for k in sorted(heading_path)]
        
        tokens = self.count_tokens(content)

        return Section(
            heading_path=sorted_path or None,
            heading_level=heading_level,
            content=content,
            block_ids=block_ids or None,
            tokens_count=tokens
        )
