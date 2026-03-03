from pathlib import Path
from loguru import logger

from src.core.config import config
from src.core.models.document import Document
from src.infrastructure.filesystem.obsidian_parser import parse_obsidian_document


class VaultScanner:
    """
    Scans an Obsidian vault directory for markdown files and returns
    a list of parsed Document objects ready for the indexing pipeline.

    Skips hidden directories (e.g. .obsidian, .trash) and empty files.
    """

    def __init__(self, excluded_dirs: set[str] | None = None) -> None:
        self._excluded_dirs = excluded_dirs or config.EXCLUDED_DIRS

    def load_vault(self, vault_dir: str | Path | None = None) -> list[Document]:
        resolved = Path(vault_dir or config.VAULT_PATH).expanduser().resolve()

        if not resolved.exists():
            logger.error(f"Vault path does not exist: {resolved}")
            return []
        if not resolved.is_dir():
            logger.error(f"Vault path is not a directory: {resolved}")
            return []

        logger.info(f"Scanning vault: {resolved}")

        documents: list[Document] = []

        for md_file in sorted(resolved.rglob("*.md")):
            if self._is_excluded(md_file):
                continue

            try:
                doc = self._load_single_file(md_file, resolved)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"Skipped {md_file.name}: {e}")

        logger.info(
            f"Vault scan complete: {len(documents)} document(s) loaded "
            f"from {resolved}"
        )
        return documents

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_excluded(self, file_path: Path) -> bool:
        """Return True if any part of the path matches an excluded dir
        or starts with '.' (hidden directories)."""
        for part in file_path.parts:
            if part.startswith(".") or part in self._excluded_dirs:
                return True
        return False

    def _load_single_file(self, file_path: Path, vault_dir: Path) -> Document | None:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            logger.debug(f"Skipping empty file: {file_path.name}")
            return None

        relative_path = file_path.relative_to(vault_dir).as_posix()
        parsed = parse_obsidian_document(raw)

        title = (
            parsed.frontmatter.get("title")
            or file_path.stem
        )

        raw_aliases = parsed.frontmatter.get("aliases") or []
        if isinstance(raw_aliases, str):
            raw_aliases = [raw_aliases]
        aliases = [str(a) for a in raw_aliases]

        return Document(
            id=Document.build_id("Obsidian", relative_path),
            source_type="Obsidian",
            source_path=relative_path,
            title=title,
            aliases=aliases,
            tags=parsed.tags,
            mtime=file_path.stat().st_mtime,
            content=raw,
            links=parsed.links,
        )
