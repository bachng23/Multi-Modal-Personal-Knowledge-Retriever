import json
import sqlite3
from pathlib import Path

from loguru import logger

from src.core.models.chunk import Chunk, ChunkMetadata
from src.core.models.indexState import IndexState


class ParentStoreManager:
    """SQLite store cho parent chunks + index state.

    Tables:
    - parent_chunks: full-text parent chunks (LLM context)
    - index_state: document indexing metadata (change detection)
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self) -> None:
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    id              TEXT PRIMARY KEY,
                    document_id     TEXT NOT NULL,
                    chunk_index     INTEGER NOT NULL,
                    content         TEXT NOT NULL,
                    metadata_json   TEXT NOT NULL,
                    created_at      TEXT DEFAULT (datetime('now')),
                    UNIQUE(document_id, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS idx_parent_doc_id
                    ON parent_chunks(document_id);

                CREATE TABLE IF NOT EXISTS index_state (
                    document_id     TEXT PRIMARY KEY,
                    source_type     TEXT NOT NULL,
                    source_path     TEXT NOT NULL,
                    mtime           REAL NOT NULL,
                    content_hash    TEXT NOT NULL,
                    chunk_ids_json  TEXT NOT NULL,
                    chunker_version TEXT NOT NULL,
                    embed_provider  TEXT NOT NULL,
                    embed_model     TEXT NOT NULL,
                    updated_at      TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_state_source
                    ON index_state(source_type, source_path);
            """)


    def save_parents(self, chunks: list[Chunk]) -> None:
        rows = [
            (
                chunk.id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.content,
                chunk.chunk_metadata.model_dump_json(),
            )
            for chunk in chunks
            if chunk.chunk_metadata.chunk_type == "parent"
        ]
        with self._conn:
            self._conn.executemany(
                """INSERT OR REPLACE INTO parent_chunks
                   (id, document_id, chunk_index, content, metadata_json)
                   VALUES (?, ?, ?, ?, ?)""",
                rows,
            )
        logger.debug(f"Saved {len(rows)} parent chunk(s)")

    def get_parent(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute(
            "SELECT * FROM parent_chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def get_parents_batch(self, chunk_ids: list[str]) -> list[Chunk]:
        """
        Lấy nhiều parent chunks trong 1 query. Dùng cho retrieval pipeline.

        Search trả về N child chunks → N parent_chunk_ids → gọi hàm này 1 lần.
        Tránh N+1 query problem khi gọi get_parent() trong vòng lặp.
        """
        if not chunk_ids:
            return []
        placeholders = ", ".join("?" for _ in chunk_ids)
        rows = self._conn.execute(
            f"SELECT * FROM parent_chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_parents_by_document(self, document_id: str) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT * FROM parent_chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def delete_parents_by_document(self, document_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM parent_chunks WHERE document_id = ?",
                (document_id,),
            )


    def save_index_state(self, state: IndexState) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO index_state
                   (document_id, source_type, source_path, mtime,
                    content_hash, chunk_ids_json, chunker_version,
                    embed_provider, embed_model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    state.document_id,
                    state.source_type,
                    state.source_path,
                    state.mtime,
                    state.content_hash,
                    json.dumps(state.chunk_ids),
                    state.chunker_version,
                    state.embed_provider,
                    state.embed_model,
                ),
            )

    def get_index_state(self, document_id: str) -> IndexState | None:
        row = self._conn.execute(
            "SELECT * FROM index_state WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_index_state(row)
    
    def get_index_states_batch(self, document_ids: list[str]) -> list[IndexState]:
        if not document_ids:
            return []
        placeholders = ", ".join("?" for _ in document_ids)
        rows = self._conn.execute(
            f"SELECT * FROM index_state WHERE document_id IN ({placeholders})",
            document_ids,
        ).fetchall()
        return [self._row_to_index_state(r) for r in rows]
        
    def get_index_state_by_path(self, source_path: str) -> IndexState | None:
        """Lookup index state by source_path using the existing SQL index.
        Much faster than loading all states when the vault is large."""
        row = self._conn.execute(
            "SELECT * FROM index_state WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_index_state(row)

    def get_all_index_states(self) -> list[IndexState]:
        rows = self._conn.execute("SELECT * FROM index_state").fetchall()
        return [self._row_to_index_state(r) for r in rows]

    def delete_index_state(self, document_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM index_state WHERE document_id = ?",
                (document_id,),
            )

    def needs_reindex(
        self,
        document_id: str,
        content_hash: str,
        chunker_version: str,
        embed_model: str,
    ) -> bool:
        """Check if the document needs re-indexing based on its index state."""

        state = self.get_index_state(document_id)
        if state is None:
            return True
        return (
            state.content_hash != content_hash
            or state.chunker_version != chunker_version
            or state.embed_model != embed_model
        )

    def delete_document_data(self, document_id: str) -> None:
        self.delete_parents_by_document(document_id)
        self.delete_index_state(document_id)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            chunk_metadata=ChunkMetadata.model_validate_json(row["metadata_json"]),
        )

    @staticmethod
    def _row_to_index_state(row: sqlite3.Row) -> IndexState:
        return IndexState(
            document_id=row["document_id"],
            source_type=row["source_type"],
            source_path=row["source_path"],
            mtime=row["mtime"],
            content_hash=row["content_hash"],
            chunk_ids=json.loads(row["chunk_ids_json"]),
            chunker_version=row["chunker_version"],
            embed_provider=row["embed_provider"],
            embed_model=row["embed_model"],
        )