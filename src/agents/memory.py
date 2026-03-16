import aiosqlite
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.core.config import config


async def create_memory_checkpointer() -> AsyncSqliteSaver:
    """Create an async SQLite checkpointer for persistent conversation memory."""
    db_path = config.DATA_DIR / "memory.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path))
    return AsyncSqliteSaver(conn)


def create_inmemory_checkpointer() -> MemorySaver:
    """Create an in-memory checkpointer (for langgraph dev / testing)."""
    return MemorySaver()
