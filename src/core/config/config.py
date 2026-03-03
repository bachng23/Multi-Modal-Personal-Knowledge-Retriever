import os
from pathlib import Path

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
PARENT_CHUNK_OVERLAP = 50
MIN_PARENT_SIZE = 2000
MAX_CHUNK_SIZE = 4000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- Token Counting ---
TIKTOKEN_ENCODING = "cl100k_base"

# --- Database Configuration ---
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
SQLITE_DB_PATH = DATA_DIR / "store.db"

# --- Qdrant (Docker local server) ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_PREFER_GRPC = True
QDRANT_COLLECTION_NAME = "child_chunks"
QDRANT_DISTANCE = "Cosine"
QDRANT_ON_DISK = True

# --- OpenRouter ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# --- Embedding ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "openai/text-embedding-3-small")
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1536"))

# --- LLM ---
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

# --- Indexing ---
CHUNKER_VERSION = "v1.0"
INDEX_CONCURRENCY = int(os.getenv("INDEX_CONCURRENCY", "10"))

# --- Obsidian Vault ---
VAULT_PATH = Path(os.getenv("VAULT_PATH", ""))
EXCLUDED_DIRS: set[str] = {"templates", "attachments", "archive", "img"}
