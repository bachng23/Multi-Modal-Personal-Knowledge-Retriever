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
