import hashlib
from pathlib import Path
from typing import Any

def stable_hashing(data: Any) -> str:
    hash = hashlib.blake2b(digest_size=16)
    hash.update(data.encode("utf-8"))
    return hash.hexdigest()

def normalize_rel_path(rel_path: str | Path) -> Path:
    """Normalize rel path from different OS"""
    
    return Path(rel_path).as_posix()
