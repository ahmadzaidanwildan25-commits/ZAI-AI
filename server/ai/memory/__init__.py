from __future__ import annotations

from .memory_store import (
    MemoryRecord,
    MemorySession,
    MemoryStore,
    create_memory_store,
    self_test,
)

# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------
StoredMemory = MemoryRecord
get_memory_store = create_memory_store

# Alias / Helper jika modul lain memanggil reset_memory_store
def reset_memory_store(*args, **kwargs):
    store = create_memory_store()
    if hasattr(store, "reset"):
        return store.reset(*args, **kwargs)

__all__ = [
    "MemoryRecord",
    "MemorySession",
    "MemoryStore",
    "StoredMemory",
    "create_memory_store",
    "get_memory_store",
    "reset_memory_store",
    "self_test",
]
