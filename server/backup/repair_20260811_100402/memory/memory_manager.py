"""
ZAI Memory Manager
Super ZAI - Memory Management Layer

Menghubungkan AIBrain dengan persistent memory database.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from memory.memory_database import (
    save_memory,
    get_memory,
    search_memories,
    get_all_memories,
    get_important_memories,
    delete_memory,
    clear_memories,
    count_memories,
    database_status,
)


class MemoryManager:
    """
    Lapisan manajemen memory ZAI.

    MemoryManager bertugas menjadi perantara antara
    AI Brain dan database memory.
    """

    VERSION = "0.1.0"

    def __init__(
        self,
        default_importance: int = 5,
    ) -> None:

        self.default_importance = max(
            1,
            min(
                int(default_importance),
                10,
            ),
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        key: str,
        value: str,
        category: str = "general",
        importance: Optional[int] = None,
    ) -> bool:
        """
        Menyimpan memory ZAI.
        """

        key = str(key or "").strip()
        value = str(value or "").strip()

        if not key or not value:
            return False

        if importance is None:
            importance = self.default_importance

        try:

            save_memory(
                key=key,
                value=value,
                category=category,
                importance=importance,
            )

            return True

        except Exception:
            return False

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        key: str,
    ) -> Optional[str]:
        """
        Mengambil satu memory.
        """

        try:
            return get_memory(key)

        except Exception:
            return None

    # ========================================================
    # EXISTS
    # ========================================================

    def exists(
        self,
        key: str,
    ) -> bool:
        """
        Mengecek apakah memory tersedia.
        """

        return self.get(key) is not None

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Mencari memory berdasarkan query.
        """

        try:
            return search_memories(
                query=query,
                limit=limit,
            )

        except Exception:
            return []

    # ========================================================
    # ALL
    # ========================================================

    def all(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Mengambil seluruh memory.
        """

        try:
            return get_all_memories(
                limit=limit,
            )

        except Exception:
            return []

    # ========================================================
    # IMPORTANT
    # ========================================================

    def important(
        self,
        limit: int = 5,
        minimum_importance: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Mengambil memory penting.
        """

        try:
            return get_important_memories(
                limit=limit,
                minimum_importance=minimum_importance,
            )

        except Exception:
            return []

    # ========================================================
    # DELETE
    # ========================================================

    def delete(
        self,
        key: str,
    ) -> bool:
        """
        Menghapus memory berdasarkan key.
        """

        try:
            return delete_memory(key)

        except Exception:
            return False

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self) -> bool:
        """
        Menghapus seluruh memory.

        Fungsi ini disediakan untuk maintenance/testing.
        """

        try:
            clear_memories()
            return True

        except Exception:
            return False

    # ========================================================
    # COUNT
    # ========================================================

    def count(self) -> int:
        """
        Menghitung memory.
        """

        try:
            return count_memories()

        except Exception:
            return 0

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> Dict[str, Any]:
        """
        Mengambil status memory system.
        """

        try:

            database = database_status()

            return {
                "engine": "MemoryManager",
                "version": self.VERSION,
                "database": database,
                "memory_count": self.count(),
                "status": "READY",
            }

        except Exception as exc:

            return {
                "engine": "MemoryManager",
                "version": self.VERSION,
                "database": {},
                "memory_count": 0,
                "status": "ERROR",
                "error": str(exc),
            }

    # ========================================================
    # STATS
    # ========================================================

    def stats(self) -> Dict[str, Any]:
        """
        Statistik MemoryManager.
        """

        return {
            "engine": "MemoryManager",
            "version": self.VERSION,
            "persistent_memory": True,
            "database": "SQLite",
            "memory_count": self.count(),
            "status": "READY",
        }


# ============================================================
# SINGLETON
# ============================================================

_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """
    Mengambil singleton MemoryManager ZAI.
    """

    global _memory_manager

    if _memory_manager is None:
        _memory_manager = MemoryManager()

    return _memory_manager
