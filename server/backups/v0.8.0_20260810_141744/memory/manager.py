from __future__ import annotations

import re
from typing import Optional, Any

from .database import (
    count_memories,
    delete_memory,
    get_all_memories,
    get_important_memories,
    get_memory,
    save_memory,
    search_memories,
)


class MemoryManager:
    """
    Super ZAI Memory Intelligence Engine.

    Responsibilities:
    - Detect memory commands.
    - Automatically classify memory categories.
    - Automatically calculate importance.
    - Prevent duplicate memories.
    - Save/update long-term memories.
    - Search memories using relevance scoring.
    - Retrieve important memories.
    - Build clean memory context.
    - Handle memory commands directly.
    """

    MEMORY_LIMIT = 5
    SEARCH_LIMIT = 10
    IMPORTANT_LIMIT = 10
    CONTEXT_LIMIT = 5

    MAX_CONTENT_LENGTH = 1000
    MAX_KEY_LENGTH = 300

    # ============================================================
    # INIT
    # ============================================================

    def __init__(self) -> None:
        self.memory_limit = self.MEMORY_LIMIT

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalisasi teks untuk perbandingan memory.
        """
        if not text:
            return ""

        text = str(text)

        text = text.strip().lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text

    # ============================================================
    # CLEAN TEXT
    # ============================================================

    @classmethod
    def clean_content(cls, text: str) -> str:
        """
        Membersihkan memory sebelum disimpan.
        """
        if not text:
            return ""

        text = str(text).strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        if len(text) > cls.MAX_CONTENT_LENGTH:
            text = text[:cls.MAX_CONTENT_LENGTH].rstrip()

        return text

    # ============================================================
    # CATEGORY DETECTION
    # ============================================================

    @classmethod
    def detect_category(
        cls,
        content: str,
    ) -> str:
        """
        Menentukan kategori memory secara otomatis.
        """

        text = cls.normalize(content)

        # --------------------------------------------------------
        # PROJECT
        # --------------------------------------------------------

        project_keywords = [
            "proyek",
            "project",
            "sedang membangun",
            "sedang membuat",
            "membangun zai",
            "membuat zai",
            "super zai",
            "aplikasi saya",
            "aplikasi kami",
            "bisnis saya",
            "usaha saya",
        ]

        if any(
            keyword in text
            for keyword in project_keywords
        ):
            return "project"

        # --------------------------------------------------------
        # PREFERENCE
        # --------------------------------------------------------

        preference_keywords = [
            "saya suka",
            "saya tidak suka",
            "saya lebih suka",
            "saya ingin",
            "saya mau",
            "favorit saya",
            "kesukaan saya",
            "preferensi saya",
            "lebih nyaman",
            "jangan gunakan",
        ]

        if any(
            keyword in text
            for keyword in preference_keywords
        ):
            return "preference"

        # --------------------------------------------------------
        # GOAL
        # --------------------------------------------------------

        goal_keywords = [
            "target saya",
            "tujuan saya",
            "impian saya",
            "cita-cita saya",
            "saya ingin mencapai",
            "saya ingin menjadi",
            "saya ingin mendapatkan",
            "goal saya",
        ]

        if any(
            keyword in text
            for keyword in goal_keywords
        ):
            return "goal"

        # --------------------------------------------------------
        # WORK
        # --------------------------------------------------------

        work_keywords = [
            "pekerjaan saya",
            "kerjaan saya",
            "pekerjaan",
            "kantor saya",
            "bisnis saya",
            "usaha saya",
            "pekerjaan utama",
        ]

        if any(
            keyword in text
            for keyword in work_keywords
        ):
            return "work"

        # --------------------------------------------------------
        # PERSONAL
        # --------------------------------------------------------

        personal_keywords = [
            "nama saya",
            "umur saya",
            "saya tinggal",
            "alamat saya",
            "keluarga saya",
            "teman saya",
            "pasangan saya",
            "pacar saya",
        ]

        if any(
            keyword in text
            for keyword in personal_keywords
        ):
            return "personal"

        # --------------------------------------------------------
        # SYSTEM
        # --------------------------------------------------------

        system_keywords = [
            "system",
            "sistem",
            "zai",
            "server",
            "database",
            "memory",
            "model",
            "ollama",
            "python",
            "fastapi",
        ]

        if any(
            keyword in text
            for keyword in system_keywords
        ):
            return "system"

        return "general"

    # ============================================================
    # IMPORTANCE
    # ============================================================

    @classmethod
    def calculate_importance(
        cls,
        content: str,
        category: Optional[str] = None,
    ) -> int:
        """
        Menghitung tingkat kepentingan memory 1-10.
        """

        text = cls.normalize(content)

        category = (
            category
            or cls.detect_category(text)
        )

        score = 5

        # --------------------------------------------------------
        # CATEGORY BOOST
        # --------------------------------------------------------

        category_bonus = {
            "goal": 3,
            "project": 3,
            "personal": 2,
            "preference": 2,
            "work": 2,
            "system": 1,
            "general": 0,
        }

        score += category_bonus.get(
            category,
            0,
        )

        # --------------------------------------------------------
        # IMPORTANT EXPRESSIONS
        # --------------------------------------------------------

        strong_keywords = [
            "penting",
            "wajib",
            "harus",
            "ingat selalu",
            "jangan lupa",
            "selamanya",
            "utama",
            "prioritas",
            "target utama",
        ]

        for keyword in strong_keywords:
            if keyword in text:
                score += 1

        # --------------------------------------------------------
        # LONG-TERM EXPRESSIONS
        # --------------------------------------------------------

        long_term_keywords = [
            "kedepannya",
            "ke depannya",
            "jangka panjang",
            "seterusnya",
            "mulai sekarang",
            "mulai saat ini",
            "untuk masa depan",
        ]

        for keyword in long_term_keywords:
            if keyword in text:
                score += 1

        # --------------------------------------------------------
        # CLAMP
        # --------------------------------------------------------

        return max(
            1,
            min(
                score,
                10,
            ),
        )

    # ============================================================
    # MEMORY COMMAND DETECTION
    # ============================================================

    def detect_memory_command(
        self,
        text: str,
    ) -> Optional[dict]:
        """
        Mendeteksi perintah memory.
        """

        normalized = self.normalize(text)

        if not normalized:
            return None

        # ========================================================
        # SAVE
        # ========================================================

        remember_patterns = [
            r"^ingat(?: bahwa)?\s+(.+)$",
            r"^ingat ya[, ]+(.+)$",
            r"^tolong ingat(?: bahwa)?\s+(.+)$",
            r"^simpan(?: bahwa)?\s+(.+)$",
            r"^catat(?: bahwa)?\s+(.+)$",
        ]

        for pattern in remember_patterns:

            match = re.match(
                pattern,
                normalized,
            )

            if match:

                content = (
                    match.group(1)
                    .strip()
                )

                return {
                    "action": "save",
                    "content": content,
                    "category": self.detect_category(
                        content
                    ),
                    "importance": self.calculate_importance(
                        content
                    ),
                }

        # ========================================================
        # COUNT
        # ========================================================

        count_patterns = [
            r"^berapa memory\??$",
            r"^berapa banyak memory\??$",
            r"^jumlah memory\??$",
            r"^memory ada berapa\??$",
        ]

        for pattern in count_patterns:

            if re.match(
                pattern,
                normalized,
            ):

                return {
                    "action": "count",
                }

        # ========================================================
        # QUERY
        # ========================================================

        query_patterns = [
            r"^apa yang kamu ingat.*$",
            r"^apa yang kamu ingat tentang saya.*$",
            r"^memory saya.*$",
            r"^lihat memory.*$",
            r"^tampilkan memory.*$",
            r"^apa saja memory.*$",
        ]

        for pattern in query_patterns:

            if re.match(
                pattern,
                normalized,
            ):

                return {
                    "action": "query",
                    "query": normalized,
                }

        # ========================================================
        # DELETE
        # ========================================================

        delete_patterns = [
            r"^hapus memory\s+(.+)$",
            r"^hapus memory tentang\s+(.+)$",
            r"^lupakan\s+(.+)$",
            r"^hapus ingatan\s+(.+)$",
        ]

        for pattern in delete_patterns:

            match = re.match(
                pattern,
                normalized,
            )

            if match:

                key = (
                    match.group(1)
                    .strip()
                )

                return {
                    "action": "delete",
                    "key": key,
                }

        return None

    # ============================================================
    # KEY GENERATION
    # ============================================================

    @classmethod
    def generate_key(
        cls,
        content: str,
    ) -> str:
        """
        Membuat key stabil dari isi memory.
        """

        content = cls.clean_content(
            content
        )

        return content[:cls.MAX_KEY_LENGTH]

    # ============================================================
    # DUPLICATE DETECTION
    # ============================================================

    def find_duplicate(
        self,
        content: str,
    ) -> Optional[dict]:
        """
        Mencari memory yang sama secara normalized.
        """

        normalized_content = self.normalize(
            content
        )

        if not normalized_content:
            return None

        results = self.search_memories(
            content,
            limit=self.SEARCH_LIMIT,
        )

        for item in results:

            existing_value = self.normalize(
                str(
                    item.get(
                        "value",
                        "",
                    )
                )
            )

            existing_key = self.normalize(
                str(
                    item.get(
                        "key",
                        "",
                    )
                )
            )

            if (
                existing_value
                == normalized_content
            ):
                return item

            if (
                existing_key
                == normalized_content
            ):
                return item

        return None

    # ============================================================
    # SAVE
    # ============================================================

    def save(
        self,
        content: str,
        category: Optional[str] = None,
        importance: Optional[int] = None,
        key: Optional[str] = None,
    ) -> dict:
        """
        Menyimpan memory secara intelligent.
        """

        content = self.clean_content(
            content
        )

        if not content:
            return {
                "success": False,
                "error": "Memory kosong.",
            }

        category = (
            category
            or self.detect_category(
                content
            )
        )

        if importance is None:
            importance = (
                self.calculate_importance(
                    content,
                    category,
                )
            )

        importance = max(
            1,
            min(
                int(importance),
                10,
            ),
        )

        memory_key = (
            key
            or self.generate_key(
                content
            )
        )

        duplicate = self.find_duplicate(
            content
        )

        # --------------------------------------------------------
        # DUPLICATE
        # --------------------------------------------------------

        if duplicate:

            existing_key = str(
                duplicate.get(
                    "key",
                    memory_key,
                )
            )

            existing_value = str(
                duplicate.get(
                    "value",
                    content,
                )
            )

            # Database save_memory menggunakan key sebagai
            # identifier utama. Dengan key existing, memory
            # akan diperbarui.
            save_memory(
                existing_key,
                content,
                category=category,
                importance=importance,
            )

            return {
                "success": True,
                "action": "updated",
                "duplicate": True,
                "key": existing_key,
                "value": content,
                "category": category,
                "importance": importance,
                "previous_value": existing_value,
            }

        # --------------------------------------------------------
        # NEW MEMORY
        # --------------------------------------------------------

        save_memory(
            memory_key,
            content,
            category=category,
            importance=importance,
        )

        return {
            "success": True,
            "action": "created",
            "duplicate": False,
            "key": memory_key,
            "value": content,
            "category": category,
            "importance": importance,
        }

    # ============================================================
    # COMPATIBILITY SAVE
    # ============================================================

    def save_memory(
        self,
        key: str,
        value: str,
        category: str = "general",
        importance: Optional[int] = None,
    ) -> dict:
        """
        Compatibility wrapper.
        """

        value = self.clean_content(
            value
        )

        if importance is None:
            importance = (
                self.calculate_importance(
                    value,
                    category,
                )
            )

        save_memory(
            key.strip(),
            value,
            category=category,
            importance=importance,
        )

        return {
            "success": True,
            "key": key.strip(),
            "value": value,
            "category": category,
            "importance": importance,
        }

    # ============================================================
    # GET
    # ============================================================

    def get(
        self,
        key: str,
    ) -> Optional[str]:
        """
        Mengambil memory berdasarkan key.
        """

        return get_memory(
            key.strip()
        )

    # ============================================================
    # COMPATIBILITY GET
    # ============================================================

    def get_memory(
        self,
        key: str,
    ) -> Optional[str]:

        return self.get(key)

    # ============================================================
    # SEARCH
    # ============================================================

    def search_memories(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Pencarian memory dasar dari database.
        """

        query = query.strip()

        if not query:
            return []

        try:
            results = search_memories(
                query,
                limit=max(
                    limit,
                    self.SEARCH_LIMIT,
                ),
            )
        except Exception:
            return []

        return [
            dict(item)
            for item in results
        ]

    # ============================================================
    # INTELLIGENT SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Intelligent memory search.

        Ranking:
        1. Exact match
        2. Word overlap
        3. Importance
        4. Recency dari database
        """

        query_normalized = self.normalize(
            query
        )

        if not query_normalized:
            return []

        results = self.search_memories(
            query,
            limit=max(
                limit * 3,
                self.SEARCH_LIMIT,
            ),
        )

        if not results:
            return []

        query_words = set(
            re.findall(
                r"\w+",
                query_normalized,
            )
        )

        ranked = []

        for item in results:

            key = self.normalize(
                str(
                    item.get(
                        "key",
                        "",
                    )
                )
            )

            value = self.normalize(
                str(
                    item.get(
                        "value",
                        "",
                    )
                )
            )

            combined = (
                f"{key} {value}"
            )

            item_words = set(
                re.findall(
                    r"\w+",
                    combined,
                )
            )

            overlap = len(
                query_words
                & item_words
            )

            exact = (
                query_normalized == key
                or query_normalized == value
            )

            importance = int(
                item.get(
                    "importance",
                    5,
                )
                or 5
            )

            score = 0.0

            # Exact match mendapat prioritas tertinggi.
            if exact:
                score += 100.0

            # Word overlap.
            score += (
                overlap * 10.0
            )

            # Importance.
            score += (
                importance * 2.0
            )

            # Substring.
            if (
                query_normalized in combined
            ):
                score += 15.0

            ranked.append(
                (
                    score,
                    item,
                )
            )

        ranked.sort(
            key=lambda pair: pair[0],
            reverse=True,
        )

        return [
            item
            for _, item
            in ranked[:limit]
        ]

    # ============================================================
    # IMPORTANT
    # ============================================================

    def important(
        self,
        limit: int = 10,
    ) -> list[dict]:
        """
        Mengambil memory paling penting.
        """

        try:
            results = get_important_memories(
                limit=limit
            )
        except TypeError:
            results = get_important_memories()

        except Exception:
            return []

        return [
            dict(item)
            for item in results[:limit]
        ]

    # ============================================================
    # COMPATIBILITY IMPORTANT
    # ============================================================

    def get_important_memories(
        self,
        limit: int = 10,
    ) -> list[dict]:

        return self.important(
            limit
        )

    # ============================================================
    # ALL
    # ============================================================

    def all(
        self,
        limit: int = 100,
    ) -> list[dict]:
        """
        Mengambil seluruh memory terbaru.
        """

        try:
            return [
                dict(item)
                for item in get_all_memories(
                    limit
                )
            ]
        except Exception:
            return []

    # ============================================================
    # COUNT
    # ============================================================

    def count(self) -> int:
        """
        Jumlah seluruh memory.
        """

        try:
            return int(
                count_memories()
            )
        except Exception:
            return 0

    # ============================================================
    # DELETE
    # ============================================================

    def delete(
        self,
        key: str,
    ) -> bool:
        """
        Menghapus memory.
        """

        key = key.strip()

        if not key:
            return False

        return bool(
            delete_memory(
                key
            )
        )

    # ============================================================
    # COMPATIBILITY DELETE
    # ============================================================

    def delete_memory(
        self,
        key: str,
    ) -> bool:

        return self.delete(
            key
        )

    # ============================================================
    # BUILD CONTEXT
    # ============================================================

    def build_context(
        self,
        query: str = "",
        limit: Optional[int] = None,
    ) -> str:
        """
        Membuat context memory untuk LLM.

        Memory yang paling relevan dan penting
        akan dimasukkan terlebih dahulu.
        """

        limit = (
            limit
            or self.CONTEXT_LIMIT
        )

        results: list[dict] = []

        # --------------------------------------------------------
        # QUERY RELEVANCE
        # --------------------------------------------------------

        if query.strip():

            results = self.search(
                query,
                limit=limit,
            )

        # --------------------------------------------------------
        # FALLBACK IMPORTANT MEMORY
        # --------------------------------------------------------

        if not results:

            results = self.important(
                limit=limit
            )

        if not results:
            return ""

        lines = []

        seen = set()

        for item in results:

            key = str(
                item.get(
                    "key",
                    "",
                )
            ).strip()

            value = str(
                item.get(
                    "value",
                    "",
                )
            ).strip()

            category = str(
                item.get(
                    "category",
                    "general",
                )
            ).strip()

            if not value:
                continue

            normalized = self.normalize(
                value
            )

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            lines.append(
                f"- [{category}] "
                f"{key}: {value}"
            )

            if len(lines) >= limit:
                break

        return "\n".join(
            lines
        )

    # ============================================================
    # HANDLE COMMAND
    # ============================================================

    def handle_command(
        self,
        text: str,
    ) -> dict:
        """
        Menangani command memory secara langsung.
        """

        command = (
            self.detect_memory_command(
                text
            )
        )

        if not command:

            return {
                "type": "memory",
                "action": "none",
                "success": False,
                "message": "Bukan memory command.",
            }

        action = command.get(
            "action"
        )

        # ========================================================
        # SAVE
        # ========================================================

        if action == "save":

            result = self.save(
                command.get(
                    "content",
                    "",
                ),
                category=command.get(
                    "category"
                ),
                importance=command.get(
                    "importance"
                ),
            )

            return {
                "type": "memory",
                "action": "save",
                **result,
            }

        # ========================================================
        # COUNT
        # ========================================================

        if action == "count":

            total = self.count()

            return {
                "type": "memory",
                "action": "count",
                "success": True,
                "count": total,
            }

        # ========================================================
        # QUERY
        # ========================================================

        if action == "query":

            memories = self.important(
                limit=self.IMPORTANT_LIMIT
            )

            if not memories:

                memories = self.all(
                    limit=self.IMPORTANT_LIMIT
                )

            return {
                "type": "memory",
                "action": "query",
                "success": True,
                "count": len(memories),
                "memories": memories,
            }

        # ========================================================
        # DELETE
        # ========================================================

        if action == "delete":

            key = str(
                command.get(
                    "key",
                    "",
                )
            ).strip()

            deleted = self.delete(
                key
            )

            return {
                "type": "memory",
                "action": "delete",
                "success": deleted,
                "key": key,
            }

        return {
            "type": "memory",
            "action": action,
            "success": False,
            "message": "Memory action tidak dikenali.",
        }

    # ============================================================
    # MEMORY SUMMARY
    # ============================================================

    def stats(self) -> dict:
        """
        Statistik memory untuk monitoring ZAI.
        """

        total = self.count()

        important = self.important(
            limit=self.IMPORTANT_LIMIT
        )

        return {
            "enabled": True,
            "count": total,
            "important_count": len(
                important
            ),
            "engine": "MemoryManager",
            "intelligence": True,
            "deduplication": True,
            "auto_category": True,
            "importance_scoring": True,
            "relevance_search": True,
        }


# ================================================================
# SINGLETON
# ================================================================

_memory_manager: Optional[
    MemoryManager
] = None


def get_memory_manager() -> MemoryManager:
    """
    Singleton MemoryManager.
    """

    global _memory_manager

    if _memory_manager is None:

        _memory_manager = (
            MemoryManager()
        )

    return _memory_manager