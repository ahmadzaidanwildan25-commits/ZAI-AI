from __future__ import annotations

import re
from typing import Optional

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
    Super ZAI Long-Term Memory Manager.

    Responsibilities:
    - Detect explicit memory commands.
    - Save memories.
    - Retrieve memories.
    - Search memories.
    - Delete memories.
    - Build compact memory context for the AI.
    - Keep database operations lightweight.
    """

    MEMORY_LIMIT = 5
    IMPORTANT_LIMIT = 5
    SEARCH_LIMIT = 5

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self) -> None:
        """
        MemoryManager tidak membutuhkan koneksi database permanen.

        Database connection dibuat oleh database.py hanya ketika
        dibutuhkan sehingga lebih aman dan ringan.
        """
        pass

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize(text: str) -> str:
        """
        Membersihkan whitespace dan membuat text lowercase.
        """

        return " ".join(
            text.strip().lower().split()
        )

    # ========================================================
    # MEMORY COMMAND DETECTION
    # ========================================================

    def detect_memory_command(
        self,
        text: str,
    ) -> Optional[dict]:
        """
        Mendeteksi perintah memory eksplisit dari user.

        Contoh:

        ingat saya suka coding
        simpan bahwa saya sedang membangun Super ZAI
        catat saya menggunakan Windows 11
        """

        normalized = self.normalize(text)

        # ----------------------------------------------------
        # SAVE / REMEMBER
        # ----------------------------------------------------

        remember_patterns = [
            r"^ingat(?: bahwa)? (.+)$",
            r"^ingat ya[, ]+(.+)$",
            r"^tolong ingat(?: bahwa)? (.+)$",
            r"^simpan(?: bahwa)? (.+)$",
            r"^catat(?: bahwa)? (.+)$",
        ]

        for pattern in remember_patterns:
            match = re.match(
                pattern,
                normalized,
            )

            if match:
                content = match.group(1).strip()

                if not content:
                    return None

                return {
                    "action": "save",
                    "content": content,
                }

        # ----------------------------------------------------
        # FORGET / DELETE
        # ----------------------------------------------------

        forget_patterns = [
            r"^lupakan (.+)$",
            r"^hapus memory (.+)$",
            r"^hapus ingatan (.+)$",
            r"^lupakan bahwa (.+)$",
        ]

        for pattern in forget_patterns:
            match = re.match(
                pattern,
                normalized,
            )

            if match:
                content = match.group(1).strip()

                if not content:
                    return None

                return {
                    "action": "delete",
                    "content": content,
                }

        # ----------------------------------------------------
        # MEMORY COUNT
        # ----------------------------------------------------

        count_patterns = [
            r"^berapa memory.*$",
            r"^berapa ingatan.*$",
            r"^berapa banyak memory.*$",
        ]

        for pattern in count_patterns:
            if re.match(pattern, normalized):
                return {
                    "action": "count",
                }

        # ----------------------------------------------------
        # MEMORY SEARCH / RECALL
        # ----------------------------------------------------

        recall_patterns = [
            r"^apa yang kamu ingat tentang (.+)$",
            r"^apa yang kamu ingat mengenai (.+)$",
            r"^ingatanku tentang (.+)$",
            r"^cari memory tentang (.+)$",
            r"^cari ingatan tentang (.+)$",
        ]

        for pattern in recall_patterns:
            match = re.match(
                pattern,
                normalized,
            )

            if match:
                content = match.group(1).strip()

                if not content:
                    return None

                return {
                    "action": "search",
                    "content": content,
                }

        # ----------------------------------------------------
        # SHOW IMPORTANT
        # ----------------------------------------------------

        important_patterns = [
            r"^memory penting$",
            r"^ingatan penting$",
            r"^tampilkan memory penting$",
            r"^tampilkan ingatan penting$",
        ]

        for pattern in important_patterns:
            if re.match(pattern, normalized):
                return {
                    "action": "important",
                }

        return None

    # ========================================================
    # CATEGORY DETECTION
    # ========================================================

    @staticmethod
    def detect_category(content: str) -> str:
        """
        Menentukan kategori memory sederhana.
        """

        text = content.lower()

        project_keywords = [
            "membangun",
            "project",
            "proyek",
            "aplikasi",
            "app",
            "super zai",
            "zai",
            "coding",
            "program",
        ]

        preference_keywords = [
            "suka",
            "tidak suka",
            "favorit",
            "senang",
            "lebih suka",
        ]

        personal_keywords = [
            "nama saya",
            "namaku",
            "saya tinggal",
            "umur saya",
            "saya berumur",
        ]

        for keyword in project_keywords:
            if keyword in text:
                return "project"

        for keyword in preference_keywords:
            if keyword in text:
                return "preference"

        for keyword in personal_keywords:
            if keyword in text:
                return "personal"

        return "general"

    # ========================================================
    # IMPORTANCE
    # ========================================================

    @staticmethod
    def calculate_importance(
        content: str,
        category: str,
    ) -> int:
        """
        Menghitung prioritas memory.

        Skala:
        1  = rendah
        5  = normal
        10 = sangat penting
        """

        score = 5

        important_words = [
            "nama saya",
            "namaku",
            "saya adalah",
            "saya sedang membangun",
            "super zai",
            "proyek utama",
            "ingat ini",
            "penting",
        ]

        for keyword in important_words:
            if keyword in content.lower():
                score += 2

        if category in {
            "personal",
            "project",
        }:
            score += 1

        return max(
            1,
            min(score, 10),
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        content: str,
        category: Optional[str] = None,
        importance: Optional[int] = None,
    ) -> dict:
        """
        Menyimpan memory.

        Karena database menggunakan key sebagai identifier,
        content digunakan sebagai key default agar memory
        sederhana tetap dapat ditemukan kembali.
        """

        content = content.strip()

        if not content:
            return {
                "success": False,
                "error": "Memory content is empty.",
            }

        final_category = (
            category.strip()
            if category
            else self.detect_category(content)
        )

        final_importance = (
            importance
            if importance is not None
            else self.calculate_importance(
                content,
                final_category,
            )
        )

        final_importance = max(
            1,
            min(
                int(final_importance),
                10,
            ),
        )

        # Database versi sekarang mendukung key/value/category.
        # Importance ditangani oleh database.py versi yang sudah
        # kita test sebelumnya.
        save_memory(
            key=content,
            value=content,
            category=final_category,
            importance=final_importance,
        )

        return {
            "success": True,
            "key": content,
            "value": content,
            "category": final_category,
            "importance": final_importance,
        }

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

        key = key.strip()

        if not key:
            return None

        return get_memory(key)

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        limit: int = SEARCH_LIMIT,
    ) -> list[dict]:
        """
        Mencari memory berdasarkan query.
        """

        query = query.strip()

        if not query:
            return []

        return search_memories(
            query=query,
            limit=min(
                max(limit, 1),
                20,
            ),
        )

    # ========================================================
    # IMPORTANT
    # ========================================================

    def important(
        self,
        limit: int = IMPORTANT_LIMIT,
    ) -> list[dict]:
        """
        Mengambil memory paling penting.
        """

        return get_important_memories(
            limit=min(
                max(limit, 1),
                20,
            ),
        )

    # ========================================================
    # ALL
    # ========================================================

    def all(
        self,
        limit: int = 100,
    ) -> list[dict]:
        """
        Mengambil seluruh memory terbaru.
        """

        return get_all_memories(
            limit=min(
                max(limit, 1),
                500,
            ),
        )

    # ========================================================
    # DELETE
    # ========================================================

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

        return delete_memory(key)

    # ========================================================
    # COUNT
    # ========================================================

    def count(self) -> int:
        """
        Menghitung total memory.
        """

        return count_memories()

    # ========================================================
    # HANDLE COMMAND
    # ========================================================

    def handle_command(
        self,
        text: str,
    ) -> Optional[dict]:
        """
        Menjalankan perintah memory jika ditemukan.

        Return None berarti:
        bukan memory command.
        """

        command = self.detect_memory_command(text)

        if command is None:
            return None

        action = command["action"]

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        if action == "save":

            result = self.save(
                command["content"],
            )

            return {
                "type": "memory",
                "action": "save",
                **result,
            }

        # ----------------------------------------------------
        # DELETE
        # ----------------------------------------------------

        if action == "delete":

            content = command["content"]

            deleted = self.delete(
                content,
            )

            return {
                "type": "memory",
                "action": "delete",
                "success": deleted,
                "key": content,
            }

        # ----------------------------------------------------
        # COUNT
        # ----------------------------------------------------

        if action == "count":

            return {
                "type": "memory",
                "action": "count",
                "success": True,
                "count": self.count(),
            }

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        if action == "search":

            results = self.search(
                command["content"],
            )

            return {
                "type": "memory",
                "action": "search",
                "success": True,
                "query": command["content"],
                "results": results,
            }

        # ----------------------------------------------------
        # IMPORTANT
        # ----------------------------------------------------

        if action == "important":

            results = self.important()

            return {
                "type": "memory",
                "action": "important",
                "success": True,
                "results": results,
            }

        return None

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    def build_context(
        self,
        query: str,
    ) -> str:
        """
        Membangun memory context ringkas untuk diberikan
        kepada model AI.

        Strategi:

        1. Ambil memory penting.
        2. Cari memory yang relevan dengan query.
        3. Hindari duplicate.
        4. Batasi jumlah memory.
        5. Batasi ukuran context.
        """

        query = query.strip()

        if not query:
            return ""

        collected: list[dict] = []
        seen: set[int] = set()

        # ----------------------------------------------------
        # IMPORTANT MEMORY
        # ----------------------------------------------------

        try:
            important = self.important(
                self.IMPORTANT_LIMIT,
            )
        except Exception:
            important = []

        for memory in important:

            memory_id = memory.get("id")

            if memory_id is not None:
                if memory_id in seen:
                    continue

                seen.add(memory_id)

            collected.append(memory)

        # ----------------------------------------------------
        # RELEVANT MEMORY
        # ----------------------------------------------------

        try:
            relevant = self.search(
                query,
                self.SEARCH_LIMIT,
            )
        except Exception:
            relevant = []

        for memory in relevant:

            memory_id = memory.get("id")

            if memory_id is not None:

                if memory_id in seen:
                    continue

                seen.add(memory_id)

            collected.append(memory)

        # ----------------------------------------------------
        # LIMIT
        # ----------------------------------------------------

        collected = collected[
            : self.MEMORY_LIMIT
        ]

        if not collected:
            return ""

        # ----------------------------------------------------
        # FORMAT
        # ----------------------------------------------------

        lines: list[str] = []

        for memory in collected:

            category = str(
                memory.get(
                    "category",
                    "general",
                )
            )

            key = str(
                memory.get(
                    "key",
                    "",
                )
            )

            value = str(
                memory.get(
                    "value",
                    "",
                )
            )

            if not value:
                continue

            lines.append(
                f"- [{category}] {value}"
            )

        if not lines:
            return ""

        return (
            "MEMORY CONTEXT:\n"
            + "\n".join(lines)
        )