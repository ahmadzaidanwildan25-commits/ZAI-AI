import re
from typing import Optional

from .database import (
    count_memories,
    delete_memory,
    get_important_memories,
    get_memory,
    save_memory,
    search_memories,
)


class MemoryManager:
    """
    High-speed memory manager for Super ZAI.

    Responsibilities:

    - Detect explicit memory commands.
    - Store long-term memories.
    - Search memories.
    - Build memory context.
    - Avoid unnecessary database work.
    """

    MEMORY_LIMIT = 5

    def __init__(self) -> None:
        pass

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize(text: str) -> str:
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

        normalized = self.normalize(text)

        # ----------------------------------------------------
        # REMEMBER / INGAT
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

                return {
                    "action": "save",
                    "content": content,
                }

        # ----------------------------------------------------
        # FORGET
        # ----------------------------------------------------

        forget_patterns = [
            r"^lupakan (.+)$",
            r"^hapus ingatan (.+)$",
            r"^jangan ingat (.+)$",
        ]

        for pattern in forget_patterns:

            match = re.match(
                pattern,
                normalized,
            )

            if match:

                content = match.group(1).strip()

                return {
                    "action": "forget",
                    "content": content,
                }

        # ----------------------------------------------------
        # SHOW MEMORY
        # ----------------------------------------------------

        show_patterns = [
            "apa yang kamu ingat tentang saya",
            "apa yang kamu ingat",
            "tampilkan ingatan",
            "lihat ingatan",
            "memory saya",
            "ingatanku",
        ]

        if normalized in show_patterns:

            return {
                "action": "show",
            }

        return None

    # ========================================================
    # AUTOMATIC MEMORY EXTRACTION
    # ========================================================

    def detect_explicit_fact(
        self,
        text: str,
    ) -> Optional[dict]:

        normalized = self.normalize(text)

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        name_patterns = [
            r"nama saya (.+)",
            r"namaku (.+)",
            r"saya bernama (.+)",
            r"panggil saya (.+)",
        ]

        for pattern in name_patterns:

            match = re.search(
                pattern,
                normalized,
            )

            if match:

                name = match.group(1).strip()

                if 1 <= len(name) <= 80:

                    return {
                        "category": "profile",
                        "key": "name",
                        "value": name,
                        "importance": 10,
                    }

        # ----------------------------------------------------
        # PREFERENCE
        # ----------------------------------------------------

        preference_patterns = [
            r"saya suka (.+)",
            r"aku suka (.+)",
            r"saya senang (.+)",
            r"aku senang (.+)",
        ]

        for pattern in preference_patterns:

            match = re.search(
                pattern,
                normalized,
            )

            if match:

                value = match.group(1).strip()

                if 1 <= len(value) <= 300:

                    return {
                        "category": "preference",
                        "key": f"suka_{value[:50]}",
                        "value": value,
                        "importance": 6,
                    }

        # ----------------------------------------------------
        # PROJECT
        # ----------------------------------------------------

        project_patterns = [
            r"proyek saya (.+)",
            r"project saya (.+)",
            r"aplikasi saya (.+)",
        ]

        for pattern in project_patterns:

            match = re.search(
                pattern,
                normalized,
            )

            if match:

                value = match.group(1).strip()

                if 1 <= len(value) <= 300:

                    return {
                        "category": "project",
                        "key": "current_project",
                        "value": value,
                        "importance": 8,
                    }

        return None

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        category: str,
        key: str,
        value: str,
        importance: int = 5,
    ) -> int:

        importance = max(
            1,
            min(10, importance),
        )

        return save_memory(
            category=category,
            key=key,
            value=value,
            importance=importance,
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        limit: int = MEMORY_LIMIT,
    ) -> list[dict]:

        rows = search_memories(
            query=query,
            limit=limit,
        )

        return [
            {
                "id": row["id"],
                "category": row["category"],
                "key": row["memory_key"],
                "value": row["memory_value"],
                "importance": row["importance"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    # ========================================================
    # IMPORTANT MEMORIES
    # ========================================================

    def important(
        self,
        limit: int = 10,
    ) -> list[dict]:

        rows = get_important_memories(
            limit=limit,
        )

        return [
            {
                "id": row["id"],
                "category": row["category"],
                "key": row["memory_key"],
                "value": row["memory_value"],
                "importance": row["importance"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    # ========================================================
    # CONTEXT
    # ========================================================

    def build_context(
        self,
        user_message: str,
    ) -> str:

        memories = self.search(
            user_message,
            limit=self.MEMORY_LIMIT,
        )

        if not memories:
            return ""

        lines = [
            "RELEVANT USER MEMORY:",
        ]

        for memory in memories:

            lines.append(
                f"- "
                f"{memory['category']}/"
                f"{memory['key']}: "
                f"{memory['value']}"
            )

        return "\n".join(lines)

    # ========================================================
    # FORGET
    # ========================================================

    def forget(
        self,
        category: str,
        key: str,
    ) -> bool:

        return delete_memory(
            category=category,
            key=key,
        )

    # ========================================================
    # COUNT
    # ========================================================

    def count(self) -> int:
        return count_memories()

    # ========================================================
    # HANDLE COMMAND
    # ========================================================

    def handle_command(
        self,
        text: str,
    ) -> Optional[dict]:

        command = self.detect_memory_command(
            text
        )

        if not command:
            return None

        action = command["action"]

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        if action == "save":

            content = command["content"]

            memory = self.detect_explicit_fact(
                content
            )

            if memory is None:

                memory = {
                    "category": "general",
                    "key": content[:80],
                    "value": content,
                    "importance": 5,
                }

            memory_id = self.save(
                category=memory["category"],
                key=memory["key"],
                value=memory["value"],
                importance=memory["importance"],
            )

            return {
                "action": "save",
                "success": True,
                "memory_id": memory_id,
                "memory": memory,
            }

        # ----------------------------------------------------
        # FORGET
        # ----------------------------------------------------

        if action == "forget":

            content = command["content"]

            memories = self.search(
                content,
                limit=5,
            )

            deleted = 0

            for memory in memories:

                if self.forget(
                    memory["category"],
                    memory["key"],
                ):
                    deleted += 1

            return {
                "action": "forget",
                "success": True,
                "deleted": deleted,
            }

        # ----------------------------------------------------
        # SHOW
        # ----------------------------------------------------

        if action == "show":

            memories = self.important(
                limit=20
            )

            return {
                "action": "show",
                "success": True,
                "memories": memories,
            }

        return None