import sqlite3
from pathlib import Path
from typing import Optional

# ============================================================
# ZAI MEMORY DATABASE
# SUPER ZAI - PERSISTENT MEMORY ENGINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "zai_memory.db"


# ============================================================
# CONNECTION
# ============================================================

def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        str(DATABASE_PATH),
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DATABASE INITIALIZATION + MIGRATION
# ============================================================

def initialize_database() -> None:
    """
    Membuat database memory ZAI.

    Database lama tetap dipertahankan.
    Kolom baru akan ditambahkan otomatis melalui migration.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = get_connection()

    try:
        cursor = connection.cursor()

        # ----------------------------------------------------
        # MAIN TABLE
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL DEFAULT 'general',
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL,
                importance INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # ----------------------------------------------------
        # MIGRATION
        # ----------------------------------------------------

        cursor.execute(
            "PRAGMA table_info(memories)"
        )

        columns = {
            row["name"]
            for row in cursor.fetchall()
        }

        # Add category if old DB does not have it.
        if "category" not in columns:

            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN category
                TEXT NOT NULL DEFAULT 'general'
                """
            )

        # Add key if old DB does not have it.
        if "key" not in columns:

            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN key
                TEXT
                """
            )

        # Add value if old DB does not have it.
        if "value" not in columns:

            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN value
                TEXT
                """
            )

        # Add importance.
        if "importance" not in columns:

            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN importance
                INTEGER NOT NULL DEFAULT 1
                """
            )

        # Add timestamps.
        if "created_at" not in columns:

            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN created_at
                DATETIME DEFAULT CURRENT_TIMESTAMP
                """
            )

        if "updated_at" not in columns:

            cursor.execute(
                """
                ALTER TABLE memories
                ADD COLUMN updated_at
                DATETIME DEFAULT CURRENT_TIMESTAMP
                """
            )

        # ----------------------------------------------------
        # INDEXES
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_memories_key
            ON memories(key)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_memories_category
            ON memories(category)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_memories_importance
            ON memories(importance)
            """
        )

        # ----------------------------------------------------
        # CLEAN OLD NULL DATA
        # ----------------------------------------------------

        cursor.execute(
            """
            UPDATE memories
            SET category = 'general'
            WHERE category IS NULL
               OR TRIM(category) = ''
            """
        )

        cursor.execute(
            """
            UPDATE memories
            SET importance = 1
            WHERE importance IS NULL
            """
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# SAVE MEMORY
# ============================================================

def save_memory(
    key: str,
    value: str,
    category: str = "general",
    importance: int = 1,
) -> None:

    key = key.strip()
    value = value.strip()
    category = category.strip() or "general"

    if not key or not value:
        return

    try:
        importance = int(importance)
    except (TypeError, ValueError):
        importance = 1

    importance = max(
        1,
        min(importance, 10),
    )

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id
            FROM memories
            WHERE key = ?
            LIMIT 1
            """,
            (key,),
        )

        existing = cursor.fetchone()

        if existing:

            cursor.execute(
                """
                UPDATE memories
                SET
                    value = ?,
                    category = ?,
                    importance = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    value,
                    category,
                    importance,
                    existing["id"],
                ),
            )

        else:

            cursor.execute(
                """
                INSERT INTO memories (
                    category,
                    key,
                    value,
                    importance
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    category,
                    key,
                    value,
                    importance,
                ),
            )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# GET MEMORY
# ============================================================

def get_memory(
    key: str,
) -> Optional[str]:

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT value
            FROM memories
            WHERE key = ?
            LIMIT 1
            """,
            (key.strip(),),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return str(row["value"])

    finally:

        connection.close()


# ============================================================
# SEARCH MEMORIES
# ============================================================

def search_memories(
    query: str,
    limit: int = 10,
) -> list[dict]:

    query = query.strip()

    if not query:
        return []

    limit = max(
        1,
        min(int(limit), 100),
    )

    connection = get_connection()

    try:

        cursor = connection.cursor()

        pattern = f"%{query}%"

        cursor.execute(
            """
            SELECT
                id,
                category,
                key,
                value,
                importance,
                created_at,
                updated_at
            FROM memories
            WHERE
                key LIKE ?
                OR value LIKE ?
                OR category LIKE ?
            ORDER BY
                importance DESC,
                updated_at DESC
            LIMIT ?
            """,
            (
                pattern,
                pattern,
                pattern,
                limit,
            ),
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# IMPORTANT MEMORIES
# ============================================================

def get_important_memories(
    limit: int = 10,
) -> list[dict]:

    limit = max(
        1,
        min(int(limit), 100),
    )

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                category,
                key,
                value,
                importance,
                created_at,
                updated_at
            FROM memories
            WHERE importance >= 5
            ORDER BY
                importance DESC,
                updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# GET ALL MEMORIES
# ============================================================

def get_all_memories(
    limit: int = 100,
) -> list[dict]:

    limit = max(
        1,
        min(int(limit), 500),
    )

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                category,
                key,
                value,
                importance,
                created_at,
                updated_at
            FROM memories
            ORDER BY
                importance DESC,
                updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# DELETE MEMORY
# ============================================================

def delete_memory(
    key: str,
) -> bool:

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM memories
            WHERE key = ?
            """,
            (key.strip(),),
        )

        deleted = cursor.rowcount > 0

        connection.commit()

        return deleted

    finally:

        connection.close()


# ============================================================
# CLEAR ALL MEMORY
# ============================================================

def clear_memories() -> None:

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM memories
            """
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# COUNT MEMORIES
# ============================================================

def count_memories() -> int:

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM memories
            """
        )

        row = cursor.fetchone()

        if row is None:
            return 0

        return int(row["total"])

    finally:

        connection.close()


# ============================================================
# DATABASE STATUS
# ============================================================

def database_status() -> dict:

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            "PRAGMA table_info(memories)"
        )

        columns = [
            row["name"]
            for row in cursor.fetchall()
        ]

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM memories
            """
        )

        row = cursor.fetchone()

        total = (
            int(row["total"])
            if row
            else 0
        )

        return {
            "database": str(DATABASE_PATH),
            "exists": DATABASE_PATH.exists(),
            "table": "memories",
            "columns": columns,
            "memory_count": total,
        }

    finally:

        connection.close()


# ============================================================
# INITIALIZE ON IMPORT
# ============================================================

initialize_database()
