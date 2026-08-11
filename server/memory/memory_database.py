"""
ZAI Memory Database
Super ZAI - Persistent Memory Core

Database:
    data/zai_memory.db
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Any


# ============================================================
# PATH CONFIGURATION
# ============================================================

SERVER_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = SERVER_DIR / "data"

DATABASE_PATH = DATA_DIR / "zai_memory.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection() -> sqlite3.Connection:
    """
    Membuka koneksi SQLite database ZAI.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=10,
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# SCHEMA MIGRATION
# ============================================================

def _ensure_column(
    cursor: sqlite3.Cursor,
    column_name: str,
    column_definition: str,
) -> None:
    """
    Menambahkan kolom jika belum tersedia.

    Digunakan agar database lama tetap kompatibel
    ketika schema ZAI mengalami upgrade.
    """

    cursor.execute(
        "PRAGMA table_info(memories)"
    )

    columns = {
        str(row["name"])
        for row in cursor.fetchall()
    }

    if column_name not in columns:
        cursor.execute(
            f"""
            ALTER TABLE memories
            ADD COLUMN {column_name} {column_definition}
            """
        )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database() -> None:
    """
    Membuat database dan melakukan migrasi schema
    tanpa menghapus memory yang sudah ada.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = get_connection()

    try:
        cursor = connection.cursor()

        # ----------------------------------------------------
        # BASE TABLE
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL DEFAULT 'general',
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                importance INTEGER NOT NULL DEFAULT 5,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # ----------------------------------------------------
        # MIGRATION
        # ----------------------------------------------------

        _ensure_column(
            cursor,
            "category",
            "TEXT NOT NULL DEFAULT 'general'",
        )

        _ensure_column(
            cursor,
            "key",
            "TEXT NOT NULL DEFAULT ''",
        )

        _ensure_column(
            cursor,
            "value",
            "TEXT NOT NULL DEFAULT ''",
        )

        _ensure_column(
            cursor,
            "importance",
            "INTEGER NOT NULL DEFAULT 5",
        )

        _ensure_column(
            cursor,
            "created_at",
            "DATETIME",
        )

        _ensure_column(
            cursor,
            "updated_at",
            "DATETIME",
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
            ON memories(importance DESC)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_memories_updated
            ON memories(updated_at DESC)
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
    importance: int = 5,
) -> None:
    """
    Menyimpan atau memperbarui memory.
    """

    key = str(key or "").strip()
    value = str(value or "").strip()
    category = str(category or "").strip() or "general"

    if not key or not value:
        return

    importance = max(
        1,
        min(
            int(importance),
            10,
        ),
    )

    initialize_database()

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id
            FROM memories
            WHERE LOWER(key) = LOWER(?)
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
    """
    Mengambil satu memory berdasarkan key.
    """

    key = str(key or "").strip()

    if not key:
        return None

    initialize_database()

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT value
            FROM memories
            WHERE LOWER(key) = LOWER(?)
            LIMIT 1
            """,
            (key,),
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
) -> list[dict[str, Any]]:
    """
    Mencari memory berdasarkan key, value, atau category.
    """

    query = str(query or "").strip()

    if not query:
        return []

    limit = max(
        1,
        min(
            int(limit),
            100,
        ),
    )

    initialize_database()

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
# GET ALL MEMORIES
# ============================================================

def get_all_memories(
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Mengambil memory terbaru berdasarkan importance
    dan waktu update.
    """

    limit = max(
        1,
        min(
            int(limit),
            500,
        ),
    )

    initialize_database()

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
# IMPORTANT MEMORIES
# ============================================================

def get_important_memories(
    limit: int = 5,
    minimum_importance: int = 7,
) -> list[dict[str, Any]]:
    """
    Mengambil memory yang dianggap penting.
    """

    limit = max(
        1,
        min(
            int(limit),
            100,
        ),
    )

    minimum_importance = max(
        1,
        min(
            int(minimum_importance),
            10,
        ),
    )

    initialize_database()

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
            WHERE importance >= ?
            ORDER BY
                importance DESC,
                updated_at DESC
            LIMIT ?
            """,
            (
                minimum_importance,
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
# DELETE MEMORY
# ============================================================

def delete_memory(
    key: str,
) -> bool:
    """
    Menghapus memory berdasarkan key
    secara case-insensitive.
    """

    key = str(key or "").strip()

    if not key:
        return False

    initialize_database()

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM memories
            WHERE LOWER(key) = LOWER(?)
            """,
            (key,),
        )

        deleted = cursor.rowcount > 0

        connection.commit()

        return deleted

    finally:
        connection.close()


# ============================================================
# CLEAR MEMORIES
# ============================================================

def clear_memories() -> None:
    """
    Menghapus seluruh memory ZAI.
    """

    initialize_database()

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
    """
    Menghitung jumlah seluruh memory ZAI.
    """

    initialize_database()

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

def database_status() -> dict[str, Any]:
    """
    Informasi status database memory.
    """

    initialize_database()

    return {
        "database": "zai_memory.db",
        "path": str(DATABASE_PATH),
        "exists": DATABASE_PATH.exists(),
        "memory_count": count_memories(),
        "status": "READY",
    }


# ============================================================
# AUTO INITIALIZATION
# ============================================================

initialize_database()
