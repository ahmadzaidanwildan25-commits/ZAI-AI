from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Optional

from memory.database import count_memories, get_all_memories
from memory.manager import MemoryManager


class IntentHandlers:
    """
    Handler execution layer for Super ZAI.

    Tugas:
    - Menjalankan aksi berdasarkan intent.
    - Menghubungkan intent dengan MemoryManager.
    - Menyediakan response standar.
    - Menjaga handler tetap aman dan terisolasi.
    """

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
    ) -> None:
        self.memory = memory_manager or MemoryManager()

        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "greeting": self.handle_greeting,
            "identity": self.handle_identity,
            "memory_save": self.handle_memory_save,
            "memory_query": self.handle_memory_query,
            "memory_delete": self.handle_memory_delete,
            "memory_count": self.handle_memory_count,
            "status": self.handle_status,
            "help": self.handle_help,
            "calculation": self.handle_calculation,
            "coding": self.handle_coding,
            "search": self.handle_search,
            "general": self.handle_general,
        }

    # =========================================================
    # CORE
    # =========================================================

    def has_handler(self, intent: str) -> bool:
        return intent.strip().lower() in self._handlers

    def available_intents(self) -> list[str]:
        return sorted(self._handlers.keys())

    def handle(
        self,
        intent: str,
        text: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:

        normalized_intent = str(intent).strip().lower()

        handler = self._handlers.get(normalized_intent)

        if handler is None:
            return self._result(
                success=False,
                intent=normalized_intent,
                message="Intent belum memiliki handler.",
            )

        try:
            return handler(text=text, **kwargs)

        except Exception as error:
            return self._result(
                success=False,
                intent=normalized_intent,
                message="Terjadi kesalahan saat menjalankan intent.",
                error=str(error),
            )

    # =========================================================
    # RESULT
    # =========================================================

    @staticmethod
    def _result(
        success: bool,
        intent: str,
        message: str = "",
        **extra: Any,
    ) -> dict[str, Any]:

        result: dict[str, Any] = {
            "success": success,
            "intent": intent,
            "message": message,
        }

        result.update(extra)

        return result

    # =========================================================
    # GREETING
    # =========================================================

    def handle_greeting(
        self,
        text: str = "",
        **_: Any,
    ) -> dict[str, Any]:

        return self._result(
            True,
            "greeting",
            "Halo! Saya ZAI. Siap membantu Anda.",
        )

    # =========================================================
    # IDENTITY
    # =========================================================

    def handle_identity(
        self,
        text: str = "",
        **_: Any,
    ) -> dict[str, Any]:

        return self._result(
            True,
            "identity",
            (
                "Saya ZAI, intelligence core dari Super ZAI. "
                "Saya dirancang untuk membantu Anda dengan "
                "percakapan, memory, analisis, coding, dan "
                "berbagai kemampuan AI."
            ),
        )

    # =========================================================
    # MEMORY SAVE
    # =========================================================

    def handle_memory_save(
        self,
        text: str = "",
        content: Optional[str] = None,
        category: str = "user",
        importance: int = 10,
        **_: Any,
    ) -> dict[str, Any]:

        value = (
            content.strip()
            if content
            else text.strip()
        )

        if not value:
            return self._result(
                False,
                "memory_save",
                "Memory yang ingin disimpan kosong.",
            )

        try:
            result = self.memory.save(
                value,
                category=category,
                importance=importance,
            )

            if isinstance(result, dict):
                return self._result(
                    bool(result.get("success", True)),
                    "memory_save",
                    "Baik. Saya akan mengingatnya.",
                    memory=result,
                )

            return self._result(
                True,
                "memory_save",
                "Baik. Saya akan mengingatnya.",
            )

        except AttributeError:
            # Compatibility dengan MemoryManager versi sebelumnya.
            try:
                result = self.memory.save_memory(
                    value,
                    value,
                    category=category,
                    importance=importance,
                )
            except TypeError:
                result = self.memory.save_memory(
                    value,
                    value,
                    category=category,
                )

            return self._result(
                True,
                "memory_save",
                "Baik. Saya akan mengingatnya.",
                memory=result,
            )

    # =========================================================
    # MEMORY QUERY
    # =========================================================

    def handle_memory_query(
        self,
        text: str = "",
        query: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:

        search_query = (
            query.strip()
            if query
            else text.strip()
        )

        try:
            if search_query:
                try:
                    memories = self.memory.search(
                        search_query,
                        limit=10,
                    )
                except AttributeError:
                    memories = self.memory.search_memories(
                        search_query,
                        limit=10,
                    )
            else:
                memories = get_all_memories(
                    limit=10,
                )

        except Exception as error:
            return self._result(
                False,
                "memory_query",
                "Gagal membaca memory.",
                error=str(error),
            )

        if not memories:
            return self._result(
                True,
                "memory_query",
                "Saya belum menemukan memory yang relevan.",
                memories=[],
            )

        lines: list[str] = []

        for item in memories:
            category = str(
                item.get("category", "general")
            ).strip()

            value = str(
                item.get("value", "")
            ).strip()

            if value:
                lines.append(
                    f"- [{category}] {value}"
                )

        message = (
            "Memory yang saya temukan:\n"
            + "\n".join(lines)
        )

        return self._result(
            True,
            "memory_query",
            message,
            memories=memories,
        )

    # =========================================================
    # MEMORY DELETE
    # =========================================================

    def handle_memory_delete(
        self,
        text: str = "",
        key: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:

        target = (
            key.strip()
            if key
            else text.strip()
        )

        if not target:
            return self._result(
                False,
                "memory_delete",
                "Key memory tidak diberikan.",
            )

        try:
            try:
                deleted = self.memory.delete(
                    target
                )
            except AttributeError:
                deleted = self.memory.delete_memory(
                    target
                )

        except Exception as error:
            return self._result(
                False,
                "memory_delete",
                "Gagal menghapus memory.",
                error=str(error),
            )

        if deleted:
            return self._result(
                True,
                "memory_delete",
                "Baik. Memory tersebut sudah saya hapus.",
                deleted=True,
            )

        return self._result(
            True,
            "memory_delete",
            "Saya tidak menemukan memory tersebut.",
            deleted=False,
        )

    # =========================================================
    # MEMORY COUNT
    # =========================================================

    def handle_memory_count(
        self,
        text: str = "",
        **_: Any,
    ) -> dict[str, Any]:

        try:
            total = count_memories()

            return self._result(
                True,
                "memory_count",
                f"Saat ini saya memiliki {total} memory.",
                count=total,
            )

        except Exception as error:
            return self._result(
                False,
                "memory_count",
                "Gagal menghitung memory.",
                error=str(error),
            )

    # =========================================================
    # STATUS
    # =========================================================

    def handle_status(
        self,
        text: str = "",
        **_: Any,
    ) -> dict[str, Any]:

        return self._result(
            True,
            "status",
            "ZAI ONLINE. Memory aktif. Intent Engine aktif.",
            status="ONLINE",
            memory=True,
            intent_engine=True,
        )

    # =========================================================
    # HELP
    # =========================================================

    def handle_help(
        self,
        text: str = "",
        **_: Any,
    ) -> dict[str, Any]:

        return self._result(
            True,
            "help",
            (
                "Saya dapat membantu dengan:\n"
                "- Percakapan umum\n"
                "- Memory jangka panjang\n"
                "- Perhitungan\n"
                "- Coding\n"
                "- Pencarian\n"
                "- Status sistem\n"
                "- Informasi tentang ZAI"
            ),
            intents=self.available_intents(),
        )

    # =========================================================
    # SAFE CALCULATOR
    # =========================================================

    _OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    _UNARY_OPERATORS: dict[
        type[ast.unaryop],
        Callable[[Any], Any],
    ] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _safe_eval(
        self,
        node: ast.AST,
    ) -> float | int:

        if isinstance(node, ast.Expression):
            return self._safe_eval(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise ValueError("Boolean tidak diperbolehkan.")

            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Nilai tidak valid.")

        if isinstance(node, ast.BinOp):
            operation = self._OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError("Operator tidak didukung.")

            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)

            if isinstance(node.op, ast.Pow):
                if abs(right) > 100:
                    raise ValueError("Pangkat terlalu besar.")

            return operation(left, right)

        if isinstance(node, ast.UnaryOp):
            operation = self._UNARY_OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError("Operator unary tidak didukung.")

            return operation(
                self._safe_eval(node.operand)
            )

        raise ValueError(
            "Ekspresi hanya boleh berisi operasi matematika sederhana."
        )

    @staticmethod
    def _extract_expression(text: str) -> str:

        expression = text.strip()

        prefixes = (
            "hitung",
            "calculate",
            "calc",
            "berapa hasil",
        )

        lowered = expression.lower()

        for prefix in prefixes:
            if lowered.startswith(prefix):
                expression = expression[
                    len(prefix):
                ].strip()
                break

        return expression

    def handle_calculation(
        self,
        text: str = "",
        expression: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:

        raw_expression = (
            expression.strip()
            if expression
            else self._extract_expression(text)
        )

        if not raw_expression:
            return self._result(
                False,
                "calculation",
                "Ekspresi matematika kosong.",
            )

        if len(raw_expression) > 200:
            return self._result(
                False,
                "calculation",
                "Ekspresi terlalu panjang.",
            )

        try:
            tree = ast.parse(
                raw_expression,
                mode="eval",
            )

            value = self._safe_eval(tree)

            return self._result(
                True,
                "calculation",
                f"Hasilnya adalah {value}.",
                expression=raw_expression,
                result=value,
            )

        except Exception as error:
            return self._result(
                False,
                "calculation",
                "Ekspresi matematika tidak valid.",
                error=str(error),
            )

    # =========================================================
    # CODING
    # =========================================================

    def handle_coding(
        self,
        text: str = "",
        **_: Any,
    ) -> dict[str, Any]:

        return self._result(
            True,
            "coding",
            (
                "Mode coding aktif. "
                "Jelaskan bahasa pemrograman, fitur, "
                "atau kode yang ingin dibuat."
            ),
            requires_generation=True,
        )

    # =========================================================
    # SEARCH
    # =========================================================

    def handle_search(
        self,
        text: str = "",
        query: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:

        search_query = (
            query.strip()
            if query
            else text.strip()
        )

        return self._result(
            True,
            "search",
            (
                "Permintaan pencarian terdeteksi. "
                "Search engine akan menangani permintaan ini."
            ),
            query=search_query,
            requires_search=True,
        )

    # =========================================================
    # GENERAL
    # =========================================================

    def handle_general(
        self,
        text: str = "",
        **_: Any,
    ) -> dict[str, Any]:

        return self._result(
            True,
            "general",
            "",
            requires_generation=True,
        )


_default_handlers: Optional[IntentHandlers] = None


def get_intent_handlers() -> IntentHandlers:
    """
    Singleton handler engine.
    """

    global _default_handlers

    if _default_handlers is None:
        _default_handlers = IntentHandlers()

    return _default_handlers


__all__ = [
    "IntentHandlers",
    "get_intent_handlers",
]
