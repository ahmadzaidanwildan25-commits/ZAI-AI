from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ToolResult:
    success: bool
    tool: str
    response: Optional[str] = None
    data: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "tool": self.tool,
            "response": self.response,
            "data": self.data,
            "error": self.error,
        }


class ToolEngine:
    """
    Super ZAI Tool Execution Layer.

    v0.9.0

    Menjalankan tool berdasarkan route dari Cognitive Pipeline.
    """

    VERSION = "0.9.0"

    def __init__(self):
        self._tools = {
            "calculator": self.handle_calculator,
            "weather": self.handle_weather,
            "search": self.handle_search,
        }

    # ======================================================
    # TOOL REGISTRY
    # ======================================================

    def tools(self) -> list[str]:
        return list(self._tools.keys())

    def has_tool(self, route: str) -> bool:
        return route in self._tools

    # ======================================================
    # MAIN EXECUTOR
    # ======================================================

    def execute(
        self,
        route: str,
        message: str,
        metadata: Optional[dict] = None,
    ) -> ToolResult:

        route = str(route or "").strip().lower()

        handler = self._tools.get(route)

        if handler is None:
            return ToolResult(
                success=False,
                tool=route,
                error=f"Tool '{route}' tidak tersedia.",
            )

        try:
            return handler(message, metadata or {})

        except Exception as error:
            return ToolResult(
                success=False,
                tool=route,
                error=str(error),
            )

    # ======================================================
    # CALCULATOR
    # ======================================================

    def handle_calculator(
        self,
        message: str,
        metadata: dict,
    ) -> ToolResult:

        expression = self._extract_expression(message)

        if not expression:
            return ToolResult(
                success=False,
                tool="calculator",
                error="Ekspresi matematika tidak ditemukan.",
            )

        try:
            result = self._safe_eval(expression)

            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 10)

            return ToolResult(
                success=True,
                tool="calculator",
                response=f"Hasilnya adalah {result}.",
                data={
                    "expression": expression,
                    "result": result,
                },
            )

        except Exception as error:
            return ToolResult(
                success=False,
                tool="calculator",
                error=f"Perhitungan gagal: {error}",
            )

    # ======================================================
    # SAFE CALCULATOR
    # ======================================================

    def _extract_expression(self, message: str) -> str:

        text = str(message or "").strip()

        patterns = [
            r"^(?:hitung|calculate|calculator)\s+(.+)$",
            r"^(?:berapa hasil dari)\s+(.+)$",
            r"^(?:berapa)\s+([0-9\.\+\-\*\/\(\)\%\^\s]+)$",
        ]

        for pattern in patterns:
            match = re.match(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return match.group(1).strip()

        if re.fullmatch(
            r"[0-9\.\+\-\*\/\(\)\%\^\s]+",
            text,
        ):
            return text

        return ""

    def _safe_eval(self, expression: str) -> Any:

        expression = expression.replace("^", "**")

        if len(expression) > 200:
            raise ValueError("Ekspresi terlalu panjang.")

        tree = ast.parse(
            expression,
            mode="eval",
        )

        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def evaluate(node):

            if isinstance(node, ast.Expression):
                return evaluate(node.body)

            if isinstance(node, ast.Constant):

                if isinstance(node.value, (int, float)):
                    return node.value

                raise ValueError(
                    "Hanya angka yang diperbolehkan."
                )

            if isinstance(node, ast.UnaryOp):

                operation = operators.get(type(node.op))

                if operation is None:
                    raise ValueError(
                        "Operator tidak diperbolehkan."
                    )

                return operation(
                    evaluate(node.operand)
                )

            if isinstance(node, ast.BinOp):

                operation = operators.get(type(node.op))

                if operation is None:
                    raise ValueError(
                        "Operator tidak diperbolehkan."
                    )

                left = evaluate(node.left)
                right = evaluate(node.right)

                if isinstance(node.op, ast.Pow):
                    if abs(right) > 100:
                        raise ValueError(
                            "Pangkat terlalu besar."
                        )

                return operation(left, right)

            raise ValueError(
                "Ekspresi tidak aman."
            )

        result = evaluate(tree)

        if isinstance(result, (int, float)):

            if abs(result) > 10**100:
                raise ValueError(
                    "Hasil terlalu besar."
                )

        return result

    # ======================================================
    # WEATHER
    # ======================================================

    def handle_weather(
        self,
        message: str,
        metadata: dict,
    ) -> ToolResult:

        return ToolResult(
            success=False,
            tool="weather",
            response=None,
            data={
                "requires_location": True,
                "message": message,
            },
            error=(
                "Weather executor belum terhubung "
                "ke weather provider."
            ),
        )

    # ======================================================
    # SEARCH
    # ======================================================

    def handle_search(
        self,
        message: str,
        metadata: dict,
    ) -> ToolResult:

        return ToolResult(
            success=False,
            tool="search",
            response=None,
            data={
                "query": message,
            },
            error=(
                "Search executor belum terhubung "
                "ke search provider."
            ),
        )

    # ======================================================
    # STATUS
    # ======================================================

    def stats(self) -> dict:

        return {
            "engine": "ToolEngine",
            "version": self.VERSION,
            "tools": self.tools(),
            "calculator": True,
            "weather": True,
            "search": True,
            "safe_calculator": True,
            "status": "READY",
        }


_tool_engine: Optional[ToolEngine] = None


def get_tool_engine() -> ToolEngine:

    global _tool_engine

    if _tool_engine is None:
        _tool_engine = ToolEngine()

    return _tool_engine
