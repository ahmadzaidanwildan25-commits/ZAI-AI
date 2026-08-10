"""
Super ZAI Planner.

Converts user intent into an executable plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PlanStep:
    step_id: int
    action: str
    tool: Optional[str] = None
    arguments: Dict[str, Any] = None
    description: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        if self.arguments is None:
            self.arguments = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "tool": self.tool,
            "arguments": self.arguments,
            "description": self.description,
            "required": self.required,
        }


class Planner:
    """
    Deterministic first-stage planner.

    Later this layer can be upgraded to LLM-assisted planning.
    """

    TOOL_KEYWORDS = {
        "calculator": (
            "hitung",
            "kalkulator",
            "berapa",
            "jumlah",
            "persen",
            "perkalian",
            "pembagian",
            "tambah",
            "kurang",
        ),
        "weather": (
            "cuaca",
            "suhu",
            "hujan",
            "panas",
            "dingin",
            "kelembapan",
        ),
        "search": (
            "cari",
            "search",
            "berita",
            "terbaru",
            "informasi",
            "riset",
            "penelitian",
        ),
        "fetch": (
            "buka",
            "baca",
            "ambil halaman",
            "ambil website",
            "fetch",
            "url",
            "link",
        ),
    }

    def __init__(
        self,
        max_steps: int = 8,
    ) -> None:
        self.max_steps = max(1, max_steps)

    def _detect_tool(
        self,
        text: str,
    ) -> Optional[str]:

        normalized = text.lower()

        scores: Dict[str, int] = {}

        for tool, keywords in self.TOOL_KEYWORDS.items():
            score = sum(
                1
                for keyword in keywords
                if keyword in normalized
            )

            if score:
                scores[tool] = score

        if not scores:
            return None

        return max(
            scores,
            key=scores.get,
        )

    def create_plan(
        self,
        user_message: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:

        text = user_message.strip()

        if not text:
            return []

        tool = self._detect_tool(text)

        steps: List[PlanStep] = []

        if tool:
            steps.append(
                PlanStep(
                    step_id=1,
                    action="execute_tool",
                    tool=tool,
                    arguments={
                        "query": text,
                    },
                    description=(
                        f"Gunakan tool {tool} "
                        "untuk memenuhi permintaan pengguna."
                    ),
                )
            )

            steps.append(
                PlanStep(
                    step_id=2,
                    action="analyze_result",
                    description=(
                        "Analisis hasil tool sebelum memberikan jawaban."
                    ),
                )
            )

            steps.append(
                PlanStep(
                    step_id=3,
                    action="generate_response",
                    description=(
                        "Susun jawaban akhir berdasarkan hasil "
                        "dan konteks percakapan."
                    ),
                )
            )

        else:
            steps.append(
                PlanStep(
                    step_id=1,
                    action="reason",
                    description=(
                        "Analisis permintaan menggunakan AI Brain."
                    ),
                )
            )

            steps.append(
                PlanStep(
                    step_id=2,
                    action="generate_response",
                    description=(
                        "Buat jawaban yang relevan dan jelas."
                    ),
                )
            )

        return [
            step.to_dict()
            for step in steps[: self.max_steps]
        ]