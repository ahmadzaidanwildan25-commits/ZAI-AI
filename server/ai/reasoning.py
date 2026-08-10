"""
Super ZAI Reasoning Engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ReasoningEngine:
    """
    Reasoning layer.

    Current version:
    - analyzes tool results
    - determines whether execution succeeded
    - prepares structured observations

    Future versions:
    - LLM reasoning
    - multi-step reasoning
    - contradiction detection
    - confidence scoring
    """

    def __init__(
        self,
        llm_client: Any = None,
    ) -> None:
        self.llm_client = llm_client

    def analyze_tool_result(
        self,
        tool: str,
        result: Any,
    ) -> Dict[str, Any]:

        if hasattr(result, "to_dict"):
            payload = result.to_dict()
        elif isinstance(result, dict):
            payload = result
        else:
            payload = {
                "success": True,
                "response": str(result),
            }

        success = bool(
            payload.get("success", False)
        )

        response = str(
            payload.get("response", "")
        )

        error = payload.get("error")

        return {
            "tool": tool,
            "success": success,
            "response": response,
            "data": payload.get("data", {}),
            "error": error,
            "usable": success and bool(response.strip()),
        }

    def should_continue(
        self,
        observations: List[Dict[str, Any]],
        current_step: int,
        max_steps: int,
    ) -> bool:

        if current_step >= max_steps:
            return False

        if not observations:
            return True

        latest = observations[-1]

        if not latest.get("success", False):
            return False

        return False

    def build_reasoning_summary(
        self,
        context: Any,
    ) -> str:

        observations = getattr(
            context,
            "observations",
            [],
        )

        if not observations:
            return ""

        successful = [
            item
            for item in observations
            if item.get("success")
        ]

        if not successful:
            return "Tool tidak menghasilkan hasil yang dapat digunakan."

        return "\n".join(
            item.get("response", "")
            for item in successful
            if item.get("response")
        )