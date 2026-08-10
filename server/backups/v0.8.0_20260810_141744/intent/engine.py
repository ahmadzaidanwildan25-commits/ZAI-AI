from __future__ import annotations

import re
from typing import Dict, Any, Optional


class IntentEngine:
    """
    ZAI Intent Engine
    -----------------
    Mengubah pesan user menjadi intent terstruktur.

    Fokus versi ini:
    - greeting
    - identity
    - memory
    - status
    - help
    - calculation
    - weather
    - coding
    - search
    - general
    """

    VERSION = "0.8.0"

    def __init__(self):
        self.patterns = [
            (
                "greeting",
                [
                    r"^halo zai$",
                    r"^halo$",
                    r"^hai$",
                    r"^hi$",
                    r"^hello$",
                    r"^hey$",
                ],
                0.95,
            ),

            (
                "identity",
                [
                    r"^siapa kamu\??$",
                    r"^kamu siapa\??$",
                    r"^siapa zai\??$",
                    r"^apa itu zai\??$",
                ],
                0.95,
            ),

            (
                "memory_save",
                [
                    r"^ingat(?: bahwa)?\s+.+$",
                    r"^simpan(?: bahwa)?\s+.+$",
                    r"^tolong ingat\s+.+$",
                ],
                0.95,
            ),

            (
                "memory_query",
                [
                    r"^apa yang kamu ingat.*$",
                    r"^apa yang kamu tahu tentang saya.*$",
                    r"^memory saya.*$",
                    r"^ingatanku.*$",
                    r"^apa saja yang kamu ingat.*$",
                ],
                0.95,
            ),

            (
                "memory_delete",
                [
                    r"^hapus memory.*$",
                    r"^hapus memori.*$",
                    r"^lupakan memory.*$",
                    r"^lupakan memori.*$",
                ],
                0.95,
            ),

            (
                "memory_count",
                [
                    r"^berapa memory\??$",
                    r"^berapa memori\??$",
                    r"^jumlah memory\??$",
                    r"^jumlah memori\??$",
                ],
                0.95,
            ),

            (
                "status",
                [
                    r"^status zai\??$",
                    r"^status sistem\??$",
                    r"^cek status zai\??$",
                ],
                0.95,
            ),

            (
                "help",
                [
                    r"^help$",
                    r"^bantuan$",
                    r"^tolong$",
                    r"^apa yang bisa kamu lakukan\??$",
                ],
                0.95,
            ),

            (
                "weather",
                [
                    r".*\bcuaca\b.*",
                    r".*\bsuhu\b.*",
                    r".*\bhujan\b.*",
                    r".*\bpanas\b.*",
                    r".*\bdingin\b.*",
                    r".*\bkelembapan\b.*",
                    r".*\bkelembaban\b.*",
                    r".*\bangin\b.*",
                    r".*\bcerah\b.*",
                    r".*\bmendung\b.*",
                    r".*\bprakiraan\b.*",
                    r".*\bforecast\b.*",
                ],
                0.95,
            ),

            (
                "calculation",
                [
                    r"^hitung\b.*$",
                    r"^kalkulasi\b.*$",
                    r"^berapa hasil\b.*$",
                    r"^calculate\b.*$",
                ],
                0.85,
            ),

            (
                "coding",
                [
                    r".*\bbuatkan kode\b.*",
                    r".*\bbuat kode\b.*",
                    r".*\bprogram\b.*",
                    r".*\bpython\b.*",
                    r".*\bdart\b.*",
                    r".*\bflutter\b.*",
                    r".*\bjavascript\b.*",
                    r".*\btypescript\b.*",
                    r".*\bfastapi\b.*",
                    r".*\bdebug\b.*",
                    r".*\bcoding\b.*",
                ],
                0.85,
            ),

            (
                "search",
                [
                    r".*\bcari\b.*",
                    r".*\bpencarian\b.*",
                    r".*\bberita terbaru\b.*",
                    r".*\binformasi terbaru\b.*",
                    r".*\bresearch\b.*",
                    r".*\briset\b.*",
                ],
                0.85,
            ),
        ]

    def normalize(self, text: str) -> str:
        text = str(text or "")
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def analyze(self, text: str) -> Dict[str, Any]:
        normalized = self.normalize(text)

        if not normalized:
            return {
                "intent": "general",
                "confidence": 0.0,
                "normalized_text": "",
                "matched_pattern": None,
                "high_confidence": False,
            }

        for intent, patterns, confidence in self.patterns:
            for pattern in patterns:
                if re.fullmatch(pattern, normalized):
                    return {
                        "intent": intent,
                        "confidence": confidence,
                        "normalized_text": normalized,
                        "matched_pattern": pattern,
                        "high_confidence": confidence >= 0.90,
                    }

        if any(
            word in normalized
            for word in [
                "buatkan kode",
                "buat kode",
                "python",
                "flutter",
                "dart",
                "javascript",
                "program",
                "debug",
                "coding",
            ]
        ):
            return {
                "intent": "coding",
                "confidence": 0.85,
                "normalized_text": normalized,
                "matched_pattern": None,
                "high_confidence": False,
            }

        if any(
            word in normalized
            for word in [
                "cari",
                "berita",
                "informasi terbaru",
                "riset",
                "research",
            ]
        ):
            return {
                "intent": "search",
                "confidence": 0.85,
                "normalized_text": normalized,
                "matched_pattern": None,
                "high_confidence": False,
            }

        return {
            "intent": "general",
            "confidence": 0.50,
            "normalized_text": normalized,
            "matched_pattern": None,
            "high_confidence": False,
        }

    def get_intent(self, text: str) -> str:
        return self.analyze(text)["intent"]


_engine: Optional[IntentEngine] = None


def get_intent_engine() -> IntentEngine:
    global _engine

    if _engine is None:
        _engine = IntentEngine()

    return _engine
