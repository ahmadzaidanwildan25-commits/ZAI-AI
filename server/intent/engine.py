"""
Super ZAI Intent Engine.

Version:
    0.11.0

Responsibilities:
    - classify user intent
    - extract entities
    - normalize user requests
    - prepare tool arguments
    - detect calculator expressions
    - detect weather locations
    - detect search queries
    - detect URLs for fetch
    - provide confidence scoring
    - remain dependency-light
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urlparse


ENGINE_VERSION = "0.11.0"


@dataclass
class IntentResult:
    """
    Structured result produced by IntentEngine.
    """

    intent: str
    confidence: float
    normalized_text: str
    entities: Dict[str, Any] = field(
        default_factory=dict
    )
    tool: Optional[str] = None
    arguments: Dict[str, Any] = field(
        default_factory=dict
    )
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "normalized_text": self.normalized_text,
            "entities": self.entities,
            "tool": self.tool,
            "arguments": self.arguments,
            "reasoning": self.reasoning,
        }


class IntentEngine:
    """
    ZAI's deterministic intent and entity engine.

    This layer intentionally does not call an LLM.

    The goal is to make tool calls predictable before
    the request reaches ToolEngine.
    """

    VERSION = ENGINE_VERSION

    INTENTS = (
        "empty",
        "greeting",
        "calculator",
        "weather",
        "search",
        "fetch",
        "memory_save",
        "memory_recall",
        "identity",
        "general",
    )

    CALCULATOR_WORDS = (
        "hitung",
        "kalkulasi",
        "kalkulator",
        "berapa hasil",
        "berapa",
        "jumlah",
        "ditambah",
        "dikurangi",
        "dikali",
        "dibagi",
        "persen",
        "akar",
        "pangkat",
    )

    WEATHER_WORDS = (
        "cuaca",
        "weather",
        "suhu",
        "temperatur",
        "hujan",
        "kelembapan",
        "kelembaban",
        "angin",
        "prakiraan",
        "ramalan cuaca",
    )

    SEARCH_WORDS = (
        "cari",
        "carikan",
        "search",
        "pencarian",
        "berita terbaru",
        "berita terkini",
        "informasi terbaru",
        "informasi terkini",
        "apa yang terbaru",
        "terbaru tentang",
        "terkini tentang",
    )

    FETCH_WORDS = (
        "buka",
        "ambil halaman",
        "fetch",
        "akses url",
        "akses website",
        "baca website",
        "baca halaman",
    )

    GREETINGS = {
        "halo",
        "hai",
        "hi",
        "hello",
        "hey",
        "pagi",
        "siang",
        "sore",
        "malam",
        "halo zai",
        "hai zai",
        "hello zai",
    }

    IDENTITY_PHRASES = (
        "siapa kamu",
        "kamu siapa",
        "siapa nama kamu",
        "apa nama kamu",
        "namamu siapa",
        "kamu adalah siapa",
    )

    MEMORY_SAVE_PATTERNS = (
        r"^nama saya\s+(.+)$",
        r"^nama ku\s+(.+)$",
        r"^namaku\s+(.+)$",
        r"^umur saya\s+(\d+)$",
        r"^usia saya\s+(\d+)$",
        r"^saya tinggal di\s+(.+)$",
        r"^alamat saya\s+(.+)$",
        r"^hobi saya\s+(.+)$",
    )

    MEMORY_RECALL_PHRASES = (
        "siapa nama saya",
        "apa nama saya",
        "namaku siapa",
        "berapa umur saya",
        "usia saya berapa",
        "saya tinggal dimana",
        "saya tinggal di mana",
        "alamat saya apa",
        "apa hobi saya",
        "hobi saya apa",
    )

    CITY_ALIASES = {
        "jakarta": "Jakarta",
        "jakarta pusat": "Jakarta",
        "jakarta selatan": "Jakarta",
        "jakarta barat": "Jakarta",
        "jakarta timur": "Jakarta",
        "jakarta utara": "Jakarta",
        "bogor": "Bogor",
        "depok": "Depok",
        "bekasi": "Bekasi",
        "tangerang": "Tangerang",
        "bandung": "Bandung",
        "surabaya": "Surabaya",
        "semarang": "Semarang",
        "yogyakarta": "Yogyakarta",
        "jogja": "Yogyakarta",
        "denpasar": "Denpasar",
        "bali": "Denpasar",
        "makassar": "Makassar",
        "medan": "Medan",
        "palembang": "Palembang",
        "malang": "Malang",
        "lombok": "Mataram",
        "mataram": "Mataram",
    }

    OPERATOR_REPLACEMENTS = (
        (r"\bditambah\b", "+"),
        (r"\bditambahkan\b", "+"),
        (r"\btambah\b", "+"),
        (r"\bplus\b", "+"),
        (r"\bdikurangi\b", "-"),
        (r"\bkurang\b", "-"),
        (r"\bminus\b", "-"),
        (r"\bdikali\b", "*"),
        (r"\bkalikan\b", "*"),
        (r"\bkali\b", "*"),
        (r"\bperkalian\b", "*"),
        (r"\b×\b", "*"),
        (r"\bx\b", "*"),
        (r"\bdibagi\b", "/"),
        (r"\bbagi\b", "/"),
        (r"\bpembagian\b", "/"),
        (r"\b÷\b", "/"),
        (r"\bpangkat\b", "**"),
    )

    def __init__(self) -> None:
        self._stats = {
            "analyze_calls": 0,
            "calculator_detected": 0,
            "weather_detected": 0,
            "search_detected": 0,
            "fetch_detected": 0,
            "memory_save_detected": 0,
            "memory_recall_detected": 0,
            "greeting_detected": 0,
            "identity_detected": 0,
            "general_detected": 0,
        }

    # =========================================================
    # PUBLIC API
    # =========================================================

    def analyze(
        self,
        message: str,
    ) -> Dict[str, Any]:
        """
        Analyze a user message and return a dictionary.

        This is the primary API expected by ZAI's router
        and orchestrator.
        """

        self._stats["analyze_calls"] += 1

        text = self._clean_text(message)

        if not text:
            result = IntentResult(
                intent="empty",
                confidence=1.0,
                normalized_text="",
                reasoning="Pesan kosong.",
            )
            return result.to_dict()

        intent = self._detect_intent(text)

        entities = self._extract_entities(
            text,
            intent,
        )

        normalized_text = self._normalize_request(
            text,
            intent,
            entities,
        )

        tool = self._tool_for_intent(intent)

        arguments = self._build_tool_arguments(
            text=text,
            intent=intent,
            entities=entities,
            normalized_text=normalized_text,
        )

        confidence = self._confidence(
            text,
            intent,
            entities,
        )

        reasoning = self._reasoning(
            intent,
            entities,
        )

        self._stats[
            f"{intent}_detected"
        ] = (
            self._stats.get(
                f"{intent}_detected",
                0,
            )
            + 1
        )

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            normalized_text=normalized_text,
            entities=entities,
            tool=tool,
            arguments=arguments,
            reasoning=reasoning,
        )

        return result.to_dict()

    def classify(
        self,
        message: str,
    ) -> str:
        """
        Return only the intent name.
        """

        return str(
            self.analyze(message).get(
                "intent",
                "general",
            )
        )

    def extract_entities(
        self,
        message: str,
    ) -> Dict[str, Any]:
        """
        Return extracted entities.
        """

        result = self.analyze(message)

        return dict(
            result.get(
                "entities",
                {},
            )
        )

    def get_tool_arguments(
        self,
        message: str,
    ) -> Dict[str, Any]:
        """
        Return sanitized arguments for ToolEngine.
        """

        result = self.analyze(message)

        return dict(
            result.get(
                "arguments",
                {},
            )
        )

    def stats(self) -> Dict[str, Any]:
        """
        Return engine diagnostics.
        """

        calls = self._stats[
            "analyze_calls"
        ]

        return {
            "engine": "IntentEngine",
            "version": self.VERSION,
            "intents": list(
                self.INTENTS
            ),
            "capabilities": {
                "classification": True,
                "entity_extraction": True,
                "calculator_normalization": True,
                "weather_city_extraction": True,
                "search_query_extraction": True,
                "url_extraction": True,
                "memory_detection": True,
                "greeting_detection": True,
                "identity_detection": True,
            },
            "runtime": {
                **self._stats,
                "success_rate_percent": (
                    100.0
                    if calls
                    else 0.0
                ),
            },
            "status": "READY",
        }

    # =========================================================
    # INTENT DETECTION
    # =========================================================

    def _detect_intent(
        self,
        text: str,
    ) -> str:

        lowered = text.lower().strip()

        if lowered in self.GREETINGS:
            return "greeting"

        if any(
            phrase in lowered
            for phrase in self.IDENTITY_PHRASES
        ):
            return "identity"

        if any(
            phrase in lowered
            for phrase in self.MEMORY_RECALL_PHRASES
        ):
            return "memory_recall"

        if self._is_memory_save(
            lowered
        ):
            return "memory_save"

        if self._looks_like_calculator(
            lowered
        ):
            return "calculator"

        if self._looks_like_weather(
            lowered
        ):
            return "weather"

        if self._looks_like_fetch(
            lowered
        ):
            return "fetch"

        if self._looks_like_search(
            lowered
        ):
            return "search"

        return "general"

    # =========================================================
    # ENTITY EXTRACTION
    # =========================================================

    def _extract_entities(
        self,
        text: str,
        intent: str,
    ) -> Dict[str, Any]:

        entities: Dict[str, Any] = {}

        if intent == "calculator":
            expression = self._extract_calculator_expression(
                text
            )

            if expression:
                entities[
                    "expression"
                ] = expression

        elif intent == "weather":
            city = self._extract_city(
                text
            )

            if city:
                entities[
                    "city"
                ] = city

        elif intent == "search":
            query = self._extract_search_query(
                text
            )

            if query:
                entities[
                    "query"
                ] = query

        elif intent == "fetch":
            url = self._extract_url(
                text
            )

            if url:
                entities[
                    "url"
                ] = url

        elif intent == "memory_save":
            memory = self._extract_memory(
                text
            )

            if memory:
                entities.update(
                    memory
                )

        return entities

    # =========================================================
    # CALCULATOR
    # =========================================================

    def _looks_like_calculator(
        self,
        text: str,
    ) -> bool:

        if any(
            word in text
            for word in self.CALCULATOR_WORDS
        ):
            return True

        if re.search(
            r"\d+\s*[\+\-\*\/\%\^\(\)]\s*\d+",
            text,
        ):
            return True

        return False

    def _extract_calculator_expression(
        self,
        text: str,
    ) -> Optional[str]:

        lowered = text.lower().strip()

        expression = lowered

        prefixes = (
            "hitung",
            "kalkulasi",
            "kalkulator",
            "berapa hasil",
            "berapa",
            "jumlah",
        )

        for prefix in prefixes:
            if expression.startswith(
                prefix
            ):
                expression = expression[
                    len(prefix):
                ].strip()
                break

        for pattern, replacement in (
            self.OPERATOR_REPLACEMENTS
        ):
            expression = re.sub(
                pattern,
                f" {replacement} ",
                expression,
                flags=re.IGNORECASE,
            )

        expression = expression.replace(
            "^",
            "**",
        )

        expression = re.sub(
            r"(?<=\d)\s*[xX]\s*(?=\d)",
            "*",
            expression,
        )

        expression = re.sub(
            r"[^0-9\+\-\*\/\%\(\)\.\s]",
            " ",
            expression,
        )

        expression = re.sub(
            r"\s+",
            " ",
            expression,
        ).strip()

        if not expression:
            return None

        if not re.search(
            r"\d",
            expression,
        ):
            return None

        if not re.search(
            r"[\+\-\*\/\%]",
            expression,
        ):
            return None

        return expression

    # =========================================================
    # WEATHER
    # =========================================================

    def _looks_like_weather(
        self,
        text: str,
    ) -> bool:

        return any(
            word in text
            for word in self.WEATHER_WORDS
        )

    def _extract_city(
        self,
        text: str,
    ) -> Optional[str]:

        lowered = text.lower()

        aliases = sorted(
            self.CITY_ALIASES.items(),
            key=lambda item: len(
                item[0]
            ),
            reverse=True,
        )

        for alias, canonical in aliases:
            if re.search(
                rf"\b{re.escape(alias)}\b",
                lowered,
            ):
                return canonical

        patterns = (
            r"\bdi\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,2})",
            r"\bdi\s+kota\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,2})",
            r"\bkota\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,2})",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                candidate = (
                    match.group(1)
                    .strip()
                    .rstrip("?!.")
                )

                if candidate:
                    return candidate.title()

        return None

    # =========================================================
    # SEARCH
    # =========================================================

    def _looks_like_search(
        self,
        text: str,
    ) -> bool:

        if any(
            word in text
            for word in self.SEARCH_WORDS
        ):
            return True

        return False

    def _extract_search_query(
        self,
        text: str,
    ) -> Optional[str]:

        query = text.strip()

        patterns = (
            r"^\s*cari(?:kan)?\s+(.+)$",
            r"^\s*search\s+(.+)$",
            r"^\s*pencarian\s+(.+)$",
            r"^\s*cari informasi\s+(.+)$",
            r"^\s*carikan informasi\s+(.+)$",
        )

        for pattern in patterns:
            match = re.match(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            if match:
                query = match.group(1)
                break

        query = re.sub(
            r"\s+",
            " ",
            query,
        ).strip()

        return query or None

    # =========================================================
    # FETCH
    # =========================================================

    def _looks_like_fetch(
        self,
        text: str,
    ) -> bool:

        return bool(
            self._extract_url(text)
        )

    def _extract_url(
        self,
        text: str,
    ) -> Optional[str]:

        match = re.search(
            r"https?://[^\s<>\[\]\(\)]+",
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        url = match.group(0).rstrip(
            ".,!?;:)]}"
        )

        parsed = urlparse(url)

        if parsed.scheme not in {
            "http",
            "https",
        }:
            return None

        if not parsed.netloc:
            return None

        return url

    # =========================================================
    # MEMORY
    # =========================================================

    def _is_memory_save(
        self,
        text: str,
    ) -> bool:

        return any(
            re.match(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern in self.MEMORY_SAVE_PATTERNS
        )

    def _extract_memory(
        self,
        text: str,
    ) -> Dict[str, Any]:

        lowered = text.lower().strip()

        patterns = (
            (
                r"^nama saya\s+(.+)$",
                "name",
            ),
            (
                r"^nama ku\s+(.+)$",
                "name",
            ),
            (
                r"^namaku\s+(.+)$",
                "name",
            ),
            (
                r"^umur saya\s+(\d+)$",
                "age",
            ),
            (
                r"^usia saya\s+(\d+)$",
                "age",
            ),
            (
                r"^saya tinggal di\s+(.+)$",
                "location",
            ),
            (
                r"^alamat saya\s+(.+)$",
                "address",
            ),
            (
                r"^hobi saya\s+(.+)$",
                "hobby",
            ),
        )

        for pattern, key in patterns:
            match = re.match(
                pattern,
                lowered,
                flags=re.IGNORECASE,
            )

            if match:
                value = (
                    match.group(1)
                    .strip()
                )

                return {
                    "memory_key": key,
                    "memory_value": value,
                }

        return {}

    # =========================================================
    # TOOL ARGUMENTS
    # =========================================================

    def _build_tool_arguments(
        self,
        text: str,
        intent: str,
        entities: Dict[str, Any],
        normalized_text: str,
    ) -> Dict[str, Any]:

        if intent == "calculator":
            expression = entities.get(
                "expression"
            )

            return {
                "expression": expression
                or normalized_text,
                "query": expression
                or normalized_text,
            }

        if intent == "weather":
            city = entities.get(
                "city"
            )

            return {
                "city": city,
                "query": city
                or normalized_text,
            }

        if intent == "search":
            query = entities.get(
                "query"
            )

            return {
                "query": query
                or normalized_text,
            }

        if intent == "fetch":
            url = entities.get(
                "url"
            )

            return {
                "url": url,
            }

        return {}

    # =========================================================
    # NORMALIZATION
    # =========================================================

    def _normalize_request(
        self,
        text: str,
        intent: str,
        entities: Dict[str, Any],
    ) -> str:

        if intent == "calculator":
            return str(
                entities.get(
                    "expression",
                    text,
                )
            )

        if intent == "weather":
            city = entities.get(
                "city"
            )

            if city:
                return (
                    f"Cuaca {city}"
                )

        if intent == "search":
            query = entities.get(
                "query"
            )

            if query:
                return query

        if intent == "fetch":
            url = entities.get(
                "url"
            )

            if url:
                return url

        return text.strip()

    # =========================================================
    # TOOL MAPPING
    # =========================================================

    @staticmethod
    def _tool_for_intent(
        intent: str,
    ) -> Optional[str]:

        mapping = {
            "calculator": "calculator",
            "weather": "weather",
            "search": "search",
            "fetch": "fetch",
        }

        return mapping.get(
            intent
        )

    # =========================================================
    # CONFIDENCE
    # =========================================================

    def _confidence(
        self,
        text: str,
        intent: str,
        entities: Dict[str, Any],
    ) -> float:

        if intent == "empty":
            return 1.0

        if intent == "greeting":
            return 0.99

        if intent == "identity":
            return 0.99

        if intent == "memory_recall":
            return 0.98

        if intent == "memory_save":
            return 0.99

        if intent == "calculator":
            if entities.get(
                "expression"
            ):
                return 0.98

            return 0.85

        if intent == "weather":
            if entities.get(
                "city"
            ):
                return 0.98

            return 0.82

        if intent == "search":
            if entities.get(
                "query"
            ):
                return 0.96

            return 0.80

        if intent == "fetch":
            if entities.get(
                "url"
            ):
                return 0.99

            return 0.70

        return 0.50

    # =========================================================
    # REASONING DESCRIPTION
    # =========================================================

    @staticmethod
    def _reasoning(
        intent: str,
        entities: Dict[str, Any],
    ) -> str:

        if intent == "calculator":
            return (
                "Permintaan dikenali sebagai "
                "perhitungan matematika."
            )

        if intent == "weather":
            city = entities.get(
                "city"
            )

            if city:
                return (
                    f"Permintaan cuaca terdeteksi "
                    f"untuk {city}."
                )

            return (
                "Permintaan cuaca terdeteksi "
                "tetapi lokasi belum pasti."
            )

        if intent == "search":
            return (
                "Permintaan membutuhkan "
                "pencarian informasi."
            )

        if intent == "fetch":
            return (
                "Permintaan berisi URL yang "
                "dapat diambil oleh fetch tool."
            )

        if intent == "memory_save":
            return (
                "Pengguna memberikan informasi "
                "yang berpotensi disimpan."
            )

        if intent == "memory_recall":
            return (
                "Pengguna meminta informasi "
                "yang kemungkinan berasal dari memory."
            )

        if intent == "greeting":
            return (
                "Pesan merupakan sapaan."
            )

        if intent == "identity":
            return (
                "Pengguna menanyakan identitas ZAI."
            )

        return (
            "Permintaan umum belum membutuhkan "
            "tool khusus."
        )

    # =========================================================
    # CLEANING
    # =========================================================

    @staticmethod
    def _clean_text(
        message: Any,
    ) -> str:

        if message is None:
            return ""

        text = str(
            message
        ).replace(
            "\x00",
            " ",
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text


_ENGINE: Optional[
    IntentEngine
] = None


def get_intent_engine() -> IntentEngine:
    """
    Return singleton IntentEngine.
    """

    global _ENGINE

    if _ENGINE is None:
        _ENGINE = IntentEngine()

    return _ENGINE


__all__ = [
    "IntentEngine",
    "IntentResult",
    "get_intent_engine",
]
