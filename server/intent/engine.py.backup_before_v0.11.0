"""
Super ZAI Intent Engine.

Responsible for:
- text normalization
- intent classification
- confidence scoring
- entity extraction
- tool requirement detection
- lightweight command understanding

This layer does not execute tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional


@dataclass
class IntentResult:
    """
    Structured result produced by IntentEngine.
    """

    intent: str
    confidence: float
    text: str
    entities: Dict[str, Any] = field(default_factory=dict)
    requires_tool: bool = False
    tool: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "text": self.text,
            "entities": dict(self.entities),
            "requires_tool": self.requires_tool,
            "tool": self.tool,
            "metadata": dict(self.metadata),
        }


class IntentEngine:
    """
    Deterministic intent classifier for the ZAI cognitive layer.

    Design goals:
    - fast
    - predictable
    - Indonesian-first
    - extensible
    - no tool execution
    - safe fallback
    """

    VERSION = "0.11.0"

    INTENTS = (
        "greeting",
        "identity",
        "help",
        "status",
        "calculation",
        "weather",
        "search",
        "fetch",
        "memory_save",
        "memory_recall",
        "memory_count",
        "memory_forget",
        "coding",
        "conversation",
        "general",
    )

    def __init__(self) -> None:
        self._last_result: Optional[IntentResult] = None

    # ============================================================
    # PUBLIC API
    # ============================================================

    def analyze(
        self,
        message: str,
        conversation: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IntentResult:

        text = self.normalize(message)

        if not text:
            result = IntentResult(
                intent="general",
                confidence=0.0,
                text="",
                metadata={
                    "empty": True,
                },
            )
            self._last_result = result
            return result

        entities = self.extract_entities(text)

        intent, confidence, tool = self._classify(
            text,
            entities,
        )

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            text=text,
            entities=entities,
            requires_tool=tool is not None,
            tool=tool,
            metadata={
                "engine": "IntentEngine",
                "version": self.VERSION,
                "conversation_available": bool(conversation),
                "metadata_available": bool(metadata),
            },
        )

        self._last_result = result
        return result

    def classify(
        self,
        message: str,
        conversation: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        return self.analyze(
            message=message,
            conversation=conversation,
            metadata=metadata,
        ).to_dict()

    def detect(
        self,
        message: str,
    ) -> IntentResult:

        return self.analyze(message)

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def normalize(message: Any) -> str:

        if message is None:
            return ""

        text = str(message).strip().lower()

        text = (
            text
            .replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
        )

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ============================================================
    # ENTITY EXTRACTION
    # ============================================================

    def extract_entities(
        self,
        text: str,
    ) -> Dict[str, Any]:

        entities: Dict[str, Any] = {}

        # --------------------------------------------------------
        # CALCULATION EXPRESSION
        # --------------------------------------------------------

        expression = self._extract_expression(text)

        if expression:
            entities["expression"] = expression

        # --------------------------------------------------------
        # CITY
        # --------------------------------------------------------

        city = self._extract_city(text)

        if city:
            entities["city"] = city

        # --------------------------------------------------------
        # URL
        # --------------------------------------------------------

        urls = re.findall(
            r"https?://[^\s]+",
            text,
        )

        if urls:
            entities["url"] = urls[0].rstrip(".,!?")

        # --------------------------------------------------------
        # SEARCH QUERY
        # --------------------------------------------------------

        search_query = self._extract_search_query(text)

        if search_query:
            entities["query"] = search_query

        # --------------------------------------------------------
        # MEMORY KEY / VALUE
        # --------------------------------------------------------

        memory = self._extract_memory(text)

        if memory:
            entities.update(memory)

        return entities

    # ============================================================
    # CLASSIFICATION
    # ============================================================

    def _classify(
        self,
        text: str,
        entities: Dict[str, Any],
    ):

        # --------------------------------------------------------
        # GREETING
        # --------------------------------------------------------

        greeting_patterns = (
            "halo",
            "hai",
            "hi",
            "hello",
            "hey",
            "selamat pagi",
            "selamat siang",
            "selamat sore",
            "selamat malam",
        )

        if self._starts_with_any(text, greeting_patterns):
            return "greeting", 0.98, None

        # --------------------------------------------------------
        # IDENTITY
        # --------------------------------------------------------

        identity_patterns = (
            "siapa kamu",
            "kamu siapa",
            "apa kamu",
            "siapa nama kamu",
            "nama kamu siapa",
            "jelaskan tentang dirimu",
        )

        if self._contains_any(text, identity_patterns):
            return "identity", 0.98, None

        # --------------------------------------------------------
        # HELP
        # --------------------------------------------------------

        help_patterns = (
            "help",
            "bantuan",
            "apa yang bisa kamu lakukan",
            "kamu bisa apa",
            "fitur kamu",
            "cara menggunakan kamu",
        )

        if self._contains_any(text, help_patterns):
            return "help", 0.97, None

        # --------------------------------------------------------
        # STATUS
        # --------------------------------------------------------

        status_patterns = (
            "status zai",
            "status kamu",
            "apakah kamu online",
            "kamu online",
            "cek status",
            "status sistem",
        )

        if self._contains_any(text, status_patterns):
            return "status", 0.97, None

        # --------------------------------------------------------
        # MEMORY SAVE
        # --------------------------------------------------------

        if self._is_memory_save(text):
            return "memory_save", 0.97, None

        # --------------------------------------------------------
        # MEMORY COUNT
        # --------------------------------------------------------

        memory_count_patterns = (
            "berapa memory",
            "berapa memori",
            "jumlah memory",
            "jumlah memori",
            "memory saya ada berapa",
            "memori saya ada berapa",
        )

        if self._contains_any(
            text,
            memory_count_patterns,
        ):
            return "memory_count", 0.97, None

        # --------------------------------------------------------
        # MEMORY FORGET
        # --------------------------------------------------------

        if self._is_memory_forget(text):
            return "memory_forget", 0.96, None

        # --------------------------------------------------------
        # MEMORY RECALL
        # --------------------------------------------------------

        if self._is_memory_recall(text):
            return "memory_recall", 0.96, None

        # --------------------------------------------------------
        # CALCULATION
        # --------------------------------------------------------

        if self._is_calculation(text, entities):
            return "calculation", 0.95, "calculator"

        # --------------------------------------------------------
        # WEATHER
        # --------------------------------------------------------

        if self._is_weather(text):
            return "weather", 0.95, "weather"

        # --------------------------------------------------------
        # FETCH
        # --------------------------------------------------------

        if (
            "url" in entities
            and self._contains_any(
                text,
                (
                    "buka",
                    "ambil",
                    "fetch",
                    "akses",
                    "baca halaman",
                    "ambil halaman",
                ),
            )
        ):
            return "fetch", 0.94, "fetch"

        # --------------------------------------------------------
        # SEARCH
        # --------------------------------------------------------

        if self._is_search(text):
            return "search", 0.94, "search"

        # --------------------------------------------------------
        # CODING
        # --------------------------------------------------------

        coding_patterns = (
            "coding",
            "kode",
            "program",
            "programming",
            "python",
            "dart",
            "flutter",
            "fastapi",
            "javascript",
            "typescript",
            "debug",
            "error kode",
            "buatkan kode",
        )

        if self._contains_any(
            text,
            coding_patterns,
        ):
            return "coding", 0.90, None

        # --------------------------------------------------------
        # CONVERSATION
        # --------------------------------------------------------

        conversation_patterns = (
            "apa kabar",
            "terima kasih",
            "makasih",
            "oke",
            "ok",
            "baik",
            "mantap",
            "sip",
            "iya",
            "ya",
        )

        if self._contains_any(
            text,
            conversation_patterns,
        ):
            return "conversation", 0.88, None

        # --------------------------------------------------------
        # FALLBACK
        # --------------------------------------------------------

        return "general", 0.50, None

    # ============================================================
    # CALCULATION DETECTION
    # ============================================================

    def _is_calculation(
        self,
        text: str,
        entities: Dict[str, Any],
    ) -> bool:

        if "expression" in entities:
            return True

        keywords = (
            "hitung",
            "berapa hasil",
            "jumlahkan",
            "kurangkan",
            "kalikan",
            "bagikan",
            "perkalian",
            "pembagian",
            "persentase",
        )

        if self._contains_any(text, keywords):
            return bool(
                re.search(
                    r"\d",
                    text,
                )
            )

        return bool(
            re.fullmatch(
                r"[\d\s\+\-\*\/\(\)\.\,%\^]+",
                text,
            )
        )

    def _extract_expression(
        self,
        text: str,
    ) -> Optional[str]:

        working = text

        replacements = {
            " dikali ": " * ",
            " kali ": " * ",
            " x ": " * ",
            " dibagi ": " / ",
            " bagi ": " / ",
            " ditambah ": " + ",
            " tambah ": " + ",
            " dikurangi ": " - ",
            " kurang ": " - ",
        }

        for source, target in replacements.items():
            working = working.replace(
                source,
                target,
            )

        working = re.sub(
            r"^(tolong\s+)?hitung\s*",
            "",
            working,
        )

        working = re.sub(
            r"^(berapa hasil|hasil dari)\s*",
            "",
            working,
        )

        working = working.strip()

        if not re.search(
            r"\d",
            working,
        ):
            return None

        if not re.fullmatch(
            r"[\d\s\+\-\*\/\(\)\.\,%\^]+",
            working,
        ):
            return None

        return working.strip()

    # ============================================================
    # WEATHER
    # ============================================================

    def _is_weather(
        self,
        text: str,
    ) -> bool:

        patterns = (
            "cuaca",
            "weather",
            "suhu",
            "temperatur",
            "hujan",
            "kelembapan",
            "kelembaban",
            "angin di",
        )

        return self._contains_any(
            text,
            patterns,
        )

    def _extract_city(
        self,
        text: str,
    ) -> Optional[str]:

        patterns = (
            r"\bdi\s+([a-zA-Z][a-zA-Z\s\-]{1,40}?)(?:\s+sekarang|\s+saat ini|\s+hari ini|\?|$)",
            r"\bkota\s+([a-zA-Z][a-zA-Z\s\-]{1,40}?)(?:\s+sekarang|\s+saat ini|\s+hari ini|\?|$)",
            r"\bcuaca\s+([a-zA-Z][a-zA-Z\s\-]{1,40}?)(?:\s+sekarang|\s+saat ini|\s+hari ini|\?|$)",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                re.IGNORECASE,
            )

            if match:
                city = match.group(1).strip()
                city = re.sub(
                    r"\s+",
                    " ",
                    city,
                )

                if city:
                    return city.title()

        return None

    # ============================================================
    # SEARCH
    # ============================================================

    def _is_search(
        self,
        text: str,
    ) -> bool:

        patterns = (
            "cari ",
            "carikan ",
            "search ",
            "pencarian ",
            "temukan ",
            "berita terbaru",
            "informasi terbaru",
        )

        return self._contains_any(
            text,
            patterns,
        )

    def _extract_search_query(
        self,
        text: str,
    ) -> Optional[str]:

        prefixes = (
            "cari ",
            "carikan ",
            "search ",
            "pencarian ",
            "temukan ",
        )

        for prefix in prefixes:
            if text.startswith(prefix):
                value = text[len(prefix):].strip()

                if value:
                    return value

        return None

    # ============================================================
    # MEMORY
    # ============================================================

    def _is_memory_save(
        self,
        text: str,
    ) -> bool:

        patterns = (
            r"^ingat(?:lah)?\s+",
            r"^tolong ingat\s+",
            r"^simpan\s+",
            r"^nama saya\s+",
            r"^nama ku\s+",
            r"^namaku\s+",
            r"^umur saya\s+",
            r"^usia saya\s+",
            r"^saya tinggal di\s+",
            r"^alamat saya\s+",
            r"^hobi saya\s+",
        )

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            for pattern in patterns
        )

    def _is_memory_recall(
        self,
        text: str,
    ) -> bool:

        patterns = (
            "siapa nama saya",
            "nama saya siapa",
            "apa nama saya",
            "umur saya berapa",
            "usia saya berapa",
            "saya tinggal dimana",
            "saya tinggal di mana",
            "apa yang kamu ingat tentang saya",
            "apa yang kamu ingat",
            "ingat apa tentang saya",
        )

        return self._contains_any(
            text,
            patterns,
        )

    def _is_memory_forget(
        self,
        text: str,
    ) -> bool:

        patterns = (
            "lupakan ",
            "lupakanlah ",
            "hapus memory ",
            "hapus memori ",
            "jangan ingat ",
            "forget ",
        )

        return self._contains_any(
            text,
            patterns,
        )

    def _extract_memory(
        self,
        text: str,
    ) -> Dict[str, Any]:

        result: Dict[str, Any] = {}

        patterns = (
            (
                r"^nama saya\s+(.+)$",
                "nama",
            ),
            (
                r"^nama ku\s+(.+)$",
                "nama",
            ),
            (
                r"^namaku\s+(.+)$",
                "nama",
            ),
            (
                r"^umur saya\s+(\d+)$",
                "umur",
            ),
            (
                r"^usia saya\s+(\d+)$",
                "umur",
            ),
            (
                r"^saya tinggal di\s+(.+)$",
                "alamat",
            ),
            (
                r"^alamat saya\s+(.+)$",
                "alamat",
            ),
            (
                r"^hobi saya\s+(.+)$",
                "hobi",
            ),
        )

        for pattern, key in patterns:
            match = re.match(
                pattern,
                text,
                re.IGNORECASE,
            )

            if match:
                result["memory_key"] = key
                result["memory_value"] = match.group(1).strip()
                break

        if text.startswith("ingat "):
            value = text[6:].strip()

            if value:
                result["memory_text"] = value

        return result

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _contains_any(
        text: str,
        patterns,
    ) -> bool:

        return any(
            pattern in text
            for pattern in patterns
        )

    @staticmethod
    def _starts_with_any(
        text: str,
        patterns,
    ) -> bool:

        return any(
            text == pattern
            or text.startswith(pattern + " ")
            for pattern in patterns
        )

    # ============================================================
    # STATS
    # ============================================================

    def stats(self) -> Dict[str, Any]:

        return {
            "engine": "IntentEngine",
            "version": self.VERSION,
            "intents": list(self.INTENTS),
            "intent_count": len(self.INTENTS),
            "normalization": True,
            "entity_extraction": True,
            "calculation_detection": True,
            "weather_detection": True,
            "search_detection": True,
            "memory_detection": True,
            "status": "READY",
        }


_ENGINE: Optional[IntentEngine] = None


def get_intent_engine() -> IntentEngine:

    global _ENGINE

    if _ENGINE is None:
        _ENGINE = IntentEngine()

    return _ENGINE


__all__ = [
    "IntentEngine",
    "IntentResult",
    "get_intent_engine",
]
