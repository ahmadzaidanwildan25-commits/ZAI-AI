# ============================================================
# ZAI INTENT ENGINE
# SUPER ZAI
# VERSION 0.6.0
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# INTENT RESULT
# ============================================================

@dataclass
class IntentResult:
    """
    Hasil deteksi intent ZAI.
    """

    intent: str
    confidence: float
    mode: str = "normal"
    entities: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "mode": self.mode,
            "entities": self.entities,
        }


# ============================================================
# INTENT ENGINE
# ============================================================

class IntentEngine:
    """
    Intent & Command Engine untuk Super ZAI.

    Fungsi utama:

    - greeting
    - identity
    - memory_save
    - memory_search
    - memory_delete
    - memory_count
    - help
    - conversation
    """

    VERSION = "0.6.0"

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    GREETING_PATTERNS = [
        r"^halo$",
        r"^halo zai$",
        r"^hai$",
        r"^hai zai$",
        r"^hi$",
        r"^hi zai$",
        r"^hello$",
        r"^hello zai$",
        r"^hey$",
        r"^hey zai$",
        r"^selamat pagi$",
        r"^selamat siang$",
        r"^selamat sore$",
        r"^selamat malam$",
    ]

    # --------------------------------------------------------
    # IDENTITY
    # --------------------------------------------------------

    IDENTITY_PATTERNS = [
        r"^siapa kamu$",
        r"^kamu siapa$",
        r"^siapa dirimu$",
        r"^kamu itu siapa$",
        r"^apa kamu$",
        r"^jelaskan siapa kamu$",
        r"^perkenalkan dirimu$",
    ]

    # --------------------------------------------------------
    # MEMORY COUNT
    # --------------------------------------------------------

    MEMORY_COUNT_PATTERNS = [
        r"^berapa memory$",
        r"^berapa memori$",
        r"^berapa banyak memory$",
        r"^berapa banyak memori$",
        r"^jumlah memory$",
        r"^jumlah memori$",
        r"^cek memory$",
        r"^cek memori$",
        r"^hitung memory$",
        r"^hitung memori$",
    ]

    # --------------------------------------------------------
    # MEMORY SEARCH
    # --------------------------------------------------------

    MEMORY_SEARCH_PATTERNS = [
        r"^apa yang kamu ingat tentang (.+)$",
        r"^apa yang kamu ingat mengenai (.+)$",
        r"^apa yang kamu tahu tentang (.+)$",
        r"^cari memory tentang (.+)$",
        r"^cari memori tentang (.+)$",
        r"^cari ingatan tentang (.+)$",
        r"^ingatanku tentang (.+)$",
        r"^memory tentang (.+)$",
        r"^memori tentang (.+)$",
    ]

    # --------------------------------------------------------
    # MEMORY DELETE
    # --------------------------------------------------------

    MEMORY_DELETE_PATTERNS = [
        r"^hapus memory (.+)$",
        r"^hapus memori (.+)$",
        r"^hapus memory tentang (.+)$",
        r"^hapus memori tentang (.+)$",
        r"^lupakan memory (.+)$",
        r"^lupakan memori (.+)$",
        r"^lupakan (.+)$",
    ]

    # --------------------------------------------------------
    # MEMORY SAVE
    # --------------------------------------------------------

    MEMORY_SAVE_PATTERNS = [
        r"^ingat (.+)$",
        r"^ingat bahwa (.+)$",
        r"^ingat ya (.+)$",
        r"^tolong ingat (.+)$",
        r"^tolong ingat bahwa (.+)$",
        r"^simpan (.+)$",
        r"^simpan bahwa (.+)$",
        r"^catat (.+)$",
        r"^catat bahwa (.+)$",
    ]

    # --------------------------------------------------------
    # HELP
    # --------------------------------------------------------

    HELP_PATTERNS = [
        r"^help$",
        r"^bantuan$",
        r"^tolong$",
        r"^apa yang bisa kamu lakukan$",
        r"^kamu bisa apa$",
        r"^fitur kamu apa saja$",
        r"^fitur zai$",
    ]

    # ========================================================
    # INIT
    # ========================================================

    def __init__(self) -> None:
        pass

    # ========================================================
    # NORMALIZE
    # ========================================================

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalisasi input user.
        """

        if not text:
            return ""

        text = text.strip().lower()

        # Hilangkan whitespace berlebihan.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        # Hilangkan tanda baca sederhana
        # di akhir kalimat.
        text = re.sub(
            r"[!?.,]+$",
            "",
            text,
        )

        return text.strip()

    # ========================================================
    # MATCH
    # ========================================================

    @staticmethod
    def _match_patterns(
        text: str,
        patterns: list[str],
    ) -> Optional[re.Match]:

        for pattern in patterns:

            match = re.match(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return match

        return None

    # ========================================================
    # DETECT
    # ========================================================

    def detect(
        self,
        text: str,
        requested_mode: str = "auto",
    ) -> IntentResult:

        normalized = self.normalize(text)

        if not normalized:

            return IntentResult(
                intent="empty",
                confidence=1.0,
                mode="fast",
            )

        mode = self._detect_mode(
            normalized,
            requested_mode,
        )

        # ----------------------------------------------------
        # GREETING
        # ----------------------------------------------------

        if self._match_patterns(
            normalized,
            self.GREETING_PATTERNS,
        ):

            return IntentResult(
                intent="greeting",
                confidence=0.99,
                mode="fast",
            )

        # ----------------------------------------------------
        # IDENTITY
        # ----------------------------------------------------

        if self._match_patterns(
            normalized,
            self.IDENTITY_PATTERNS,
        ):

            return IntentResult(
                intent="identity",
                confidence=0.99,
                mode="fast",
            )

        # ----------------------------------------------------
        # MEMORY COUNT
        # ----------------------------------------------------

        if self._match_patterns(
            normalized,
            self.MEMORY_COUNT_PATTERNS,
        ):

            return IntentResult(
                intent="memory_count",
                confidence=0.99,
                mode="fast",
            )

        # ----------------------------------------------------
        # MEMORY DELETE
        # ----------------------------------------------------

        match = self._match_patterns(
            normalized,
            self.MEMORY_DELETE_PATTERNS,
        )

        if match:

            target = match.group(1).strip()

            if target:

                return IntentResult(
                    intent="memory_delete",
                    confidence=0.98,
                    mode="fast",
                    entities={
                        "target": target,
                    },
                )

        # ----------------------------------------------------
        # MEMORY SEARCH
        # ----------------------------------------------------

        match = self._match_patterns(
            normalized,
            self.MEMORY_SEARCH_PATTERNS,
        )

        if match:

            target = match.group(1).strip()

            if target:

                return IntentResult(
                    intent="memory_search",
                    confidence=0.98,
                    mode="fast",
                    entities={
                        "query": target,
                    },
                )

        # ----------------------------------------------------
        # MEMORY SAVE
        # ----------------------------------------------------

        match = self._match_patterns(
            normalized,
            self.MEMORY_SAVE_PATTERNS,
        )

        if match:

            content = match.group(1).strip()

            # Jangan salah mendeteksi kalimat
            # "ingat?" sebagai memory.
            if content and content not in {
                "?",
                "apa",
                "apa?",
            }:

                return IntentResult(
                    intent="memory_save",
                    confidence=0.97,
                    mode="fast",
                    entities={
                        "content": content,
                    },
                )

        # ----------------------------------------------------
        # HELP
        # ----------------------------------------------------

        if self._match_patterns(
            normalized,
            self.HELP_PATTERNS,
        ):

            return IntentResult(
                intent="help",
                confidence=0.99,
                mode="fast",
            )

        # ----------------------------------------------------
        # CONVERSATION
        # ----------------------------------------------------

        return IntentResult(
            intent="conversation",
            confidence=0.70,
            mode=mode,
        )

    # ========================================================
    # MODE DETECTION
    # ========================================================

    def _detect_mode(
        self,
        text: str,
        requested_mode: str,
    ) -> str:

        requested = (
            requested_mode
            or "auto"
        ).strip().lower()

        if requested in {
            "fast",
            "normal",
            "deep",
        }:

            return requested

        # ----------------------------------------------------
        # FAST
        # ----------------------------------------------------

        fast_keywords = [
            "jawab singkat",
            "singkat saja",
            "secara singkat",
            "berapa",
            "siapa",
            "kapan",
            "dimana",
            "apa itu",
        ]

        for keyword in fast_keywords:

            if (
                keyword in text
                and len(text) < 180
            ):

                return "fast"

        # ----------------------------------------------------
        # DEEP
        # ----------------------------------------------------

        deep_keywords = [
            "analisis mendalam",
            "analisa mendalam",
            "jelaskan secara mendalam",
            "bandingkan secara detail",
            "buat arsitektur",
            "debug",
            "debugging",
            "riset",
            "research",
            "program lengkap",
            "kode lengkap",
            "full code",
            "buat sistem",
            "arsitektur",
            "dari awal sampai akhir",
            "production",
            "production ready",
        ]

        for keyword in deep_keywords:

            if keyword in text:

                return "deep"

        return "normal"

    # ========================================================
    # QUICK CHECK
    # ========================================================

    def is_command(
        self,
        text: str,
    ) -> bool:

        result = self.detect(text)

        return result.intent != "conversation"

    # ========================================================
    # INTENT ONLY
    # ========================================================

    def get_intent(
        self,
        text: str,
    ) -> str:

        return self.detect(text).intent

    # ========================================================
    # VERSION
    # ========================================================

    def info(self) -> dict:

        return {
            "name": "ZAI Intent Engine",
            "version": self.VERSION,
            "status": "ONLINE",
            "intents": [
                "empty",
                "greeting",
                "identity",
                "memory_save",
                "memory_search",
                "memory_delete",
                "memory_count",
                "help",
                "conversation",
            ],
        }


# ============================================================
# GLOBAL INSTANCE
# ============================================================

intent_engine = IntentEngine()