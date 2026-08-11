from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import (
    parse_qs,
    unquote,
    urljoin,
    urlparse,
)

from .agent_result import AgentResult
from .base_agent import BaseAgent


# ============================================================================
# ZAI RESEARCH AGENT
# ============================================================================
#
# ResearchAgent adalah agent penelitian ZAI.
#
# Fokus utama:
#
#   - research task analysis
#   - query extraction
#   - topic detection
#   - URL extraction
#   - source normalization
#   - source classification
#   - text cleaning
#   - evidence extraction
#   - keyword extraction
#   - entity extraction
#   - claim extraction
#   - source quality scoring
#   - research planning
#   - synthesis preparation
#   - confidence scoring
#   - duplicate detection
#   - citation preparation
#   - research result formatting
#
# Agent ini sengaja tidak mengeksekusi arbitrary code.
#
# HTTP/network provider dibuat sebagai extension point.
# Default behavior aman: agent dapat menganalisis task dan data
# penelitian yang diberikan kepadanya tanpa melakukan network request.
#
# Provider eksternal dapat ditambahkan kemudian oleh ZAI Research Engine.
#
# ============================================================================


# ============================================================================
# CONSTANTS
# ============================================================================

AGENT_NAME = "research_agent"
AGENT_VERSION = "1.0.0"

AGENT_DESCRIPTION = (
    "Research specialist agent untuk melakukan perencanaan penelitian, "
    "analisis sumber, ekstraksi evidence, evaluasi kualitas sumber, "
    "sintesis informasi, dan penyusunan hasil riset ZAI."
)

AGENT_CAPABILITIES: tuple[str, ...] = (
    "research",
    "research_planning",
    "query_analysis",
    "topic_detection",
    "source_analysis",
    "source_scoring",
    "source_normalization",
    "url_extraction",
    "text_cleaning",
    "keyword_extraction",
    "entity_extraction",
    "claim_extraction",
    "evidence_extraction",
    "citation_preparation",
    "duplicate_detection",
    "research_synthesis",
    "confidence_scoring",
)

MAX_TASK_LENGTH = 100_000
MAX_TEXT_LENGTH = 500_000
MAX_SOURCE_COUNT = 100
MAX_KEYWORDS = 50
MAX_ENTITIES = 100
MAX_CLAIMS = 100
MAX_EVIDENCE = 200

MIN_WORD_LENGTH = 3

DEFAULT_CONFIDENCE = 0.20

# Common Indonesian stop words.
INDONESIAN_STOPWORDS: frozenset[str] = frozenset(
    {
        "yang",
        "dan",
        "atau",
        "dari",
        "untuk",
        "dengan",
        "pada",
        "dalam",
        "adalah",
        "ini",
        "itu",
        "ke",
        "di",
        "dengan",
        "sebagai",
        "oleh",
        "karena",
        "agar",
        "juga",
        "akan",
        "dapat",
        "bisa",
        "lebih",
        "kurang",
        "sangat",
        "sudah",
        "telah",
        "belum",
        "tidak",
        "bukan",
        "apa",
        "siapa",
        "mana",
        "kapan",
        "bagaimana",
        "mengapa",
        "sebuah",
        "suatu",
        "para",
        "terhadap",
        "tentang",
        "antara",
        "hingga",
        "sampai",
        "serta",
        "namun",
        "tetapi",
        "jika",
        "bila",
        "maka",
        "sehingga",
        "dengan",
        "tanpa",
        "secara",
        "bagi",
        "mereka",
        "kami",
        "kita",
        "saya",
        "anda",
        "dia",
        "ia",
        "nya",
        "the",
        "and",
        "or",
        "for",
        "with",
        "from",
        "this",
        "that",
        "these",
        "those",
        "are",
        "was",
        "were",
        "been",
        "being",
        "have",
        "has",
        "had",
        "not",
        "but",
        "can",
        "could",
        "would",
        "should",
        "about",
        "into",
        "over",
        "under",
        "after",
        "before",
        "between",
        "during",
        "through",
        "more",
        "most",
        "less",
        "than",
        "very",
        "research",
        "riset",
    }
)


URL_PATTERN = re.compile(
    r"""
    (?:
        https?://
        |
        www\.
    )
    [^\s<>"'`]+
    """,
    re.IGNORECASE | re.VERBOSE,
)


DOMAIN_PATTERN = re.compile(
    r"""
    \b
    (?:
        [a-z0-9-]+\.)+
        [a-z]{2,}
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


EMAIL_PATTERN = re.compile(
    r"""
    \b
    [A-Z0-9._%+-]+
    @
    [A-Z0-9.-]+\.[A-Z]{2,}
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


WORD_PATTERN = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9_-]*"
)


SENTENCE_PATTERN = re.compile(
    r"""
    (?<=[.!?])
    \s+
    |
    (?<=\n)
    \s*
    """,
    re.VERBOSE,
)


NUMBER_PATTERN = re.compile(
    r"""
    (?<!\w)
    (?:
        \d+(?:[.,]\d+)?
        |
        \d{1,3}(?:[.,]\d{3})+
    )
    (?:\s*[%$€£¥]|[A-Za-z]+)?
    """,
    re.VERBOSE,
)


YEAR_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\b"
)


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(slots=True)
class ResearchSource:
    """
    Representasi source penelitian.
    """

    url: str = ""
    title: str = ""
    domain: str = ""
    source_type: str = "unknown"

    authority_score: float = 0.0
    relevance_score: float = 0.0
    freshness_score: float = 0.0
    quality_score: float = 0.0

    accessible: bool = True

    snippet: str = ""
    text: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "source_type": self.source_type,
            "authority_score": round(self.authority_score, 4),
            "relevance_score": round(self.relevance_score, 4),
            "freshness_score": round(self.freshness_score, 4),
            "quality_score": round(self.quality_score, 4),
            "accessible": self.accessible,
            "snippet": self.snippet,
            "text_length": len(self.text),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ResearchClaim:
    """
    Representasi claim yang ditemukan dalam research material.
    """

    claim_id: str
    text: str

    confidence: float = 0.0

    source_urls: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    claim_type: str = "general"

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "source_urls": list(self.source_urls),
            "evidence_ids": list(self.evidence_ids),
            "claim_type": self.claim_type,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ResearchEvidence:
    """
    Evidence yang mendukung sebuah claim.
    """

    evidence_id: str
    text: str

    source_url: str = ""
    source_title: str = ""

    relevance_score: float = 0.0
    strength_score: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "relevance_score": round(self.relevance_score, 4),
            "strength_score": round(self.strength_score, 4),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ResearchPlan:
    """
    Research plan yang dibuat sebelum proses sintesis.
    """

    objective: str
    queries: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    required_sources: int = 3

    source_preferences: list[str] = field(default_factory=list)

    steps: list[str] = field(default_factory=list)

    risks: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "queries": list(self.queries),
            "topics": list(self.topics),
            "required_sources": self.required_sources,
            "source_preferences": list(self.source_preferences),
            "steps": list(self.steps),
            "risks": list(self.risks),
            "metadata": dict(self.metadata),
        }


# ============================================================================
# RESEARCH AGENT
# ============================================================================


class ResearchAgent(BaseAgent):
    """
    Agent penelitian utama ZAI.

    Agent ini berfungsi sebagai research specialist dan preparation layer
    sebelum data diteruskan ke engine pencarian / web connector / LLM.

    Prinsip desain:

        1. Deterministic
        2. Observable
        3. Extensible
        4. Safe by default
        5. Tidak menjalankan arbitrary code
        6. Tidak menganggap source sebagai fakta hanya karena URL tersedia
        7. Memisahkan evidence dari synthesis
    """

    name = AGENT_NAME
    version = AGENT_VERSION
    description = AGENT_DESCRIPTION
    capabilities = AGENT_CAPABILITIES

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------

    def __init__(
        self,
        *,
        max_sources: int = MAX_SOURCE_COUNT,
        max_keywords: int = MAX_KEYWORDS,
        max_claims: int = MAX_CLAIMS,
        max_evidence: int = MAX_EVIDENCE,
        network_enabled: bool = False,
    ) -> None:
        super().__init__()

        self.max_sources = max(
            1,
            min(max_sources, MAX_SOURCE_COUNT),
        )

        self.max_keywords = max(
            1,
            min(max_keywords, MAX_KEYWORDS),
        )

        self.max_claims = max(
            1,
            min(max_claims, MAX_CLAIMS),
        )

        self.max_evidence = max(
            1,
            min(max_evidence, MAX_EVIDENCE),
        )

        self.network_enabled = bool(network_enabled)

        self.research_count = 0
        self.plan_count = 0
        self.synthesis_count = 0

        self.last_plan: ResearchPlan | None = None

        self._source_cache: dict[str, ResearchSource] = {}

    # ----------------------------------------------------------------------
    # BaseAgent information
    # ----------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """
        Return informasi lengkap agent.
        """

        base = super().info()

        base.update(
            {
                "domain": "research",
                "network_enabled": self.network_enabled,
                "max_sources": self.max_sources,
                "max_keywords": self.max_keywords,
                "max_claims": self.max_claims,
                "max_evidence": self.max_evidence,
                "research_count": self.research_count,
                "plan_count": self.plan_count,
                "synthesis_count": self.synthesis_count,
                "source_cache_size": len(self._source_cache),
            }
        )

        return base

    # ----------------------------------------------------------------------
    # Health
    # ----------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """
        Health information untuk monitoring ZAI.
        """

        total = self.execution_count

        success_rate = (
            (self.success_count / total) * 100
            if total
            else 0.0
        )

        return {
            "agent": self.name,
            "version": self.version,
            "status": "HEALTHY",
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(success_rate, 2),
            "research_count": self.research_count,
            "plan_count": self.plan_count,
            "synthesis_count": self.synthesis_count,
            "network_enabled": self.network_enabled,
            "source_cache_size": len(self._source_cache),
        }

    # ----------------------------------------------------------------------
    # Main execution
    # ----------------------------------------------------------------------

    async def run(
        self,
        task: str,
        result: AgentResult,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Entry point ResearchAgent.
        """

        started = time.perf_counter()

        normalized_task = self.normalize_task(task)

        result.add_observation(
            "research_agent_started",
            agent=self.name,
            task_length=len(normalized_task),
        )

        if not normalized_task:
            result.success = False
            result.response = (
                "ResearchAgent menerima task kosong."
            )

            result.add_error(
                "Task penelitian tidak boleh kosong."
            )

            result.status = "failed"

            return result

        if len(normalized_task) > MAX_TASK_LENGTH:
            result.success = False
            result.response = (
                "Task penelitian terlalu panjang."
            )

            result.add_error(
                f"Task melebihi batas {MAX_TASK_LENGTH} karakter."
            )

            result.status = "failed"

            return result

        result.add_observation(
            "task_normalized",
            original_length=len(task),
            normalized_length=len(normalized_task),
        )

        research_mode = self.detect_research_mode(
            normalized_task
        )

        result.add_observation(
            "research_mode_detected",
            mode=research_mode,
        )

        plan = self.create_plan(
            normalized_task
        )

        self.last_plan = plan
        self.plan_count += 1

        result.add_observation(
            "research_plan_created",
            query_count=len(plan.queries),
            topic_count=len(plan.topics),
            required_sources=plan.required_sources,
        )

        supplied_sources = kwargs.get(
            "sources",
            [],
        )

        supplied_text = kwargs.get(
            "text",
            "",
        )

        source_objects = self.normalize_sources(
            supplied_sources
        )

        if source_objects:
            result.add_observation(
                "sources_normalized",
                source_count=len(source_objects),
            )

        if supplied_text:
            result.add_observation(
                "research_text_received",
                text_length=len(str(supplied_text)),
            )

        keywords = self.extract_keywords(
            normalized_task
        )

        entities = self.extract_entities(
            normalized_task
        )

        urls = self.extract_urls(
            normalized_task
        )

        result.add_observation(
            "research_signals_extracted",
            keyword_count=len(keywords),
            entity_count=len(entities),
            url_count=len(urls),
        )

        if urls:
            url_sources = [
                self.source_from_url(url)
                for url in urls
            ]

            source_objects.extend(
                url_sources
            )

            source_objects = self.deduplicate_sources(
                source_objects
            )

        for source in source_objects:
            source.relevance_score = self.score_source_relevance(
                source,
                keywords,
            )

            source.quality_score = self.calculate_source_quality(
                source
            )

        source_objects.sort(
            key=lambda item: item.quality_score,
            reverse=True,
        )

        claims: list[ResearchClaim] = []
        evidence: list[ResearchEvidence] = []

        if supplied_text:
            cleaned_text = self.clean_text(
                str(supplied_text)
            )

            evidence = self.extract_evidence(
                cleaned_text,
                source_objects,
                keywords,
            )

            claims = self.extract_claims(
                cleaned_text,
                source_objects,
                evidence,
            )

            result.add_observation(
                "material_analyzed",
                text_length=len(cleaned_text),
                claim_count=len(claims),
                evidence_count=len(evidence),
            )

        confidence = self.calculate_research_confidence(
            task=normalized_task,
            sources=source_objects,
            claims=claims,
            evidence=evidence,
        )

        response = self.build_research_response(
            task=normalized_task,
            mode=research_mode,
            plan=plan,
            sources=source_objects,
            claims=claims,
            evidence=evidence,
            confidence=confidence,
        )

        result.response = response
        result.success = True
        result.status = "completed"

        self.research_count += 1

        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        result.metadata.update(
            {
                "agent": self.name,
                "agent_version": self.version,
                "research_mode": research_mode,
                "task_length": len(normalized_task),
                "keyword_count": len(keywords),
                "entity_count": len(entities),
                "url_count": len(urls),
                "source_count": len(source_objects),
                "claim_count": len(claims),
                "evidence_count": len(evidence),
                "research_confidence": round(
                    confidence,
                    4,
                ),
                "latency_ms": latency_ms,
                "network_enabled": self.network_enabled,
            }
        )

        result.add_observation(
            "research_completed",
            source_count=len(source_objects),
            claim_count=len(claims),
            evidence_count=len(evidence),
            confidence=round(confidence, 4),
            latency_ms=latency_ms,
        )

        return result

    # ----------------------------------------------------------------------
    # Task normalization
    # ----------------------------------------------------------------------

    @staticmethod
    def normalize_task(
        task: str,
    ) -> str:
        """
        Normalisasi task penelitian.
        """

        if task is None:
            return ""

        value = str(task)

        value = value.replace(
            "\x00",
            " ",
        )

        value = html.unescape(
            value
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    # ----------------------------------------------------------------------
    # Research mode
    # ----------------------------------------------------------------------

    @classmethod
    def detect_research_mode(
        cls,
        task: str,
    ) -> str:
        """
        Menentukan jenis penelitian.
        """

        normalized = task.lower()

        if any(
            keyword in normalized
            for keyword in (
                "bandingkan",
                "compare",
                "comparison",
                "vs",
                "versus",
            )
        ):
            return "comparison"

        if any(
            keyword in normalized
            for keyword in (
                "terbaru",
                "latest",
                "terkini",
                "today",
                "hari ini",
                "update",
                "news",
                "berita",
            )
        ):
            return "current_information"

        if any(
            keyword in normalized
            for keyword in (
                "harga",
                "price",
                "biaya",
                "cost",
                "tarif",
            )
        ):
            return "market_research"

        if any(
            keyword in normalized
            for keyword in (
                "cara",
                "how",
                "tutorial",
                "panduan",
                "guide",
            )
        ):
            return "procedural_research"

        if any(
            keyword in normalized
            for keyword in (
                "akademik",
                "jurnal",
                "paper",
                "ilmiah",
                "academic",
                "scientific",
            )
        ):
            return "academic_research"

        if any(
            keyword in normalized
            for keyword in (
                "produk",
                "product",
                "software",
                "aplikasi",
                "laptop",
                "hp",
            )
        ):
            return "product_research"

        return "general_research"

    # ----------------------------------------------------------------------
    # Research planning
    # ----------------------------------------------------------------------

    def create_plan(
        self,
        task: str,
    ) -> ResearchPlan:
        """
        Membuat research plan.
        """

        mode = self.detect_research_mode(
            task
        )

        keywords = self.extract_keywords(
            task,
            limit=10,
        )

        topics = self.detect_topics(
            task
        )

        queries = self.generate_queries(
            task=task,
            keywords=keywords,
            mode=mode,
        )

        preferences = self.source_preferences_for_mode(
            mode
        )

        steps = [
            "Normalisasi research objective.",
            "Identifikasi topik dan keyword utama.",
            "Bangun query penelitian.",
            "Kumpulkan dan normalisasi source.",
            "Evaluasi relevansi dan kualitas source.",
            "Ekstraksi evidence dan claim.",
            "Deteksi konflik atau duplikasi informasi.",
            "Sintesis hasil penelitian.",
            "Hitung confidence.",
            "Siapkan citation metadata.",
        ]

        risks = [
            "Source dapat tidak tersedia.",
            "Informasi dapat sudah kedaluwarsa.",
            "Satu source tidak cukup untuk kesimpulan kuat.",
            "Claim dapat membutuhkan verifikasi tambahan.",
        ]

        if mode == "current_information":
            risks.append(
                "Informasi terkini memerlukan sumber dengan timestamp terbaru."
            )

        if mode == "comparison":
            risks.append(
                "Perbandingan harus menggunakan kriteria yang konsisten."
            )

        required_sources = 3

        if mode == "academic_research":
            required_sources = 5

        elif mode == "comparison":
            required_sources = 4

        return ResearchPlan(
            objective=task,
            queries=queries,
            topics=topics,
            required_sources=required_sources,
            source_preferences=preferences,
            steps=steps,
            risks=risks,
            metadata={
                "mode": mode,
                "generated_at": self.now_iso(),
            },
        )

    # ----------------------------------------------------------------------
    # Query generation
    # ----------------------------------------------------------------------

    @classmethod
    def generate_queries(
        cls,
        *,
        task: str,
        keywords: Sequence[str],
        mode: str,
    ) -> list[str]:
        """
        Membuat query penelitian dari task.
        """

        queries: list[str] = []

        base = cls.normalize_task(
            task
        )

        if base:
            queries.append(
                base
            )

        keyword_query = " ".join(
            keywords[:8]
        )

        if keyword_query:
            queries.append(
                keyword_query
            )

        if mode == "comparison":
            queries.append(
                f"{base} comparison"
            )

        elif mode == "academic_research":
            queries.append(
                f"{base} research paper"
            )

        elif mode == "current_information":
            queries.append(
                f"{base} latest"
            )

        elif mode == "market_research":
            queries.append(
                f"{base} price market"
            )

        elif mode == "procedural_research":
            queries.append(
                f"{base} guide tutorial"
            )

        cleaned: list[str] = []

        seen: set[str] = set()

        for query in queries:
            query = cls.normalize_task(
                query
            )

            key = query.lower()

            if not query:
                continue

            if key in seen:
                continue

            seen.add(key)
            cleaned.append(query)

        return cleaned[:10]

    # ----------------------------------------------------------------------
    # Topic detection
    # ----------------------------------------------------------------------

    @classmethod
    def detect_topics(
        cls,
        task: str,
    ) -> list[str]:
        """
        Deteksi topik dari keyword dominan.
        """

        keywords = cls.extract_keywords(
            task,
            limit=15,
        )

        topics: list[str] = []

        for keyword in keywords:
            if keyword not in topics:
                topics.append(
                    keyword
                )

        return topics[:10]

    # ----------------------------------------------------------------------
    # Source preferences
    # ----------------------------------------------------------------------

    @staticmethod
    def source_preferences_for_mode(
        mode: str,
    ) -> list[str]:
        """
        Prioritas tipe source berdasarkan research mode.
        """

        mapping: dict[str, list[str]] = {
            "academic_research": [
                "academic",
                "government",
                "institutional",
                "official",
            ],
            "current_information": [
                "official",
                "news",
                "government",
                "institutional",
            ],
            "market_research": [
                "official",
                "marketplace",
                "company",
                "review",
            ],
            "comparison": [
                "official",
                "independent",
                "review",
                "academic",
            ],
            "product_research": [
                "official",
                "technical",
                "review",
                "documentation",
            ],
            "procedural_research": [
                "official",
                "documentation",
                "technical",
                "tutorial",
            ],
            "general_research": [
                "official",
                "government",
                "academic",
                "institutional",
                "reputable_media",
            ],
        }

        return list(
            mapping.get(
                mode,
                mapping["general_research"],
            )
        )

    # ----------------------------------------------------------------------
    # URL extraction
    # ----------------------------------------------------------------------

    @classmethod
    def extract_urls(
        cls,
        text: str,
    ) -> list[str]:
        """
        Extract URL dari text.
        """

        if not text:
            return []

        matches = URL_PATTERN.findall(
            text
        )

        urls: list[str] = []

        seen: set[str] = set()

        for raw_url in matches:
            url = raw_url.strip()

            url = url.rstrip(
                ".,;:!?)]}"
            )

            if url.startswith(
                "www."
            ):
                url = (
                    "https://"
                    + url
                )

            if not url:
                continue

            normalized = cls.normalize_url(
                url
            )

            if normalized in seen:
                continue

            seen.add(normalized)
            urls.append(normalized)

        return urls

    # ----------------------------------------------------------------------
    # URL normalization
    # ----------------------------------------------------------------------

    @staticmethod
    def normalize_url(
        url: str,
    ) -> str:
        """
        Normalisasi URL.
        """

        value = (
            str(url)
            .strip()
        )

        if not value:
            return ""

        if value.startswith(
            "www."
        ):
            value = (
                "https://"
                + value
            )

        parsed = urlparse(
            value
        )

        if not parsed.scheme:
            value = (
                "https://"
                + value
            )

            parsed = urlparse(
                value
            )

        scheme = (
            parsed.scheme.lower()
        )

        netloc = (
            parsed.netloc.lower()
        )

        path = (
            parsed.path
            or ""
        )

        if path != "/":
            path = path.rstrip(
                "/"
            )

        return (
            f"{scheme}://"
            f"{netloc}"
            f"{path}"
            + (
                f"?{parsed.query}"
                if parsed.query
                else ""
            )
        )

    # ----------------------------------------------------------------------
    # Source creation
    # ----------------------------------------------------------------------

    def source_from_url(
        self,
        url: str,
    ) -> ResearchSource:
        """
        Membuat ResearchSource dari URL.
        """

        normalized = self.normalize_url(
            url
        )

        cached = self._source_cache.get(
            normalized
        )

        if cached is not None:
            return cached

        parsed = urlparse(
            normalized
        )

        domain = (
            parsed.netloc.lower()
        )

        source_type = self.classify_domain(
            domain
        )

        source = ResearchSource(
            url=normalized,
            title=self.title_from_url(
                normalized
            ),
            domain=domain,
            source_type=source_type,
            authority_score=self.authority_score_for_domain(
                domain
            ),
            relevance_score=0.0,
            freshness_score=self.freshness_score_from_url(
                normalized
            ),
            quality_score=0.0,
            accessible=True,
        )

        source.quality_score = (
            self.calculate_source_quality(
                source
            )
        )

        self._source_cache[
            normalized
        ] = source

        return source

    # ----------------------------------------------------------------------
    # Source normalization
    # ----------------------------------------------------------------------

    def normalize_sources(
        self,
        sources: Any,
    ) -> list[ResearchSource]:
        """
        Normalisasi source dari berbagai bentuk input.
        """

        if sources is None:
            return []

        if isinstance(
            sources,
            Mapping,
        ):
            sources = [
                sources
            ]

        elif isinstance(
            str,
            type(sources),
        ):
            sources = [
                sources
            ]

        try:
            items = list(
                sources
            )
        except TypeError:
            items = [
                sources
            ]

        normalized: list[ResearchSource] = []

        for item in items:
            if len(normalized) >= self.max_sources:
                break

            source = self.normalize_source_item(
                item
            )

            if source is not None:
                normalized.append(
                    source
                )

        return self.deduplicate_sources(
            normalized
        )

    # ----------------------------------------------------------------------
    # Individual source normalization
    # ----------------------------------------------------------------------

    def normalize_source_item(
        self,
        item: Any,
    ) -> ResearchSource | None:
        """
        Normalisasi satu source.
        """

        if isinstance(
            item,
            ResearchSource,
        ):
            return item

        if isinstance(
            item,
            str,
        ):
            value = item.strip()

            if not value:
                return None

            if (
                value.startswith(
                    "http://"
                )
                or value.startswith(
                    "https://"
                )
                or value.startswith(
                    "www."
                )
            ):
                return self.source_from_url(
                    value
                )

            return ResearchSource(
                title="Provided text",
                source_type="provided_text",
                snippet=value[:500],
                text=value[:MAX_TEXT_LENGTH],
                authority_score=0.5,
                relevance_score=0.5,
                freshness_score=0.5,
                quality_score=0.5,
            )

        if isinstance(
            item,
            Mapping,
        ):
            url = str(
                item.get(
                    "url",
                    "",
                )
                or ""
            )

            title = str(
                item.get(
                    "title",
                    "",
                )
                or ""
            )

            text = str(
                item.get(
                    "text",
                    item.get(
                        "content",
                        "",
                    ),
                )
                or ""
            )

            snippet = str(
                item.get(
                    "snippet",
                    "",
                )
                or ""
            )

            if url:
                source = self.source_from_url(
                    url
                )

                if title:
                    source.title = title

                if text:
                    source.text = text[
                        :MAX_TEXT_LENGTH
                    ]

                if snippet:
                    source.snippet = snippet[
                        :2_000
                    ]

                metadata = item.get(
                    "metadata",
                    {},
                )

                if isinstance(
                    metadata,
                    Mapping,
                ):
                    source.metadata.update(
                        metadata
                    )

                return source

            if text:
                return ResearchSource(
                    title=title
                    or "Provided material",
                    source_type="provided_text",
                    snippet=(
                        snippet
                        or text[:500]
                    ),
                    text=text[
                        :MAX_TEXT_LENGTH
                    ],
                    authority_score=0.5,
                    relevance_score=0.5,
                    freshness_score=0.5,
                    quality_score=0.5,
                )

        return None

    # ----------------------------------------------------------------------
    # Source deduplication
    # ----------------------------------------------------------------------

    @staticmethod
    def deduplicate_sources(
        sources: Sequence[ResearchSource],
    ) -> list[ResearchSource]:
        """
        Menghapus source duplikat.
        """

        result: list[ResearchSource] = []

        seen: set[str] = set()

        for source in sources:
            key = (
                source.url.lower().strip()
                if source.url
                else hashlib.sha256(
                    source.text.encode(
                        "utf-8",
                        errors="ignore",
                    )
                ).hexdigest()
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(source)

        return result

    # ----------------------------------------------------------------------
    # Domain classification
    # ----------------------------------------------------------------------

    @staticmethod
    def classify_domain(
        domain: str,
    ) -> str:
        """
        Klasifikasi domain secara heuristik.
        """

        domain = domain.lower()

        if (
            domain.endswith(
                ".gov"
            )
            or ".gov." in domain
        ):
            return "government"

        if (
            domain.endswith(
                ".edu"
            )
            or ".ac." in domain
        ):
            return "academic"

        if any(
            marker in domain
            for marker in (
                "journal",
                "researchgate",
                "sciencedirect",
                "springer",
                "nature.com",
                "arxiv",
            )
        ):
            return "academic"

        if any(
            marker in domain
            for marker in (
                "github.com",
                "gitlab.com",
                "stackoverflow.com",
                "developer.",
                "docs.",
            )
        ):
            return "technical"

        if any(
            marker in domain
            for marker in (
                "reuters.com",
                "bbc.",
                "cnn.com",
                "nytimes.com",
                "kompas.com",
                "tempo.co",
                "detik.com",
            )
        ):
            return "news"

        if any(
            marker in domain
            for marker in (
                "amazon.",
                "tokopedia.",
                "shopee.",
                "blibli.",
            )
        ):
            return "marketplace"

        return "website"

    # ----------------------------------------------------------------------
    # Authority scoring
    # ----------------------------------------------------------------------

    @staticmethod
    def authority_score_for_domain(
        domain: str,
    ) -> float:
        """
        Heuristic authority score.
        """

        domain = domain.lower()

        if (
            domain.endswith(".gov")
            or ".gov." in domain
        ):
            return 0.95

        if (
            domain.endswith(".edu")
            or ".ac." in domain
        ):
            return 0.92

        if any(
            marker in domain
            for marker in (
                "who.int",
                "worldbank.org",
                "imf.org",
                "un.org",
                "oecd.org",
            )
        ):
            return 0.98

        if any(
            marker in domain
            for marker in (
                "nature.com",
                "sciencedirect.com",
                "springer.com",
                "arxiv.org",
            )
        ):
            return 0.94

        if any(
            marker in domain
            for marker in (
                "reuters.com",
                "bbc.com",
                "nytimes.com",
            )
        ):
            return 0.86

        if any(
            marker in domain
            for marker in (
                "github.com",
                "stackoverflow.com",
                "developer.mozilla.org",
            )
        ):
            return 0.82

        return 0.50

    # ----------------------------------------------------------------------
    # Freshness score
    # ----------------------------------------------------------------------

    @staticmethod
    def freshness_score_from_url(
        url: str,
    ) -> float:
        """
        Estimasi freshness berdasarkan tahun pada URL.
        """

        years = [
            int(value)
            for value in YEAR_PATTERN.findall(
                url
            )
        ]

        if not years:
            return 0.50

        current_year = (
            datetime.now(
                timezone.utc
            ).year
        )

        newest = max(
            years
        )

        age = max(
            0,
            current_year - newest,
        )

        if age == 0:
            return 1.0

        if age == 1:
            return 0.90

        if age == 2:
            return 0.80

        if age <= 5:
            return 0.65

        if age <= 10:
            return 0.45

        return 0.25

    # ----------------------------------------------------------------------
    # Source relevance
    # ----------------------------------------------------------------------

    @classmethod
    def score_source_relevance(
        cls,
        source: ResearchSource,
        keywords: Sequence[str],
    ) -> float:
        """
        Mengukur relevansi source berdasarkan keyword.
        """

        if not keywords:
            return 0.50

        haystack = " ".join(
            (
                source.title,
                source.snippet,
                source.text[:10_000],
                source.domain,
            )
        ).lower()

        matches = 0

        for keyword in keywords:
            if keyword.lower() in haystack:
                matches += 1

        ratio = (
            matches / len(keywords)
        )

        return min(
            1.0,
            ratio * 1.25,
        )

    # ----------------------------------------------------------------------
    # Source quality
    # ----------------------------------------------------------------------

    @staticmethod
    def calculate_source_quality(
        source: ResearchSource,
    ) -> float:
        """
        Menghitung kualitas source.
        """

        authority = (
            source.authority_score
        )

        relevance = (
            source.relevance_score
        )

        freshness = (
            source.freshness_score
        )

        accessible = (
            1.0
            if source.accessible
            else 0.0
        )

        score = (
            authority * 0.40
            + relevance * 0.30
            + freshness * 0.15
            + accessible * 0.15
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    # ----------------------------------------------------------------------
    # Keyword extraction
    # ----------------------------------------------------------------------

    @classmethod
    def extract_keywords(
        cls,
        text: str,
        *,
        limit: int | None = None,
    ) -> list[str]:
        """
        Extract keyword sederhana berbasis frequency.
        """

        if not text:
            return []

        words = [
            word.lower()
            for word in WORD_PATTERN.findall(
                text
            )
        ]

        filtered: list[str] = []

        for word in words:
            if len(word) < MIN_WORD_LENGTH:
                continue

            if word.isdigit():
                continue

            if word in INDONESIAN_STOPWORDS:
                continue

            filtered.append(
                word
            )

        counter = Counter(
            filtered
        )

        target = (
            limit
            if limit is not None
            else cls.__dict__.get(
                "max_keywords",
                MAX_KEYWORDS,
            )
        )

        target = max(
            1,
            min(
                int(target),
                MAX_KEYWORDS,
            ),
        )

        return [
            word
            for word, _count in counter.most_common(
                target
            )
        ]

    # ----------------------------------------------------------------------
    # Entity extraction
    # ----------------------------------------------------------------------

    @classmethod
    def extract_entities(
        cls,
        text: str,
    ) -> list[str]:
        """
        Extract entity heuristically.

        Entity yang dideteksi:
            - URL/domain
            - email
            - proper-case phrases
            - angka penting
        """

        if not text:
            return []

        entities: list[str] = []

        for url in cls.extract_urls(
            text
        ):
            if url not in entities:
                entities.append(
                    url
                )

        for email in EMAIL_PATTERN.findall(
            text
        ):
            if email not in entities:
                entities.append(
                    email
                )

        proper_pattern = re.compile(
            r"\b"
            r"(?:"
            r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ0-9-]*"
            r"(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ0-9-]*){0,4}"
            r")"
            r"\b"
        )

        for match in proper_pattern.findall(
            text
        ):
            value = match.strip()

            if len(value) < 3:
                continue

            if value.lower() in INDONESIAN_STOPWORDS:
                continue

            if value not in entities:
                entities.append(
                    value
                )

        for number in NUMBER_PATTERN.findall(
            text
        ):
            if number not in entities:
                entities.append(
                    number
                )

        return entities[
            :MAX_ENTITIES
        ]

    # ----------------------------------------------------------------------
    # Text cleaning
    # ----------------------------------------------------------------------

    @staticmethod
    def clean_text(
        text: str,
    ) -> str:
        """
        Membersihkan HTML, whitespace, dan artefak umum.
        """

        if not text:
            return ""

        value = html.unescape(
            str(text)
        )

        value = re.sub(
            r"<script\b[^>]*>.*?</script>",
            " ",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        )

        value = re.sub(
            r"<style\b[^>]*>.*?</style>",
            " ",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        )

        value = re.sub(
            r"<[^>]+>",
            " ",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()[
            :MAX_TEXT_LENGTH
        ]

    # ----------------------------------------------------------------------
    # Sentence extraction
    # ----------------------------------------------------------------------

    @classmethod
    def split_sentences(
        cls,
        text: str,
    ) -> list[str]:
        """
        Memecah text menjadi sentence.
        """

        cleaned = cls.clean_text(
            text
        )

        if not cleaned:
            return []

        sentences = re.split(
            SENTENCE_PATTERN,
            cleaned,
        )

        result: list[str] = []

        for sentence in sentences:
            value = sentence.strip()

            if not value:
                continue

            result.append(
                value
            )

        return result

    # ----------------------------------------------------------------------
    # Evidence extraction
    # ----------------------------------------------------------------------

    def extract_evidence(
        self,
        text: str,
        sources: Sequence[ResearchSource],
        keywords: Sequence[str],
    ) -> list[ResearchEvidence]:
        """
        Extract evidence dari material.
        """

        sentences = self.split_sentences(
            text
        )

        evidence: list[ResearchEvidence] = []

        for index, sentence in enumerate(
            sentences
        ):
            if len(evidence) >= self.max_evidence:
                break

            relevance = self.sentence_relevance(
                sentence,
                keywords,
            )

            if relevance <= 0.0:
                continue

            source = self.best_source_for_text(
                sentence,
                sources,
            )

            strength = self.evidence_strength(
                sentence
            )

            evidence_id = (
                f"ev-{index + 1:04d}-"
                f"{hashlib.sha1(sentence.encode('utf-8')).hexdigest()[:8]}"
            )

            evidence.append(
                ResearchEvidence(
                    evidence_id=evidence_id,
                    text=sentence,
                    source_url=(
                        source.url
                        if source
                        else ""
                    ),
                    source_title=(
                        source.title
                        if source
                        else ""
                    ),
                    relevance_score=relevance,
                    strength_score=strength,
                )
            )

        return evidence

    # ----------------------------------------------------------------------
    # Sentence relevance
    # ----------------------------------------------------------------------

    @classmethod
    def sentence_relevance(
        cls,
        sentence: str,
        keywords: Sequence[str],
    ) -> float:
        """
        Hitung relevansi sentence.
        """

        if not sentence:
            return 0.0

        if not keywords:
            return 0.5

        lower = sentence.lower()

        matches = sum(
            1
            for keyword in keywords
            if keyword.lower() in lower
        )

        if matches == 0:
            return 0.0

        return min(
            1.0,
            matches / max(
                1,
                min(
                    len(keywords),
                    5,
                ),
            ),
        )

    # ----------------------------------------------------------------------
    # Evidence strength
    # ----------------------------------------------------------------------

    @staticmethod
    def evidence_strength(
        sentence: str,
    ) -> float:
        """
        Estimasi kekuatan evidence.
        """

        score = 0.35

        if NUMBER_PATTERN.search(
            sentence
        ):
            score += 0.15

        if YEAR_PATTERN.search(
            sentence
        ):
            score += 0.10

        if any(
            token in sentence.lower()
            for token in (
                "menurut",
                "berdasarkan",
                "data",
                "study",
                "research",
                "penelitian",
                "laporan",
                "report",
            )
        ):
            score += 0.20

        if len(sentence) > 100:
            score += 0.05

        if len(sentence) > 200:
            score += 0.05

        return min(
            1.0,
            score,
        )

    # ----------------------------------------------------------------------
    # Best source matching
    # ----------------------------------------------------------------------

    @classmethod
    def best_source_for_text(
        cls,
        text: str,
        sources: Sequence[ResearchSource],
    ) -> ResearchSource | None:
        """
        Memilih source paling relevan terhadap text.
        """

        if not sources:
            return None

        text_lower = text.lower()

        best: ResearchSource | None = None
        best_score = -1.0

        for source in sources:
            source_text = " ".join(
                (
                    source.title,
                    source.domain,
                    source.snippet,
                )
            ).lower()

            score = 0.0

            for token in cls.extract_keywords(
                text,
                limit=10,
            ):
                if token in source_text:
                    score += 0.1

            score += (
                source.quality_score
                * 0.25
            )

            if (
                source.domain
                and source.domain in text_lower
            ):
                score += 0.3

            if score > best_score:
                best_score = score
                best = source

        return best

    # ----------------------------------------------------------------------
    # Claim extraction
    # ----------------------------------------------------------------------

    def extract_claims(
        self,
        text: str,
        sources: Sequence[ResearchSource],
        evidence: Sequence[ResearchEvidence],
    ) -> list[ResearchClaim]:
        """
        Extract claim dari evidence.
        """

        claims: list[ResearchClaim] = []

        seen: set[str] = set()

        for index, item in enumerate(
            evidence
        ):
            normalized = self.normalize_claim_text(
                item.text
            )

            if not normalized:
                continue

            key = normalized.lower()

            if key in seen:
                continue

            seen.add(key)

            source_urls: list[str] = []

            if item.source_url:
                source_urls.append(
                    item.source_url
                )

            claim_type = self.classify_claim(
                normalized
            )

            confidence = self.claim_confidence(
                item,
                sources,
            )

            claim_id = (
                f"claim-{index + 1:04d}-"
                f"{hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:8]}"
            )

            claims.append(
                ResearchClaim(
                    claim_id=claim_id,
                    text=normalized,
                    confidence=confidence,
                    source_urls=source_urls,
                    evidence_ids=[
                        item.evidence_id
                    ],
                    claim_type=claim_type,
                )
            )

            if len(claims) >= self.max_claims:
                break

        return claims

    # ----------------------------------------------------------------------
    # Claim normalization
    # ----------------------------------------------------------------------

    @staticmethod
    def normalize_claim_text(
        text: str,
    ) -> str:
        """
        Normalisasi claim.
        """

        value = (
            str(text)
            .strip()
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value[:2_000]

    # ----------------------------------------------------------------------
    # Claim classification
    # ----------------------------------------------------------------------

    @staticmethod
    def classify_claim(
        text: str,
    ) -> str:
        """
        Klasifikasi claim sederhana.
        """

        lower = text.lower()

        if NUMBER_PATTERN.search(
            text
        ):
            return "quantitative"

        if any(
            token in lower
            for token in (
                "menurut",
                "berdasarkan",
                "penelitian",
                "study",
                "research",
            )
        ):
            return "evidence_based"

        if any(
            token in lower
            for token in (
                "meningkat",
                "menurun",
                "lebih tinggi",
                "lebih rendah",
                "increase",
                "decrease",
            )
        ):
            return "trend"

        if any(
            token in lower
            for token in (
                "adalah",
                "merupakan",
                "is",
                "are",
                "means",
            )
        ):
            return "definition"

        return "general"

    # ----------------------------------------------------------------------
    # Claim confidence
    # ----------------------------------------------------------------------

    @staticmethod
    def claim_confidence(
        evidence: ResearchEvidence,
        sources: Sequence[ResearchSource],
    ) -> float:
        """
        Hitung confidence claim.
        """

        confidence = (
            evidence.relevance_score
            * 0.45
            + evidence.strength_score
            * 0.35
        )

        if evidence.source_url:
            source = next(
                (
                    source
                    for source in sources
                    if source.url
                    == evidence.source_url
                ),
                None,
            )

            if source:
                confidence += (
                    source.quality_score
                    * 0.20
                )
            else:
                confidence += 0.05

        return min(
            1.0,
            max(
                0.0,
                confidence,
            ),
        )

    # ----------------------------------------------------------------------
    # Research confidence
    # ----------------------------------------------------------------------

    @staticmethod
    def calculate_research_confidence(
        *,
        task: str,
        sources: Sequence[ResearchSource],
        claims: Sequence[ResearchClaim],
        evidence: Sequence[ResearchEvidence],
    ) -> float:
        """
        Confidence keseluruhan research.
        """

        score = DEFAULT_CONFIDENCE

        if task:
            score += 0.10

        if sources:
            source_quality = sum(
                source.quality_score
                for source in sources
            ) / len(sources)

            score += (
                source_quality
                * 0.30
            )

        if evidence:
            evidence_quality = sum(
                item.strength_score
                for item in evidence
            ) / len(evidence)

            score += (
                evidence_quality
                * 0.20
            )

        if claims:
            claim_quality = sum(
                claim.confidence
                for claim in claims
            ) / len(claims)

            score += (
                claim_quality
                * 0.30
            )

        if len(sources) >= 3:
            score += 0.10

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    # ----------------------------------------------------------------------
    # Response builder
    # ----------------------------------------------------------------------

    def build_research_response(
        self,
        *,
        task: str,
        mode: str,
        plan: ResearchPlan,
        sources: Sequence[ResearchSource],
        claims: Sequence[ResearchClaim],
        evidence: Sequence[ResearchEvidence],
        confidence: float,
    ) -> str:
        """
        Membuat response terstruktur.
        """

        lines: list[str] = []

        lines.append(
            "ZAI Research Agent"
        )

        lines.append(
            f"Mode: {mode}"
        )

        lines.append(
            f"Objective: {task}"
        )

        lines.append(
            ""
        )

        lines.append(
            "Research Plan:"
        )

        for index, step in enumerate(
            plan.steps,
            start=1,
        ):
            lines.append(
                f"{index}. {step}"
            )

        lines.append(
            ""
        )

        if plan.queries:
            lines.append(
                "Suggested Queries:"
            )

            for query in plan.queries:
                lines.append(
                    f"- {query}"
                )

            lines.append(
                ""
            )

        if sources:
            lines.append(
                "Sources:"
            )

            for source in sources[
                :self.max_sources
            ]:
                lines.append(
                    "- "
                    + (
                        source.title
                        or source.url
                        or source.domain
                        or "Unnamed source"
                    )
                    + (
                        f" [{source.url}]"
                        if source.url
                        else ""
                    )
                    + (
                        f" | quality={source.quality_score:.2f}"
                    )
                )

            lines.append(
                ""
            )

        if claims:
            lines.append(
                "Claims:"
            )

            for claim in claims[
                :self.max_claims
            ]:
                lines.append(
                    f"- [{claim.confidence:.2f}] "
                    f"{claim.text}"
                )

            lines.append(
                ""
            )

        if evidence:
            lines.append(
                "Evidence:"
            )

            for item in evidence[
                :self.max_evidence
            ]:
                lines.append(
                    f"- [{item.strength_score:.2f}] "
                    f"{item.text}"
                )

            lines.append(
                ""
            )

        lines.append(
            "Research Confidence: "
            f"{confidence:.2f}"
        )

        if not sources:
            lines.append(
                ""
            )

            lines.append(
                "Catatan: belum ada source "
                "eksternal yang diberikan. "
                "ResearchAgent baru membuat "
                "research plan dan analisis awal."
            )

        return "\n".join(
            lines
        )

    # ----------------------------------------------------------------------
    # Citation preparation
    # ----------------------------------------------------------------------

    def prepare_citations(
        self,
        sources: Sequence[ResearchSource],
    ) -> list[dict[str, Any]]:
        """
        Menyiapkan metadata citation.
        """

        citations: list[dict[str, Any]] = []

        for index, source in enumerate(
            sources,
            start=1,
        ):
            if not source.url:
                continue

            citations.append(
                {
                    "citation_id": f"[{index}]",
                    "url": source.url,
                    "title": source.title,
                    "domain": source.domain,
                    "source_type": source.source_type,
                    "quality_score": round(
                        source.quality_score,
                        4,
                    ),
                }
            )

        return citations

    # ----------------------------------------------------------------------
    # Duplicate claim detection
    # ----------------------------------------------------------------------

    @classmethod
    def detect_duplicate_claims(
        cls,
        claims: Sequence[ResearchClaim],
    ) -> list[tuple[str, str]]:
        """
        Deteksi claim yang kemungkinan duplikat.
        """

        duplicates: list[
            tuple[str, str]
        ] = []

        normalized: list[
            tuple[str, str]
        ] = []

        for claim in claims:
            key = cls.normalize_for_similarity(
                claim.text
            )

            normalized.append(
                (
                    claim.claim_id,
                    key,
                )
            )

        for index, first in enumerate(
            normalized
        ):
            for second in normalized[
                index + 1:
            ]:
                similarity = cls.text_similarity(
                    first[1],
                    second[1],
                )

                if similarity >= 0.85:
                    duplicates.append(
                        (
                            first[0],
                            second[0],
                        )
                    )

        return duplicates

    # ----------------------------------------------------------------------
    # Similarity normalization
    # ----------------------------------------------------------------------

    @classmethod
    def normalize_for_similarity(
        cls,
        text: str,
    ) -> str:
        """
        Normalisasi text untuk similarity.
        """

        words = [
            word.lower()
            for word in WORD_PATTERN.findall(
                text
            )
            if len(word) >= 3
            and word.lower()
            not in INDONESIAN_STOPWORDS
        ]

        return " ".join(
            words
        )

    # ----------------------------------------------------------------------
    # Text similarity
    # ----------------------------------------------------------------------

    @staticmethod
    def text_similarity(
        first: str,
        second: str,
    ) -> float:
        """
        Jaccard-like similarity.
        """

        first_set = set(
            first.split()
        )

        second_set = set(
            second.split()
        )

        if not first_set and not second_set:
            return 1.0

        if not first_set or not second_set:
            return 0.0

        intersection = len(
            first_set
            & second_set
        )

        union = len(
            first_set
            | second_set
        )

        return (
            intersection / union
            if union
            else 0.0
        )

    # ----------------------------------------------------------------------
    # Conflict detection
    # ----------------------------------------------------------------------

    @classmethod
    def detect_conflicts(
        cls,
        claims: Sequence[ResearchClaim],
    ) -> list[dict[str, Any]]:
        """
        Mencari indikasi konflik sederhana.
        """

        conflicts: list[
            dict[str, Any]
        ] = []

        for index, first in enumerate(
            claims
        ):
            first_lower = first.text.lower()

            for second in claims[
                index + 1:
            ]:
                second_lower = (
                    second.text.lower()
                )

                same_subject = (
                    cls.shared_keyword_ratio(
                        first_lower,
                        second_lower,
                    )
                    >= 0.35
                )

                if not same_subject:
                    continue

                opposite_pairs = (
                    (
                        "meningkat",
                        "menurun",
                    ),
                    (
                        "increase",
                        "decrease",
                    ),
                    (
                        "naik",
                        "turun",
                    ),
                    (
                        "lebih tinggi",
                        "lebih rendah",
                    ),
                )

                for positive, negative in opposite_pairs:
                    if (
                        positive in first_lower
                        and negative in second_lower
                    ) or (
                        negative in first_lower
                        and positive in second_lower
                    ):
                        conflicts.append(
                            {
                                "claim_a": first.claim_id,
                                "claim_b": second.claim_id,
                                "type": "potential_conflict",
                                "reason": (
                                    "Claims menggunakan "
                                    "indikator berlawanan."
                                ),
                            }
                        )

                        break

        return conflicts

    # ----------------------------------------------------------------------
    # Shared keyword ratio
    # ----------------------------------------------------------------------

    @classmethod
    def shared_keyword_ratio(
        cls,
        first: str,
        second: str,
    ) -> float:
        """
        Rasio keyword yang sama.
        """

        first_words = set(
            cls.extract_keywords(
                first,
                limit=20,
            )
        )

        second_words = set(
            cls.extract_keywords(
                second,
                limit=20,
            )
        )

        if not first_words:
            return 0.0

        return len(
            first_words
            & second_words
        ) / len(
            first_words
        )

    # ----------------------------------------------------------------------
    # URL title
    # ----------------------------------------------------------------------

    @staticmethod
    def title_from_url(
        url: str,
    ) -> str:
        """
        Membuat title fallback dari URL.
        """

        parsed = urlparse(
            url
        )

        path = (
            parsed.path
            .strip("/")
            .split("/")
        )

        if path:
            last = path[-1]

            if last:
                last = unquote(
                    last
                )

                last = re.sub(
                    r"[-_]+",
                    " ",
                    last,
                )

                last = re.sub(
                    r"\.[A-Za-z0-9]+$",
                    "",
                    last,
                )

                if last:
                    return last.title()

        return (
            parsed.netloc
            or "Research Source"
        )

    # ----------------------------------------------------------------------
    # Related URLs
    # ----------------------------------------------------------------------

    @staticmethod
    def resolve_relative_url(
        base_url: str,
        relative_url: str,
    ) -> str:
        """
        Resolve relative URL.
        """

        return urljoin(
            base_url,
            relative_url,
        )

    # ----------------------------------------------------------------------
    # Query parameter extraction
    # ----------------------------------------------------------------------

    @staticmethod
    def extract_query_parameters(
        url: str,
    ) -> dict[str, list[str]]:
        """
        Extract query parameter dari URL.
        """

        parsed = urlparse(
            url
        )

        return {
            key: values
            for key, values in parse_qs(
                parsed.query
            ).items()
        }

    # ----------------------------------------------------------------------
    # Domain extraction
    # ----------------------------------------------------------------------

    @staticmethod
    def extract_domains(
        text: str,
    ) -> list[str]:
        """
        Extract domain.
        """

        domains: list[str] = []

        seen: set[str] = set()

        for domain in DOMAIN_PATTERN.findall(
            text
        ):
            value = domain.lower()

            if value in seen:
                continue

            seen.add(value)
            domains.append(
                value
            )

        return domains

    # ----------------------------------------------------------------------
    # Numbers
    # ----------------------------------------------------------------------

    @staticmethod
    def extract_numbers(
        text: str,
    ) -> list[str]:
        """
        Extract angka dari text.
        """

        return NUMBER_PATTERN.findall(
            text
        )

    # ----------------------------------------------------------------------
    # Years
    # ----------------------------------------------------------------------

    @staticmethod
    def extract_years(
        text: str,
    ) -> list[int]:
        """
        Extract tahun.
        """

        return [
            int(year)
            for year in YEAR_PATTERN.findall(
                text
            )
        ]

    # ----------------------------------------------------------------------
    # Research statistics
    # ----------------------------------------------------------------------

    @staticmethod
    def research_statistics(
        sources: Sequence[ResearchSource],
        claims: Sequence[ResearchClaim],
        evidence: Sequence[ResearchEvidence],
    ) -> dict[str, Any]:
        """
        Statistik research material.
        """

        quality_values = [
            source.quality_score
            for source in sources
        ]

        claim_values = [
            claim.confidence
            for claim in claims
        ]

        evidence_values = [
            item.strength_score
            for item in evidence
        ]

        return {
            "source_count": len(
                sources
            ),
            "claim_count": len(
                claims
            ),
            "evidence_count": len(
                evidence
            ),
            "average_source_quality": (
                round(
                    sum(
                        quality_values
                    ) / len(
                        quality_values
                    ),
                    4,
                )
                if quality_values
                else 0.0
            ),
            "average_claim_confidence": (
                round(
                    sum(
                        claim_values
                    ) / len(
                        claim_values
                    ),
                    4,
                )
                if claim_values
                else 0.0
            ),
            "average_evidence_strength": (
                round(
                    sum(
                        evidence_values
                    ) / len(
                        evidence_values
                    ),
                    4,
                )
                if evidence_values
                else 0.0
            ),
        }

    # ----------------------------------------------------------------------
    # Source ranking
    # ----------------------------------------------------------------------

    @staticmethod
    def rank_sources(
        sources: Sequence[ResearchSource],
    ) -> list[ResearchSource]:
        """
        Ranking source berdasarkan quality.
        """

        return sorted(
            sources,
            key=lambda source: (
                source.quality_score,
                source.authority_score,
                source.relevance_score,
                source.freshness_score,
            ),
            reverse=True,
        )

    # ----------------------------------------------------------------------
    # Source filtering
    # ----------------------------------------------------------------------

    @staticmethod
    def filter_high_quality_sources(
        sources: Sequence[ResearchSource],
        threshold: float = 0.70,
    ) -> list[ResearchSource]:
        """
        Filter source berkualitas tinggi.
        """

        return [
            source
            for source in sources
            if source.quality_score
            >= threshold
        ]

    # ----------------------------------------------------------------------
    # Research material validation
    # ----------------------------------------------------------------------

    @staticmethod
    def validate_material(
        text: str,
    ) -> dict[str, Any]:
        """
        Validasi material penelitian.
        """

        if text is None:
            return {
                "valid": False,
                "reason": "Material is None",
                "length": 0,
            }

        value = str(
            text
        )

        if not value.strip():
            return {
                "valid": False,
                "reason": "Material kosong",
                "length": 0,
            }

        if len(value) > MAX_TEXT_LENGTH:
            return {
                "valid": False,
                "reason": "Material terlalu panjang",
                "length": len(value),
            }

        return {
            "valid": True,
            "reason": "",
            "length": len(value),
        }

    # ----------------------------------------------------------------------
    # Safe JSON serialization
    # ----------------------------------------------------------------------

    @staticmethod
    def to_json(
        value: Any,
    ) -> str:
        """
        Serialize object secara aman.
        """

        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # ----------------------------------------------------------------------
    # ISO timestamp
    # ----------------------------------------------------------------------

    @staticmethod
    def now_iso() -> str:
        """
        UTC timestamp.
        """

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ----------------------------------------------------------------------
    # Async research preparation
    # ----------------------------------------------------------------------

    async def prepare_research(
        self,
        task: str,
    ) -> ResearchPlan:
        """
        Async wrapper untuk research planning.
        """

        await asyncio.sleep(
            0
        )

        plan = self.create_plan(
            self.normalize_task(
                task
            )
        )

        self.last_plan = plan

        self.plan_count += 1

        return plan

    # ----------------------------------------------------------------------
    # Async source analysis
    # ----------------------------------------------------------------------

    async def analyze_sources(
        self,
        sources: Any,
        task: str = "",
    ) -> list[ResearchSource]:
        """
        Async source analysis.
        """

        await asyncio.sleep(
            0
        )

        normalized = self.normalize_sources(
            sources
        )

        keywords = self.extract_keywords(
            task
        )

        for source in normalized:
            source.relevance_score = (
                self.score_source_relevance(
                    source,
                    keywords,
                )
            )

            source.quality_score = (
                self.calculate_source_quality(
                    source
                )
            )

        return self.rank_sources(
            normalized
        )

    # ----------------------------------------------------------------------
    # Async synthesis
    # ----------------------------------------------------------------------

    async def synthesize(
        self,
        task: str,
        *,
        sources: Any = None,
        text: str = "",
    ) -> dict[str, Any]:
        """
        Async research synthesis.
        """

        started = time.perf_counter()

        normalized_task = self.normalize_task(
            task
        )

        source_objects = self.normalize_sources(
            sources
        )

        keywords = self.extract_keywords(
            normalized_task
        )

        if text:
            cleaned = self.clean_text(
                text
            )

            evidence = self.extract_evidence(
                cleaned,
                source_objects,
                keywords,
            )

            claims = self.extract_claims(
                cleaned,
                source_objects,
                evidence,
            )

        else:
            evidence = []

            claims = []

        confidence = (
            self.calculate_research_confidence(
                task=normalized_task,
                sources=source_objects,
                claims=claims,
                evidence=evidence,
            )
        )

        conflicts = (
            self.detect_conflicts(
                claims
            )
        )

        duplicates = (
            self.detect_duplicate_claims(
                claims
            )
        )

        statistics = (
            self.research_statistics(
                source_objects,
                claims,
                evidence,
            )
        )

        self.synthesis_count += 1

        latency_ms = round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            2,
        )

        return {
            "success": True,
            "task": normalized_task,
            "confidence": round(
                confidence,
                4,
            ),
            "sources": [
                source.to_dict()
                for source in source_objects
            ],
            "claims": [
                claim.to_dict()
                for claim in claims
            ],
            "evidence": [
                item.to_dict()
                for item in evidence
            ],
            "conflicts": conflicts,
            "duplicate_claims": duplicates,
            "statistics": statistics,
            "citations": self.prepare_citations(
                source_objects
            ),
            "latency_ms": latency_ms,
        }

    # ----------------------------------------------------------------------
    # Research summary
    # ----------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """
        Summary agent.
        """

        return {
            "agent": self.name,
            "version": self.version,
            "status": "READY",
            "research_count": self.research_count,
            "plan_count": self.plan_count,
            "synthesis_count": self.synthesis_count,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "source_cache_size": len(
                self._source_cache
            ),
        }

    # ----------------------------------------------------------------------
    # Cache management
    # ----------------------------------------------------------------------

    def clear_source_cache(self) -> None:
        """
        Bersihkan cache source.
        """

        self._source_cache.clear()

    # ----------------------------------------------------------------------
    # Source cache inspection
    # ----------------------------------------------------------------------

    def cached_sources(self) -> list[dict[str, Any]]:
        """
        Return cache source.
        """

        return [
            source.to_dict()
            for source in self._source_cache.values()
        ]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ResearchAgent",
    "ResearchSource",
    "ResearchClaim",
    "ResearchEvidence",
    "ResearchPlan",
]