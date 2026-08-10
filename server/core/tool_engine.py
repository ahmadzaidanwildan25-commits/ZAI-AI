from __future__ import annotations

import ast
import html
import ipaddress
import math
import operator
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import (
    parse_qs,
    quote_plus,
    unquote,
    urljoin,
    urlparse,
)

import httpx


# ============================================================
# SUPER ZAI - TOOL ENGINE
# Version 1.7.0
# ============================================================

ENGINE_NAME = "ToolEngine"
ENGINE_VERSION = "1.7.0"


# ============================================================
# TOOL RESULT
# ============================================================

@dataclass
class ToolResult:
    success: bool
    tool: str
    response: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tool": self.tool,
            "response": self.response,
            "data": self.data,
            "error": self.error,
        }


# ============================================================
# CACHE ENTRY
# ============================================================

@dataclass
class CacheEntry:
    value: Any
    created_at: float
    ttl: float

    def valid(self) -> bool:
        return (time.monotonic() - self.created_at) < self.ttl


# ============================================================
# SAFE CACHE
# ============================================================

class TTLCache:
    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self._items: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._items.get(key)

            if entry is None:
                return None

            if not entry.valid():
                self._items.pop(key, None)
                return None

            return entry.value

    def set(self, key: str, value: Any, ttl: float) -> None:
        with self._lock:
            self._items[key] = CacheEntry(
                value=value,
                created_at=time.monotonic(),
                ttl=ttl,
            )

            self._cleanup()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def size(self) -> int:
        with self._lock:
            self._cleanup()
            return len(self._items)

    def _cleanup(self) -> None:
        now = time.monotonic()

        expired = [
            key
            for key, entry in self._items.items()
            if now - entry.created_at >= entry.ttl
        ]

        for key in expired:
            self._items.pop(key, None)

        if len(self._items) <= self.max_entries:
            return

        ordered = sorted(
            self._items.items(),
            key=lambda item: item[1].created_at,
        )

        overflow = len(self._items) - self.max_entries

        for key, _ in ordered[:overflow]:
            self._items.pop(key, None)


# ============================================================
# SAFE CALCULATOR
# ============================================================

class SafeCalculator:
    """
    AST-based calculator.

    Supported:
        + - * / // % **
        unary + -
        parentheses
        common math functions
        constants pi/e/tau
    """

    MAX_EXPRESSION_LENGTH = 300

    ALLOWED_BINARY_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    ALLOWED_UNARY_OPERATORS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    FUNCTIONS: Dict[str, Callable[..., float]] = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "fabs": math.fabs,
        "floor": math.floor,
        "ceil": math.ceil,
        "degrees": math.degrees,
        "radians": math.radians,
    }

    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }

    def evaluate(self, expression: str) -> float:
        if not isinstance(expression, str):
            raise ValueError("Ekspresi harus berupa teks.")

        expression = expression.strip()

        if not expression:
            raise ValueError("Ekspresi kosong.")

        if len(expression) > self.MAX_EXPRESSION_LENGTH:
            raise ValueError("Ekspresi terlalu panjang.")

        tree = ast.parse(expression, mode="eval")

        result = self._eval(tree.body)

        if isinstance(result, bool):
            raise ValueError("Boolean tidak diperbolehkan.")

        if not math.isfinite(float(result)):
            raise ValueError("Hasil tidak valid atau tidak terbatas.")

        return result

    def _eval(self, node: ast.AST) -> float:

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                if isinstance(node.value, bool):
                    raise ValueError("Boolean tidak diperbolehkan.")

                value = float(node.value)

                if not math.isfinite(value):
                    raise ValueError("Angka tidak valid.")

                return value

            raise ValueError("Tipe konstanta tidak diperbolehkan.")

        if isinstance(node, ast.Num):
            return float(node.n)

        if isinstance(node, ast.Name):
            if node.id in self.CONSTANTS:
                return self.CONSTANTS[node.id]

            raise ValueError(
                f"Nama '{node.id}' tidak diperbolehkan."
            )

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)

            if operator_type not in self.ALLOWED_BINARY_OPERATORS:
                raise ValueError("Operator tidak diperbolehkan.")

            left = self._eval(node.left)
            right = self._eval(node.right)

            if operator_type is ast.Pow:
                if abs(right) > 1000:
                    raise ValueError(
                        "Eksponen terlalu besar."
                    )

            operation = self.ALLOWED_BINARY_OPERATORS[
                operator_type
            ]

            try:
                return operation(left, right)
            except ZeroDivisionError:
                raise ValueError("Tidak dapat membagi dengan nol.")
            except OverflowError:
                raise ValueError("Hasil terlalu besar.")

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)

            if operator_type not in self.ALLOWED_UNARY_OPERATORS:
                raise ValueError("Operator unary tidak diperbolehkan.")

            operand = self._eval(node.operand)

            return self.ALLOWED_UNARY_OPERATORS[
                operator_type
            ](operand)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Pemanggilan fungsi tidak diperbolehkan.")

            function_name = node.func.id

            if function_name not in self.FUNCTIONS:
                raise ValueError(
                    f"Fungsi '{function_name}' tidak diperbolehkan."
                )

            if node.keywords:
                raise ValueError(
                    "Keyword arguments tidak diperbolehkan."
                )

            args = [
                self._eval(argument)
                for argument in node.args
            ]

            try:
                return self.FUNCTIONS[function_name](*args)
            except Exception as exc:
                raise ValueError(
                    f"Perhitungan fungsi gagal: {exc}"
                )

        raise ValueError(
            f"Ekspresi '{type(node).__name__}' tidak diperbolehkan."
        )


# ============================================================
# TOOL ENGINE
# ============================================================

class ToolEngine:

    def __init__(self):

        self.engine = ENGINE_NAME
        self.version = ENGINE_VERSION

        # ----------------------------------------------------
        # TOOLS
        # ----------------------------------------------------

        self.tools = [
            "calculator",
            "weather",
            "search",
            "fetch",
        ]

        self.calculator_enabled = True
        self.weather_enabled = True
        self.search_enabled = True
        self.fetch_enabled = True

        # ----------------------------------------------------
        # PROVIDERS
        # ----------------------------------------------------

        self.weather_provider = "open-meteo"

        self.search_provider = ["bing"]
        self.search_fallback = False
        self.search_method = "GET"

        # ----------------------------------------------------
        # SEARCH CONFIG
        # ----------------------------------------------------

        self.search_result_limit = 5
        self.search_retries = 2

        self.search_connect_timeout = 10.0
        self.search_read_timeout = 15.0

        # ----------------------------------------------------
        # FETCH CONFIG
        # ----------------------------------------------------

        self.fetch_connect_timeout = 10.0
        self.fetch_read_timeout = 20.0

        self.fetch_max_bytes = 2_000_000
        self.fetch_max_text = 20_000

        self.fetch_max_redirects = 5

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        self.search_cache_enabled = True
        self.search_cache_ttl = 120
        self.search_cache_max_entries = 100

        self.fetch_cache_enabled = True
        self.fetch_cache_ttl = 120
        self.fetch_cache_max_entries = 100

        self.search_cache = TTLCache(
            max_entries=self.search_cache_max_entries
        )

        self.fetch_cache = TTLCache(
            max_entries=self.fetch_cache_max_entries
        )

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        self.timeout_protection = True
        self.ast_evaluator = True
        self.controlled_provider_failure = True

        self.bing_redirect_decoder = True
        self.bing_html_parser = True

        self.url_normalization = True
        self.search_deduplication = True

        self.web_fetch = True
        self.html_text_extraction = True

        self.ssrf_protection = True
        self.redirect_protection = True

        # ----------------------------------------------------
        # CALCULATOR
        # ----------------------------------------------------

        self.calculator = SafeCalculator()

        # ----------------------------------------------------
        # HTTP CLIENT
        # ----------------------------------------------------

        self.user_agent = (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/151.0.0.0 Safari/537.36 "
            "Super-ZAI/1.7.0"
        )

        # ----------------------------------------------------
        # RUNTIME STATS
        # ----------------------------------------------------

        self._stats_lock = threading.RLock()

        self.runtime = {
            "calculator_calls": 0,
            "weather_calls": 0,
            "search_calls": 0,
            "fetch_calls": 0,

            "calculator_success": 0,
            "weather_success": 0,
            "search_success": 0,
            "fetch_success": 0,

            "calculator_failures": 0,
            "weather_failures": 0,
            "search_failures": 0,
            "fetch_failures": 0,

            "search_cache_hits": 0,
            "search_cache_misses": 0,

            "fetch_cache_hits": 0,
            "fetch_cache_misses": 0,

            "search_results_total": 0,
            "fetch_bytes_total": 0,

            "search_latency_total_ms": 0.0,
            "fetch_latency_total_ms": 0.0,
            "weather_latency_total_ms": 0.0,
            "calculator_latency_total_ms": 0.0,

            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
        }

    # ========================================================
    # PUBLIC EXECUTE
    # ========================================================

    def execute(
        self,
        tool: str,
        query: str,
    ) -> ToolResult:

        tool = str(tool or "").strip().lower()

        with self._stats_lock:
            self.runtime["requests_total"] += 1

        try:

            if tool == "calculator":
                return self._execute_calculator(query)

            if tool == "weather":
                return self._execute_weather(query)

            if tool == "search":
                return self._execute_search(query)

            if tool == "fetch":
                return self._execute_fetch(query)

            result = ToolResult(
                success=False,
                tool=tool,
                response=f"Tool '{tool}' tidak tersedia.",
                data={
                    "available_tools": self.tools,
                },
                error="unknown_tool",
            )

            with self._stats_lock:
                self.runtime["requests_failed"] += 1

            return result

        except Exception as exc:

            with self._stats_lock:
                self.runtime["requests_failed"] += 1

            return ToolResult(
                success=False,
                tool=tool,
                response="Tool gagal diproses.",
                data={},
                error=str(exc),
            )

    # ========================================================
    # CALCULATOR
    # ========================================================

    def _execute_calculator(
        self,
        query: str,
    ) -> ToolResult:

        start = time.perf_counter()

        with self._stats_lock:
            self.runtime["calculator_calls"] += 1

        try:

            expression = self._extract_calculator_expression(
                query
            )

            result = self.calculator.evaluate(expression)

            formatted = self._format_number(result)

            latency = self._latency_ms(start)

            with self._stats_lock:
                self.runtime["calculator_success"] += 1
                self.runtime[
                    "calculator_latency_total_ms"
                ] += latency
                self.runtime["requests_success"] += 1

            return ToolResult(
                success=True,
                tool="calculator",
                response=f"Hasilnya adalah {formatted}.",
                data={
                    "expression": expression,
                    "result": result,
                    "safe": True,
                    "latency_ms": latency,
                },
            )

        except Exception as exc:

            latency = self._latency_ms(start)

            with self._stats_lock:
                self.runtime["calculator_failures"] += 1
                self.runtime[
                    "calculator_latency_total_ms"
                ] += latency

            return ToolResult(
                success=False,
                tool="calculator",
                response="Perhitungan tidak dapat diproses.",
                data={
                    "safe": True,
                    "latency_ms": latency,
                },
                error=str(exc),
            )

    def _extract_calculator_expression(
        self,
        query: str,
    ) -> str:

        text = str(query or "").strip()

        prefixes = [
            "hitung",
            "calculate",
            "calculator",
            "berapa",
            "hasil",
        ]

        lowered = text.lower()

        for prefix in prefixes:
            if lowered.startswith(prefix):
                text = text[len(prefix):].strip()
                break

        text = text.replace("×", "*")
        text = text.replace("÷", "/")
        text = text.replace("^", "**")

        return text

    # ========================================================
    # WEATHER
    # ========================================================

    def _execute_weather(
        self,
        query: str,
    ) -> ToolResult:

        start = time.perf_counter()

        with self._stats_lock:
            self.runtime["weather_calls"] += 1

        try:

            city = self._extract_city(query)

            if not city:
                raise ValueError(
                    "Kota cuaca tidak ditemukan."
                )

            coordinates = self._geocode_city(city)

            latitude = coordinates["latitude"]
            longitude = coordinates["longitude"]

            url = "https://api.open-meteo.com/v1/forecast"

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "apparent_temperature,"
                    "relative_humidity_2m,"
                    "precipitation,"
                    "wind_speed_10m,"
                    "weather_code"
                ),
                "timezone": "auto",
            }

            timeout = httpx.Timeout(
                connect=10.0,
                read=15.0,
                write=10.0,
                pool=10.0,
            )

            response = httpx.get(
                url,
                params=params,
                timeout=timeout,
                headers={
                    "User-Agent": self.user_agent,
                },
                follow_redirects=True,
            )

            response.raise_for_status()

            payload = response.json()

            current = payload.get("current", {})

            temperature = current.get(
                "temperature_2m"
            )

            feels_like = current.get(
                "apparent_temperature"
            )

            humidity = current.get(
                "relative_humidity_2m"
            )

            precipitation = current.get(
                "precipitation"
            )

            wind = current.get(
                "wind_speed_10m"
            )

            weather_code = current.get(
                "weather_code"
            )

            description = self._weather_description(
                weather_code
            )

            latency = self._latency_ms(start)

            response_text = (
                f"Cuaca di {coordinates['city']}: "
                f"{description}. "
                f"Suhu {temperature}°C, "
                f"terasa seperti {feels_like}°C, "
                f"kelembapan {humidity}%, "
                f"angin {wind} km/jam. "
                f"Curah hujan saat ini "
                f"{precipitation} mm."
            )

            data = {
                "city": coordinates["city"],
                "country": coordinates["country"],
                "latitude": latitude,
                "longitude": longitude,
                "temperature_c": temperature,
                "feels_like_c": feels_like,
                "humidity_percent": humidity,
                "precipitation_mm": precipitation,
                "wind_kmh": wind,
                "weather_code": weather_code,
                "description": description,
                "latency_ms": latency,
                "provider": self.weather_provider,
            }

            with self._stats_lock:
                self.runtime["weather_success"] += 1
                self.runtime[
                    "weather_latency_total_ms"
                ] += latency
                self.runtime["requests_success"] += 1

            return ToolResult(
                success=True,
                tool="weather",
                response=response_text,
                data=data,
            )

        except Exception as exc:

            latency = self._latency_ms(start)

            with self._stats_lock:
                self.runtime["weather_failures"] += 1
                self.runtime[
                    "weather_latency_total_ms"
                ] += latency

            return ToolResult(
                success=False,
                tool="weather",
                response="Data cuaca tidak dapat diperoleh.",
                data={
                    "latency_ms": latency,
                    "provider": self.weather_provider,
                },
                error=str(exc),
            )

    def _extract_city(
        self,
        query: str,
    ) -> str:

        text = str(query or "").strip()

        prefixes = [
            "cuaca",
            "weather",
            "cek cuaca",
            "cek weather",
        ]

        lowered = text.lower()

        for prefix in prefixes:
            if lowered.startswith(prefix):
                text = text[len(prefix):].strip()
                break

        return text.strip()

    def _geocode_city(
        self,
        city: str,
    ) -> Dict[str, Any]:

        url = "https://geocoding-api.open-meteo.com/v1/search"

        params = {
            "name": city,
            "count": 1,
            "language": "id",
            "format": "json",
        }

        timeout = httpx.Timeout(
            connect=10.0,
            read=15.0,
            write=10.0,
            pool=10.0,
        )

        response = httpx.get(
            url,
            params=params,
            timeout=timeout,
            headers={
                "User-Agent": self.user_agent,
            },
        )

        response.raise_for_status()

        payload = response.json()

        results = payload.get("results") or []

        if not results:
            raise ValueError(
                f"Kota '{city}' tidak ditemukan."
            )

        result = results[0]

        return {
            "city": result.get("name") or city,
            "country": result.get(
                "country",
                "Unknown",
            ),
            "latitude": result["latitude"],
            "longitude": result["longitude"],
        }

    def _weather_description(
        self,
        code: Any,
    ) -> str:

        mapping = {
            0: "cerah",
            1: "cerah sebagian",
            2: "berawan sebagian",
            3: "mendung",
            45: "berkabut",
            48: "kabut tebal",
            51: "gerimis ringan",
            53: "gerimis sedang",
            55: "gerimis lebat",
            56: "gerimis beku ringan",
            57: "gerimis beku lebat",
            61: "hujan ringan",
            63: "hujan sedang",
            65: "hujan lebat",
            66: "hujan beku ringan",
            67: "hujan beku lebat",
            71: "salju ringan",
            73: "salju sedang",
            75: "salju lebat",
            77: "butiran salju",
            80: "hujan ringan",
            81: "hujan sedang",
            82: "hujan lebat",
            85: "salju ringan",
            86: "salju lebat",
            95: "badai petir",
            96: "badai petir dengan hujan es",
            99: "badai petir kuat dengan hujan es",
        }

        try:
            return mapping.get(
                int(code),
                "kondisi cuaca tidak diketahui",
            )
        except Exception:
            return "kondisi cuaca tidak diketahui"

    # ========================================================
    # SEARCH
    # ========================================================

    def _execute_search(
        self,
        query: str,
    ) -> ToolResult:

        start = time.perf_counter()

        with self._stats_lock:
            self.runtime["search_calls"] += 1

        clean_query = self._clean_search_query(query)

        if not clean_query:
            return ToolResult(
                success=False,
                tool="search",
                response="Query pencarian kosong.",
                error="empty_query",
            )

        cache_key = self._cache_key(
            "search",
            clean_query,
        )

        if self.search_cache_enabled:

            cached = self.search_cache.get(cache_key)

            if cached is not None:

                with self._stats_lock:
                    self.runtime["search_cache_hits"] += 1

                cached_result = dict(cached)

                cached_result["cache_hit"] = True

                latency = self._latency_ms(start)

                cached_result["latency_ms"] = latency

                return ToolResult(
                    success=True,
                    tool="search",
                    response=self._format_search_response(
                        clean_query,
                        cached_result["results"],
                    ),
                    data=cached_result,
                )

            with self._stats_lock:
                self.runtime["search_cache_misses"] += 1

        try:

            result = self._bing_search(
                clean_query
            )

            if self.search_cache_enabled:
                self.search_cache.set(
                    cache_key,
                    result,
                    self.search_cache_ttl,
                )

            latency = self._latency_ms(start)

            result = dict(result)
            result["latency_ms"] = latency
            result["cache_hit"] = False

            with self._stats_lock:
                self.runtime["search_success"] += 1
                self.runtime[
                    "search_results_total"
                ] += result.get("count", 0)
                self.runtime[
                    "search_latency_total_ms"
                ] += latency
                self.runtime["requests_success"] += 1

            return ToolResult(
                success=True,
                tool="search",
                response=self._format_search_response(
                    clean_query,
                    result["results"],
                ),
                data=result,
            )

        except Exception as exc:

            latency = self._latency_ms(start)

            with self._stats_lock:
                self.runtime["search_failures"] += 1
                self.runtime[
                    "search_latency_total_ms"
                ] += latency

            return ToolResult(
                success=False,
                tool="search",
                response="Pencarian gagal dilakukan.",
                data={
                    "query": clean_query,
                    "results": [],
                    "count": 0,
                    "provider": "bing",
                    "latency_ms": latency,
                },
                error=str(exc),
            )

    def _clean_search_query(
        self,
        query: str,
    ) -> str:

        text = str(query or "").strip()

        prefixes = [
            "cari",
            "search",
            "carikan",
            "tolong cari",
            "cari informasi",
            "cari info",
        ]

        lowered = text.lower()

        for prefix in prefixes:
            if lowered.startswith(prefix):
                text = text[len(prefix):].strip()
                break

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

    def _bing_search(
        self,
        query: str,
    ) -> Dict[str, Any]:

        url = "https://www.bing.com/search"

        params = {
            "q": query,
            "count": self.search_result_limit,
            "setlang": "id",
        }

        timeout = httpx.Timeout(
            connect=self.search_connect_timeout,
            read=self.search_read_timeout,
            write=10.0,
            pool=10.0,
        )

        last_error: Optional[Exception] = None

        for attempt in range(
            1,
            self.search_retries + 1,
        ):

            try:

                response = httpx.get(
                    url,
                    params=params,
                    timeout=timeout,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": (
                            "text/html,"
                            "application/xhtml+xml,"
                            "application/xml;q=0.9,"
                            "*/*;q=0.8"
                        ),
                        "Accept-Language": (
                            "id-ID,id;q=0.9,en;q=0.8"
                        ),
                    },
                    follow_redirects=True,
                )

                response.raise_for_status()

                results = self._parse_bing_results(
                    response.text
                )

                return {
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "status_code": response.status_code,
                    "attempt": attempt,
                    "final_url": str(response.url),
                    "html_length": len(response.text),
                    "provider": "bing",
                }

            except Exception as exc:

                last_error = exc

                if attempt < self.search_retries:
                    time.sleep(0.25 * attempt)

        raise RuntimeError(
            f"Bing search gagal: {last_error}"
        )

    # ========================================================
    # BING PARSER
    # ========================================================

    def _parse_bing_results(
        self,
        raw_html: str,
    ) -> List[Dict[str, str]]:

        from html.parser import HTMLParser

        class BingParser(HTMLParser):

            def __init__(self):
                super().__init__(
                    convert_charrefs=True
                )

                self.results = []

                self.current_result = None
                self.current_tag = None

                self.in_li = False
                self.in_title = False
                self.in_snippet = False

            def handle_starttag(
                self,
                tag,
                attrs,
            ):

                attrs_dict = dict(attrs)

                classes = (
                    attrs_dict.get(
                        "class",
                        "",
                    )
                    or ""
                )

                href = attrs_dict.get(
                    "href"
                )

                if tag == "li" and (
                    "b_algo" in classes
                    or "b_ans" in classes
                ):

                    self.in_li = True

                    self.current_result = {
                        "title": "",
                        "url": "",
                        "snippet": "",
                    }

                if not self.in_li:
                    return

                if tag == "h2":
                    self.in_title = True

                if tag == "p":
                    self.in_snippet = True

                if tag == "a" and href:
                    if (
                        not self.current_result["url"]
                        and self._looks_like_result_link(
                            href
                        )
                    ):
                        self.current_result["url"] = href

            def handle_endtag(
                self,
                tag,
            ):

                if tag == "h2":
                    self.in_title = False

                if tag == "p":
                    self.in_snippet = False

                if tag == "li" and self.in_li:

                    result = self.current_result

                    if result:

                        title = self._clean_text(
                            result["title"]
                        )

                        snippet = self._clean_text(
                            result["snippet"]
                        )

                        url = self._clean_url(
                            result["url"]
                        )

                        if (
                            title
                            and url
                            and self._is_http_url(url)
                        ):

                            self.results.append(
                                {
                                    "title": title,
                                    "url": url,
                                    "snippet": snippet,
                                    "domain": self._domain(
                                        url
                                    ),
                                }
                            )

                    self.current_result = None
                    self.in_li = False

            def handle_data(
                self,
                data,
            ):

                if not self.in_li:
                    return

                if self.current_result is None:
                    return

                if self.in_title:
                    self.current_result["title"] += (
                        " " + data
                    )

                elif self.in_snippet:
                    self.current_result["snippet"] += (
                        " " + data
                    )

            @staticmethod
            def _clean_text(
                value: str,
            ) -> str:

                value = html.unescape(
                    value or ""
                )

                value = re.sub(
                    r"\s+",
                    " ",
                    value,
                )

                return value.strip()

            @staticmethod
            def _looks_like_result_link(
                href: str,
            ) -> bool:

                return (
                    href.startswith("http://")
                    or href.startswith("https://")
                    or href.startswith("/ck/")
                )

            @staticmethod
            def _clean_url(
                href: str,
            ) -> str:

                return ToolEngine._normalize_url(
                    ToolEngine._decode_bing_redirect(
                        href
                    )
                )

            @staticmethod
            def _is_http_url(
                href: str,
            ) -> bool:

                parsed = urlparse(href)

                return parsed.scheme in {
                    "http",
                    "https",
                }

            @staticmethod
            def _domain(
                href: str,
            ) -> str:

                return (
                    urlparse(href)
                    .netloc
                    .lower()
                    .split("@")[-1]
                    .split(":")[0]
                )

        parser = BingParser()

        parser.feed(raw_html)

        results = parser.results

        # ----------------------------------------------------
        # SECONDARY REGEX FALLBACK
        # ----------------------------------------------------

        if not results:

            results = self._regex_bing_fallback(
                raw_html
            )

        # ----------------------------------------------------
        # DEDUPLICATION
        # ----------------------------------------------------

        if self.search_deduplication:

            unique = []
            seen = set()

            for result in results:

                normalized = self._normalize_url(
                    result.get("url", "")
                )

                if not normalized:
                    continue

                key = normalized.lower()

                if key in seen:
                    continue

                seen.add(key)

                result["url"] = normalized

                result["domain"] = self._domain(
                    normalized
                )

                unique.append(result)

                if len(unique) >= self.search_result_limit:
                    break

            results = unique

        return results[
            : self.search_result_limit
        ]

    def _regex_bing_fallback(
        self,
        raw_html: str,
    ) -> List[Dict[str, str]]:

        results = []

        pattern = re.compile(
            r'<li[^>]+class="[^"]*b_algo[^"]*"'
            r'[^>]*>(.*?)</li>',
            re.I | re.S,
        )

        blocks = pattern.findall(
            raw_html
        )

        for block in blocks:

            title_match = re.search(
                r"<h2[^>]*>"
                r"(.*?)"
                r"</h2>",
                block,
                re.I | re.S,
            )

            href_match = re.search(
                r'<a[^>]+href="([^"]+)"',
                block,
                re.I | re.S,
            )

            snippet_match = re.search(
                r"<p[^>]*>"
                r"(.*?)"
                r"</p>",
                block,
                re.I | re.S,
            )

            if not title_match:
                continue

            if not href_match:
                continue

            title = self._strip_html(
                title_match.group(1)
            )

            href = self._normalize_url(
                self._decode_bing_redirect(
                    html.unescape(
                        href_match.group(1)
                    )
                )
            )

            snippet = ""

            if snippet_match:
                snippet = self._strip_html(
                    snippet_match.group(1)
                )

            if not self._is_http_url(href):
                continue

            results.append(
                {
                    "title": title,
                    "url": href,
                    "snippet": snippet,
                    "domain": self._domain(href),
                }
            )

            if len(results) >= self.search_result_limit:
                break

        return results

    # ========================================================
    # BING REDIRECT DECODER
    # ========================================================

    @staticmethod
    def _decode_bing_redirect(
        url: str,
    ) -> str:

        if not url:
            return ""

        url = html.unescape(
            url
        )

        parsed = urlparse(url)

        # Already direct
        if parsed.scheme in {
            "http",
            "https",
        } and (
            parsed.netloc.lower()
            not in {
                "www.bing.com",
                "bing.com",
            }
        ):

            return url

        # Relative Bing redirect
        if url.startswith("/ck/"):
            url = urljoin(
                "https://www.bing.com",
                url,
            )
            parsed = urlparse(url)

        if parsed.netloc.lower() not in {
            "www.bing.com",
            "bing.com",
        }:
            return url

        params = parse_qs(
            parsed.query
        )

        encoded_targets = params.get(
            "u",
            []
        )

        if not encoded_targets:
            return url

        encoded = encoded_targets[0]

        try:

            decoded = unquote(
                encoded
            )

            # Bing sometimes uses a1 + base64 style.
            if decoded.startswith("a1"):
                import base64

                payload = decoded[2:]

                try:
                    padding = "=" * (
                        (-len(payload)) % 4
                    )

                    decoded = base64.urlsafe_b64decode(
                        payload + padding
                    ).decode(
                        "utf-8",
                        errors="ignore",
                    )

                except Exception:
                    pass

            if decoded.startswith(
                ("http://", "https://")
            ):
                return decoded

        except Exception:
            pass

        return url

    # ========================================================
    # SEARCH RESPONSE
    # ========================================================

    def _format_search_response(
        self,
        query: str,
        results: List[Dict[str, str]],
    ) -> str:

        if not results:
            return (
                f"Tidak ditemukan hasil untuk: "
                f"{query}"
            )

        lines = [
            f"Hasil pencarian untuk: {query}",
            "",
        ]

        for index, result in enumerate(
            results,
            start=1,
        ):

            title = result.get(
                "title",
                "Tanpa judul",
            )

            domain = result.get(
                "domain",
                "",
            )

            snippet = result.get(
                "snippet",
                "",
            )

            url = result.get(
                "url",
                "",
            )

            lines.append(
                f"{index}. {title}"
            )

            if domain:
                lines.append(
                    f"   Sumber: {domain}"
                )

            if snippet:
                lines.append(
                    f"   {snippet}"
                )

            if url:
                lines.append(
                    f"   {url}"
                )

            lines.append("")

        return "\n".join(
            lines
        ).strip()

    # ========================================================
    # FETCH
    # ========================================================

    def _execute_fetch(
        self,
        query: str,
    ) -> ToolResult:

        start = time.perf_counter()

        with self._stats_lock:
            self.runtime["fetch_calls"] += 1

        try:

            url = self._extract_url(
                query
            )

            if not url:
                raise ValueError(
                    "URL tidak ditemukan."
                )

            url = self._normalize_url(
                url
            )

            self._validate_url(
                url
            )

            cache_key = self._cache_key(
                "fetch",
                url,
            )

            if self.fetch_cache_enabled:

                cached = self.fetch_cache.get(
                    cache_key
                )

                if cached is not None:

                    with self._stats_lock:
                        self.runtime[
                            "fetch_cache_hits"
                        ] += 1

                    cached_data = dict(
                        cached
                    )

                    cached_data[
                        "cache_hit"
                    ] = True

                    latency = self._latency_ms(
                        start
                    )

                    cached_data[
                        "latency_ms"
                    ] = latency

                    return ToolResult(
                        success=True,
                        tool="fetch",
                        response=self._format_fetch_response(
                            cached_data
                        ),
                        data=cached_data,
                    )

                with self._stats_lock:
                    self.runtime[
                        "fetch_cache_misses"
                    ] += 1

            result = self._fetch_url(
                url
            )

            if self.fetch_cache_enabled:
                self.fetch_cache.set(
                    cache_key,
                    result,
                    self.fetch_cache_ttl,
                )

            latency = self._latency_ms(
                start
            )

            result = dict(result)

            result[
                "latency_ms"
            ] = latency

            result[
                "cache_hit"
            ] = False

            with self._stats_lock:
                self.runtime[
                    "fetch_success"
                ] += 1

                self.runtime[
                    "fetch_bytes_total"
                ] += result.get(
                    "bytes",
                    0,
                )

                self.runtime[
                    "fetch_latency_total_ms"
                ] += latency

                self.runtime[
                    "requests_success"
                ] += 1

            return ToolResult(
                success=True,
                tool="fetch",
                response=self._format_fetch_response(
                    result
                ),
                data=result,
            )

        except Exception as exc:

            latency = self._latency_ms(
                start
            )

            with self._stats_lock:
                self.runtime[
                    "fetch_failures"
                ] += 1

                self.runtime[
                    "fetch_latency_total_ms"
                ] += latency

            return ToolResult(
                success=False,
                tool="fetch",
                response="Halaman tidak dapat diambil.",
                data={
                    "latency_ms": latency,
                },
                error=str(exc),
            )

    def _extract_url(
        self,
        query: str,
    ) -> str:

        text = str(
            query or ""
        ).strip()

        # Markdown URL
        markdown = re.search(
            r"\[[^\]]+\]"
            r"\(\s*(https?://[^)\s]+)"
            r"\s*\)",
            text,
            re.I,
        )

        if markdown:
            return markdown.group(1)

        # Plain URL
        match = re.search(
            r"https?://[^\s<>\"]+",
            text,
            re.I,
        )

        if match:
            return match.group(0).rstrip(
                ".,);]}>"
            )

        return ""

    def _fetch_url(
        self,
        url: str,
    ) -> Dict[str, Any]:

        timeout = httpx.Timeout(
            connect=self.fetch_connect_timeout,
            read=self.fetch_read_timeout,
            write=10.0,
            pool=10.0,
        )

        headers = {
            "User-Agent": self.user_agent,
            "Accept": (
                "text/html,"
                "application/xhtml+xml,"
                "application/json,"
                "text/plain,"
                "*/*;q=0.8"
            ),
            "Accept-Language": (
                "id-ID,id;q=0.9,en;q=0.8"
            ),
        }

        with httpx.Client(
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
            max_redirects=self.fetch_max_redirects,
        ) as client:

            response = client.get(
                url
            )

            response.raise_for_status()

            final_url = self._normalize_url(
                str(response.url)
            )

            self._validate_url(
                final_url
            )

            raw_bytes = response.content

            if len(raw_bytes) > self.fetch_max_bytes:
                raw_bytes = raw_bytes[
                    : self.fetch_max_bytes
                ]

                truncated_bytes = True
            else:
                truncated_bytes = False

            content_type = (
                response.headers.get(
                    "content-type",
                    "",
                )
                .lower()
            )

            text = self._decode_response_text(
                raw_bytes,
                response.headers.get(
                    "content-type",
                    "",
                ),
            )

            is_html = (
                "text/html" in content_type
                or "application/xhtml" in content_type
                or "<html" in text.lower()
            )

            if is_html:
                extracted = self._extract_html_text(
                    text
                )

                title = self._extract_html_title(
                    text
                )

            else:
                extracted = self._clean_plain_text(
                    text
                )

                title = ""

            if len(extracted) > self.fetch_max_text:

                extracted = extracted[
                    : self.fetch_max_text
                ]

                truncated_text = True

            else:
                truncated_text = False

            return {
                "url": url,
                "final_url": final_url,
                "title": title,
                "status_code": response.status_code,
                "content_type": content_type,
                "text": extracted,
                "text_length": len(extracted),
                "bytes": len(response.content),
                "downloaded_bytes": len(raw_bytes),
                "truncated": (
                    truncated_bytes
                    or truncated_text
                ),
                "provider": "httpx",
            }

    def _format_fetch_response(
        self,
        data: Dict[str, Any],
    ) -> str:

        title = data.get(
            "title"
        )

        text = data.get(
            "text",
            "",
        )

        if title:
            header = (
                f"Halaman: {title}"
            )
        else:
            header = "Halaman"

        if not text:
            return header

        return (
            f"{header}\n\n"
            f"{text}"
        )

    # ========================================================
    # HTML TEXT EXTRACTION
    # ========================================================

    def _extract_html_title(
        self,
        raw_html: str,
    ) -> str:

        match = re.search(
            r"<title[^>]*>"
            r"(.*?)"
            r"</title>",
            raw_html,
            re.I | re.S,
        )

        if not match:
            return ""

        return self._clean_plain_text(
            html.unescape(
                match.group(1)
            )
        )

    def _extract_html_text(
        self,
        raw_html: str,
    ) -> str:

        text = raw_html

        # Remove non-content elements.
        text = re.sub(
            r"<script\b[^>]*>.*?</script>",
            " ",
            text,
            flags=re.I | re.S,
        )

        text = re.sub(
            r"<style\b[^>]*>.*?</style>",
            " ",
            text,
            flags=re.I | re.S,
        )

        text = re.sub(
            r"<noscript\b[^>]*>.*?</noscript>",
            " ",
            text,
            flags=re.I | re.S,
        )

        text = re.sub(
            r"<svg\b[^>]*>.*?</svg>",
            " ",
            text,
            flags=re.I | re.S,
        )

        text = re.sub(
            r"<iframe\b[^>]*>.*?</iframe>",
            " ",
            text,
            flags=re.I | re.S,
        )

        # Structural HTML tags -> line breaks.
        text = re.sub(
            r"</?(?:p|div|section|article|main|header|footer|"
            r"li|ul|ol|h1|h2|h3|h4|h5|h6|br|tr|td|th)"
            r"\b[^>]*>",
            "\n",
            text,
            flags=re.I,
        )

        # Remove remaining tags.
        text = re.sub(
            r"<[^>]+>",
            " ",
            text,
        )

        text = html.unescape(
            text
        )

        return self._clean_plain_text(
            text
        )

    @staticmethod
    def _strip_html(
        text: str,
    ) -> str:

        text = re.sub(
            r"<[^>]+>",
            " ",
            text,
        )

        return ToolEngine._clean_plain_text(
            html.unescape(text)
        )

    @staticmethod
    def _clean_plain_text(
        text: str,
    ) -> str:

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n[ \t]+",
            "\n",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    @staticmethod
    def _decode_response_text(
        raw_bytes: bytes,
        content_type: str,
    ) -> str:

        charset = None

        match = re.search(
            r"charset=([a-zA-Z0-9._-]+)",
            content_type or "",
            re.I,
        )

        if match:
            charset = match.group(1)

        encodings = []

        if charset:
            encodings.append(
                charset
            )

        encodings.extend(
            [
                "utf-8",
                "latin-1",
            ]
        )

        for encoding in encodings:

            try:
                return raw_bytes.decode(
                    encoding,
                    errors="replace",
                )
            except Exception:
                continue

        return raw_bytes.decode(
            "utf-8",
            errors="replace",
        )

    # ========================================================
    # URL SECURITY
    # ========================================================

    def _validate_url(
        self,
        url: str,
    ) -> None:

        parsed = urlparse(
            url
        )

        if parsed.scheme not in {
            "http",
            "https",
        }:
            raise ValueError(
                "Hanya HTTP/HTTPS yang diperbolehkan."
            )

        if not parsed.hostname:
            raise ValueError(
                "Hostname URL tidak valid."
            )

        hostname = parsed.hostname.lower()

        blocked_hosts = {
            "localhost",
            "localhost.localdomain",
            "metadata.google.internal",
            "metadata",
        }

        if hostname in blocked_hosts:
            raise ValueError(
                "Akses ke hostname internal diblokir."
            )

        if hostname.endswith(
            ".local"
        ):
            raise ValueError(
                "Domain lokal diblokir."
            )

        # IP literal check
        try:

            ip = ipaddress.ip_address(
                hostname
            )

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                raise ValueError(
                    "Akses ke alamat IP internal diblokir."
                )

        except ValueError as exc:

            # If it was actually an IP and blocked,
            # preserve security exception.
            if "internal" in str(exc).lower():
                raise

            # Otherwise it is a normal hostname.
            pass

        # Resolve hostname and check returned IPs.
        if self.ssrf_protection:

            try:

                addresses = socket.getaddrinfo(
                    hostname,
                    parsed.port or (
                        443
                        if parsed.scheme == "https"
                        else 80
                    ),
                    type=socket.SOCK_STREAM,
                )

                for address in addresses:

                    ip_text = address[
                        4
                    ][0]

                    try:
                        ip = ipaddress.ip_address(
                            ip_text
                        )

                    except ValueError:
                        continue

                    if (
                        ip.is_private
                        or ip.is_loopback
                        or ip.is_link_local
                        or ip.is_multicast
                        or ip.is_reserved
                        or ip.is_unspecified
                    ):
                        raise ValueError(
                            "Hostname mengarah ke alamat internal."
                        )

            except socket.gaierror:
                raise ValueError(
                    "Hostname tidak dapat di-resolve."
                )

    # ========================================================
    # URL NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_url(
        url: str,
    ) -> str:

        if not url:
            return ""

        url = html.unescape(
            url.strip()
        )

        url = url.replace(
            "\\/",
            "/",
        )

        # Remove accidental markdown wrapper.
        markdown = re.match(
            r"^\[([^\]]+)\]\((https?://[^)]+)\)$",
            url,
            re.I,
        )

        if markdown:
            url = markdown.group(2)

        parsed = urlparse(
            url
        )

        if parsed.scheme not in {
            "http",
            "https",
        }:
            return url

        scheme = parsed.scheme.lower()

        hostname = (
            parsed.hostname
            or ""
        ).lower()

        try:
            port = parsed.port

        except ValueError:
            port = None

        netloc = hostname

        if port:
            default_port = (
                443
                if scheme == "https"
                else 80
            )

            if port != default_port:
                netloc = (
                    f"{hostname}:{port}"
                )

        path = parsed.path or "/"

        return (
            f"{scheme}://"
            f"{netloc}"
            f"{path}"
            + (
                f"?{parsed.query}"
                if parsed.query
                else ""
            )
            + (
                f"#{parsed.fragment}"
                if parsed.fragment
                else ""
            )
        )

    # ========================================================
    # CACHE KEY
    # ========================================================

    @staticmethod
    def _cache_key(
        namespace: str,
        value: str,
    ) -> str:

        normalized = re.sub(
            r"\s+",
            " ",
            str(value or "").strip().lower(),
        )

        return (
            f"{namespace}:"
            f"{normalized}"
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _domain(
        url: str,
    ) -> str:

        try:

            return (
                urlparse(url)
                .netloc
                .lower()
                .split("@")[-1]
                .split(":")[0]
            )

        except Exception:
            return ""

    @staticmethod
    def _is_http_url(
        url: str,
    ) -> bool:

        try:

            return urlparse(
                url
            ).scheme in {
                "http",
                "https",
            }

        except Exception:
            return False

    @staticmethod
    def _latency_ms(
        start: float,
    ) -> float:

        return round(
            (
                time.perf_counter()
                - start
            )
            * 1000,
            2,
        )

    @staticmethod
    def _format_number(
        value: Any,
    ) -> str:

        if isinstance(
            value,
            float,
        ):

            if value.is_integer():
                return str(
                    int(value)
                )

            return f"{value:.12g}"

        return str(value)

    # ========================================================
    # STATS
    # ========================================================

    def stats(self) -> Dict[str, Any]:

        with self._stats_lock:

            runtime = dict(
                self.runtime
            )

        search_calls = max(
            runtime["search_calls"],
            1,
        )

        fetch_calls = max(
            runtime["fetch_calls"],
            1,
        )

        weather_calls = max(
            runtime["weather_calls"],
            1,
        )

        calculator_calls = max(
            runtime["calculator_calls"],
            1,
        )

        runtime["search_average_latency_ms"] = round(
            runtime[
                "search_latency_total_ms"
            ]
            / search_calls,
            2,
        )

        runtime["fetch_average_latency_ms"] = round(
            runtime[
                "fetch_latency_total_ms"
            ]
            / fetch_calls,
            2,
        )

        runtime["weather_average_latency_ms"] = round(
            runtime[
                "weather_latency_total_ms"
            ]
            / weather_calls,
            2,
        )

        runtime[
            "calculator_average_latency_ms"
        ] = round(
            runtime[
                "calculator_latency_total_ms"
            ]
            / calculator_calls,
            2,
        )

        total_requests = max(
            runtime[
                "requests_total"
            ],
            1,
        )

        runtime[
            "success_rate_percent"
        ] = round(
            (
                runtime[
                    "requests_success"
                ]
                / total_requests
            )
            * 100,
            2,
        )

        search_cache_total = (
            runtime[
                "search_cache_hits"
            ]
            + runtime[
                "search_cache_misses"
            ]
        )

        fetch_cache_total = (
            runtime[
                "fetch_cache_hits"
            ]
            + runtime[
                "fetch_cache_misses"
            ]
        )

        runtime[
            "search_cache_hit_rate_percent"
        ] = (
            round(
                (
                    runtime[
                        "search_cache_hits"
                    ]
                    / search_cache_total
                )
                * 100,
                2,
            )
            if search_cache_total
            else 0.0
        )

        runtime[
            "fetch_cache_hit_rate_percent"
        ] = (
            round(
                (
                    runtime[
                        "fetch_cache_hits"
                    ]
                    / fetch_cache_total
                )
                * 100,
                2,
            )
            if fetch_cache_total
            else 0.0
        )

        return {
            "engine": self.engine,
            "version": self.version,

            "tools": list(
                self.tools
            ),

            "calculator": (
                self.calculator_enabled
            ),

            "weather": (
                self.weather_enabled
            ),

            "search": (
                self.search_enabled
            ),

            "fetch": (
                self.fetch_enabled
            ),

            "safe_calculator": True,

            "weather_provider": (
                self.weather_provider
            ),

            "search_provider": list(
                self.search_provider
            ),

            "search_fallback": (
                self.search_fallback
            ),

            "search_method": (
                self.search_method
            ),

            "search_result_limit": (
                self.search_result_limit
            ),

            "search_retries": (
                self.search_retries
            ),

            "search_connect_timeout": (
                self.search_connect_timeout
            ),

            "search_read_timeout": (
                self.search_read_timeout
            ),

            "fetch_connect_timeout": (
                self.fetch_connect_timeout
            ),

            "fetch_read_timeout": (
                self.fetch_read_timeout
            ),

            "fetch_max_bytes": (
                self.fetch_max_bytes
            ),

            "fetch_max_text": (
                self.fetch_max_text
            ),

            "fetch_max_redirects": (
                self.fetch_max_redirects
            ),

            "search_cache": (
                self.search_cache_enabled
            ),

            "search_cache_ttl": (
                self.search_cache_ttl
            ),

            "search_cache_entries": (
                self.search_cache.size()
            ),

            "search_cache_max_entries": (
                self.search_cache_max_entries
            ),

            "fetch_cache": (
                self.fetch_cache_enabled
            ),

            "fetch_cache_ttl": (
                self.fetch_cache_ttl
            ),

            "fetch_cache_entries": (
                self.fetch_cache.size()
            ),

            "fetch_cache_max_entries": (
                self.fetch_cache_max_entries
            ),

            "timeout_protection": (
                self.timeout_protection
            ),

            "ast_evaluator": (
                self.ast_evaluator
            ),

            "controlled_provider_failure": (
                self.controlled_provider_failure
            ),

            "bing_redirect_decoder": (
                self.bing_redirect_decoder
            ),

            "bing_html_parser": (
                self.bing_html_parser
            ),

            "url_normalization": (
                self.url_normalization
            ),

            "search_deduplication": (
                self.search_deduplication
            ),

            "web_fetch": (
                self.web_fetch
            ),

            "html_text_extraction": (
                self.html_text_extraction
            ),

            "ssrf_protection": (
                self.ssrf_protection
            ),

            "redirect_protection": (
                self.redirect_protection
            ),

            "runtime": runtime,

            "status": "READY",
        }

    # ========================================================
    # CACHE CONTROL
    # ========================================================

    def clear_cache(
        self,
        tool: Optional[str] = None,
    ) -> Dict[str, Any]:

        normalized = (
            str(tool or "")
            .strip()
            .lower()
        )

        if not normalized:

            self.search_cache.clear()
            self.fetch_cache.clear()

            return {
                "success": True,
                "cleared": [
                    "search",
                    "fetch",
                ],
            }

        if normalized == "search":

            self.search_cache.clear()

            return {
                "success": True,
                "cleared": [
                    "search",
                ],
            }

        if normalized == "fetch":

            self.fetch_cache.clear()

            return {
                "success": True,
                "cleared": [
                    "fetch",
                ],
            }

        return {
            "success": False,
            "error": (
                "Tool cache tidak dikenal."
            ),
        }


# ============================================================
# SINGLETON
# ============================================================

_engine_instance: Optional[
    ToolEngine
] = None

_engine_lock = threading.RLock()


def get_tool_engine() -> ToolEngine:

    global _engine_instance

    with _engine_lock:

        if _engine_instance is None:
            _engine_instance = ToolEngine()

        return _engine_instance