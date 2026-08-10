from __future__ import annotations

import ast
import operator
import re
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any, Optional

import httpx


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

    Version 1.1.0

    Tools:
        - calculator
        - weather
        - search

    Features:
        - Safe AST calculator
        - Open-Meteo weather provider
        - DuckDuckGo HTML search fallback
        - Tool registry
        - Timeout protection
        - Result normalization
        - Provider metadata
        - Defensive error handling
    """

    VERSION = "1.1.0"

    WEATHER_GEOCODE_URL = (
        "https://geocoding-api.open-meteo.com/v1/search"
    )

    WEATHER_URL = (
        "https://api.open-meteo.com/v1/forecast"
    )

    SEARCH_URL = (
        "https://html.duckduckgo.com/html/"
    )

    DEFAULT_TIMEOUT = 10.0
    GEOCODE_TIMEOUT = 8.0

    MAX_CALCULATOR_LENGTH = 200
    MAX_POWER_EXPONENT = 100
    MAX_ABSOLUTE_RESULT = 10**100

    MAX_SEARCH_RESULTS = 5

    def __init__(self):
        self._tools = {
            "calculator": self.handle_calculator,
            "weather": self.handle_weather,
            "search": self.handle_search,
        }

    # ==========================================================
    # REGISTRY
    # ==========================================================

    def tools(self) -> list[str]:
        return list(self._tools.keys())

    def has_tool(self, route: str) -> bool:
        normalized = str(route or "").strip().lower()
        return normalized in self._tools

    # ==========================================================
    # MAIN EXECUTOR
    # ==========================================================

    def execute(
        self,
        route: str,
        message: str,
        metadata: Optional[dict] = None,
    ) -> ToolResult:

        normalized_route = (
            str(route or "")
            .strip()
            .lower()
        )

        handler = self._tools.get(
            normalized_route
        )

        if handler is None:
            return ToolResult(
                success=False,
                tool=normalized_route,
                error=(
                    f"Tool '{normalized_route}' "
                    "tidak tersedia."
                ),
            )

        try:
            return handler(
                str(message or "").strip(),
                metadata or {},
            )

        except Exception as error:
            return ToolResult(
                success=False,
                tool=normalized_route,
                error=str(error),
            )

    # ==========================================================
    # CALCULATOR
    # ==========================================================

    def handle_calculator(
        self,
        message: str,
        metadata: dict,
    ) -> ToolResult:

        expression = self._extract_expression(
            message
        )

        if not expression:
            return ToolResult(
                success=False,
                tool="calculator",
                error=(
                    "Ekspresi matematika "
                    "tidak ditemukan."
                ),
            )

        try:
            result = self._safe_eval(
                expression
            )

            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(
                        result,
                        10,
                    )

            return ToolResult(
                success=True,
                tool="calculator",
                response=(
                    f"Hasilnya adalah {result}."
                ),
                data={
                    "expression": expression,
                    "result": result,
                    "safe": True,
                },
            )

        except Exception as error:
            return ToolResult(
                success=False,
                tool="calculator",
                error=(
                    f"Perhitungan gagal: {error}"
                ),
            )

    def _extract_expression(
        self,
        message: str,
    ) -> str:

        text = str(
            message or ""
        ).strip()

        patterns = [
            r"^(?:hitung|calculate|calculator)\s+(.+)$",
            r"^(?:berapa hasil dari)\s+(.+)$",
            (
                r"^(?:berapa)\s+"
                r"([0-9\.\+\-\*\/\(\)\%\^\s]+)$"
            ),
        ]

        for pattern in patterns:
            match = re.match(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return (
                    match.group(1)
                    .strip()
                )

        if re.fullmatch(
            r"[0-9\.\+\-\*\/\(\)\%\^\s]+",
            text,
        ):
            return text

        return ""

    def _safe_eval(
        self,
        expression: str,
    ) -> Any:

        expression = (
            expression
            .replace("^", "**")
            .strip()
        )

        if not expression:
            raise ValueError(
                "Ekspresi kosong."
            )

        if len(expression) > (
            self.MAX_CALCULATOR_LENGTH
        ):
            raise ValueError(
                "Ekspresi terlalu panjang."
            )

        try:
            tree = ast.parse(
                expression,
                mode="eval",
            )
        except SyntaxError:
            raise ValueError(
                "Format ekspresi tidak valid."
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

        def validate_number(
            value: Any,
        ) -> None:

            if not isinstance(
                value,
                (int, float),
            ):
                raise ValueError(
                    "Hanya angka yang diperbolehkan."
                )

            if isinstance(
                value,
                float,
            ):
                if value != value:
                    raise ValueError(
                        "NaN tidak diperbolehkan."
                    )

                if value in (
                    float("inf"),
                    float("-inf"),
                ):
                    raise ValueError(
                        "Infinity tidak diperbolehkan."
                    )

            if abs(value) > (
                self.MAX_ABSOLUTE_RESULT
            ):
                raise ValueError(
                    "Nilai terlalu besar."
                )

        def evaluate(node):

            if isinstance(
                node,
                ast.Expression,
            ):
                return evaluate(
                    node.body
                )

            if isinstance(
                node,
                ast.Constant,
            ):

                if isinstance(
                    node.value,
                    bool,
                ):
                    raise ValueError(
                        "Boolean tidak diperbolehkan."
                    )

                if isinstance(
                    node.value,
                    (int, float),
                ):
                    validate_number(
                        node.value
                    )
                    return node.value

                raise ValueError(
                    "Hanya angka yang diperbolehkan."
                )

            if isinstance(
                node,
                ast.UnaryOp,
            ):

                operation = operators.get(
                    type(node.op)
                )

                if operation is None:
                    raise ValueError(
                        "Operator tidak diperbolehkan."
                    )

                operand = evaluate(
                    node.operand
                )

                result = operation(
                    operand
                )

                validate_number(result)

                return result

            if isinstance(
                node,
                ast.BinOp,
            ):

                operation = operators.get(
                    type(node.op)
                )

                if operation is None:
                    raise ValueError(
                        "Operator tidak diperbolehkan."
                    )

                left = evaluate(
                    node.left
                )

                right = evaluate(
                    node.right
                )

                if isinstance(
                    node.op,
                    ast.Pow,
                ):

                    if abs(right) > (
                        self.MAX_POWER_EXPONENT
                    ):
                        raise ValueError(
                            "Pangkat terlalu besar."
                        )

                    if (
                        abs(left) > 10**10
                        and abs(right) > 10
                    ):
                        raise ValueError(
                            "Operasi pangkat terlalu besar."
                        )

                if isinstance(
                    node.op,
                    (
                        ast.Div,
                        ast.FloorDiv,
                        ast.Mod,
                    ),
                ):
                    if right == 0:
                        raise ValueError(
                            "Pembagian dengan nol."
                        )

                result = operation(
                    left,
                    right,
                )

                validate_number(result)

                return result

            raise ValueError(
                "Ekspresi tidak aman."
            )

        result = evaluate(tree)

        validate_number(result)

        return result

    # ==========================================================
    # WEATHER
    # ==========================================================

    def handle_weather(
        self,
        message: str,
        metadata: dict,
    ) -> ToolResult:

        started = time.perf_counter()

        try:
            location = self._extract_location(
                message,
                metadata,
            )

            if not location:
                return ToolResult(
                    success=False,
                    tool="weather",
                    data={
                        "requires_location": True,
                        "message": message,
                    },
                    error=(
                        "Lokasi belum diketahui. "
                        "Contoh: cuaca Jakarta."
                    ),
                )

            latitude = location.get(
                "latitude"
            )

            longitude = location.get(
                "longitude"
            )

            city = location.get(
                "city",
                "lokasi",
            )

            if (
                latitude is None
                or longitude is None
            ):
                return ToolResult(
                    success=False,
                    tool="weather",
                    error=(
                        "Koordinat lokasi tidak valid."
                    ),
                )

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "precipitation,"
                    "weather_code,"
                    "wind_speed_10m"
                ),
                "timezone": "auto",
                "forecast_days": 1,
            }

            with httpx.Client(
                timeout=self.DEFAULT_TIMEOUT
            ) as client:

                response = client.get(
                    self.WEATHER_URL,
                    params=params,
                )

                response.raise_for_status()

                payload = response.json()

            current = payload.get(
                "current",
                {},
            )

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

            description = (
                self._weather_description(
                    weather_code
                )
            )

            latency_ms = round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                2,
            )

            response_text = (
                f"Cuaca di {city}: "
                f"{description}. "
                f"Suhu {temperature}°C, "
                f"terasa seperti "
                f"{feels_like}°C, "
                f"kelembapan "
                f"{humidity}%, "
                f"angin "
                f"{wind} km/jam."
            )

            if precipitation is not None:
                response_text += (
                    " Curah hujan saat ini "
                    f"{precipitation} mm."
                )

            return ToolResult(
                success=True,
                tool="weather",
                response=response_text,
                data={
                    "city": city,
                    "country": location.get(
                        "country",
                        "",
                    ),
                    "latitude": latitude,
                    "longitude": longitude,
                    "temperature_c": temperature,
                    "feels_like_c": feels_like,
                    "humidity_percent": humidity,
                    "precipitation_mm": precipitation,
                    "wind_kmh": wind,
                    "weather_code": weather_code,
                    "description": description,
                    "latency_ms": latency_ms,
                    "provider": "open-meteo",
                },
            )

        except httpx.HTTPStatusError as error:
            return ToolResult(
                success=False,
                tool="weather",
                error=(
                    "Weather provider "
                    f"mengembalikan HTTP "
                    f"{error.response.status_code}."
                ),
            )

        except httpx.RequestError as error:
            return ToolResult(
                success=False,
                tool="weather",
                error=(
                    "Weather provider "
                    f"tidak dapat diakses: {error}"
                ),
            )

        except Exception as error:
            return ToolResult(
                success=False,
                tool="weather",
                error=(
                    f"Weather gagal: {error}"
                ),
            )

    def _extract_location(
        self,
        message: str,
        metadata: dict,
    ) -> Optional[dict]:

        metadata_location = (
            metadata.get("location")
        )

        if isinstance(
            metadata_location,
            dict,
        ):

            latitude = (
                metadata_location.get(
                    "latitude"
                )
            )

            longitude = (
                metadata_location.get(
                    "longitude"
                )
            )

            if (
                latitude is not None
                and longitude is not None
            ):
                return metadata_location

        text = str(
            message or ""
        ).strip()

        if not text:
            return None

        location_name = None

        patterns = [
            r"\b(?:di|kota)\s+(.+)$",
            r"^(?:cuaca|weather)\s+(.+)$",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            candidate = (
                match.group(1)
                .strip()
            )

            candidate = re.sub(
                r"\s+(?:sekarang|hari ini)$",
                "",
                candidate,
                flags=re.IGNORECASE,
            ).strip()

            if candidate:
                location_name = candidate
                break

        if not location_name:
            return None

        try:

            with httpx.Client(
                timeout=self.GEOCODE_TIMEOUT
            ) as client:

                response = client.get(
                    self.WEATHER_GEOCODE_URL,
                    params={
                        "name": location_name,
                        "count": 1,
                        "language": "id",
                        "format": "json",
                    },
                )

                response.raise_for_status()

                payload = response.json()

            results = payload.get(
                "results",
                [],
            )

            if not results:
                return None

            item = results[0]

            latitude = item.get(
                "latitude"
            )

            longitude = item.get(
                "longitude"
            )

            if (
                latitude is None
                or longitude is None
            ):
                return None

            return {
                "city": item.get(
                    "name",
                    location_name,
                ),
                "country": item.get(
                    "country",
                    "",
                ),
                "latitude": latitude,
                "longitude": longitude,
            }

        except Exception:
            return None

    @staticmethod
    def _weather_description(
        code: Any,
    ) -> str:

        descriptions = {
            0: "cerah",
            1: "sebagian besar cerah",
            2: "berawan sebagian",
            3: "mendung",
            45: "berkabut",
            48: "kabut tebal",
            51: "gerimis ringan",
            53: "gerimis",
            55: "gerimis lebat",
            56: "gerimis beku ringan",
            57: "gerimis beku lebat",
            61: "hujan ringan",
            63: "hujan",
            65: "hujan lebat",
            66: "hujan beku ringan",
            67: "hujan beku lebat",
            71: "salju ringan",
            73: "salju",
            75: "salju lebat",
            77: "butiran salju",
            80: "hujan lokal",
            81: "hujan lokal",
            82: "hujan lokal lebat",
            85: "salju lokal ringan",
            86: "salju lokal lebat",
            95: "badai petir",
            96: "badai petir dengan hujan es",
            99: "badai petir kuat",
        }

        return descriptions.get(
            code,
            "kondisi cuaca tidak diketahui",
        )

    # ==========================================================
    # SEARCH
    # ==========================================================

    def handle_search(
        self,
        message: str,
        metadata: dict,
    ) -> ToolResult:

        query = self._extract_search_query(
            message
        )

        if not query:
            return ToolResult(
                success=False,
                tool="search",
                error=(
                    "Query pencarian kosong."
                ),
            )

        started = time.perf_counter()

        try:

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "Chrome/151.0 Safari/537.36"
                )
            }

            with httpx.Client(
                timeout=self.DEFAULT_TIMEOUT,
                headers=headers,
                follow_redirects=True,
            ) as client:

                response = client.post(
                    self.SEARCH_URL,
                    data={
                        "q": query,
                    },
                )

                response.raise_for_status()

                html = response.text

            results = (
                self._parse_search_results(
                    html
                )
            )

            latency_ms = round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                2,
            )

            if not results:
                return ToolResult(
                    success=False,
                    tool="search",
                    data={
                        "query": query,
                        "results": [],
                        "count": 0,
                        "latency_ms": latency_ms,
                        "provider": "duckduckgo",
                    },
                    error=(
                        "Tidak ada hasil pencarian "
                        "yang ditemukan."
                    ),
                )

            lines = [
                f"Hasil pencarian untuk: {query}",
                "",
            ]

            for index, item in enumerate(
                results,
                start=1,
            ):

                title = (
                    item.get("title")
                    or "Tanpa judul"
                )

                lines.append(
                    f"{index}. {title}"
                )

                snippet = (
                    item.get("snippet")
                    or ""
                ).strip()

                if snippet:
                    lines.append(
                        f"   {snippet}"
                    )

                url = (
                    item.get("url")
                    or ""
                ).strip()

                if url:
                    lines.append(
                        f"   {url}"
                    )

            return ToolResult(
                success=True,
                tool="search",
                response="\n".join(lines),
                data={
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "latency_ms": latency_ms,
                    "provider": "duckduckgo",
                },
            )

        except httpx.HTTPStatusError as error:
            return ToolResult(
                success=False,
                tool="search",
                error=(
                    "Search provider "
                    f"mengembalikan HTTP "
                    f"{error.response.status_code}."
                ),
            )

        except httpx.RequestError as error:
            return ToolResult(
                success=False,
                tool="search",
                error=(
                    "Search provider "
                    f"tidak dapat diakses: {error}"
                ),
            )

        except Exception as error:
            return ToolResult(
                success=False,
                tool="search",
                error=(
                    f"Search gagal: {error}"
                ),
            )

    @staticmethod
    def _extract_search_query(
        message: str,
    ) -> str:

        text = str(
            message or ""
        ).strip()

        patterns = [
            (
                r"^(?:cari|search|carikan|"
                r"telusuri)\s+(.+)$"
            ),
            (
                r"^(?:cari berita terbaru "
                r"tentang)\s+(.+)$"
            ),
            (
                r"^(?:berita terbaru "
                r"tentang)\s+(.+)$"
            ),
        ]

        for pattern in patterns:

            match = re.match(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return (
                    match.group(1)
                    .strip()
                )

        return text

    @staticmethod
    def _parse_search_results(
        html: str,
    ) -> list[dict]:

        class Parser(HTMLParser):

            def __init__(self):
                super().__init__()

                self.results = []

                self.current = None

                self.capture_title = False
                self.capture_snippet = False

                self.title_depth = 0
                self.snippet_depth = 0

            def handle_starttag(
                self,
                tag,
                attrs,
            ):

                attributes = dict(attrs)

                classes = (
                    attributes.get(
                        "class",
                        "",
                    )
                    or ""
                )

                if (
                    tag == "a"
                    and
                    "result__a"
                    in classes.split()
                ):

                    self.current = {
                        "title": "",
                        "url": (
                            attributes.get(
                                "href",
                                "",
                            )
                            or ""
                        ),
                        "snippet": "",
                    }

                    self.capture_title = True
                    self.title_depth = 1
                    return

                if (
                    self.current
                    and
                    (
                        "result__snippet"
                        in classes.split()
                    )
                ):

                    self.capture_snippet = True
                    self.snippet_depth = 1
                    return

                if self.capture_title:
                    self.title_depth += 1

                if self.capture_snippet:
                    self.snippet_depth += 1

            def handle_endtag(
                self,
                tag,
            ):

                if self.capture_title:

                    self.title_depth -= 1

                    if self.title_depth <= 0:

                        self.capture_title = False
                        self.title_depth = 0

                if self.capture_snippet:

                    self.snippet_depth -= 1

                    if self.snippet_depth <= 0:

                        self.capture_snippet = False
                        self.snippet_depth = 0

                if (
                    self.current
                    and not self.capture_title
                    and not self.capture_snippet
                    and self.current.get(
                        "title"
                    )
                ):

                    self._commit_current()

            def handle_data(
                self,
                data,
            ):

                cleaned = unescape(
                    data.strip()
                )

                if not cleaned:
                    return

                if (
                    self.current
                    and self.capture_title
                ):

                    self.current[
                        "title"
                    ] += cleaned

                elif (
                    self.current
                    and self.capture_snippet
                ):

                    self.current[
                        "snippet"
                    ] += cleaned

            def _commit_current(self):

                if not self.current:
                    return

                title = re.sub(
                    r"\s+",
                    " ",
                    self.current.get(
                        "title",
                        "",
                    ),
                ).strip()

                snippet = re.sub(
                    r"\s+",
                    " ",
                    self.current.get(
                        "snippet",
                        "",
                    ),
                ).strip()

                url = (
                    self.current.get(
                        "url",
                        "",
                    )
                    or ""
                ).strip()

                item = {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }

                if (
                    title
                    and item not in self.results
                    and len(self.results) < 5
                ):
                    self.results.append(
                        item
                    )

                self.current = None

        parser = Parser()

        try:
            parser.feed(
                str(html or "")
            )
            parser.close()
        except Exception:
            return []

        return parser.results[:5]

    # ==========================================================
    # STATUS
    # ==========================================================

    def stats(self) -> dict:

        return {
            "engine": "ToolEngine",
            "version": self.VERSION,
            "tools": self.tools(),
            "calculator": True,
            "weather": True,
            "search": True,
            "safe_calculator": True,
            "weather_provider": "open-meteo",
            "search_provider": "duckduckgo",
            "timeout_protection": True,
            "ast_evaluator": True,
            "status": "READY",
        }


# ==============================================================
# SINGLETON
# ==============================================================

_tool_engine: Optional[ToolEngine] = None


def get_tool_engine() -> ToolEngine:

    global _tool_engine

    if _tool_engine is None:
        _tool_engine = ToolEngine()

    return _tool_engine