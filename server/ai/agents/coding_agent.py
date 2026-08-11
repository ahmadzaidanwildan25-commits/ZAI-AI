from __future__ import annotations

import ast
import json
import re
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .agent_result import AgentResult
from .base_agent import BaseAgent


class CodingAgent(BaseAgent):
    """
    ================================================================
    SUPER ZAI - CODING AGENT
    ================================================================

    CodingAgent adalah agent khusus pemrograman untuk ZAI.

    Fungsi utama:

    1. Mendeteksi bahasa pemrograman.
    2. Mendeteksi code block Markdown.
    3. Mengekstrak source code.
    4. Menganalisis struktur kode.
    5. Memvalidasi syntax Python.
    6. Mendeteksi pola berbahaya secara statis.
    7. Mendeteksi kemungkinan bug umum.
    8. Memberikan informasi mengenai kode.
    9. Menghasilkan AgentResult.
    10. Terintegrasi dengan BaseAgent.
    11. Dapat digunakan AgentRouter.
    12. Dapat digunakan AgentRuntime.
    13. Tidak menjalankan source code pengguna secara langsung.

    Catatan keamanan:

    CodingAgent melakukan STATIC ANALYSIS.

    Agent ini TIDAK mengeksekusi:
        eval()
        exec()
        subprocess
        shell command
        arbitrary Python source

    Source code hanya dianalisis sebagai teks / AST.

    ================================================================
    VERSION
    ================================================================
    """

    name = "coding_agent"
    version = "1.2.0"

    description = (
        "Coding AI agent untuk analisis, debugging, "
        "deteksi bahasa, ekstraksi kode, dan reasoning pemrograman ZAI."
    )

    capabilities = (
        "coding",
        "code_analysis",
        "code_generation",
        "debugging",
        "language_detection",
        "syntax_validation",
        "code_extraction",
        "security_analysis",
        "static_analysis",
        "python_analysis",
        "task_response",
    )

    # ================================================================
    # CODE BLOCK PATTERN
    # ================================================================
    #
    # INI adalah bagian yang memperbaiki error:
    #
    # AttributeError:
    # CodingAgent has no attribute CODE_BLOCK_PATTERN
    #
    # Pattern menangkap:
    #
    # ```python
    # print("hello")
    # ```
    #
    # ```py
    # ...
    # ```
    #
    # ```
    # ...
    # ```
    #
    # ```javascript
    # ...
    # ```
    #
    # ================================================================

    CODE_BLOCK_PATTERN = re.compile(
        r"""
        ```
        [ \t]*
        (?P<language>[A-Za-z0-9_+#.\-]*)?
        [ \t]*
        \r?\n
        (?P<code>.*?)
        \r?\n?
        ```
        """,
        re.DOTALL | re.VERBOSE,
    )

    INLINE_CODE_PATTERN = re.compile(
        r"`(?P<code>[^`\n]+)`"
    )

    # ================================================================
    # LANGUAGE ALIASES
    # ================================================================

    LANGUAGE_ALIASES: dict[str, str] = {
        "py": "python",
        "python3": "python",
        "python-3": "python",
        "python3.x": "python",
        "py3": "python",

        "js": "javascript",
        "node": "javascript",
        "nodejs": "javascript",
        "jsx": "javascript",

        "ts": "typescript",
        "tsx": "typescript",

        "c++": "cpp",
        "cc": "cpp",
        "hpp": "cpp",
        "h++": "cpp",

        "c#": "csharp",
        "cs": "csharp",

        "golang": "go",

        "rb": "ruby",
        "rs": "rust",

        "sh": "shell",
        "bash": "shell",
        "zsh": "shell",

        "ps": "powershell",
        "ps1": "powershell",

        "yml": "yaml",

        "md": "markdown",

        "text": "text",
        "txt": "text",
    }

    SUPPORTED_LANGUAGES = (
        "python",
        "javascript",
        "typescript",
        "java",
        "kotlin",
        "dart",
        "c",
        "cpp",
        "csharp",
        "go",
        "rust",
        "php",
        "ruby",
        "swift",
        "shell",
        "powershell",
        "sql",
        "html",
        "css",
        "scss",
        "yaml",
        "json",
        "xml",
        "markdown",
        "text",
        "unknown",
    )

    # ================================================================
    # LANGUAGE KEYWORDS
    # ================================================================

    LANGUAGE_KEYWORDS: dict[str, tuple[str, ...]] = {
        "python": (
            "def ",
            "import ",
            "from ",
            "class ",
            "async def ",
            "await ",
            "elif ",
            "None",
            "True",
            "False",
            "print(",
            "self.",
            "__init__",
            "pip ",
            "pytest",
            "django",
            "flask",
            "fastapi",
            "pydantic",
        ),

        "javascript": (
            "const ",
            "let ",
            "var ",
            "function ",
            "=>",
            "console.log",
            "require(",
            "module.exports",
            "document.",
            "window.",
            "npm ",
            "node ",
        ),

        "typescript": (
            "interface ",
            "type ",
            "enum ",
            "namespace ",
            "public ",
            "private ",
            ": string",
            ": number",
            ": boolean",
            "as ",
            "tsx",
        ),

        "java": (
            "public class ",
            "private class ",
            "public static void main",
            "System.out.",
            "import java.",
            "extends ",
            "implements ",
            "new ",
        ),

        "kotlin": (
            "fun ",
            "val ",
            "var ",
            "data class ",
            "object ",
            "companion object",
            "println(",
            "import kotlin.",
        ),

        "dart": (
            "void main(",
            "Future<",
            "Widget",
            "BuildContext",
            "StatelessWidget",
            "StatefulWidget",
            "import 'package:",
            "flutter",
            "dart:",
        ),

        "c": (
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "int main(",
            "printf(",
            "scanf(",
            "malloc(",
            "free(",
        ),

        "cpp": (
            "#include <iostream>",
            "std::",
            "cout",
            "cin",
            "using namespace std",
            "int main(",
            "vector<",
            "string",
        ),

        "csharp": (
            "using System;",
            "namespace ",
            "public class ",
            "private class ",
            "Console.WriteLine",
            "static void Main",
            "async Task",
        ),

        "go": (
            "package main",
            "func main(",
            "fmt.Println",
            "go ",
            "goroutine",
            "chan ",
            "defer ",
        ),

        "rust": (
            "fn main(",
            "let mut ",
            "use std::",
            "println!",
            "cargo ",
            "impl ",
            "trait ",
        ),

        "php": (
            "<?php",
            "$_POST",
            "$_GET",
            "function ",
            "echo ",
            "namespace ",
            "use ",
        ),

        "ruby": (
            "def ",
            "end",
            "puts ",
            "require ",
            "class ",
            "module ",
            "attr_accessor",
        ),

        "swift": (
            "import Foundation",
            "import UIKit",
            "import SwiftUI",
            "let ",
            "var ",
            "func ",
            "struct ",
            "class ",
        ),

        "shell": (
            "#!/bin/bash",
            "#!/bin/sh",
            "echo ",
            "chmod ",
            "mkdir ",
            "grep ",
            "sed ",
            "awk ",
        ),

        "powershell": (
            "Get-",
            "Set-",
            "New-",
            "Remove-",
            "Write-Host",
            "$env:",
            "$PSVersionTable",
            "Invoke-",
        ),

        "sql": (
            "SELECT ",
            "INSERT INTO ",
            "UPDATE ",
            "DELETE FROM ",
            "CREATE TABLE ",
            "ALTER TABLE ",
            "DROP TABLE ",
            "JOIN ",
            "WHERE ",
        ),

        "html": (
            "<html",
            "<!DOCTYPE html",
            "<body",
            "<div",
            "<span",
            "<head",
            "<script",
        ),

        "css": (
            "{",
            "color:",
            "display:",
            "margin:",
            "padding:",
            "font-size:",
            "background:",
        ),

        "yaml": (
            "services:",
            "version:",
            "dependencies:",
            "name:",
            "description:",
        ),

        "json": (
            '{"',
            "[{",
            '": ',
        ),
    }

    # ================================================================
    # SECURITY PATTERNS
    # ================================================================

    SECURITY_PATTERNS: tuple[tuple[str, str, str], ...] = (
        (
            "python_eval",
            r"\beval\s*\(",
            "Penggunaan eval() dapat mengeksekusi input dinamis."
        ),
        (
            "python_exec",
            r"\bexec\s*\(",
            "Penggunaan exec() dapat mengeksekusi source dinamis."
        ),
        (
            "python_os_system",
            r"\bos\.system\s*\(",
            "os.system() menjalankan command sistem."
        ),
        (
            "python_subprocess",
            r"\bsubprocess\.",
            "subprocess dapat menjalankan proses eksternal."
        ),
        (
            "python_pickle",
            r"\bpickle\.(load|loads)\s*\(",
            "pickle dari sumber tidak terpercaya dapat berbahaya."
        ),
        (
            "python_yaml_unsafe",
            r"\byaml\.load\s*\(",
            "yaml.load() perlu loader aman."
        ),
        (
            "shell_rm_rf",
            r"\brm\s+-rf\b",
            "rm -rf dapat menghapus data secara destruktif."
        ),
        (
            "powershell_encoded",
            r"-EncodedCommand\b",
            "Encoded PowerShell command perlu diperiksa."
        ),
        (
            "powershell_invoke_expression",
            r"\bInvoke-Expression\b",
            "Invoke-Expression dapat mengeksekusi command dinamis."
        ),
        (
            "javascript_eval",
            r"\beval\s*\(",
            "eval() pada JavaScript dapat mengeksekusi string dinamis."
        ),
        (
            "javascript_innerhtml",
            r"\.innerHTML\s*=",
            "innerHTML dapat menyebabkan XSS jika input tidak disanitasi."
        ),
        (
            "sql_dynamic",
            r"""(?i)(SELECT|INSERT|UPDATE|DELETE).*[\+\|]\s*[A-Za-z_$]""",
            "SQL string dinamis berpotensi menyebabkan SQL injection."
        ),
        (
            "hardcoded_password",
            r"""(?i)\b(password|passwd|pwd)\s*=\s*["'][^"']+["']""",
            "Kemungkinan password hardcoded."
        ),
        (
            "hardcoded_api_key",
            r"""(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token)\s*=\s*["'][^"']+["']""",
            "Kemungkinan credential hardcoded."
        ),
    )

    # ================================================================
    # BUG PATTERNS
    # ================================================================

    BUG_PATTERNS: tuple[tuple[str, str, str], ...] = (
        (
            "python_bare_except",
            r"except\s*:",
            "Bare except dapat menyembunyikan error."
        ),
        (
            "python_mutable_default",
            r"def\s+\w+\([^)]*=\s*(\[\]|\{\})",
            "Mutable default argument dapat menyebabkan state tidak terduga."
        ),
        (
            "python_compare_none",
            r"==\s*None",
            "Lebih baik gunakan 'is None'."
        ),
        (
            "python_compare_true",
            r"==\s*True",
            "Pertimbangkan pengecekan boolean langsung."
        ),
        (
            "todo",
            r"\bTODO\b",
            "Terdapat TODO yang belum diselesaikan."
        ),
        (
            "fixme",
            r"\bFIXME\b",
            "Terdapat FIXME yang belum diselesaikan."
        ),
        (
            "javascript_var",
            r"\bvar\s+\w+",
            "Gunakan let/const bila memungkinkan."
        ),
        (
            "javascript_loose_equal",
            r"(?<![=!])==(?!=)",
            "Pertimbangkan strict equality ===."
        ),
        (
            "empty_catch",
            r"catch\s*\([^)]*\)\s*\{\s*\}",
            "Empty catch dapat menyembunyikan error."
        ),
    )

    # ================================================================
    # COMMAND / INTENT PATTERNS
    # ================================================================

    COMMAND_PATTERNS: tuple[tuple[str, str], ...] = (
        (
            "debug",
            r"\b(debug|debugging|perbaiki|perbaikan|fix|error|bug)\b",
        ),
        (
            "analyze",
            r"\b(analyze|analisa|analysis|analisis|jelaskan kode)\b",
        ),
        (
            "generate",
            r"\b(generate|buatkan|buat kode|coding|program)\b",
        ),
        (
            "review",
            r"\b(review|review code|cek kode|periksa kode)\b",
        ),
        (
            "refactor",
            r"\b(refactor|rapikan|optimalkan struktur)\b",
        ),
        (
            "explain",
            r"\b(explain|jelaskan|terangkan)\b",
        ),
        (
            "test",
            r"\b(test|testing|unit test|pytest)\b",
        ),
        (
            "security",
            r"\b(security|keamanan|secure|vulnerability|kerentanan)\b",
        ),
    )

    # ================================================================
    # DATACLASSES
    # ================================================================

    @dataclass
    class CodeBlock:
        language: str
        code: str
        start: int
        end: int
        index: int
        raw_language: str = ""

        def to_dict(self) -> dict[str, Any]:
            return {
                "index": self.index,
                "language": self.language,
                "raw_language": self.raw_language,
                "start": self.start,
                "end": self.end,
                "line_count": len(self.code.splitlines()),
                "character_count": len(self.code),
                "code": self.code,
            }

    @dataclass
    class SecurityFinding:
        rule: str
        severity: str
        message: str
        line: Optional[int] = None

        def to_dict(self) -> dict[str, Any]:
            return {
                "rule": self.rule,
                "severity": self.severity,
                "message": self.message,
                "line": self.line,
            }

    @dataclass
    class BugFinding:
        rule: str
        severity: str
        message: str
        line: Optional[int] = None

        def to_dict(self) -> dict[str, Any]:
            return {
                "rule": self.rule,
                "severity": self.severity,
                "message": self.message,
                "line": self.line,
            }

    # ================================================================
    # INITIALIZATION
    # ================================================================

    def __init__(self) -> None:
        super().__init__()

        self.analysis_count = 0
        self.syntax_check_count = 0
        self.language_detection_count = 0
        self.code_extraction_count = 0

        self.security_finding_count = 0
        self.bug_finding_count = 0

    # ================================================================
    # INFO
    # ================================================================

    def info(self) -> dict[str, Any]:
        data = super().info()

        data.update(
            {
                "analysis_count": self.analysis_count,
                "syntax_check_count": self.syntax_check_count,
                "language_detection_count": (
                    self.language_detection_count
                ),
                "code_extraction_count": (
                    self.code_extraction_count
                ),
                "security_finding_count": (
                    self.security_finding_count
                ),
                "bug_finding_count": (
                    self.bug_finding_count
                ),
            }
        )

        return data

    # ================================================================
    # NORMALIZE LANGUAGE
    # ================================================================

    @classmethod
    def normalize_language(
        cls,
        language: Optional[str],
    ) -> str:
        """
        Normalisasi nama bahasa.

        Contoh:

            py -> python
            js -> javascript
            ts -> typescript
            ps1 -> powershell
        """

        value = str(language or "").strip().lower()

        if not value:
            return "unknown"

        value = value.replace(" ", "")

        if value in cls.LANGUAGE_ALIASES:
            return cls.LANGUAGE_ALIASES[value]

        if value in cls.SUPPORTED_LANGUAGES:
            return value

        return "unknown"

    # ================================================================
    # DETECT LANGUAGE
    # ================================================================

    def detect_language(
        self,
        task: str,
    ) -> str:
        """
        Mendeteksi bahasa pemrograman.

        Prioritas:

        1. Markdown code fence language.
        2. Extension-like hints.
        3. Syntax / keyword scoring.
        4. Python AST.
        5. Unknown.
        """

        self.language_detection_count += 1

        text = str(task or "")

        if not text.strip():
            return "unknown"

        blocks = self.extract_code_blocks(text)

        if blocks:
            languages = [
                block.language
                for block in blocks
                if block.language != "unknown"
            ]

            if languages:
                return self._most_common_language(
                    languages
                )

            source = "\n".join(
                block.code
                for block in blocks
            )
        else:
            source = text

        explicit = self._detect_explicit_language_hint(
            text
        )

        if explicit != "unknown":
            return explicit

        scores = self._score_languages(source)

        if scores:
            best_language, best_score = max(
                scores.items(),
                key=lambda item: item[1],
            )

            if best_score > 0:
                return best_language

        python_result = self._validate_python_syntax(
            source
        )

        if python_result["valid"]:
            return "python"

        return "unknown"

    # ================================================================
    # EXPLICIT LANGUAGE HINT
    # ================================================================

    @classmethod
    def _detect_explicit_language_hint(
        cls,
        text: str,
    ) -> str:
        normalized = str(text or "").lower()

        patterns = (
            (r"\bpython\b", "python"),
            (r"\bjavascript\b", "javascript"),
            (r"\btypescript\b", "typescript"),
            (r"\bjava\b", "java"),
            (r"\bkotlin\b", "kotlin"),
            (r"\bdart\b", "dart"),
            (r"\bc\+\+\b", "cpp"),
            (r"\bc#\b", "csharp"),
            (r"\bgolang\b", "go"),
            (r"\brust\b", "rust"),
            (r"\bphp\b", "php"),
            (r"\bruby\b", "ruby"),
            (r"\bswift\b", "swift"),
            (r"\bpowershell\b", "powershell"),
            (r"\bshell\b", "shell"),
            (r"\bbash\b", "shell"),
            (r"\bsql\b", "sql"),
            (r"\bhtml\b", "html"),
            (r"\bcss\b", "css"),
            (r"\byaml\b", "yaml"),
            (r"\bjson\b", "json"),
        )

        for pattern, language in patterns:
            if re.search(pattern, normalized):
                return language

        return "unknown"

    # ================================================================
    # SCORE LANGUAGES
    # ================================================================

    @classmethod
    def _score_languages(
        cls,
        source: str,
    ) -> dict[str, int]:
        text = str(source or "")

        scores: dict[str, int] = {
            language: 0
            for language in cls.LANGUAGE_KEYWORDS
        }

        lower_text = text.lower()

        for language, keywords in (
            cls.LANGUAGE_KEYWORDS.items()
        ):
            for keyword in keywords:
                if keyword.lower() in lower_text:
                    scores[language] += 1

        # Extra strong signals.

        if re.search(
            r"^\s*def\s+\w+\s*\(",
            text,
            re.MULTILINE,
        ):
            scores["python"] += 5

        if re.search(
            r"^\s*import\s+\w+",
            text,
            re.MULTILINE,
        ):
            scores["python"] += 2

        if re.search(
            r"\bconsole\.log\s*\(",
            text,
        ):
            scores["javascript"] += 5

        if re.search(
            r"\binterface\s+\w+",
            text,
        ):
            scores["typescript"] += 5

        if re.search(
            r"\bfunc\s+main\s*\(",
            text,
        ):
            scores["go"] += 6

        if re.search(
            r"\bfn\s+main\s*\(",
            text,
        ):
            scores["rust"] += 6

        if re.search(
            r"<\/?[A-Za-z][^>]*>",
            text,
        ):
            scores["html"] += 3

        if re.search(
            r"\bSELECT\s+.+\s+FROM\s+",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            scores["sql"] += 6

        if re.search(
            r"^\s*[{[]",
            text,
        ) and re.search(
            r'["\'][^"\']+["\']\s*:',
            text,
        ):
            scores["json"] += 3

        return {
            language: score
            for language, score in scores.items()
            if score > 0
        }

    # ================================================================
    # EXTRACT CODE BLOCKS
    # ================================================================

    @classmethod
    def extract_code_blocks(
        cls,
        task: str,
    ) -> list["CodingAgent.CodeBlock"]:
        """
        Mengambil semua fenced Markdown code block.

        Contoh:

        ```python
        print("Halo")
        ```

        akan menjadi CodeBlock(language="python").
        """

        text = str(task or "")

        matches = cls.CODE_BLOCK_PATTERN.finditer(
            text
        )

        blocks: list[CodingAgent.CodeBlock] = []

        for index, match in enumerate(matches, start=1):
            raw_language = (
                match.group("language") or ""
            ).strip()

            language = cls.normalize_language(
                raw_language
            )

            code = (
                match.group("code") or ""
            ).strip("\r\n")

            blocks.append(
                cls.CodeBlock(
                    language=language,
                    code=code,
                    start=match.start(),
                    end=match.end(),
                    index=index,
                    raw_language=raw_language,
                )
            )

        return blocks

    # ================================================================
    # EXTRACT FIRST CODE
    # ================================================================

    @classmethod
    def extract_code(
        cls,
        task: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Mengembalikan code block pertama yang cocok.
        """

        blocks = cls.extract_code_blocks(task)

        if not blocks:
            return ""

        wanted = cls.normalize_language(
            language
        )

        if wanted != "unknown":
            for block in blocks:
                if block.language == wanted:
                    return block.code

        return blocks[0].code

    # ================================================================
    # EXTRACT INLINE CODE
    # ================================================================

    @classmethod
    def extract_inline_code(
        cls,
        task: str,
    ) -> list[str]:
        text = str(task or "")

        return [
            match.group("code")
            for match in cls.INLINE_CODE_PATTERN.finditer(
                text
            )
        ]

    # ================================================================
    # MOST COMMON LANGUAGE
    # ================================================================

    @staticmethod
    def _most_common_language(
        languages: Iterable[str],
    ) -> str:
        counts: dict[str, int] = {}

        for language in languages:
            counts[language] = (
                counts.get(language, 0) + 1
            )

        if not counts:
            return "unknown"

        return max(
            counts.items(),
            key=lambda item: item[1],
        )[0]

    # ================================================================
    # PYTHON SYNTAX VALIDATION
    # ================================================================

    @classmethod
    def validate_python(
        cls,
        code: str,
    ) -> dict[str, Any]:
        """
        Public helper untuk validasi syntax Python.
        """

        return cls._validate_python_syntax(code)

    @classmethod
    def _validate_python_syntax(
        cls,
        code: str,
    ) -> dict[str, Any]:
        source = str(code or "")

        if not source.strip():
            return {
                "valid": False,
                "error": "Source code kosong.",
                "line": None,
                "offset": None,
                "type": "empty_source",
            }

        try:
            tree = ast.parse(
                source,
                mode="exec",
            )

            return {
                "valid": True,
                "error": None,
                "line": None,
                "offset": None,
                "type": None,
                "node_count": sum(
                    1
                    for _ in ast.walk(tree)
                ),
            }

        except SyntaxError as exc:
            return {
                "valid": False,
                "error": str(exc),
                "line": exc.lineno,
                "offset": exc.offset,
                "type": "syntax_error",
                "text": exc.text,
            }

        except Exception as exc:
            return {
                "valid": False,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
                "line": None,
                "offset": None,
                "type": "validation_error",
            }

    # ================================================================
    # PYTHON AST SUMMARY
    # ================================================================

    @classmethod
    def python_ast_summary(
        cls,
        code: str,
    ) -> dict[str, Any]:
        """
        Membuat ringkasan AST Python tanpa mengeksekusi kode.
        """

        validation = cls._validate_python_syntax(
            code
        )

        if not validation["valid"]:
            return {
                "valid": False,
                "error": validation["error"],
                "line": validation["line"],
                "offset": validation["offset"],
            }

        tree = ast.parse(
            str(code or ""),
            mode="exec",
        )

        functions: list[str] = []
        classes: list[str] = []
        imports: list[str] = []
        variables: list[str] = []

        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                functions.append(node.name)

            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

            elif isinstance(
                node,
                (
                    ast.Import,
                    ast.ImportFrom,
                ),
            ):
                if isinstance(node, ast.Import):
                    imports.extend(
                        alias.name
                        for alias in node.names
                    )
                else:
                    module = (
                        node.module or ""
                    )
                    imports.append(module)

            elif isinstance(
                node,
                ast.Assign,
            ):
                for target in node.targets:
                    if isinstance(
                        target,
                        ast.Name,
                    ):
                        variables.append(
                            target.id
                        )

            elif isinstance(
                node,
                ast.AnnAssign,
            ):
                if isinstance(
                    node.target,
                    ast.Name,
                ):
                    variables.append(
                        node.target.id
                    )

        return {
            "valid": True,
            "node_count": sum(
                1
                for _ in ast.walk(tree)
            ),
            "functions": sorted(
                set(functions)
            ),
            "classes": sorted(
                set(classes)
            ),
            "imports": sorted(
                set(
                    item
                    for item in imports
                    if item
                )
            ),
            "variables": sorted(
                set(variables)
            ),
        }

    # ================================================================
    # SECURITY ANALYSIS
    # ================================================================

    @classmethod
    def security_scan(
        cls,
        code: str,
    ) -> list["CodingAgent.SecurityFinding"]:
        source = str(code or "")

        findings: list[
            CodingAgent.SecurityFinding
        ] = []

        for rule, pattern, message in (
            cls.SECURITY_PATTERNS
        ):
            try:
                regex = re.compile(
                    pattern,
                    re.MULTILINE,
                )
            except re.error:
                continue

            for match in regex.finditer(source):
                line = (
                    source.count(
                        "\n",
                        0,
                        match.start(),
                    )
                    + 1
                )

                severity = (
                    cls._security_severity(rule)
                )

                findings.append(
                    cls.SecurityFinding(
                        rule=rule,
                        severity=severity,
                        message=message,
                        line=line,
                    )
                )

        return findings

    # ================================================================
    # SECURITY SEVERITY
    # ================================================================

    @staticmethod
    def _security_severity(
        rule: str,
    ) -> str:
        critical = {
            "python_eval",
            "python_exec",
            "python_os_system",
            "python_subprocess",
            "powershell_encoded",
            "powershell_invoke_expression",
        }

        high = {
            "python_pickle",
            "sql_dynamic",
            "hardcoded_password",
            "hardcoded_api_key",
        }

        medium = {
            "python_yaml_unsafe",
            "javascript_eval",
            "javascript_innerhtml",
            "shell_rm_rf",
        }

        if rule in critical:
            return "critical"

        if rule in high:
            return "high"

        if rule in medium:
            return "medium"

        return "low"

    # ================================================================
    # BUG ANALYSIS
    # ================================================================

    @classmethod
    def bug_scan(
        cls,
        code: str,
    ) -> list["CodingAgent.BugFinding"]:
        source = str(code or "")

        findings: list[
            CodingAgent.BugFinding
        ] = []

        for rule, pattern, message in (
            cls.BUG_PATTERNS
        ):
            try:
                regex = re.compile(
                    pattern,
                    re.MULTILINE,
                )
            except re.error:
                continue

            for match in regex.finditer(source):
                line = (
                    source.count(
                        "\n",
                        0,
                        match.start(),
                    )
                    + 1
                )

                severity = (
                    cls._bug_severity(rule)
                )

                findings.append(
                    cls.BugFinding(
                        rule=rule,
                        severity=severity,
                        message=message,
                        line=line,
                    )
                )

        return findings

    # ================================================================
    # BUG SEVERITY
    # ================================================================

    @staticmethod
    def _bug_severity(
        rule: str,
    ) -> str:
        high = {
            "python_mutable_default",
        }

        medium = {
            "python_bare_except",
            "javascript_loose_equal",
            "empty_catch",
        }

        if rule in high:
            return "high"

        if rule in medium:
            return "medium"

        return "low"

    # ================================================================
    # DETECT COMMAND / INTENT
    # ================================================================

    @classmethod
    def detect_command(
        cls,
        task: str,
    ) -> str:
        text = str(task or "")

        for command, pattern in (
            cls.COMMAND_PATTERNS
        ):
            if re.search(
                pattern,
                text,
                re.IGNORECASE,
            ):
                return command

        return "general"

    # ================================================================
    # TASK TYPE
    # ================================================================

    @classmethod
    def detect_task_type(
        cls,
        task: str,
    ) -> str:
        command = cls.detect_command(task)

        mapping = {
            "debug": "debugging",
            "analyze": "analysis",
            "generate": "generation",
            "review": "code_review",
            "refactor": "refactoring",
            "explain": "explanation",
            "test": "testing",
            "security": "security_analysis",
            "general": "coding",
        }

        return mapping.get(
            command,
            "coding",
        )

    # ================================================================
    # CODE METRICS
    # ================================================================

    @classmethod
    def code_metrics(
        cls,
        code: str,
    ) -> dict[str, Any]:
        source = str(code or "")

        lines = source.splitlines()

        non_empty = [
            line
            for line in lines
            if line.strip()
        ]

        comments = [
            line
            for line in lines
            if line.strip().startswith(
                (
                    "#",
                    "//",
                    "/*",
                    "*",
                )
            )
        ]

        return {
            "characters": len(source),
            "lines": len(lines),
            "non_empty_lines": len(non_empty),
            "empty_lines": (
                len(lines)
                - len(non_empty)
            ),
            "comment_lines": len(comments),
            "comment_ratio": (
                round(
                    len(comments)
                    / len(lines),
                    4,
                )
                if lines
                else 0.0
            ),
            "words": len(
                re.findall(
                    r"\b\w+\b",
                    source,
                )
            ),
        }

    # ================================================================
    # CODE COMPLEXITY - BASIC
    # ================================================================

    @classmethod
    def estimate_complexity(
        cls,
        code: str,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        source = str(code or "")

        lang = cls.normalize_language(
            language
        )

        branch_patterns = {
            "python": (
                r"\bif\b",
                r"\belif\b",
                r"\bfor\b",
                r"\bwhile\b",
                r"\bexcept\b",
                r"\bmatch\b",
                r"\bcase\b",
            ),
            "javascript": (
                r"\bif\b",
                r"\belse if\b",
                r"\bfor\b",
                r"\bwhile\b",
                r"\bswitch\b",
                r"\bcatch\b",
            ),
        }

        patterns = branch_patterns.get(
            lang,
            (
                r"\bif\b",
                r"\bfor\b",
                r"\bwhile\b",
                r"\bswitch\b",
                r"\bcatch\b",
            ),
        )

        branches = 0

        for pattern in patterns:
            branches += len(
                re.findall(
                    pattern,
                    source,
                    re.IGNORECASE,
                )
            )

        complexity = 1 + branches

        if complexity <= 5:
            level = "low"
        elif complexity <= 10:
            level = "medium"
        elif complexity <= 20:
            level = "high"
        else:
            level = "very_high"

        return {
            "language": lang,
            "branch_count": branches,
            "estimated_cyclomatic_complexity": (
                complexity
            ),
            "level": level,
        }

    # ================================================================
    # NORMALIZE TASK
    # ================================================================

    @staticmethod
    def normalize_task(
        task: str,
    ) -> str:
        return re.sub(
            r"\s+",
            " ",
            str(task or "").strip(),
        )

    # ================================================================
    # BUILD RESPONSE
    # ================================================================

    @classmethod
    def build_analysis_response(
        cls,
        *,
        language: str,
        task_type: str,
        block_count: int,
        security_count: int,
        bug_count: int,
        syntax_valid: Optional[bool],
    ) -> str:
        parts: list[str] = []

        parts.append(
            f"ZAI Coding Agent menganalisis task "
            f"dengan bahasa terdeteksi: {language}."
        )

        parts.append(
            f"Jenis task: {task_type}."
        )

        parts.append(
            f"Code block ditemukan: {block_count}."
        )

        if syntax_valid is True:
            parts.append(
                "Syntax Python: VALID."
            )
        elif syntax_valid is False:
            parts.append(
                "Syntax Python: TIDAK VALID."
            )

        parts.append(
            f"Security finding: {security_count}."
        )

        parts.append(
            f"Potential bug finding: {bug_count}."
        )

        return " ".join(parts)

    # ================================================================
    # STATIC ANALYSIS
    # ================================================================

    def analyze_code(
        self,
        task: str,
    ) -> dict[str, Any]:
        self.analysis_count += 1

        started = time.perf_counter()

        text = str(task or "")

        blocks = self.extract_code_blocks(text)

        if blocks:
            source = "\n\n".join(
                block.code
                for block in blocks
            )

            detected_language = (
                self.detect_language(text)
            )
        else:
            source = text
            detected_language = (
                self.detect_language(text)
            )

        task_type = self.detect_task_type(
            text
        )

        security_findings = (
            self.security_scan(source)
        )

        bug_findings = (
            self.bug_scan(source)
        )

        python_validation: Optional[
            dict[str, Any]
        ] = None

        python_ast: Optional[
            dict[str, Any]
        ] = None

        if detected_language == "python":
            self.syntax_check_count += 1

            python_validation = (
                self.validate_python(source)
            )

            if python_validation["valid"]:
                python_ast = (
                    self.python_ast_summary(
                        source
                    )
                )

        metrics = self.code_metrics(
            source
        )

        complexity = (
            self.estimate_complexity(
                source,
                detected_language,
            )
        )

        self.security_finding_count += (
            len(security_findings)
        )

        self.bug_finding_count += (
            len(bug_findings)
        )

        latency_ms = round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            4,
        )

        return {
            "analysis_id": str(
                uuid.uuid4()
            ),
            "language": detected_language,
            "task_type": task_type,
            "code_block_count": len(
                blocks
            ),
            "code_blocks": [
                block.to_dict()
                for block in blocks
            ],
            "metrics": metrics,
            "complexity": complexity,
            "python_validation": (
                python_validation
            ),
            "python_ast": python_ast,
            "security_findings": [
                finding.to_dict()
                for finding in security_findings
            ],
            "bug_findings": [
                finding.to_dict()
                for finding in bug_findings
            ],
            "security_count": len(
                security_findings
            ),
            "bug_count": len(
                bug_findings
            ),
            "latency_ms": latency_ms,
        }

    # ================================================================
    # RUN
    # ================================================================

    async def run(
        self,
        task: str,
        result: AgentResult,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Main execution method BaseAgent.

        Tidak menjalankan arbitrary user code.
        """

        started = time.perf_counter()

        normalized_task = (
            self.normalize_task(task)
        )

        result.add_observation(
            "coding_agent_started",
            agent=self.name,
            task_length=len(
                normalized_task
            ),
        )

        result.add_observation(
            "task_normalized",
            original_length=len(
                str(task or "")
            ),
            normalized_length=len(
                normalized_task
            ),
        )

        if not normalized_task:
            result.add_warning(
                "Task coding kosong."
            )

            result.response = (
                "ZAI Coding Agent menerima "
                "task kosong."
            )

            result.success = False

            if hasattr(
                result,
                "status",
            ):
                result.status = "failed"

            return result

        try:
            analysis = self.analyze_code(
                normalized_task
            )

            language = analysis[
                "language"
            ]

            task_type = analysis[
                "task_type"
            ]

            block_count = analysis[
                "code_block_count"
            ]

            security_count = analysis[
                "security_count"
            ]

            bug_count = analysis[
                "bug_count"
            ]

            python_validation = analysis[
                "python_validation"
            ]

            syntax_valid: Optional[
                bool
            ] = None

            if python_validation is not None:
                syntax_valid = bool(
                    python_validation[
                        "valid"
                    ]
                )

            result.add_observation(
                "language_detected",
                language=language,
            )

            result.add_observation(
                "task_type_detected",
                task_type=task_type,
            )

            result.add_observation(
                "code_blocks_extracted",
                count=block_count,
            )

            if python_validation is not None:
                result.add_observation(
                    "python_syntax_checked",
                    valid=syntax_valid,
                )

            result.add_observation(
                "security_scan_completed",
                findings=security_count,
            )

            result.add_observation(
                "bug_scan_completed",
                findings=bug_count,
            )

            response = (
                self.build_analysis_response(
                    language=language,
                    task_type=task_type,
                    block_count=block_count,
                    security_count=security_count,
                    bug_count=bug_count,
                    syntax_valid=syntax_valid,
                )
            )

            result.response = response

            result.metadata.update(
                {
                    "agent": self.name,
                    "agent_version": self.version,
                    "task_type": task_type,
                    "language": language,
                    "code_block_count": block_count,
                    "security_count": security_count,
                    "bug_count": bug_count,
                    "analysis": analysis,
                }
            )

            result.add_observation(
                "response_generated",
                response_length=len(
                    response
                ),
            )

            result.complete(
                response,
                latency_ms=round(
                    (
                        time.perf_counter()
                        - started
                    )
                    * 1000,
                    4,
                ),
            )

            return result

        except Exception as exc:
            result.add_error(
                f"{type(exc).__name__}: {exc}"
            )

            result.response = (
                "ZAI Coding Agent gagal "
                "menganalisis task."
            )

            result.success = False

            if hasattr(
                result,
                "status",
            ):
                result.status = "failed"

            return result

    # ================================================================
    # QUICK ANALYSIS
    # ================================================================

    def quick_analysis(
        self,
        task: str,
    ) -> dict[str, Any]:
        """
        Analisis cepat tanpa AgentResult.
        """

        language = self.detect_language(
            task
        )

        blocks = self.extract_code_blocks(
            task
        )

        source = "\n".join(
            block.code
            for block in blocks
        )

        if not source:
            source = str(task or "")

        security = self.security_scan(
            source
        )

        bugs = self.bug_scan(
            source
        )

        return {
            "language": language,
            "task_type": self.detect_task_type(
                task
            ),
            "code_block_count": len(
                blocks
            ),
            "security_count": len(
                security
            ),
            "bug_count": len(
                bugs
            ),
        }

    # ================================================================
    # IS CODE
    # ================================================================

    @classmethod
    def is_code(
        cls,
        task: str,
    ) -> bool:
        text = str(task or "").strip()

        if not text:
            return False

        if cls.extract_code_blocks(text):
            return True

        language = cls._detect_explicit_language_hint(
            text
        )

        if language != "unknown":
            return True

        scores = cls._score_languages(
            text
        )

        return bool(scores)

    # ================================================================
    # IS PYTHON
    # ================================================================

    @classmethod
    def is_python(
        cls,
        task: str,
    ) -> bool:
        return (
            cls.detect_language_static(
                task
            )
            == "python"
        )

    # ================================================================
    # STATIC LANGUAGE DETECTION
    # ================================================================

    @classmethod
    def detect_language_static(
        cls,
        task: str,
    ) -> str:
        """
        Versi classmethod sehingga bisa dipanggil:

            CodingAgent.detect_language_static(...)
        """

        text = str(task or "")

        blocks = cls.extract_code_blocks(
            text
        )

        if blocks:
            explicit = [
                block.language
                for block in blocks
                if block.language != "unknown"
            ]

            if explicit:
                return cls._most_common_language(
                    explicit
                )

        explicit = (
            cls._detect_explicit_language_hint(
                text
            )
        )

        if explicit != "unknown":
            return explicit

        scores = cls._score_languages(
            text
        )

        if scores:
            return max(
                scores.items(),
                key=lambda item: item[1],
            )[0]

        return "unknown"

    # ================================================================
    # FORMAT ANALYSIS
    # ================================================================

    @classmethod
    def format_analysis(
        cls,
        analysis: dict[str, Any],
    ) -> str:
        language = analysis.get(
            "language",
            "unknown",
        )

        task_type = analysis.get(
            "task_type",
            "coding",
        )

        blocks = analysis.get(
            "code_block_count",
            0,
        )

        security = analysis.get(
            "security_count",
            0,
        )

        bugs = analysis.get(
            "bug_count",
            0,
        )

        lines: list[str] = []

        lines.append(
            "=== ZAI CODING ANALYSIS ==="
        )

        lines.append(
            f"Language : {language}"
        )

        lines.append(
            f"Task     : {task_type}"
        )

        lines.append(
            f"Blocks   : {blocks}"
        )

        lines.append(
            f"Security : {security}"
        )

        lines.append(
            f"Bug      : {bugs}"
        )

        validation = analysis.get(
            "python_validation"
        )

        if validation:
            lines.append(
                "Python Syntax : "
                + (
                    "VALID"
                    if validation.get(
                        "valid"
                    )
                    else "INVALID"
                )
            )

            if validation.get(
                "error"
            ):
                lines.append(
                    "Python Error  : "
                    + str(
                        validation[
                            "error"
                        ]
                    )
                )

        return "\n".join(lines)

    # ================================================================
    # SERIALIZE ANALYSIS
    # ================================================================

    @classmethod
    def analysis_to_json(
        cls,
        analysis: dict[str, Any],
    ) -> str:
        return json.dumps(
            analysis,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # ================================================================
    # DEBUG SUGGESTIONS
    # ================================================================

    @classmethod
    def debugging_suggestions(
        cls,
        analysis: dict[str, Any],
    ) -> list[str]:
        suggestions: list[str] = []

        validation = analysis.get(
            "python_validation"
        )

        if validation and not validation.get(
            "valid"
        ):
            suggestions.append(
                "Perbaiki syntax Python "
                "berdasarkan line dan offset "
                "pada syntax error."
            )

        for finding in analysis.get(
            "security_findings",
            [],
        ):
            rule = finding.get(
                "rule",
                "",
            )

            if rule == "python_eval":
                suggestions.append(
                    "Hindari eval() untuk input "
                    "yang tidak dipercaya."
                )

            elif rule == "python_exec":
                suggestions.append(
                    "Hindari exec() pada source "
                    "dinamis."
                )

            elif rule == "python_subprocess":
                suggestions.append(
                    "Validasi command dan gunakan "
                    "API proses yang terkontrol."
                )

            elif rule == "hardcoded_password":
                suggestions.append(
                    "Pindahkan password ke secret "
                    "manager atau environment variable."
                )

            elif rule == "hardcoded_api_key":
                suggestions.append(
                    "Gunakan environment variable "
                    "atau secret manager."
                )

        for finding in analysis.get(
            "bug_findings",
            [],
        ):
            message = finding.get(
                "message"
            )

            if message:
                suggestions.append(
                    str(message)
                )

        if not suggestions:
            suggestions.append(
                "Tidak ditemukan masalah statis "
                "utama pada pemeriksaan dasar."
            )

        return list(
            dict.fromkeys(
                suggestions
            )
        )

    # ================================================================
    # PUBLIC REVIEW
    # ================================================================

    def review(
        self,
        task: str,
    ) -> dict[str, Any]:
        analysis = self.analyze_code(
            task
        )

        analysis[
            "suggestions"
        ] = self.debugging_suggestions(
            analysis
        )

        analysis[
            "formatted"
        ] = self.format_analysis(
            analysis
        )

        return analysis

    # ================================================================
    # CODE BLOCK COUNT
    # ================================================================

    @classmethod
    def code_block_count(
        cls,
        task: str,
    ) -> int:
        return len(
            cls.extract_code_blocks(
                task
            )
        )

    # ================================================================
    # LINE NUMBER HELPER
    # ================================================================

    @staticmethod
    def line_number(
        source: str,
        position: int,
    ) -> int:
        text = str(source or "")

        position = max(
            0,
            min(
                position,
                len(text),
            ),
        )

        return (
            text.count(
                "\n",
                0,
                position,
            )
            + 1
        )

    # ================================================================
    # CLEAN CODE
    # ================================================================

    @staticmethod
    def clean_code(
        code: str,
    ) -> str:
        """
        Membersihkan indentation akibat Markdown.
        """

        source = str(code or "")

        source = source.replace(
            "\r\n",
            "\n",
        )

        source = source.replace(
            "\r",
            "\n",
        )

        return textwrap.dedent(
            source
        ).strip()

    # ================================================================
    # REMOVE CODE FENCES
    # ================================================================

    @classmethod
    def remove_code_fences(
        cls,
        code: str,
    ) -> str:
        text = str(code or "")

        text = re.sub(
            r"^\s*```[A-Za-z0-9_+#.\-]*\s*\n",
            "",
            text,
            count=1,
        )

        text = re.sub(
            r"\n\s*```\s*$",
            "",
            text,
            count=1,
        )

        return cls.clean_code(
            text
        )

    # ================================================================
    # LANGUAGE FROM EXTENSION
    # ================================================================

    @classmethod
    def language_from_extension(
        cls,
        filename: str,
    ) -> str:
        name = str(
            filename or ""
        ).strip().lower()

        extension_map = {
            ".py": "python",
            ".pyw": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".kt": "kotlin",
            ".dart": "dart",
            ".c": "c",
            ".h": "c",
            ".cc": "cpp",
            ".cpp": "cpp",
            ".cxx": "cpp",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift",
            ".sh": "shell",
            ".bash": "shell",
            ".zsh": "shell",
            ".ps1": "powershell",
            ".sql": "sql",
            ".html": "html",
            ".htm": "html",
            ".css": "css",
            ".scss": "scss",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".xml": "xml",
            ".md": "markdown",
            ".txt": "text",
        }

        for extension, language in (
            extension_map.items()
        ):
            if name.endswith(extension):
                return language

        return "unknown"

    # ================================================================
    # AGENT HEALTH
    # ================================================================

    def health(self) -> dict[str, Any]:
        return {
            "agent": self.name,
            "version": self.version,
            "status": "HEALTHY",
            "capabilities": list(
                self.capabilities
            ),
            "execution_count": (
                self.execution_count
            ),
            "success_count": (
                self.success_count
            ),
            "failure_count": (
                self.failure_count
            ),
            "analysis_count": (
                self.analysis_count
            ),
            "syntax_check_count": (
                self.syntax_check_count
            ),
            "language_detection_count": (
                self.language_detection_count
            ),
            "code_extraction_count": (
                self.code_extraction_count
            ),
            "security_finding_count": (
                self.security_finding_count
            ),
            "bug_finding_count": (
                self.bug_finding_count
            ),
        }


# ====================================================================
# SELF TEST
# ====================================================================

def _self_test() -> dict[str, Any]:
    """
    Internal test untuk CodingAgent.
    """

    agent = CodingAgent()

    source = '''```python
def hello():
    print("Halo ZAI")

hello()
```'''

    language = agent.detect_language(
        source
    )

    assert language == "python", (
        f"Expected python, got {language}"
    )

    blocks = agent.extract_code_blocks(
        source
    )

    assert len(blocks) == 1

    assert blocks[0].language == "python"

    assert "def hello" in blocks[0].code

    validation = agent.validate_python(
        blocks[0].code
    )

    assert validation["valid"] is True

    metrics = agent.code_metrics(
        blocks[0].code
    )

    assert metrics["lines"] > 0

    return {
        "status": "PASS",
        "language": language,
        "blocks": len(blocks),
        "python_valid": validation[
            "valid"
        ],
        "lines": metrics["lines"],
    }


if __name__ == "__main__":
    print(
        json.dumps(
            _self_test(),
            ensure_ascii=False,
            indent=2,
        )
    )