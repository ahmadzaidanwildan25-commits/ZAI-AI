from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base_agent import BaseAgent
from .agent_result import AgentResult


class SystemCommandError(RuntimeError):
    """Error untuk operasi command sistem yang gagal."""


@dataclass
class SystemCommandResult:
    """
    Representasi hasil command sistem.

    Command yang dijalankan SystemAgent dibatasi pada command
    diagnostik/read-only yang sudah diizinkan.
    """

    command: str
    args: list[str]
    success: bool
    return_code: int
    stdout: str
    stderr: str
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SystemAgent(BaseAgent):
    """
    ZAI System Agent.

    Agent ini bertugas menangani informasi dan diagnostik sistem
    secara aman.

    Fokus utama:
    - informasi OS
    - informasi Python
    - informasi CPU
    - informasi RAM
    - informasi disk
    - informasi jaringan
    - hostname
    - IP address lokal
    - uptime
    - environment dasar
    - status runtime
    - health check
    - diagnosis sederhana
    - command diagnostik read-only

    SystemAgent TIDAK menjalankan arbitrary shell command.

    Tujuannya adalah menyediakan fondasi System Intelligence
    untuk Super ZAI tanpa memberikan agent kemampuan eksekusi
    command berbahaya secara default.
    """

    name = "system_agent"

    version = "1.0.0"

    description = (
        "System specialist agent untuk monitoring, diagnostik, "
        "informasi perangkat, OS, resource, jaringan, dan runtime ZAI."
    )

    capabilities = (
        "system",
        "system_information",
        "os_information",
        "hardware_information",
        "cpu_information",
        "memory_information",
        "disk_information",
        "network_information",
        "hostname_detection",
        "ip_detection",
        "python_information",
        "environment_information",
        "uptime_detection",
        "system_health",
        "system_diagnostics",
        "safe_command_execution",
        "read_only_commands",
    )

    # ------------------------------------------------------------------
    # SAFE COMMAND POLICY
    # ------------------------------------------------------------------

    SAFE_COMMANDS: dict[str, tuple[str, ...]] = {
        "windows": (
            "hostname",
            "whoami",
            "ver",
            "systeminfo",
            "ipconfig",
            "tasklist",
        ),
        "linux": (
            "hostname",
            "whoami",
            "uname",
            "uptime",
            "df",
            "free",
            "ip",
        ),
        "darwin": (
            "hostname",
            "whoami",
            "uname",
            "uptime",
            "df",
            "vm_stat",
        ),
    }

    BLOCKED_COMMAND_TOKENS = (
        "format",
        "shutdown",
        "restart",
        "reboot",
        "del ",
        "erase ",
        "rmdir",
        "rm ",
        "rm\t",
        "mkfs",
        "diskpart",
        "reg delete",
        "reg add",
        "net user",
        "net localgroup",
        "sc delete",
        "sc stop",
        "taskkill",
        "kill ",
        "pkill",
        "chmod",
        "chown",
        "powershell",
        "pwsh",
        "cmd /c",
        "bash -c",
        "sh -c",
        "curl ",
        "wget ",
        "invoke-webrequest",
        "iex ",
        "start-process",
        "set-executionpolicy",
    )

    # ------------------------------------------------------------------
    # REQUEST KEYWORDS
    # ------------------------------------------------------------------

    SYSTEM_KEYWORDS = (
        "system",
        "sistem",
        "os",
        "operating system",
        "windows",
        "linux",
        "macos",
        "mac",
        "komputer",
        "computer",
        "pc",
        "laptop",
        "device",
        "perangkat",
        "mesin",
        "machine",
    )

    CPU_KEYWORDS = (
        "cpu",
        "processor",
        "prosesor",
        "core",
        "cores",
        "thread",
        "threads",
        "processor usage",
        "cpu usage",
        "cpu information",
    )

    MEMORY_KEYWORDS = (
        "ram",
        "memory",
        "memori",
        "memory usage",
        "ram usage",
        "available memory",
        "free memory",
    )

    DISK_KEYWORDS = (
        "disk",
        "storage",
        "penyimpanan",
        "harddisk",
        "hard disk",
        "ssd",
        "hdd",
        "drive",
        "disk space",
        "free space",
    )

    NETWORK_KEYWORDS = (
        "network",
        "jaringan",
        "internet",
        "wifi",
        "wi-fi",
        "ethernet",
        "ip",
        "ip address",
        "alamat ip",
        "hostname",
        "dns",
        "connection",
        "koneksi",
    )

    PYTHON_KEYWORDS = (
        "python",
        "python version",
        "python environment",
        "venv",
        "virtual environment",
        "pip",
    )

    HEALTH_KEYWORDS = (
        "health",
        "healthy",
        "status",
        "sehat",
        "diagnostic",
        "diagnostics",
        "diagnosa",
        "diagnostik",
        "check system",
        "cek sistem",
        "system check",
    )

    UPTIME_KEYWORDS = (
        "uptime",
        "berapa lama",
        "menyala",
        "boot time",
        "waktu hidup",
    )

    ENVIRONMENT_KEYWORDS = (
        "environment",
        "env",
        "environment variable",
        "variabel environment",
        "path",
        "working directory",
        "current directory",
    )

    INFO_PHRASES = (
        "system information",
        "system info",
        "informasi sistem",
        "info sistem",
        "informasi komputer",
        "info komputer",
        "informasi perangkat",
        "info perangkat",
        "computer information",
        "computer info",
    )

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        command_timeout: float = 10.0,
        max_output_length: int = 20_000,
    ) -> None:
        super().__init__()

        self.command_timeout = max(1.0, float(command_timeout))
        self.max_output_length = max(1_000, int(max_output_length))

        self.system_name = platform.system().lower() or "unknown"

        self.command_count = 0
        self.command_success_count = 0
        self.command_failure_count = 0

        self.diagnostic_count = 0
        self.health_check_count = 0

        self.started_at = datetime.now(timezone.utc)

        self._last_system_snapshot: dict[str, Any] | None = None
        self._last_health_snapshot: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # BASIC INFORMATION
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """
        Informasi agent yang kompatibel dengan AgentRegistry.
        """

        data = super().info()

        data.update(
            {
                "domain": "system",
                "platform": self.system_name,
                "command_execution": True,
                "safe_command_only": True,
                "arbitrary_shell": False,
                "command_timeout": self.command_timeout,
                "command_count": self.command_count,
                "command_success_count": self.command_success_count,
                "command_failure_count": self.command_failure_count,
                "diagnostic_count": self.diagnostic_count,
                "health_check_count": self.health_check_count,
            }
        )

        return data

    # ------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """
        Health check internal SystemAgent.
        """

        self.health_check_count += 1

        try:
            snapshot = self.collect_system_snapshot()

            status = "HEALTHY"

            issues: list[str] = []

            memory = snapshot.get("memory", {})
            disk = snapshot.get("disk", {})

            memory_percent = memory.get("percent")
            disk_percent = disk.get("percent")

            if isinstance(memory_percent, (int, float)):
                if memory_percent >= 95:
                    status = "DEGRADED"
                    issues.append("Penggunaan RAM sangat tinggi.")
                elif memory_percent >= 85 and status == "HEALTHY":
                    status = "WARNING"
                    issues.append("Penggunaan RAM tinggi.")

            if isinstance(disk_percent, (int, float)):
                if disk_percent >= 95:
                    status = "DEGRADED"
                    issues.append("Penggunaan disk sangat tinggi.")
                elif disk_percent >= 85 and status == "HEALTHY":
                    status = "WARNING"
                    issues.append("Penggunaan disk tinggi.")

            result = {
                "agent": self.name,
                "version": self.version,
                "status": status,
                "platform": self.system_name,
                "issues": issues,
                "execution_count": self.execution_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "success_rate": self._success_rate(),
                "command_count": self.command_count,
                "command_success_count": self.command_success_count,
                "command_failure_count": self.command_failure_count,
                "diagnostic_count": self.diagnostic_count,
                "health_check_count": self.health_check_count,
                "system": snapshot,
            }

            self._last_health_snapshot = result

            return result

        except Exception as exc:
            return {
                "agent": self.name,
                "version": self.version,
                "status": "DEGRADED",
                "error": f"{type(exc).__name__}: {exc}",
                "execution_count": self.execution_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "success_rate": self._success_rate(),
            }

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------

    async def run(
        self,
        task: str,
        result: AgentResult,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Entry point utama BaseAgent.
        """

        normalized_task = self.normalize_task(task)

        result.add_observation(
            "system_agent_started",
            agent=self.name,
            task_length=len(task or ""),
            normalized_length=len(normalized_task),
        )

        if not normalized_task:
            result.add_warning(
                "Task kosong. SystemAgent menggunakan mode system_info."
            )

            response = self.format_system_information(
                self.collect_system_snapshot()
            )

            result.response = response
            result.complete(response)

            return result

        mode = self.detect_mode(normalized_task)

        result.add_observation(
            "system_mode_detected",
            mode=mode,
        )

        try:
            if mode == "health":
                self.diagnostic_count += 1

                health_data = self.health()

                result.metadata.update(
                    {
                        "system_mode": mode,
                        "diagnostic": True,
                        "health_status": health_data.get("status"),
                    }
                )

                response = self.format_health(health_data)

            elif mode == "cpu":
                snapshot = self.collect_cpu_information()

                result.metadata.update(
                    {
                        "system_mode": mode,
                    }
                )

                response = self.format_cpu(snapshot)

            elif mode == "memory":
                snapshot = self.collect_memory_information()

                result.metadata.update(
                    {
                        "system_mode": mode,
                    }
                )

                response = self.format_memory(snapshot)

            elif mode == "disk":
                snapshot = self.collect_disk_information()

                result.metadata.update(
                    {
                        "system_mode": mode,
                    }
                )

                response = self.format_disk(snapshot)

            elif mode == "network":
                snapshot = self.collect_network_information()

                result.metadata.update(
                    {
                        "system_mode": mode,
                    }
                )

                response = self.format_network(snapshot)

            elif mode == "python":
                snapshot = self.collect_python_information()

                result.metadata.update(
                    {
                        "system_mode": mode,
                    }
                )

                response = self.format_python(snapshot)

            elif mode == "uptime":
                snapshot = self.collect_uptime_information()

                result.metadata.update(
                    {
                        "system_mode": mode,
                    }
                )

                response = self.format_uptime(snapshot)

            elif mode == "environment":
                snapshot = self.collect_environment_information()

                result.metadata.update(
                    {
                        "system_mode": mode,
                    }
                )

                response = self.format_environment(snapshot)

            else:
                snapshot = self.collect_system_snapshot()

                result.metadata.update(
                    {
                        "system_mode": "system_info",
                        "snapshot_keys": list(snapshot.keys()),
                    }
                )

                response = self.format_system_information(snapshot)

            result.response = response

            result.add_observation(
                "system_response_generated",
                mode=mode,
                response_length=len(response),
            )

            result.complete(response)

            return result

        except Exception as exc:
            result.add_error(
                f"{type(exc).__name__}: {exc}"
            )

            result.response = (
                f"{self.name} gagal mengumpulkan informasi sistem: "
                f"{type(exc).__name__}: {exc}"
            )

            return result

    # ------------------------------------------------------------------
    # TASK NORMALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_task(task: str | None) -> str:
        if task is None:
            return ""

        text = str(task)

        text = text.replace("\x00", " ")

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ------------------------------------------------------------------
    # MODE DETECTION
    # ------------------------------------------------------------------

    def detect_mode(self, task: str) -> str:
        """
        Menentukan mode berdasarkan task user.
        """

        text = task.lower().strip()

        if self._contains_any(text, self.HEALTH_KEYWORDS):
            return "health"

        if self._contains_any(text, self.CPU_KEYWORDS):
            return "cpu"

        if self._contains_any(text, self.MEMORY_KEYWORDS):
            return "memory"

        if self._contains_any(text, self.DISK_KEYWORDS):
            return "disk"

        if self._contains_any(text, self.NETWORK_KEYWORDS):
            return "network"

        if self._contains_any(text, self.PYTHON_KEYWORDS):
            return "python"

        if self._contains_any(text, self.UPTIME_KEYWORDS):
            return "uptime"

        if self._contains_any(text, self.ENVIRONMENT_KEYWORDS):
            return "environment"

        return "system_info"

    @staticmethod
    def _contains_any(
        text: str,
        keywords: tuple[str, ...],
    ) -> bool:
        return any(
            keyword in text
            for keyword in keywords
        )

    # ------------------------------------------------------------------
    # COMPLETE SYSTEM SNAPSHOT
    # ------------------------------------------------------------------

    def collect_system_snapshot(self) -> dict[str, Any]:
        """
        Mengumpulkan snapshot sistem tanpa library eksternal.
        """

        started = time.perf_counter()

        snapshot = {
            "timestamp": self._utc_now(),
            "platform": self.collect_os_information(),
            "cpu": self.collect_cpu_information(),
            "memory": self.collect_memory_information(),
            "disk": self.collect_disk_information(),
            "network": self.collect_network_information(),
            "python": self.collect_python_information(),
            "uptime": self.collect_uptime_information(),
            "environment": self.collect_environment_information(),
        }

        snapshot["collection_latency_ms"] = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        self._last_system_snapshot = snapshot

        return snapshot

    # ------------------------------------------------------------------
    # OS INFORMATION
    # ------------------------------------------------------------------

    def collect_os_information(self) -> dict[str, Any]:
        uname = platform.uname()

        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "node": uname.node,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        }

    # ------------------------------------------------------------------
    # CPU INFORMATION
    # ------------------------------------------------------------------

    def collect_cpu_information(self) -> dict[str, Any]:
        logical = os.cpu_count() or 1

        physical = self._get_physical_cpu_count()

        load_average: list[float] | None = None

        try:
            load_average = [
                round(float(value), 2)
                for value in os.getloadavg()
            ]
        except (AttributeError, OSError):
            load_average = None

        cpu_usage = self._get_cpu_usage_windows()

        return {
            "logical_cores": logical,
            "physical_cores": physical,
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
            "usage_percent": cpu_usage,
            "load_average": load_average,
        }

    def _get_physical_cpu_count(self) -> int | None:
        """
        Mengambil physical core count jika memungkinkan.

        Tidak bergantung pada psutil.
        """

        if self.system_name == "windows":
            try:
                output = subprocess.check_output(
                    [
                        "wmic",
                        "cpu",
                        "get",
                        "NumberOfCores",
                        "/value",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )

                values = re.findall(
                    r"NumberOfCores\s*=\s*(\d+)",
                    output,
                    flags=re.IGNORECASE,
                )

                if values:
                    return sum(
                        int(value)
                        for value in values
                    )

            except (
                subprocess.SubprocessError,
                FileNotFoundError,
                OSError,
                ValueError,
            ):
                pass

        return None

    def _get_cpu_usage_windows(self) -> float | None:
        """
        CPU usage Windows menggunakan PowerShell WMI/CIM.

        Jika command tidak tersedia, mengembalikan None.
        """

        if self.system_name != "windows":
            return None

        command = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "(Get-CimInstance Win32_Processor | "
                "Measure-Object -Property LoadPercentage -Average).Average"
            ),
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=4,
                creationflags=getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0,
                ),
            )

            if completed.returncode != 0:
                return None

            value = completed.stdout.strip()

            if not value:
                return None

            return round(float(value), 2)

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            OSError,
            ValueError,
        ):
            return None

    # ------------------------------------------------------------------
    # MEMORY INFORMATION
    # ------------------------------------------------------------------

    def collect_memory_information(self) -> dict[str, Any]:
        """
        Mengambil memory information.

        Menggunakan:
        - Windows API melalui ctypes
        - Linux /proc/meminfo
        - fallback None
        """

        if self.system_name == "windows":
            result = self._collect_windows_memory()

            if result:
                return result

        if self.system_name == "linux":
            result = self._collect_linux_memory()

            if result:
                return result

        return {
            "total_bytes": None,
            "available_bytes": None,
            "used_bytes": None,
            "free_bytes": None,
            "percent": None,
            "total_gb": None,
            "available_gb": None,
            "used_gb": None,
        }

    def _collect_windows_memory(self) -> dict[str, Any] | None:
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()

            status.dwLength = ctypes.sizeof(
                MEMORYSTATUSEX
            )

            success = ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)
            )

            if not success:
                return None

            total = int(status.ullTotalPhys)
            available = int(status.ullAvailPhys)
            used = total - available

            return {
                "total_bytes": total,
                "available_bytes": available,
                "used_bytes": used,
                "free_bytes": available,
                "percent": float(status.dwMemoryLoad),
                "total_gb": self._bytes_to_gb(total),
                "available_gb": self._bytes_to_gb(available),
                "used_gb": self._bytes_to_gb(used),
            }

        except Exception:
            return None

    def _collect_linux_memory(self) -> dict[str, Any] | None:
        path = Path("/proc/meminfo")

        if not path.exists():
            return None

        try:
            data = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            values: dict[str, int] = {}

            for line in data.splitlines():
                match = re.match(
                    r"^([A-Za-z_]+):\s+(\d+)\s+kB",
                    line,
                )

                if match:
                    values[match.group(1)] = (
                        int(match.group(2)) * 1024
                    )

            total = values.get("MemTotal")
            available = values.get("MemAvailable")

            if total is None or available is None:
                return None

            used = total - available

            percent = (
                (used / total) * 100
                if total
                else None
            )

            return {
                "total_bytes": total,
                "available_bytes": available,
                "used_bytes": used,
                "free_bytes": values.get("MemFree"),
                "percent": round(percent, 2)
                if percent is not None
                else None,
                "total_gb": self._bytes_to_gb(total),
                "available_gb": self._bytes_to_gb(available),
                "used_gb": self._bytes_to_gb(used),
            }

        except OSError:
            return None

    # ------------------------------------------------------------------
    # DISK INFORMATION
    # ------------------------------------------------------------------

    def collect_disk_information(
        self,
        path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Mengambil kapasitas disk.

        Default:
        - Windows: current drive
        - Linux/macOS: root filesystem
        """

        if path is None:
            if self.system_name == "windows":
                path = Path.cwd().anchor or "C:\\"
            else:
                path = "/"

        try:
            usage = shutil.disk_usage(path)

            total = int(usage.total)
            used = int(usage.used)
            free = int(usage.free)

            percent = (
                used / total * 100
                if total
                else 0.0
            )

            return {
                "path": str(path),
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": free,
                "percent": round(percent, 2),
                "total_gb": self._bytes_to_gb(total),
                "used_gb": self._bytes_to_gb(used),
                "free_gb": self._bytes_to_gb(free),
            }

        except OSError as exc:
            return {
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
                "total_bytes": None,
                "used_bytes": None,
                "free_bytes": None,
                "percent": None,
            }

    # ------------------------------------------------------------------
    # NETWORK INFORMATION
    # ------------------------------------------------------------------

    def collect_network_information(self) -> dict[str, Any]:
        hostname = socket.gethostname()

        addresses: list[str] = []

        try:
            infos = socket.getaddrinfo(
                hostname,
                None,
                socket.AF_INET,
            )

            for info in infos:
                address = info[4][0]

                if address not in addresses:
                    addresses.append(address)

        except socket.gaierror:
            pass

        local_ip = self._detect_local_ip()

        if local_ip and local_ip not in addresses:
            addresses.insert(
                0,
                local_ip,
            )

        return {
            "hostname": hostname,
            "local_ip": local_ip,
            "addresses": addresses,
            "fqdn": socket.getfqdn(),
        }

    def _detect_local_ip(self) -> str | None:
        """
        Mendeteksi IP lokal menggunakan UDP socket.

        Tidak mengirim data aplikasi.
        """

        sock: socket.socket | None = None

        try:
            sock = socket.socket(
                socket.AF_INET,
                socket.SOCK_DGRAM,
            )

            sock.settimeout(1.0)

            sock.connect(
                ("8.8.8.8", 80)
            )

            address = sock.getsockname()[0]

            return address

        except OSError:
            return None

        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # PYTHON INFORMATION
    # ------------------------------------------------------------------

    def collect_python_information(self) -> dict[str, Any]:
        executable = sys.executable

        version_info = sys.version_info

        virtual_env = (
            os.environ.get("VIRTUAL_ENV")
            or os.environ.get("CONDA_PREFIX")
        )

        return {
            "version": platform.python_version(),
            "version_full": sys.version,
            "major": version_info.major,
            "minor": version_info.minor,
            "micro": version_info.micro,
            "implementation": platform.python_implementation(),
            "executable": executable,
            "prefix": sys.prefix,
            "base_prefix": getattr(
                sys,
                "base_prefix",
                None,
            ),
            "virtual_environment": virtual_env,
            "venv_active": bool(virtual_env),
        }

    # ------------------------------------------------------------------
    # UPTIME
    # ------------------------------------------------------------------

    def collect_uptime_information(self) -> dict[str, Any]:
        now = time.time()

        if self.system_name == "windows":
            milliseconds = self._windows_uptime_ms()

            if milliseconds is not None:
                seconds = milliseconds / 1000.0

                return {
                    "available": True,
                    "seconds": round(seconds, 2),
                    "human": self._format_duration(seconds),
                }

        if self.system_name == "linux":
            path = Path("/proc/uptime")

            try:
                if path.exists():
                    content = path.read_text(
                        encoding="utf-8"
                    ).strip()

                    seconds = float(
                        content.split()[0]
                    )

                    return {
                        "available": True,
                        "seconds": round(seconds, 2),
                        "human": self._format_duration(seconds),
                    }

            except (
                OSError,
                ValueError,
                IndexError,
            ):
                pass

        return {
            "available": False,
            "seconds": None,
            "human": "Unavailable",
            "timestamp": now,
        }

    def _windows_uptime_ms(self) -> int | None:
        try:
            import ctypes

            return int(
                ctypes.windll.kernel32.GetTickCount64()
            )

        except Exception:
            return None

    # ------------------------------------------------------------------
    # ENVIRONMENT INFORMATION
    # ------------------------------------------------------------------

    def collect_environment_information(self) -> dict[str, Any]:
        cwd = os.getcwd()

        path_value = os.environ.get(
            "PATH",
            "",
        )

        path_entries = [
            item
            for item in path_value.split(os.pathsep)
            if item
        ]

        virtual_env = os.environ.get(
            "VIRTUAL_ENV"
        )

        return {
            "cwd": cwd,
            "pythonpath": os.environ.get(
                "PYTHONPATH"
            ),
            "virtual_env": virtual_env,
            "venv_active": bool(virtual_env),
            "path_entry_count": len(path_entries),
            "path_entries": path_entries[:50],
            "shell": os.environ.get(
                "SHELL"
            ) or os.environ.get(
                "COMSPEC"
            ),
            "user": os.environ.get(
                "USERNAME"
            ) or os.environ.get(
                "USER"
            ),
        }

    # ------------------------------------------------------------------
    # SAFE COMMAND EXECUTION
    # ------------------------------------------------------------------

    async def execute_safe_command(
        self,
        command: str,
        args: list[str] | tuple[str, ...] | None = None,
        *,
        timeout: float | None = None,
    ) -> SystemCommandResult:
        """
        Menjalankan command diagnostik yang masuk whitelist.

        Tidak menerima shell=True.

        Tidak menerima pipeline, redirection,
        command chaining, atau shell expansion.
        """

        command = str(command).strip()

        if not command:
            raise SystemCommandError(
                "Command tidak boleh kosong."
            )

        args_list = [
            str(value)
            for value in (args or [])
        ]

        self._validate_safe_command(
            command,
            args_list,
        )

        self.command_count += 1

        started = time.perf_counter()

        try:
            completed = await asyncio.to_thread(
                self._run_subprocess,
                command,
                args_list,
                timeout or self.command_timeout,
            )

            duration_ms = round(
                (time.perf_counter() - started) * 1000,
                2,
            )

            stdout = self._truncate_output(
                completed.stdout
            )

            stderr = self._truncate_output(
                completed.stderr
            )

            success = (
                completed.returncode == 0
            )

            if success:
                self.command_success_count += 1
            else:
                self.command_failure_count += 1

            return SystemCommandResult(
                command=command,
                args=args_list,
                success=success,
                return_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
            )

        except Exception:
            self.command_failure_count += 1
            raise

    def _validate_safe_command(
        self,
        command: str,
        args: list[str],
    ) -> None:
        command_lower = command.lower().strip()

        if any(
            token in command_lower
            for token in self.BLOCKED_COMMAND_TOKENS
        ):
            raise SystemCommandError(
                f"Command '{command}' diblokir oleh security policy."
            )

        if any(
            token in argument.lower()
            for argument in args
            for token in self.BLOCKED_COMMAND_TOKENS
        ):
            raise SystemCommandError(
                "Argument command mengandung token yang diblokir."
            )

        # Mencegah shell operators.
        forbidden_symbols = (
            "|",
            ">",
            "<",
            "&&",
            "||",
            ";",
            "`",
            "$(",
        )

        combined = " ".join(
            [command, *args]
        )

        if any(
            symbol in combined
            for symbol in forbidden_symbols
        ):
            raise SystemCommandError(
                "Shell operator tidak diperbolehkan."
            )

        allowed = self.SAFE_COMMANDS.get(
            self.system_name,
            (),
        )

        executable_name = Path(
            command
        ).name.lower()

        normalized_allowed = {
            Path(item).name.lower()
            for item in allowed
        }

        if executable_name not in normalized_allowed:
            raise SystemCommandError(
                f"Command '{command}' tidak masuk whitelist "
                f"platform '{self.system_name}'."
            )

    def _run_subprocess(
        self,
        command: str,
        args: list[str],
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        executable = shutil.which(command)

        if executable is None:
            raise SystemCommandError(
                f"Executable '{command}' tidak ditemukan."
            )

        creationflags = getattr(
            subprocess,
            "CREATE_NO_WINDOW",
            0,
        )

        return subprocess.run(
            [
                executable,
                *args,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
            creationflags=creationflags,
        )

    # ------------------------------------------------------------------
    # DIAGNOSTIC COMMAND HELPERS
    # ------------------------------------------------------------------

    async def command_hostname(
        self,
    ) -> SystemCommandResult:
        return await self.execute_safe_command(
            "hostname"
        )

    async def command_whoami(
        self,
    ) -> SystemCommandResult:
        return await self.execute_safe_command(
            "whoami"
        )

    async def command_system_info(
        self,
    ) -> SystemCommandResult:
        if self.system_name == "windows":
            return await self.execute_safe_command(
                "systeminfo"
            )

        return await self.execute_safe_command(
            "uname",
            ["-a"],
        )

    async def command_network_info(
        self,
    ) -> SystemCommandResult:
        if self.system_name == "windows":
            return await self.execute_safe_command(
                "ipconfig"
            )

        return await self.execute_safe_command(
            "ip",
            ["addr"],
        )

    async def command_process_info(
        self,
    ) -> SystemCommandResult:
        if self.system_name == "windows":
            return await self.execute_safe_command(
                "tasklist"
            )

        raise SystemCommandError(
            "Process listing command belum diaktifkan "
            "untuk platform ini."
        )

    # ------------------------------------------------------------------
    # FORMATTING
    # ------------------------------------------------------------------

    def format_system_information(
        self,
        snapshot: dict[str, Any],
    ) -> str:
        platform_info = snapshot.get(
            "platform",
            {},
        )

        cpu = snapshot.get(
            "cpu",
            {},
        )

        memory = snapshot.get(
            "memory",
            {},
        )

        disk = snapshot.get(
            "disk",
            {},
        )

        network = snapshot.get(
            "network",
            {},
        )

        python_info = snapshot.get(
            "python",
            {},
        )

        uptime = snapshot.get(
            "uptime",
            {},
        )

        lines = [
            "ZAI System Agent",
            f"Status: READY",
            "",
            "=== OPERATING SYSTEM ===",
            f"System: {self._display(platform_info.get('system'))}",
            f"Release: {self._display(platform_info.get('release'))}",
            f"Version: {self._display(platform_info.get('version'))}",
            f"Architecture: {self._display(platform_info.get('architecture'))}",
            f"Machine: {self._display(platform_info.get('machine'))}",
            f"Hostname: {self._display(platform_info.get('hostname'))}",
            "",
            "=== CPU ===",
            f"Processor: {self._display(cpu.get('processor'))}",
            f"Physical cores: {self._display(cpu.get('physical_cores'))}",
            f"Logical cores: {self._display(cpu.get('logical_cores'))}",
            f"CPU usage: {self._format_percent(cpu.get('usage_percent'))}",
            "",
            "=== MEMORY ===",
            f"Total RAM: {self._format_gb(memory.get('total_gb'))}",
            f"Used RAM: {self._format_gb(memory.get('used_gb'))}",
            f"Available RAM: {self._format_gb(memory.get('available_gb'))}",
            f"RAM usage: {self._format_percent(memory.get('percent'))}",
            "",
            "=== DISK ===",
            f"Path: {self._display(disk.get('path'))}",
            f"Total: {self._format_gb(disk.get('total_gb'))}",
            f"Used: {self._format_gb(disk.get('used_gb'))}",
            f"Free: {self._format_gb(disk.get('free_gb'))}",
            f"Disk usage: {self._format_percent(disk.get('percent'))}",
            "",
            "=== NETWORK ===",
            f"Hostname: {self._display(network.get('hostname'))}",
            f"Local IP: {self._display(network.get('local_ip'))}",
            f"FQDN: {self._display(network.get('fqdn'))}",
            "",
            "=== PYTHON ===",
            f"Version: {self._display(python_info.get('version'))}",
            f"Implementation: {self._display(python_info.get('implementation'))}",
            f"Executable: {self._display(python_info.get('executable'))}",
            f"Virtual environment: {self._display(python_info.get('virtual_environment'))}",
            "",
            "=== UPTIME ===",
            f"Uptime: {self._display(uptime.get('human'))}",
            "",
            "=== COLLECTION ===",
            f"Latency: {self._display(snapshot.get('collection_latency_ms'))} ms",
        ]

        return "\n".join(lines)

    def format_health(
        self,
        health_data: dict[str, Any],
    ) -> str:
        lines = [
            "ZAI System Health",
            f"Status: {health_data.get('status', 'UNKNOWN')}",
            f"Platform: {health_data.get('platform', 'UNKNOWN')}",
            "",
            "=== AGENT ===",
            f"Execution count: {health_data.get('execution_count', 0)}",
            f"Success count: {health_data.get('success_count', 0)}",
            f"Failure count: {health_data.get('failure_count', 0)}",
            f"Success rate: {health_data.get('success_rate', 0.0)}%",
            "",
            "=== COMMAND ===",
            f"Command count: {health_data.get('command_count', 0)}",
            f"Command success: {health_data.get('command_success_count', 0)}",
            f"Command failure: {health_data.get('command_failure_count', 0)}",
            "",
            "=== ISSUES ===",
        ]

        issues = health_data.get(
            "issues",
            [],
        )

        if issues:
            lines.extend(
                f"- {issue}"
                for issue in issues
            )
        else:
            lines.append(
                "- Tidak ada masalah utama yang terdeteksi."
            )

        return "\n".join(lines)

    def format_cpu(
        self,
        data: dict[str, Any],
    ) -> str:
        return "\n".join(
            [
                "ZAI CPU Information",
                f"Processor: {self._display(data.get('processor'))}",
                f"Physical cores: {self._display(data.get('physical_cores'))}",
                f"Logical cores: {self._display(data.get('logical_cores'))}",
                f"Architecture: {self._display(data.get('architecture'))}",
                f"CPU usage: {self._format_percent(data.get('usage_percent'))}",
                f"Load average: {self._display(data.get('load_average'))}",
            ]
        )

    def format_memory(
        self,
        data: dict[str, Any],
    ) -> str:
        return "\n".join(
            [
                "ZAI Memory Information",
                f"Total: {self._format_gb(data.get('total_gb'))}",
                f"Used: {self._format_gb(data.get('used_gb'))}",
                f"Available: {self._format_gb(data.get('available_gb'))}",
                f"Free: {self._format_gb(data.get('free_bytes') / (1024 ** 3) if isinstance(data.get('free_bytes'), (int, float)) else None)}",
                f"Usage: {self._format_percent(data.get('percent'))}",
            ]
        )

    def format_disk(
        self,
        data: dict[str, Any],
    ) -> str:
        return "\n".join(
            [
                "ZAI Disk Information",
                f"Path: {self._display(data.get('path'))}",
                f"Total: {self._format_gb(data.get('total_gb'))}",
                f"Used: {self._format_gb(data.get('used_gb'))}",
                f"Free: {self._format_gb(data.get('free_gb'))}",
                f"Usage: {self._format_percent(data.get('percent'))}",
            ]
        )

    def format_network(
        self,
        data: dict[str, Any],
    ) -> str:
        addresses = data.get(
            "addresses",
            [],
        )

        lines = [
            "ZAI Network Information",
            f"Hostname: {self._display(data.get('hostname'))}",
            f"Local IP: {self._display(data.get('local_ip'))}",
            f"FQDN: {self._display(data.get('fqdn'))}",
            "",
            "Addresses:",
        ]

        if addresses:
            lines.extend(
                f"- {address}"
                for address in addresses
            )
        else:
            lines.append(
                "- Tidak ada address tambahan."
            )

        return "\n".join(lines)

    def format_python(
        self,
        data: dict[str, Any],
    ) -> str:
        return "\n".join(
            [
                "ZAI Python Environment",
                f"Version: {self._display(data.get('version'))}",
                f"Implementation: {self._display(data.get('implementation'))}",
                f"Executable: {self._display(data.get('executable'))}",
                f"Prefix: {self._display(data.get('prefix'))}",
                f"Base prefix: {self._display(data.get('base_prefix'))}",
                f"Virtual environment: {self._display(data.get('virtual_environment'))}",
                f"Venv active: {self._display(data.get('venv_active'))}",
            ]
        )

    def format_uptime(
        self,
        data: dict[str, Any],
    ) -> str:
        return "\n".join(
            [
                "ZAI System Uptime",
                f"Available: {self._display(data.get('available'))}",
                f"Uptime: {self._display(data.get('human'))}",
                f"Seconds: {self._display(data.get('seconds'))}",
            ]
        )

    def format_environment(
        self,
        data: dict[str, Any],
    ) -> str:
        path_entries = data.get(
            "path_entries",
            [],
        )

        lines = [
            "ZAI Environment Information",
            f"Working directory: {self._display(data.get('cwd'))}",
            f"Python path: {self._display(data.get('pythonpath'))}",
            f"Virtual environment: {self._display(data.get('virtual_env'))}",
            f"Venv active: {self._display(data.get('venv_active'))}",
            f"User: {self._display(data.get('user'))}",
            f"Shell: {self._display(data.get('shell'))}",
            f"PATH entries: {data.get('path_entry_count', 0)}",
        ]

        if path_entries:
            lines.append("")
            lines.append("PATH:")

            lines.extend(
                f"- {entry}"
                for entry in path_entries
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # UTILITY METHODS
    # ------------------------------------------------------------------

    def _success_rate(self) -> float:
        if self.execution_count <= 0:
            return 0.0

        return round(
            (
                self.success_count
                / self.execution_count
            ) * 100,
            2,
        )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _bytes_to_gb(
        value: int | float | None,
    ) -> float | None:
        if value is None:
            return None

        return round(
            float(value) / (1024 ** 3),
            2,
        )

    @staticmethod
    def _format_duration(
        seconds: float,
    ) -> str:
        seconds = max(
            0,
            int(seconds),
        )

        days, remainder = divmod(
            seconds,
            86400,
        )

        hours, remainder = divmod(
            remainder,
            3600,
        )

        minutes, seconds = divmod(
            remainder,
            60,
        )

        parts: list[str] = []

        if days:
            parts.append(
                f"{days} hari"
            )

        if hours:
            parts.append(
                f"{hours} jam"
            )

        if minutes:
            parts.append(
                f"{minutes} menit"
            )

        if not parts or seconds:
            parts.append(
                f"{seconds} detik"
            )

        return " ".join(parts)

    @staticmethod
    def _format_gb(
        value: float | int | None,
    ) -> str:
        if value is None:
            return "Unavailable"

        return f"{float(value):.2f} GB"

    @staticmethod
    def _format_percent(
        value: float | int | None,
    ) -> str:
        if value is None:
            return "Unavailable"

        return f"{float(value):.2f}%"

    @staticmethod
    def _display(
        value: Any,
    ) -> str:
        if value is None:
            return "Unavailable"

        if isinstance(value, bool):
            return "Yes" if value else "No"

        return str(value)

    def _truncate_output(
        self,
        value: str,
    ) -> str:
        value = value or ""

        if len(value) <= self.max_output_length:
            return value

        return (
            value[: self.max_output_length]
            + "\n...[OUTPUT TRUNCATED BY ZAI]..."
        )

    # ------------------------------------------------------------------
    # PUBLIC SNAPSHOT HELPERS
    # ------------------------------------------------------------------

    def last_snapshot(
        self,
    ) -> dict[str, Any] | None:
        return self._last_system_snapshot

    def last_health(
        self,
    ) -> dict[str, Any] | None:
        return self._last_health_snapshot

    def uptime_seconds(
        self,
    ) -> float | None:
        data = self.collect_uptime_information()

        value = data.get(
            "seconds"
        )

        if isinstance(value, (int, float)):
            return float(value)

        return None

    # ------------------------------------------------------------------
    # DIAGNOSTIC SUMMARY
    # ------------------------------------------------------------------

    def diagnostic_summary(self) -> dict[str, Any]:
        """
        Snapshot ringkas yang mudah digunakan oleh orchestrator.
        """

        snapshot = self.collect_system_snapshot()

        memory = snapshot.get(
            "memory",
            {},
        )

        disk = snapshot.get(
            "disk",
            {},
        )

        cpu = snapshot.get(
            "cpu",
            {},
        )

        return {
            "agent": self.name,
            "version": self.version,
            "status": "READY",
            "platform": self.system_name,
            "hostname": snapshot.get(
                "network",
                {},
            ).get("hostname"),
            "cpu_usage": cpu.get(
                "usage_percent"
            ),
            "memory_usage": memory.get(
                "percent"
            ),
            "disk_usage": disk.get(
                "percent"
            ),
            "uptime": snapshot.get(
                "uptime",
                {},
            ).get("human"),
            "python_version": snapshot.get(
                "python",
                {},
            ).get("version"),
            "collection_latency_ms": snapshot.get(
                "collection_latency_ms"
            ),
        }

    # ------------------------------------------------------------------
    # COMMAND RESULT FORMAT
    # ------------------------------------------------------------------

    @staticmethod
    def format_command_result(
        command_result: SystemCommandResult,
    ) -> str:
        status = (
            "SUCCESS"
            if command_result.success
            else "FAILED"
        )

        lines = [
            "ZAI System Command",
            f"Status: {status}",
            f"Command: {command_result.command}",
            f"Return code: {command_result.return_code}",
            f"Latency: {command_result.duration_ms} ms",
            "",
            "STDOUT:",
            command_result.stdout or "(empty)",
        ]

        if command_result.stderr:
            lines.extend(
                [
                    "",
                    "STDERR:",
                    command_result.stderr,
                ]
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serialisasi lengkap SystemAgent.
        """

        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": list(
                self.capabilities
            ),
            "platform": self.system_name,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self._success_rate(),
            "command_count": self.command_count,
            "command_success_count": self.command_success_count,
            "command_failure_count": self.command_failure_count,
            "diagnostic_count": self.diagnostic_count,
            "health_check_count": self.health_check_count,
            "command_timeout": self.command_timeout,
            "safe_command_only": True,
            "arbitrary_shell": False,
            "started_at": self.started_at.isoformat(),
        }


# ============================================================================
# SELF TEST
# ============================================================================

def _self_test() -> dict[str, Any]:
    """
    Self-test internal.

    Tidak menjalankan command berbahaya.
    """

    agent = SystemAgent()

    snapshot = agent.collect_system_snapshot()

    assert isinstance(
        snapshot,
        dict,
    )

    assert "platform" in snapshot
    assert "cpu" in snapshot
    assert "memory" in snapshot
    assert "disk" in snapshot
    assert "network" in snapshot
    assert "python" in snapshot
    assert "uptime" in snapshot
    assert "environment" in snapshot

    mode_tests = {
        "cpu": agent.detect_mode(
            "berapa cpu komputer saya"
        ),
        "memory": agent.detect_mode(
            "berapa ram yang tersedia"
        ),
        "disk": agent.detect_mode(
            "cek storage disk"
        ),
        "network": agent.detect_mode(
            "cek jaringan dan ip"
        ),
        "python": agent.detect_mode(
            "versi python saya berapa"
        ),
        "uptime": agent.detect_mode(
            "berapa lama komputer menyala"
        ),
        "environment": agent.detect_mode(
            "cek environment python"
        ),
        "health": agent.detect_mode(
            "cek health sistem"
        ),
    }

    for expected, actual in mode_tests.items():
        assert actual == expected, (
            f"Expected mode {expected}, "
            f"got {actual}"
        )

    assert agent.info()["name"] == "system_agent"

    health = agent.health()

    assert isinstance(
        health,
        dict,
    )

    return {
        "success": True,
        "agent": agent.name,
        "version": agent.version,
        "snapshot_keys": list(
            snapshot.keys()
        ),
        "mode_tests": mode_tests,
        "health_status": health.get(
            "status"
        ),
    }


if __name__ == "__main__":
    print(
        "SystemAgent module."
    )
    print(
        "Gunakan: "
        "python -m ai.agents.system_agent"
    )
    print()

    try:
        test_result = _self_test()

        print(
            test_result
        )

        print(
            "SYSTEM_AGENT_SELF_TEST_OK"
        )

    except Exception as exc:
        print(
            f"SYSTEM_AGENT_SELF_TEST_FAILED: "
            f"{type(exc).__name__}: {exc}"
        )
        raise