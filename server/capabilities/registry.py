from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import importlib
import json


@dataclass
class Capability:
    name: str
    category: str
    status: str
    description: str
    module: Optional[str] = None
    entrypoint: Optional[str] = None
    version: Optional[str] = None
    health: bool = False
    source: str = "registry"

    def to_dict(self) -> dict:
        return asdict(self)


class CapabilityRegistry:
    """
    Central registry for ZAI capabilities.

    Status values:
        ACTIVE
        PARTIAL
        NOT_ACTIVE
        PLACEHOLDER
        ERROR
    """

    VERSION = "1.0.0"

    def __init__(self) -> None:
        self._capabilities: Dict[str, Capability] = {}
        self._created_at = datetime.now(timezone.utc).isoformat()

        self._register_builtin_capabilities()
        self.refresh_health()

    # --------------------------------------------------
    # REGISTRATION
    # --------------------------------------------------

    def register(
        self,
        name: str,
        category: str,
        status: str,
        description: str,
        module: Optional[str] = None,
        entrypoint: Optional[str] = None,
        version: Optional[str] = None,
        health: bool = False,
        source: str = "registry",
    ) -> None:

        status = status.upper()

        allowed = {
            "ACTIVE",
            "PARTIAL",
            "NOT_ACTIVE",
            "PLACEHOLDER",
            "ERROR",
        }

        if status not in allowed:
            raise ValueError(
                f"Invalid capability status: {status}"
            )

        self._capabilities[name] = Capability(
            name=name,
            category=category,
            status=status,
            description=description,
            module=module,
            entrypoint=entrypoint,
            version=version,
            health=health,
            source=source,
        )

    # --------------------------------------------------
    # BUILT-IN CAPABILITIES
    # --------------------------------------------------

    def _register_builtin_capabilities(self) -> None:

        # CORE AI
        self.register(
            "brain",
            "intelligence",
            "ACTIVE",
            "Core cognitive brain.",
            "ai.brain",
            "AIBrain",
            "0.11.0",
        )

        self.register(
            "cognitive_orchestrator",
            "intelligence",
            "ACTIVE",
            "Central cognitive orchestration.",
            "ai.cognitive_orchestrator",
            "CognitiveOrchestrator",
            "0.11.0",
        )

        self.register(
            "agent_loop",
            "intelligence",
            "ACTIVE",
            "Agent execution loop.",
            "ai.agent_loop",
            "AgentLoop",
            "0.11.0",
        )

        self.register(
            "context_manager",
            "intelligence",
            "ACTIVE",
            "Conversation and reasoning context.",
            "ai.context",
            "ContextManager",
            "0.11.0",
        )

        self.register(
            "planner",
            "intelligence",
            "ACTIVE",
            "Task planning engine.",
            "ai.planner",
            "Planner",
            "0.11.0",
        )

        self.register(
            "reasoning_engine",
            "intelligence",
            "ACTIVE",
            "Reasoning engine.",
            "ai.reasoning",
            "ReasoningEngine",
            "0.11.0",
        )

        # MEMORY
        self.register(
            "memory",
            "memory",
            "ACTIVE",
            "Persistent SQLite memory system.",
            "memory.memory_manager",
            "MemoryManager",
            "0.1.0",
        )

        self.register(
            "memory_database",
            "memory",
            "ACTIVE",
            "Persistent memory database.",
            "memory.memory_database",
            "MemoryDatabase",
        )

        # TOOLS
        self.register(
            "tool_engine",
            "tools",
            "ACTIVE",
            "Central tool execution engine.",
            "core.tool_engine",
            "ToolEngine",
            "1.7.0",
        )

        self.register(
            "calculator",
            "tools",
            "ACTIVE",
            "Safe mathematical calculator.",
            "core.tool_engine",
            "calculator",
        )

        self.register(
            "weather",
            "tools",
            "ACTIVE",
            "Weather lookup capability.",
            "core.tool_engine",
            "weather",
        )

        self.register(
            "search",
            "tools",
            "ACTIVE",
            "Web search capability.",
            "core.tool_engine",
            "search",
        )

        self.register(
            "fetch",
            "tools",
            "ACTIVE",
            "Web resource fetching.",
            "core.tool_engine",
            "fetch",
        )

        # AGENTS
        self.register(
            "coding_agent",
            "agents",
            "NOT_ACTIVE",
            "Software development agent.",
            "core.agents.coding",
        )

        self.register(
            "debugger_agent",
            "agents",
            "NOT_ACTIVE",
            "Debugging and diagnosis agent.",
            "core.agents.debugger",
        )

        self.register(
            "developer_agent",
            "agents",
            "NOT_ACTIVE",
            "Development orchestration agent.",
            "core.agents.developer",
        )

        self.register(
            "research_agent",
            "agents",
            "NOT_ACTIVE",
            "Research agent.",
            "core.agents.research",
        )

        self.register(
            "file_agent",
            "agents",
            "NOT_ACTIVE",
            "File management agent.",
            "core.agents.file",
        )

        self.register(
            "browser_agent",
            "agents",
            "NOT_ACTIVE",
            "Browser automation agent.",
            "core.agents.browser",
        )

        self.register(
            "computer_agent",
            "agents",
            "NOT_ACTIVE",
            "Computer control agent.",
            "core.agents.computer",
        )

        self.register(
            "android_agent",
            "agents",
            "NOT_ACTIVE",
            "Android control agent.",
            "core.agents.android",
        )

        self.register(
            "testing_agent",
            "agents",
            "NOT_ACTIVE",
            "Testing and validation agent.",
            "core.agents.testing",
        )

        self.register(
            "deployment_agent",
            "agents",
            "NOT_ACTIVE",
            "Deployment agent.",
            "core.agents.deployment",
        )

        self.register(
            "automation_agent",
            "agents",
            "NOT_ACTIVE",
            "Automation agent.",
            "core.agents.automation",
        )

        # KNOWLEDGE
        self.register(
            "knowledge",
            "knowledge",
            "NOT_ACTIVE",
            "Knowledge management system.",
            "core.knowledge",
        )

        self.register(
            "rag",
            "knowledge",
            "NOT_ACTIVE",
            "Retrieval augmented generation.",
            "core.knowledge.rag",
        )

        self.register(
            "embeddings",
            "knowledge",
            "NOT_ACTIVE",
            "Embedding/index system.",
            "core.knowledge.embeddings",
        )

        self.register(
            "web_knowledge",
            "knowledge",
            "PARTIAL",
            "Web-based knowledge acquisition.",
            "core.knowledge.web",
        )

        # VISION
        self.register(
            "vision",
            "vision",
            "NOT_ACTIVE",
            "Computer vision subsystem.",
            "core.vision",
        )

        self.register(
            "ocr",
            "vision",
            "NOT_ACTIVE",
            "Optical character recognition.",
            "core.vision.ocr",
        )

        self.register(
            "screen_analysis",
            "vision",
            "NOT_ACTIVE",
            "Screen understanding.",
            "core.vision.screen",
        )

        self.register(
            "object_detection",
            "vision",
            "NOT_ACTIVE",
            "Object detection.",
            "core.vision.object_detection",
        )

        # VOICE
        self.register(
            "voice",
            "voice",
            "NOT_ACTIVE",
            "Voice subsystem.",
            "core.voice",
        )

        self.register(
            "speech_to_text",
            "voice",
            "NOT_ACTIVE",
            "Speech recognition.",
            "core.voice.stt",
        )

        self.register(
            "text_to_speech",
            "voice",
            "NOT_ACTIVE",
            "Speech synthesis.",
            "core.voice.tts",
        )

        self.register(
            "wake_word",
            "voice",
            "NOT_ACTIVE",
            "Wake word detection.",
            "core.voice.wake_word",
        )

        # AUTOMATION
        self.register(
            "automation",
            "automation",
            "PARTIAL",
            "Automation subsystem.",
            "core.automation",
        )

        self.register(
            "workflows",
            "automation",
            "NOT_ACTIVE",
            "Workflow execution.",
            "core.automation.workflows",
        )

        self.register(
            "schedules",
            "automation",
            "NOT_ACTIVE",
            "Scheduled automation.",
            "core.automation.schedules",
        )

        self.register(
            "triggers",
            "automation",
            "NOT_ACTIVE",
            "Event-driven triggers.",
            "core.automation.triggers",
        )

        # SECURITY
        self.register(
            "security",
            "security",
            "PARTIAL",
            "ZAI security subsystem.",
            "core.security",
        )

        self.register(
            "approval_manager",
            "security",
            "ACTIVE",
            "Human approval gate.",
            "core.security.approvals.approval_manager",
            "ApprovalManager",
        )

        self.register(
            "audit_logger",
            "security",
            "ACTIVE",
            "Security and activity audit logging.",
            "core.security.audit.audit_logger",
            "AuditLogger",
        )

        self.register(
            "permission_manager",
            "security",
            "ACTIVE",
            "Permission management.",
            "core.security.permissions.permission_manager",
            "PermissionManager",
        )

        # EVOLUTION
        self.register(
            "evolution",
            "evolution",
            "PARTIAL",
            "Self-analysis and improvement framework.",
            "core.evolution",
        )

        self.register(
            "benchmark",
            "evolution",
            "ACTIVE",
            "AI benchmark subsystem.",
            "core.evolution.benchmark.benchmark_engine",
            "BenchmarkEngine",
        )

        self.register(
            "diagnostics",
            "evolution",
            "ACTIVE",
            "System diagnostics.",
            "core.evolution.diagnostics.diagnostics",
            "Diagnostics",
        )

        self.register(
            "improvement",
            "evolution",
            "ACTIVE",
            "Improvement engine.",
            "core.evolution.improvement.improvement_engine",
            "ImprovementEngine",
        )

        self.register(
            "self_analysis",
            "evolution",
            "ACTIVE",
            "Self-analysis engine.",
            "core.evolution.self_analysis.self_analysis",
            "SelfAnalysis",
        )

        self.register(
            "weakness_detection",
            "evolution",
            "ACTIVE",
            "Weakness detection engine.",
            "core.evolution.weakness_detection.weakness_detector",
            "WeaknessDetector",
        )

        self.register(
            "rollback",
            "evolution",
            "ACTIVE",
            "Safe rollback system.",
            "core.evolution.rollback.rollback_manager",
            "RollbackManager",
        )

    # --------------------------------------------------
    # HEALTH CHECK
    # --------------------------------------------------

    def _check_module(self, module_name: Optional[str]) -> bool:

        if not module_name:
            return False

        try:
            importlib.import_module(module_name)
            return True
        except Exception:
            return False

    def refresh_health(self) -> None:

        for capability in self._capabilities.values():

            if capability.module:
                capability.health = self._check_module(
                    capability.module
                )

    # --------------------------------------------------
    # QUERY
    # --------------------------------------------------

    def get(self, name: str) -> Optional[Capability]:
        return self._capabilities.get(name)

    def all(self) -> List[dict]:
        return [
            capability.to_dict()
            for capability in self._capabilities.values()
        ]

    def active(self) -> List[dict]:
        return [
            capability.to_dict()
            for capability in self._capabilities.values()
            if capability.status == "ACTIVE"
        ]

    def partial(self) -> List[dict]:
        return [
            capability.to_dict()
            for capability in self._capabilities.values()
            if capability.status == "PARTIAL"
        ]

    def inactive(self) -> List[dict]:
        return [
            capability.to_dict()
            for capability in self._capabilities.values()
            if capability.status == "NOT_ACTIVE"
        ]

    # --------------------------------------------------
    # CATEGORY
    # --------------------------------------------------

    def by_category(self, category: str) -> List[dict]:

        return [
            capability.to_dict()
            for capability in self._capabilities.values()
            if capability.category == category
        ]

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    def summary(self) -> dict:

        capabilities = list(self._capabilities.values())

        active = sum(
            1 for x in capabilities
            if x.status == "ACTIVE"
        )

        partial = sum(
            1 for x in capabilities
            if x.status == "PARTIAL"
        )

        inactive = sum(
            1 for x in capabilities
            if x.status == "NOT_ACTIVE"
        )

        placeholder = sum(
            1 for x in capabilities
            if x.status == "PLACEHOLDER"
        )

        errors = sum(
            1 for x in capabilities
            if x.status == "ERROR"
        )

        healthy = sum(
            1 for x in capabilities
            if x.health
        )

        total = len(capabilities)

        percentage = (
            round((active / total) * 100, 2)
            if total
            else 0
        )

        return {
            "registry_version": self.VERSION,
            "total": total,
            "active": active,
            "partial": partial,
            "not_active": inactive,
            "placeholder": placeholder,
            "error": errors,
            "healthy_modules": healthy,
            "activation_percent": percentage,
            "created_at": self._created_at,
        }

    # --------------------------------------------------
    # EXPORT
    # --------------------------------------------------

    def export(self, path: str) -> str:

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        payload = {
            "registry": self.VERSION,
            "summary": self.summary(),
            "capabilities": self.all(),
        }

        target.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        return str(target)

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    def report(self) -> str:

        summary = self.summary()

        lines = []

        lines.append("")
        lines.append("=" * 70)
        lines.append("                 ZAI CAPABILITY REPORT")
        lines.append("=" * 70)

        lines.append(
            f"Registry Version : {summary['registry_version']}"
        )

        lines.append(
            f"Total            : {summary['total']}"
        )

        lines.append(
            f"ACTIVE           : {summary['active']}"
        )

        lines.append(
            f"PARTIAL          : {summary['partial']}"
        )

        lines.append(
            f"NOT ACTIVE       : {summary['not_active']}"
        )

        lines.append(
            f"PLACEHOLDER      : {summary['placeholder']}"
        )

        lines.append(
            f"ERROR            : {summary['error']}"
        )

        lines.append(
            f"Healthy Modules  : {summary['healthy_modules']}"
        )

        lines.append(
            f"Activation       : {summary['activation_percent']}%"
        )

        lines.append("-" * 70)

        for capability in self._capabilities.values():

            status = capability.status

            if status == "ACTIVE":
                marker = "[ACTIVE]"
            elif status == "PARTIAL":
                marker = "[PARTIAL]"
            elif status == "NOT_ACTIVE":
                marker = "[OFF]"
            elif status == "ERROR":
                marker = "[ERROR]"
            else:
                marker = "[PLACEHOLDER]"

            health = "HEALTHY" if capability.health else "UNHEALTHY"

            lines.append(
                f"{marker:<13} "
                f"{capability.category:<15} "
                f"{capability.name:<25} "
                f"{health}"
            )

        lines.append("=" * 70)

        return "\n".join(lines)
