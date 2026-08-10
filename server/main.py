from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

import asyncio
import httpx
import json
import time

from memory.database import (
    initialize_database,
    count_memories,
)

from memory.manager import MemoryManager


# ============================================================
# ZAI SERVER
# SUPER ZAI - AI CORE
# VERSION 0.6.0
# ============================================================

APP_NAME = "ZAI"
APP_VERSION = "0.6.0"

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_URL}/api/chat"
OLLAMA_GENERATE_URL = f"{OLLAMA_URL}/api/generate"

MODEL = "qwen3:8b"

# Keep model loaded in RAM.
KEEP_ALIVE = -1


# ============================================================
# PERFORMANCE
# ============================================================

MAX_HISTORY = 8
MAX_HISTORY_CHARS = 3000
MAX_MESSAGE_CHARS = 12000

MEMORY_CONTEXT_LIMIT = 5

CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = None


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="ZAI AI Core",
    version=APP_VERSION,
    description="Super ZAI local AI backend.",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[list[Message]] = None
    mode: Optional[str] = "auto"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ZAI, the intelligence core of Super ZAI.

You are a personal AI assistant.

CORE BEHAVIOR:

- Answer naturally.
- Be accurate and useful.
- Understand the user's intent.
- Use relevant memory when available.
- Never invent personal information.
- Never pretend to know something that is not available.
- Do not reveal hidden reasoning.
- Respond in the user's language.
- Keep simple questions concise.
- Give useful detail for complex tasks.
- Do not unnecessarily repeat yourself.
- Maintain conversation context.

MEMORY:

You may receive a section called MEMORY CONTEXT.

MEMORY CONTEXT contains information previously saved by the user.

Use it naturally when it is relevant.

Do not mention the internal database, SQLite,
MemoryManager, implementation, or memory system
unless the user explicitly asks about those systems.

If memory contains the answer to a personal question,
use that memory.

If memory does not contain the answer,
do not invent one.

IMPORTANT:

The memory context is reference information.
The latest user message always has priority.

You are the intelligence core of Super ZAI.

Future capabilities may include:

- long-term memory
- web access
- files
- computer control
- automation
- voice
- devices
- agents
- projects
- knowledge systems
""".strip()


# ============================================================
# GLOBAL SERVICES
# ============================================================

client: Optional[httpx.AsyncClient] = None

memory_manager: Optional[MemoryManager] = None


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    global client
    global memory_manager

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    initialize_database()

    # --------------------------------------------------------
    # MEMORY MANAGER
    # --------------------------------------------------------

    memory_manager = MemoryManager()

    # --------------------------------------------------------
    # HTTP CLIENT
    # --------------------------------------------------------

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=READ_TIMEOUT,
            write=30.0,
            pool=30.0,
        ),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
        ),
    )

    # --------------------------------------------------------
    # SERVER INFORMATION
    # --------------------------------------------------------

    print("=" * 64)
    print("ZAI AI CORE")
    print("=" * 64)
    print(f"Version : {APP_VERSION}")
    print(f"Model   : {MODEL}")
    print(f"Ollama  : {OLLAMA_URL}")
    print("Memory  : ENABLED")
    print("Stream  : ENABLED")
    print("Think   : DISABLED")
    print("Keep    : INFINITE")
    print(f"Memories: {count_memories()}")
    print("Status  : SERVER READY")
    print("=" * 64)

    # --------------------------------------------------------
    # BACKGROUND WARMUP
    # --------------------------------------------------------

    asyncio.create_task(
        background_warmup()
    )


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
async def shutdown_event():
    global client

    if client is not None:
        await client.aclose()
        client = None


# ============================================================
# CLIENT HELPER
# ============================================================

def get_client() -> httpx.AsyncClient:
    if client is None:
        raise RuntimeError(
            "HTTP client is not initialized."
        )

    return client


# ============================================================
# MEMORY MANAGER HELPER
# ============================================================

def get_memory_manager() -> MemoryManager:
    if memory_manager is None:
        raise RuntimeError(
            "Memory manager is not initialized."
        )

    return memory_manager


# ============================================================
# MODE NORMALIZATION
# ============================================================

def normalize_mode(
    mode: Optional[str],
) -> str:

    if not mode:
        return "auto"

    mode = mode.lower().strip()

    if mode not in {
        "auto",
        "fast",
        "normal",
        "deep",
    }:
        return "auto"

    return mode


# ============================================================
# MODE DETECTION
# ============================================================

def detect_mode(
    message: str,
    requested_mode: str,
) -> str:

    if requested_mode != "auto":
        return requested_mode

    text = message.lower().strip()

    # --------------------------------------------------------
    # VERY SIMPLE
    # --------------------------------------------------------

    fast_exact = {
        "halo",
        "hai",
        "hi",
        "hello",
        "hey",
        "oke",
        "ok",
        "ya",
        "tidak",
        "makasih",
        "terima kasih",
        "terima kasih zai",
        "siapa kamu",
        "apa kabar",
        "selamat pagi",
        "selamat siang",
        "selamat sore",
        "selamat malam",
    }

    if text in fast_exact:
        return "fast"

    # --------------------------------------------------------
    # FAST KEYWORDS
    # --------------------------------------------------------

    fast_keywords = [
        "jawab singkat",
        "singkat saja",
        "secara singkat",
        "berapa hasil",
        "arti kata",
        "apa itu",
        "siapa",
        "kapan",
        "dimana",
        "berapa",
    ]

    for keyword in fast_keywords:

        if keyword in text and len(text) < 180:
            return "fast"

    # --------------------------------------------------------
    # DEEP TASKS
    # --------------------------------------------------------

    deep_keywords = [
        "analisis mendalam",
        "analisa mendalam",
        "bandingkan secara detail",
        "buat arsitektur",
        "debug",
        "debugging",
        "jelaskan secara mendalam",
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


# ============================================================
# GENERATION OPTIONS
# ============================================================

def select_options(mode: str) -> dict:

    if mode == "fast":

        return {
            "temperature": 0.15,
            "top_p": 0.75,
            "num_predict": 96,
        }

    if mode == "normal":

        return {
            "temperature": 0.25,
            "top_p": 0.85,
            "num_predict": 512,
        }

    if mode == "deep":

        return {
            "temperature": 0.35,
            "top_p": 0.90,
            "num_predict": 1536,
        }

    return {
        "temperature": 0.25,
        "top_p": 0.85,
        "num_predict": 512,
    }


# ============================================================
# HISTORY BUILDER
# ============================================================

def build_messages(
    user_message: str,
    history: Optional[list[Message]],
    memory_context: str = "",
) -> list[dict]:

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # --------------------------------------------------------
    # MEMORY CONTEXT
    # --------------------------------------------------------

    if memory_context.strip():

        messages.append(
            {
                "role": "system",
                "content": (
                    "MEMORY CONTEXT:\n\n"
                    + memory_context.strip()
                ),
            }
        )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    if history:

        recent_history = history[-MAX_HISTORY:]

        for item in recent_history:

            role = item.role.lower().strip()

            if role not in {
                "user",
                "assistant",
            }:
                continue

            content = item.content.strip()

            if not content:
                continue

            if len(content) > MAX_HISTORY_CHARS:
                content = content[
                    :MAX_HISTORY_CHARS
                ]

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    # --------------------------------------------------------
    # CURRENT USER MESSAGE
    # --------------------------------------------------------

    message = user_message.strip()

    if len(message) > MAX_MESSAGE_CHARS:
        message = message[
            :MAX_MESSAGE_CHARS
        ]

    messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    return messages


# ============================================================
# MEMORY CONTEXT
# ============================================================

def build_memory_context(
    query: str,
) -> str:

    manager = get_memory_manager()

    try:

        # MemoryManager yang sudah kita test:
        # build_context(query)

        context = manager.build_context(
            query
        )

        if not context:
            return ""

        # build_context() sudah menghasilkan
        # format MEMORY CONTEXT.
        if context.startswith(
            "MEMORY CONTEXT:"
        ):

            context = context[
                len("MEMORY CONTEXT:"):
            ].strip()

        return context.strip()

    except Exception as error:

        print(
            "[ZAI] Memory context error:",
            error,
        )

        return ""


# ============================================================
# MEMORY COMMAND
# ============================================================

def process_memory_command(
    text: str,
) -> Optional[dict]:

    manager = get_memory_manager()

    try:

        command = manager.detect_memory_command(
            text
        )

        return command

    except AttributeError:

        return None

    except Exception as error:

        print(
            "[ZAI] Memory command error:",
            error,
        )

        return None


# ============================================================
# MEMORY COMMAND EXECUTION
# ============================================================

def execute_memory_command(
    command: dict,
) -> Optional[str]:

    manager = get_memory_manager()

    action = command.get(
        "action"
    )

    # ========================================================
    # SAVE
    # ========================================================

    if action == "save":

        content = str(
            command.get(
                "content",
                "",
            )
        ).strip()

        if not content:
            return None

        try:

            result = manager.save(
                content,
                category="project",
                importance=10,
            )

            if isinstance(result, dict):

                if result.get(
                    "success",
                    True,
                ):

                    return (
                        "Baik. Saya akan "
                        "mengingatnya."
                    )

            return (
                "Baik. Saya akan "
                "mengingatnya."
            )

        except AttributeError:

            # Compatibility fallback
            # dengan MemoryManager versi lama.

            try:

                result = manager.save_memory(
                    content,
                    content,
                    category="project",
                    importance=10,
                )

            except TypeError:

                result = manager.save_memory(
                    content,
                    content,
                    category="project",
                )

            if isinstance(result, dict):

                if result.get(
                    "success",
                    True,
                ):

                    return (
                        "Baik. Saya akan "
                        "mengingatnya."
                    )

            return (
                "Baik. Saya akan "
                "mengingatnya."
            )

    # ========================================================
    # DELETE
    # ========================================================

    if action == "delete":

        key = str(
            command.get(
                "key",
                "",
            )
        ).strip()

        if not key:
            return None

        try:

            result = manager.delete(
                key
            )

        except AttributeError:

            result = manager.delete_memory(
                key
            )

        if result:

            return (
                "Baik. Memory tersebut "
                "sudah saya hapus."
            )

        return (
            "Saya tidak menemukan "
            "memory tersebut."
        )

    return None


# ============================================================
# WARMUP
# ============================================================

async def perform_warmup() -> dict:

    payload = {
        "model": MODEL,
        "prompt": "",
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_predict": 1,
        },
    }

    started = time.perf_counter()

    try:

        response = await get_client().post(
            OLLAMA_GENERATE_URL,
            json=payload,
        )

        latency_ms = round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            2,
        )

        return {
            "success": (
                response.status_code == 200
            ),
            "model": MODEL,
            "latency_ms": latency_ms,
        }

    except Exception as error:

        return {
            "success": False,
            "model": MODEL,
            "error": str(error),
        }


async def background_warmup():

    await asyncio.sleep(0.5)

    try:

        result = await perform_warmup()

        print(
            "[ZAI] Background warmup:",
            result,
        )

    except Exception as error:

        print(
            "[ZAI] Warmup error:",
            error,
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "ONLINE",
        "model": MODEL,
        "memory": True,
        "streaming": True,
        "thinking": False,
        "keep_alive": True,
    }


# ============================================================
# PING
# ============================================================

@app.get("/ping")
async def ping():

    return {
        "pong": True,
        "service": APP_NAME,
        "status": "ONLINE",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    started = time.perf_counter()

    try:

        response = await get_client().get(
            f"{OLLAMA_URL}/api/tags"
        )

        latency_ms = round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            2,
        )

        if response.status_code != 200:

            return {
                "status": "DEGRADED",
                "ollama": False,
                "memory": True,
                "latency_ms": latency_ms,
            }

        data = response.json()

        models = [
            item.get("name")
            for item in data.get(
                "models",
                [],
            )
        ]

        model_available = MODEL in models

        return {
            "status": (
                "ONLINE"
                if model_available
                else "DEGRADED"
            ),
            "ollama": True,
            "model": MODEL,
            "model_available": model_available,
            "memory": True,
            "memory_count": count_memories(),
            "latency_ms": latency_ms,
        }

    except Exception as error:

        return {
            "status": "OFFLINE",
            "ollama": False,
            "memory": True,
            "error": str(error),
        }


# ============================================================
# MEMORY STATUS
# ============================================================

@app.get("/memory")
async def memory_status():

    return {
        "enabled": True,
        "count": count_memories(),
    }


# ============================================================
# WARMUP ENDPOINT
# ============================================================

@app.post("/warmup")
async def warmup():

    return await perform_warmup()


# ============================================================
# OLLAMA STREAM
# ============================================================

async def ollama_stream(
    user_message: str,
    history: Optional[list[Message]],
    requested_mode: str,
):

    # ========================================================
    # MODE
    # ========================================================

    normalized_mode = normalize_mode(
        requested_mode
    )

    mode = detect_mode(
        user_message,
        normalized_mode,
    )

    options = select_options(
        mode
    )

    # ========================================================
    # MEMORY COMMAND
    # ========================================================

    memory_command = process_memory_command(
        user_message
    )

    if memory_command:

        memory_response = execute_memory_command(
            memory_command
        )

        if memory_response:

            yield (
                json.dumps(
                    {
                        "type": "start",
                        "mode": "fast",
                        "latency_ms": 0,
                        "memory": True,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            yield (
                json.dumps(
                    {
                        "type": "token",
                        "content": memory_response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            yield (
                json.dumps(
                    {
                        "type": "done",
                        "mode": "fast",
                        "total_latency_ms": 0,
                        "token_count": 1,
                        "memory": True,
                        "memory_used": True,
                        "done": True,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            return

    # ========================================================
    # MEMORY CONTEXT
    # ========================================================

    memory_context = build_memory_context(
        user_message
    )

    # ========================================================
    # MESSAGES
    # ========================================================

    messages = build_messages(
        user_message,
        history,
        memory_context,
    )

    # ========================================================
    # OLLAMA PAYLOAD
    # ========================================================

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "think": False,
        "options": options,
    }

    started = time.perf_counter()

    first_token_time = None
    token_count = 0

    try:

        async with get_client().stream(
            "POST",
            OLLAMA_CHAT_URL,
            json=payload,
        ) as response:

            # =================================================
            # HTTP ERROR
            # =================================================

            if response.status_code != 200:

                body = await response.aread()

                error_text = body.decode(
                    "utf-8",
                    errors="replace",
                )

                yield (
                    json.dumps(
                        {
                            "type": "error",
                            "message": error_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                return

            # =================================================
            # STREAM
            # =================================================

            async for line in response.aiter_lines():

                if not line:
                    continue

                try:

                    data = json.loads(line)

                except json.JSONDecodeError:

                    continue

                message_data = data.get(
                    "message",
                    {},
                )

                content = message_data.get(
                    "content",
                    "",
                )

                # =============================================
                # TOKEN
                # =============================================

                if content:

                    token_count += 1

                    if first_token_time is None:

                        first_token_time = (
                            time.perf_counter()
                        )

                        first_token_latency_ms = round(
                            (
                                first_token_time
                                - started
                            )
                            * 1000,
                            2,
                        )

                        yield (
                            json.dumps(
                                {
                                    "type": "start",
                                    "mode": mode,
                                    "latency_ms": (
                                        first_token_latency_ms
                                    ),
                                    "memory_used": bool(
                                        memory_context
                                    ),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    yield (
                        json.dumps(
                            {
                                "type": "token",
                                "content": content,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                # =============================================
                # DONE
                # =============================================

                if data.get("done"):

                    total_latency_ms = round(
                        (
                            time.perf_counter()
                            - started
                        )
                        * 1000,
                        2,
                    )

                    yield (
                        json.dumps(
                            {
                                "type": "done",
                                "mode": mode,
                                "total_latency_ms": (
                                    total_latency_ms
                                ),
                                "token_count": token_count,
                                "memory_used": bool(
                                    memory_context
                                ),
                                "done": True,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    break

    # ========================================================
    # CONNECTION ERROR
    # ========================================================

    except httpx.ConnectError:

        yield (
            json.dumps(
                {
                    "type": "error",
                    "message": (
                        "Tidak dapat terhubung "
                        "ke Ollama di "
                        "127.0.0.1:11434."
                    ),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    # ========================================================
    # TIMEOUT
    # ========================================================

    except httpx.ReadTimeout:

        yield (
            json.dumps(
                {
                    "type": "error",
                    "message": (
                        "Ollama membutuhkan waktu "
                        "terlalu lama untuk "
                        "memberikan respons."
                    ),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    # ========================================================
    # CANCELLED
    # ========================================================

    except asyncio.CancelledError:

        raise

    # ========================================================
    # UNKNOWN ERROR
    # ========================================================

    except Exception as error:

        yield (
            json.dumps(
                {
                    "type": "error",
                    "message": str(error),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
async def chat(
    request: ChatRequest,
):

    message = request.message.strip()

    if not message:

        return {
            "success": False,
            "error": "Message cannot be empty.",
        }

    return StreamingResponse(
        ollama_stream(
            user_message=message,
            history=request.history,
            requested_mode=(
                request.mode or "auto"
            ),
        ),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# SPEED STATUS
# ============================================================

@app.get("/speed")
async def speed():

    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "model": MODEL,
        "streaming": True,
        "thinking": False,
        "keep_alive": True,
        "memory": True,
        "max_history": MAX_HISTORY,
        "modes": {
            "fast": {
                "num_predict": 96,
            },
            "normal": {
                "num_predict": 512,
            },
            "deep": {
                "num_predict": 1536,
            },
        },
    }


# ============================================================
# SERVER INFORMATION
# ============================================================

@app.get("/info")
async def info():

    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "model": MODEL,
        "ollama_url": OLLAMA_URL,
        "streaming": True,
        "thinking": False,
        "keep_alive": KEEP_ALIVE,
        "memory": True,
        "memory_count": count_memories(),
        "max_history": MAX_HISTORY,
        "max_history_chars": MAX_HISTORY_CHARS,
        "max_message_chars": MAX_MESSAGE_CHARS,
    }