from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import time
import asyncio
from memory.database import initialize_database
from memory.manager import MemoryManager

# ============================================================
# SUPER ZAI
# INTELLIGENCE CORE + ROUTER
# ============================================================

APP_NAME = "ZAI"
APP_VERSION = "0.4.0"

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_URL}/api/chat"
OLLAMA_GENERATE_URL = f"{OLLAMA_URL}/api/generate"

MODEL = "qwen3:8b"

# Keep model loaded in RAM.
KEEP_ALIVE = -1


# ============================================================
# SPEED CONFIGURATION
# ============================================================

MAX_HISTORY = 8
MAX_HISTORY_CHARS = 3000
MAX_MESSAGE_CHARS = 12000

CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = None


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Super ZAI AI Core",
    version=APP_VERSION,
    description="Super ZAI local intelligence core.",
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
You are ZAI, a highly capable personal AI assistant.

Your name is ZAI.

Core behavior:

- Answer the user's actual question.
- Be accurate and useful.
- Respond naturally.
- Respond in the user's language.
- Keep simple answers concise.
- Give more detail when the task requires it.
- Never expose hidden chain-of-thought or private reasoning.
- Do not claim that a tool was used when it was not used.
- Do not invent current information.
- Prioritize speed for simple requests.
- For complex requests, provide a complete and useful answer.

You are the intelligence core of Super ZAI.

Your architecture can eventually contain:
- memory
- web
- files
- computer control
- automation
- voice
- devices
- agents
- projects
- knowledge

At this stage, the local Qwen model is the primary reasoning engine.
""".strip()


# ============================================================
# HTTP CLIENT
# ============================================================

client: Optional[httpx.AsyncClient] = None
memory_manager = MemoryManager()

@app.on_event("startup")
async def startup_event():
    global client

    initialize_database()

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
    print("Status  : SERVER READY")
    print("=" * 64)

    asyncio.create_task(background_warmup())

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
# MODE NORMALIZATION
# ============================================================

def normalize_mode(mode: Optional[str]) -> str:

    if not mode:
        return "auto"

    mode = mode.lower().strip()

    allowed_modes = {
        "auto",
        "fast",
        "normal",
        "deep",
    }

    if mode not in allowed_modes:
        return "auto"

    return mode


# ============================================================
# INTELLIGENCE ROUTER
# ============================================================

def route_intent(message: str) -> str:
    """
    Determine which ZAI capability should handle the request.

    Current capabilities:
        chat
        web
        memory
        agent
        coding
    """

    text = message.lower().strip()

    # --------------------------------------------------------
    # WEB
    # --------------------------------------------------------

    web_patterns = [
        "sekarang",
        "saat ini",
        "hari ini",
        "terbaru",
        "terkini",
        "latest",
        "berita",
        "news",
        "harga hari ini",
        "cuaca",
        "presiden",
        "siapa yang menjabat",
        "jadwal hari ini",
        "informasi terbaru",
    ]

    for pattern in web_patterns:
        if pattern in text:
            return "web"

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    memory_patterns = [
        "ingat ini",
        "ingat bahwa",
        "ingat kalau",
        "simpan ini",
        "simpan bahwa",
        "catat ini",
        "jangan lupa",
        "apa yang kamu ingat",
        "kamu ingat",
        "ingat nama saya",
        "ingat saya",
    ]

    for pattern in memory_patterns:
        if pattern in text:
            return "memory"

    # --------------------------------------------------------
    # AGENT
    # --------------------------------------------------------

    agent_patterns = [
        "jalankan",
        "eksekusi",
        "buka aplikasi",
        "buka program",
        "buat file lalu",
        "buat project lalu",
        "perbaiki lalu jalankan",
        "kerjakan semuanya",
        "lakukan semuanya",
        "otomatis",
        "automatically",
        "selesaikan sampai selesai",
    ]

    for pattern in agent_patterns:
        if pattern in text:
            return "agent"

    # --------------------------------------------------------
    # CODING
    # --------------------------------------------------------

    coding_patterns = [
        "coding",
        "kode",
        "program",
        "flutter",
        "dart",
        "python",
        "fastapi",
        "javascript",
        "typescript",
        "html",
        "css",
        "api",
        "debug",
        "debugging",
        "error coding",
        "buat aplikasi",
        "buat sistem",
        "buat program",
        "full code",
        "kode lengkap",
    ]

    for pattern in coding_patterns:
        if pattern in text:
            return "coding"

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return "chat"


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
    # FAST
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
    # DEEP
    # --------------------------------------------------------

    deep_keywords = [
        "analisis mendalam",
        "analisa mendalam",
        "bandingkan secara detail",
        "buat arsitektur",
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
        "debug project",
    ]

    for keyword in deep_keywords:
        if keyword in text:
            return "deep"

    return "normal"


# ============================================================
# GENERATION OPTIONS
# ============================================================

def select_options(mode: str) -> dict:

    # FAST
    if mode == "fast":
        return {
            "temperature": 0.15,
            "top_p": 0.75,
            "num_predict": 96,
        }

    # NORMAL
    if mode == "normal":
        return {
            "temperature": 0.25,
            "top_p": 0.85,
            "num_predict": 512,
        }

    # DEEP
    if mode == "deep":
        return {
            "temperature": 0.35,
            "top_p": 0.90,
            "num_predict": 1536,
        }

    # FALLBACK
    return {
        "temperature": 0.25,
        "top_p": 0.85,
        "num_predict": 512,
    }

@app.get("/memory")
async def memory_status():

    return {
        "enabled": True,
        "total_memories": memory_manager.count(),
    }
@app.get("/memory/list")
async def memory_list():

    return {
        "success": True,
        "memories": memory_manager.important(
            limit=50
        ),
    }
@app.get("/memory/search")
async def memory_search(
    q: str,
):

    return {
        "success": True,
        "query": q,
        "memories": memory_manager.search(
            q,
            limit=10,
        ),
    }

# ============================================================
# HISTORY
# ============================================================
def build_messages(user_message, memory_context=None, history=None):
    messages = []

    if memory_context:
        messages.append(
            {
                "role": "system",
                "content": memory_context,
            }
        )

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
                content = content[:MAX_HISTORY_CHARS]

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    message = user_message.strip()

    if len(message) > MAX_MESSAGE_CHARS:
        message = message[:MAX_MESSAGE_CHARS]

    messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    return messages

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
            ) * 1000,
            2,
        )

        return {
            "success": response.status_code == 200,
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
        "streaming": True,
        "thinking": False,
        "keep_alive": True,
        "router": True,
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
            ) * 1000,
            2,
        )

        if response.status_code != 200:

            return {
                "status": "DEGRADED",
                "ollama": False,
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
            "latency_ms": latency_ms,
        }

    except Exception as error:

        return {
            "status": "OFFLINE",
            "ollama": False,
            "error": str(error),
        }


# ============================================================
# WARMUP ENDPOINT
# ============================================================

@app.post("/warmup")
async def warmup():

    return await perform_warmup()


# ============================================================
# ROUTER TEST
# ============================================================

@app.get("/route")
async def route(message: str):

    clean_message = message.strip()

    if not clean_message:

        return {
            "success": False,
            "error": "Message cannot be empty.",
        }

    requested_mode = normalize_mode("auto")

    mode = detect_mode(
        clean_message,
        requested_mode,
    )

    intent = route_intent(
        clean_message
    )

    return {
        "success": True,
        "message": clean_message,
        "intent": intent,
        "mode": mode,
    }


# ============================================================
# OLLAMA STREAM
# ============================================================

async def ollama_stream(
    user_message: str,
    history: Optional[list[Message]],
    requested_mode: str,
):

    normalized_mode = normalize_mode(
        requested_mode
    )

    mode = detect_mode(
        user_message,
        normalized_mode,
    )

    intent = route_intent(
        user_message
    )

    options = select_options(mode)

    messages = build_messages(
        user_message,
        history,
    )

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
                            ) * 1000,
                            2,
                        )

                        yield (
                            json.dumps(
                                {
                                    "type": "start",
                                    "intent": intent,
                                    "mode": mode,
                                    "latency_ms": (
                                        first_token_latency_ms
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

                if data.get("done"):

                    total_latency_ms = round(
                        (
                            time.perf_counter()
                            - started
                        ) * 1000,
                        2,
                    )

                    yield (
                        json.dumps(
                            {
                                "type": "done",
                                "intent": intent,
                                "mode": mode,
                                "total_latency_ms": (
                                    total_latency_ms
                                ),
                                "token_count": token_count,
                                "done": True,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    break

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

    except httpx.ReadTimeout:

        yield (
            json.dumps(
                {
                    "type": "error",
                    "message": (
                        "Ollama membutuhkan waktu "
                        "terlalu lama untuk memberikan "
                        "respons."
                    ),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    except asyncio.CancelledError:

        raise

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
async def chat(request: ChatRequest):

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
        "router": True,
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
        "router": True,
        "capabilities": [
            "chat",
            "web",
            "memory",
            "coding",
            "agent",
        ],
        "max_history": MAX_HISTORY,
        "max_history_chars": MAX_HISTORY_CHARS,
        "max_message_chars": MAX_MESSAGE_CHARS,
    }