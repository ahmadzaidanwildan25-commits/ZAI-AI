from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import time
import asyncio


# ============================================================
# ZAI SERVER
# SUPER ZAI - HIGH SPEED AI CORE
# ============================================================

APP_NAME = "ZAI"
APP_VERSION = "0.3.0"

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_URL}/api/chat"
OLLAMA_GENERATE_URL = f"{OLLAMA_URL}/api/generate"

MODEL = "qwen3:8b"

# Keep model loaded in memory.
# -1 = keep loaded indefinitely.
KEEP_ALIVE = -1

# ============================================================
# SPEED CONFIGURATION
# ============================================================

# Jangan biarkan history terlalu panjang.
# History panjang = prompt semakin besar = response semakin lambat.
MAX_HISTORY = 8

# Batas karakter setiap message history.
MAX_HISTORY_CHARS = 3000

# Batas karakter pesan user.
MAX_MESSAGE_CHARS = 12000

# HTTP timeout.
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = None

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="ZAI AI Core",
    version=APP_VERSION,
    description="High-speed local AI backend for Super ZAI.",
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
You are ZAI, a fast and capable personal AI assistant.

Core behavior:
- Answer directly.
- Be accurate and useful.
- Understand the user's intent.
- Do not reveal hidden reasoning.
- Do not unnecessarily over-explain.
- Keep simple answers short.
- For complex tasks, provide the necessary detail.
- Respond naturally in the user's language.
- Prioritize speed without sacrificing important information.

You are the intelligence core of Super ZAI.
Future capabilities may include:
memory, web, files, computer control, automation,
voice, devices, agents, projects, and knowledge.

For now, use the available local model.
""".strip()


# ============================================================
# HTTP CLIENT
# ============================================================

client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def startup_event():
    global client

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
    print("Stream  : ENABLED")
    print("Think   : DISABLED")
    print("Keep    : INFINITE")
    print("Status  : SERVER READY")
    print("=" * 64)

    # Warmup dijalankan di background.
    # Server tidak perlu menunggu warmup selesai.
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
        raise RuntimeError("HTTP client is not initialized.")

    return client


# ============================================================
# MODE
# ============================================================

def normalize_mode(mode: Optional[str]) -> str:
    if not mode:
        return "auto"

    mode = mode.lower().strip()

    if mode not in {"auto", "fast", "normal", "deep"}:
        return "auto"

    return mode


def detect_mode(
    message: str,
    requested_mode: str,
) -> str:

    if requested_mode != "auto":
        return requested_mode

    text = message.lower().strip()

    # --------------------------------------------------------
    # VERY SHORT / SIMPLE REQUEST
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
    # SPEED KEYWORDS
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

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    return "normal"


# ============================================================
# GENERATION OPTIONS
# ============================================================

def select_options(mode: str) -> dict:
    # FAST:
    # Output pendek supaya token generation cepat.
    if mode == "fast":
        return {
            "temperature": 0.15,
            "top_p": 0.75,
            "num_predict": 96,
        }

    # NORMAL:
    # Default ZAI.
    if mode == "normal":
        return {
            "temperature": 0.25,
            "top_p": 0.85,
            "num_predict": 512,
        }

    # DEEP:
    # Dipakai hanya untuk pekerjaan berat.
    if mode == "deep":
        return {
            "temperature": 0.35,
            "top_p": 0.9,
            "num_predict": 1536,
        }

    # Default fallback.
    return {
        "temperature": 0.25,
        "top_p": 0.85,
        "num_predict": 512,
    }

# ============================================================
# HISTORY
# ============================================================

def build_messages(
    user_message: str,
    history: Optional[list[Message]],
) -> list[dict]:

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    if history:

        # Ambil hanya history terbaru.
        recent_history = history[-MAX_HISTORY:]

        for item in recent_history:

            role = item.role.lower().strip()

            if role not in {"user", "assistant"}:
                continue

            content = item.content.strip()

            if not content:
                continue

            # Batasi ukuran history.
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
            (time.perf_counter() - started) * 1000,
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

    # Beri waktu server selesai startup.
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
            (time.perf_counter() - started) * 1000,
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
            for item in data.get("models", [])
        ]

        model_available = MODEL in models

        return {
            "status": "ONLINE" if model_available else "DEGRADED",
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
# OLLAMA STREAM
# ============================================================

async def ollama_stream(
    user_message: str,
    history: Optional[list[Message]],
    requested_mode: str,
):

    # --------------------------------------------------------
    # MODE
    # --------------------------------------------------------

    normalized_mode = normalize_mode(
        requested_mode
    )

    mode = detect_mode(
        user_message,
        normalized_mode,
    )

    options = select_options(mode)

    # --------------------------------------------------------
    # MESSAGES
    # --------------------------------------------------------

    messages = build_messages(
        user_message,
        history,
    )

    # --------------------------------------------------------
    # OLLAMA PAYLOAD
    # --------------------------------------------------------

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,

        # Keep model in memory.
        "keep_alive": KEEP_ALIVE,

        # Qwen3 thinking disabled.
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

            # ------------------------------------------------
            # HTTP ERROR
            # ------------------------------------------------

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

            # ------------------------------------------------
            # STREAM
            # ------------------------------------------------

            async for line in response.aiter_lines():

                if not line:
                    continue

                try:

                    data = json.loads(line)

                except json.JSONDecodeError:

                    continue

                # ------------------------------------------------
                # MESSAGE
                # ------------------------------------------------

                message_data = data.get(
                    "message",
                    {},
                )

                content = message_data.get(
                    "content",
                    "",
                )

                # ------------------------------------------------
                # FIRST TOKEN
                # ------------------------------------------------

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
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    # ----------------------------------------
                    # TOKEN
                    # ----------------------------------------

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

                # ------------------------------------------------
                # DONE
                # ------------------------------------------------

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
                                "done": True,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    break

    # ========================================================
    # ERRORS
    # ========================================================

    except httpx.ConnectError:

        yield (
            json.dumps(
                {
                    "type": "error",
                    "message": (
                        "Tidak dapat terhubung ke Ollama "
                        "di 127.0.0.1:11434."
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
                        "Ollama membutuhkan waktu terlalu lama "
                        "untuk memberikan respons."
                    ),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    except asyncio.CancelledError:

        # User/client membatalkan stream.
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
            requested_mode=request.mode or "auto",
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
        "model": MODEL,
        "streaming": True,
        "thinking": False,
        "keep_alive": True,
        "max_history": MAX_HISTORY,
        "modes": {
            "fast": {
                "num_predict": 256,
            },
            "normal": {
                "num_predict": 768,
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
        "max_history": MAX_HISTORY,
        "max_history_chars": MAX_HISTORY_CHARS,
        "max_message_chars": MAX_MESSAGE_CHARS,
    }