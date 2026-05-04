"""Real-time speech-to-text server using faster-whisper + WebRTC VAD.

The browser streams 16 kHz mono PCM (int16) frames over a WebSocket. The
server runs voice activity detection on 30 ms frames, accumulates speech
into utterances, and transcribes each finalized utterance with
faster-whisper. Partial hypotheses are emitted while the user is still
speaking so the UI feels live.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("live-stt")

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples
FRAME_BYTES = FRAME_SAMPLES * 2  # int16

# Tunables via env vars so the same server runs on CPU laptop and GPU box.
MODEL_SIZE = os.getenv("STT_MODEL", "small")
DEVICE = os.getenv("STT_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("STT_COMPUTE", "int8" if DEVICE == "cpu" else "float16")
LANGUAGE = os.getenv("STT_LANGUAGE", "fr")
VAD_AGGRESSIVENESS = int(os.getenv("STT_VAD", "2"))  # 0..3
SILENCE_MS_END = int(os.getenv("STT_SILENCE_MS", "600"))
PARTIAL_EVERY_MS = int(os.getenv("STT_PARTIAL_MS", "700"))
MIN_UTTERANCE_MS = int(os.getenv("STT_MIN_UTTERANCE_MS", "300"))
MAX_UTTERANCE_MS = int(os.getenv("STT_MAX_UTTERANCE_MS", "20000"))

log.info("Loading faster-whisper model=%s device=%s compute=%s lang=%s",
         MODEL_SIZE, DEVICE, COMPUTE_TYPE, LANGUAGE)
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
log.info("Model loaded.")

app = FastAPI()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


@dataclass
class Session:
    vad: webrtcvad.Vad = field(default_factory=lambda: webrtcvad.Vad(VAD_AGGRESSIVENESS))
    pcm_buffer: bytearray = field(default_factory=bytearray)  # leftover bytes < one frame
    utterance: bytearray = field(default_factory=bytearray)   # current utterance audio
    in_speech: bool = False
    silence_ms: int = 0
    speech_ms: int = 0
    last_partial_ms: int = 0
    seq: int = 0  # monotonic id assigned to each finalized utterance


def transcribe_pcm(pcm_bytes: bytes, language: str) -> str:
    """Run faster-whisper on raw int16 PCM bytes. Returns concatenated text."""
    if not pcm_bytes:
        return ""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _info = model.transcribe(
        audio,
        language=language,
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return "".join(seg.text for seg in segments).strip()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = Session()
    language = LANGUAGE
    loop = asyncio.get_running_loop()
    log.info("Client connected")

    async def send(payload: dict) -> None:
        try:
            await ws.send_json(payload)
        except Exception:
            pass

    try:
        while True:
            message = await ws.receive()
            if "text" in message and message["text"] is not None:
                # Control messages: {"type":"config","language":"fr"} or {"type":"flush"}
                try:
                    import json
                    cmd = json.loads(message["text"])
                except Exception:
                    continue
                if cmd.get("type") == "config" and cmd.get("language"):
                    language = cmd["language"]
                    log.info("Language set to %s", language)
                elif cmd.get("type") == "flush":
                    await _finalize(session, language, send, loop)
                continue

            data = message.get("bytes")
            if not data:
                continue

            session.pcm_buffer.extend(data)

            while len(session.pcm_buffer) >= FRAME_BYTES:
                frame = bytes(session.pcm_buffer[:FRAME_BYTES])
                del session.pcm_buffer[:FRAME_BYTES]
                is_speech = session.vad.is_speech(frame, SAMPLE_RATE)

                if is_speech:
                    session.utterance.extend(frame)
                    session.speech_ms += FRAME_MS
                    session.silence_ms = 0
                    session.in_speech = True
                elif session.in_speech:
                    # Trailing silence still belongs to the utterance for context.
                    session.utterance.extend(frame)
                    session.silence_ms += FRAME_MS

                # Emit partials while we're inside an utterance.
                if (
                    session.in_speech
                    and session.speech_ms >= MIN_UTTERANCE_MS
                    and session.speech_ms - session.last_partial_ms >= PARTIAL_EVERY_MS
                ):
                    session.last_partial_ms = session.speech_ms
                    pcm = bytes(session.utterance)
                    asyncio.create_task(_emit_partial(pcm, language, send, loop, session.seq))

                # End-of-utterance: enough trailing silence, or hard cap reached.
                if session.in_speech and (
                    session.silence_ms >= SILENCE_MS_END
                    or session.speech_ms >= MAX_UTTERANCE_MS
                ):
                    await _finalize(session, language, send, loop)
    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as exc:
        log.exception("WebSocket error: %s", exc)
    finally:
        await _finalize(session, language, send, loop)


async def _emit_partial(pcm: bytes, language: str, send, loop, seq: int) -> None:
    t0 = time.monotonic()
    text = await loop.run_in_executor(None, transcribe_pcm, pcm, language)
    if text:
        await send({"type": "partial", "seq": seq, "text": text,
                    "latency_ms": int((time.monotonic() - t0) * 1000)})


async def _finalize(session: Session, language: str, send, loop) -> None:
    if session.speech_ms < MIN_UTTERANCE_MS or not session.utterance:
        session.utterance.clear()
        session.in_speech = False
        session.silence_ms = 0
        session.speech_ms = 0
        session.last_partial_ms = 0
        return

    pcm = bytes(session.utterance)
    seq = session.seq
    session.seq += 1
    session.utterance.clear()
    session.in_speech = False
    session.silence_ms = 0
    session.speech_ms = 0
    session.last_partial_ms = 0

    t0 = time.monotonic()
    text = await loop.run_in_executor(None, transcribe_pcm, pcm, language)
    await send({"type": "final", "seq": seq, "text": text,
                "latency_ms": int((time.monotonic() - t0) * 1000)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
