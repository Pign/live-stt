"""Real-time speech-to-text server using sherpa-onnx streaming zipformer.

The browser streams 16 kHz mono PCM (int16) frames over a WebSocket. The
server feeds them to a sherpa-onnx OnlineRecognizer which performs streaming
ASR with built-in endpoint detection. Partial hypotheses are emitted as the
user speaks; on endpoint detection a final transcript is emitted and the
stream is reset for the next utterance.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import numpy as np
import sherpa_onnx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("live-stt")

SAMPLE_RATE = 16_000
MODEL_DIR = Path(os.getenv(
    "STT_MODEL_DIR",
    "models/sherpa-onnx-streaming-zipformer-fr-2023-04-14",
))


def _resolve(pattern: str) -> str:
    matches = sorted(MODEL_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Missing {pattern} under {MODEL_DIR}. Run ./download_model.sh first."
        )
    # Prefer int8 weights if available (faster on CPU).
    int8 = [m for m in matches if "int8" in m.name]
    return str((int8 or matches)[0])


log.info("Loading sherpa-onnx streaming zipformer from %s", MODEL_DIR)
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=str(MODEL_DIR / "tokens.txt"),
    encoder=_resolve("encoder-*.onnx"),
    decoder=_resolve("decoder-*.onnx"),
    joiner=_resolve("joiner-*.onnx"),
    num_threads=int(os.getenv("STT_THREADS", "2")),
    sample_rate=SAMPLE_RATE,
    feature_dim=80,
    enable_endpoint_detection=True,
    rule1_min_trailing_silence=2.4,
    rule2_min_trailing_silence=1.2,
    rule3_min_utterance_length=20.0,
    decoding_method="greedy_search",
    provider=os.getenv("STT_PROVIDER", "cpu"),
)
log.info("Model loaded.")

app = FastAPI()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    stream = recognizer.create_stream()
    last_partial = ""
    seq = 0
    log.info("Client connected")

    try:
        while True:
            message = await ws.receive()

            if "text" in message and message["text"] is not None:
                try:
                    cmd = json.loads(message["text"])
                except Exception:
                    continue
                if cmd.get("type") == "flush":
                    # Pad with silence to flush the encoder, then drain.
                    stream.accept_waveform(SAMPLE_RATE, np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
                    stream.input_finished()
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)
                    text = recognizer.get_result(stream).text.strip()
                    if text:
                        await ws.send_json({"type": "final", "seq": seq, "text": text})
                        seq += 1
                    recognizer.reset(stream)
                    last_partial = ""
                continue

            data = message.get("bytes")
            if not data:
                continue

            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            stream.accept_waveform(SAMPLE_RATE, samples)

            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            text = recognizer.get_result(stream).text.strip()

            if recognizer.is_endpoint(stream):
                if text:
                    await ws.send_json({"type": "final", "seq": seq, "text": text})
                    seq += 1
                recognizer.reset(stream)
                last_partial = ""
            elif text and text != last_partial:
                await ws.send_json({"type": "partial", "seq": seq, "text": text})
                last_partial = text

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception:
        log.exception("WebSocket error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
