"""
WebSocket Stream Handler
========================
Manages real-time audio streaming from browser clients.
Buffers incoming PCM chunks, runs them through the DSP pipeline,
and emits results at high frequency for live UI updates.
"""

import asyncio
import numpy as np
import json
from fastapi import WebSocket, WebSocketDisconnect
from dsp_pipeline import DSPPipeline, DEFAULT_SR

# ─── CONFIG ─────────────────────────────────────────────────────────────────
CHUNK_DURATION_S = 0.5      # Process every 0.5 seconds of audio
OVERLAP_RATIO = 0.25        # 25% overlap between consecutive chunks
MAX_BUFFER_S = 10.0         # Maximum buffer size before forced flush


class AudioStreamBuffer:
    """Accumulates incoming PCM bytes into processable NumPy chunks."""

    def __init__(self, sr: int = DEFAULT_SR, chunk_duration: float = CHUNK_DURATION_S):
        self.sr = sr
        self.chunk_size = int(sr * chunk_duration)
        self.overlap_size = int(self.chunk_size * OVERLAP_RATIO)
        self.buffer = np.array([], dtype=np.float32)
        self.max_buffer_samples = int(sr * MAX_BUFFER_S)

    def append(self, pcm_bytes: bytes) -> list[np.ndarray]:
        """Append raw PCM float32 bytes and return ready chunks."""
        new_samples = np.frombuffer(pcm_bytes, dtype=np.float32)
        self.buffer = np.concatenate([self.buffer, new_samples])

        # Safety: prevent unbounded buffer growth
        if len(self.buffer) > self.max_buffer_samples:
            self.buffer = self.buffer[-self.max_buffer_samples:]

        chunks = []
        while len(self.buffer) >= self.chunk_size:
            chunks.append(self.buffer[: self.chunk_size].copy())
            # Advance by (chunk_size - overlap) to maintain overlap
            advance = self.chunk_size - self.overlap_size
            self.buffer = self.buffer[advance:]

        return chunks

    def flush(self) -> np.ndarray | None:
        """Return any remaining buffered audio."""
        if len(self.buffer) > self.sr * 0.1:  # At least 100ms
            chunk = self.buffer.copy()
            self.buffer = np.array([], dtype=np.float32)
            return chunk
        return None

    def reset(self):
        self.buffer = np.array([], dtype=np.float32)


class StreamManager:
    """Handles WebSocket lifecycle for real-time audio analysis."""

    def __init__(self):
        self.pipeline = DSPPipeline(sr=DEFAULT_SR)
        self.active_connections: dict[str, WebSocket] = {}

    async def handle_connection(self, websocket: WebSocket, client_id: str):
        """Main WebSocket handler for a single client."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        buffer = AudioStreamBuffer(sr=DEFAULT_SR)

        try:
            # Send initial handshake
            await websocket.send_json({
                "type": "connected",
                "config": {
                    "sample_rate": DEFAULT_SR,
                    "chunk_duration": CHUNK_DURATION_S,
                    "format": "float32_pcm",
                },
            })

            while True:
                # Receive binary PCM data
                data = await websocket.receive()

                if "bytes" in data:
                    pcm_data = data["bytes"]
                    chunks = buffer.append(pcm_data)

                    for chunk in chunks:
                        # Run DSP in executor to avoid blocking the event loop
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, self.pipeline.process_chunk, chunk
                        )
                        await websocket.send_json({
                            "type": "analysis",
                            "data": result,
                        })

                elif "text" in data:
                    msg = json.loads(data["text"])

                    if msg.get("type") == "stop":
                        # Process remaining buffer
                        remaining = buffer.flush()
                        if remaining is not None:
                            result = await asyncio.get_event_loop().run_in_executor(
                                None, self.pipeline.process_chunk, remaining
                            )
                            await websocket.send_json({
                                "type": "analysis_final",
                                "data": result,
                            })
                        buffer.reset()
                        await websocket.send_json({"type": "stopped"})

                    elif msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            print(f"WebSocket client disconnected: {client_id}")
        except Exception as e:
            import traceback
            print(f"Exception in WebSocket handler for client {client_id}:")
            traceback.print_exc()
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
            except Exception:
                pass
        finally:
            self.active_connections.pop(client_id, None)
            buffer.reset()


# Singleton instance
stream_manager = StreamManager()
