# server/main.py

import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import custom audio pipeline modules
from server.pipeline.ring_buffer import RingBuffer
from server.pipeline.vad_filter import VADFilter

app = FastAPI(title="Real-Time WebRTC/WebSocket Audio Pipeline")

# Initialize Voice Activity Detection filter
vad = VADFilter(threshold=0.5)

# Serve static files from client directory for UI frontend
app.mount("/client", StaticFiles(directory="client"), name="client")


@app.get("/")
async def get_index():
    """Serves the primary web interface UI."""
    return FileResponse("client/index.html")


@app.websocket("/ws/audio")
async def audio_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint handling live PCM16 audio streaming, ring buffering,
    and Silero VAD evaluation.
    """
    await websocket.accept()
    print("[WEBSOCKET] Client connected for audio streaming.")

    # Initialize a 3.0-second (48,000 samples at 16kHz) ring buffer for this session
    # with a 200ms stride (3,200 samples) between evaluation checks.
    ring_buffer = RingBuffer(capacity_samples=48000, stride_samples=3200)

    try:
        while True:
            # Receive binary audio data from the browser's AudioWorklet
            data = await websocket.receive_bytes()

            # 1. Convert incoming raw binary Int16 PCM to float32 normalized [-1.0, 1.0]
            pcm16_data = np.frombuffer(data, dtype=np.int16)
            float32_data = pcm16_data.astype(np.float32) / 32768.0

            # 2. Append normalized audio chunk into ring buffer
            # 3. Check if stride threshold (200ms) is reached to trigger VAD evaluation
            if ring_buffer.append(float32_data):
                # Retrieve current accumulated 3-second audio window
                audio_window = ring_buffer.get_window()

                # Evaluate VAD across 512-sample sub-chunks inside the window
                speech_detected = vad.is_speech(audio_window)

                if speech_detected:
                    # Echo success status payload back to WebSocket client for browser display
                    response = {
                        "status": "SPEECH_DETECTED",
                        "buffer_samples": len(audio_window),
                        "message": "Speech detected in active ring buffer."
                    }
                    await websocket.send_text(json.dumps(response))
                else:
                    # Echo drop/silence status payload back to WebSocket client
                    response = {
                        "status": "SILENCE_OR_NOISE_DROPPED",
                        "buffer_samples": len(audio_window),
                        "message": "Audio classified as background noise or silence."
                    }
                    await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("[WEBSOCKET] Client audio streaming session disconnected.")
    except Exception as e:
        print(f"[WEBSOCKET ERROR]: {e}")
        await websocket.close()