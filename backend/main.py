"""
DSP Voice Camouflage Detection — FastAPI Application
=====================================================
REST endpoints for file upload analysis +
WebSocket endpoint for real-time microphone streaming.
"""

import io
import uuid
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dsp_pipeline import DSPPipeline, DEFAULT_SR
from stream_handler import stream_manager

# ─── APP ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DSP Voice Camouflage Detection API",
    version="1.0.0",
    description="Detects natural, modified, and AI-generated voices via DSP analysis",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = DSPPipeline(sr=DEFAULT_SR)


# ─── REST: Health ───────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "engine": "DSP Voice Camouflage Detector v1.0"}


# ─── REST: File Upload Analysis ────────────────────────────────────────────
ALLOWED_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/flac", "audio/x-flac",
    "audio/mp4", "audio/m4a", "audio/x-m4a",
    "application/octet-stream",  # fallback for some browsers
}

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".mp4"}


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    """Analyze an uploaded audio file through the full DSP pipeline."""
    import os

    # Validate file extension
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save to a temporary file on disk first
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")

    try:
        # Write binary content to disk
        contents = await file.read()
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(contents)

        # Load audio using the file path (allows soundfile/audioread to detect format)
        y, sr = librosa.load(temp_file_path, sr=DEFAULT_SR, mono=True)

        if len(y) < DEFAULT_SR * 0.3:  # Minimum 300ms
            raise HTTPException(status_code=400, detail="Audio too short. Minimum 300ms required.")

        if len(y) > DEFAULT_SR * 120:  # Max 2 minutes
            y = y[: DEFAULT_SR * 120]

        # Run full pipeline
        result = pipeline.process_signal(y)

        return JSONResponse(content={
            "success": True,
            "filename": filename,
            "duration_s": round(len(y) / DEFAULT_SR, 2),
            "sample_rate": DEFAULT_SR,
            **result,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


# ─── REST: Model Retraining & Feedback ─────────────────────────────────────
@app.post("/api/train")
async def train_with_sample(ground_truth: str = Form(...), file: UploadFile = File(...)):
    """Reject online retraining until a reviewed real-audio training workflow exists."""
    raise HTTPException(
        status_code=409,
        detail=("Online retraining is disabled. Feedback must be reviewed and evaluated "
                "against a held-out real-audio dataset before publishing a model."),
    )

    # Kept below temporarily as a reference for the future reviewed-training job.
    import os
    from dsp_pipeline import ml_classifier

    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio file format.")

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"train_{uuid.uuid4()}{ext}")

    try:
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)

        y, sr = librosa.load(temp_file_path, sr=DEFAULT_SR, mono=True)
        result = pipeline.process_signal(y)

        # Retrain model with feedback sample
        retrain_result = ml_classifier.add_sample_and_retrain(result["features"], ground_truth)
        stats = ml_classifier.get_model_stats()

        return JSONResponse(content={
            "success": True,
            "message": f"Sample added with ground truth label '{ground_truth}'. Model retrained!",
            "retrain_status": retrain_result,
            "model_stats": stats
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


@app.get("/api/model-stats")
async def get_model_stats():
    """Retrieve current ML model accuracy, estimator count, and feature importances."""
    from dsp_pipeline import ml_classifier
    return JSONResponse(content=ml_classifier.get_model_stats())

@app.websocket("/ws/audio/{client_id}")
async def websocket_audio(websocket: WebSocket, client_id: str):
    """Real-time audio analysis via WebSocket.

    Client sends raw Float32 PCM audio chunks as binary messages.
    Server responds with JSON analysis results per processed chunk.
    """
    await stream_manager.handle_connection(websocket, client_id)


# ─── WebSocket: Fallback (auto-assign ID) ─────────────────────────────────
@app.websocket("/ws/audio")
async def websocket_audio_auto(websocket: WebSocket):
    """Auto-assign client ID if none provided."""
    client_id = str(uuid.uuid4())[:8]
    await stream_manager.handle_connection(websocket, client_id)


# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
