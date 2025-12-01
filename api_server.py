"""
Chatterbox TTS HTTP API Server

A FastAPI-based HTTP API for the Chatterbox Text-to-Speech system.
Supports the [pause:Xs] custom pause tag feature (e.g., [pause:0.5s], [pause:1.0s]).

Usage:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

API Endpoints:
    POST /tts - Generate speech from text
    GET /health - Health check endpoint
"""

import io
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# Global model instance
class ModelManager:
    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.model = None
        self.active_users = 0
        self.lock = threading.Lock()
        self.timer = None
        self.unload_timeout = 5.0  # 5 seconds

    def get(self):
        return ModelContext(self)

    def _acquire(self):
        with self.lock:
            if self.timer:
                self.timer.cancel()
                self.timer = None
            
            if self.model is None:
                self.model = self.model_factory()
            
            self.active_users += 1
            return self.model

    def _release(self):
        with self.lock:
            self.active_users -= 1
            if self.active_users == 0:
                self.timer = threading.Timer(self.unload_timeout, self.unload_model)
                self.timer.start()

    def unload_model(self):
        with self.lock:
            if self.active_users == 0 and self.model is not None:
                print("Unloading model due to inactivity...")
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                print("Model unloaded.")

class ModelContext:
    def __init__(self, manager):
        self.manager = manager
        self.model = None

    def __enter__(self):
        self.model = self.manager._acquire()
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager._release()

def create_tts_model():
    from chatterbox.tts import ChatterboxTTS
    device = get_device()
    print(f"Loading ChatterboxTTS model on {device}...")
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded successfully!")
    return model

def create_multilingual_model():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    device = get_device()
    print(f"Loading ChatterboxMultilingualTTS model on {device}...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    print("Multilingual model loaded successfully!")
    return model

tts_manager = ModelManager(create_tts_model)
multilingual_manager = ModelManager(create_multilingual_model)

def get_model():
    return tts_manager.get()

def get_multilingual_model():
    return multilingual_manager.get()



class TTSRequest(BaseModel):
    """Request model for TTS generation."""
    text: str = Field(..., description="Text to synthesize. Supports [pause:Xs] tags for custom pauses (e.g., [pause:0.5s]).")
    exaggeration: float = Field(default=0.5, ge=0.0, le=2.0, description="Emotion exaggeration level (0.5 = neutral)")
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="CFG weight for pace control")
    temperature: float = Field(default=0.8, ge=0.05, le=5.0, description="Sampling temperature")
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0, description="Repetition penalty")
    min_p: float = Field(default=0.05, ge=0.0, le=1.0, description="Min-p sampling parameter")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    use_auto_editor: bool = Field(default=False, description="Enable artifact cleaning")
    ae_threshold: float = Field(default=0.06, ge=0.0, le=1.0, description="Auto-editor volume threshold")
    ae_margin: float = Field(default=0.2, ge=0.0, le=1.0, description="Auto-editor boundary protection")
    speed: float = Field(default=1.0, ge=0.1, le=3.0, description="Playback speed multiplier (1.0 = normal)")


class MultilingualTTSRequest(BaseModel):
    """Request model for multilingual TTS generation."""
    text: str = Field(..., description="Text to synthesize")
    language_id: str = Field(..., description="Language code (e.g., 'en', 'fr', 'zh', 'ja')")
    exaggeration: float = Field(default=0.5, ge=0.0, le=2.0, description="Emotion exaggeration level")
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="CFG weight for pace control")
    temperature: float = Field(default=0.8, ge=0.05, le=5.0, description="Sampling temperature")
    repetition_penalty: float = Field(default=2.0, ge=1.0, le=3.0, description="Repetition penalty")
    min_p: float = Field(default=0.05, ge=0.0, le=1.0, description="Min-p sampling parameter")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")


class OpenAISpeechRequest(BaseModel):
    """Compatibility model for OpenAI-style TTS requests (as in docs)."""
    input: str = Field(..., description="Text to convert to speech", min_length=1, max_length=3000)
    voice: Optional[str] = Field(default=None, description="Voice name or library id")
    response_format: Optional[str] = Field(default="wav", description="Ignored - API returns WAV")
    speed: float = Field(default=1.0, ge=0.1, le=3.0, description="Playback speed multiplier (1.0 = normal)")
    exaggeration: Optional[float] = Field(default=None, ge=0.25, le=2.0)
    cfg_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    temperature: Optional[float] = Field(default=None, ge=0.05, le=5.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Pre-load the model if PRELOAD_MODEL is set
    if os.environ.get("PRELOAD_MODEL", "false").lower() == "true":
        with get_model():
            pass
    yield
    # Shutdown: cleanup if needed
    pass


# Initialize FastAPI app
app = FastAPI(
    title="Chatterbox TTS API",
    description="HTTP API for Chatterbox Text-to-Speech with support for [pause:Xs] tags (e.g., [pause:0.5s])",
    version="1.0.0",
    lifespan=lifespan,
)


def get_device():
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "device": get_device(),
        "model_loaded": tts_manager.model is not None,
        "multilingual_model_loaded": multilingual_manager.model is not None,
    })


@app.get("/languages")
async def list_languages():
    """List supported languages for multilingual TTS."""
    from chatterbox.mtl_tts import SUPPORTED_LANGUAGES
    return JSONResponse({
        "languages": SUPPORTED_LANGUAGES
    })


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Generate speech from text.
    
    Supports [pause:Xs] tags for custom pauses. Example:
    - "Hello[pause:0.5s]world" - 0.5 second pause between words
    - "Wait[pause:1.0s]for it" - 1 second pause
    
    Returns: WAV audio file
    """
    try:
        with get_model() as tts_model:
            # Generate audio
            wav = tts_model.generate(
                text=request.text,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty,
                min_p=request.min_p,
                top_p=request.top_p,
                use_auto_editor=request.use_auto_editor,
                ae_threshold=request.ae_threshold,
                ae_margin=request.ae_margin,
                speed=request.speed,
            )
            
            # Convert to bytes
            buffer = io.BytesIO()
            torchaudio.save(buffer, wav, tts_model.sr, format="wav")
            buffer.seek(0)
            
            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"}
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech")
async def openai_style_speech(req: OpenAISpeechRequest):
    """OpenAI-compatible endpoint that maps `input` to local TTS generator and supports `speed`."""
    try:
        # NOTE: Updated to use context manager to prevent crashes with ModelManager
        with get_model() as tts_model:
            # Use provided override values or fall back to defaults
            exaggeration = req.exaggeration if req.exaggeration is not None else 0.5
            cfg_weight = req.cfg_weight if req.cfg_weight is not None else 0.5
            temperature = req.temperature if req.temperature is not None else 0.8

            wav = tts_model.generate(
                text=req.input,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                speed=req.speed,
            )

            buffer = io.BytesIO()
            torchaudio.save(buffer, wav, tts_model.sr, format="wav")
            buffer.seek(0)

            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"}
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "chatterbox-tts-1",
                "object": "model",
                "created": 1677649963,
                "owned_by": "resemble-ai",
            }
        ]
    })


@app.get("/config")
async def get_config():
    cfg = {
        "server": {"host": "0.0.0.0", "port": int(os.environ.get("PORT", 8000))},
        "model": {"device": get_device(), "voice_sample_path": os.environ.get("VOICE_SAMPLE_PATH", "./voice-sample.mp3"), "model_cache_dir": os.environ.get("MODEL_CACHE_DIR", "./models")},
        "defaults": {"exaggeration": float(os.environ.get("EXAGGERATION", 0.5)), "cfg_weight": float(os.environ.get("CFG_WEIGHT", 0.5)), "temperature": float(os.environ.get("TEMPERATURE", 0.8)), "max_chunk_length": int(os.environ.get("MAX_CHUNK_LENGTH", 280)), "max_total_length": int(os.environ.get("MAX_TOTAL_LENGTH", 3000))}
    }
    return JSONResponse(cfg)


@app.post("/tts/with-voice")
async def text_to_speech_with_voice(
    text: str = Form(..., description="Text to synthesize. Supports [pause:Xs] tags (e.g., [pause:0.5s])."),
    voice_file: UploadFile = File(..., description="Reference voice audio file (WAV format)"),
    exaggeration: float = Form(default=0.5),
    cfg_weight: float = Form(default=0.5),
    temperature: float = Form(default=0.8),
    repetition_penalty: float = Form(default=1.2),
    min_p: float = Form(default=0.05),
    top_p: float = Form(default=1.0),
    use_auto_editor: bool = Form(default=False),
    speed: float = Form(default=1.0),
):
    """
    Generate speech with a custom voice reference.
    
    Upload a WAV file to clone the voice characteristics.
    Supports [pause:Xs] tags for custom pauses (e.g., [pause:0.5s]).
    
    Returns: WAV audio file
    """
    try:
        with get_model() as tts_model:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                content = await voice_file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                # Generate audio with voice reference
                wav = tts_model.generate(
                    text=text,
                    audio_prompt_path=tmp_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                    use_auto_editor=use_auto_editor,
                    speed=speed,
                )
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
            
            # Convert to bytes
            buffer = io.BytesIO()
            torchaudio.save(buffer, wav, tts_model.sr, format="wav")
            buffer.seek(0)
            
            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"}
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/multilingual")
async def multilingual_text_to_speech(request: MultilingualTTSRequest):
    """
    Generate multilingual speech from text.
    
    Supports 23 languages. Use /languages endpoint to see supported language codes.
    
    Returns: WAV audio file
    """
    try:
        with get_multilingual_model() as mtl_model:
            # Generate audio
            wav = mtl_model.generate(
                text=request.text,
                language_id=request.language_id,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty,
                min_p=request.min_p,
                top_p=request.top_p,
                speed=1.0,
            )
            
            # Convert to bytes
            buffer = io.BytesIO()
            torchaudio.save(buffer, wav, mtl_model.sr, format="wav")
            buffer.seek(0)
            
            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"}
            )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)