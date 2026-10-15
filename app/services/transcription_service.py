"""
SentriMail Transcription Service
---------------------------------
Audio transcription service using OpenAI Whisper.

Fixes applied:
- Bug #1 (MODEL_COMPARISON.md §4.1): Added missing `tempfile` and `os` imports
  that caused NameError crashes on every audio upload request.
- Bug #3 (MODEL_COMPARISON.md §4.2): Whisper model is now pre-loaded as a
  module-level singleton to eliminate the 1-3 second per-request load spike
  and prevent RAM allocation on every audio upload.
"""

import os
import tempfile
import logging

logger = logging.getLogger(__name__)

# ── Singleton Whisper model ────────────────────────────────────────────────────
# Pre-loaded once at module import (triggered by startup_event in main.py).
# Avoids the 1-3 second whisper.load_model("base") call per request.
_whisper_model = None
_whisper_loaded = False


def _load_whisper_model() -> None:
    """Load the Whisper base model once and cache it as a module-level singleton."""
    global _whisper_model, _whisper_loaded
    if _whisper_loaded:
        return
    _whisper_loaded = True
    try:
        import whisper
        _whisper_model = whisper.load_model("base")
        logger.info("Whisper 'base' model pre-loaded successfully.")
    except Exception as exc:
        _whisper_model = None
        logger.warning("Whisper model failed to pre-load: %s", exc)


async def transcribe_audio_file(audio_bytes: bytes) -> str:
    """Transcribe audio bytes to text using the pre-loaded Whisper singleton.

    Returns an empty string when Whisper is unavailable or transcription fails.
    """
    if not _whisper_loaded:
        _load_whisper_model()

    if _whisper_model is None:
        logger.warning("Whisper model is not available; returning empty transcription.")
        return ""

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = _whisper_model.transcribe(tmp_path)
        text = result.get("text", "")
        return text.strip()
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
