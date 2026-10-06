"""
SentriMail Transcription Service
---------------------------------
Audio transcription service using OpenAI Whisper.
"""

async def transcribe_audio_file(audio_bytes: bytes) -> str:
    try:
        import whisper
    except Exception as e:
        print(f"Whisper availability check failed: {e}")
        return ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = whisper.load_model("base")
        result = model.transcribe(tmp_path)
        text = result.get("text", "")
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        text = ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return text.strip()
