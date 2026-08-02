"""
voice_input.py

Transcribes recorded audio to text using OpenAI's Whisper model, so
voice input can be analyzed by the same detection pipeline as typed
text. Uses the smallest ("tiny") model by default for speed - a
security-demo use case doesn't need large-model accuracy.

First run downloads the model (~75MB for "tiny") from OpenAI's servers -
needs a normal internet connection once. If that fails (no internet, a
restrictive network, etc.), this raises a clear, specific error rather
than a cryptic one, so the app can show a helpful message instead of
crashing.
"""

import tempfile
import os
import shutil
import sys
import stat

# Whisper needs ffmpeg to decode audio, and hardcodes the literal command
# name "ffmpeg" - it has no option to point at a custom path. Rather than
# requiring users to install ffmpeg system-wide and edit their PATH (real
# friction hit during testing - Windows doesn't ship ffmpeg), use
# imageio-ffmpeg's bundled binary. That binary isn't literally named
# "ffmpeg" though (e.g. "ffmpeg-linux-x86_64-v7.0.2"), so PATH alone
# doesn't help - a correctly-named copy has to exist somewhere on PATH.
# This runs once at import time, before any transcription.
def _ensure_ffmpeg_on_path():
    if shutil.which("ffmpeg"):
        return  # already available, nothing to do
    try:
        import imageio_ffmpeg
        bundled_path = imageio_ffmpeg.get_ffmpeg_exe()
        target_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        target_dir = os.path.join(tempfile.gettempdir(), "ai_soc_ffmpeg")
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, target_name)
        if not os.path.exists(target_path):
            shutil.copy2(bundled_path, target_path)
            if sys.platform != "win32":
                st = os.stat(target_path)
                os.chmod(target_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        if target_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = target_dir + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass  # falls through to whatever ffmpeg (if any) is already on PATH

_ensure_ffmpeg_on_path()

_model = None
_model_load_error = None


def get_model(model_size="tiny"):
    """Loads (and caches) the Whisper model. Raises RuntimeError with a
    clear message if the download fails, instead of letting urllib's
    raw HTTPError surface."""
    global _model, _model_load_error
    if _model is not None:
        return _model
    if _model_load_error is not None:
        raise _model_load_error
    try:
        import whisper
        _model = whisper.load_model(model_size)
        return _model
    except Exception as e:
        _model_load_error = RuntimeError(
            f"Couldn't load the Whisper speech-to-text model ({e}). "
            f"This needs internet access on first run to download the "
            f"model (~75MB). Check your connection and try again."
        )
        raise _model_load_error


def transcribe_audio_bytes(audio_bytes: bytes, suffix=".wav") -> str:
    """Takes raw audio bytes (e.g. from st.audio_input), saves to a temp
    file, and returns the transcribed text. Raises RuntimeError with a
    clear message on failure."""
    model = get_model()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        result = model.transcribe(tmp_path)
        return result["text"].strip()
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")
    finally:
        os.unlink(tmp_path)
