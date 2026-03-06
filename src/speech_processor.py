"""
Speech Processor Module
Transcribes audio recordings using OpenAI Whisper (local or API).
Also extracts prosodic features for confidence analysis.
"""

import io
import os
import time
import wave
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import whisper as _whisper
    _whisper_local = True
except ImportError:
    _whisper_local = False

try:
    from openai import OpenAI as _OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

try:
    import librosa
    _librosa_available = True
except ImportError:
    _librosa_available = False


# ---------------------------------------------------------------------------
# Transcription backends
# ---------------------------------------------------------------------------
class LocalWhisperTranscriber:
    """Runs Whisper locally (requires `openai-whisper` package + ffmpeg)."""

    def __init__(self, model_size: str = "base"):
        if not _whisper_local:
            raise ImportError("Install: pip install openai-whisper")
        print(f"[Whisper] Loading local model: {model_size} ...")
        self.model = _whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> dict:
        result = self.model.transcribe(audio_path, fp16=False)
        return {
            "text": result["text"].strip(),
            "language": result.get("language", "en"),
            "segments": result.get("segments", []),
        }


class OpenAIWhisperTranscriber:
    """Uses the OpenAI Whisper API (requires API key, no local GPU needed)."""

    def __init__(self):
        if not _openai_available:
            raise ImportError("Install: pip install openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY env variable.")
        self.client = _OpenAI(api_key=api_key)

    def transcribe(self, audio_path: str) -> dict:
        with open(audio_path, "rb") as f:
            result = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
            )
        return {
            "text": result.text.strip(),
            "language": getattr(result, "language", "en"),
            "segments": getattr(result, "segments", []),
        }


class MockTranscriber:
    """Returns a placeholder transcript — for UI development and testing."""

    _SAMPLE = (
        "I have extensive experience building machine learning pipelines. "
        "In my last role, I led a team of three data scientists to develop a "
        "churn prediction model that reduced customer attrition by 18 percent. "
        "I'm comfortable working with Python, scikit-learn, and cloud platforms. "
        "I believe strong communication between technical and business teams is "
        "critical for project success, and I always make sure to document my work thoroughly."
    )

    def transcribe(self, audio_path: str) -> dict:
        time.sleep(0.5)  # Simulate latency
        return {
            "text": self._SAMPLE,
            "language": "en",
            "segments": [],
        }


# ---------------------------------------------------------------------------
# Prosodic / acoustic feature extractor
# ---------------------------------------------------------------------------
class AcousticAnalyzer:
    """
    Extracts speaking-rate, pitch variance, pause ratio, and energy
    from a WAV file to support confidence scoring.
    Falls back to dummy values if librosa is not installed.
    """

    def analyze(self, audio_path: str, sample_rate: int = 16000) -> dict:
        if not _librosa_available:
            return self._dummy_features()

        try:
            y, sr = librosa.load(audio_path, sr=sample_rate)

            # ── Duration ──────────────────────────────────────────────────────
            duration = librosa.get_duration(y=y, sr=sr)

            # ── Silence / pause ratio ─────────────────────────────────────────
            intervals = librosa.effects.split(y, top_db=30)
            speech_duration = sum(e - s for s, e in intervals) / sr
            pause_ratio = 1.0 - (speech_duration / duration) if duration > 0 else 0.5

            # ── Pitch (F0) statistics ──────────────────────────────────────────
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
            )
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
            pitch_mean = float(np.nanmean(voiced_f0)) if len(voiced_f0) else 150.0
            pitch_std = float(np.nanstd(voiced_f0)) if len(voiced_f0) else 30.0

            # ── RMS energy ────────────────────────────────────────────────────
            rms = librosa.feature.rms(y=y)[0]
            energy_mean = float(np.mean(rms))
            energy_std = float(np.std(rms))

            # ── Speaking rate proxy (zero-crossing rate) ──────────────────────
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

            return {
                "duration_seconds": round(duration, 2),
                "pause_ratio": round(pause_ratio, 3),
                "pitch_mean_hz": round(pitch_mean, 1),
                "pitch_std_hz": round(pitch_std, 1),
                "energy_mean": round(energy_mean, 5),
                "energy_std": round(energy_std, 5),
                "zcr_proxy": round(zcr, 5),
            }
        except Exception as e:
            print(f"[AcousticAnalyzer] Error: {e}")
            return self._dummy_features()

    @staticmethod
    def _dummy_features() -> dict:
        return {
            "duration_seconds": 45.0,
            "pause_ratio": 0.18,
            "pitch_mean_hz": 165.0,
            "pitch_std_hz": 28.0,
            "energy_mean": 0.04,
            "energy_std": 0.015,
            "zcr_proxy": 0.08,
        }


# ---------------------------------------------------------------------------
# Unified processor
# ---------------------------------------------------------------------------
class SpeechProcessor:
    """
    High-level class combining transcription + acoustic analysis.

    Usage:
        processor = SpeechProcessor(backend="local_whisper", whisper_model="base")
        result = processor.process("answer.wav")
        # result["transcript"], result["acoustic_features"]
    """

    def __init__(
        self,
        backend: str = "mock",
        whisper_model: str = "base",
    ):
        if backend == "local_whisper":
            self.transcriber = LocalWhisperTranscriber(model_size=whisper_model)
        elif backend == "openai_whisper":
            self.transcriber = OpenAIWhisperTranscriber()
        else:
            self.transcriber = MockTranscriber()

        self.acoustic_analyzer = AcousticAnalyzer()

    def process(self, audio_path: str) -> dict:
        """
        Transcribe audio and extract acoustic features.
        Returns a dict with keys: transcript, language, segments, acoustic_features.
        """
        transcription = self.transcriber.transcribe(audio_path)
        acoustic = self.acoustic_analyzer.analyze(audio_path)

        return {
            "transcript": transcription["text"],
            "language": transcription["language"],
            "segments": transcription["segments"],
            "acoustic_features": acoustic,
        }

    @staticmethod
    def save_audio_bytes(audio_bytes: bytes, suffix: str = ".wav") -> str:
        """Write raw bytes to a temp file and return the path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            return f.name
