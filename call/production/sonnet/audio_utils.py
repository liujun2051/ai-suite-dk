# 文件2: `audio_utils.py`

```python
"""
audio_utils.py - Audio processing utilities for VoiceBot

Handles:
  - Raw binary PCM parsing from mod_audio_stream WebSocket frames
  - PCM resampling (8kHz ↔ 16kHz ↔ 24kHz)
  - Voice Activity Detection (WebRTC VAD + energy-based fallback)
  - Audio normalization
  - Audio chunking / buffering helpers
  - RMS / dBFS calculation
  - Silence generation
  - Audio format validation

mod_audio_stream sends raw binary PCM frames (no JSON wrapper):
  - Frame = raw L16 (signed 16-bit little-endian) PCM samples
  - Sample rate: configured in FreeSWITCH dialplan (8000 or 16000)
  - Channels: 1 (mono)
  - Each WebSocket message = one audio chunk (typically 20ms worth of samples)
"""

import array
import audioop
import logging
import math
import struct
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    logger.warning("numpy not available — resampling will use audioop (lower quality)")

try:
    import webrtcvad
    _WEBRTCVAD_AVAILABLE = True
except ImportError:
    _WEBRTCVAD_AVAILABLE = False
    logger.warning("webrtcvad not available — falling back to energy-based VAD")

try:
    from scipy import signal as scipy_signal
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_WIDTH_BYTES   = 2          # 16-bit PCM
SAMPLE_MAX_VALUE     = 32767      # max amplitude for int16
SAMPLE_MIN_VALUE     = -32768
CHANNELS             = 1          # always mono for telephony

# Supported sample rates
SUPPORTED_RATES      = (8000, 16000, 24000, 48000)

# WebRTC VAD only supports these frame durations (ms)
WEBRTC_VALID_FRAME_MS = (10, 20, 30)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class VADState(Enum):
    SILENCE = auto()
    SPEECH  = auto()


class ResampleQuality(Enum):
    LOW    = "low"     # audioop linear interpolation — fastest
    MEDIUM = "medium"  # numpy linear interpolation
    HIGH   = "high"    # scipy polyphase filter — best quality


@dataclass
class VADResult:
    """Result of a single VAD analysis frame."""
    state: VADState
    is_speech: bool
    energy_dbfs: float
    frame_duration_ms: int
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def is_silence(self) -> bool:
        return not self.is_speech


@dataclass
class AudioChunk:
    """
    A single chunk of raw PCM audio with metadata.
    data: raw bytes (signed 16-bit little-endian, mono)
    """
    data: bytes
    sample_rate: int
    timestamp: float = field(default_factory=time.monotonic)
    duration_ms: float = 0.0
    sequence: int = 0

    def __post_init__(self):
        if self.duration_ms == 0.0:
            self.duration_ms = calc_duration_ms(
                len(self.data), self.sample_rate
            )

    @property
    def num_samples(self) -> int:
        return len(self.data) // SAMPLE_WIDTH_BYTES

    @property
    def is_empty(self) -> bool:
        return len(self.data) == 0


@dataclass
class SpeechSegment:
    """
    A complete speech utterance detected by VAD.
    Contains the accumulated PCM bytes of a user's spoken turn.
    """
    audio_data: bytes
    sample_rate: int
    start_time: float
    end_time: float
    energy_dbfs: float
    frame_count: int

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def num_samples(self) -> int:
        return len(self.audio_data) // SAMPLE_WIDTH_BYTES

    def __repr__(self) -> str:
        return (
            f"SpeechSegment(duration={self.duration_ms:.0f}ms "
            f"bytes={len(self.audio_data)} "
            f"energy={self.energy_dbfs:.1f}dBFS)"
        )


# ---------------------------------------------------------------------------
# PCM math helpers
# ---------------------------------------------------------------------------

def calc_duration_ms(num_bytes: int, sample_rate: int) -> float:
    """Calculate audio duration in milliseconds from byte count."""
    if sample_rate <= 0:
        return 0.0
    num_samples = num_bytes // SAMPLE_WIDTH_BYTES
    return (num_samples / sample_rate) * 1000.0


def calc_num_bytes(duration_ms: float, sample_rate: int) -> int:
    """Calculate number of bytes for a given duration."""
    num_samples = int((duration_ms / 1000.0) * sample_rate)
    return num_samples * SAMPLE_WIDTH_BYTES


def calc_rms(pcm_bytes: bytes) -> float:
    """
    Calculate RMS amplitude of PCM audio.
    Returns value in range [0.0, 32767.0].
    Returns 0.0 for empty or silent audio.
    """
    if not pcm_bytes:
        return 0.0
    # audioop.rms is fast and doesn't need numpy
    try:
        return float(audioop.rms(pcm_bytes, SAMPLE_WIDTH_BYTES))
    except Exception:
        return 0.0


def calc_dbfs(pcm_bytes: bytes) -> float:
    """
    Calculate audio level in dBFS (decibels relative to full scale).
    0 dBFS = maximum possible amplitude (32767).
    Silence ≈ -96 dBFS (for 16-bit audio).
    Returns -96.0 for silent/empty audio.
    """
    rms = calc_rms(pcm_bytes)
    if rms <= 0:
        return -96.0
    return 20.0 * math.log10(rms / SAMPLE_MAX_VALUE)


def dbfs_to_linear(dbfs: float) -> float:
    """Convert dBFS value to linear scale [0.0, 1.0]."""
    return 10.0 ** (dbfs / 20.0)


def is_silent(pcm_bytes: bytes, threshold_dbfs: float = -40.0) -> bool:
    """Return True if audio energy is below threshold."""
    return calc_dbfs(pcm_bytes) < threshold_dbfs


def generate_silence(duration_ms: float, sample_rate: int) -> bytes:
    """Generate silent PCM audio (all zeros) of given duration."""
    num_bytes = calc_num_bytes(duration_ms, sample_rate)
    return b"\x00" * num_bytes


def generate_tone(
    frequency_hz: float,
    duration_ms: float,
    sample_rate: int,
    amplitude: float = 0.3,
) -> bytes:
    """
    Generate a pure sine wave tone.
    Useful for testing audio pipeline end-to-end.
    amplitude: 0.0 to 1.0
    """
    num_samples = int((duration_ms / 1000.0) * sample_rate)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        sample = int(amplitude * SAMPLE_MAX_VALUE * math.sin(2 * math.pi * frequency_hz * t))
        sample = max(SAMPLE_MIN_VALUE, min(SAMPLE_MAX_VALUE, sample))
        samples.append(sample)
    return struct.pack(f"<{num_samples}h", *samples)


def pcm_to_int16_list(pcm_bytes: bytes) -> List[int]:
    """Unpack raw PCM bytes to list of signed 16-bit integers."""
    num_samples = len(pcm_bytes) // SAMPLE_WIDTH_BYTES
    return list(struct.unpack(f"<{num_samples}h", pcm_bytes[:num_samples * SAMPLE_WIDTH_BYTES]))


def int16_list_to_pcm(samples: List[int]) -> bytes:
    """Pack list of signed 16-bit integers to raw PCM bytes."""
    return struct.pack(f"<{len(samples)}h", *samples)


def clamp_pcm(pcm_bytes: bytes) -> bytes:
    """Clamp all samples to valid int16 range."""
    samples = pcm_to_int16_list(pcm_bytes)
    clamped = [max(SAMPLE_MIN_VALUE, min(SAMPLE_MAX_VALUE, s)) for s in samples]
    return int16_list_to_pcm(clamped)


def mix_pcm(a: bytes, b: bytes) -> bytes:
    """
    Mix two PCM streams by averaging samples.
    Streams must be same length, same sample rate.
    """
    if len(a) != len(b):
        # Pad shorter one with silence
        maxlen = max(len(a), len(b))
        a = a.ljust(maxlen, b"\x00")
        b = b.ljust(maxlen, b"\x00")
    try:
        return audioop.add(a, b, SAMPLE_WIDTH_BYTES)
    except Exception:
        return a


def pcm_gain(pcm_bytes: bytes, gain_db: float) -> bytes:
    """
    Apply gain (in dB) to PCM audio.
    Positive = louder, negative = quieter.
    Clips to valid int16 range.
    """
    if gain_db == 0.0:
        return pcm_bytes
    linear_gain = 10.0 ** (gain_db / 20.0)
    try:
        # audioop.mul applies linear gain factor
        return audioop.mul(pcm_bytes, SAMPLE_WIDTH_BYTES, linear_gain)
    except Exception:
        return pcm_bytes


def normalize_pcm(
    pcm_bytes: bytes,
    target_dbfs: float = -20.0,
) -> bytes:
    """
    Normalize audio to target dBFS level.
    Calculates required gain and applies it.
    """
    if not pcm_bytes:
        return pcm_bytes
    current_dbfs = calc_dbfs(pcm_bytes)
    if current_dbfs <= -90.0:
        return pcm_bytes  # too silent to normalize meaningfully
    gain_db = target_dbfs - current_dbfs
    # Safety: don't apply extreme gain (would amplify noise)
    gain_db = max(-30.0, min(30.0, gain_db))
    return pcm_gain(pcm_bytes, gain_db)


def validate_pcm(pcm_bytes: bytes, sample_rate: int) -> Tuple[bool, str]:
    """
    Validate PCM data.
    Returns (is_valid, error_message).
    """
    if not pcm_bytes:
        return False, "Empty PCM data"
    if len(pcm_bytes) % SAMPLE_WIDTH_BYTES != 0:
        return False, (
            f"PCM length {len(pcm_bytes)} not divisible by "
            f"sample_width={SAMPLE_WIDTH_BYTES}"
        )
    if sample_rate not in SUPPORTED_RATES:
        return False, (
            f"Unsupported sample rate {sample_rate}. "
            f"Supported: {SUPPORTED_RATES}"
        )
    return True, ""


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_pcm(
    pcm_bytes: bytes,
    from_rate: int,
    to_rate: int,
    quality: ResampleQuality = ResampleQuality.MEDIUM,
) -> bytes:
    """
    Resample PCM audio from one sample rate to another.

    Args:
        pcm_bytes:  Raw 16-bit signed PCM bytes
        from_rate:  Source sample rate (Hz)
        to_rate:    Target sample rate (Hz)
        quality:    Resampling quality / algorithm

    Returns:
        Resampled raw 16-bit signed PCM bytes
    """
    if from_rate == to_rate:
        return pcm_bytes
    if not pcm_bytes:
        return pcm_bytes

    # Validate
    ok, err = validate_pcm(pcm_bytes, from_rate)
    if not ok:
        logger.warning("resample_pcm: invalid input: %s", err)
        return pcm_bytes

    try:
        if quality == ResampleQuality.LOW or not _NUMPY_AVAILABLE:
            return _resample_audioop(pcm_bytes, from_rate, to_rate)
        elif quality == ResampleQuality.MEDIUM or not _SCIPY_AVAILABLE:
            return _resample_numpy(pcm_bytes, from_rate, to_rate)
        else:
            return _resample_scipy(pcm_bytes, from_rate, to_rate)
    except Exception as e:
        logger.error(
            "Resampling failed (%d→%d): %s — falling back to audioop",
            from_rate, to_rate, e
        )
        return _resample_audioop(pcm_bytes, from_rate, to_rate)


def _resample_audioop(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """
    Resample using Python's built-in audioop.ratecv.
    Lowest quality but zero dependencies.
    Uses linear interpolation internally.
    """
    # audioop.ratecv signature:
    # ratecv(fragment, width, nchannels, inrate, outrate, state, weightA=1, weightB=0)
    resampled, _ = audioop.ratecv(
        pcm_bytes,
        SAMPLE_WIDTH_BYTES,
        CHANNELS,
        from_rate,
        to_rate,
        None,       # state (None = start fresh)
        1,          # weightA
        0,          # weightB
    )
    return resampled


def _resample_numpy(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """
    Resample using numpy linear interpolation.
    Medium quality, good performance.
    """
    # Unpack to numpy int16 array
    num_samples = len(pcm_bytes) // SAMPLE_WIDTH_BYTES
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

    # Calculate output length
    out_length = int(num_samples * to_rate / from_rate)
    if out_length <= 0:
        return b""

    # Linear interpolation via numpy
    x_old = np.linspace(0, 1, num_samples)
    x_new = np.linspace(0, 1, out_length)
    resampled = np.interp(x_new, x_old, samples)

    # Convert back to int16
    resampled = np.clip(resampled, SAMPLE_MIN_VALUE, SAMPLE_MAX_VALUE)
    return resampled.astype(np.int16).tobytes()


def _resample_scipy(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """
    Resample using scipy polyphase filter.
    Highest quality, best for audio.
    """
    num_samples = len(pcm_bytes) // SAMPLE_WIDTH_BYTES
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

    # Calculate GCD to reduce ratio
    from math import gcd
    g = gcd(from_rate, to_rate)
    up   = to_rate   // g
    down = from_rate // g

    resampled = scipy_signal.resample_poly(samples, up, down)
    resampled = np.clip(resampled, SAMPLE_MIN_VALUE, SAMPLE_MAX_VALUE)
    return resampled.astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Audio chunking
# ---------------------------------------------------------------------------

def chunk_pcm(
    pcm_bytes: bytes,
    chunk_ms: int,
    sample_rate: int,
    pad_last: bool = True,
) -> Generator[bytes, None, None]:
    """
    Split PCM audio into fixed-size chunks.

    Args:
        pcm_bytes:  Input PCM bytes
        chunk_ms:   Desired chunk duration in milliseconds
        sample_rate: Sample rate of audio
        pad_last:   If True, pad the last chunk with silence to full size

    Yields:
        PCM byte chunks of exactly chunk_ms duration (if pad_last=True)
    """
    chunk_bytes = calc_num_bytes(chunk_ms, sample_rate)
    offset = 0
    while offset < len(pcm_bytes):
        chunk = pcm_bytes[offset:offset + chunk_bytes]
        if len(chunk) < chunk_bytes and pad_last:
            chunk = chunk + b"\x00" * (chunk_bytes - len(chunk))
        if chunk:
            yield chunk
        offset += chunk_bytes


def concat_pcm(chunks: List[bytes]) -> bytes:
    """Concatenate a list of PCM byte chunks."""
    return b"".join(chunks)


class PCMRingBuffer:
    """
    Thread-safe ring buffer for PCM audio data.
    Drops oldest data when full (never blocks on write).
    Used to buffer audio between the WebSocket receiver and the processing pipeline.
    """

    def __init__(self, max_duration_ms: float, sample_rate: int):
        self._sample_rate  = sample_rate
        self._max_bytes    = calc_num_bytes(max_duration_ms, sample_rate)
        self._buffer       = bytearray()
        self._lock         = threading.Lock()
        self._total_written = 0
        self._total_dropped = 0

    def write(self, pcm_bytes: bytes) -> int:
        """
        Write PCM bytes to buffer.
        If buffer would overflow, drops oldest data.
        Returns number of bytes actually written.
        """
        if not pcm_bytes:
            return 0
        with self._lock:
            available = self._max_bytes - len(self._buffer)
            if len(pcm_bytes) > available:
                # Drop oldest to make room
                drop = len(pcm_bytes) - available
                # Align to sample boundary
                drop = (drop // SAMPLE_WIDTH_BYTES) * SAMPLE_WIDTH_BYTES
                del self._buffer[:drop]
                self._total_dropped += drop
                logger.debug(
                    "PCMRingBuffer overflow: dropped %d bytes", drop
                )
            self._buffer.extend(pcm_bytes)
            self._total_written += len(pcm_bytes)
            return len(pcm_bytes)

    def read(self, num_bytes: int) -> bytes:
        """
        Read up to num_bytes from buffer.
        Returns whatever is available (may be less than requested).
        """
        with self._lock:
            # Align to sample boundary
            num_bytes = (num_bytes // SAMPLE_WIDTH_BYTES) * SAMPLE_WIDTH_BYTES
            chunk = bytes(self._buffer[:num_bytes])
            del self._buffer[:len(chunk)]
            return chunk

    def read_chunk(self, chunk_ms: int) -> Optional[bytes]:
        """
        Read exactly one chunk of chunk_ms duration.
        Returns None if not enough data available yet.
        """
        needed = calc_num_bytes(chunk_ms, self._sample_rate)
        with self._lock:
            if len(self._buffer) < needed:
                return None
            chunk = bytes(self._buffer[:needed])
            del self._buffer[:needed]
            return chunk

    def read_all(self) -> bytes:
        """Read and clear all available data."""
        with self._lock:
            data = bytes(self._buffer)
            self._buffer.clear()
            return data

    def peek(self, num_bytes: int) -> bytes:
        """Read without consuming."""
        with self._lock:
            return bytes(self._buffer[:num_bytes])

    def available_bytes(self) -> int:
        with self._lock:
            return len(self._buffer)

    def available_ms(self) -> float:
        return calc_duration_ms(self.available_bytes(), self._sample_rate)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "available_bytes": len(self._buffer),
                "available_ms": calc_duration_ms(len(self._buffer), self._sample_rate),
                "max_bytes": self._max_bytes,
                "total_written": self._total_written,
                "total_dropped": self._total_dropped,
            }


# ---------------------------------------------------------------------------
# Voice Activity Detection
# ---------------------------------------------------------------------------

class EnergyVAD:
    """
    Simple energy-based VAD.
    Uses RMS energy and a configurable dBFS threshold.
    No external dependencies.
    Suitable as fallback when webrtcvad is not available.
    """

    def __init__(
        self,
        threshold_dbfs: float       = -40.0,
        min_speech_frames: int      = 3,
        min_silence_frames: int     = 10,
        frame_ms: int               = 20,
        sample_rate: int            = 16000,
    ):
        self.threshold_dbfs     = threshold_dbfs
        self.min_speech_frames  = min_speech_frames
        self.min_silence_frames = min_silence_frames
        self.frame_ms           = frame_ms
        self.sample_rate        = sample_rate

        self._state             = VADState.SILENCE
        self._consecutive_speech  = 0
        self._consecutive_silence = 0

    def process_frame(self, pcm_frame: bytes) -> VADResult:
        """
        Process a single PCM frame.
        Frame must be exactly frame_ms of audio.
        """
        dbfs = calc_dbfs(pcm_frame)
        raw_is_speech = dbfs >= self.threshold_dbfs

        if raw_is_speech:
            self._consecutive_speech  += 1
            self._consecutive_silence  = 0
        else:
            self._consecutive_silence += 1
            self._consecutive_speech   = 0

        # Apply hysteresis to reduce chatter
        if (self._state == VADState.SILENCE
                and self._consecutive_speech >= self.min_speech_frames):
            self._state = VADState.SPEECH

        elif (self._state == VADState.SPEECH
                and self._consecutive_silence >= self.min_silence_frames):
            self._state = VADState.SILENCE

        return VADResult(
            state=self._state,
            is_speech=(self._state == VADState.SPEECH),
            energy_dbfs=dbfs,
            frame_duration_ms=self.frame_ms,
        )

    def reset(self) -> None:
        self._state = VADState.SILENCE
        self._consecutive_speech  = 0
        self._consecutive_silence = 0


class WebRTCVAD:
    """
    WebRTC-based VAD wrapper.
    Higher accuracy than energy VAD, especially in noisy conditions.
    Requires: pip install webrtcvad
    """

    def __init__(
        self,
        aggressiveness: int     = 2,
        sample_rate: int        = 16000,
        frame_ms: int           = 20,
        smoothing_frames: int   = 5,
        smoothing_threshold: int = 3,
        energy_threshold_dbfs: float = -50.0,
    ):
        if not _WEBRTCVAD_AVAILABLE:
            raise RuntimeError(
                "webrtcvad not installed. "
                "Install with: pip install webrtcvad"
            )
        if frame_ms not in WEBRTC_VALID_FRAME_MS:
            raise ValueError(
                f"frame_ms must be one of {WEBRTC_VALID_FRAME_MS}, got {frame_ms}"
            )
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(
                f"WebRTC VAD supports 8000/16000/32000/48000 Hz, got {sample_rate}"
            )

        self.aggressiveness       = aggressiveness
        self.sample_rate          = sample_rate
        self.frame_ms             = frame_ms
        self.smoothing_frames     = smoothing_frames
        self.smoothing_threshold  = smoothing_threshold
        self.energy_threshold_dbfs = energy_threshold_dbfs

        self._vad = webrtcvad.Vad(aggressiveness)
        self._frame_bytes = calc_num_bytes(frame_ms, sample_rate)

        # Smoothing window: deque of booleans
        self._history: Deque[bool] = deque(maxlen=smoothing_frames)
        self._state = VADState.SILENCE

    def process_frame(self, pcm_frame: bytes) -> VADResult:
        """
        Process a single PCM frame of exactly frame_ms duration.

        If the frame is not exactly the right size, it will be truncated
        or padded to fit. This is intentional for robustness.
        """
        # Ensure correct frame size
        expected = self._frame_bytes
        if len(pcm_frame) < expected:
            pcm_frame = pcm_frame + b"\x00" * (expected - len(pcm_frame))
        elif len(pcm_frame) > expected:
            pcm_frame = pcm_frame[:expected]

        dbfs = calc_dbfs(pcm_frame)

        # Hard energy gate — if audio is essentially silent, skip WebRTC
        if dbfs < self.energy_threshold_dbfs:
            raw_speech = False
        else:
            try:
                raw_speech = self._vad.is_speech(pcm_frame, self.sample_rate)
            except Exception as e:
                logger.warning("WebRTC VAD error: %s", e)
                raw_speech = False

        # Smooth decision using majority vote over history
        self._history.append(raw_speech)
        speech_count = sum(self._history)
        smoothed_speech = speech_count >= self.smoothing_threshold

        self._state = VADState.SPEECH if smoothed_speech else VADState.SILENCE

        return VADResult(
            state=self._state,
            is_speech=smoothed_speech,
            energy_dbfs=dbfs,
            frame_duration_ms=self.frame_ms,
        )

    def reset(self) -> None:
        self._history.clear()
        self._state = VADState.SILENCE

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes


def create_vad(
    use_webrtc: bool            = True,
    aggressiveness: int         = 2,
    sample_rate: int            = 16000,
    frame_ms: int               = 20,
    smoothing_frames: int       = 5,
    smoothing_threshold: int    = 3,
    energy_threshold_dbfs: float = -40.0,
) -> "WebRTCVAD | EnergyVAD":
    """
    Factory function: create the best available VAD.
    Falls back to EnergyVAD if webrtcvad is not installed.
    """
    if use_webrtc and _WEBRTCVAD_AVAILABLE:
        logger.info(
            "Creating WebRTC VAD (aggressiveness=%d, rate=%d, frame=%dms)",
            aggressiveness, sample_rate, frame_ms,
        )
        return WebRTCVAD(
            aggressiveness=aggressiveness,
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            smoothing_frames=smoothing_frames,
            smoothing_threshold=smoothing_threshold,
            energy_threshold_dbfs=energy_threshold_dbfs,
        )
    else:
        if use_webrtc and not _WEBRTCVAD_AVAILABLE:
            logger.warning(
                "webrtcvad not available, falling back to EnergyVAD"
            )
        logger.info(
            "Creating Energy VAD (threshold=%.1f dBFS, rate=%d, frame=%dms)",
            energy_threshold_dbfs, sample_rate, frame_ms,
        )
        return EnergyVAD(
            threshold_dbfs=energy_threshold_dbfs,
            frame_ms=frame_ms,
            sample_rate=sample_rate,
        )


# ---------------------------------------------------------------------------
# Utterance detector
# ---------------------------------------------------------------------------

class UtteranceDetector:
    """
    Stateful utterance boundary detector.

    Sits on top of a VAD and accumulates frames, emitting complete
    SpeechSegment objects when an utterance ends (silence after speech).

    Also detects interruptions: speech during bot playback.

    Usage:
        detector = UtteranceDetector(vad, config)
        for pcm_frame in audio_stream:
            result = detector.process(pcm_frame)
            if result:
                # complete utterance ready
                handle_utterance(result)
    """

    def __init__(
        self,
        vad:                    "WebRTCVAD | EnergyVAD",
        sample_rate:            int   = 16000,
        frame_ms:               int   = 20,
        min_speech_ms:          int   = 200,
        silence_ms:             int   = 700,
        max_utterance_ms:       int   = 30_000,
        interruption_speech_ms: int   = 250,
        interruption_silence_ms:int   = 200,
        on_speech_start:        Optional[Callable[[], None]] = None,
        on_speech_end:          Optional[Callable[[], None]] = None,
        on_interruption:        Optional[Callable[[], None]] = None,
    ):
        self._vad               = vad
        self.sample_rate        = sample_rate
        self.frame_ms           = frame_ms
        self.min_speech_ms      = min_speech_ms
        self.silence_ms         = silence_ms
        self.max_utterance_ms   = max_utterance_ms
        self.interruption_speech_ms  = interruption_speech_ms
        self.interruption_silence_ms = interruption_silence_ms

        self.on_speech_start    = on_speech_start
        self.on_speech_end      = on_speech_end
        self.on_interruption    = on_interruption

        # State
        self._state             = VADState.SILENCE
        self._speech_frames:    List[bytes] = []
        self._speech_ms:        float = 0.0
        self._silence_ms:       float = 0.0
        self._utterance_start:  Optional[float] = None
        self._last_energy:      float = -96.0

        # Interruption tracking
        self._bot_is_speaking:       bool  = False
        self._interruption_speech_ms: float = 0.0
        self._interruption_fired:    bool  = False

        self._lock = threading.Lock()

        # Frame bytes (for splitting oversized input)
        self._frame_bytes = calc_num_bytes(frame_ms, sample_rate)

    def set_bot_speaking(self, is_speaking: bool) -> None:
        """
        Call this to tell the detector whether the bot is currently
        playing audio. When True, interruption detection is active.
        """
        with self._lock:
            self._bot_is_speaking = is_speaking
            if not is_speaking:
                self._interruption_speech_ms = 0.0
                self._interruption_fired     = False

    def process(self, pcm_bytes: bytes) -> Optional[SpeechSegment]:
        """
        Process incoming PCM audio (any size).
        Internally splits into frame_ms chunks.

        Returns a SpeechSegment when a complete utterance is detected,
        otherwise returns None.
        """
        # Split into VAD-sized frames
        result: Optional[SpeechSegment] = None
        for frame in self._split_frames(pcm_bytes):
            seg = self._process_frame(frame)
            if seg is not None:
                result = seg
        return result

    def _split_frames(self, pcm_bytes: bytes) -> Generator[bytes, None, None]:
        """Split arbitrary-length PCM into exactly frame_ms frames."""
        offset = 0
        while offset + self._frame_bytes <= len(pcm_bytes):
            yield pcm_bytes[offset:offset + self._frame_bytes]
            offset += self._frame_bytes
        # Handle leftover (pad to full frame)
        leftover = pcm_bytes[offset:]
        if leftover:
            padded = leftover + b"\x00" * (self._frame_bytes - len(leftover))
            yield padded

    def _process_frame(self, frame: bytes) -> Optional[SpeechSegment]:
        """Core frame-by-frame state machine."""
        with self._lock:
            vad_result = self._vad.process_frame(frame)
            self._last_energy = vad_result.energy_dbfs

            # ---- Interruption detection ----
            if self._bot_is_speaking and not self._interruption_fired:
                if vad_result.is_speech:
                    self._interruption_speech_ms += self.frame_ms
                    if self._interruption_speech_ms >= self.interruption_speech_ms:
                        self._interruption_fired = True
                        logger.info(
                            "Interruption detected! "
                            "speech_ms=%.0f threshold=%.0f",
                            self._interruption_speech_ms,
                            self.interruption_speech_ms,
                        )
                        if self.on_interruption:
                            # Fire callback outside lock to avoid deadlock
                            threading.Thread(
                                target=self.on_interruption, daemon=True
                            ).start()
                else:
                    # Reset interruption counter on silence
                    self._interruption_speech_ms = max(
                        0.0,
                        self._interruption_speech_ms - self.frame_ms
                    )

            # ---- Utterance state machine ----
            if vad_result.is_speech:
                if self._state == VADState.SILENCE:
                    # Transition: silence → speech
                    self._state = VADState.SPEECH
                    self._utterance_start = time.monotonic()
                    self._speech_frames.clear()
                    self._speech_ms = 0.0
                    self._silence_ms = 0.0
                    logger.debug("Speech started (energy=%.1fdBFS)", self._last_energy)
                    if self.on_speech_start:
                        threading.Thread(
                            target=self.on_speech_start, daemon=True
                        ).start()

                self._speech_frames.append(frame)
                self._speech_ms += self.frame_ms
                self._silence_ms = 0.0

                # Max utterance length guard
                if self._speech_ms >= self.max_utterance_ms:
                    logger.warning(
                        "Utterance exceeded max duration (%.0fms), forcing end",
                        self.max_utterance_ms,
                    )
                    return self._finalize_utterance()

            else:  # silence frame
                if self._state == VADState.SPEECH:
                    self._silence_ms += self.frame_ms
                    # Keep appending silence frames so we have natural trailing silence
                    self._speech_frames.append(frame)

                    if self._silence_ms >= self.silence_ms:
                        # Check minimum speech duration
                        if self._speech_ms >= self.min_speech_ms:
                            logger.debug(
                                "Speech ended (speech=%.0fms, silence=%.0fms)",
                                self._speech_ms, self._silence_ms,
                            )
                            if self.on_speech_end:
                                threading.Thread(
                                    target=self.on_speech_end, daemon=True
                                ).start()
                            return self._finalize_utterance()
                        else:
                            # Too short — treat as noise, reset
                            logger.debug(
                                "Speech too short (%.0fms < %.0fms min), discarding",
                                self._speech_ms, self.min_speech_ms,
                            )
                            self._reset_state()

            return None

    def _finalize_utterance(self) -> SpeechSegment:
        """Build and return a SpeechSegment, then reset state."""
        audio = concat_pcm(self._speech_frames)
        start = self._utterance_start or time.monotonic()
        end   = time.monotonic()
        seg = SpeechSegment(
            audio_data=audio,
            sample_rate=self.sample_rate,
            start_time=start,
            end_time=end,
            energy_dbfs=self._last_energy,
            frame_count=len(self._speech_frames),
        )
        self._reset_state()
        return seg

    def _reset_state(self) -> None:
        self._state = VADState.SILENCE
        self._speech_frames.clear()
        self._speech_ms   = 0.0
        self._silence_ms  = 0.0
        self._utterance_start = None

    def reset(self) -> None:
        """Full reset (call on session end or error)."""
        with self._lock:
            self._reset_state()
            self._vad.reset()
            self._bot_is_speaking = False
            self._interruption_speech_ms = 0.0
            self._interruption_fired = False

    @property
    def current_state(self) -> VADState:
        with self._lock:
            return self._state

    @property
    def current_energy_dbfs(self) -> float:
        with self._lock:
            return self._last_energy

    @property
    def is_speech_active(self) -> bool:
        with self._lock:
            return self._state == VADState.SPEECH


# ---------------------------------------------------------------------------
# Audio pipeline helper: frame splitter for mod_audio_stream
# ---------------------------------------------------------------------------

class ModAudioStreamFrameParser:
    """
    Parses raw binary WebSocket frames from mod_audio_stream.

    mod_audio_stream sends raw PCM binary frames (no JSON wrapper).
    Each WebSocket message is a binary frame containing L16 PCM samples.

    This class:
      - Validates incoming frames
      - Accumulates partial frames into a buffer
      - Emits complete, aligned chunks of the requested size
      - Tracks statistics
    """

    def __init__(
        self,
        sample_rate: int    = 16000,
        frame_ms:    int    = 20,
        output_ms:   int    = 100,
    ):
        """
        Args:
            sample_rate: Expected sample rate from FreeSWITCH
            frame_ms:    Size of frames sent by mod_audio_stream (typically 20ms)
            output_ms:   Size of chunks to emit to downstream processing
        """
        self.sample_rate  = sample_rate
        self.frame_ms     = frame_ms
        self.output_ms    = output_ms

        self._input_frame_bytes  = calc_num_bytes(frame_ms, sample_rate)
        self._output_chunk_bytes = calc_num_bytes(output_ms, sample_rate)

        self._buffer         = bytearray()
        self._lock           = threading.Lock()
        self._frames_received = 0
        self._bytes_received  = 0
        self._bytes_dropped   = 0
        self._chunks_emitted  = 0

    def feed(self, raw_frame: bytes) -> List[bytes]:
        """
        Feed a raw binary WebSocket frame from mod_audio_stream.

        Args:
            raw_frame: Raw bytes from WebSocket message.
                       Must be binary (not text/JSON).

        Returns:
            List of complete output chunks (may be empty if buffer
            not yet full enough to emit a complete output chunk).
        """
        if not raw_frame:
            return []

        # Basic sanity check: must be divisible by sample width
        if len(raw_frame) % SAMPLE_WIDTH_BYTES != 0:
            # Trim to aligned boundary
            aligned_len = (len(raw_frame) // SAMPLE_WIDTH_BYTES) * SAMPLE_WIDTH_BYTES
            dropped = len(raw_frame) - aligned_len
            raw_frame = raw_frame[:aligned_len]
            self._bytes_dropped += dropped
            logger.warning(
                "ModAudioStreamFrameParser: unaligned frame, "
                "dropped %d bytes", dropped
            )

        chunks_out: List[bytes] = []

        with self._lock:
            self._frames_received += 1
            self._bytes_received  += len(raw_frame)
            self._buffer.extend(raw_frame)

            # Emit complete output chunks
            while len(self._buffer) >= self._output_chunk_bytes:
                chunk = bytes(self._buffer[:self._output_chunk_bytes])
                del self._buffer[:self._output_chunk_bytes]
                chunks_out.append(chunk)
                self._chunks_emitted += 1

        return chunks_out

    def flush(self) -> Optional[bytes]:
        """
        Flush any remaining bytes in buffer (pad with silence to full chunk).
        Call on session end.
        """
        with self._lock:
            if not self._buffer:
                return None
            # Pad to output chunk size
            remainder = bytes(self._buffer)
            padded = remainder + b"\x00" * (
                self._output_chunk_bytes - len(remainder)
            )
            self._buffer.clear()
            return padded

    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()

    @property
    def buffered_bytes(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def buffered_ms(self) -> float:
        return calc_duration_ms(self.buffered_bytes, self.sample_rate)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "frames_received":  self._frames_received,
                "bytes_received":   self._bytes_received,
                "bytes_dropped":    self._bytes_dropped,
                "chunks_emitted":   self._chunks_emitted,
                "buffered_bytes":   len(self._buffer),
                "buffered_ms":      calc_duration_ms(
                    len(self._buffer), self.sample_rate
                ),
            }


# ---------------------------------------------------------------------------
# Audio format conversion utilities
# ---------------------------------------------------------------------------

def pcm_to_wav_bytes(
    pcm_bytes: bytes,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = SAMPLE_WIDTH_BYTES,
) -> bytes:
    """
    Wrap raw PCM bytes in a WAV file header.
    Returns complete WAV file bytes (in-memory, no file I/O).
    """
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def wav_bytes_to_pcm(wav_bytes: bytes) -> Tuple[bytes, int, int]:
    """
    Extract raw PCM from WAV file bytes.
    Returns (pcm_bytes, sample_rate, channels).
    """
    import io
    import wave
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sample_rate = wf.getframerate()
        channels    = wf.getnchannels()
        pcm_bytes   = wf.readframes(wf.getnframes())
    return pcm_bytes, sample_rate, channels


def stereo_to_mono(pcm_bytes: bytes) -> bytes:
    """Convert stereo (2-channel interleaved) PCM to mono by averaging channels."""
    try:
        return audioop.tomono(pcm_bytes, SAMPLE_WIDTH_BYTES, 0.5, 0.5)
    except Exception as e:
        logger.error("stereo_to_mono failed: %s", e)
        return pcm_bytes


def mono_to_stereo(pcm_bytes: bytes) -> bytes:
    """Convert mono PCM to stereo (duplicate channel)."""
    try:
        return audioop.tostereo(pcm_bytes, SAMPLE_WIDTH_BYTES, 1.0, 1.0)
    except Exception as e:
        logger.error("mono_to_stereo failed: %s", e)
        return pcm_bytes


def ulaw_to_pcm(ulaw_bytes: bytes) -> bytes:
    """Convert μ-law (G.711) encoded audio to 16-bit linear PCM."""
    try:
        return audioop.ulaw2lin(ulaw_bytes, SAMPLE_WIDTH_BYTES)
    except Exception as e:
        logger.error("ulaw_to_pcm failed: %s", e)
        return b""


def pcm_to_ulaw(pcm_bytes: bytes) -> bytes:
    """Convert 16-bit linear PCM to μ-law (G.711)."""
    try:
        return audioop.lin2ulaw(pcm_bytes, SAMPLE_WIDTH_BYTES)
    except Exception as e:
        logger.error("pcm_to_ulaw failed: %s", e)
        return b""


def alaw_to_pcm(alaw_bytes: bytes) -> bytes:
    """Convert A-law (G.711) encoded audio to 16-bit linear PCM."""
    try:
        return audioop.alaw2lin(alaw_bytes, SAMPLE_WIDTH_BYTES)
    except Exception as e:
        logger.error("alaw_to_pcm failed: %s", e)
        return b""


def pcm_to_alaw(pcm_bytes: bytes) -> bytes:
    """Convert 16-bit linear PCM to A-law (G.711)."""
    try:
        return audioop.lin2alaw(pcm_bytes, SAMPLE_WIDTH_BYTES)
    except Exception as e:
        logger.error("pcm_to_alaw failed: %s", e)
        return b""


# ---------------------------------------------------------------------------
# Audio pipeline builder (convenience factory)
# ---------------------------------------------------------------------------

@dataclass
class AudioPipelineConfig:
    """
    Configuration for building a complete audio processing pipeline.
    Wraps the relevant fields from AppConfig for this module.
    """
    fs_sample_rate:              int   = 16000
    model_sample_rate:           int   = 16000
    fs_chunk_ms:                 int   = 20
    model_chunk_ms:              int   = 100
    enable_resampling:           bool  = True
    enable_normalization:        bool  = True
    normalization_target_dbfs:   float = -20.0
    use_webrtc_vad:              bool  = True
    vad_aggressiveness:          int   = 2
    vad_frame_ms:                int   = 20
    vad_smoothing_frames:        int   = 5
    vad_smoothing_threshold:     int   = 3
    energy_threshold_dbfs:       float = -40.0
    min_speech_ms:               int   = 200
    silence_ms:                  int   = 700
    max_utterance_ms:            int   = 30_000
    interruption_speech_ms:      int   = 250
    interruption_silence_ms:     int   = 200
    max_buffer_duration_ms:      float = 5000.0


class AudioPipeline:
    """
    Complete audio processing pipeline for one call session.

    Wraps:
      - ModAudioStreamFrameParser  (raw frame → aligned chunks)
      - Resampler                  (FS rate → model rate)
      - Normalizer                 (level adjustment)
      - VAD + UtteranceDetector    (speech segmentation)

    Usage:
        pipeline = AudioPipeline(cfg)
        pipeline.set_callbacks(on_utterance=handle_utterance,
                               on_interruption=handle_interruption)

        # For each WebSocket binary frame from mod_audio_stream:
        pipeline.feed_frame(raw_bytes)

        # When bot starts/stops speaking:
        pipeline.set_bot_speaking(True)
        pipeline.set_bot_speaking(False)
    """

    def __init__(
        self,
        cfg: AudioPipelineConfig,
        on_utterance:    Optional[Callable[[SpeechSegment], None]] = None,
        on_speech_start: Optional[Callable[[], None]]              = None,
        on_speech_end:   Optional[Callable[[], None]]              = None,
        on_interruption: Optional[Callable[[], None]]              = None,
    ):
        self._cfg    = cfg
        self._on_utterance    = on_utterance
        self._on_speech_start = on_speech_start
        self._on_speech_end   = on_speech_end
        self._on_interruption = on_interruption

        # Frame parser: splits raw WebSocket frames into aligned chunks
        self._parser = ModAudioStreamFrameParser(
            sample_rate=cfg.fs_sample_rate,
            frame_ms=cfg.fs_chunk_ms,
            output_ms=cfg.vad_frame_ms,   # emit frames at VAD frame size
        )

        # VAD
        self._vad = create_vad(
            use_webrtc=cfg.use_webrtc_vad,
            aggressiveness=cfg.vad_aggressiveness,
            sample_rate=cfg.model_sample_rate,
            frame_ms=cfg.vad_frame_ms,
            smoothing_frames=cfg.vad_smoothing_frames,
            smoothing_threshold=cfg.vad_smoothing_threshold,
            energy_threshold_dbfs=cfg.energy_threshold_dbfs,
        )

        # Utterance detector
        self._detector = UtteranceDetector(
            vad=self._vad,
            sample_rate=cfg.model_sample_rate,
            frame_ms=cfg.vad_frame_ms,
            min_speech_ms=cfg.min_speech_ms,
            silence_ms=cfg.silence_ms,
            max_utterance_ms=cfg.max_utterance_ms,
            interruption_speech_ms=cfg.interruption_speech_ms,
            interruption_silence_ms=cfg.interruption_silence_ms,
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
            on_interruption=on_interruption,
        )

        # Resampling needed?
        self._needs_resample = (
            cfg.enable_resampling
            and cfg.fs_sample_rate != cfg.model_sample_rate
        )

        # Stats
        self._total_frames_fed     = 0
        self._total_bytes_received = 0
        self._total_utterances     = 0

        logger.info(
            "AudioPipeline created: FS=%dHz model=%dHz resample=%s "
            "vad=%s normalize=%s",
            cfg.fs_sample_rate, cfg.model_sample_rate,
            self._needs_resample,
            "webrtc" if cfg.use_webrtc_vad and _WEBRTCVAD_AVAILABLE else "energy",
            cfg.enable_normalization,
        )

    def feed_frame(self, raw_bytes: bytes) -> None:
        """
        Feed a raw binary WebSocket frame from mod_audio_stream.
        Processes synchronously in the calling thread.
        Call this from the WebSocket message handler.
        """
        if not raw_bytes:
            return

        self._total_frames_fed     += 1
        self._total_bytes_received += len(raw_bytes)

        # Step 1: Parse into VAD-sized frames
        frames = self._parser.feed(raw_bytes)

        for frame in frames:
            # Step 2: Resample if needed
            if self._needs_resample:
                frame = resample_pcm(
                    frame,
                    self._cfg.fs_sample_rate,
                    self._cfg.model_sample_rate,
                    quality=ResampleQuality.MEDIUM,
                )

            # Step 3: Normalize
            if self._cfg.enable_normalization:
                frame = normalize_pcm(
                    frame,
                    target_dbfs=self._cfg.normalization_target_dbfs,
                )

            # Step 4: VAD + utterance detection
            utterance = self._detector.process(frame)
            if utterance is not None:
                self._total_utterances += 1
                logger.info(
                    "Utterance #%d complete: %s",
                    self._total_utterances, utterance
                )
                if self._on_utterance:
                    try:
                        self._on_utterance(utterance)
                    except Exception as e:
                        logger.error("on_utterance callback error: %s", e)

    def set_bot_speaking(self, is_speaking: bool) -> None:
        """Notify pipeline whether the bot is currently speaking."""
        self._detector.set_bot_speaking(is_speaking)

    def reset(self) -> None:
        """Reset all stateful components (call on session end)."""
        self._parser.reset()
        self._detector.reset()

    @property
    def current_energy_dbfs(self) -> float:
        return self._detector.current_energy_dbfs

    @property
    def is_user_speaking(self) -> bool:
        return self._detector.is_speech_active

    @property
    def stats(self) -> dict:
        return {
            "total_frames_fed":     self._total_frames_fed,
            "total_bytes_received": self._total_bytes_received,
            "total_utterances":     self._total_utterances,
            "current_energy_dbfs":  self.current_energy_dbfs,
            "is_user_speaking":     self.is_user_speaking,
            "parser":               self._parser.stats,
        }
```

---

`audio_utils.py` 完成 ✅ 约 **780行**

包含：

| 组件 | 说明 |
|------|------|
| `calc_rms / calc_dbfs` | 音频能量计算 |
| `resample_pcm` | 三档质量重采样（audioop/numpy/scipy） |
| `normalize_pcm / pcm_gain` | 音量归一化 |
| `generate_silence / generate_tone` | 静音/测试音生成 |
| `PCMRingBuffer` | 线程安全环形缓冲 |
| `EnergyVAD` | 能量阈值VAD（无依赖fallback） |
| `WebRTCVAD` | WebRTC VAD封装（高精度） |
| `UtteranceDetector` | 完整的语音段检测状态机 + 打断检测 |
| `ModAudioStreamFrameParser` | 解析mod_audio_stream原始二进制帧 |
| `AudioPipeline` | 完整流水线工厂（一键组装所有组件） |
| 格式转换 | PCM↔WAV↔μ-law↔A-law |

---

**下一个：`esl.py`** — FreeSWITCH ESL连接、事件监听、uuid_break命令。准备好了就说继续！
