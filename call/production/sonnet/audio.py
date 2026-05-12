# 文件4: `audio.py`

```python
"""
audio.py - WebSocket audio server for mod_audio_stream

Handles:
  - WebSocket server that mod_audio_stream connects to
  - Receives raw binary PCM frames from FreeSWITCH (no JSON wrapper)
  - Sends raw binary PCM audio back to FreeSWITCH for playback
  - Per-connection AudioStream objects managing full duplex audio
  - Playback queue with interruption support
  - Backpressure handling
  - Connection lifecycle management
  - Integration with AudioPipeline (from audio_utils.py)

mod_audio_stream WebSocket protocol (raw binary mode):
  - Client (FreeSWITCH) connects to ws://our-server:8765
  - First message: JSON metadata with call info
  - Subsequent inbound messages: raw binary PCM (L16, mono)
  - Outbound messages we send: raw binary PCM (L16, mono)
  - Connection closes when call ends

FreeSWITCH dialplan configuration:
  <action application="audio_stream" data="ws://127.0.0.1:8765 16000 1"/>

```

---

`audio.py` 完成 ✅ 约 **700行**

包含：

| 组件 | 说明 |
|------|------|
| `CallMetadata` | 解析mod_audio_stream首帧JSON元数据 |
| `PlaybackQueue` | 有界异步播放队列，支持原子clear()打断 |
| `AudioStream` | 单通话全双工音频流管理，receive/send双loop |
| `_receive_loop` | 接收FS原始PCM，区分首帧JSON和后续binary |
| `_send_loop` | 定时drain队列发回FS，underrun时补静音 |
| `begin_speaking / end_speaking` | 精确控制bot说话状态 |
| `interrupt()` | 立即打断：清队列+通知pipeline |
| `AudioServer` | WebSocket服务器，管理所有并发连接 |
| `create_audio_server` | 从AppConfig一键创建 |

---

**下一个：`ai_client.py`** — MiniCPM-o 4.5 全双工WebSocket客户端，音频流输入输出，打断信号。准备好就说继续！
  
"""

import asyncio
import json
import logging
import time
import uuid
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Coroutine, Dict,
    List, Optional, Set
)

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

from audio_utils import (
    AudioPipeline,
    AudioPipelineConfig,
    SpeechSegment,
    PCMRingBuffer,
    calc_duration_ms,
    calc_num_bytes,
    generate_silence,
    SAMPLE_WIDTH_BYTES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How long to wait for playback queue item before checking stop flag (seconds)
PLAYBACK_DRAIN_INTERVAL  = 0.005   # 5ms
# Maximum audio we will buffer for playback (milliseconds)
MAX_PLAYBACK_BUFFER_MS   = 10_000  # 10 seconds
# Size of each PCM chunk sent to FreeSWITCH (milliseconds)
SEND_CHUNK_MS            = 20      # match mod_audio_stream frame size
# How long to wait between send attempts when throttling (seconds)
SEND_THROTTLE_SLEEP      = 0.001   # 1ms
# Maximum consecutive send errors before giving up
MAX_SEND_ERRORS          = 10


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class AudioStreamState(Enum):
    """Lifecycle states of a single mod_audio_stream WebSocket connection."""
    CONNECTING   = auto()   # WebSocket just connected, awaiting metadata
    READY        = auto()   # Metadata received, audio flowing
    INTERRUPTING = auto()   # Mid-interruption, draining queues
    CLOSING      = auto()   # Graceful shutdown in progress
    CLOSED       = auto()   # Connection closed


@dataclass
class CallMetadata:
    """
    Metadata extracted from the first JSON message sent by mod_audio_stream.
    FreeSWITCH sends this immediately after WebSocket connection.
    """
    call_uuid:      str
    caller_id_name: str  = ""
    caller_id_num:  str  = ""
    destination:    str  = ""
    sample_rate:    int  = 16000
    channels:       int  = 1
    direction:      str  = "inbound"
    extra:          Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: str) -> "CallMetadata":
        """
        Parse call metadata from the first WebSocket message.
        mod_audio_stream sends a JSON object with call variables.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid metadata JSON: {e}") from e

        return cls(
            call_uuid=data.get(
                "uuid",
                data.get("Unique-ID", str(uuid.uuid4()))
            ),
            caller_id_name=data.get(
                "Caller-Caller-ID-Name",
                data.get("caller_id_name", ""),
            ),
            caller_id_num=data.get(
                "Caller-Caller-ID-Number",
                data.get("caller_id_num", ""),
            ),
            destination=data.get(
                "Caller-Destination-Number",
                data.get("destination", ""),
            ),
            sample_rate=int(data.get("sample_rate", 16000)),
            channels=int(data.get("channels", 1)),
            direction=data.get(
                "Call-Direction",
                data.get("direction", "inbound"),
            ),
            extra=data,
        )

    def __repr__(self) -> str:
        return (
            f"CallMetadata(uuid={self.call_uuid[:8]}... "
            f"from={self.caller_id_num!r} "
            f"to={self.destination!r} "
            f"rate={self.sample_rate})"
        )


@dataclass
class AudioStreamStats:
    """Runtime statistics for one audio stream."""
    bytes_received:    int   = 0
    bytes_sent:        int   = 0
    frames_received:   int   = 0
    frames_sent:       int   = 0
    utterances:        int   = 0
    interruptions:     int   = 0
    playback_underruns:int   = 0
    send_errors:       int   = 0
    connected_at:      float = field(default_factory=time.time)
    last_rx_at:        Optional[float] = None
    last_tx_at:        Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        return time.time() - self.connected_at

    def to_dict(self) -> Dict:
        return {
            "bytes_received":     self.bytes_received,
            "bytes_sent":         self.bytes_sent,
            "frames_received":    self.frames_received,
            "frames_sent":        self.frames_sent,
            "utterances":         self.utterances,
            "interruptions":      self.interruptions,
            "playback_underruns": self.playback_underruns,
            "send_errors":        self.send_errors,
            "duration_seconds":   round(self.duration_seconds, 2),
        }


# ---------------------------------------------------------------------------
# PlaybackQueue
# ---------------------------------------------------------------------------

class PlaybackQueue:
    """
    Async queue for PCM audio chunks to be sent to FreeSWITCH.

    Features:
      - Bounded by max_duration_ms to prevent unbounded memory growth
      - Supports atomic clear() for interruption
      - Tracks total buffered duration
      - Signals when queue becomes non-empty (wake up sender)
    """

    def __init__(self, max_duration_ms: float, sample_rate: int):
        self._max_duration_ms = max_duration_ms
        self._sample_rate     = sample_rate
        self._queue:          asyncio.Queue = asyncio.Queue()
        self._buffered_ms:    float = 0.0
        self._lock:           asyncio.Lock = asyncio.Lock()
        self._total_enqueued: int   = 0
        self._total_dropped:  int   = 0

    async def put(self, pcm_chunk: bytes) -> bool:
        """
        Enqueue a PCM chunk for playback.
        Returns False (and drops chunk) if buffer is full.
        """
        chunk_ms = calc_duration_ms(len(pcm_chunk), self._sample_rate)

        async with self._lock:
            if self._buffered_ms + chunk_ms > self._max_duration_ms:
                self._total_dropped += 1
                logger.warning(
                    "PlaybackQueue full (%.0f/%.0fms), dropping chunk",
                    self._buffered_ms, self._max_duration_ms
                )
                return False
            self._buffered_ms  += chunk_ms
            self._total_enqueued += 1

        await self._queue.put(pcm_chunk)
        return True

    async def get(self, timeout: float = 0.1) -> Optional[bytes]:
        """
        Dequeue a PCM chunk.
        Returns None if queue is empty after timeout.
        """
        try:
            chunk = await asyncio.wait_for(
                self._queue.get(), timeout=timeout
            )
            chunk_ms = calc_duration_ms(len(chunk), self._sample_rate)
            async with self._lock:
                self._buffered_ms = max(0.0, self._buffered_ms - chunk_ms)
            return chunk
        except asyncio.TimeoutError:
            return None

    def clear(self) -> int:
        """
        Atomically clear all pending playback audio.
        Called on interruption.
        Returns number of chunks dropped.
        """
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break

        # Reset buffered duration
        asyncio.get_event_loop().create_task(self._reset_buffered())
        self._total_dropped += dropped

        if dropped > 0:
            logger.info(
                "PlaybackQueue cleared: dropped %d chunks", dropped
            )
        return dropped

    async def _reset_buffered(self) -> None:
        async with self._lock:
            self._buffered_ms = 0.0

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()

    @property
    def buffered_ms(self) -> float:
        return self._buffered_ms

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> Dict:
        return {
            "buffered_ms":    round(self._buffered_ms, 1),
            "qsize":          self._queue.qsize(),
            "total_enqueued": self._total_enqueued,
            "total_dropped":  self._total_dropped,
        }


# ---------------------------------------------------------------------------
# AudioStream  (one per WebSocket connection / call)
# ---------------------------------------------------------------------------

class AudioStream:
    """
    Manages the full-duplex audio stream for a single FreeSWITCH call.

    Lifecycle:
      1. Created when WebSocket connects
      2. Receives first JSON message → extracts CallMetadata
      3. Receives raw binary PCM frames → feeds AudioPipeline → VAD/utterance
      4. Sends raw binary PCM back to FreeSWITCH for playback
      5. Supports immediate interruption (clear queue + notify)
      6. Closed when WebSocket disconnects

    Thread model:
      - receive_loop: reads from WebSocket, feeds pipeline  (asyncio task)
      - send_loop:    drains PlaybackQueue, writes to WebSocket (asyncio task)
      - Both tasks run in the same event loop
    """

    def __init__(
        self,
        ws:          WebSocketServerProtocol,
        pipeline_cfg: AudioPipelineConfig,
        on_utterance:    Optional[Callable[[str, SpeechSegment], Coroutine]] = None,
        on_connected:    Optional[Callable[["AudioStream"], Coroutine]]      = None,
        on_disconnected: Optional[Callable[["AudioStream"], Coroutine]]      = None,
        on_interruption: Optional[Callable[[str], Coroutine]]                = None,
    ):
        """
        Args:
            ws:              WebSocket connection from mod_audio_stream
            pipeline_cfg:    Audio pipeline configuration
            on_utterance:    Async callback(call_uuid, SpeechSegment)
                             called when user finishes speaking
            on_connected:    Async callback(AudioStream) on ready
            on_disconnected: Async callback(AudioStream) on close
            on_interruption: Async callback(call_uuid) when user interrupts
        """
        self._ws             = ws
        self._pipeline_cfg   = pipeline_cfg
        self._on_utterance   = on_utterance
        self._on_connected   = on_connected
        self._on_disconnected = on_disconnected
        self._on_interruption = on_interruption

        # Set after first message parsed
        self.metadata:   Optional[CallMetadata] = None
        self.call_uuid:  str = ""

        # State
        self._state      = AudioStreamState.CONNECTING
        self._state_lock = asyncio.Lock()

        # Stats
        self.stats = AudioStreamStats()

        # Playback queue
        self._playback_queue = PlaybackQueue(
            max_duration_ms=MAX_PLAYBACK_BUFFER_MS,
            sample_rate=pipeline_cfg.model_sample_rate,
        )

        # Audio pipeline (created after metadata received)
        self._pipeline: Optional[AudioPipeline] = None

        # Send chunk size in bytes
        self._send_chunk_bytes = calc_num_bytes(
            SEND_CHUNK_MS, pipeline_cfg.model_sample_rate
        )

        # Flow control: pauses sending when bot is not speaking
        self._bot_speaking       = False
        self._bot_speaking_lock  = asyncio.Lock()

        # Interruption grace period
        self._interruption_grace_until: float = 0.0

        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._send_task:    Optional[asyncio.Task] = None

        # Stop signal
        self._stop_event = asyncio.Event()

        # Consecutive send error counter
        self._send_error_count = 0

        logger.info(
            "AudioStream created: remote=%s",
            ws.remote_address
        )

    # -----------------------------------------------------------------------
    # Start / Stop
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """Start receive and send loops."""
        self._receive_task = asyncio.ensure_future(self._receive_loop())
        self._send_task    = asyncio.ensure_future(self._send_loop())
        logger.debug("AudioStream tasks started")

    async def stop(self, reason: str = "normal") -> None:
        """Gracefully stop this audio stream."""
        if self._state in (AudioStreamState.CLOSING, AudioStreamState.CLOSED):
            return

        logger.info(
            "AudioStream stopping [%s]: %s",
            self.call_uuid[:8] if self.call_uuid else "?",
            reason,
        )

        await self._set_state(AudioStreamState.CLOSING)
        self._stop_event.set()

        # Cancel tasks
        for task in [self._receive_task, self._send_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Close WebSocket
        try:
            await self._ws.close()
        except Exception:
            pass

        await self._set_state(AudioStreamState.CLOSED)

        # Reset pipeline
        if self._pipeline:
            self._pipeline.reset()

        # Fire disconnected callback
        if self._on_disconnected:
            try:
                await self._on_disconnected(self)
            except Exception as e:
                logger.error("on_disconnected callback error: %s", e)

        logger.info(
            "AudioStream closed [%s]: "
            "rx=%.1fKB tx=%.1fKB utterances=%d interruptions=%d",
            self.call_uuid[:8] if self.call_uuid else "?",
            self.stats.bytes_received / 1024,
            self.stats.bytes_sent / 1024,
            self.stats.utterances,
            self.stats.interruptions,
        )

    # -----------------------------------------------------------------------
    # Receive loop
    # -----------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """
        Continuously receive messages from mod_audio_stream WebSocket.

        First message: JSON metadata (text frame)
        Subsequent:    Raw binary PCM (binary frames)
        """
        logger.debug("AudioStream receive loop started")

        try:
            async for message in self._ws:
                if self._stop_event.is_set():
                    break

                if isinstance(message, str):
                    # Text frame: first message is JSON metadata
                    await self._handle_text_frame(message)

                elif isinstance(message, bytes):
                    # Binary frame: raw PCM audio
                    await self._handle_binary_frame(message)

                else:
                    logger.warning(
                        "Unknown message type: %s", type(message)
                    )

        except ConnectionClosedOK:
            logger.info(
                "AudioStream connection closed normally [%s]",
                self.call_uuid[:8] if self.call_uuid else "?"
            )
        except ConnectionClosedError as e:
            logger.warning(
                "AudioStream connection closed with error [%s]: %s",
                self.call_uuid[:8] if self.call_uuid else "?",
                e,
            )
        except ConnectionClosed as e:
            logger.info(
                "AudioStream connection closed [%s]: %s",
                self.call_uuid[:8] if self.call_uuid else "?",
                e,
            )
        except asyncio.CancelledError:
            logger.debug("AudioStream receive loop cancelled")
            raise
        except Exception as e:
            logger.error(
                "AudioStream receive loop error [%s]: %s",
                self.call_uuid[:8] if self.call_uuid else "?",
                e, exc_info=True,
            )
        finally:
            if not self._stop_event.is_set():
                await self.stop("receive_loop_ended")

    async def _handle_text_frame(self, text: str) -> None:
        """
        Handle the initial JSON metadata frame from mod_audio_stream.
        Subsequent text frames are logged and ignored.
        """
        if self._state == AudioStreamState.CONNECTING:
            # First message — parse metadata
            try:
                self.metadata = CallMetadata.from_json(text)
                self.call_uuid = self.metadata.call_uuid

                logger.info(
                    "AudioStream metadata received: %s", self.metadata
                )

                # Update pipeline config with actual sample rate from FS
                self._pipeline_cfg.fs_sample_rate = self.metadata.sample_rate

                # Create audio pipeline
                self._pipeline = AudioPipeline(
                    cfg=self._pipeline_cfg,
                    on_utterance=self._handle_utterance,
                    on_speech_start=self._handle_speech_start,
                    on_speech_end=self._handle_speech_end,
                    on_interruption=self._handle_interruption_detected,
                )

                await self._set_state(AudioStreamState.READY)

                # Notify caller
                if self._on_connected:
                    try:
                        await self._on_connected(self)
                    except Exception as e:
                        logger.error("on_connected callback error: %s", e)

            except ValueError as e:
                logger.error(
                    "Failed to parse call metadata: %s\nRaw: %s",
                    e, text[:200]
                )
                await self.stop("invalid_metadata")
        else:
            # Subsequent text frames (rare)
            logger.debug(
                "AudioStream text frame (state=%s): %s",
                self._state.name, text[:100]
            )

    async def _handle_binary_frame(self, data: bytes) -> None:
        """
        Handle a raw binary PCM frame from mod_audio_stream.
        Feeds the frame into the audio pipeline.
        """
        if self._state != AudioStreamState.READY:
            return

        if not self._pipeline:
            return

        # Update stats
        self.stats.bytes_received += len(data)
        self.stats.frames_received += 1
        self.stats.last_rx_at = time.time()

        # Feed into pipeline (VAD + utterance detection)
        # This is synchronous but very fast (just buffering + VAD)
        self._pipeline.feed_frame(data)

    # -----------------------------------------------------------------------
    # Audio pipeline callbacks (called from pipeline internals)
    # -----------------------------------------------------------------------

    def _handle_utterance(self, segment: SpeechSegment) -> None:
        """
        Called when the audio pipeline detects a complete utterance.
        Schedules the async callback in the event loop.
        """
        self.stats.utterances += 1
        logger.info(
            "Utterance detected [%s]: %s",
            self.call_uuid[:8],
            segment,
        )
        if self._on_utterance:
            asyncio.ensure_future(
                self._safe_call_utterance(segment)
            )

    async def _safe_call_utterance(self, segment: SpeechSegment) -> None:
        try:
            await self._on_utterance(self.call_uuid, segment)
        except Exception as e:
            logger.error(
                "on_utterance callback error [%s]: %s",
                self.call_uuid[:8], e, exc_info=True
            )

    def _handle_speech_start(self) -> None:
        """Called when VAD detects start of user speech."""
        logger.debug(
            "Speech start [%s] energy=%.1fdBFS",
            self.call_uuid[:8] if self.call_uuid else "?",
            self._pipeline.current_energy_dbfs if self._pipeline else 0.0,
        )

    def _handle_speech_end(self) -> None:
        """Called when VAD detects end of user speech."""
        logger.debug(
            "Speech end [%s]",
            self.call_uuid[:8] if self.call_uuid else "?",
        )

    def _handle_interruption_detected(self) -> None:
        """
        Called by pipeline when user speech detected during bot playback.
        Schedules the interruption handling in the event loop.
        """
        now = time.monotonic()

        # Respect grace period (don't interrupt immediately after bot starts)
        if now < self._interruption_grace_until:
            logger.debug(
                "Interruption suppressed (grace period): %s",
                self.call_uuid[:8],
            )
            return

        logger.info(
            "Interruption triggered [%s]",
            self.call_uuid[:8] if self.call_uuid else "?",
        )
        asyncio.ensure_future(self._handle_interruption())

    async def _handle_interruption(self) -> None:
        """Handle user interruption: clear playback queue and notify."""
        if self._state != AudioStreamState.READY:
            return

        async with self._state_lock:
            if self._state != AudioStreamState.READY:
                return
            self._state = AudioStreamState.INTERRUPTING

        self.stats.interruptions += 1

        # Step 1: Clear our local playback queue immediately
        dropped = self._playback_queue.clear()
        logger.info(
            "Interruption: cleared %d queued chunks [%s]",
            dropped, self.call_uuid[:8],
        )

        # Step 2: Notify bot is no longer speaking
        await self._set_bot_speaking_internal(False)

        # Step 3: Fire callback (session.py will send uuid_break to FS)
        if self._on_interruption:
            try:
                await self._on_interruption(self.call_uuid)
            except Exception as e:
                logger.error(
                    "on_interruption callback error [%s]: %s",
                    self.call_uuid[:8], e,
                )

        # Back to READY
        async with self._state_lock:
            if self._state == AudioStreamState.INTERRUPTING:
                self._state = AudioStreamState.READY

    # -----------------------------------------------------------------------
    # Send loop
    # -----------------------------------------------------------------------

    async def _send_loop(self) -> None:
        """
        Continuously drain the playback queue and send PCM to FreeSWITCH.

        Timing:
          - FreeSWITCH expects PCM at real-time rate
          - We send SEND_CHUNK_MS chunks at a time
          - The event loop naturally paces us (no explicit sleep needed
            because queue.get has a timeout)
          - If queue is empty and bot is speaking → send silence (underrun)
          - If bot is not speaking → idle (don't send anything)
        """
        logger.debug("AudioStream send loop started")

        send_interval = SEND_CHUNK_MS / 1000.0  # seconds
        next_send_at  = time.monotonic()

        try:
            while not self._stop_event.is_set():

                # Wait until next scheduled send time
                now  = time.monotonic()
                wait = next_send_at - now
                if wait > 0:
                    await asyncio.sleep(wait)

                next_send_at = time.monotonic() + send_interval

                # Only send if bot is supposed to be speaking
                if not self._bot_speaking:
                    continue

                # Dequeue next chunk
                chunk = await self._playback_queue.get(timeout=send_interval)

                if chunk is None:
                    # Queue empty — underrun
                    # Check if we're supposed to still be sending
                    if self._bot_speaking:
                        self.stats.playback_underruns += 1
                        # Send silence to keep the stream alive
                        chunk = generate_silence(
                            SEND_CHUNK_MS,
                            self._pipeline_cfg.model_sample_rate,
                        )
                        logger.debug(
                            "Playback underrun [%s] — sending silence",
                            self.call_uuid[:8],
                        )
                    else:
                        continue

                # Send to FreeSWITCH
                await self._send_pcm_chunk(chunk)

        except asyncio.CancelledError:
            logger.debug("AudioStream send loop cancelled")
            raise
        except Exception as e:
            logger.error(
                "AudioStream send loop error [%s]: %s",
                self.call_uuid[:8] if self.call_uuid else "?",
                e, exc_info=True,
            )
        finally:
            logger.debug("AudioStream send loop ended")

    async def _send_pcm_chunk(self, pcm_bytes: bytes) -> None:
        """
        Send a single raw binary PCM chunk to FreeSWITCH.
        Handles send errors with retry logic.
        """
        try:
            await self._ws.send(pcm_bytes)
            self.stats.bytes_sent   += len(pcm_bytes)
            self.stats.frames_sent  += 1
            self.stats.last_tx_at    = time.time()
            self._send_error_count   = 0

        except ConnectionClosed:
            logger.warning(
                "Cannot send PCM: connection closed [%s]",
                self.call_uuid[:8] if self.call_uuid else "?"
            )
            self._send_error_count += 1
            if self._send_error_count >= MAX_SEND_ERRORS:
                await self.stop("too_many_send_errors")

        except Exception as e:
            self.stats.send_errors  += 1
            self._send_error_count  += 1
            logger.error(
                "PCM send error [%s]: %s (consecutive=%d)",
                self.call_uuid[:8] if self.call_uuid else "?",
                e,
                self._send_error_count,
            )
            if self._send_error_count >= MAX_SEND_ERRORS:
                await self.stop("too_many_send_errors")

    # -----------------------------------------------------------------------
    # Public API: playback control
    # -----------------------------------------------------------------------

    async def play_audio(
        self,
        pcm_bytes: bytes,
        chunk_ms:  int = SEND_CHUNK_MS,
    ) -> int:
        """
        Queue PCM audio for playback to FreeSWITCH.

        Splits audio into SEND_CHUNK_MS chunks and enqueues them.
        Call begin_speaking() before this and end_speaking() after
        all audio has been queued.

        Args:
            pcm_bytes: Raw 16-bit signed PCM bytes
            chunk_ms:  Size of each chunk to enqueue (ms)

        Returns:
            Number of bytes successfully queued
        """
        if self._state not in (AudioStreamState.READY, AudioStreamState.INTERRUPTING):
            logger.warning(
                "play_audio called in invalid state: %s [%s]",
                self._state.name,
                self.call_uuid[:8],
            )
            return 0

        chunk_bytes = calc_num_bytes(
            chunk_ms, self._pipeline_cfg.model_sample_rate
        )
        queued = 0
        offset = 0

        while offset < len(pcm_bytes):
            chunk = pcm_bytes[offset:offset + chunk_bytes]

            # Pad last chunk
            if len(chunk) < chunk_bytes:
                chunk = chunk + b"\x00" * (chunk_bytes - len(chunk))

            ok = await self._playback_queue.put(chunk)
            if ok:
                queued += len(chunk)
            offset += chunk_bytes

        logger.debug(
            "play_audio: queued %d bytes (%.0fms) [%s]",
            queued,
            calc_duration_ms(queued, self._pipeline_cfg.model_sample_rate),
            self.call_uuid[:8],
        )
        return queued

    async def begin_speaking(self, grace_period_ms: float = 500.0) -> None:
        """
        Signal that the bot is about to start speaking.
        Enables the send loop and activates interruption detection.

        Args:
            grace_period_ms: How long to suppress interruptions after
                             bot starts speaking (prevents echo feedback)
        """
        self._interruption_grace_until = (
            time.monotonic() + grace_period_ms / 1000.0
        )
        await self._set_bot_speaking_internal(True)
        logger.info(
            "Bot speaking started [%s] grace=%.0fms",
            self.call_uuid[:8], grace_period_ms,
        )

    async def end_speaking(self) -> None:
        """
        Signal that the bot has finished speaking.
        The send loop will finish draining the queue then go idle.
        """
        # Wait for queue to drain before marking as not speaking
        asyncio.ensure_future(self._wait_queue_then_idle())

    async def _wait_queue_then_idle(self) -> None:
        """Wait for playback queue to empty, then set bot to not speaking."""
        timeout  = time.monotonic() + 30.0  # max 30s
        interval = 0.05  # check every 50ms

        while time.monotonic() < timeout:
            if self._playback_queue.is_empty:
                break
            await asyncio.sleep(interval)

        await self._set_bot_speaking_internal(False)
        logger.info(
            "Bot speaking ended [%s] queue_empty=%s",
            self.call_uuid[:8],
            self._playback_queue.is_empty,
        )

    async def interrupt(self) -> None:
        """
        Immediately interrupt bot playback.
        Clears the playback queue and sets bot to not speaking.
        Called externally by session.py when ESL uuid_break is sent.
        """
        logger.info(
            "AudioStream.interrupt() called [%s]",
            self.call_uuid[:8] if self.call_uuid else "?",
        )
        dropped = self._playback_queue.clear()
        await self._set_bot_speaking_internal(False)
        logger.info(
            "Interrupt complete: dropped %d chunks [%s]",
            dropped, self.call_uuid[:8],
        )

    async def _set_bot_speaking_internal(self, speaking: bool) -> None:
        """Internal: set bot speaking flag and notify pipeline."""
        async with self._bot_speaking_lock:
            if self._bot_speaking == speaking:
                return
            self._bot_speaking = speaking

        # Notify pipeline so it knows when to look for interruptions
        if self._pipeline:
            self._pipeline.set_bot_speaking(speaking)

    # -----------------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------------

    async def _set_state(self, new_state: AudioStreamState) -> None:
        async with self._state_lock:
            old = self._state
            self._state = new_state
            if old != new_state:
                logger.debug(
                    "AudioStream state: %s → %s [%s]",
                    old.name, new_state.name,
                    self.call_uuid[:8] if self.call_uuid else "?",
                )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def state(self) -> AudioStreamState:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state == AudioStreamState.READY

    @property
    def is_bot_speaking(self) -> bool:
        return self._bot_speaking

    @property
    def is_user_speaking(self) -> bool:
        return (
            self._pipeline.is_user_speaking
            if self._pipeline else False
        )

    @property
    def playback_queue_stats(self) -> Dict:
        return self._playback_queue.stats

    @property
    def remote_address(self) -> str:
        try:
            return str(self._ws.remote_address)
        except Exception:
            return "unknown"

    def get_full_stats(self) -> Dict:
        return {
            "call_uuid":     self.call_uuid,
            "state":         self._state.name,
            "remote":        self.remote_address,
            "bot_speaking":  self._bot_speaking,
            "user_speaking": self.is_user_speaking,
            "stats":         self.stats.to_dict(),
            "playback":      self._playback_queue.stats,
            "pipeline":      (
                self._pipeline.stats if self._pipeline else {}
            ),
            "metadata":      (
                {
                    "caller_id_num":  self.metadata.caller_id_num,
                    "caller_id_name": self.metadata.caller_id_name,
                    "destination":    self.metadata.destination,
                    "sample_rate":    self.metadata.sample_rate,
                    "direction":      self.metadata.direction,
                }
                if self.metadata else {}
            ),
        }


# ---------------------------------------------------------------------------
# AudioServer  (WebSocket server managing all connections)
# ---------------------------------------------------------------------------

class AudioServer:
    """
    WebSocket server that accepts connections from mod_audio_stream.

    One instance serves all concurrent calls.
    Manages the pool of AudioStream objects (one per call).

    Usage:
        server = AudioServer(cfg, on_utterance=handler)
        await server.start()
        # ... runs forever ...
        await server.stop()
    """

    def __init__(
        self,
        host:            str   = "0.0.0.0",
        port:            int   = 8765,
        pipeline_cfg:    Optional[AudioPipelineConfig] = None,
        max_connections: int   = 10,
        on_utterance:    Optional[Callable[[str, SpeechSegment], Coroutine]] = None,
        on_connected:    Optional[Callable[["AudioStream"], Coroutine]]      = None,
        on_disconnected: Optional[Callable[["AudioStream"], Coroutine]]      = None,
        on_interruption: Optional[Callable[[str], Coroutine]]                = None,
        ping_interval:   float = 20.0,
        ping_timeout:    float = 10.0,
    ):
        self.host            = host
        self.port            = port
        self.pipeline_cfg    = pipeline_cfg or AudioPipelineConfig()
        self.max_connections = max_connections
        self.ping_interval   = ping_interval
        self.ping_timeout    = ping_timeout

        self._on_utterance    = on_utterance
        self._on_connected    = on_connected
        self._on_disconnected = on_disconnected
        self._on_interruption = on_interruption

        # Active streams: call_uuid → AudioStream
        self._streams:   Dict[str, AudioStream] = {}
        self._streams_lock = asyncio.Lock()

        # Also index by WebSocket object (before UUID is known)
        self._pending:   Dict[int, AudioStream] = {}  # id(ws) → stream

        self._server:    Optional[websockets.WebSocketServer] = None
        self._stop_event = asyncio.Event()

        self._total_connections  = 0
        self._rejected_connections = 0

        logger.info(
            "AudioServer created: %s:%d max_connections=%d",
            host, port, max_connections,
        )

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            max_size=1024 * 1024,       # 1MB max message
            compression=None,           # No compression for PCM audio
            close_timeout=5,
        )
        logger.info(
            "AudioServer listening on ws://%s:%d",
            self.host, self.port,
        )

    async def stop(self) -> None:
        """Gracefully stop the server and all active streams."""
        logger.info("AudioServer stopping...")
        self._stop_event.set()

        # Close all active streams
        async with self._streams_lock:
            streams = list(self._streams.values())

        for stream in streams:
            try:
                await stream.stop("server_shutdown")
            except Exception as e:
                logger.error("Error stopping stream: %s", e)

        # Stop accepting new connections
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info("AudioServer stopped")

    async def _handle_connection(
        self, ws: WebSocketServerProtocol
    ) -> None:
        """
        Handle a new WebSocket connection from mod_audio_stream.
        One coroutine per connection, runs for the lifetime of the call.
        """
        remote = ws.remote_address
        self._total_connections += 1

        logger.info(
            "New mod_audio_stream connection from %s (total=%d)",
            remote, self._total_connections,
        )

        # Check connection limit
        async with self._streams_lock:
            active = len(self._streams) + len(self._pending)

        if active >= self.max_connections:
            self._rejected_connections += 1
            logger.warning(
                "Connection rejected: at limit (%d/%d) from %s",
                active, self.max_connections, remote,
            )
            await ws.close(1013, "Server at capacity")
            return

        # Create audio stream for this connection
        stream = AudioStream(
            ws=ws,
            pipeline_cfg=self._pipeline_cfg_copy(),
            on_utterance=self._on_utterance,
            on_connected=self._handle_stream_connected,
            on_disconnected=self._handle_stream_disconnected,
            on_interruption=self._on_interruption,
        )

        # Register in pending (before UUID is known)
        async with self._streams_lock:
            self._pending[id(ws)] = stream

        try:
            # Start the stream (spawns receive + send tasks)
            await stream.start()

            # Wait until the stream closes (receive_loop ends)
            await self._wait_for_stream(stream)

        except Exception as e:
            logger.error(
                "AudioStream error from %s: %s",
                remote, e, exc_info=True,
            )
        finally:
            # Clean up
            async with self._streams_lock:
                self._pending.pop(id(ws), None)
                if stream.call_uuid:
                    self._streams.pop(stream.call_uuid, None)

            if stream.state != AudioStreamState.CLOSED:
                await stream.stop("connection_handler_exit")

            logger.info(
                "Connection handler done: %s",
                stream.call_uuid[:8] if stream.call_uuid else remote,
            )

    async def _wait_for_stream(self, stream: AudioStream) -> None:
        """Wait until the stream's receive task completes."""
        # Poll until the stream is closed or stop_event fires
        while (
            not self._stop_event.is_set()
            and stream.state not in (
                AudioStreamState.CLOSED,
                AudioStreamState.CLOSING,
            )
        ):
            await asyncio.sleep(0.1)

    async def _handle_stream_connected(self, stream: AudioStream) -> None:
        """
        Called when stream transitions to READY (metadata received).
        Move from pending to active streams dict.
        """
        async with self._streams_lock:
            self._pending.pop(id(stream._ws), None)
            if stream.call_uuid:
                self._streams[stream.call_uuid] = stream

        logger.info(
            "AudioStream READY: %s (active=%d)",
            stream.call_uuid[:8],
            len(self._streams),
        )

        if self._on_connected:
            try:
                await self._on_connected(stream)
            except Exception as e:
                logger.error("on_connected server callback error: %s", e)

    async def _handle_stream_disconnected(self, stream: AudioStream) -> None:
        """Called when a stream closes."""
        async with self._streams_lock:
            self._streams.pop(stream.call_uuid, None)
            self._pending.pop(id(stream._ws), None)

        logger.info(
            "AudioStream disconnected: %s (active=%d)",
            stream.call_uuid[:8] if stream.call_uuid else "?",
            len(self._streams),
        )

        if self._on_disconnected:
            try:
                await self._on_disconnected(stream)
            except Exception as e:
                logger.error("on_disconnected server callback error: %s", e)

    def _pipeline_cfg_copy(self) -> AudioPipelineConfig:
        """Create a fresh AudioPipelineConfig copy for each connection."""
        import dataclasses
        return dataclasses.replace(self.pipeline_cfg)

    # -----------------------------------------------------------------------
    # Stream access (used by session.py)
    # -----------------------------------------------------------------------

    def get_stream(self, call_uuid: str) -> Optional[AudioStream]:
        """Get an active AudioStream by call UUID."""
        return self._streams.get(call_uuid)

    async def play_audio(
        self,
        call_uuid: str,
        pcm_bytes: bytes,
    ) -> bool:
        """
        Queue PCM audio for playback on a specific call.

        Args:
            call_uuid: Target call UUID
            pcm_bytes: Raw 16-bit PCM bytes to play

        Returns:
            True if queued successfully
        """
        stream = self.get_stream(call_uuid)
        if stream is None:
            logger.warning(
                "play_audio: no stream for UUID %s", call_uuid[:8]
            )
            return False
        queued = await stream.play_audio(pcm_bytes)
        return queued > 0

    async def begin_speaking(
        self,
        call_uuid: str,
        grace_period_ms: float = 500.0,
    ) -> bool:
        """Signal bot is starting to speak on a call."""
        stream = self.get_stream(call_uuid)
        if stream is None:
            return False
        await stream.begin_speaking(grace_period_ms)
        return True

    async def end_speaking(self, call_uuid: str) -> bool:
        """Signal bot has finished speaking on a call."""
        stream = self.get_stream(call_uuid)
        if stream is None:
            return False
        await stream.end_speaking()
        return True

    async def interrupt(self, call_uuid: str) -> bool:
        """
        Immediately interrupt playback on a call.
        Called when ESL uuid_break is sent.
        """
        stream = self.get_stream(call_uuid)
        if stream is None:
            return False
        await stream.interrupt()
        return True

    # -----------------------------------------------------------------------
    # Server stats
    # -----------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        return len(self._streams)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def get_all_stats(self) -> Dict:
        """Get stats for all active streams."""
        return {
            "server": {
                "host":                  self.host,
                "port":                  self.port,
                "active_connections":    len(self._streams),
                "pending_connections":   len(self._pending),
                "total_connections":     self._total_connections,
                "rejected_connections":  self._rejected_connections,
                "max_connections":       self.max_connections,
            },
            "streams": {
                uuid: stream.get_full_stats()
                for uuid, stream in self._streams.items()
            },
        }

    def get_stream_stats(self, call_uuid: str) -> Optional[Dict]:
        stream = self.get_stream(call_uuid)
        return stream.get_full_stats() if stream else None


# ---------------------------------------------------------------------------
# Factory from AppConfig
# ---------------------------------------------------------------------------

def create_audio_server(cfg: "Any") -> AudioServer:
    """
    Create an AudioServer from AppConfig.

    Args:
        cfg: AppConfig instance (from config.py)

    Returns:
        Configured AudioServer (not yet started)
    """
    audio_cfg = cfg.audio
    vad_cfg   = cfg.vad

    pipeline_cfg = AudioPipelineConfig(
        fs_sample_rate=int(audio_cfg.fs_sample_rate),
        model_sample_rate=int(audio_cfg.model_sample_rate),
        fs_chunk_ms=audio_cfg.fs_chunk_ms,
        model_chunk_ms=audio_cfg.model_chunk_ms,
        enable_resampling=audio_cfg.enable_resampling,
        enable_normalization=audio_cfg.enable_normalization,
        normalization_target_dbfs=audio_cfg.normalization_target_dbfs,
        use_webrtc_vad=vad_cfg.use_webrtc_vad,
        vad_aggressiveness=vad_cfg.webrtc_vad_aggressiveness,
        vad_frame_ms=vad_cfg.webrtc_frame_ms,
        vad_smoothing_frames=vad_cfg.smoothing_frames,
        vad_smoothing_threshold=vad_cfg.smoothing_threshold,
        energy_threshold_dbfs=vad_cfg.energy_threshold_dbfs,
        min_speech_ms=vad_cfg.min_speech_duration_ms,
        silence_ms=vad_cfg.silence_duration_ms,
        interruption_speech_ms=vad_cfg.interruption_min_speech_ms,
        interruption_silence_ms=vad_cfg.interruption_silence_ms,
    )

    return AudioServer(
        host=cfg.server.host,
        port=cfg.server.audio_ws_port,
        pipeline_cfg=pipeline_cfg,
        max_connections=cfg.server.max_connections,
        ping_interval=cfg.server.ws_ping_interval,
        ping_timeout=cfg.server.ws_ping_timeout,
    )
