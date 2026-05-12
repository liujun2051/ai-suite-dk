# 文件5: `ai_client.py`

```python
"""
ai_client.py - MiniCPM-o 4.5 full-duplex WebSocket client

Handles:
  - WebSocket connection to MiniCPM-o 4.5 official API
  - Full-duplex audio streaming (send user audio, receive bot audio simultaneously)
  - Session management (one WS session per call)
  - Sending raw PCM audio chunks to the model
  - Receiving raw PCM audio chunks from the model
  - Interrupt signal handling
  - Automatic reconnection on error
  - Token/audio usage tracking
  - Graceful session teardown

MiniCPM-o 4.5 WebSocket protocol:
  Outbound (us → model):
    - {"type": "session.create", "session": {...}}         # init session
    - {"type": "input_audio_buffer.append", "audio": <b64 PCM>}  # stream audio
    - {"type": "input_audio_buffer.commit"}                # user turn complete
    - {"type": "response.cancel"}                          # interrupt
    - {"type": "session.update", "session": {...}}         # update config

  Inbound (model → us):
    - {"type": "session.created", ...}                     # session ready
    - {"type": "response.audio.delta", "delta": <b64 PCM>}# audio chunk
    - {"type": "response.audio.done"}                      # audio complete
    - {"type": "response.done"}                            # full response done
    - {"type": "error", "error": {...}}                    # error
    - {"type": "input_audio_buffer.speech_started"}        # model detected speech
    - {"type": "input_audio_buffer.speech_stopped"}        # model detected silence

Reference: https://www.minimaxi.com/document/guides/voice-call

 
```

---

`ai_client.py` 完成 ✅ 约 **720行**

包含：

| 组件 | 说明 |
|------|------|
| `SessionConfig` | MiniCPM-o会话配置，序列化为API格式 |
| `AudioResponse` | 单个音频响应chunk，携带PCM bytes |
| `UsageStats` | Token/音频用量追踪 |
| `LatencyTracker` | 首包延迟追踪，p95统计 |
| `_msg_*` | 所有协议消息构建函数 |
| `MiniCPMClient._receiver_loop` | 接收所有模型消息，dispatch到handler |
| `_message_handlers` | 完整消息类型映射表（18种消息类型） |
| `_audio_sender_loop` | 异步drain队列，批量发送音频 |
| `send_audio()` | 非阻塞入队，供session.py调用 |
| `interrupt()` | 3步打断：clear buffer → cancel → wait |
| `commit_audio()` | 手动触发模型响应（VAD关闭时） |
| `send_text_message()` | 中途注入文本消息 |
| `_reconnect()` | 指数退避自动重连 |
| `create_minicpm_client` | AppConfig工厂函数 |

---

**下一个：`session.py`** — 这是整个系统的大脑，把ESL + AudioStream + MiniCPMClient全部串联起来，管理完整的通话状态机。准备好就说继续！

"""

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, AsyncGenerator, Callable,
    Coroutine, Dict, List, Optional
)

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
    WebSocketException,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How long to wait for session.created after connecting (seconds)
SESSION_INIT_TIMEOUT    = 15.0
# How long to wait for any response to start (seconds)
RESPONSE_START_TIMEOUT  = 10.0
# Interval between audio chunks sent to model (seconds)
AUDIO_SEND_INTERVAL     = 0.1    # 100ms
# Max audio chunk size to send at once (bytes) — prevents WS frame overload
MAX_AUDIO_CHUNK_BYTES   = 32_000
# How long model can be silent before we consider response done (seconds)
MODEL_SILENCE_TIMEOUT   = 2.0
# Max reconnect attempts per session
MAX_RECONNECT_ATTEMPTS  = 5
# Base reconnect delay (seconds)
RECONNECT_BASE_DELAY    = 1.0


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class ClientState(Enum):
    IDLE         = auto()   # Not connected
    CONNECTING   = auto()   # WS connecting
    READY        = auto()   # Session created, ready to stream
    STREAMING    = auto()   # Actively sending user audio
    RESPONDING   = auto()   # Model is generating response
    INTERRUPTING = auto()   # Interrupt in progress
    CLOSING      = auto()   # Graceful shutdown
    CLOSED       = auto()   # Fully closed
    ERROR        = auto()   # Unrecoverable error


class ResponseState(Enum):
    IDLE       = auto()   # No active response
    GENERATING = auto()   # Model is generating audio
    DONE       = auto()   # Response complete
    CANCELLED  = auto()   # Response was interrupted


@dataclass
class SessionConfig:
    """
    Configuration sent to MiniCPM-o when creating/updating a session.
    Maps to the official API session object.
    """
    model:           str   = "MiniCPMo-4.5"
    system_prompt:   str   = "你是一个专业的语音助手，回答简洁自然。"
    voice_id:        str   = "default"
    language:        str   = "zh-CN"
    temperature:     float = 0.7
    input_audio_format:  str = "pcm16"     # pcm16 | g711_ulaw | g711_alaw
    output_audio_format: str = "pcm16"     # pcm16
    input_sample_rate:   int = 16000
    output_sample_rate:  int = 16000
    turn_detection: Dict[str, Any] = field(default_factory=lambda: {
        "type":                    "server_vad",
        "threshold":               0.5,
        "prefix_padding_ms":       300,
        "silence_duration_ms":     500,
        "create_response":         True,
    })

    def to_api_dict(self) -> Dict[str, Any]:
        """Serialize to MiniCPM-o API session format."""
        return {
            "model": self.model,
            "instructions": self.system_prompt,
            "voice": self.voice_id,
            "input_audio_format":  self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "input_audio_transcription": {
                "model": "whisper-1",
                "language": self.language,
            },
            "turn_detection": self.turn_detection,
            "temperature": self.temperature,
            "modalities": ["audio", "text"],
        }


@dataclass
class AudioResponse:
    """
    A single audio response chunk from the model.
    Carries raw decoded PCM bytes.
    """
    pcm_bytes:   bytes
    sample_rate: int
    is_final:    bool    = False    # True on last chunk of a response
    timestamp:   float   = field(default_factory=time.monotonic)
    sequence:    int     = 0

    @property
    def duration_ms(self) -> float:
        from audio_utils import calc_duration_ms
        return calc_duration_ms(len(self.pcm_bytes), self.sample_rate)


@dataclass
class UsageStats:
    """Token and audio usage tracking."""
    input_audio_ms:   float = 0.0
    output_audio_ms:  float = 0.0
    input_tokens:     int   = 0
    output_tokens:    int   = 0
    total_tokens:     int   = 0
    responses:        int   = 0
    interruptions:    int   = 0
    reconnections:    int   = 0
    errors:           int   = 0

    def to_dict(self) -> Dict:
        return {
            "input_audio_ms":  round(self.input_audio_ms, 1),
            "output_audio_ms": round(self.output_audio_ms, 1),
            "input_tokens":    self.input_tokens,
            "output_tokens":   self.output_tokens,
            "total_tokens":    self.total_tokens,
            "responses":       self.responses,
            "interruptions":   self.interruptions,
            "reconnections":   self.reconnections,
            "errors":          self.errors,
        }


@dataclass
class LatencyTracker:
    """Tracks time-to-first-audio latency per response."""
    _response_start:  Optional[float] = field(default=None, repr=False)
    _first_audio_at:  Optional[float] = field(default=None, repr=False)
    samples:          List[float]     = field(default_factory=list)

    def response_started(self) -> None:
        self._response_start = time.monotonic()
        self._first_audio_at = None

    def first_audio_received(self) -> Optional[float]:
        """Returns latency in ms, or None if already recorded."""
        if self._response_start is None:
            return None
        if self._first_audio_at is not None:
            return None   # already recorded for this response
        self._first_audio_at = time.monotonic()
        latency_ms = (self._first_audio_at - self._response_start) * 1000
        self.samples.append(latency_ms)
        return latency_ms

    @property
    def p95_ms(self) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def mean_ms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)


# ---------------------------------------------------------------------------
# MiniCPM-o message builders
# ---------------------------------------------------------------------------

def _msg_session_create(cfg: SessionConfig) -> str:
    return json.dumps({
        "type":    "session.update",
        "session": cfg.to_api_dict(),
    })


def _msg_audio_append(pcm_bytes: bytes) -> str:
    """Encode PCM bytes as base64 and build input_audio_buffer.append message."""
    audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
    return json.dumps({
        "type":  "input_audio_buffer.append",
        "audio": audio_b64,
    })


def _msg_audio_commit() -> str:
    """Signal end of user audio turn (server_vad disabled mode)."""
    return json.dumps({"type": "input_audio_buffer.commit"})


def _msg_response_cancel() -> str:
    """Cancel current model response (interrupt)."""
    return json.dumps({"type": "response.cancel"})


def _msg_response_create() -> str:
    """Manually trigger response generation."""
    return json.dumps({"type": "response.create"})


def _msg_clear_audio_buffer() -> str:
    """Clear the server-side input audio buffer."""
    return json.dumps({"type": "input_audio_buffer.clear"})


def _msg_session_update(updates: Dict[str, Any]) -> str:
    return json.dumps({
        "type":    "session.update",
        "session": updates,
    })


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class MiniCPMClient:
    """
    Full-duplex WebSocket client for MiniCPM-o 4.5.

    One instance per call session.
    Manages a single WebSocket connection to the MiniCPM-o API.

    Audio flow:
      User audio → send_audio() → WebSocket → MiniCPM-o
      MiniCPM-o  → WebSocket   → on_audio_response callback

    Usage:
        client = MiniCPMClient(api_key="...", config=session_cfg)
        await client.connect()

        # Stream user audio
        await client.send_audio(pcm_bytes)

        # Interrupt model response
        await client.interrupt()

        # Disconnect
        await client.disconnect()
    """

    def __init__(
        self,
        api_key:         str,
        api_base_url:    str   = "wss://api.minimaxi.chat/v1/realtime",
        config:          Optional[SessionConfig] = None,
        on_audio_chunk:  Optional[Callable[[AudioResponse], Coroutine]] = None,
        on_response_done:Optional[Callable[[Dict], Coroutine]]          = None,
        on_speech_start: Optional[Callable[[], Coroutine]]              = None,
        on_speech_stop:  Optional[Callable[[], Coroutine]]              = None,
        on_error:        Optional[Callable[[str, Dict], Coroutine]]     = None,
        on_connected:    Optional[Callable[[], Coroutine]]              = None,
        on_disconnected: Optional[Callable[[], Coroutine]]              = None,
        reconnect_enabled: bool  = True,
        reconnect_max:     int   = MAX_RECONNECT_ATTEMPTS,
    ):
        """
        Args:
            api_key:          MiniCPM-o API key
            api_base_url:     WebSocket endpoint URL
            config:           Session configuration (model, voice, etc.)
            on_audio_chunk:   Async callback(AudioResponse) for each audio chunk
                              This is the hot path — called for every PCM chunk
                              the model outputs. Should be fast.
            on_response_done: Async callback(metadata_dict) when response complete
            on_speech_start:  Async callback when model detects user started speaking
            on_speech_stop:   Async callback when model detects user stopped speaking
            on_error:         Async callback(error_code, error_dict) on API error
            on_connected:     Async callback when WebSocket session is ready
            on_disconnected:  Async callback when connection closes
            reconnect_enabled: Auto-reconnect on unexpected disconnect
            reconnect_max:    Max reconnect attempts
        """
        self._api_key          = api_key
        self._api_base_url     = api_base_url.rstrip("/")
        self._config           = config or SessionConfig()
        self._on_audio_chunk   = on_audio_chunk
        self._on_response_done = on_response_done
        self._on_speech_start  = on_speech_start
        self._on_speech_stop   = on_speech_stop
        self._on_error         = on_error
        self._on_connected     = on_connected
        self._on_disconnected  = on_disconnected
        self._reconnect_enabled = reconnect_enabled
        self._reconnect_max     = reconnect_max

        # WebSocket
        self._ws:     Optional[websockets.WebSocketClientProtocol] = None
        self._state   = ClientState.IDLE
        self._session_id: str = ""

        # Response tracking
        self._response_state    = ResponseState.IDLE
        self._response_seq:     int  = 0      # increments per response
        self._audio_chunk_seq:  int  = 0      # increments per audio chunk

        # Audio send queue
        # Filled by send_audio(), drained by _audio_sender_loop()
        self._audio_send_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

        # Stats and latency
        self.usage    = UsageStats()
        self.latency  = LatencyTracker()

        # Tasks
        self._receiver_task: Optional[asyncio.Task] = None
        self._sender_task:   Optional[asyncio.Task] = None
        self._stop_event     = asyncio.Event()

        # Reconnect state
        self._reconnect_count = 0

        # Pending audio bytes (accumulated between commits)
        self._input_audio_ms  = 0.0

        logger.info(
            "MiniCPMClient created: url=%s model=%s",
            self._api_base_url, self._config.model,
        )

    # -----------------------------------------------------------------------
    # Connection lifecycle
    # -----------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect to MiniCPM-o API and initialize session.
        Starts receiver and sender background tasks.
        """
        self._stop_event.clear()
        self._state = ClientState.CONNECTING

        url = self._build_url()
        headers = self._build_headers()

        logger.info("Connecting to MiniCPM-o: %s", url)

        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    url,
                    extra_headers=headers,
                    ping_interval=30,
                    ping_timeout=10,
                    max_size=10 * 1024 * 1024,   # 10MB for audio
                    compression=None,
                ),
                timeout=SESSION_INIT_TIMEOUT,
            )
        except (OSError, asyncio.TimeoutError, WebSocketException) as e:
            self._state = ClientState.ERROR
            logger.error("MiniCPM-o connect failed: %s", e)
            raise ConnectionError(f"MiniCPM-o connect failed: {e}") from e

        logger.info("MiniCPM-o WebSocket connected")

        # Start background tasks
        self._receiver_task = asyncio.ensure_future(self._receiver_loop())
        self._sender_task   = asyncio.ensure_future(self._audio_sender_loop())

        # Send session configuration
        await self._init_session()

    async def _init_session(self) -> None:
        """Send session.update to configure the model session."""
        init_msg = _msg_session_create(self._config)
        await self._send_raw(init_msg)
        logger.debug("Session init sent, waiting for session.created...")

        # Wait for session to be confirmed
        # The receiver_loop will set state to READY when session.created arrives
        try:
            await asyncio.wait_for(
                self._wait_for_state(ClientState.READY),
                timeout=SESSION_INIT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"MiniCPM-o session init timeout after {SESSION_INIT_TIMEOUT}s"
            )

        logger.info(
            "MiniCPM-o session ready: %s",
            self._session_id or "(no session id)"
        )

    async def disconnect(self) -> None:
        """Gracefully disconnect from MiniCPM-o."""
        if self._state in (ClientState.CLOSING, ClientState.CLOSED):
            return

        logger.info("MiniCPMClient disconnecting...")
        self._state = ClientState.CLOSING
        self._stop_event.set()

        # Cancel background tasks
        for task in [self._receiver_task, self._sender_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Close WebSocket
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._state = ClientState.CLOSED
        logger.info(
            "MiniCPMClient disconnected. Usage: %s Latency p95=%.0fms",
            self.usage.to_dict(),
            self.latency.p95_ms,
        )

        if self._on_disconnected:
            try:
                await self._on_disconnected()
            except Exception as e:
                logger.error("on_disconnected callback error: %s", e)

    async def _wait_for_state(self, target: ClientState) -> None:
        """Poll until state reaches target."""
        while self._state != target:
            if self._state in (ClientState.ERROR, ClientState.CLOSED):
                raise ConnectionError(
                    f"Connection failed while waiting for state {target.name}"
                )
            await asyncio.sleep(0.05)

    def _build_url(self) -> str:
        """Build the full WebSocket URL with model parameter."""
        return f"{self._api_base_url}?model={self._config.model}"

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers for WebSocket upgrade."""
        return {
            "Authorization":  f"Bearer {self._api_key}",
            "Content-Type":   "application/json",
            "User-Agent":     "VoiceBot/1.0",
        }

    # -----------------------------------------------------------------------
    # Background: receiver loop
    # -----------------------------------------------------------------------

    async def _receiver_loop(self) -> None:
        """
        Continuously receive and dispatch messages from MiniCPM-o.
        Runs as a background asyncio task.
        This is the most critical loop — all model output comes through here.
        """
        logger.debug("MiniCPM-o receiver loop started")

        try:
            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(
                        self._ws.recv(),
                        timeout=60.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "MiniCPM-o receiver: no message for 60s"
                    )
                    continue

                if isinstance(raw, bytes):
                    # Binary frame: treat as raw PCM (unlikely but handle it)
                    await self._handle_binary_message(raw)
                else:
                    # Text frame: JSON message
                    await self._handle_json_message(raw)

        except ConnectionClosedOK:
            logger.info("MiniCPM-o connection closed normally")
        except ConnectionClosedError as e:
            logger.warning("MiniCPM-o connection closed with error: %s", e)
            await self._handle_disconnect(str(e))
        except ConnectionClosed as e:
            logger.info("MiniCPM-o connection closed: %s", e)
        except asyncio.CancelledError:
            logger.debug("MiniCPM-o receiver loop cancelled")
            raise
        except Exception as e:
            logger.error(
                "MiniCPM-o receiver loop error: %s",
                e, exc_info=True,
            )
            await self._handle_disconnect(str(e))
        finally:
            logger.debug("MiniCPM-o receiver loop ended")

    async def _handle_json_message(self, raw: str) -> None:
        """Parse and dispatch a JSON message from MiniCPM-o."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("MiniCPM-o invalid JSON: %s | raw=%s", e, raw[:200])
            return

        msg_type = msg.get("type", "")
        logger.debug("MiniCPM-o ← %s", msg_type)

        # Dispatch by type
        handler = self._message_handlers.get(msg_type)
        if handler:
            await handler(self, msg)
        else:
            logger.debug("MiniCPM-o unhandled message type: %s", msg_type)

    async def _handle_binary_message(self, data: bytes) -> None:
        """Handle unexpected binary frame (treat as raw PCM if in RESPONDING state)."""
        if self._state == ClientState.RESPONDING:
            logger.debug(
                "MiniCPM-o binary frame: %d bytes (treating as PCM)", len(data)
            )
            await self._emit_audio_chunk(data, is_final=False)

    # -----------------------------------------------------------------------
    # Message handlers (one per message type)
    # -----------------------------------------------------------------------

    async def _on_session_created(self, msg: Dict) -> None:
        """Model confirmed session is ready."""
        self._session_id = (
            msg.get("session", {}).get("id", "")
            or msg.get("session_id", "")
        )
        self._state = ClientState.READY
        logger.info(
            "MiniCPM-o session created: id=%s",
            self._session_id or "(unknown)",
        )
        if self._on_connected:
            try:
                await self._on_connected()
            except Exception as e:
                logger.error("on_connected callback error: %s", e)

    async def _on_session_updated(self, msg: Dict) -> None:
        """Session configuration was updated."""
        logger.debug("MiniCPM-o session updated")
        if self._state == ClientState.CONNECTING:
            # First update response = session is ready
            self._state = ClientState.READY
            if self._on_connected:
                try:
                    await self._on_connected()
                except Exception as e:
                    logger.error("on_connected callback error: %s", e)

    async def _on_response_audio_delta(self, msg: Dict) -> None:
        """
        Model sent an audio chunk.
        This is the hot path — decode base64 PCM and fire callback ASAP.
        """
        if self._state not in (ClientState.RESPONDING, ClientState.READY):
            logger.debug(
                "Ignoring audio delta in state %s", self._state.name
            )
            return

        self._state = ClientState.RESPONDING

        # Decode base64 audio
        audio_b64 = msg.get("delta", "")
        if not audio_b64:
            logger.warning("MiniCPM-o audio delta: empty delta field")
            return

        try:
            pcm_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            logger.error("MiniCPM-o audio decode error: %s", e)
            self.usage.errors += 1
            return

        if not pcm_bytes:
            return

        # Track latency for first chunk
        latency = self.latency.first_audio_received()
        if latency is not None:
            logger.info(
                "MiniCPM-o first audio chunk: %.0fms latency, %d bytes",
                latency, len(pcm_bytes),
            )

        await self._emit_audio_chunk(pcm_bytes, is_final=False)

    async def _on_response_audio_done(self, msg: Dict) -> None:
        """Model finished sending audio for this response."""
        logger.info(
            "MiniCPM-o response.audio.done (seq=%d)", self._response_seq
        )
        self._response_state = ResponseState.DONE

        # Emit final chunk marker
        await self._emit_audio_chunk(b"", is_final=True)

    async def _on_response_done(self, msg: Dict) -> None:
        """
        Full response is complete (audio + any text).
        May include usage statistics.
        """
        self._response_state = ResponseState.DONE
        self.usage.responses += 1

        # Extract usage if present
        response = msg.get("response", {})
        usage    = response.get("usage", {})
        if usage:
            self.usage.input_tokens  += usage.get("input_tokens", 0)
            self.usage.output_tokens += usage.get("output_tokens", 0)
            self.usage.total_tokens  += usage.get("total_tokens", 0)

        logger.info(
            "MiniCPM-o response done #%d | "
            "tokens in=%d out=%d | latency p95=%.0fms",
            self._response_seq,
            self.usage.input_tokens,
            self.usage.output_tokens,
            self.latency.p95_ms,
        )

        # Return to ready state
        if self._state == ClientState.RESPONDING:
            self._state = ClientState.READY

        if self._on_response_done:
            try:
                await self._on_response_done(response)
            except Exception as e:
                logger.error("on_response_done callback error: %s", e)

    async def _on_response_cancelled(self, msg: Dict) -> None:
        """Model confirmed response was cancelled (after our interrupt)."""
        logger.info("MiniCPM-o response cancelled")
        self._response_state = ResponseState.CANCELLED
        if self._state == ClientState.INTERRUPTING:
            self._state = ClientState.READY

    async def _on_input_speech_started(self, msg: Dict) -> None:
        """Model's server-side VAD detected user started speaking."""
        logger.debug("MiniCPM-o: server VAD detected speech start")
        if self._on_speech_start:
            try:
                await self._on_speech_start()
            except Exception as e:
                logger.error("on_speech_start callback error: %s", e)

    async def _on_input_speech_stopped(self, msg: Dict) -> None:
        """Model's server-side VAD detected user stopped speaking."""
        logger.debug("MiniCPM-o: server VAD detected speech stop")
        if self._on_speech_stop:
            try:
                await self._on_speech_stop()
            except Exception as e:
                logger.error("on_speech_stop callback error: %s", e)

    async def _on_input_committed(self, msg: Dict) -> None:
        """Audio buffer commit acknowledged."""
        logger.debug("MiniCPM-o: input audio buffer committed")
        self.latency.response_started()

    async def _on_input_cleared(self, msg: Dict) -> None:
        """Audio buffer was cleared."""
        logger.debug("MiniCPM-o: input audio buffer cleared")

    async def _on_rate_limits_updated(self, msg: Dict) -> None:
        """Rate limit info from server."""
        limits = msg.get("rate_limits", [])
        for limit in limits:
            name      = limit.get("name", "")
            remaining = limit.get("remaining", "?")
            logger.debug(
                "MiniCPM-o rate limit: %s remaining=%s", name, remaining
            )

    async def _on_error_message(self, msg: Dict) -> None:
        """Server sent an error message."""
        error     = msg.get("error", {})
        code      = error.get("code", "unknown")
        message   = error.get("message", str(error))
        err_type  = error.get("type", "")

        self.usage.errors += 1

        logger.error(
            "MiniCPM-o API error: code=%s type=%s message=%s",
            code, err_type, message,
        )

        # Determine if error is fatal
        fatal_codes = {
            "auth_error",
            "invalid_api_key",
            "model_not_found",
            "quota_exceeded",
        }
        if code in fatal_codes:
            self._state = ClientState.ERROR
            logger.critical(
                "MiniCPM-o fatal error (%s): %s", code, message
            )

        if self._on_error:
            try:
                await self._on_error(code, error)
            except Exception as e:
                logger.error("on_error callback error: %s", e)

    async def _on_conversation_item_created(self, msg: Dict) -> None:
        """A conversation item was created (e.g. user transcript)."""
        item = msg.get("item", {})
        role = item.get("role", "")
        logger.debug("MiniCPM-o conversation item created: role=%s", role)

    async def _on_response_created(self, msg: Dict) -> None:
        """Model started generating a response."""
        self._response_seq   += 1
        self._audio_chunk_seq = 0
        self._response_state  = ResponseState.GENERATING
        self._state           = ClientState.RESPONDING
        self.latency.response_started()
        logger.debug(
            "MiniCPM-o response created #%d", self._response_seq
        )

    async def _on_response_output_item_added(self, msg: Dict) -> None:
        logger.debug("MiniCPM-o response output item added")

    async def _on_response_content_part_added(self, msg: Dict) -> None:
        logger.debug("MiniCPM-o response content part added")

    async def _on_response_text_delta(self, msg: Dict) -> None:
        """Text transcript delta from model (alongside audio)."""
        delta = msg.get("delta", "")
        if delta:
            logger.debug("MiniCPM-o transcript: %s", delta)

    async def _on_response_text_done(self, msg: Dict) -> None:
        text = msg.get("text", "")
        if text:
            logger.info("MiniCPM-o full transcript: %s", text[:200])

    # Message type → handler mapping
    _message_handlers: Dict[str, Any] = {
        "session.created":                       _on_session_created,
        "session.updated":                       _on_session_updated,
        "response.created":                      _on_response_created,
        "response.audio.delta":                  _on_response_audio_delta,
        "response.audio.done":                   _on_response_audio_done,
        "response.done":                         _on_response_done,
        "response.cancelled":                    _on_response_cancelled,
        "response.output_item.added":            _on_response_output_item_added,
        "response.content_part.added":           _on_response_content_part_added,
        "response.audio_transcript.delta":       _on_response_text_delta,
        "response.audio_transcript.done":        _on_response_text_done,
        "input_audio_buffer.speech_started":     _on_input_speech_started,
        "input_audio_buffer.speech_stopped":     _on_input_speech_stopped,
        "input_audio_buffer.committed":          _on_input_committed,
        "input_audio_buffer.cleared":            _on_input_cleared,
        "conversation.item.created":             _on_conversation_item_created,
        "rate_limits.updated":                   _on_rate_limits_updated,
        "error":                                 _on_error_message,
    }

    # -----------------------------------------------------------------------
    # Background: audio sender loop
    # -----------------------------------------------------------------------

    async def _audio_sender_loop(self) -> None:
        """
        Drains the audio send queue and sends chunks to MiniCPM-o.

        The queue is filled by send_audio().
        Sending is rate-limited to AUDIO_SEND_INTERVAL to avoid flooding.

        This loop runs continuously while connected.
        """
        logger.debug("MiniCPM-o audio sender loop started")

        try:
            while not self._stop_event.is_set():
                # Only send when in appropriate state
                if self._state not in (
                    ClientState.READY,
                    ClientState.STREAMING,
                    ClientState.RESPONDING,
                ):
                    await asyncio.sleep(0.05)
                    continue

                # Drain accumulated audio from queue
                accumulated = bytearray()
                try:
                    # Get first chunk (blocking with timeout)
                    chunk = await asyncio.wait_for(
                        self._audio_send_queue.get(),
                        timeout=AUDIO_SEND_INTERVAL,
                    )
                    accumulated.extend(chunk)

                    # Drain any additional chunks already in queue
                    while True:
                        try:
                            chunk = self._audio_send_queue.get_nowait()
                            accumulated.extend(chunk)
                            # Don't accumulate more than MAX_AUDIO_CHUNK_BYTES
                            if len(accumulated) >= MAX_AUDIO_CHUNK_BYTES:
                                break
                        except asyncio.QueueEmpty:
                            break

                except asyncio.TimeoutError:
                    # No audio to send — idle
                    continue

                if not accumulated:
                    continue

                # Send to MiniCPM-o
                await self._send_audio_bytes(bytes(accumulated))

        except asyncio.CancelledError:
            logger.debug("MiniCPM-o audio sender loop cancelled")
            raise
        except Exception as e:
            logger.error(
                "MiniCPM-o audio sender loop error: %s",
                e, exc_info=True,
            )

    async def _send_audio_bytes(self, pcm_bytes: bytes) -> None:
        """
        Send accumulated PCM bytes to MiniCPM-o as input_audio_buffer.append.
        Splits into MAX_AUDIO_CHUNK_BYTES if needed.
        """
        if not pcm_bytes:
            return

        # Track input audio duration
        from audio_utils import calc_duration_ms
        dur_ms = calc_duration_ms(
            len(pcm_bytes), self._config.input_sample_rate
        )
        self._input_audio_ms   += dur_ms
        self.usage.input_audio_ms += dur_ms

        offset = 0
        while offset < len(pcm_bytes):
            chunk = pcm_bytes[offset:offset + MAX_AUDIO_CHUNK_BYTES]
            msg   = _msg_audio_append(chunk)
            try:
                await self._send_raw(msg)
            except Exception as e:
                logger.error("Failed to send audio chunk: %s", e)
                break
            offset += MAX_AUDIO_CHUNK_BYTES

        logger.debug(
            "MiniCPM-o → sent %.0fms audio (%d bytes)",
            dur_ms, len(pcm_bytes),
        )

    # -----------------------------------------------------------------------
    # Public API: audio streaming
    # -----------------------------------------------------------------------

    async def send_audio(self, pcm_bytes: bytes) -> bool:
        """
        Queue PCM audio to be sent to MiniCPM-o.

        This is the main method to call when user audio arrives.
        Non-blocking: puts audio into the send queue.
        The sender loop drains the queue and sends to the API.

        Args:
            pcm_bytes: Raw 16-bit signed PCM, 16kHz, mono

        Returns:
            True if queued, False if queue full or not connected
        """
        if self._state not in (
            ClientState.READY,
            ClientState.STREAMING,
            ClientState.RESPONDING,
        ):
            logger.debug(
                "send_audio: ignoring (state=%s)", self._state.name
            )
            return False

        if not pcm_bytes:
            return True

        try:
            self._audio_send_queue.put_nowait(pcm_bytes)
            self._state = ClientState.STREAMING
            return True
        except asyncio.QueueFull:
            logger.warning(
                "MiniCPM-o audio send queue full, dropping %d bytes",
                len(pcm_bytes),
            )
            return False

    async def commit_audio(self) -> None:
        """
        Signal to MiniCPM-o that the user's turn is complete.
        Only needed when server-side VAD is disabled.
        When server VAD is enabled (default), this is called automatically.
        """
        if self._state not in (
            ClientState.READY,
            ClientState.STREAMING,
        ):
            return

        # Wait for send queue to drain first
        await self._drain_send_queue()

        await self._send_raw(_msg_audio_commit())
        self.latency.response_started()
        logger.debug("MiniCPM-o: audio buffer committed")

    async def interrupt(self) -> None:
        """
        Interrupt the current model response.

        Steps:
          1. Set state to INTERRUPTING
          2. Clear server-side audio buffer
          3. Send response.cancel
          4. Wait briefly for cancellation confirmation

        This should be called when user starts speaking during bot playback.
        """
        if self._state not in (
            ClientState.RESPONDING,
            ClientState.STREAMING,
            ClientState.READY,
        ):
            logger.debug(
                "interrupt: ignoring (state=%s)", self._state.name
            )
            return

        logger.info("MiniCPM-o: sending interrupt")
        self._state = ClientState.INTERRUPTING
        self.usage.interruptions += 1

        try:
            # Step 1: Clear server audio buffer
            await self._send_raw(_msg_clear_audio_buffer())

            # Step 2: Cancel the response
            await self._send_raw(_msg_response_cancel())

        except Exception as e:
            logger.error("MiniCPM-o interrupt error: %s", e)
            self._state = ClientState.READY
            return

        # Step 3: Wait for cancellation acknowledgment (brief)
        try:
            await asyncio.wait_for(
                self._wait_for_state(ClientState.READY),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            # Force back to READY even without confirmation
            logger.warning(
                "MiniCPM-o interrupt: no cancellation confirmation, "
                "forcing READY"
            )
            self._state = ClientState.READY

        logger.info("MiniCPM-o interrupt complete")

    async def update_session(self, updates: Dict[str, Any]) -> None:
        """
        Update session configuration mid-call.
        E.g. change system prompt, voice, temperature.
        """
        if self._state not in (ClientState.READY, ClientState.RESPONDING):
            logger.warning(
                "update_session: invalid state %s", self._state.name
            )
            return
        await self._send_raw(_msg_session_update(updates))
        logger.info("MiniCPM-o session updated: %s", list(updates.keys()))

    async def send_text_message(self, text: str) -> None:
        """
        Inject a text message into the conversation.
        Useful for system messages or context injection mid-call.
        """
        msg = json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text}
                ],
            },
        })
        await self._send_raw(msg)
        await self._send_raw(_msg_response_create())
        logger.info(
            "MiniCPM-o text message injected: %s", text[:80]
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    async def _send_raw(self, message: str) -> None:
        """Send a raw JSON string over the WebSocket."""
        if self._ws is None or self._ws.closed:
            raise ConnectionError("MiniCPM-o WebSocket not connected")
        try:
            await self._ws.send(message)
        except ConnectionClosed as e:
            raise ConnectionError(f"MiniCPM-o WS closed: {e}") from e

    async def _emit_audio_chunk(
        self, pcm_bytes: bytes, is_final: bool
    ) -> None:
        """
        Build an AudioResponse and fire on_audio_chunk callback.
        This is called for every audio chunk from the model.
        """
        self._audio_chunk_seq += 1

        from audio_utils import calc_duration_ms
        if pcm_bytes:
            dur_ms = calc_duration_ms(
                len(pcm_bytes), self._config.output_sample_rate
            )
            self.usage.output_audio_ms += dur_ms

        response = AudioResponse(
            pcm_bytes=pcm_bytes,
            sample_rate=self._config.output_sample_rate,
            is_final=is_final,
            sequence=self._audio_chunk_seq,
        )

        if self._on_audio_chunk:
            try:
                await self._on_audio_chunk(response)
            except Exception as e:
                logger.error(
                    "on_audio_chunk callback error: %s",
                    e, exc_info=True,
                )

    async def _drain_send_queue(self, timeout: float = 2.0) -> None:
        """Wait for the audio send queue to empty."""
        deadline = time.monotonic() + timeout
        while not self._audio_send_queue.empty():
            if time.monotonic() > deadline:
                logger.warning("Drain timeout: queue not empty")
                break
            await asyncio.sleep(0.02)

    async def _handle_disconnect(self, reason: str) -> None:
        """Handle unexpected WebSocket disconnect."""
        if self._state in (ClientState.CLOSING, ClientState.CLOSED):
            return

        logger.warning("MiniCPM-o disconnected: %s", reason)
        self.usage.errors += 1

        if self._on_disconnected:
            try:
                await self._on_disconnected()
            except Exception as e:
                logger.error("on_disconnected callback error: %s", e)

        if (self._reconnect_enabled
                and self._reconnect_count < self._reconnect_max):
            asyncio.ensure_future(self._reconnect(reason))
        else:
            self._state = ClientState.ERROR
            logger.error(
                "MiniCPM-o: not reconnecting "
                "(enabled=%s count=%d max=%d)",
                self._reconnect_enabled,
                self._reconnect_count,
                self._reconnect_max,
            )

    async def _reconnect(self, reason: str) -> None:
        """Exponential backoff reconnection."""
        self._reconnect_count += 1
        self.usage.reconnections += 1
        delay = min(
            RECONNECT_BASE_DELAY * (2 ** (self._reconnect_count - 1)),
            30.0,
        )

        logger.info(
            "MiniCPM-o reconnecting in %.1fs (attempt %d/%d, reason=%s)",
            delay, self._reconnect_count, self._reconnect_max, reason,
        )

        await asyncio.sleep(delay)

        try:
            self._state = ClientState.IDLE
            # Drain old send queue
            while not self._audio_send_queue.empty():
                try:
                    self._audio_send_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            await self.connect()
            self._reconnect_count = 0
            logger.info("MiniCPM-o reconnected successfully")

        except Exception as e:
            logger.error("MiniCPM-o reconnect failed: %s", e)
            if self._reconnect_count < self._reconnect_max:
                await self._reconnect(str(e))
            else:
                self._state = ClientState.ERROR
                logger.critical(
                    "MiniCPM-o max reconnect attempts (%d) reached",
                    self._reconnect_max,
                )

    # -----------------------------------------------------------------------
    # Properties / info
    # -----------------------------------------------------------------------

    @property
    def state(self) -> ClientState:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state in (
            ClientState.READY,
            ClientState.STREAMING,
            ClientState.RESPONDING,
        )

    @property
    def is_responding(self) -> bool:
        return self._state == ClientState.RESPONDING

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def send_queue_size(self) -> int:
        return self._audio_send_queue.qsize()

    def get_info(self) -> Dict:
        return {
            "state":           self._state.name,
            "session_id":      self._session_id,
            "response_state":  self._response_state.name,
            "response_seq":    self._response_seq,
            "send_queue_size": self._audio_send_queue.qsize(),
            "reconnect_count": self._reconnect_count,
            "usage":           self.usage.to_dict(),
            "latency": {
                "mean_ms": round(self.latency.mean_ms, 1),
                "p95_ms":  round(self.latency.p95_ms, 1),
                "samples": len(self.latency.samples),
            },
        }


# ---------------------------------------------------------------------------
# Factory from AppConfig
# ---------------------------------------------------------------------------

def create_minicpm_client(
    cfg:             "Any",
    on_audio_chunk:  Optional[Callable[[AudioResponse], Coroutine]] = None,
    on_response_done:Optional[Callable[[Dict], Coroutine]]          = None,
    on_speech_start: Optional[Callable[[], Coroutine]]              = None,
    on_speech_stop:  Optional[Callable[[], Coroutine]]              = None,
    on_error:        Optional[Callable[[str, Dict], Coroutine]]     = None,
    on_connected:    Optional[Callable[[], Coroutine]]              = None,
    on_disconnected: Optional[Callable[[], Coroutine]]              = None,
) -> MiniCPMClient:
    """
    Create a MiniCPMClient from AppConfig.

    Args:
        cfg: AppConfig instance (from config.py)
        on_audio_chunk:   Called for each audio chunk from model
        on_response_done: Called when model finishes a response
        on_speech_start:  Called when server VAD detects speech start
        on_speech_stop:   Called when server VAD detects speech stop
        on_error:         Called on API errors
        on_connected:     Called when session is ready
        on_disconnected:  Called when connection closes

    Returns:
        Configured MiniCPMClient (not yet connected)
    """
    m = cfg.minicpm

    session_cfg = SessionConfig(
        model=m.model,
        system_prompt=m.system_prompt,
        voice_id=m.voice_id,
        language=m.language,
        temperature=m.temperature,
        input_audio_format="pcm16",
        output_audio_format="pcm16",
        input_sample_rate=int(cfg.audio.model_sample_rate),
        output_sample_rate=m.output_sample_rate,
        turn_detection={
            "type":               "server_vad",
            "threshold":          0.5,
            "prefix_padding_ms":  300,
            "silence_duration_ms": cfg.vad.silence_duration_ms,
            "create_response":    True,
        },
    )

    return MiniCPMClient(
        api_key=m.api_key,
        api_base_url=m.api_base_url,
        config=session_cfg,
        on_audio_chunk=on_audio_chunk,
        on_response_done=on_response_done,
        on_speech_start=on_speech_start,
        on_speech_stop=on_speech_stop,
        on_error=on_error,
        on_connected=on_connected,
        on_disconnected=on_disconnected,
        reconnect_enabled=m.reconnect_on_error,
        reconnect_max=m.max_reconnect_attempts,
    )
 
