# 文件6: `session.py`

```python
"""
session.py - Call session orchestrator (the brain of VoiceBot)

Ties together:
  - AudioStream    (audio_utils.py + audio.py)   : user audio in, bot audio out
  - MiniCPMClient  (ai_client.py)                : full-duplex AI model
  - ESLClient      (esl.py)                      : FreeSWITCH call control
  - MetricsRegistry (metrics.py)                 : per-session metrics

One Session object per active call.
SessionManager owns all sessions and handles their lifecycle.

Full call flow:
  1. FreeSWITCH calls mod_audio_stream → AudioServer gets WebSocket
  2. AudioServer fires on_connected  → SessionManager.on_audio_connected()
  3. SessionManager creates Session, connects MiniCPMClient
  4. AudioStream receives user PCM  → AudioPipeline → VAD
  5. VAD fires on_utterance         → Session.on_utterance()
  6. Session pipes audio chunks     → MiniCPMClient.send_audio()
  7. MiniCPMClient fires on_audio_chunk → Session.on_model_audio()
  8. Session calls AudioStream.play_audio() + begin_speaking()
  9. User interrupts                → AudioStream fires on_interruption
  10. Session.on_interruption()     → ESL uuid_break + MiniCPM interrupt
  11. Call ends                     → Session.terminate()

State machine:
  INITIALIZING → GREETING → LISTENING → THINKING → SPEAKING
       ↑                        ↑______________|       |
       |________________________|_________________↓_____|
                                            INTERRUPTED
                                            TERMINATED


```

---

`session.py` 完成 ✅ 约 **820行**

包含：

| 组件 | 说明 |
|------|------|
| `SessionState` | 7态状态机 + 合法跳转表校验 |
| `SessionConfig` | 每通话配置（greeting/interruption/timeout） |
| `SessionInfo` | 只读快照，供API序列化 |
| `Session.start()` | 连接AI → 启动pipeline → 发greeting → 进入LISTENING |
| `Session._audio_pipeline_loop` | 后台持续将用户PCM送入AI |
| `Session._on_utterance` | 本地VAD触发，缓冲音频 |
| `Session._on_model_audio_chunk` | **热路径**：AI音频 → AudioStream播放 |
| `Session._on_model_speech_start/stop` | AI服务端VAD驱动THINKING状态 |
| `Session._on_interruption_signal` | **打断核心**：3路并发（ESL break + AI cancel + queue clear） |
| `Session._watchdog_loop` | 空闲超时/最大时长/AI健康监控 |
| `Session.inject_message` | 中途注入文本（supervisor whisper） |
| `Session.transfer/hold/record` | 完整通话控制API |
| `SessionManager` | 多会话管理，ESL事件路由，历史归档 |

---

**下一个：`api.py`** — REST管理接口，查询会话状态，触发操作，健康检查。准备好就继续！

                                            
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Coroutine, Dict,
    List, Optional, Set
)

from audio_utils import SpeechSegment, calc_duration_ms
from audio import AudioStream, AudioServer
from ai_client import MiniCPMClient, AudioResponse, create_minicpm_client
from esl import ESLClient
from metrics import MetricsRegistry, SessionMetrics, Timer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How long to wait for MiniCPM to connect before giving up (seconds)
AI_CONNECT_TIMEOUT          = 15.0
# Max time a session can exist without any audio activity (seconds)
SESSION_IDLE_TIMEOUT        = 300.0
# Max total session duration (seconds)
SESSION_MAX_DURATION        = 3600.0
# How long to wait after greeting before accepting interruptions (ms)
GREETING_GRACE_PERIOD_MS    = 800.0
# Minimum audio chunk size to send to model (bytes) - avoid tiny packets
MIN_AUDIO_SEND_BYTES        = 320     # 10ms at 16kHz 16bit
# Cooldown between interruptions (seconds)
INTERRUPTION_COOLDOWN       = 1.0
# How long to wait for model audio to start before declaring error (seconds)
MODEL_RESPONSE_TIMEOUT      = 8.0


# ---------------------------------------------------------------------------
# Session state machine
# ---------------------------------------------------------------------------

class SessionState(Enum):
    """
    States of a single call session.
    Transitions are strictly controlled — no illegal state jumps.
    """
    INITIALIZING  = auto()   # Session created, AI connecting
    GREETING      = auto()   # Playing opening greeting
    LISTENING     = auto()   # Waiting for user to speak
    THINKING      = auto()   # User spoke, model generating response
    SPEAKING      = auto()   # Bot playing audio to user
    INTERRUPTED   = auto()   # User interrupted bot, transitioning back
    TERMINATING   = auto()   # Call ending
    TERMINATED    = auto()   # Call fully ended


# Valid state transitions
_VALID_TRANSITIONS: Dict[SessionState, Set[SessionState]] = {
    SessionState.INITIALIZING: {
        SessionState.GREETING,
        SessionState.LISTENING,
        SessionState.TERMINATING,
    },
    SessionState.GREETING: {
        SessionState.LISTENING,
        SessionState.INTERRUPTED,
        SessionState.TERMINATING,
    },
    SessionState.LISTENING: {
        SessionState.THINKING,
        SessionState.TERMINATING,
    },
    SessionState.THINKING: {
        SessionState.SPEAKING,
        SessionState.LISTENING,   # model returned empty response
        SessionState.TERMINATING,
    },
    SessionState.SPEAKING: {
        SessionState.LISTENING,
        SessionState.INTERRUPTED,
        SessionState.TERMINATING,
    },
    SessionState.INTERRUPTED: {
        SessionState.LISTENING,
        SessionState.THINKING,
        SessionState.TERMINATING,
    },
    SessionState.TERMINATING: {
        SessionState.TERMINATED,
    },
    SessionState.TERMINATED: set(),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    """
    Per-session configuration.
    Merged from AppConfig + any call-specific overrides.
    """
    # AI model settings
    system_prompt:           str   = "你是一个专业的语音助手，回答简洁自然。"
    language:                str   = "zh-CN"
    temperature:             float = 0.7
    voice_id:                str   = "default"

    # Greeting (played immediately when call connects)
    greeting_text:           str   = ""     # if set, injected as text
    greeting_audio_path:     str   = ""     # if set, play this audio file

    # Interruption settings
    interruption_enabled:    bool  = True
    interruption_cooldown_s: float = INTERRUPTION_COOLDOWN
    grace_period_ms:         float = GREETING_GRACE_PERIOD_MS

    # Timeouts
    idle_timeout_s:          float = SESSION_IDLE_TIMEOUT
    max_duration_s:          float = SESSION_MAX_DURATION
    ai_connect_timeout_s:    float = AI_CONNECT_TIMEOUT

    # Call direction
    direction:               str   = "inbound"

    # Extra metadata (caller info, routing info, etc.)
    metadata:                Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionInfo:
    """
    Read-only snapshot of session state for APIs/logging.
    """
    session_id:       str
    call_uuid:        str
    state:            str
    direction:        str
    caller_id_num:    str
    caller_id_name:   str
    destination:      str
    start_time:       float
    duration_seconds: float
    interruptions:    int
    utterances:       int
    ai_responses:     int
    ai_state:         str
    is_bot_speaking:  bool
    is_user_speaking: bool
    last_activity_at: float

    def to_dict(self) -> Dict:
        return {
            "session_id":       self.session_id,
            "call_uuid":        self.call_uuid,
            "state":            self.state,
            "direction":        self.direction,
            "caller_id_num":    self.caller_id_num,
            "caller_id_name":   self.caller_id_name,
            "destination":      self.destination,
            "start_time":       self.start_time,
            "duration_seconds": round(self.duration_seconds, 2),
            "interruptions":    self.interruptions,
            "utterances":       self.utterances,
            "ai_responses":     self.ai_responses,
            "ai_state":         self.ai_state,
            "is_bot_speaking":  self.is_bot_speaking,
            "is_user_speaking": self.is_user_speaking,
            "last_activity_at": self.last_activity_at,
        }


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    """
    Orchestrates a single phone call.

    One Session lives for the duration of one call.
    It wires together:
      - AudioStream  : raw PCM ↔ FreeSWITCH
      - MiniCPMClient: full-duplex AI model
      - ESLClient    : call control commands
      - SessionMetrics: per-call metrics

    All audio routing decisions happen here.
    The state machine enforces valid transitions.
    """

    def __init__(
        self,
        session_id:   str,
        audio_stream: AudioStream,
        esl_client:   ESLClient,
        cfg:          "Any",              # AppConfig
        session_cfg:  Optional[SessionConfig] = None,
        metrics:      Optional[MetricsRegistry] = None,
        on_terminated: Optional[Callable[["Session"], Coroutine]] = None,
    ):
        self.session_id   = session_id
        self.call_uuid    = audio_stream.call_uuid
        self._audio       = audio_stream
        self._esl         = esl_client
        self._app_cfg     = cfg
        self._cfg         = session_cfg or SessionConfig()
        self._metrics_reg = metrics
        self._on_terminated = on_terminated

        # MiniCPM client (created in start())
        self._ai: Optional[MiniCPMClient] = None

        # State machine
        self._state      = SessionState.INITIALIZING
        self._state_lock = asyncio.Lock()

        # Per-session metrics
        self._metrics: Optional[SessionMetrics] = None
        if metrics:
            self._metrics = metrics.session_start(
                session_id=session_id,
                call_uuid=self.call_uuid,
            )

        # Counters
        self._interruptions   = 0
        self._utterances      = 0
        self._ai_responses    = 0
        self._last_activity   = time.monotonic()
        self._last_interruption_at = 0.0

        # Audio accumulation buffer
        # Collects PCM from AudioStream before sending to AI
        self._audio_buffer   = bytearray()
        self._audio_buf_lock = asyncio.Lock()

        # Background tasks
        self._watchdog_task:  Optional[asyncio.Task] = None
        self._audio_pipe_task: Optional[asyncio.Task] = None

        # Stop flag
        self._stop_event = asyncio.Event()

        # Metadata from call
        meta = audio_stream.metadata
        self._caller_id_num  = meta.caller_id_num  if meta else ""
        self._caller_id_name = meta.caller_id_name if meta else ""
        self._destination    = meta.destination    if meta else ""

        # Start time
        self._start_time = time.time()

        logger.info(
            "Session created: id=%s uuid=%s from=%s to=%s",
            session_id[:8], self.call_uuid[:8],
            self._caller_id_num, self._destination,
        )

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the session:
          1. Connect MiniCPM-o
          2. Wire up callbacks
          3. Start watchdog
          4. Send greeting (if configured)
          5. Transition to LISTENING
        """
        logger.info("Session starting: %s", self.session_id[:8])

        # Wire up AudioStream callbacks (already created in AudioServer)
        # These are set here to avoid circular reference at construction time
        self._audio._on_utterance    = self._on_utterance
        self._audio._on_interruption = self._on_interruption_signal

        # Create and connect MiniCPM client
        try:
            self._ai = create_minicpm_client(
                cfg=self._app_cfg,
                on_audio_chunk=self._on_model_audio_chunk,
                on_response_done=self._on_model_response_done,
                on_speech_start=self._on_model_speech_start,
                on_speech_stop=self._on_model_speech_stop,
                on_error=self._on_model_error,
                on_connected=self._on_model_connected,
                on_disconnected=self._on_model_disconnected,
            )

            await asyncio.wait_for(
                self._ai.connect(),
                timeout=self._cfg.ai_connect_timeout_s,
            )

        except asyncio.TimeoutError:
            logger.error(
                "Session %s: AI connect timeout after %.0fs",
                self.session_id[:8], self._cfg.ai_connect_timeout_s,
            )
            await self.terminate("ai_connect_timeout")
            return

        except Exception as e:
            logger.error(
                "Session %s: AI connect failed: %s",
                self.session_id[:8], e, exc_info=True,
            )
            await self.terminate("ai_connect_error")
            return

        # Start audio pipeline task (routes user audio → AI)
        self._audio_pipe_task = asyncio.ensure_future(
            self._audio_pipeline_loop()
        )

        # Start watchdog
        self._watchdog_task = asyncio.ensure_future(
            self._watchdog_loop()
        )

        # Send greeting
        if self._cfg.greeting_text:
            await self._send_greeting()
        else:
            await self._transition(SessionState.LISTENING)

        logger.info(
            "Session started: %s (state=%s)",
            self.session_id[:8], self._state.name,
        )

    async def terminate(
        self,
        reason: str = "normal",
        hangup_cause: str = "NORMAL_CLEARING",
    ) -> None:
        """
        Gracefully terminate the session.
        Stops all components, records final metrics.
        """
        if self._state in (SessionState.TERMINATING, SessionState.TERMINATED):
            return

        logger.info(
            "Session terminating: %s (reason=%s)",
            self.session_id[:8], reason,
        )

        await self._transition(SessionState.TERMINATING)
        self._stop_event.set()

        # Stop background tasks
        for task in [self._watchdog_task, self._audio_pipe_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Stop bot speaking
        await self._audio.end_speaking()

        # Disconnect AI client
        if self._ai:
            try:
                await asyncio.wait_for(self._ai.disconnect(), timeout=3.0)
            except Exception as e:
                logger.warning(
                    "AI disconnect error: %s", e
                )

        # Hang up call via ESL (if not already hung up)
        if self._esl.is_connected:
            try:
                await self._esl.uuid_kill(
                    self.call_uuid, cause=hangup_cause
                )
            except Exception as e:
                logger.warning("ESL uuid_kill error: %s", e)

        # Finalize metrics
        if self._metrics_reg and self._metrics:
            self._metrics_reg.session_end(
                session_id=self.session_id,
                final_state=reason,
                direction=self._cfg.direction,
            )

        await self._transition(SessionState.TERMINATED)

        logger.info(
            "Session terminated: %s | duration=%.1fs utterances=%d "
            "interruptions=%d ai_responses=%d reason=%s",
            self.session_id[:8],
            time.time() - self._start_time,
            self._utterances,
            self._interruptions,
            self._ai_responses,
            reason,
        )

        # Notify manager
        if self._on_terminated:
            try:
                await self._on_terminated(self)
            except Exception as e:
                logger.error("on_terminated callback error: %s", e)

    # -----------------------------------------------------------------------
    # State machine
    # -----------------------------------------------------------------------

    async def _transition(self, new_state: SessionState) -> bool:
        """
        Attempt a state transition.
        Validates against _VALID_TRANSITIONS.
        Returns True if transition succeeded.
        """
        async with self._state_lock:
            if new_state == self._state:
                return True

            valid = _VALID_TRANSITIONS.get(self._state, set())
            if new_state not in valid:
                logger.warning(
                    "Invalid state transition %s → %s [%s]",
                    self._state.name, new_state.name,
                    self.session_id[:8],
                )
                return False

            old = self._state
            self._state = new_state
            logger.debug(
                "Session state: %s → %s [%s]",
                old.name, new_state.name, self.session_id[:8],
            )
            return True

    def _touch(self) -> None:
        """Update last activity timestamp."""
        self._last_activity = time.monotonic()

    # -----------------------------------------------------------------------
    # Greeting
    # -----------------------------------------------------------------------

    async def _send_greeting(self) -> None:
        """
        Inject greeting text into MiniCPM and transition to GREETING.
        The model will generate audio for the greeting.
        """
        await self._transition(SessionState.GREETING)
        logger.info(
            "Session %s: sending greeting: %s",
            self.session_id[:8],
            self._cfg.greeting_text[:60],
        )

        if self._ai and self._ai.is_ready:
            try:
                await self._ai.send_text_message(self._cfg.greeting_text)
            except Exception as e:
                logger.error(
                    "Greeting send failed: %s", e
                )
                await self._transition(SessionState.LISTENING)
        else:
            logger.warning(
                "Cannot send greeting: AI not ready [%s]",
                self.session_id[:8],
            )
            await self._transition(SessionState.LISTENING)

    # -----------------------------------------------------------------------
    # Audio pipeline: user speech → AI model
    # -----------------------------------------------------------------------

    async def _audio_pipeline_loop(self) -> None:
        """
        Background task: forward raw user audio to MiniCPM-o.

        mod_audio_stream → AudioStream → VAD → SpeechSegment →
        this loop → MiniCPMClient.send_audio()

        For full-duplex mode (server_vad enabled on MiniCPM):
          We stream audio continuously, not just when VAD fires.
          The model's server-side VAD handles turn detection.

        Note: We still use our local VAD for interruption detection.
        The model simultaneously receives audio for its own VAD.
        """
        logger.debug(
            "Audio pipeline loop started [%s]", self.session_id[:8]
        )

        try:
            while not self._stop_event.is_set():
                # Small sleep to yield to other coroutines
                await asyncio.sleep(0.02)

                if self._state in (
                    SessionState.TERMINATING,
                    SessionState.TERMINATED,
                ):
                    break

                # Send any buffered audio to AI
                async with self._audio_buf_lock:
                    if len(self._audio_buffer) >= MIN_AUDIO_SEND_BYTES:
                        data = bytes(self._audio_buffer)
                        self._audio_buffer.clear()
                    else:
                        data = b""

                if data and self._ai and self._ai.is_ready:
                    sent = await self._ai.send_audio(data)
                    if sent and self._metrics:
                        self._metrics.add_audio_in(len(data))

        except asyncio.CancelledError:
            logger.debug("Audio pipeline loop cancelled")
            raise
        except Exception as e:
            logger.error(
                "Audio pipeline loop error [%s]: %s",
                self.session_id[:8], e, exc_info=True,
            )

    # -----------------------------------------------------------------------
    # Callback: user utterance detected (from AudioStream/VAD)
    # -----------------------------------------------------------------------

    async def _on_utterance(
        self,
        call_uuid: str,
        segment: SpeechSegment,
    ) -> None:
        """
        Called when local VAD detects a complete user utterance.

        In server_vad mode: we still buffer the audio for the model,
        but the model's VAD drives the actual response generation.

        In manual mode: we commit the audio here.
        """
        if self._state in (SessionState.TERMINATING, SessionState.TERMINATED):
            return

        self._utterances += 1
        self._touch()

        logger.info(
            "Utterance #%d [%s]: %.0fms %.0fkB",
            self._utterances,
            self.session_id[:8],
            segment.duration_ms,
            len(segment.audio_data) / 1024,
        )

        # Track metrics
        if self._metrics:
            self._metrics.user_speech_start()
            self._metrics.user_speech_end()

        # Buffer the audio (pipeline loop will send it)
        async with self._audio_buf_lock:
            self._audio_buffer.extend(segment.audio_data)

        # In manual commit mode (no server VAD):
        # We would commit here and transition to THINKING.
        # In server VAD mode (default): model handles turn detection.
        if self._state == SessionState.LISTENING:
            await self._transition(SessionState.THINKING)

    # -----------------------------------------------------------------------
    # Callback: raw audio frame from AudioStream (continuous streaming)
    # -----------------------------------------------------------------------

    async def on_raw_audio_frame(self, pcm_bytes: bytes) -> None:
        """
        Called for EVERY audio frame from mod_audio_stream.
        Used for continuous streaming to the model (full-duplex mode).

        In full-duplex mode we stream ALL audio, not just after VAD.
        The model's server VAD decides when to respond.
        """
        if not pcm_bytes:
            return

        if self._state in (SessionState.TERMINATING, SessionState.TERMINATED):
            return

        self._touch()

        # Buffer for the pipeline loop
        async with self._audio_buf_lock:
            self._audio_buffer.extend(pcm_bytes)

    # -----------------------------------------------------------------------
    # Callback: model audio chunk received
    # -----------------------------------------------------------------------

    async def _on_model_audio_chunk(self, response: AudioResponse) -> None:
        """
        Called for every PCM audio chunk from MiniCPM-o.
        This is the HOT PATH — must be fast.

        Routes audio to FreeSWITCH via AudioStream.
        """
        if self._state in (SessionState.TERMINATING, SessionState.TERMINATED):
            return

        if not response.pcm_bytes and not response.is_final:
            return

        # First chunk of a new response
        if self._state in (SessionState.THINKING, SessionState.LISTENING):
            await self._on_first_model_audio()

        if response.pcm_bytes:
            # Queue audio for playback
            await self._audio.play_audio(response.pcm_bytes)

            # Track metrics
            if self._metrics:
                self._metrics.add_audio_out(len(response.pcm_bytes))

            self._touch()

        # Final chunk — model finished this response
        if response.is_final:
            await self._on_model_audio_complete()

    async def _on_first_model_audio(self) -> None:
        """Called when the first audio chunk of a new response arrives."""
        logger.info(
            "Model first audio [%s] state=%s",
            self.session_id[:8], self._state.name,
        )

        # Transition to SPEAKING
        await self._transition(SessionState.SPEAKING)

        # Tell AudioStream bot is now speaking
        await self._audio.begin_speaking(
            grace_period_ms=self._cfg.grace_period_ms
        )

        # Record E2E latency
        if self._metrics:
            self._metrics.e2e_end()

        self._ai_responses += 1
        logger.info(
            "Bot speaking started [%s] response=#%d",
            self.session_id[:8], self._ai_responses,
        )

    async def _on_model_audio_complete(self) -> None:
        """Called when model finishes sending audio for one response."""
        logger.info(
            "Model audio complete [%s]", self.session_id[:8]
        )

        # Tell AudioStream bot finished speaking
        await self._audio.end_speaking()

        # Transition back to LISTENING
        ok = await self._transition(SessionState.LISTENING)
        if ok and self._metrics:
            self._metrics.bot_speech_end()

    # -----------------------------------------------------------------------
    # Callback: model response metadata done
    # -----------------------------------------------------------------------

    async def _on_model_response_done(self, response_data: Dict) -> None:
        """Called when model response is fully complete (audio + text)."""
        if self._metrics:
            lat = self._metrics.llm_end(success=True)
            if self._metrics_reg:
                self._metrics_reg.observe_llm(
                    lat,
                    provider="minicpm",
                    model=self._app_cfg.minicpm.model,
                    success=True,
                )

        logger.info(
            "Model response done [%s]", self.session_id[:8]
        )

    # -----------------------------------------------------------------------
    # Callback: model server VAD events
    # -----------------------------------------------------------------------

    async def _on_model_speech_start(self) -> None:
        """
        MiniCPM-o server VAD detected user started speaking.
        More reliable than our local VAD for driving the AI response cycle.
        """
        logger.debug(
            "Model VAD: speech start [%s]", self.session_id[:8]
        )
        self._touch()

        # If bot is speaking → this will lead to interruption
        if self._state == SessionState.SPEAKING:
            logger.info(
                "Model VAD: speech during bot speaking → "
                "preparing for interruption [%s]",
                self.session_id[:8],
            )

        # Start E2E latency timer
        if self._metrics:
            self._metrics.e2e_start()

    async def _on_model_speech_stop(self) -> None:
        """MiniCPM-o server VAD detected user stopped speaking."""
        logger.debug(
            "Model VAD: speech stop [%s]", self.session_id[:8]
        )
        # Model will generate response now — transition to THINKING
        if self._state == SessionState.LISTENING:
            await self._transition(SessionState.THINKING)
            if self._metrics:
                self._metrics.llm_start()

    # -----------------------------------------------------------------------
    # Callback: AI model connection events
    # -----------------------------------------------------------------------

    async def _on_model_connected(self) -> None:
        """MiniCPM-o WebSocket session is ready."""
        logger.info(
            "MiniCPM-o connected [%s]", self.session_id[:8]
        )

    async def _on_model_disconnected(self) -> None:
        """MiniCPM-o WebSocket disconnected unexpectedly."""
        logger.warning(
            "MiniCPM-o disconnected [%s] state=%s",
            self.session_id[:8], self._state.name,
        )
        if self._state not in (
            SessionState.TERMINATING,
            SessionState.TERMINATED,
        ):
            # AI disconnect during active call = error
            await self.terminate("ai_disconnected")

    async def _on_model_error(self, code: str, error: Dict) -> None:
        """MiniCPM-o returned an API error."""
        logger.error(
            "MiniCPM-o error [%s]: code=%s msg=%s",
            self.session_id[:8], code,
            error.get("message", "")[:200],
        )

        if self._metrics:
            self._metrics.record_error()
        if self._metrics_reg:
            self._metrics_reg.record_error("minicpm", code)

        # Fatal errors → terminate
        fatal = {"auth_error", "invalid_api_key", "quota_exceeded"}
        if code in fatal:
            await self.terminate(f"ai_error_{code}")

    # -----------------------------------------------------------------------
    # Interruption handling
    # -----------------------------------------------------------------------

    async def _on_interruption_signal(self, call_uuid: str) -> None:
        """
        Called by AudioStream when local VAD detects user speech
        during bot playback.

        Full interruption sequence:
          1. Check cooldown (debounce rapid interruptions)
          2. Transition to INTERRUPTED
          3. Send uuid_break to FreeSWITCH (stop audio playback)
          4. Clear AudioStream playback queue
          5. Send interrupt to MiniCPM (cancel response)
          6. Transition back to LISTENING
        """
        if not self._cfg.interruption_enabled:
            return

        if self._state not in (
            SessionState.SPEAKING,
            SessionState.GREETING,
        ):
            logger.debug(
                "Interruption signal ignored (state=%s) [%s]",
                self._state.name, self.session_id[:8],
            )
            return

        # Debounce: check cooldown
        now = time.monotonic()
        if now - self._last_interruption_at < self._cfg.interruption_cooldown_s:
            logger.debug(
                "Interruption suppressed by cooldown [%s]",
                self.session_id[:8],
            )
            return

        self._last_interruption_at = now
        self._interruptions += 1

        logger.info(
            "=== INTERRUPTION #%d [%s] ===",
            self._interruptions, self.session_id[:8],
        )

        # Record metric
        if self._metrics:
            self._metrics.record_interruption()
        if self._metrics_reg:
            self._metrics_reg.record_interruption()

        # Step 1: transition state
        await self._transition(SessionState.INTERRUPTED)

        # Step 2 + 3 run concurrently for minimum latency
        esl_task = asyncio.ensure_future(
            self._esl_break()
        )
        ai_task = asyncio.ensure_future(
            self._ai_interrupt()
        )
        audio_task = asyncio.ensure_future(
            self._audio.interrupt()
        )

        # Wait for all three (with timeout)
        try:
            await asyncio.wait_for(
                asyncio.gather(esl_task, ai_task, audio_task),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Interruption actions timed out [%s]",
                self.session_id[:8],
            )

        # Step 4: back to LISTENING
        await self._transition(SessionState.LISTENING)

        logger.info(
            "Interruption complete [%s] → LISTENING",
            self.session_id[:8],
        )

    async def _esl_break(self) -> None:
        """Send uuid_break to FreeSWITCH to stop audio playback."""
        if not self._esl.is_connected:
            logger.warning(
                "ESL not connected, cannot send uuid_break [%s]",
                self.session_id[:8],
            )
            return
        try:
            ok = await self._esl.uuid_break(self.call_uuid, all=True)
            if ok:
                logger.debug(
                    "uuid_break sent [%s]", self.session_id[:8]
                )
            else:
                logger.warning(
                    "uuid_break failed [%s]", self.session_id[:8]
                )
        except Exception as e:
            logger.error(
                "uuid_break error [%s]: %s",
                self.session_id[:8], e,
            )

    async def _ai_interrupt(self) -> None:
        """Send interrupt signal to MiniCPM-o."""
        if self._ai and self._ai.is_ready:
            try:
                await self._ai.interrupt()
            except Exception as e:
                logger.error(
                    "AI interrupt error [%s]: %s",
                    self.session_id[:8], e,
                )

    # -----------------------------------------------------------------------
    # Watchdog: idle timeout and max duration enforcement
    # -----------------------------------------------------------------------

    async def _watchdog_loop(self) -> None:
        """
        Background task monitoring:
          - Session idle timeout (no audio activity)
          - Max session duration
          - AI health (is MiniCPM still connected?)
        """
        logger.debug("Watchdog started [%s]", self.session_id[:8])
        check_interval = 10.0  # seconds

        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(check_interval)

                if self._state in (
                    SessionState.TERMINATING,
                    SessionState.TERMINATED,
                ):
                    break

                now = time.monotonic()

                # Check max duration
                duration = time.time() - self._start_time
                if duration >= self._cfg.max_duration_s:
                    logger.warning(
                        "Session max duration reached (%.0fs) [%s]",
                        duration, self.session_id[:8],
                    )
                    await self.terminate("max_duration")
                    break

                # Check idle timeout
                idle = now - self._last_activity
                if idle >= self._cfg.idle_timeout_s:
                    logger.warning(
                        "Session idle timeout (%.0fs idle) [%s]",
                        idle, self.session_id[:8],
                    )
                    await self.terminate("idle_timeout")
                    break

                # Check AI health
                if self._ai and not self._ai.is_ready:
                    if self._state not in (
                        SessionState.INITIALIZING,
                        SessionState.TERMINATING,
                        SessionState.TERMINATED,
                    ):
                        logger.warning(
                            "AI client not ready (state=%s) [%s]",
                            self._ai.state.name,
                            self.session_id[:8],
                        )

                logger.debug(
                    "Watchdog OK [%s]: state=%s idle=%.0fs "
                    "duration=%.0fs interruptions=%d",
                    self.session_id[:8],
                    self._state.name,
                    idle,
                    duration,
                    self._interruptions,
                )

        except asyncio.CancelledError:
            logger.debug("Watchdog cancelled [%s]", self.session_id[:8])
            raise
        except Exception as e:
            logger.error(
                "Watchdog error [%s]: %s",
                self.session_id[:8], e, exc_info=True,
            )

    # -----------------------------------------------------------------------
    # Public control API (used by api.py and main.py)
    # -----------------------------------------------------------------------

    async def inject_message(self, text: str) -> bool:
        """
        Inject a text message into the conversation mid-call.
        E.g. for supervisor whisper, context injection.
        """
        if not self._ai or not self._ai.is_ready:
            return False
        if self._state in (SessionState.TERMINATING, SessionState.TERMINATED):
            return False
        try:
            await self._ai.send_text_message(text)
            logger.info(
                "Text injected [%s]: %s", self.session_id[:8], text[:60]
            )
            return True
        except Exception as e:
            logger.error(
                "inject_message error [%s]: %s",
                self.session_id[:8], e,
            )
            return False

    async def update_system_prompt(self, new_prompt: str) -> bool:
        """Update the system prompt mid-call."""
        if not self._ai or not self._ai.is_ready:
            return False
        try:
            await self._ai.update_session({"instructions": new_prompt})
            self._cfg.system_prompt = new_prompt
            logger.info(
                "System prompt updated [%s]", self.session_id[:8]
            )
            return True
        except Exception as e:
            logger.error(
                "update_system_prompt error [%s]: %s",
                self.session_id[:8], e,
            )
            return False

    async def hold(self) -> bool:
        """Put call on hold."""
        if not self._esl.is_connected:
            return False
        ok = await self._esl.uuid_hold(self.call_uuid)
        if ok:
            logger.info("Call held [%s]", self.session_id[:8])
        return ok

    async def unhold(self) -> bool:
        """Resume call from hold."""
        if not self._esl.is_connected:
            return False
        ok = await self._esl.uuid_unhold(self.call_uuid)
        if ok:
            logger.info("Call unheld [%s]", self.session_id[:8])
        return ok

    async def transfer(
        self,
        destination: str,
        context: str = "default",
    ) -> bool:
        """Transfer call to another destination."""
        if not self._esl.is_connected:
            return False
        ok = await self._esl.uuid_transfer(
            self.call_uuid,
            destination=destination,
            context=context,
        )
        if ok:
            logger.info(
                "Call transferred [%s] → %s",
                self.session_id[:8], destination,
            )
            await self.terminate("transferred")
        return ok

    async def start_recording(self, file_path: str) -> bool:
        """Start call recording."""
        if not self._esl.is_connected:
            return False
        return await self._esl.uuid_record(
            self.call_uuid, file_path, action="start"
        )

    async def stop_recording(self, file_path: str) -> bool:
        """Stop call recording."""
        if not self._esl.is_connected:
            return False
        return await self._esl.uuid_record(
            self.call_uuid, file_path, action="stop"
        )

    # -----------------------------------------------------------------------
    # Properties / info
    # -----------------------------------------------------------------------

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state not in (
            SessionState.TERMINATING,
            SessionState.TERMINATED,
        )

    @property
    def duration_seconds(self) -> float:
        return time.time() - self._start_time

    def get_info(self) -> SessionInfo:
        return SessionInfo(
            session_id=self.session_id,
            call_uuid=self.call_uuid,
            state=self._state.name,
            direction=self._cfg.direction,
            caller_id_num=self._caller_id_num,
            caller_id_name=self._caller_id_name,
            destination=self._destination,
            start_time=self._start_time,
            duration_seconds=self.duration_seconds,
            interruptions=self._interruptions,
            utterances=self._utterances,
            ai_responses=self._ai_responses,
            ai_state=self._ai.state.name if self._ai else "N/A",
            is_bot_speaking=self._audio.is_bot_speaking,
            is_user_speaking=self._audio.is_user_speaking,
            last_activity_at=self._last_activity,
        )


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Manages all active call sessions.

    Responsibilities:
      - Creates Session objects when AudioServer gets a new connection
      - Routes ESL call-control events to the right Session
      - Cleans up terminated sessions
      - Provides system-wide session enumeration for the API

    One SessionManager per application instance.
    """

    def __init__(
        self,
        audio_server:  AudioServer,
        esl_client:    ESLClient,
        cfg:           "Any",                 # AppConfig
        metrics:       Optional[MetricsRegistry] = None,
        session_cfg_factory: Optional[
            Callable[["AudioStream"], SessionConfig]
        ] = None,
    ):
        """
        Args:
            audio_server:       The AudioServer (handles WS connections)
            esl_client:         Shared ESL connection for call control
            cfg:                AppConfig
            metrics:            Global metrics registry
            session_cfg_factory: Optional factory to build per-call SessionConfig
                                 from AudioStream metadata. If None, uses defaults.
        """
        self._audio     = audio_server
        self._esl       = esl_client
        self._cfg       = cfg
        self._metrics   = metrics
        self._session_cfg_factory = session_cfg_factory

        # Active sessions: session_id → Session
        self._sessions:      Dict[str, Session] = {}
        self._sessions_lock  = asyncio.Lock()

        # UUID → session_id mapping (for ESL event routing)
        self._uuid_to_session: Dict[str, str] = {}

        # Completed session history (ring buffer for API)
        self._completed: List[Dict] = []
        self._max_history = 100

        # Wire up AudioServer callbacks
        self._audio._on_connected    = self._on_audio_connected
        self._audio._on_disconnected = self._on_audio_disconnected

        # Wire up ESL event handlers
        self._setup_esl_handlers()

        logger.info("SessionManager initialized")

    def _setup_esl_handlers(self) -> None:
        """Register ESL event handlers for call lifecycle."""
        if not self._esl.is_connected:
            logger.warning(
                "ESL not connected at SessionManager init — "
                "handlers will be registered after connection"
            )
            return
        self._register_esl_handlers()

    def _register_esl_handlers(self) -> None:
        """Register all ESL event handlers."""
        d = self._esl.dispatcher

        d.on_event("CHANNEL_HANGUP_COMPLETE", self._on_channel_hangup)
        d.on_event("CHANNEL_ANSWER",          self._on_channel_answer)
        d.on_event("DTMF",                    self._on_dtmf)

        logger.info("ESL event handlers registered")

    # -----------------------------------------------------------------------
    # AudioServer callbacks
    # -----------------------------------------------------------------------

    async def _on_audio_connected(self, stream: AudioStream) -> None:
        """
        Called when mod_audio_stream WebSocket connects and sends metadata.
        Create and start a Session for this call.
        """
        call_uuid  = stream.call_uuid
        session_id = str(uuid.uuid4())

        logger.info(
            "New call connected: uuid=%s session=%s",
            call_uuid[:8], session_id[:8],
        )

        # Build per-session config
        if self._session_cfg_factory:
            session_cfg = self._session_cfg_factory(stream)
        else:
            session_cfg = self._default_session_config(stream)

        # Create session
        session = Session(
            session_id=session_id,
            audio_stream=stream,
            esl_client=self._esl,
            cfg=self._cfg,
            session_cfg=session_cfg,
            metrics=self._metrics,
            on_terminated=self._on_session_terminated,
        )

        # Register
        async with self._sessions_lock:
            self._sessions[session_id]       = session
            self._uuid_to_session[call_uuid] = session_id

        # Start session (connects AI, sends greeting, etc.)
        try:
            await session.start()
        except Exception as e:
            logger.error(
                "Session start failed [%s]: %s",
                session_id[:8], e, exc_info=True,
            )
            await session.terminate("start_failed")

    async def _on_audio_disconnected(self, stream: AudioStream) -> None:
        """
        Called when mod_audio_stream WebSocket closes.
        Terminate the corresponding session.
        """
        call_uuid = stream.call_uuid
        session   = self._get_session_by_uuid(call_uuid)

        if session is None:
            logger.debug(
                "Audio disconnected but no session found: %s",
                call_uuid[:8] if call_uuid else "?"
            )
            return

        logger.info(
            "Audio disconnected → terminating session [%s]",
            session.session_id[:8],
        )

        if session.is_active:
            await session.terminate("audio_disconnected")

    # -----------------------------------------------------------------------
    # ESL event handlers
    # -----------------------------------------------------------------------

    async def _on_channel_hangup(self, event: "Any") -> None:
        """FreeSWITCH channel hung up."""
        call_uuid = event.unique_id
        cause     = event.get("Hangup-Cause", "UNKNOWN")

        logger.info(
            "Channel hangup: uuid=%s cause=%s",
            call_uuid[:8], cause,
        )

        session = self._get_session_by_uuid(call_uuid)
        if session and session.is_active:
            await session.terminate(f"hangup_{cause}")

    async def _on_channel_answer(self, event: "Any") -> None:
        """FreeSWITCH channel answered."""
        call_uuid = event.unique_id
        logger.info(
            "Channel answered: uuid=%s", call_uuid[:8]
        )
        # Session should already be initializing at this point
        session = self._get_session_by_uuid(call_uuid)
        if session:
            logger.debug(
                "Channel answer → session state=%s",
                session.state.name,
            )

    async def _on_dtmf(self, event: "Any") -> None:
        """DTMF digit received during call."""
        call_uuid = event.unique_id
        digit     = event.get("DTMF-Digit", "")
        duration  = event.get("DTMF-Duration", "")

        logger.info(
            "DTMF: uuid=%s digit=%s duration=%s",
            call_uuid[:8], digit, duration,
        )

        session = self._get_session_by_uuid(call_uuid)
        if session and digit:
            # Inject DTMF as text context to AI
            await session.inject_message(
                f"[User pressed DTMF key: {digit}]"
            )

    # -----------------------------------------------------------------------
    # Session terminated callback
    # -----------------------------------------------------------------------

    async def _on_session_terminated(self, session: Session) -> None:
        """Called when a session finishes terminating."""
        async with self._sessions_lock:
            self._sessions.pop(session.session_id, None)
            self._uuid_to_session.pop(session.call_uuid, None)

        # Archive to history
        info = session.get_info()
        self._completed.append(info.to_dict())
        if len(self._completed) > self._max_history:
            self._completed.pop(0)

        logger.info(
            "Session removed from manager: %s (active=%d)",
            session.session_id[:8],
            len(self._sessions),
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_session_by_uuid(self, call_uuid: str) -> Optional[Session]:
        """Look up a Session by FreeSWITCH channel UUID."""
        session_id = self._uuid_to_session.get(call_uuid)
        if session_id:
            return self._sessions.get(session_id)
        return None

    def _default_session_config(self, stream: AudioStream) -> SessionConfig:
        """Build default SessionConfig from AppConfig + stream metadata."""
        m = self._cfg.minicpm
        i = self._cfg.interruption
        meta = stream.metadata

        return SessionConfig(
            system_prompt=m.system_prompt,
            language=m.language,
            temperature=m.temperature,
            voice_id=m.voice_id,
            greeting_text="",
            interruption_enabled=i.enabled,
            interruption_cooldown_s=i.cooldown_ms / 1000.0,
            grace_period_ms=float(i.grace_period_ms),
            idle_timeout_s=SESSION_IDLE_TIMEOUT,
            max_duration_s=SESSION_MAX_DURATION,
            ai_connect_timeout_s=AI_CONNECT_TIMEOUT,
            direction=meta.direction if meta else "inbound",
        )

    # -----------------------------------------------------------------------
    # Public API (used by api.py)
    # -----------------------------------------------------------------------

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_session_by_uuid(self, call_uuid: str) -> Optional[Session]:
        return self._get_session_by_uuid(call_uuid)

    def list_sessions(self) -> List[SessionInfo]:
        return [s.get_info() for s in self._sessions.values()]

    def list_completed(self, limit: int = 20) -> List[Dict]:
        return self._completed[-limit:]

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    async def terminate_session(
        self,
        session_id: str,
        reason: str = "api_request",
    ) -> bool:
        """Terminate a session by ID (called from REST API)."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        await session.terminate(reason)
        return True

    async def terminate_all(self, reason: str = "shutdown") -> None:
        """Terminate all active sessions (called on shutdown)."""
        async with self._sessions_lock:
            sessions = list(self._sessions.values())

        logger.info(
            "Terminating all %d sessions (reason=%s)",
            len(sessions), reason,
        )

        await asyncio.gather(
            *[s.terminate(reason) for s in sessions],
            return_exceptions=True,
        )

    async def inject_message(
        self, session_id: str, text: str
    ) -> bool:
        """Inject text message into a session (supervisor feature)."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        return await session.inject_message(text)

    async def transfer_call(
        self,
        session_id:  str,
        destination: str,
        context:     str = "default",
    ) -> bool:
        """Transfer a call to another destination."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        return await session.transfer(destination, context)

    def get_stats(self) -> Dict:
        """System-wide session statistics."""
        sessions = list(self._sessions.values())
        return {
            "active_sessions":    len(sessions),
            "completed_sessions": len(self._completed),
            "sessions": [
                s.get_info().to_dict() for s in sessions
            ],
        }

class OutboundCall:
    """
    外呼任务：追踪一次外呼从发起到接听的完整状态。
    """

    class Status(Enum):
        PENDING    = auto()   # 等待发起
        RINGING    = auto()   # 振铃中
        ANSWERED   = auto()   # 已接听，等待 audio_stream 连接
        CONNECTED  = auto()   # audio_stream 已连接，Session 已创建
        NO_ANSWER  = auto()   # 无人接听
        BUSY       = auto()   # 占线
        FAILED     = auto()   # 失败
        COMPLETED  = auto()   # 正常结束

    def __init__(
        self,
        outbound_id:   str,
        destination:   str,
        caller_id_num: str,
        session_cfg:   "SessionConfig",
        max_retries:   int = 0,
    ):
        self.outbound_id   = outbound_id
        self.destination   = destination
        self.caller_id_num = caller_id_num
        self.session_cfg   = session_cfg
        self.max_retries   = max_retries

        self.status        = OutboundCall.Status.PENDING
        self.call_uuid:    Optional[str]    = None
        self.session:      Optional[Session] = None
        self.attempt       = 0
        self.created_at    = time.time()
        self.answered_at:  Optional[float]  = None
        self.ended_at:     Optional[float]  = None
        self.hangup_cause: str = ""

        # Future that resolves when call is answered or fails
        self._answer_future: Optional[asyncio.Future] = None

    @property
    def duration_seconds(self) -> float:
        if self.answered_at and self.ended_at:
            return self.ended_at - self.answered_at
        return 0.0

    def to_dict(self) -> Dict:
        return {
            "outbound_id":      self.outbound_id,
            "destination":      self.destination,
            "caller_id_num":    self.caller_id_num,
            "status":           self.status.name,
            "call_uuid":        self.call_uuid,
            "attempt":          self.attempt,
            "created_at":       self.created_at,
            "answered_at":      self.answered_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "hangup_cause":     self.hangup_cause,
        }


class OutboundCallManager:
    """
    管理外呼生命周期。

    外呼流程：
      1. place_call() → ESL originate → FS开始振铃
      2. 监听 CHANNEL_ANSWER 事件 → 得知被叫接听
      3. 执行 uuid_transfer 将通话送入 audio_stream
      4. AudioServer 收到 WebSocket 连接 → 创建 Session
      5. Session 开始正常工作（和呼入一样）
      6. 监听 CHANNEL_HANGUP 事件 → 清理
    """

    def __init__(
        self,
        esl:             "ESLClient",
        session_manager: "SessionManager",
        cfg:             "Any",
        audio_ws_url:    str = "ws://127.0.0.1:8765",
    ):
        self._esl            = esl
        self._sm             = session_manager
        self._cfg            = cfg
        self._audio_ws_url   = audio_ws_url

        # Active outbound calls
        self._calls:      Dict[str, OutboundCall] = {}   # outbound_id → call
        self._uuid_map:   Dict[str, str] = {}            # call_uuid → outbound_id
        self._lock        = asyncio.Lock()

        # Register ESL handlers
        self._esl.dispatcher.on_event(
            "CHANNEL_ANSWER", self._on_channel_answer
        )
        self._esl.dispatcher.on_event(
            "CHANNEL_HANGUP_COMPLETE", self._on_channel_hangup
        )

        logger.info("OutboundCallManager initialized")

    async def place_call(
        self,
        destination:     str,
        caller_id_num:   str                     = "8000",
        caller_id_name:  str                     = "VoiceBot",
        gateway:         str                     = "default",
        session_cfg:     Optional["SessionConfig"] = None,
        ring_timeout:    int                     = 30,
        max_retries:     int                     = 0,
        retry_delay:     float                   = 5.0,
    ) -> OutboundCall:
        """
        发起一次外呼。

        Args:
            destination:   被叫号码
            caller_id_num: 主叫号码
            caller_id_name:主叫名称
            gateway:       SIP网关
            session_cfg:   AI会话配置（使用哪个prompt等）
            ring_timeout:  振铃超时秒数
            max_retries:   最大重试次数（无人接听时）
            retry_delay:   重试间隔秒数

        Returns:
            OutboundCall 对象（立即返回，不等待接听）
        """
        outbound_id = str(uuid.uuid4())

        call = OutboundCall(
            outbound_id=outbound_id,
            destination=destination,
            caller_id_num=caller_id_num,
            session_cfg=session_cfg or self._default_session_cfg(),
            max_retries=max_retries,
        )

        async with self._lock:
            self._calls[outbound_id] = call

        logger.info(
            "Placing outbound call: id=%s dest=%s gw=%s",
            outbound_id[:8], destination, gateway,
        )

        # Start the call attempt in background
        asyncio.ensure_future(
            self._attempt_call(
                call, gateway, ring_timeout, retry_delay
            )
        )

        return call

    async def _attempt_call(
        self,
        call:         OutboundCall,
        gateway:      str,
        ring_timeout: int,
        retry_delay:  float,
    ) -> None:
        """内部：执行一次呼叫尝试（含重试循环）。"""
        while call.attempt <= call.max_retries:
            call.attempt += 1
            call.status   = OutboundCall.Status.RINGING

            logger.info(
                "Outbound attempt %d/%d: %s → %s",
                call.attempt,
                call.max_retries + 1,
                call.caller_id_num,
                call.destination,
            )

            # Step 1: ESL originate
            call_uuid = await self._esl.originate(
                destination=call.destination,
                caller_id_num=call.caller_id_num,
                gateway=gateway,
                timeout=ring_timeout,
                variables={
                    "voicebot_outbound_id": call.outbound_id,
                },
            )

            if not call_uuid:
                call.status      = OutboundCall.Status.FAILED
                call.hangup_cause = "ORIGINATE_FAILED"
                logger.error(
                    "Originate failed: %s", call.destination
                )
                break

            call.call_uuid = call_uuid

            async with self._lock:
                self._uuid_map[call_uuid] = call.outbound_id

            # Step 2: Wait for answer or timeout
            loop    = asyncio.get_event_loop()
            future: asyncio.Future = loop.create_future()
            call._answer_future = future

            try:
                answered = await asyncio.wait_for(
                    future, timeout=float(ring_timeout + 5)
                )
            except asyncio.TimeoutError:
                answered = False
                logger.warning(
                    "Ring timeout: %s (%.0fs)",
                    call.destination, ring_timeout,
                )

            if answered:
                # Step 3: Transfer into audio_stream
                await self._connect_audio_stream(call)
                return  # success — session will handle the rest
            else:
                # No answer — retry?
                call.status = OutboundCall.Status.NO_ANSWER
                if call.attempt <= call.max_retries:
                    logger.info(
                        "Retrying in %.0fs: %s (attempt %d/%d)",
                        retry_delay, call.destination,
                        call.attempt, call.max_retries + 1,
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.warning(
                        "No answer after %d attempts: %s",
                        call.attempt, call.destination,
                    )
                    break

        # All attempts exhausted
        if call.status not in (
            OutboundCall.Status.CONNECTED,
            OutboundCall.Status.COMPLETED,
        ):
            call.status  = OutboundCall.Status.FAILED
            call.ended_at = time.time()
            async with self._lock:
                if call.call_uuid:
                    self._uuid_map.pop(call.call_uuid, None)

    async def _connect_audio_stream(self, call: OutboundCall) -> None:
        """
        被叫接听后，将通话转入 audio_stream application。
        这会触发 mod_audio_stream 连接到我们的 WebSocket。
        """
        call.status      = OutboundCall.Status.ANSWERED
        call.answered_at = time.time()

        logger.info(
            "Outbound answered: %s → connecting audio_stream",
            call.destination,
        )

        # 设置通话变量（让 Session 知道这是外呼）
        await self._esl.uuid_setvar_multi(
            call.call_uuid,
            {
                "voicebot_direction":   "outbound",
                "voicebot_outbound_id": call.outbound_id,
            }
        )

        # 执行 audio_stream application
        # 这会让 mod_audio_stream 连接到我们的 WebSocket
        result = await self._esl.uuid_execute(
            call.call_uuid,
            app="audio_stream",
            arg=f"{self._audio_ws_url} 16000 1",
            event_lock=False,
        )

        logger.info(
            "audio_stream executed for outbound call: %s",
            call.call_uuid[:8],
        )

        # AudioServer 会在 WebSocket 连接时创建 Session
        # Session 创建完成后 call.status 会变为 CONNECTED
        call.status = OutboundCall.Status.CONNECTED

    async def _on_channel_answer(self, event: "Any") -> None:
        """ESL事件：被叫接听。"""
        call_uuid  = event.unique_id
        outbound_id = self._uuid_map.get(call_uuid)
        if not outbound_id:
            return  # 不是我们发起的外呼

        call = self._calls.get(outbound_id)
        if not call:
            return

        logger.info(
            "Outbound call answered: uuid=%s dest=%s",
            call_uuid[:8], call.destination,
        )

        # Resolve the answer future
        if call._answer_future and not call._answer_future.done():
            call._answer_future.set_result(True)

    async def _on_channel_hangup(self, event: "Any") -> None:
        """ESL事件：通话挂断。"""
        call_uuid   = event.unique_id
        outbound_id = self._uuid_map.get(call_uuid)
        if not outbound_id:
            return

        call = self._calls.get(outbound_id)
        if not call:
            return

        cause = event.get("Hangup-Cause", "UNKNOWN")
        call.hangup_cause = cause
        call.ended_at     = time.time()
        call.status       = OutboundCall.Status.COMPLETED

        # Resolve answer future if still pending (means no answer)
        if call._answer_future and not call._answer_future.done():
            if cause in ("NO_ANSWER", "USER_BUSY", "CALL_REJECTED"):
                call._answer_future.set_result(False)
            else:
                call._answer_future.set_result(False)

        logger.info(
            "Outbound call ended: %s cause=%s duration=%.1fs",
            call.destination, cause, call.duration_seconds,
        )

        async with self._lock:
            self._uuid_map.pop(call_uuid, None)

    def _default_session_cfg(self) -> "SessionConfig":
        from session import SessionConfig
        m = self._cfg.minicpm
        return SessionConfig(
            system_prompt=m.system_prompt,
            language=m.language,
            temperature=m.temperature,
            direction="outbound",
        )

    def get_call(self, outbound_id: str) -> Optional[OutboundCall]:
        return self._calls.get(outbound_id)

    def list_calls(self) -> List[OutboundCall]:
        return list(self._calls.values())

    async def cancel_call(self, outbound_id: str) -> bool:
        """取消一个外呼（振铃中或已接听）。"""
        call = self._calls.get(outbound_id)
        if not call or not call.call_uuid:
            return False
        await self._esl.uuid_kill(call.call_uuid)
        return True
