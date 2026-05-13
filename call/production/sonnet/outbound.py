"""
outbound.py - Outbound call management for VoiceBot

This is the core of the outbound calling system.

Handles:
  - Single outbound call lifecycle
  - Campaign (batch) outbound calling
  - Concurrency limiting (max N simultaneous calls)
  - Rate limiting (calls per second/minute)
  - Retry logic (no answer, busy, network error)
  - Answer detection (human vs. AMD - answering machine)
  - Call result tracking and reporting
  - Webhook callbacks on call completion
  - Priority queuing
  - DNC (Do Not Call) list checking
  - Schedule-aware dialing (business hours)

Outbound call state machine:
  PENDING
    │ place()
    ▼
  DIALING ──────────────────────────────────┐
    │ CHANNEL_PROGRESS                       │
    ▼                                        │
  RINGING ──────────────────────────────────┤
    │ CHANNEL_ANSWER                         │
    ▼                                        │
  ANSWERED                                   │
    │ uuid_execute(audio_stream)             │
    ▼                                        │
  CONNECTING                                 │
    │ AudioServer WebSocket connected        │
    ▼                                        │
  CONNECTED                                  │
    │ Session terminates                     │
    ▼                                        │
  COMPLETED          FAILED ◄────────────────┘
                     NO_ANSWER
                     BUSY
                     CANCELLED
"""

import asyncio
import logging
import time
import uuid
import json
import aiohttp
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from enum import Enum, auto
from typing import (
    Any, Callable, Coroutine, Dict,
    List, Optional, Set, Tuple
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default ring timeout (seconds)
DEFAULT_RING_TIMEOUT        = 30
# Default max retry attempts
DEFAULT_MAX_RETRIES         = 2
# Default delay between retries (seconds)
DEFAULT_RETRY_DELAY         = 60.0
# Max time to wait for audio_stream WebSocket after answer (seconds)
AUDIO_CONNECT_TIMEOUT       = 15.0
# How long to wait for ESL originate command (seconds)
ORIGINATE_TIMEOUT           = 60.0
# Interval for campaign worker to check queue (seconds)
CAMPAIGN_POLL_INTERVAL      = 0.5
# Max calls in a single campaign batch
MAX_CAMPAIGN_SIZE           = 10_000
# Webhook HTTP timeout (seconds)
WEBHOOK_TIMEOUT             = 10.0
# Max webhook retries
WEBHOOK_MAX_RETRIES         = 3


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CallState(Enum):
    """Outbound call lifecycle states."""
    PENDING      = "pending"       # In queue, not yet dialed
    DIALING      = "dialing"       # ESL originate sent
    RINGING      = "ringing"       # SIP ringing at destination
    ANSWERED     = "answered"      # Call answered, connecting audio
    CONNECTING   = "connecting"    # Waiting for mod_audio_stream WS
    CONNECTED    = "connected"     # AI session active
    COMPLETED    = "completed"     # Call ended normally
    NO_ANSWER    = "no_answer"     # Ring timeout, no answer
    BUSY         = "busy"          # Destination busy
    FAILED       = "failed"        # Technical failure
    CANCELLED    = "cancelled"     # Cancelled before answer
    REJECTED     = "rejected"      # DNC list or invalid number


class HangupCause(str, Enum):
    """FreeSWITCH hangup cause codes we care about."""
    NORMAL_CLEARING    = "NORMAL_CLEARING"
    NO_ANSWER          = "NO_ANSWER"
    USER_BUSY          = "USER_BUSY"
    CALL_REJECTED      = "CALL_REJECTED"
    UNALLOCATED_NUMBER = "UNALLOCATED_NUMBER"
    NO_ROUTE_DEST      = "NO_ROUTE_DESTINATION"
    NETWORK_OUT_OF_ORDER = "NETWORK_OUT_OF_ORDER"
    ORIGINATOR_CANCEL  = "ORIGINATOR_CANCEL"
    LOSE_RACE          = "LOSE_RACE"
    UNKNOWN            = "UNKNOWN"

    @classmethod
    def from_str(cls, s: str) -> "HangupCause":
        try:
            return cls(s)
        except ValueError:
            return cls.UNKNOWN

    @property
    def is_retryable(self) -> bool:
        """Should we retry when call ends with this cause?"""
        return self in (
            cls.NO_ANSWER,
            cls.NETWORK_OUT_OF_ORDER,
        ) if (cls := HangupCause) else False

    @property
    def is_busy(self) -> bool:
        return self == HangupCause.USER_BUSY

    @property
    def is_invalid(self) -> bool:
        return self in (
            HangupCause.UNALLOCATED_NUMBER,
            HangupCause.NO_ROUTE_DEST,
        )


class CampaignState(Enum):
    """Campaign lifecycle states."""
    CREATED   = "created"
    RUNNING   = "running"
    PAUSED    = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Priority(int, Enum):
    """Call priority levels (lower number = higher priority)."""
    CRITICAL = 1
    HIGH     = 2
    NORMAL   = 3
    LOW      = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    """
    Complete record of an outbound call attempt.
    Immutable snapshot for reporting/webhook.
    """
    outbound_id:    str
    campaign_id:    Optional[str]
    destination:    str
    caller_id_num:  str
    caller_id_name: str
    call_uuid:      Optional[str]
    state:          CallState
    hangup_cause:   str
    attempt:        int
    max_retries:    int
    created_at:     float
    dialed_at:      Optional[float]
    answered_at:    Optional[float]
    connected_at:   Optional[float]
    ended_at:       Optional[float]
    session_id:     Optional[str]
    metadata:       Dict[str, Any]
    error_message:  str

    @property
    def ring_duration_seconds(self) -> float:
        if self.dialed_at and self.answered_at:
            return self.answered_at - self.dialed_at
        return 0.0

    @property
    def call_duration_seconds(self) -> float:
        if self.answered_at and self.ended_at:
            return self.ended_at - self.answered_at
        return 0.0

    @property
    def total_duration_seconds(self) -> float:
        if self.created_at and self.ended_at:
            return self.ended_at - self.created_at
        return 0.0

    def to_dict(self) -> Dict:
        return {
            "outbound_id":           self.outbound_id,
            "campaign_id":           self.campaign_id,
            "destination":           self.destination,
            "caller_id_num":         self.caller_id_num,
            "caller_id_name":        self.caller_id_name,
            "call_uuid":             self.call_uuid,
            "state":                 self.state.value,
            "hangup_cause":          self.hangup_cause,
            "attempt":               self.attempt,
            "max_retries":           self.max_retries,
            "created_at":            self.created_at,
            "dialed_at":             self.dialed_at,
            "answered_at":           self.answered_at,
            "connected_at":          self.connected_at,
            "ended_at":              self.ended_at,
            "session_id":            self.session_id,
            "metadata":              self.metadata,
            "error_message":         self.error_message,
            "ring_duration_seconds": round(self.ring_duration_seconds, 2),
            "call_duration_seconds": round(self.call_duration_seconds, 2),
            "total_duration_seconds":round(self.total_duration_seconds, 2),
        }


@dataclass
class OutboundCallConfig:
    """
    Configuration for a single outbound call.
    """
    # Destination
    destination:    str
    caller_id_num:  str   = "8000"
    caller_id_name: str   = "VoiceBot"
    gateway:        str   = "default"

    # Retry
    ring_timeout:   int   = DEFAULT_RING_TIMEOUT
    max_retries:    int   = DEFAULT_MAX_RETRIES
    retry_delay:    float = DEFAULT_RETRY_DELAY

    # Priority
    priority:       Priority = Priority.NORMAL

    # AI session
    system_prompt:  str   = ""
    greeting_text:  str   = ""
    language:       str   = "zh-CN"
    temperature:    float = 0.7
    voice_id:       str   = "default"

    # Campaign
    campaign_id:    Optional[str] = None

    # Webhook: called when call ends
    webhook_url:    Optional[str] = None
    webhook_secret: Optional[str] = None

    # Custom metadata (passed through to webhook/records)
    metadata:       Dict[str, Any] = field(default_factory=dict)

    # Scheduling
    scheduled_at:   Optional[float] = None   # Unix timestamp, None=immediate

    def __post_init__(self):
        if not self.destination:
            raise ValueError("destination is required")


@dataclass
class CampaignConfig:
    """
    Configuration for a batch outbound campaign.
    """
    name:                str
    calls:               List[OutboundCallConfig]

    # Concurrency
    max_concurrent:      int   = 5    # max simultaneous calls
    calls_per_minute:    float = 30.0 # rate limit

    # Default values applied to all calls (overridden by per-call config)
    default_caller_id_num:  str = "8000"
    default_caller_id_name: str = "VoiceBot"
    default_gateway:        str = "default"
    default_ring_timeout:   int = 30
    default_max_retries:    int = 2
    default_retry_delay:    float = 60.0
    default_system_prompt:  str = ""
    default_greeting_text:  str = ""

    # Business hours restriction (None = no restriction)
    business_hours_start:   Optional[dt_time] = None  # e.g. time(9, 0)
    business_hours_end:     Optional[dt_time] = None  # e.g. time(18, 0)
    business_days:          List[int] = field(
        default_factory=lambda: [0,1,2,3,4]  # Mon-Fri
    )

    # Webhook for campaign-level events
    webhook_url:            Optional[str] = None

    def __post_init__(self):
        if not self.calls:
            raise ValueError("Campaign must have at least one call")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if self.calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be > 0")


# ---------------------------------------------------------------------------
# OutboundCall: single call state machine
# ---------------------------------------------------------------------------

class OutboundCall:
    """
    Manages the lifecycle of a single outbound call.

    State transitions are strictly controlled.
    Thread-safe via asyncio locks.
    """

    # Valid state transitions
    _TRANSITIONS: Dict[CallState, Set[CallState]] = {
        CallState.PENDING:    {CallState.DIALING,    CallState.CANCELLED, CallState.REJECTED},
        CallState.DIALING:    {CallState.RINGING,    CallState.FAILED,    CallState.NO_ANSWER,
                               CallState.BUSY,        CallState.CANCELLED},
        CallState.RINGING:    {CallState.ANSWERED,   CallState.NO_ANSWER, CallState.BUSY,
                               CallState.FAILED,      CallState.CANCELLED},
        CallState.ANSWERED:   {CallState.CONNECTING, CallState.FAILED,    CallState.COMPLETED},
        CallState.CONNECTING: {CallState.CONNECTED,  CallState.FAILED},
        CallState.CONNECTED:  {CallState.COMPLETED,  CallState.FAILED},
        CallState.COMPLETED:  set(),
        CallState.NO_ANSWER:  {CallState.PENDING},   # retry
        CallState.BUSY:       {CallState.PENDING},   # retry
        CallState.FAILED:     {CallState.PENDING},   # retry
        CallState.CANCELLED:  set(),
        CallState.REJECTED:   set(),
    }

    def __init__(
        self,
        cfg:          OutboundCallConfig,
        outbound_id:  Optional[str] = None,
    ):
        self.outbound_id   = outbound_id or str(uuid.uuid4())
        self.cfg           = cfg

        # State
        self._state        = CallState.PENDING
        self._state_lock   = asyncio.Lock()

        # Runtime info
        self.call_uuid:    Optional[str] = None
        self.session_id:   Optional[str] = None
        self.attempt:      int   = 0
        self.hangup_cause: str   = ""
        self.error_message:str   = ""

        # Timestamps
        self.created_at:    float          = time.time()
        self.dialed_at:     Optional[float] = None
        self.answered_at:   Optional[float] = None
        self.connected_at:  Optional[float] = None
        self.ended_at:      Optional[float] = None

        # Futures
        self._answer_future:  Optional[asyncio.Future] = None
        self._connect_future: Optional[asyncio.Future] = None

        # Callbacks
        self._on_state_change: Optional[
            Callable[["OutboundCall", CallState, CallState], Coroutine]
        ] = None

        logger.debug(
            "OutboundCall created: id=%s dest=%s",
            self.outbound_id[:8], cfg.destination,
        )

    # -----------------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------------

    @property
    def state(self) -> CallState:
        return self._state

    async def transition(
        self, new_state: CallState, reason: str = ""
    ) -> bool:
        """
        Attempt state transition.
        Returns True if successful, False if invalid transition.
        """
        async with self._state_lock:
            valid = self._TRANSITIONS.get(self._state, set())
            if new_state not in valid:
                logger.warning(
                    "Invalid transition %s → %s [%s] reason=%s",
                    self._state.value, new_state.value,
                    self.outbound_id[:8], reason,
                )
                return False

            old_state  = self._state
            self._state = new_state

            logger.info(
                "Call %s → %s [%s]%s",
                old_state.value, new_state.value,
                self.outbound_id[:8],
                f" ({reason})" if reason else "",
            )

        # Fire state change callback outside lock
        if self._on_state_change:
            try:
                await self._on_state_change(self, old_state, new_state)
            except Exception as e:
                logger.error(
                    "State change callback error [%s]: %s",
                    self.outbound_id[:8], e,
                )
        return True

    def reset_for_retry(self) -> None:
        """Reset mutable state for retry attempt."""
        self.call_uuid      = None
        self.dialed_at      = None
        self.answered_at    = None
        self.connected_at   = None
        self.hangup_cause   = ""
        self._answer_future  = None
        self._connect_future = None

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._state in (
            CallState.DIALING,
            CallState.RINGING,
            CallState.ANSWERED,
            CallState.CONNECTING,
            CallState.CONNECTED,
        )

    @property
    def is_terminal(self) -> bool:
        return self._state in (
            CallState.COMPLETED,
            CallState.CANCELLED,
            CallState.REJECTED,
        )

    @property
    def should_retry(self) -> bool:
        retryable = self._state in (
            CallState.NO_ANSWER,
            CallState.BUSY,
            CallState.FAILED,
        )
        return retryable and self.attempt < self.cfg.max_retries + 1

    @property
    def ring_duration_seconds(self) -> float:
        if self.dialed_at and self.answered_at:
            return self.answered_at - self.dialed_at
        return 0.0

    @property
    def call_duration_seconds(self) -> float:
        if self.answered_at and self.ended_at:
            return self.ended_at - self.answered_at
        return 0.0

    def to_record(self) -> CallRecord:
        return CallRecord(
            outbound_id=self.outbound_id,
            campaign_id=self.cfg.campaign_id,
            destination=self.cfg.destination,
            caller_id_num=self.cfg.caller_id_num,
            caller_id_name=self.cfg.caller_id_name,
            call_uuid=self.call_uuid,
            state=self._state,
            hangup_cause=self.hangup_cause,
            attempt=self.attempt,
            max_retries=self.cfg.max_retries,
            created_at=self.created_at,
            dialed_at=self.dialed_at,
            answered_at=self.answered_at,
            connected_at=self.connected_at,
            ended_at=self.ended_at,
            session_id=self.session_id,
            metadata=self.cfg.metadata,
            error_message=self.error_message,
        )

    def to_dict(self) -> Dict:
        return self.to_record().to_dict()

    def __repr__(self) -> str:
        return (
            f"OutboundCall(id={self.outbound_id[:8]} "
            f"dest={self.cfg.destination} "
            f"state={self._state.value} "
            f"attempt={self.attempt})"
        )


# ---------------------------------------------------------------------------
# Webhook sender
# ---------------------------------------------------------------------------

class WebhookSender:
    """
    Sends HTTP POST webhooks on call events.
    Retries on failure with exponential backoff.
    Supports HMAC-SHA256 signature for verification.
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=WEBHOOK_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def send(
        self,
        url:     str,
        payload: Dict,
        secret:  Optional[str] = None,
    ) -> bool:
        """
        Send webhook with retry.
        Returns True if delivered successfully.
        """
        body = json.dumps(payload, default=str).encode()

        headers = {
            "Content-Type":  "application/json",
            "User-Agent":    "VoiceBot-Webhook/1.0",
            "X-VoiceBot-Event": payload.get("event", "call.updated"),
            "X-Timestamp":   str(int(time.time())),
        }

        # HMAC signature
        if secret:
            import hmac as _hmac
            import hashlib
            sig = _hmac.new(
                secret.encode(),
                body,
                hashlib.sha256,
            ).hexdigest()
            headers["X-Signature-SHA256"] = f"sha256={sig}"

        session = await self._get_session()

        for attempt in range(1, WEBHOOK_MAX_RETRIES + 1):
            try:
                async with session.post(
                    url, data=body, headers=headers
                ) as resp:
                    if resp.status < 300:
                        logger.debug(
                            "Webhook delivered: %s status=%d",
                            url, resp.status,
                        )
                        return True
                    else:
                        logger.warning(
                            "Webhook failed: %s status=%d (attempt %d/%d)",
                            url, resp.status,
                            attempt, WEBHOOK_MAX_RETRIES,
                        )
            except Exception as e:
                logger.warning(
                    "Webhook error: %s error=%s (attempt %d/%d)",
                    url, e, attempt, WEBHOOK_MAX_RETRIES,
                )

            if attempt < WEBHOOK_MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)

        logger.error(
            "Webhook delivery failed after %d attempts: %s",
            WEBHOOK_MAX_RETRIES, url,
        )
        return False

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


# ---------------------------------------------------------------------------
# DNC (Do Not Call) list
# ---------------------------------------------------------------------------

class DNCList:
    """
    In-memory Do Not Call list.
    Numbers on this list are rejected before dialing.
    Can be loaded from file or populated via API.
    """

    def __init__(self):
        self._numbers: Set[str] = set()
        self._lock    = asyncio.Lock()

    async def add(self, number: str) -> None:
        normalized = self._normalize(number)
        async with self._lock:
            self._numbers.add(normalized)
        logger.info("DNC: added %s", normalized)

    async def remove(self, number: str) -> None:
        normalized = self._normalize(number)
        async with self._lock:
            self._numbers.discard(normalized)

    async def check(self, number: str) -> bool:
        """Returns True if number is on DNC list."""
        normalized = self._normalize(number)
        async with self._lock:
            return normalized in self._numbers

    async def load_file(self, path: str) -> int:
        """Load DNC numbers from a text file (one per line)."""
        count = 0
        try:
            with open(path, "r") as f:
                for line in f:
                    num = line.strip()
                    if num and not num.startswith("#"):
                        await self.add(num)
                        count += 1
            logger.info("DNC: loaded %d numbers from %s", count, path)
        except FileNotFoundError:
            logger.warning("DNC file not found: %s", path)
        return count

    def _normalize(self, number: str) -> str:
        """Normalize: strip whitespace, dashes, spaces."""
        return "".join(c for c in number if c.isdigit() or c == "+")

    @property
    def size(self) -> int:
        return len(self._numbers)


# ---------------------------------------------------------------------------
# Rate limiter for outbound calls
# ---------------------------------------------------------------------------

class OutboundRateLimiter:
    """
    Token bucket rate limiter for outbound call dialing.
    Prevents overloading FreeSWITCH or SIP trunk.
    """

    def __init__(self, calls_per_minute: float):
        self._cpm          = calls_per_minute
        self._interval     = 60.0 / calls_per_minute  # seconds per call
        self._last_call_at = 0.0
        self._lock         = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until we can make the next call."""
        async with self._lock:
            now  = time.monotonic()
            wait = self._last_call_at + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_at = time.monotonic()

    def update_rate(self, calls_per_minute: float) -> None:
        self._cpm      = calls_per_minute
        self._interval = 60.0 / calls_per_minute


# ---------------------------------------------------------------------------
# Campaign: batch outbound calling
# ---------------------------------------------------------------------------

class Campaign:
    """
    Manages a batch of outbound calls.

    Features:
      - Priority queue (higher priority calls dial first)
      - Concurrency control (max N simultaneous)
      - Rate limiting (calls per minute)
      - Business hours enforcement
      - Pause/resume
      - Real-time progress tracking
      - Per-call retry scheduling
    """

    def __init__(
        self,
        cfg:             CampaignConfig,
        campaign_id:     Optional[str] = None,
    ):
        self.campaign_id  = campaign_id or str(uuid.uuid4())
        self.cfg          = cfg
        self.name         = cfg.name

        self._state       = CampaignState.CREATED
        self._state_lock  = asyncio.Lock()

        # Priority queue: (priority, created_at, OutboundCall)
        self._queue:      asyncio.PriorityQueue = asyncio.PriorityQueue()
        # Retry queue: (retry_after_timestamp, OutboundCall)
        self._retry_queue: List[Tuple[float, OutboundCall]] = []

        # All calls in this campaign
        self._all_calls:  Dict[str, OutboundCall] = {}

        # Active (currently dialing/connected)
        self._active:     Set[str] = set()  # outbound_ids
        self._active_lock = asyncio.Lock()

        # Concurrency semaphore
        self._semaphore   = asyncio.Semaphore(cfg.max_concurrent)

        # Rate limiter
        self._rate_limiter = OutboundRateLimiter(cfg.calls_per_minute)

        # Stats
        self.created_at   = time.time()
        self.started_at:  Optional[float] = None
        self.ended_at:    Optional[float] = None

        # Tasks
        self._worker_task:  Optional[asyncio.Task] = None
        self._retry_task:   Optional[asyncio.Task] = None
        self._stop_event    = asyncio.Event()

        # Populate queue from config
        self._total = len(cfg.calls)
        self._populate_queue()

        logger.info(
            "Campaign created: id=%s name=%s calls=%d "
            "concurrent=%d rate=%.1f/min",
            self.campaign_id[:8], self.name,
            self._total, cfg.max_concurrent, cfg.calls_per_minute,
        )

    def _populate_queue(self) -> None:
        """Load all calls from config into priority queue."""
        for call_cfg in self.cfg.calls:
            # Apply campaign defaults where call doesn't override
            if not call_cfg.caller_id_num:
                call_cfg.caller_id_num = self.cfg.default_caller_id_num
            if not call_cfg.caller_id_name:
                call_cfg.caller_id_name = self.cfg.default_caller_id_name
            if not call_cfg.gateway:
                call_cfg.gateway = self.cfg.default_gateway
            if call_cfg.ring_timeout == DEFAULT_RING_TIMEOUT:
                call_cfg.ring_timeout = self.cfg.default_ring_timeout
            if call_cfg.max_retries == DEFAULT_MAX_RETRIES:
                call_cfg.max_retries = self.cfg.default_max_retries
            if call_cfg.retry_delay == DEFAULT_RETRY_DELAY:
                call_cfg.retry_delay = self.cfg.default_retry_delay
            if not call_cfg.system_prompt:
                call_cfg.system_prompt = self.cfg.default_system_prompt
            if not call_cfg.greeting_text:
                call_cfg.greeting_text = self.cfg.default_greeting_text
            if not call_cfg.campaign_id:
                call_cfg.campaign_id = self.campaign_id

            call = OutboundCall(cfg=call_cfg)
            self._all_calls[call.outbound_id] = call
            # Queue item: (priority, timestamp, outbound_id)
            # Using outbound_id (str) to avoid comparing OutboundCall objects
            self._queue.put_nowait(
                (call_cfg.priority.value, call.created_at, call.outbound_id)
            )

    # -----------------------------------------------------------------------
    # Campaign lifecycle
    # -----------------------------------------------------------------------

    async def start(
        self,
        dial_fn: Callable[["OutboundCall"], Coroutine],
    ) -> None:
        """
        Start the campaign.
        dial_fn: async function that dials one call and returns when done.
        """
        async with self._state_lock:
            if self._state != CampaignState.CREATED:
                raise RuntimeError(
                    f"Cannot start campaign in state {self._state.name}"
                )
            self._state    = CampaignState.RUNNING
            self.started_at = time.time()

        logger.info(
            "Campaign started: %s (%s) calls=%d",
            self.campaign_id[:8], self.name, self._total,
        )

        self._worker_task = asyncio.ensure_future(
            self._worker_loop(dial_fn)
        )
        self._retry_task = asyncio.ensure_future(
            self._retry_loop()
        )

    async def pause(self) -> None:
        """Pause campaign (no new calls, active calls continue)."""
        async with self._state_lock:
            if self._state == CampaignState.RUNNING:
                self._state = CampaignState.PAUSED
                logger.info("Campaign paused: %s", self.campaign_id[:8])

    async def resume(self) -> None:
        """Resume a paused campaign."""
        async with self._state_lock:
            if self._state == CampaignState.PAUSED:
                self._state = CampaignState.RUNNING
                logger.info("Campaign resumed: %s", self.campaign_id[:8])

    async def cancel(self) -> None:
        """Cancel campaign. Active calls continue to completion."""
        async with self._state_lock:
            if self._state not in (
                CampaignState.COMPLETED, CampaignState.CANCELLED
            ):
                self._state = CampaignState.CANCELLED

        self._stop_event.set()

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
        if self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()

        logger.info("Campaign cancelled: %s", self.campaign_id[:8])

    # -----------------------------------------------------------------------
    # Worker loops
    # -----------------------------------------------------------------------

    async def _worker_loop(
        self,
        dial_fn: Callable[["OutboundCall"], Coroutine],
    ) -> None:
        """
        Main campaign worker.
        Dequeues calls, respects concurrency + rate limits,
        enforces business hours.
        """
        logger.debug(
            "Campaign worker started: %s", self.campaign_id[:8]
        )

        while not self._stop_event.is_set():
            # Check if campaign is paused
            if self._state == CampaignState.PAUSED:
                await asyncio.sleep(1.0)
                continue

            # Check if campaign is cancelled/completed
            if self._state in (
                CampaignState.CANCELLED,
                CampaignState.COMPLETED,
            ):
                break

            # Check business hours
            if not self._is_business_hours():
                wait = self._seconds_until_business_hours()
                logger.info(
                    "Campaign %s: outside business hours, "
                    "waiting %.0fs",
                    self.campaign_id[:8], wait,
                )
                await asyncio.sleep(min(wait, 60.0))
                continue

            # Try to get next call from queue
            try:
                priority, created_at, outbound_id = (
                    self._queue.get_nowait()
                )
            except asyncio.QueueEmpty:
                # Queue empty — check if all calls done
                if await self._is_complete():
                    await self._on_complete()
                    break
                await asyncio.sleep(CAMPAIGN_POLL_INTERVAL)
                continue

            call = self._all_calls.get(outbound_id)
            if not call:
                continue

            # Check scheduled time
            if call.cfg.scheduled_at:
                now = time.time()
                if call.cfg.scheduled_at > now:
                    # Not yet time — put back and wait
                    await self._queue.put(
                        (priority, created_at, outbound_id)
                    )
                    await asyncio.sleep(1.0)
                    continue

            # Rate limit
            await self._rate_limiter.acquire()

            # Concurrency limit
            async with self._semaphore:
                async with self._active_lock:
                    self._active.add(outbound_id)

                try:
                    # Dial the call (blocks until call ends or fails)
                    await dial_fn(call)
                except Exception as e:
                    logger.error(
                        "Campaign dial error [%s]: %s",
                        outbound_id[:8], e, exc_info=True,
                    )
                    await call.transition(
                        CallState.FAILED, f"dial_error: {e}"
                    )
                finally:
                    async with self._active_lock:
                        self._active.discard(outbound_id)

            # Schedule retry if needed
            if call.should_retry:
                retry_at = time.time() + call.cfg.retry_delay
                call.reset_for_retry()
                await call.transition(CallState.PENDING, "retry_scheduled")
                self._retry_queue.append((retry_at, call))
                logger.info(
                    "Call scheduled for retry in %.0fs: %s",
                    call.cfg.retry_delay, call.cfg.destination,
                )

        logger.info(
            "Campaign worker ended: %s", self.campaign_id[:8]
        )

    async def _retry_loop(self) -> None:
        """
        Moves calls from retry queue back to main queue when ready.
        """
        while not self._stop_event.is_set():
            await asyncio.sleep(5.0)

            now   = time.time()
            ready = []
            remaining = []

            for retry_at, call in self._retry_queue:
                if retry_at <= now:
                    ready.append(call)
                else:
                    remaining.append((retry_at, call))

            self._retry_queue = remaining

            for call in ready:
                logger.info(
                    "Retry: re-queuing %s (attempt %d/%d)",
                    call.cfg.destination,
                    call.attempt + 1,
                    call.cfg.max_retries + 1,
                )
                await self._queue.put(
                    (call.cfg.priority.value, time.time(), call.outbound_id)
                )

    async def _is_complete(self) -> bool:
        """Check if all calls have reached terminal state."""
        async with self._active_lock:
            if self._active:
                return False
        if self._retry_queue:
            return False
        # Check all calls are terminal
        for call in self._all_calls.values():
            if not call.is_terminal and call.state != CallState.PENDING:
                return False
        return self._queue.empty()

    async def _on_complete(self) -> None:
        async with self._state_lock:
            self._state  = CampaignState.COMPLETED
            self.ended_at = time.time()

        logger.info(
            "Campaign completed: %s (%s) "
            "total=%d answered=%d failed=%d "
            "duration=%.0fs",
            self.campaign_id[:8], self.name,
            self.stats["total"],
            self.stats["answered"],
            self.stats["failed"],
            (self.ended_at or time.time()) - (self.started_at or time.time()),
        )

    # -----------------------------------------------------------------------
    # Business hours
    # -----------------------------------------------------------------------

    def _is_business_hours(self) -> bool:
        """Check if current time is within configured business hours."""
        if not self.cfg.business_hours_start:
            return True   # No restriction

        now      = datetime.now()
        weekday  = now.weekday()

        if weekday not in self.cfg.business_days:
            return False

        current_time = now.time()
        return (
            self.cfg.business_hours_start
            <= current_time
            <= self.cfg.business_hours_end
        )

    def _seconds_until_business_hours(self) -> float:
        """Calculate seconds until next business hours window opens."""
        if not self.cfg.business_hours_start:
            return 0.0

        now = datetime.now()
        today_start = datetime.combine(
            now.date(), self.cfg.business_hours_start
        )

        if now < today_start and now.weekday() in self.cfg.business_days:
            return (today_start - now).total_seconds()

        # Find next business day
        from datetime import timedelta
        for days_ahead in range(1, 8):
            next_day = now + timedelta(days=days_ahead)
            if next_day.weekday() in self.cfg.business_days:
                next_start = datetime.combine(
                    next_day.date(), self.cfg.business_hours_start
                )
                return (next_start - now).total_seconds()

        return 3600.0  # fallback: 1 hour

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    @property
    def stats(self) -> Dict:
        calls = list(self._all_calls.values())
        by_state: Dict[str, int] = {}
        for call in calls:
            s = call.state.value
            by_state[s] = by_state.get(s, 0) + 1

        answered  = by_state.get("completed", 0)
        failed    = (
            by_state.get("failed", 0)
            + by_state.get("no_answer", 0)
            + by_state.get("busy", 0)
        )
        pending   = by_state.get("pending", 0)
        active    = len(self._active)
        progress  = (
            (self._total - pending - active) / self._total * 100
            if self._total > 0 else 0
        )

        return {
            "campaign_id":   self.campaign_id,
            "name":          self.name,
            "state":         self._state.value,
            "total":         self._total,
            "by_state":      by_state,
            "answered":      answered,
            "failed":        failed,
            "pending":       pending,
            "active":        active,
            "retrying":      len(self._retry_queue),
            "progress_pct":  round(progress, 1),
            "created_at":    self.created_at,
            "started_at":    self.started_at,
            "ended_at":      self.ended_at,
            "duration_s":    round(
                (self.ended_at or time.time())
                - (self.started_at or time.time()), 1
            ) if self.started_at else 0,
        }

    def get_calls(
        self,
        state_filter: Optional[CallState] = None,
        limit: int = 100,
    ) -> List[Dict]:
        calls = list(self._all_calls.values())
        if state_filter:
            calls = [c for c in calls if c.state == state_filter]
        return [c.to_dict() for c in calls[:limit]]


# ---------------------------------------------------------------------------
# OutboundCallManager: the top-level orchestrator
# ---------------------------------------------------------------------------

class OutboundCallManager:
    """
    Top-level manager for all outbound calls and campaigns.

    Responsibilities:
      - Single call API: place_call()
      - Campaign API:    create_campaign() / start_campaign()
      - ESL event routing (CHANNEL_ANSWER, CHANNEL_HANGUP, etc.)
      - Audio stream connection (uuid_execute audio_stream)
      - Webhook delivery
      - DNC checking
      - Completed call archiving
      - System-wide concurrency cap

    Integrates with:
      - ESLClient         (esl.py)
      - AudioServer       (audio.py)
      - SessionManager    (session.py)
      - MetricsRegistry   (metrics.py)
    """

    def __init__(
        self,
        esl:             "Any",   # ESLClient
        audio_server:    "Any",   # AudioServer
        session_manager: "Any",   # SessionManager
        cfg:             "Any",   # AppConfig
        metrics:         Optional["Any"] = None,
    ):
        self._esl      = esl
        self._audio    = audio_server
        self._sm       = session_manager
        self._cfg      = cfg
        self._metrics  = metrics

        # Audio WebSocket URL (where mod_audio_stream connects to us)
        self._audio_ws_url = (
            f"ws://{cfg.server.host}:{cfg.server.audio_ws_port}"
        )
        # For Docker/remote FS, override with actual reachable address
        env_override = os.environ.get("VOICEBOT__AUDIO_WS_OVERRIDE", "")
        if env_override:
            self._audio_ws_url = env_override

        # Single calls (not part of a campaign)
        self._calls:      Dict[str, OutboundCall] = {}
        self._calls_lock  = asyncio.Lock()

        # UUID → outbound_id (for ESL event routing)
        self._uuid_map:   Dict[str, str] = {}
        self._uuid_lock   = asyncio.Lock()

        # Campaigns
        self._campaigns:  Dict[str, Campaign] = {}
        self._camp_lock   = asyncio.Lock()

        # DNC list
        self.dnc = DNCList()

        # Webhook sender
        self._webhook = WebhookSender()

        # Completed call archive (ring buffer)
        self._completed:  List[CallRecord] = []
        self._max_archive = 1000

        # System-wide concurrent call cap
        self._global_semaphore = asyncio.Semaphore(
            cfg.server.max_connections
        )

        # Register ESL event handlers
        self._register_esl_handlers()

        logger.info(
            "OutboundCallManager initialized: "
            "audio_ws=%s max_concurrent=%d",
            self._audio_ws_url,
            cfg.server.max_connections,
        )

    def _register_esl_handlers(self) -> None:
        """Register all ESL event handlers."""
        d = self._esl.dispatcher
        d.on_event("CHANNEL_PROGRESS",       self._on_channel_progress)
        d.on_event("CHANNEL_PROGRESS_MEDIA", self._on_channel_progress)
        d.on_event("CHANNEL_ANSWER",         self._on_channel_answer)
        d.on_event("CHANNEL_HANGUP_COMPLETE", self._on_channel_hangup)
        d.on_event("CHANNEL_EXECUTE_COMPLETE", self._on_execute_complete)
        logger.info("OutboundCallManager ESL handlers registered")

    # -----------------------------------------------------------------------
    # Public API: single call
    # -----------------------------------------------------------------------

    async def place_call(
        self,
        cfg:           OutboundCallConfig,
        outbound_id:   Optional[str] = None,
    ) -> OutboundCall:
        """
        Place a single outbound call.

        Returns immediately with an OutboundCall object.
        The call proceeds asynchronously in the background.

        Args:
            cfg:         Call configuration
            outbound_id: Optional specific ID (for idempotency)

        Returns:
            OutboundCall instance (check .state for progress)
        """
        # DNC check
        if await self.dnc.check(cfg.destination):
            call = OutboundCall(cfg=cfg, outbound_id=outbound_id)
            await call.transition(CallState.REJECTED, "dnc_list")
            call.error_message = "Number on DNC list"
            call.ended_at = time.time()
            logger.warning(
                "Call rejected (DNC): %s", cfg.destination
            )
            return call

        # Create call object
        call = OutboundCall(cfg=cfg, outbound_id=outbound_id)

        # Register
        async with self._calls_lock:
            self._calls[call.outbound_id] = call

        # Register state change callback for metrics + webhook
        call._on_state_change = self._on_call_state_change

        logger.info(
            "Placing call: id=%s dest=%s gw=%s retry=%d",
            call.outbound_id[:8],
            cfg.destination,
            cfg.gateway,
            cfg.max_retries,
        )

        # Dial in background
        asyncio.ensure_future(self._dial_with_retry(call))
        return call

    async def cancel_call(self, outbound_id: str) -> bool:
        """
        Cancel a pending or active call.
        If ringing: sends uuid_kill.
        If pending: removes from queue.
        """
        async with self._calls_lock:
            call = self._calls.get(outbound_id)

        if call is None:
            return False

        if call.call_uuid and call.is_active:
            await self._esl.uuid_kill(
                call.call_uuid, cause="ORIGINATOR_CANCEL"
            )

        await call.transition(CallState.CANCELLED, "api_cancel")
        call.ended_at = time.time()
        return True

    # -----------------------------------------------------------------------
    # Public API: campaigns
    # -----------------------------------------------------------------------

    async def create_campaign(
        self, cfg: CampaignConfig
    ) -> Campaign:
        """
        Create a new outbound campaign (does not start it yet).
        """
        campaign = Campaign(cfg=cfg)

        # Wire per-call state change callback
        for call in campaign._all_calls.values():
            call._on_state_change = self._on_call_state_change

        async with self._camp_lock:
            self._campaigns[campaign.campaign_id] = campaign

        logger.info(
            "Campaign created: %s (%s) calls=%d",
            campaign.campaign_id[:8], cfg.name, len(cfg.calls),
        )
        return campaign

    async def start_campaign(self, campaign_id: str) -> bool:
        """Start a created campaign."""
        campaign = self._campaigns.get(campaign_id)
        if campaign is None:
            return False

        # Register all campaign calls
        async with self._calls_lock:
            for call in campaign._all_calls.values():
                self._calls[call.outbound_id] = call

        await campaign.start(dial_fn=self._dial_with_retry)
        return True

    async def pause_campaign(self, campaign_id: str) -> bool:
        campaign = self._campaigns.get(campaign_id)
        if campaign is None:
            return False
        await campaign.pause()
        return True

    async def resume_campaign(self, campaign_id: str) -> bool:
        campaign = self._campaigns.get(campaign_id)
        if campaign is None:
            return False
        await campaign.resume()
        return True

    async def cancel_campaign(self, campaign_id: str) -> bool:
        campaign = self._campaigns.get(campaign_id)
        if campaign is None:
            return False
        await campaign.cancel()
        return True

    # -----------------------------------------------------------------------
    # Core dialing logic
    # -----------------------------------------------------------------------

    async def _dial_with_retry(self, call: OutboundCall) -> None:
        """
        Dial a call, retrying on no-answer/busy up to max_retries.
        This is the core dial loop for both single calls and campaigns.
        """
        async with self._global_semaphore:
            while True:
                call.attempt += 1

                logger.info(
                    "Dialing: %s attempt=%d/%d id=%s",
                    call.cfg.destination,
                    call.attempt,
                    call.cfg.max_retries + 1,
                    call.outbound_id[:8],
                )

                success = await self._dial_once(call)

                if success:
                    # Call completed normally
                    break

                if not call.should_retry:
                    break

                # Wait before retry
                logger.info(
                    "Retry in %.0fs: %s (attempt %d done)",
                    call.cfg.retry_delay,
                    call.cfg.destination,
                    call.attempt,
                )
                await asyncio.sleep(call.cfg.retry_delay)
                call.reset_for_retry()
                await call.transition(CallState.PENDING, "retry")

        # Archive completed call
        await self._archive_call(call)

    async def _dial_once(self, call: OutboundCall) -> bool:
        """
        Execute a single dial attempt.
        Returns True if call was answered and completed.
        Returns False if failed/busy/no-answer (may retry).
        """
        # Step 1: ESL originate
        await call.transition(CallState.DIALING)
        call.dialed_at = time.time()

        call_uuid = await self._originate(call)

        if not call_uuid:
            await call.transition(CallState.FAILED, "originate_failed")
            call.error_message = "ESL originate returned no UUID"
            call.ended_at = time.time()
            return False

        call.call_uuid = call_uuid

        # Register UUID mapping
        async with self._uuid_lock:
            self._uuid_map[call_uuid] = call.outbound_id

        # Step 2: Wait for CHANNEL_ANSWER or terminal event
        loop   = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        call._answer_future = future

        try:
            result = await asyncio.wait_for(
                future,
                timeout=float(call.cfg.ring_timeout + 10),
            )
        except asyncio.TimeoutError:
            result = "timeout"
            logger.warning(
                "Ring timeout: %s (%.0fs)",
                call.cfg.destination, call.cfg.ring_timeout,
            )

        # Handle result
        if result == "answered":
            # Step 3: Connect audio stream
            connected = await self._connect_audio_stream(call)
            if not connected:
                await call.transition(CallState.FAILED, "audio_connect_failed")
                call.ended_at = time.time()
                return False

            # Step 4: Wait for call to end
            await self._wait_for_call_end(call)
            return True

        elif result == "busy":
            await call.transition(CallState.BUSY, "user_busy")
            call.ended_at = time.time()
            return False

        elif result in ("no_answer", "timeout"):
            await call.transition(CallState.NO_ANSWER, result)
            call.ended_at = time.time()
            return False

        else:
            # failed / rejected / cancelled
            cause = call.hangup_cause or result
            if result == "cancelled":
                await call.transition(CallState.CANCELLED, cause)
            else:
                await call.transition(CallState.FAILED, cause)
            call.ended_at = time.time()
            return False

    async def _originate(self, call: OutboundCall) -> Optional[str]:
        """
        Send FreeSWITCH originate command via ESL.
        Returns call_uuid on success, None on failure.
        """
        # Build channel variables
        var_dict = {
            "origination_caller_id_number": call.cfg.caller_id_num,
            "origination_caller_id_name":   call.cfg.caller_id_name,
            "call_timeout":                  str(call.cfg.ring_timeout),
            "ignore_early_media":            "true",
            "voicebot_outbound":             "true",
            "voicebot_outbound_id":          call.outbound_id,
            "voicebot_campaign_id":          call.cfg.campaign_id or "",
            "hangup_after_bridge":           "false",
        }

        # Add custom metadata as FS variables
        for k, v in call.cfg.metadata.items():
            safe_key = "voicebot_meta_" + k.replace("-", "_")
            var_dict[safe_key] = str(v)

        var_str = ",".join(f"{k}={v}" for k, v in var_dict.items())

        # Build destination string
        # Format: sofia/gateway/<gateway>/<destination>
        dest_str = (
            f"sofia/gateway/{call.cfg.gateway}/{call.cfg.destination}"
        )

        # originate {vars}<dest> &park()
        # &park() keeps the call alive waiting for our control
        cmd = f"originate {{{var_str}}}{dest_str} &park()"

        logger.debug(
            "ESL originate: dest=%s gw=%s timeout=%d",
            call.cfg.destination,
            call.cfg.gateway,
            call.cfg.ring_timeout,
        )

        try:
            event = await asyncio.wait_for(
                self._esl.send_bgapi(cmd, timeout=ORIGINATE_TIMEOUT),
                timeout=ORIGINATE_TIMEOUT + 5,
            )
            body = event.body or ""

            if "+OK" in body:
                call_uuid = body.replace("+OK", "").strip()
                logger.info(
                    "Originated: uuid=%s dest=%s",
                    call_uuid[:8], call.cfg.destination,
                )
                return call_uuid
            else:
                logger.error(
                    "Originate failed: dest=%s response=%s",
                    call.cfg.destination, body[:200],
                )
                call.error_message = f"originate: {body[:200]}"
                return None

        except asyncio.TimeoutError:
            logger.error(
                "Originate timeout: dest=%s (%.0fs)",
                call.cfg.destination, ORIGINATE_TIMEOUT,
            )
            call.error_message = "originate timeout"
            return None

        except Exception as e:
            logger.error(
                "Originate error: dest=%s error=%s",
                call.cfg.destination, e,
            )
            call.error_message = str(e)
            return None

    async def _connect_audio_stream(self, call: OutboundCall) -> bool:
        """
        Execute audio_stream application on the answered call.
        This makes mod_audio_stream connect to our WebSocket server.

        After this, our AudioServer will receive the WebSocket connection
        and create a Session automatically.
        """
        await call.transition(CallState.CONNECTING)

        # Set call variables so AudioStream/Session know this is outbound
        await self._esl.uuid_setvar_multi(
            call.call_uuid,
            {
                "voicebot_direction":    "outbound",
                "voicebot_outbound_id":  call.outbound_id,
                "voicebot_campaign_id":  call.cfg.campaign_id or "",
                "voicebot_destination":  call.cfg.destination,
            }
        )

        logger.info(
            "Executing audio_stream: uuid=%s url=%s",
            call.call_uuid[:8], self._audio_ws_url,
        )

        # Execute audio_stream — this blocks in FreeSWITCH until stream ends
        # We don't event_lock because we need to proceed
        try:
            await self._esl.uuid_execute(
                call.call_uuid,
                app="audio_stream",
                arg=f"{self._audio_ws_url} 16000 1",
                event_lock=False,
            )
        except Exception as e:
            logger.error(
                "audio_stream execute error [%s]: %s",
                call.call_uuid[:8], e,
            )
            call.error_message = f"audio_stream execute: {e}"
            return False

        # Wait for AudioServer to receive WebSocket from mod_audio_stream
        loop   = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        call._connect_future = future

        try:
            await asyncio.wait_for(future, timeout=AUDIO_CONNECT_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(
                "Audio stream connect timeout: %s (%.0fs)",
                call.call_uuid[:8], AUDIO_CONNECT_TIMEOUT,
            )
            call.error_message = "audio_stream connect timeout"
            return False

        call.connected_at = time.time()
        await call.transition(CallState.CONNECTED)

        logger.info(
            "Audio stream connected: %s dest=%s",
            call.outbound_id[:8], call.cfg.destination,
        )
        return True

    async def _wait_for_call_end(self, call: OutboundCall) -> None:
        """
        Wait until the call session terminates.
        Polls session state until it's terminal.
        """
        poll_interval = 1.0
        max_wait      = call.cfg.ring_timeout + 3600  # ring + 1hr max

        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval)

            # Check if session still exists
            if call.session_id:
                session = self._sm.get_session(call.session_id)
                if session is None or not session.is_active:
                    break

            # Check if call state is terminal already
            if call.is_terminal:
                break

            # Check if call_uuid is still active in FreeSWITCH
            # (detected via CHANNEL_HANGUP event)
            if call.state not in (
                CallState.CONNECTED,
                CallState.CONNECTING,
            ):
                break

        if call.state == CallState.CONNECTED:
            await call.transition(CallState.COMPLETED, "session_ended")

        call.ended_at = call.ended_at or time.time()

    # -----------------------------------------------------------------------
    # ESL event handlers
    # -----------------------------------------------------------------------

    async def _on_channel_progress(self, event: "Any") -> None:
        """SIP 180 Ringing or 183 Session Progress received."""
        call_uuid = event.unique_id
        call      = await self._get_call_by_uuid(call_uuid)
        if call is None:
            return

        if call.state == CallState.DIALING:
            await call.transition(CallState.RINGING)
            logger.info(
                "Ringing: %s uuid=%s",
                call.cfg.destination, call_uuid[:8],
            )

    async def _on_channel_answer(self, event: "Any") -> None:
        """
        Call was answered.
        Resolve the answer future so _dial_once() can proceed.
        """
        call_uuid = event.unique_id
        call      = await self._get_call_by_uuid(call_uuid)
        if call is None:
            return

        call.answered_at = time.time()
        await call.transition(CallState.ANSWERED)

        logger.info(
            "Answered: %s uuid=%s ring=%.1fs",
            call.cfg.destination,
            call_uuid[:8],
            call.ring_duration_seconds,
        )

        # Resolve future
        if call._answer_future and not call._answer_future.done():
            call._answer_future.set_result("answered")

        # Track metrics
        if self._metrics:
            self._metrics.record_interruption()   # reuse as "answered" counter

    async def _on_channel_hangup(self, event: "Any") -> None:
        """
        Call hung up.
        Handle based on current call state.
        """
        call_uuid = event.unique_id
        cause_str = event.get("Hangup-Cause", "UNKNOWN")
        cause     = HangupCause.from_str(cause_str)

        call = await self._get_call_by_uuid(call_uuid)
        if call is None:
            return

        call.hangup_cause = cause_str
        call.ended_at     = time.time()

        logger.info(
            "Hangup: %s cause=%s state=%s uuid=%s",
            call.cfg.destination,
            cause_str,
            call.state.value,
            call_uuid[:8],
        )

        # Resolve answer future if still pending (no answer case)
        if call._answer_future and not call._answer_future.done():
            if cause == HangupCause.USER_BUSY:
                call._answer_future.set_result("busy")
            elif cause in (
                HangupCause.NO_ANSWER,
                HangupCause.ORIGINATOR_CANCEL,
            ):
                call._answer_future.set_result("no_answer")
            elif cause.is_invalid:
                call._answer_future.set_result("failed")
            else:
                call._answer_future.set_result("failed")

        # Resolve connect future if still pending
        if call._connect_future and not call._connect_future.done():
            call._connect_future.set_exception(
                ConnectionError(f"Call hung up: {cause_str}")
            )

        # Clean up UUID mapping
        async with self._uuid_lock:
            self._uuid_map.pop(call_uuid, None)

        # If call was connected, mark completed
        if call.state == CallState.CONNECTED:
            await call.transition(CallState.COMPLETED, f"hangup:{cause_str}")

    async def _on_execute_complete(self, event: "Any") -> None:
        """
        CHANNEL_EXECUTE_COMPLETE for audio_stream app.
        This fires when audio_stream application exits on FreeSWITCH side.
        """
        app = event.get("Application", "")
        if app != "audio_stream":
            return

        call_uuid = event.unique_id
        call      = await self._get_call_by_uuid(call_uuid)
        if call is None:
            return

        logger.info(
            "audio_stream complete: uuid=%s state=%s",
            call_uuid[:8], call.state.value,
        )

    # -----------------------------------------------------------------------
    # Called by AudioServer when WebSocket connects
    # -----------------------------------------------------------------------

    async def on_audio_stream_connected(
        self,
        stream: "Any",   # AudioStream
    ) -> None:
        """
        Called by AudioServer/SessionManager when mod_audio_stream
        WebSocket connects for an outbound call.

        Resolves the connect future so _connect_audio_stream() can proceed.
        Also passes outbound metadata to the session config.
        """
        call_uuid   = stream.call_uuid
        outbound_id = None

        # Try to find outbound_id from call_uuid mapping first
        async with self._uuid_lock:
            outbound_id = self._uuid_map.get(call_uuid)

        # Also try metadata from stream (FS channel variable)
        if not outbound_id and stream.metadata:
            outbound_id = stream.metadata.extra.get(
                "variable_voicebot_outbound_id", ""
            )

        if not outbound_id:
            logger.warning(
                "audio_stream connected but no outbound_id found: "
                "uuid=%s", call_uuid[:8],
            )
            return

        call = self._calls.get(outbound_id)
        if not call:
            logger.warning(
                "audio_stream connected but call not found: "
                "outbound_id=%s", outbound_id[:8],
            )
            return

        logger.info(
            "Outbound audio_stream connected: id=%s uuid=%s",
            outbound_id[:8], call_uuid[:8],
        )

        # Resolve connect future
        if call._connect_future and not call._connect_future.done():
            call._connect_future.set_result(True)

    async def on_session_created(
        self,
        session_id: str,
        call_uuid:  str,
    ) -> None:
        """
        Called by SessionManager when a new Session is created
        for an outbound call. Links session_id to our OutboundCall.
        """
        async with self._uuid_lock:
            outbound_id = self._uuid_map.get(call_uuid)

        if not outbound_id:
            return

        call = self._calls.get(outbound_id)
        if call:
            call.session_id = session_id
            logger.debug(
                "Session linked to outbound call: "
                "session=%s outbound=%s",
                session_id[:8], outbound_id[:8],
            )

    # -----------------------------------------------------------------------
    # State change handler (metrics + webhook)
    # -----------------------------------------------------------------------

    async def _on_call_state_change(
        self,
        call:      OutboundCall,
        old_state: CallState,
        new_state: CallState,
    ) -> None:
        """
        Called on every state transition.
        Updates metrics and fires webhooks.
        """
        # Metrics
        if self._metrics:
            if new_state == CallState.ANSWERED:
                self._metrics.session_start(
                    session_id=call.outbound_id,
                    call_uuid=call.call_uuid or "",
                )
            if new_state in (
                CallState.COMPLETED,
                CallState.FAILED,
                CallState.NO_ANSWER,
                CallState.BUSY,
                CallState.CANCELLED,
            ):
                if call.answered_at:
                    self._metrics.observe_e2e(
                        (time.time() - call.answered_at) * 1000
                    )

        # Webhook
        if call.cfg.webhook_url:
            asyncio.ensure_future(
                self._send_webhook(call, new_state)
            )

    async def _send_webhook(
        self,
        call:      OutboundCall,
        new_state: CallState,
    ) -> None:
        """Send webhook notification for call state change."""
        payload = {
            "event":      f"call.{new_state.value}",
            "timestamp":  time.time(),
            "call":       call.to_dict(),
        }
        await self._webhook.send(
            url=call.cfg.webhook_url,
            payload=payload,
            secret=call.cfg.webhook_secret,
        )

    # -----------------------------------------------------------------------
    # Archive
    # -----------------------------------------------------------------------

    async def _archive_call(self, call: OutboundCall) -> None:
        """Move completed call to archive ring buffer."""
        record = call.to_record()
        self._completed.append(record)
        if len(self._completed) > self._max_archive:
            self._completed.pop(0)

        # Clean up active calls dict
        async with self._calls_lock:
            self._calls.pop(call.outbound_id, None)

        logger.info(
            "Call archived: id=%s dest=%s state=%s "
            "duration=%.1fs attempts=%d",
            call.outbound_id[:8],
            call.cfg.destination,
            call.state.value,
            call.call_duration_seconds,
            call.attempt,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    async def _get_call_by_uuid(
        self, call_uuid: str
    ) -> Optional[OutboundCall]:
        """Look up OutboundCall by FreeSWITCH channel UUID."""
        async with self._uuid_lock:
            outbound_id = self._uuid_map.get(call_uuid)
        if not outbound_id:
            return None
        return self._calls.get(outbound_id)

    async def shutdown(self) -> None:
        """Graceful shutdown: cancel all campaigns, wait for active calls."""
        logger.info("OutboundCallManager shutting down...")

        # Cancel all campaigns
        async with self._camp_lock:
            for campaign in self._campaigns.values():
                await campaign.cancel()

        # Wait for active calls to finish (up to 30s)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            async with self._calls_lock:
                active = [
                    c for c in self._calls.values() if c.is_active
                ]
            if not active:
                break
            logger.info(
                "Waiting for %d active outbound calls...", len(active)
            )
            await asyncio.sleep(2.0)

        await self._webhook.close()
        logger.info("OutboundCallManager shutdown complete")

    # -----------------------------------------------------------------------
    # Stats / queries
    # -----------------------------------------------------------------------

    def get_call(self, outbound_id: str) -> Optional[OutboundCall]:
        return self._calls.get(outbound_id)

    def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        return self._campaigns.get(campaign_id)

    def list_calls(
        self,
        state_filter: Optional[CallState] = None,
        limit: int = 100,
    ) -> List[Dict]:
        calls = list(self._calls.values())
        if state_filter:
            calls = [c for c in calls if c.state == state_filter]
        return [c.to_dict() for c in calls[:limit]]

    def list_campaigns(self) -> List[Dict]:
        return [c.stats for c in self._campaigns.values()]

    def list_completed(self, limit: int = 50) -> List[Dict]:
        return [r.to_dict() for r in self._completed[-limit:]]

    def get_stats(self) -> Dict:
        calls     = list(self._calls.values())
        campaigns = list(self._campaigns.values())

        by_state: Dict[str, int] = {}
        for call in calls:
            s = call.state.value
            by_state[s] = by_state.get(s, 0) + 1

        return {
            "active_calls":      len(calls),
            "active_campaigns":  len(campaigns),
            "completed_calls":   len(self._completed),
            "dnc_size":          self.dnc.size,
            "by_state":          by_state,
            "campaigns":         [c.stats for c in campaigns],
        }


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
import os
