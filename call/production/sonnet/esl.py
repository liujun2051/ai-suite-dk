# 文件3: `esl.py`

```python
"""
esl.py - FreeSWITCH Event Socket Library (ESL) client

Handles:
  - Persistent TCP connection to FreeSWITCH ESL
  - Authentication
  - Event subscription and dispatching
  - Command sending (api, bgapi, sendmsg)
  - Call control: uuid_break, uuid_kill, uuid_setvar, uuid_answer, etc.
  - Automatic reconnection with exponential backoff
  - Heartbeat monitoring
  - Background job tracking
  - Thread-safe command queue

FreeSWITCH ESL protocol:
  - Plain text protocol over TCP
  - Each message separated by \\n\\n
  - Headers: Key: Value\\n
  - Body follows Content-Length header
  - Commands sent as plain text lines
  - Responses arrive as events

Reference: https://developer.signalwire.com/freeswitch/FreeSWITCH-Explained/Client-and-Developer-Interfaces/Event-Socket-Library/
"""

import asyncio
import logging
import re
import time
import uuid
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Coroutine, Dict, List,
    Optional, Set, Tuple
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ESL_HEADER_END      = b"\n\n"
ESL_LINE_END        = b"\n"
MAX_HEADER_SIZE     = 8192      # bytes
MAX_BODY_SIZE       = 10 * 1024 * 1024   # 10 MB
RECONNECT_BASE_DELAY = 1.0      # seconds
RECONNECT_MAX_DELAY  = 30.0     # seconds
COMMAND_TIMEOUT      = 10.0     # seconds
BGAPI_TIMEOUT        = 30.0     # seconds


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ESLConnectionState(Enum):
    DISCONNECTED  = auto()
    CONNECTING    = auto()
    AUTHENTICATING = auto()
    CONNECTED     = auto()
    RECONNECTING  = auto()
    STOPPING      = auto()


class ESLEventType(str, Enum):
    """Common FreeSWITCH event types we care about."""
    CHANNEL_CREATE           = "CHANNEL_CREATE"
    CHANNEL_ANSWER           = "CHANNEL_ANSWER"
    CHANNEL_HANGUP           = "CHANNEL_HANGUP"
    CHANNEL_HANGUP_COMPLETE  = "CHANNEL_HANGUP_COMPLETE"
    CHANNEL_BRIDGE           = "CHANNEL_BRIDGE"
    CHANNEL_UNBRIDGE         = "CHANNEL_UNBRIDGE"
    CHANNEL_EXECUTE          = "CHANNEL_EXECUTE"
    CHANNEL_EXECUTE_COMPLETE = "CHANNEL_EXECUTE_COMPLETE"
    CHANNEL_PROGRESS         = "CHANNEL_PROGRESS"
    CHANNEL_PROGRESS_MEDIA   = "CHANNEL_PROGRESS_MEDIA"
    CHANNEL_PARK             = "CHANNEL_PARK"
    CHANNEL_HOLD             = "CHANNEL_HOLD"
    CHANNEL_UNHOLD           = "CHANNEL_UNHOLD"
    DTMF                     = "DTMF"
    DETECTED_SPEECH          = "DETECTED_SPEECH"
    PLAYBACK_START           = "PLAYBACK_START"
    PLAYBACK_STOP            = "PLAYBACK_STOP"
    RECORD_START             = "RECORD_START"
    RECORD_STOP              = "RECORD_STOP"
    BACKGROUND_JOB           = "BACKGROUND_JOB"
    HEARTBEAT                = "HEARTBEAT"
    SHUTDOWN                 = "SHUTDOWN"
    CUSTOM                   = "CUSTOM"
    ALL                      = "ALL"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ESLEvent:
    """
    Parsed FreeSWITCH ESL event.
    Headers are stored as a dict, body as optional string.
    """
    headers: Dict[str, str]
    body:    Optional[str] = None

    @property
    def event_name(self) -> str:
        return self.headers.get("Event-Name", "")

    @property
    def event_subclass(self) -> str:
        return self.headers.get("Event-Subclass", "")

    @property
    def unique_id(self) -> str:
        """Channel UUID (call leg identifier)."""
        return (
            self.headers.get("Unique-ID", "")
            or self.headers.get("Channel-Call-UUID", "")
        )

    @property
    def content_type(self) -> str:
        return self.headers.get("Content-Type", "")

    @property
    def reply_text(self) -> str:
        return self.headers.get("Reply-Text", "")

    @property
    def job_uuid(self) -> str:
        return self.headers.get("Job-UUID", "")

    @property
    def is_ok(self) -> bool:
        return self.reply_text.startswith("+OK")

    @property
    def is_error(self) -> bool:
        return self.reply_text.startswith("-ERR")

    def get(self, key: str, default: str = "") -> str:
        return self.headers.get(key, default)

    def __repr__(self) -> str:
        uid = self.unique_id
        return (
            f"ESLEvent(name={self.event_name!r} "
            f"ct={self.content_type!r} "
            f"uuid={uid[:8] if uid else ''}...)"
        )


@dataclass
class ESLCommand:
    """
    A pending ESL command waiting for a response.
    """
    command:    str
    future:     asyncio.Future
    sent_at:    float = field(default_factory=time.monotonic)
    timeout:    float = COMMAND_TIMEOUT


@dataclass
class BGAPIJob:
    """
    A pending background API job waiting for BACKGROUND_JOB event.
    """
    job_uuid:   str
    command:    str
    future:     asyncio.Future
    sent_at:    float = field(default_factory=time.monotonic)
    timeout:    float = BGAPI_TIMEOUT


@dataclass
class ESLConnectionInfo:
    """Current connection status snapshot."""
    state:              ESLConnectionState
    host:               str
    port:               int
    connected_at:       Optional[float]
    reconnect_attempts: int
    commands_sent:      int
    events_received:    int
    last_heartbeat:     Optional[float]

    def to_dict(self) -> Dict:
        return {
            "state":              self.state.name,
            "host":               self.host,
            "port":               self.port,
            "connected_at":       self.connected_at,
            "uptime_seconds":     (
                time.time() - self.connected_at
                if self.connected_at else None
            ),
            "reconnect_attempts": self.reconnect_attempts,
            "commands_sent":      self.commands_sent,
            "events_received":    self.events_received,
            "last_heartbeat":     self.last_heartbeat,
        }


# ---------------------------------------------------------------------------
# ESL protocol parser
# ---------------------------------------------------------------------------

class ESLProtocolParser:
    """
    Incremental parser for the FreeSWITCH ESL wire protocol.

    ESL message format:
        Header-Name: header-value\\n
        Another-Header: value\\n
        Content-Length: <N>\\n
        \\n
        <N bytes of body>

    Multiple messages arrive in a single TCP stream.
    This parser handles fragmentation and reassembly.
    """

    def __init__(self):
        self._buf        = bytearray()
        self._messages:  List[ESLEvent] = []

    def feed(self, data: bytes) -> List[ESLEvent]:
        """
        Feed raw TCP bytes into the parser.
        Returns list of fully parsed ESLEvent objects (may be empty).
        """
        self._buf.extend(data)
        self._messages.clear()
        self._parse()
        return list(self._messages)

    def _parse(self) -> None:
        while True:
            # Find the end of headers (blank line = \\n\\n)
            header_end = self._buf.find(b"\n\n")
            if header_end == -1:
                # Incomplete headers — wait for more data
                if len(self._buf) > MAX_HEADER_SIZE:
                    logger.error(
                        "ESL header too large (%d bytes), clearing buffer",
                        len(self._buf)
                    )
                    self._buf.clear()
                break

            # Parse headers
            header_bytes = self._buf[:header_end]
            headers = self._parse_headers(header_bytes)

            # Check for body
            content_length = int(headers.get("Content-Length", "0"))
            message_end = header_end + 2  # skip the \\n\\n

            if content_length > 0:
                body_end = message_end + content_length
                if len(self._buf) < body_end:
                    # Body not fully received yet
                    break
                if content_length > MAX_BODY_SIZE:
                    logger.error(
                        "ESL body too large (%d bytes), skipping",
                        content_length
                    )
                    del self._buf[:body_end]
                    continue

                body_bytes = self._buf[message_end:body_end]
                body = body_bytes.decode("utf-8", errors="replace")
                del self._buf[:body_end]
            else:
                body = None
                del self._buf[:message_end]

            event = ESLEvent(headers=headers, body=body)

            # If this is a text/event-plain content type,
            # the body itself is another set of headers + optional body
            if headers.get("Content-Type") == "text/event-plain" and body:
                event = self._parse_event_body(headers, body)

            self._messages.append(event)

    def _parse_headers(self, header_bytes: bytes) -> Dict[str, str]:
        """Parse raw header bytes into a dict."""
        headers: Dict[str, str] = {}
        for line in header_bytes.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            if b":" in line:
                key, _, value = line.partition(b":")
                headers[
                    key.strip().decode("utf-8", errors="replace")
                ] = value.strip().decode("utf-8", errors="replace")
        return headers

    def _parse_event_body(
        self, outer_headers: Dict[str, str], body: str
    ) -> ESLEvent:
        """
        For text/event-plain messages, the body contains event headers
        and optionally another nested body.
        """
        lines = body.split("\n")
        event_headers: Dict[str, str] = {}
        event_body: Optional[str] = None
        content_length = 0

        for i, line in enumerate(lines):
            line = line.rstrip("\r")
            if not line:
                # Blank line — rest is event body
                if content_length > 0:
                    event_body = "\n".join(lines[i + 1:])
                break
            if ":" in line:
                key, _, value = line.partition(":")
                k = key.strip()
                v = value.strip()
                event_headers[k] = v
                if k == "Content-Length":
                    try:
                        content_length = int(v)
                    except ValueError:
                        pass

        # Merge outer headers (but event headers take precedence)
        merged = {**outer_headers, **event_headers}
        return ESLEvent(headers=merged, body=event_body)

    def reset(self) -> None:
        self._buf.clear()
        self._messages.clear()


# ---------------------------------------------------------------------------
# Event handler registry
# ---------------------------------------------------------------------------

EventHandler = Callable[[ESLEvent], Coroutine]


class ESLEventDispatcher:
    """
    Routes incoming ESL events to registered async handlers.

    Supports:
      - Subscribe by event name (e.g. "CHANNEL_HANGUP")
      - Subscribe by UUID (all events for a specific call)
      - Wildcard (all events)
      - One-shot handlers (auto-removed after first match)
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._lock = threading.Lock()

        # event_name → list of handlers
        self._by_event: Dict[str, List[EventHandler]] = defaultdict(list)
        # uuid → list of handlers
        self._by_uuid:  Dict[str, List[EventHandler]] = defaultdict(list)
        # Wildcard handlers (receive all events)
        self._wildcard: List[EventHandler] = []
        # One-shot: (event_name, uuid) → Future
        self._oneshot:  Dict[Tuple[str, str], asyncio.Future] = {}

    def on_event(
        self,
        event_name: str,
        handler: EventHandler,
    ) -> None:
        """Register a persistent handler for a specific event name."""
        with self._lock:
            self._by_event[event_name].append(handler)

    def on_uuid(
        self,
        uuid: str,
        handler: EventHandler,
    ) -> None:
        """Register a persistent handler for all events from a specific UUID."""
        with self._lock:
            self._by_uuid[uuid].append(handler)

    def on_any(self, handler: EventHandler) -> None:
        """Register a wildcard handler (receives all events)."""
        with self._lock:
            self._wildcard.append(handler)

    def off_event(self, event_name: str, handler: EventHandler) -> None:
        with self._lock:
            handlers = self._by_event.get(event_name, [])
            if handler in handlers:
                handlers.remove(handler)

    def off_uuid(self, uuid: str, handler: EventHandler) -> None:
        with self._lock:
            handlers = self._by_uuid.get(uuid, [])
            if handler in handlers:
                handlers.remove(handler)

    def off_any(self, handler: EventHandler) -> None:
        with self._lock:
            if handler in self._wildcard:
                self._wildcard.remove(handler)

    def remove_uuid(self, uuid: str) -> None:
        """Remove all handlers for a UUID (call on hangup)."""
        with self._lock:
            self._by_uuid.pop(uuid, None)
            # Also clean up any pending one-shots for this UUID
            to_remove = [k for k in self._oneshot if k[1] == uuid]
            for k in to_remove:
                fut = self._oneshot.pop(k)
                if not fut.done():
                    fut.cancel()

    def wait_for_event(
        self,
        event_name: str,
        uuid: str = "",
        timeout: float = 10.0,
    ) -> asyncio.Future:
        """
        Return a Future that resolves to the next matching ESLEvent.
        Useful for waiting on specific events synchronously in async code.

        Example:
            event = await esl.dispatcher.wait_for_event(
                "CHANNEL_ANSWER", uuid=call_uuid, timeout=30.0
            )
        """
        key = (event_name, uuid)
        fut: asyncio.Future = self._loop.create_future()
        with self._lock:
            self._oneshot[key] = fut

        async def _timeout_guard():
            await asyncio.sleep(timeout)
            with self._lock:
                if key in self._oneshot and not self._oneshot[key].done():
                    self._oneshot.pop(key).set_exception(
                        asyncio.TimeoutError(
                            f"Timeout waiting for {event_name} uuid={uuid}"
                        )
                    )

        asyncio.ensure_future(_timeout_guard(), loop=self._loop)
        return fut

    def dispatch(self, event: ESLEvent) -> None:
        """
        Dispatch event to all matching handlers.
        Called from the ESL reader coroutine (asyncio context).
        """
        event_name = event.event_name
        event_uuid = event.unique_id

        with self._lock:
            handlers: List[EventHandler] = []

            # By event name
            handlers.extend(self._by_event.get(event_name, []))
            # By UUID
            if event_uuid:
                handlers.extend(self._by_uuid.get(event_uuid, []))
            # Wildcard
            handlers.extend(self._wildcard)

            # One-shot: event name only
            key_name = (event_name, "")
            if key_name in self._oneshot:
                fut = self._oneshot.pop(key_name)
                if not fut.done():
                    fut.set_result(event)

            # One-shot: event name + uuid
            if event_uuid:
                key_uuid = (event_name, event_uuid)
                if key_uuid in self._oneshot:
                    fut = self._oneshot.pop(key_uuid)
                    if not fut.done():
                        fut.set_result(event)

        # Fire handlers as tasks
        for handler in handlers:
            asyncio.ensure_future(
                self._safe_call(handler, event),
                loop=self._loop,
            )

    async def _safe_call(
        self, handler: EventHandler, event: ESLEvent
    ) -> None:
        """Call a handler, catching and logging exceptions."""
        try:
            await handler(event)
        except Exception as e:
            logger.error(
                "ESL event handler error [%s]: %s",
                event.event_name, e, exc_info=True
            )


# ---------------------------------------------------------------------------
# Main ESL client
# ---------------------------------------------------------------------------

class ESLClient:
    """
    Async FreeSWITCH ESL client (inbound mode).

    Connects to FreeSWITCH ESL port, authenticates, subscribes to events,
    and provides a full API for call control commands.

    Usage:
        client = ESLClient(host="127.0.0.1", port=8021, password="ClueCon")
        await client.connect()

        # Subscribe to events
        client.dispatcher.on_event("CHANNEL_HANGUP", my_hangup_handler)

        # Send commands
        await client.uuid_break(call_uuid)
        await client.uuid_kill(call_uuid)

        # Wait for specific event
        event = await client.dispatcher.wait_for_event(
            "CHANNEL_ANSWER", uuid=call_uuid
        )

        await client.disconnect()
    """

    def __init__(
        self,
        host:                     str   = "127.0.0.1",
        port:                     int   = 8021,
        password:                 str   = "ClueCon",
        connect_timeout:          float = 10.0,
        command_timeout:          float = 10.0,
        reconnect_enabled:        bool  = True,
        reconnect_interval:       float = 3.0,
        reconnect_max_attempts:   int   = 0,
        heartbeat_interval:       float = 30.0,
        subscribe_events:         Optional[List[str]] = None,
        on_connected:             Optional[Callable[[], Coroutine]] = None,
        on_disconnected:          Optional[Callable[[], Coroutine]] = None,
    ):
        self.host                   = host
        self.port                   = port
        self.password               = password
        self.connect_timeout        = connect_timeout
        self.command_timeout        = command_timeout
        self.reconnect_enabled      = reconnect_enabled
        self.reconnect_interval     = reconnect_interval
        self.reconnect_max_attempts = reconnect_max_attempts
        self.heartbeat_interval     = heartbeat_interval
        self.subscribe_events       = subscribe_events or [
            "CHANNEL_CREATE",
            "CHANNEL_ANSWER",
            "CHANNEL_HANGUP",
            "CHANNEL_HANGUP_COMPLETE",
            "CHANNEL_EXECUTE",
            "CHANNEL_EXECUTE_COMPLETE",
            "PLAYBACK_START",
            "PLAYBACK_STOP",
            "BACKGROUND_JOB",
            "HEARTBEAT",
            "DTMF",
        ]
        self._on_connected          = on_connected
        self._on_disconnected       = on_disconnected

        # Asyncio state
        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        self._reader: Optional[asyncio.StreamReader]      = None
        self._writer: Optional[asyncio.StreamWriter]      = None

        # State
        self._state       = ESLConnectionState.DISCONNECTED
        self._state_lock  = asyncio.Lock() if False else threading.Lock()

        # Parser
        self._parser      = ESLProtocolParser()

        # Dispatcher (created after loop is known)
        self._dispatcher: Optional[ESLEventDispatcher] = None

        # Pending commands queue (FIFO — ESL is sequential)
        self._command_queue: asyncio.Queue = None   # type: ignore
        self._pending_cmd:   Optional[ESLCommand] = None

        # Pending bgapi jobs
        self._bgapi_jobs: Dict[str, BGAPIJob] = {}

        # Stats
        self._commands_sent    = 0
        self._events_received  = 0
        self._reconnect_count  = 0
        self._connected_at:    Optional[float] = None
        self._last_heartbeat:  Optional[float] = None

        # Tasks
        self._reader_task:    Optional[asyncio.Task] = None
        self._writer_task:    Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

        self._stop_event:     Optional[asyncio.Event] = None

        logger.info(
            "ESLClient created: %s:%d reconnect=%s",
            host, port, reconnect_enabled
        )

    # -----------------------------------------------------------------------
    # Connection lifecycle
    # -----------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect to FreeSWITCH ESL and authenticate.
        Sets up background tasks for reading events and heartbeat.
        """
        self._loop = asyncio.get_event_loop()
        self._dispatcher = ESLEventDispatcher(self._loop)
        self._command_queue = asyncio.Queue()
        self._stop_event = asyncio.Event()

        await self._do_connect()

    async def _do_connect(self) -> None:
        """Internal: open TCP connection and authenticate."""
        self._set_state(ESLConnectionState.CONNECTING)
        logger.info("Connecting to FreeSWITCH ESL %s:%d ...", self.host, self.port)

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.connect_timeout,
            )
        except (OSError, asyncio.TimeoutError) as e:
            logger.error("ESL connect failed: %s", e)
            self._set_state(ESLConnectionState.DISCONNECTED)
            raise

        logger.info("ESL TCP connected to %s:%d", self.host, self.port)

        # Authenticate
        await self._authenticate()

        # Subscribe to events
        await self._subscribe_events()

        # Start background tasks
        self._parser.reset()
        self._reader_task = asyncio.ensure_future(self._reader_loop())
        self._writer_task = asyncio.ensure_future(self._writer_loop())
        self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())

        self._connected_at   = time.time()
        self._set_state(ESLConnectionState.CONNECTED)

        logger.info(
            "ESL connected and authenticated to %s:%d",
            self.host, self.port
        )

        if self._on_connected:
            asyncio.ensure_future(self._on_connected())

    async def _authenticate(self) -> None:
        """
        Handle ESL authentication handshake.
        FreeSWITCH sends: Content-Type: auth/request\\n\\n
        We reply:         auth <password>\\n\\n
        FreeSWITCH sends: Content-Type: command/reply\\nReply-Text: +OK accepted\\n\\n
        """
        self._set_state(ESLConnectionState.AUTHENTICATING)

        # Wait for auth/request
        auth_prompt = await self._read_raw_message()
        if not auth_prompt or "auth/request" not in auth_prompt.decode("utf-8", errors=""):
            raise ConnectionError("ESL did not send auth/request")

        # Send password
        auth_cmd = f"auth {self.password}\n\n".encode()
        self._writer.write(auth_cmd)
        await self._writer.drain()

        # Wait for reply
        reply_raw = await asyncio.wait_for(
            self._read_raw_message(),
            timeout=self.connect_timeout,
        )
        if not reply_raw:
            raise ConnectionError("No auth reply from ESL")

        reply_text = reply_raw.decode("utf-8", errors="replace")
        if "+OK accepted" not in reply_text:
            raise ConnectionError(
                f"ESL authentication failed: {reply_text!r}"
            )

        logger.info("ESL authentication successful")

    async def _read_raw_message(self) -> Optional[bytes]:
        """
        Read a single raw ESL message from the stream.
        Used only during authentication (before the reader loop starts).
        """
        buf = bytearray()
        while True:
            try:
                chunk = await asyncio.wait_for(
                    self._reader.read(4096),
                    timeout=self.connect_timeout,
                )
            except asyncio.TimeoutError:
                return None
            if not chunk:
                return None
            buf.extend(chunk)
            if b"\n\n" in buf:
                return bytes(buf)

    async def _subscribe_events(self) -> None:
        """Send event subscription command to FreeSWITCH."""
        events_str = " ".join(self.subscribe_events)
        cmd = f"event plain {events_str}\n\n"
        self._writer.write(cmd.encode())
        await self._writer.drain()
        logger.info("ESL event subscription sent: %s", events_str[:80])

    async def disconnect(self) -> None:
        """Gracefully disconnect from FreeSWITCH ESL."""
        logger.info("Disconnecting from ESL...")
        self._set_state(ESLConnectionState.STOPPING)

        if self._stop_event:
            self._stop_event.set()

        # Cancel tasks
        for task in [
            self._reader_task,
            self._writer_task,
            self._heartbeat_task,
            self._reconnect_task,
        ]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Close connection
        if self._writer:
            try:
                self._writer.write(b"exit\n\n")
                await self._writer.drain()
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass

        self._reader = None
        self._writer = None
        self._set_state(ESLConnectionState.DISCONNECTED)
        logger.info("ESL disconnected")

        if self._on_disconnected:
            asyncio.ensure_future(self._on_disconnected())

    # -----------------------------------------------------------------------
    # Background loops
    # -----------------------------------------------------------------------

    async def _reader_loop(self) -> None:
        """
        Continuously read data from ESL TCP stream and dispatch events.
        Runs as a background asyncio task.
        """
        logger.debug("ESL reader loop started")
        try:
            while not self._stop_event.is_set():
                try:
                    data = await asyncio.wait_for(
                        self._reader.read(65536),
                        timeout=60.0,
                    )
                except asyncio.TimeoutError:
                    # No data in 60s — check if connection is alive
                    logger.warning("ESL read timeout — connection may be dead")
                    await self._handle_disconnect("read timeout")
                    return

                if not data:
                    logger.warning("ESL connection closed by remote")
                    await self._handle_disconnect("remote closed")
                    return

                # Parse incoming bytes
                events = self._parser.feed(data)
                for event in events:
                    await self._handle_event(event)

        except asyncio.CancelledError:
            logger.debug("ESL reader loop cancelled")
            raise
        except Exception as e:
            logger.error("ESL reader loop error: %s", e, exc_info=True)
            await self._handle_disconnect(str(e))

    async def _writer_loop(self) -> None:
        """
        Drain the command queue and write commands to the ESL stream.
        Runs as a background asyncio task.
        Ensures sequential command/response matching.
        """
        logger.debug("ESL writer loop started")
        try:
            while not self._stop_event.is_set():
                try:
                    cmd: ESLCommand = await asyncio.wait_for(
                        self._command_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Set as pending (the reader loop will resolve it)
                self._pending_cmd = cmd

                try:
                    raw = (cmd.command + "\n\n").encode()
                    self._writer.write(raw)
                    await self._writer.drain()
                    self._commands_sent += 1
                    logger.debug("ESL >>> %s", cmd.command[:80])
                except Exception as e:
                    logger.error("ESL write error: %s", e)
                    if not cmd.future.done():
                        cmd.future.set_exception(e)
                    self._pending_cmd = None
                    await self._handle_disconnect(str(e))
                    return

        except asyncio.CancelledError:
            logger.debug("ESL writer loop cancelled")
            raise
        except Exception as e:
            logger.error("ESL writer loop error: %s", e, exc_info=True)

    async def _heartbeat_loop(self) -> None:
        """
        Periodically send a noevents command as a keepalive.
        If FreeSWITCH stops sending heartbeats, flag connection as dead.
        """
        logger.debug("ESL heartbeat loop started")
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(self.heartbeat_interval)

                if self._state != ESLConnectionState.CONNECTED:
                    continue

                # Check if we've received a heartbeat recently
                if self._last_heartbeat is not None:
                    age = time.monotonic() - self._last_heartbeat
                    if age > self.heartbeat_interval * 3:
                        logger.warning(
                            "ESL heartbeat missing for %.0fs — "
                            "connection may be dead",
                            age
                        )
                        await self._handle_disconnect("heartbeat timeout")
                        return

                # Send a keepalive (api status returns quickly)
                try:
                    await asyncio.wait_for(
                        self._raw_send("api status"),
                        timeout=5.0,
                    )
                except Exception:
                    pass

        except asyncio.CancelledError:
            logger.debug("ESL heartbeat loop cancelled")
            raise

    async def _handle_disconnect(self, reason: str) -> None:
        """Handle unexpected disconnection."""
        if self._state in (
            ESLConnectionState.STOPPING,
            ESLConnectionState.DISCONNECTED,
            ESLConnectionState.RECONNECTING,
        ):
            return

        logger.warning("ESL disconnected: %s", reason)
        self._set_state(ESLConnectionState.RECONNECTING)

        # Fail any pending command
        if self._pending_cmd and not self._pending_cmd.future.done():
            self._pending_cmd.future.set_exception(
                ConnectionError(f"ESL disconnected: {reason}")
            )
            self._pending_cmd = None

        # Fail any pending bgapi jobs
        for job in self._bgapi_jobs.values():
            if not job.future.done():
                job.future.set_exception(
                    ConnectionError(f"ESL disconnected: {reason}")
                )
        self._bgapi_jobs.clear()

        if self._on_disconnected:
            asyncio.ensure_future(self._on_disconnected())

        if self.reconnect_enabled:
            self._reconnect_task = asyncio.ensure_future(
                self._reconnect_loop()
            )

    async def _reconnect_loop(self) -> None:
        """Exponential backoff reconnection loop."""
        delay = self.reconnect_interval
        attempt = 0

        while not self._stop_event.is_set():
            attempt += 1
            self._reconnect_count += 1

            if (self.reconnect_max_attempts > 0
                    and attempt > self.reconnect_max_attempts):
                logger.error(
                    "ESL max reconnect attempts (%d) reached, giving up",
                    self.reconnect_max_attempts
                )
                self._set_state(ESLConnectionState.DISCONNECTED)
                return

            logger.info(
                "ESL reconnect attempt %d (delay=%.1fs)...",
                attempt, delay
            )

            try:
                # Clean up old tasks
                for task in [
                    self._reader_task,
                    self._writer_task,
                    self._heartbeat_task,
                ]:
                    if task and not task.done():
                        task.cancel()

                self._parser.reset()
                await self._do_connect()
                logger.info("ESL reconnected after %d attempts", attempt)
                return

            except Exception as e:
                logger.warning(
                    "ESL reconnect attempt %d failed: %s",
                    attempt, e
                )

            # Exponential backoff with jitter
            import random
            jitter = random.uniform(0, delay * 0.1)
            await asyncio.sleep(min(delay + jitter, RECONNECT_MAX_DELAY))
            delay = min(delay * 2, RECONNECT_MAX_DELAY)

    # -----------------------------------------------------------------------
    # Event handling
    # -----------------------------------------------------------------------

    async def _handle_event(self, event: ESLEvent) -> None:
        """Route incoming ESL event."""
        ct = event.content_type
        self._events_received += 1

        logger.debug("ESL <<< %s", event)

        # Command/reply — resolve pending command future
        if ct == "command/reply":
            await self._resolve_command(event)
            return

        # API response — also resolves pending command
        if ct == "api/response":
            await self._resolve_command(event)
            return

        # Auth request (happens on reconnect before auth is done)
        if ct == "auth/request":
            logger.debug("ESL received auth/request (unexpected after auth)")
            return

        # Disconnect notice
        if ct == "text/disconnect-notice":
            logger.warning("ESL received disconnect notice")
            await self._handle_disconnect("disconnect-notice")
            return

        # Regular event
        if ct in ("text/event-plain", "text/event-json", "text/event-xml"):
            event_name = event.event_name

            # Track heartbeats
            if event_name == "HEARTBEAT":
                self._last_heartbeat = time.monotonic()
                logger.debug("ESL heartbeat received")

            # Resolve background job futures
            if event_name == "BACKGROUND_JOB":
                await self._resolve_bgapi(event)

            # Dispatch to registered handlers
            self._dispatcher.dispatch(event)

    async def _resolve_command(self, event: ESLEvent) -> None:
        """Resolve the oldest pending command future."""
        if self._pending_cmd and not self._pending_cmd.future.done():
            self._pending_cmd.future.set_result(event)
        self._pending_cmd = None

    async def _resolve_bgapi(self, event: ESLEvent) -> None:
        """Resolve a pending bgapi job future by Job-UUID."""
        job_uuid = event.job_uuid
        if not job_uuid:
            return
        job = self._bgapi_jobs.pop(job_uuid, None)
        if job and not job.future.done():
            job.future.set_result(event)
            logger.debug(
                "BGAPIJob resolved: %s (%.1fms)",
                job_uuid[:8],
                (time.monotonic() - job.sent_at) * 1000,
            )

    # -----------------------------------------------------------------------
    # Command sending
    # -----------------------------------------------------------------------

    async def _raw_send(self, command: str) -> ESLEvent:
        """
        Send a raw ESL command and wait for the command/reply response.
        Commands are queued to ensure sequential execution.

        Args:
            command: ESL command string (without trailing \\n\\n)

        Returns:
            ESLEvent with the command/reply
        """
        if self._state not in (
            ESLConnectionState.CONNECTED,
            ESLConnectionState.AUTHENTICATING,
        ):
            raise ConnectionError(
                f"ESL not connected (state={self._state.name})"
            )

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        cmd = ESLCommand(
            command=command,
            future=future,
            timeout=self.command_timeout,
        )

        await self._command_queue.put(cmd)

        try:
            result = await asyncio.wait_for(
                future,
                timeout=self.command_timeout + 1.0,
            )
            return result
        except asyncio.TimeoutError:
            logger.error("ESL command timed out: %s", command[:60])
            raise TimeoutError(f"ESL command timed out: {command[:60]}")

    async def send_api(self, command: str) -> str:
        """
        Send a synchronous API command.
        Returns the response body as a string.

        Example:
            result = await esl.send_api("uuid_kill <uuid>")
        """
        event = await self._raw_send(f"api {command}")
        return event.body or event.reply_text

    async def send_bgapi(
        self,
        command: str,
        timeout: float = BGAPI_TIMEOUT,
    ) -> ESLEvent:
        """
        Send a background API command.
        Returns when the BACKGROUND_JOB event arrives.

        Example:
            result = await esl.send_bgapi("originate sofia/... &park()")
        """
        job_uuid_str = str(uuid.uuid4())

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        job = BGAPIJob(
            job_uuid=job_uuid_str,
            command=command,
            future=future,
            timeout=timeout,
        )
        self._bgapi_jobs[job_uuid_str] = job

        await self._raw_send(
            f"bgapi {command}\nJob-UUID: {job_uuid_str}"
        )

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self._bgapi_jobs.pop(job_uuid_str, None)
            raise TimeoutError(
                f"bgapi timeout after {timeout}s: {command[:60]}"
            )

    async def send_msg(
        self,
        uuid: str,
        headers: Dict[str, str],
        body: str = "",
    ) -> ESLEvent:
        """
        Send a sendmsg command to a specific channel.

        Args:
            uuid:    Channel UUID
            headers: Message headers dict
            body:    Optional message body
        """
        header_lines = "\n".join(f"{k}: {v}" for k, v in headers.items())
        if body:
            cmd = (
                f"sendmsg {uuid}\n"
                f"{header_lines}\n"
                f"Content-Length: {len(body.encode())}\n\n"
                f"{body}"
            )
        else:
            cmd = f"sendmsg {uuid}\n{header_lines}"

        return await self._raw_send(cmd)

    # -----------------------------------------------------------------------
    # Call control commands
    # -----------------------------------------------------------------------

    async def uuid_break(self, call_uuid: str, all: bool = True) -> bool:
        """
        Immediately stop audio playback on a channel.
        This is the primary mechanism for interrupting bot speech.

        Args:
            call_uuid: FreeSWITCH channel UUID
            all:       If True, break all queued audio (not just current)

        Returns:
            True if command succeeded
        """
        flag = "all" if all else ""
        cmd  = f"uuid_break {call_uuid} {flag}".strip()
        try:
            result = await self.send_api(cmd)
            ok = "+OK" in (result or "")
            if ok:
                logger.info("uuid_break OK: %s", call_uuid[:8])
            else:
                logger.warning("uuid_break failed: %s → %s", call_uuid[:8], result)
            return ok
        except Exception as e:
            logger.error("uuid_break error [%s]: %s", call_uuid[:8], e)
            return False

    async def uuid_kill(
        self,
        call_uuid: str,
        cause: str = "NORMAL_CLEARING",
    ) -> bool:
        """
        Hang up a call.

        Args:
            call_uuid: FreeSWITCH channel UUID
            cause:     Hangup cause code (SIP response reason)
        """
        cmd = f"uuid_kill {call_uuid} {cause}"
        try:
            result = await self.send_api(cmd)
            ok = "+OK" in (result or "")
            logger.info(
                "uuid_kill %s: %s (cause=%s)",
                "OK" if ok else "FAIL",
                call_uuid[:8], cause
            )
            return ok
        except Exception as e:
            logger.error("uuid_kill error [%s]: %s", call_uuid[:8], e)
            return False

    async def uuid_setvar(
        self,
        call_uuid: str,
        variable: str,
        value: str,
    ) -> bool:
        """
        Set a channel variable.

        Example:
            await esl.uuid_setvar(uuid, "record_session", "true")
        """
        cmd = f"uuid_setvar {call_uuid} {variable} {value}"
        try:
            result = await self.send_api(cmd)
            return "+OK" in (result or "")
        except Exception as e:
            logger.error(
                "uuid_setvar error [%s] %s=%s: %s",
                call_uuid[:8], variable, value, e
            )
            return False

    async def uuid_setvar_multi(
        self,
        call_uuid: str,
        variables: Dict[str, str],
    ) -> bool:
        """
        Set multiple channel variables at once.
        More efficient than calling uuid_setvar in a loop.
        """
        pairs = ";".join(f"{k}={v}" for k, v in variables.items())
        cmd   = f"uuid_setvar_multi {call_uuid} {pairs}"
        try:
            result = await self.send_api(cmd)
            return "+OK" in (result or "")
        except Exception as e:
            logger.error(
                "uuid_setvar_multi error [%s]: %s",
                call_uuid[:8], e
            )
            return False

    async def uuid_getvar(
        self,
        call_uuid: str,
        variable: str,
    ) -> Optional[str]:
        """Get a channel variable value."""
        cmd = f"uuid_getvar {call_uuid} {variable}"
        try:
            result = await self.send_api(cmd)
            if result and not result.startswith("-ERR"):
                return result.strip()
            return None
        except Exception as e:
            logger.error(
                "uuid_getvar error [%s] %s: %s",
                call_uuid[:8], variable, e
            )
            return None

    async def uuid_answer(self, call_uuid: str) -> bool:
        """Answer an incoming call."""
        cmd = f"uuid_answer {call_uuid}"
        try:
            result = await self.send_api(cmd)
            return "+OK" in (result or "")
        except Exception as e:
            logger.error("uuid_answer error [%s]: %s", call_uuid[:8], e)
            return False

    async def uuid_hold(self, call_uuid: str) -> bool:
        """Put a call on hold."""
        try:
            result = await self.send_api(f"uuid_hold {call_uuid}")
            return "+OK" in (result or "")
        except Exception as e:
            logger.error("uuid_hold error [%s]: %s", call_uuid[:8], e)
            return False

    async def uuid_unhold(self, call_uuid: str) -> bool:
        """Take a call off hold."""
        try:
            result = await self.send_api(f"uuid_hold off {call_uuid}")
            return "+OK" in (result or "")
        except Exception as e:
            logger.error("uuid_unhold error [%s]: %s", call_uuid[:8], e)
            return False

    async def uuid_transfer(
        self,
        call_uuid: str,
        destination: str,
        dialplan: str = "XML",
        context: str = "default",
    ) -> bool:
        """
        Transfer a call to another destination.

        Args:
            call_uuid:   Channel UUID
            destination: Extension or SIP URI
            dialplan:    Dialplan type (XML, Lua, etc.)
            context:     Dialplan context
        """
        cmd = f"uuid_transfer {call_uuid} {destination} {dialplan} {context}"
        try:
            result = await self.send_api(cmd)
            ok = "+OK" in (result or "")
            logger.info(
                "uuid_transfer %s: %s → %s",
                "OK" if ok else "FAIL",
                call_uuid[:8], destination
            )
            return ok
        except Exception as e:
            logger.error(
                "uuid_transfer error [%s] → %s: %s",
                call_uuid[:8], destination, e
            )
            return False

    async def uuid_bridge(
        self,
        uuid_a: str,
        uuid_b: str,
    ) -> bool:
        """Bridge two call legs together."""
        cmd = f"uuid_bridge {uuid_a} {uuid_b}"
        try:
            result = await self.send_api(cmd)
            return "+OK" in (result or "")
        except Exception as e:
            logger.error(
                "uuid_bridge error [%s <-> %s]: %s",
                uuid_a[:8], uuid_b[:8], e
            )
            return False

    async def uuid_record(
        self,
        call_uuid: str,
        file_path: str,
        action: str = "start",
        limit: int  = 0,
    ) -> bool:
        """
        Start or stop call recording.

        Args:
            call_uuid: Channel UUID
            file_path: Path for recording file
            action:    "start" | "stop" | "pause" | "resume"
            limit:     Max recording duration in seconds (0 = unlimited)
        """
        cmd = f"uuid_record {call_uuid} {action} {file_path}"
        if action == "start" and limit > 0:
            cmd += f" {limit}"
        try:
            result = await self.send_api(cmd)
            ok = "+OK" in (result or "")
            logger.info(
                "uuid_record %s %s: %s",
                action, "OK" if ok else "FAIL", call_uuid[:8]
            )
            return ok
        except Exception as e:
            logger.error(
                "uuid_record error [%s]: %s", call_uuid[:8], e
            )
            return False

    async def uuid_play(
        self,
        call_uuid: str,
        file_path: str,
        loop: int = 1,
    ) -> bool:
        """
        Play an audio file on a channel.

        Args:
            call_uuid: Channel UUID
            file_path: Path to audio file (wav, mp3, etc.)
            loop:      Number of times to repeat (1 = once)
        """
        cmd = f"uuid_displace {call_uuid} start {file_path}"
        if loop > 1:
            cmd += f" {loop}"
        try:
            result = await self.send_api(cmd)
            return "+OK" in (result or "")
        except Exception as e:
            logger.error("uuid_play error [%s]: %s", call_uuid[:8], e)
            return False

    async def uuid_send_dtmf(
        self,
        call_uuid: str,
        dtmf_digits: str,
        duration_ms: int = 100,
    ) -> bool:
        """
        Send DTMF tones on a channel.

        Args:
            call_uuid:   Channel UUID
            dtmf_digits: Digits to send (0-9, *, #, A-D)
            duration_ms: Duration of each tone in milliseconds
        """
        cmd = f"uuid_send_dtmf {call_uuid} {dtmf_digits}@{duration_ms}"
        try:
            result = await self.send_api(cmd)
            return "+OK" in (result or "")
        except Exception as e:
            logger.error(
                "uuid_send_dtmf error [%s]: %s", call_uuid[:8], e
            )
            return False

    async def uuid_execute(
        self,
        call_uuid: str,
        app: str,
        arg: str = "",
        event_lock: bool = False,
    ) -> ESLEvent:
        """
        Execute a dialplan application on a channel.

        Args:
            call_uuid:  Channel UUID
            app:        Application name (e.g. "playback", "sleep", "speak")
            arg:        Application arguments
            event_lock: Wait for CHANNEL_EXECUTE_COMPLETE before returning

        Example:
            await esl.uuid_execute(uuid, "sleep", "1000")
            await esl.uuid_execute(uuid, "playback", "/tmp/greeting.wav")
        """
        headers: Dict[str, str] = {
            "call-command": "execute",
            "execute-app-name": app,
        }
        if arg:
            headers["execute-app-arg"] = arg
        if event_lock:
            headers["event-lock"] = "true"

        if event_lock:
            # Wait for CHANNEL_EXECUTE_COMPLETE
            fut = self._dispatcher.wait_for_event(
                "CHANNEL_EXECUTE_COMPLETE",
                uuid=call_uuid,
                timeout=30.0,
            )

        result = await self.send_msg(call_uuid, headers)

        if event_lock:
            try:
                complete_event = await fut
                return complete_event
            except asyncio.TimeoutError:
                logger.warning(
                    "uuid_execute timeout waiting for CHANNEL_EXECUTE_COMPLETE: "
                    "%s %s",
                    app, call_uuid[:8]
                )
                return result

        return result

    async def uuid_flush_dtmf(self, call_uuid: str) -> bool:
        """Flush pending DTMF digits from a channel."""
        try:
            result = await self.send_api(f"uuid_flush_dtmf {call_uuid}")
            return "+OK" in (result or "")
        except Exception as e:
            logger.error("uuid_flush_dtmf error [%s]: %s", call_uuid[:8], e)
            return False

    async def get_channel_info(self, call_uuid: str) -> Optional[Dict[str, str]]:
        """
        Get all channel variables for a UUID.
        Returns a dict of variable names to values.
        """
        try:
            result = await self.send_api(f"uuid_dump {call_uuid}")
            if not result or result.startswith("-ERR"):
                return None
            info: Dict[str, str] = {}
            for line in result.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, _, value = line.partition(":")
                    info[key.strip()] = value.strip()
            return info
        except Exception as e:
            logger.error(
                "get_channel_info error [%s]: %s", call_uuid[:8], e
            )
            return None

    async def show_calls(self) -> List[Dict[str, str]]:
        """
        List all active calls on FreeSWITCH.
        Returns list of call info dicts.
        """
        try:
            result = await self.send_api("show calls as json")
            if not result:
                return []
            import json
            data = json.loads(result)
            if isinstance(data, dict):
                return data.get("rows", [])
            return []
        except Exception as e:
            logger.error("show_calls error: %s", e)
            return []

    async def global_getvar(self, variable: str) -> Optional[str]:
        """Get a global FreeSWITCH variable."""
        try:
            result = await self.send_api(f"global_getvar {variable}")
            if result and not result.startswith("-ERR"):
                return result.strip()
            return None
        except Exception as e:
            logger.error("global_getvar error [%s]: %s", variable, e)
            return None

    async def fs_status(self) -> Optional[str]:
        """Get FreeSWITCH system status."""
        try:
            return await self.send_api("status")
        except Exception as e:
            logger.error("fs_status error: %s", e)
            return None

    async def reload_xml(self) -> bool:
        """Reload FreeSWITCH XML configuration."""
        try:
            result = await self.send_api("reloadxml")
            return "+OK" in (result or "")
        except Exception as e:
            logger.error("reload_xml error: %s", e)
            return False

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _set_state(self, state: ESLConnectionState) -> None:
        old = self._state
        self._state = state
        if old != state:
            logger.info(
                "ESL state: %s → %s",
                old.name, state.name
            )

    @property
    def is_connected(self) -> bool:
        return self._state == ESLConnectionState.CONNECTED

    @property
    def dispatcher(self) -> ESLEventDispatcher:
        if self._dispatcher is None:
            raise RuntimeError("ESLClient not started — call connect() first")
        return self._dispatcher

    @property
    def connection_info(self) -> ESLConnectionInfo:
        return ESLConnectionInfo(
            state=self._state,
            host=self.host,
            port=self.port,
            connected_at=self._connected_at,
            reconnect_attempts=self._reconnect_count,
            commands_sent=self._commands_sent,
            events_received=self._events_received,
            last_heartbeat=self._last_heartbeat,
        )


# ---------------------------------------------------------------------------
# ESL client factory from AppConfig
# ---------------------------------------------------------------------------

def create_esl_client(cfg: "Any") -> ESLClient:
    """
    Create an ESLClient from AppConfig.

    Args:
        cfg: AppConfig instance (from config.py)

    Returns:
        Configured ESLClient (not yet connected)
    """
    esl_cfg = cfg.esl
    return ESLClient(
        host=esl_cfg.host,
        port=esl_cfg.port,
        password=esl_cfg.password,
        connect_timeout=esl_cfg.connect_timeout,
        command_timeout=esl_cfg.command_timeout,
        reconnect_enabled=esl_cfg.reconnect_enabled,
        reconnect_interval=esl_cfg.reconnect_interval_seconds,
        reconnect_max_attempts=esl_cfg.reconnect_max_attempts,
        heartbeat_interval=esl_cfg.heartbeat_interval_seconds,
        subscribe_events=esl_cfg.subscribe_events,
    )


# ---------------------------------------------------------------------------
# Convenience: wait for call answer with timeout
# ---------------------------------------------------------------------------

async def wait_for_answer(
    esl: ESLClient,
    call_uuid: str,
    timeout: float = 30.0,
) -> bool:
    """
    Wait for a call to be answered.

    Args:
        esl:       ESLClient instance
        call_uuid: Channel UUID to monitor
        timeout:   Max seconds to wait

    Returns:
        True if answered, False if timed out or hung up
    """
    try:
        event = await esl.dispatcher.wait_for_event(
            "CHANNEL_ANSWER",
            uuid=call_uuid,
            timeout=timeout,
        )
        logger.info("Call answered: %s", call_uuid[:8])
        return True
    except asyncio.TimeoutError:
        logger.warning(
            "Timeout waiting for answer: %s (%.0fs)",
            call_uuid[:8], timeout
        )
        return False


async def wait_for_hangup(
    esl: ESLClient,
    call_uuid: str,
    timeout: float = 3600.0,
) -> Optional[str]:
    """
    Wait for a call to hang up.

    Returns:
        Hangup cause string, or None if timed out
    """
    try:
        event = await esl.dispatcher.wait_for_event(
            "CHANNEL_HANGUP_COMPLETE",
            uuid=call_uuid,
            timeout=timeout,
        )
        cause = event.get("Hangup-Cause", "UNKNOWN")
        logger.info(
            "Call hung up: %s (cause=%s)",
            call_uuid[:8], cause
        )
        return cause
    except asyncio.TimeoutError:
        return None
```

---

`esl.py` 完成 ✅ 约 **780行**

包含：

| 组件 | 说明 |
|------|------|
| `ESLProtocolParser` | 增量式TCP流解析，处理粘包/分包，支持嵌套event body |
| `ESLEventDispatcher` | 事件路由：按名称/UUID/通配符订阅，one-shot Future支持 |
| `ESLClient` | 完整ESL客户端，reader/writer/heartbeat三个后台task |
| `_authenticate` | ESL握手流程 |
| `_reconnect_loop` | 指数退避自动重连 |
| `send_api / send_bgapi` | 同步/异步API命令 |
| `uuid_break` | **打断核心命令**，立即停止FS播放 |
| `uuid_kill / uuid_transfer / uuid_bridge` | 完整通话控制 |
| `uuid_execute` | dialplan应用执行，支持event-lock等待 |
| `uuid_record` | 通话录音 |
| `wait_for_answer / wait_for_hangup` | 便利等待函数 |

---

**下一个：`audio.py`** — WebSocket音频服务器，接收mod_audio_stream的原始PCM，管理每个通话的音频流，发送TTS音频回FreeSWITCH。准备好就说继续！
