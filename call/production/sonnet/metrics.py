"""
metrics.py - Production-grade metrics collection and reporting

Handles:
  - Per-session metrics tracking (STT/TTS/LLM/E2E latency timers)
  - System-wide rolling window statistics
  - Rate counters (calls/s, errors/s)
  - Prometheus counters, histograms, gauges
  - Background log reporter
  - Health checker
  - Outbound call metrics
  - Campaign metrics

Used by:
  - session.py    : per-session timing
  - outbound.py   : outbound call tracking
  - api.py        : metrics endpoint
  - main.py       : init and lifecycle
"""

import logging
import math
import os
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        start_http_server, CollectorRegistry,
        generate_latest, CONTENT_TYPE_LATEST,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — Prometheus disabled")

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Rolling window statistics
# ---------------------------------------------------------------------------

class RollingStats:
    """
    Thread-safe rolling window statistics.
    Keeps samples from the last `window_seconds` seconds.
    Computes count/mean/median/p95/p99/min/max/stddev on demand.
    """

    def __init__(
        self,
        window_seconds: int = 60,
        max_samples:    int = 10_000,
    ):
        self._window   = window_seconds
        self._max      = max_samples
        self._samples: deque = deque()   # (monotonic_ts, value)
        self._lock     = threading.Lock()

    def record(self, value: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._samples.append((now, value))
            self._evict(now)
            while len(self._samples) > self._max:
                self._samples.popleft()

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _values(self) -> List[float]:
        now    = time.monotonic()
        cutoff = now - self._window
        return [v for ts, v in self._samples if ts >= cutoff]

    def count(self) -> int:
        with self._lock:
            return len(self._values())

    def mean(self) -> float:
        with self._lock:
            v = self._values()
            return statistics.mean(v) if v else 0.0

    def median(self) -> float:
        with self._lock:
            v = self._values()
            return statistics.median(v) if v else 0.0

    def p95(self) -> float:
        with self._lock:
            v = sorted(self._values())
            if not v:
                return 0.0
            return v[min(int(len(v) * 0.95), len(v) - 1)]

    def p99(self) -> float:
        with self._lock:
            v = sorted(self._values())
            if not v:
                return 0.0
            return v[min(int(len(v) * 0.99), len(v) - 1)]

    def minimum(self) -> float:
        with self._lock:
            v = self._values()
            return min(v) if v else 0.0

    def maximum(self) -> float:
        with self._lock:
            v = self._values()
            return max(v) if v else 0.0

    def stddev(self) -> float:
        with self._lock:
            v = self._values()
            return statistics.stdev(v) if len(v) >= 2 else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "count":  self.count(),
            "mean":   round(self.mean(),   3),
            "median": round(self.median(), 3),
            "p95":    round(self.p95(),    3),
            "p99":    round(self.p99(),    3),
            "min":    round(self.minimum(),3),
            "max":    round(self.maximum(),3),
            "stddev": round(self.stddev(), 3),
        }


class RateCounter:
    """
    Thread-safe sliding window rate counter.
    Tracks events per second over a rolling window.
    """

    def __init__(self, window_seconds: int = 60):
        self._window  = window_seconds
        self._events: deque = deque()   # monotonic timestamps
        self._lock    = threading.Lock()
        self._total   = 0

    def increment(self, n: int = 1) -> None:
        now = time.monotonic()
        with self._lock:
            for _ in range(n):
                self._events.append(now)
            self._total += n
            self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    def rate(self) -> float:
        """Events per second over the window."""
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            return len(self._events) / self._window

    def total(self) -> int:
        with self._lock:
            return self._total

    def window_count(self) -> int:
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            return len(self._events)


# ---------------------------------------------------------------------------
# Per-session metrics
# ---------------------------------------------------------------------------

class SessionMetrics:
    """
    Tracks all timing and counter metrics for a single call session.

    Usage:
        sm = SessionMetrics(session_id, call_uuid)

        sm.stt_start()
        # ... do STT ...
        latency_ms = sm.stt_end()

        sm.llm_start()
        # ... call LLM ...
        latency_ms = sm.llm_end()

        sm.e2e_start()    # when user stops speaking
        sm.e2e_end()      # when bot starts playing audio
    """

    def __init__(self, session_id: str, call_uuid: str):
        self.session_id  = session_id
        self.call_uuid   = call_uuid
        self.created_at  = time.time()
        self.ended_at:   Optional[float] = None
        self.final_state = "active"

        self._lock = threading.Lock()

        # Counters
        self.stt_requests    = 0
        self.tts_requests    = 0
        self.llm_requests    = 0
        self.interruptions   = 0
        self.errors          = 0

        # Latency samples (ms)
        self.stt_latencies_ms: List[float] = []
        self.tts_latencies_ms: List[float] = []
        self.llm_latencies_ms: List[float] = []
        self.e2e_latencies_ms: List[float] = []

        # Audio bytes
        self.audio_bytes_in  = 0
        self.audio_bytes_out = 0

        # Speaking durations (ms)
        self.user_speaking_ms = 0.0
        self.bot_speaking_ms  = 0.0

        # Active timers (monotonic start times)
        self._stt_start:   Optional[float] = None
        self._tts_start:   Optional[float] = None
        self._llm_start:   Optional[float] = None
        self._e2e_start:   Optional[float] = None
        self._user_sp_start: Optional[float] = None
        self._bot_sp_start:  Optional[float] = None

    # ---- STT ----

    def stt_start(self) -> None:
        with self._lock:
            self._stt_start = time.monotonic()

    def stt_end(self, success: bool = True) -> float:
        with self._lock:
            if self._stt_start is None:
                return 0.0
            ms = (time.monotonic() - self._stt_start) * 1000
            self._stt_start = None
            if success:
                self.stt_requests += 1
                self.stt_latencies_ms.append(ms)
            else:
                self.errors += 1
            return ms

    # ---- TTS ----

    def tts_start(self) -> None:
        with self._lock:
            self._tts_start = time.monotonic()

    def tts_end(self, success: bool = True) -> float:
        with self._lock:
            if self._tts_start is None:
                return 0.0
            ms = (time.monotonic() - self._tts_start) * 1000
            self._tts_start = None
            if success:
                self.tts_requests += 1
                self.tts_latencies_ms.append(ms)
            else:
                self.errors += 1
            return ms

    # ---- LLM ----

    def llm_start(self) -> None:
        with self._lock:
            self._llm_start = time.monotonic()

    def llm_end(self, success: bool = True) -> float:
        with self._lock:
            if self._llm_start is None:
                return 0.0
            ms = (time.monotonic() - self._llm_start) * 1000
            self._llm_start = None
            if success:
                self.llm_requests += 1
                self.llm_latencies_ms.append(ms)
            else:
                self.errors += 1
            return ms

    # ---- E2E (user stops speaking → bot starts playing) ----

    def e2e_start(self) -> None:
        with self._lock:
            self._e2e_start = time.monotonic()

    def e2e_end(self) -> float:
        with self._lock:
            if self._e2e_start is None:
                return 0.0
            ms = (time.monotonic() - self._e2e_start) * 1000
            self._e2e_start = None
            self.e2e_latencies_ms.append(ms)
            return ms

    # ---- Speaking duration ----

    def user_speech_start(self) -> None:
        with self._lock:
            self._user_sp_start = time.monotonic()

    def user_speech_end(self) -> None:
        with self._lock:
            if self._user_sp_start is not None:
                self.user_speaking_ms += (
                    time.monotonic() - self._user_sp_start
                ) * 1000
                self._user_sp_start = None

    def bot_speech_start(self) -> None:
        with self._lock:
            self._bot_sp_start = time.monotonic()

    def bot_speech_end(self) -> None:
        with self._lock:
            if self._bot_sp_start is not None:
                self.bot_speaking_ms += (
                    time.monotonic() - self._bot_sp_start
                ) * 1000
                self._bot_sp_start = None

    # ---- Audio bytes ----

    def add_audio_in(self, n: int) -> None:
        with self._lock:
            self.audio_bytes_in += n

    def add_audio_out(self, n: int) -> None:
        with self._lock:
            self.audio_bytes_out += n

    # ---- Other events ----

    def record_interruption(self) -> None:
        with self._lock:
            self.interruptions += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    # ---- Lifecycle ----

    def finalize(self, final_state: str = "completed") -> None:
        with self._lock:
            self.ended_at    = time.time()
            self.final_state = final_state
            # Close any open timers
            now = time.monotonic()
            if self._user_sp_start:
                self.user_speaking_ms += (now - self._user_sp_start) * 1000
                self._user_sp_start = None
            if self._bot_sp_start:
                self.bot_speaking_ms += (now - self._bot_sp_start) * 1000
                self._bot_sp_start = None

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at or time.time()
        return end - self.created_at

    def _safe_stats(self, samples: List[float]) -> Dict:
        if not samples:
            return {"count": 0, "mean": 0, "p95": 0, "p99": 0,
                    "min": 0, "max": 0}
        s = sorted(samples)
        return {
            "count": len(s),
            "mean":  round(statistics.mean(s), 1),
            "p95":   round(s[min(int(len(s)*0.95), len(s)-1)], 1),
            "p99":   round(s[min(int(len(s)*0.99), len(s)-1)], 1),
            "min":   round(s[0], 1),
            "max":   round(s[-1], 1),
        }

    def to_dict(self) -> Dict:
        with self._lock:
            return {
                "session_id":      self.session_id,
                "call_uuid":       self.call_uuid,
                "final_state":     self.final_state,
                "created_at":      self.created_at,
                "ended_at":        self.ended_at,
                "duration_seconds":round(self.duration_seconds, 2),
                "counters": {
                    "stt_requests":  self.stt_requests,
                    "tts_requests":  self.tts_requests,
                    "llm_requests":  self.llm_requests,
                    "interruptions": self.interruptions,
                    "errors":        self.errors,
                },
                "latency_ms": {
                    "stt": self._safe_stats(self.stt_latencies_ms),
                    "tts": self._safe_stats(self.tts_latencies_ms),
                    "llm": self._safe_stats(self.llm_latencies_ms),
                    "e2e": self._safe_stats(self.e2e_latencies_ms),
                },
                "audio": {
                    "bytes_in":  self.audio_bytes_in,
                    "bytes_out": self.audio_bytes_out,
                },
                "speaking_ms": {
                    "user": round(self.user_speaking_ms, 1),
                    "bot":  round(self.bot_speaking_ms, 1),
                },
            }


# ---------------------------------------------------------------------------
# Prometheus metrics wrapper
# ---------------------------------------------------------------------------

class PrometheusMetrics:
    """
    All Prometheus metrics for VoiceBot.
    Safe to use even if prometheus_client is not installed
    (all methods become no-ops).
    """

    def __init__(self, namespace: str = "voicebot"):
        self._ns        = namespace
        self._available = _PROMETHEUS_AVAILABLE
        if not self._available:
            return
        self._define_metrics()

    def _define_metrics(self) -> None:
        ns = self._ns

        # ---- Counters ----
        self.calls_total = Counter(
            f"{ns}_calls_total",
            "Total calls handled",
            ["direction", "result"],
        )
        self.interruptions_total = Counter(
            f"{ns}_interruptions_total",
            "Total user interruptions",
        )
        self.errors_total = Counter(
            f"{ns}_errors_total",
            "Total errors",
            ["component", "type"],
        )
        self.audio_bytes_total = Counter(
            f"{ns}_audio_bytes_total",
            "Total audio bytes processed",
            ["direction"],
        )

        # Outbound specific
        self.outbound_calls_total = Counter(
            f"{ns}_outbound_calls_total",
            "Total outbound calls placed",
            ["result"],   # answered/no_answer/busy/failed
        )
        self.outbound_retries_total = Counter(
            f"{ns}_outbound_retries_total",
            "Total outbound call retries",
        )

        # ---- Histograms ----
        e2e_buckets = [
            0.1, 0.2, 0.3, 0.5, 0.75,
            1.0, 1.5, 2.0, 3.0, 5.0, 8.0,
        ]
        self.e2e_latency = Histogram(
            f"{ns}_e2e_latency_seconds",
            "End-to-end latency: user silence → bot audio",
            buckets=e2e_buckets,
        )

        llm_buckets = [
            0.1, 0.2, 0.5, 0.75, 1.0,
            1.5, 2.0, 3.0, 5.0, 10.0,
        ]
        self.llm_latency = Histogram(
            f"{ns}_llm_latency_seconds",
            "LLM / AI model response latency",
            ["provider", "model"],
            buckets=llm_buckets,
        )

        self.call_duration = Histogram(
            f"{ns}_call_duration_seconds",
            "Total call duration",
            ["direction"],
            buckets=[
                10, 30, 60, 120, 180,
                300, 600, 1200, 1800, 3600,
            ],
        )

        self.ring_duration = Histogram(
            f"{ns}_ring_duration_seconds",
            "Outbound ring duration (dial → answer)",
            buckets=[5, 10, 15, 20, 25, 30, 45, 60],
        )

        # ---- Gauges ----
        self.active_calls = Gauge(
            f"{ns}_active_calls",
            "Currently active calls",
            ["direction"],
        )
        self.active_campaigns = Gauge(
            f"{ns}_active_campaigns",
            "Currently active outbound campaigns",
        )
        self.esl_connected = Gauge(
            f"{ns}_esl_connected",
            "ESL connection status (1=connected)",
        )
        self.queue_depth = Gauge(
            f"{ns}_queue_depth",
            "Outbound call queue depth",
            ["campaign_id"],
        )
        self.memory_bytes = Gauge(
            f"{ns}_memory_bytes",
            "Process RSS memory usage",
        )
        self.cpu_percent = Gauge(
            f"{ns}_cpu_percent",
            "Process CPU usage percent",
        )
        self.open_fds = Gauge(
            f"{ns}_open_file_descriptors",
            "Number of open file descriptors",
        )

    # ---- Public methods (safe no-ops if prometheus unavailable) ----

    def call_started(self, direction: str = "inbound") -> None:
        if not self._available:
            return
        self.active_calls.labels(direction=direction).inc()

    def call_ended(
        self,
        direction:   str,
        result:      str,
        duration_s:  float,
    ) -> None:
        if not self._available:
            return
        self.active_calls.labels(direction=direction).dec()
        self.calls_total.labels(
            direction=direction, result=result
        ).inc()
        self.call_duration.labels(direction=direction).observe(
            duration_s
        )

    def outbound_result(self, result: str) -> None:
        if not self._available:
            return
        self.outbound_calls_total.labels(result=result).inc()

    def outbound_retry(self) -> None:
        if not self._available:
            return
        self.outbound_retries_total.inc()

    def observe_e2e(self, latency_ms: float) -> None:
        if not self._available:
            return
        self.e2e_latency.observe(latency_ms / 1000)

    def observe_llm(
        self,
        latency_ms: float,
        provider:   str = "minicpm",
        model:      str = "minicpmo-4.5",
    ) -> None:
        if not self._available:
            return
        self.llm_latency.labels(
            provider=provider, model=model
        ).observe(latency_ms / 1000)

    def observe_ring(self, duration_s: float) -> None:
        if not self._available:
            return
        self.ring_duration.observe(duration_s)

    def record_interruption(self) -> None:
        if not self._available:
            return
        self.interruptions_total.inc()

    def record_error(
        self,
        component: str,
        err_type:  str = "unknown",
    ) -> None:
        if not self._available:
            return
        self.errors_total.labels(
            component=component, type=err_type
        ).inc()

    def add_audio_bytes(self, direction: str, n: int) -> None:
        if not self._available:
            return
        self.audio_bytes_total.labels(direction=direction).inc(n)

    def set_esl_connected(self, connected: bool) -> None:
        if not self._available:
            return
        self.esl_connected.set(1 if connected else 0)

    def set_active_campaigns(self, n: int) -> None:
        if not self._available:
            return
        self.active_campaigns.set(n)

    def set_queue_depth(
        self, campaign_id: str, depth: int
    ) -> None:
        if not self._available:
            return
        self.queue_depth.labels(campaign_id=campaign_id).set(depth)

    def update_process_stats(self) -> None:
        """Update memory/CPU/FD gauges from psutil."""
        if not self._available or not _PSUTIL_AVAILABLE:
            return
        try:
            proc = psutil.Process(os.getpid())
            self.memory_bytes.set(proc.memory_info().rss)
            self.cpu_percent.set(proc.cpu_percent(interval=None))
            self.open_fds.set(proc.num_fds())
        except Exception:
            pass

    def is_available(self) -> bool:
        return self._available


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class Timer:
    """
    Context manager for timing code blocks.
    Records to both SessionMetrics and global RollingStats.

    Usage:
        with Timer("llm", session_metrics=sm, registry=reg,
                   provider="minicpm", model="minicpmo-4.5"):
            response = await ai.generate()
        # latency automatically recorded on __exit__
    """

    def __init__(
        self,
        metric:          str,
        session_metrics: Optional[SessionMetrics]  = None,
        registry:        Optional["MetricsRegistry"] = None,
        provider:        str   = "default",
        model:           str   = "default",
        success:         bool  = True,
    ):
        self.metric          = metric
        self.session_metrics = session_metrics
        self.registry        = registry
        self.provider        = provider
        self.model           = model
        self.success         = success
        self.latency_ms      = 0.0
        self._start:         Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        if self.session_metrics:
            fn = getattr(self.session_metrics, f"{self.metric}_start", None)
            if fn:
                fn()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.success = exc_type is None
        if self._start is not None:
            self.latency_ms = (time.monotonic() - self._start) * 1000

        if self.session_metrics:
            fn = getattr(
                self.session_metrics, f"{self.metric}_end", None
            )
            if fn:
                fn(success=self.success)

        if self.registry:
            if self.metric == "llm":
                self.registry.observe_llm(
                    self.latency_ms,
                    self.provider,
                    self.model,
                    self.success,
                )
            elif self.metric == "e2e":
                self.registry.observe_e2e(self.latency_ms)

        return False  # never suppress exceptions


# ---------------------------------------------------------------------------
# MetricsRegistry: the central hub
# ---------------------------------------------------------------------------

class MetricsRegistry:
    """
    Central metrics hub.

    Owns:
      - PrometheusMetrics  : Prometheus counters/histograms/gauges
      - RollingStats       : per-metric sliding window stats
      - RateCounters       : per-metric event rates
      - SessionMetrics     : per-session trackers (active calls)
      - Background reporter: periodic log output

    One instance per process, created in main.py and passed
    to all components that need it.
    """

    def __init__(
        self,
        namespace:               str = "voicebot",
        rolling_window_seconds:  int = 60,
        prometheus_port:         Optional[int] = None,
        log_report_interval_s:   int = 60,
    ):
        self._ns              = namespace
        self._window          = rolling_window_seconds
        self._prom_port       = prometheus_port
        self._report_interval = log_report_interval_s

        # Prometheus
        self.prometheus = PrometheusMetrics(namespace=namespace)

        # Global rolling stats
        self._rolling: Dict[str, RollingStats] = {
            "e2e_latency_ms":  RollingStats(rolling_window_seconds),
            "llm_latency_ms":  RollingStats(rolling_window_seconds),
            "call_duration_s": RollingStats(rolling_window_seconds),
            "ring_duration_s": RollingStats(rolling_window_seconds),
        }

        # Rate counters
        self._rates: Dict[str, RateCounter] = {
            "calls":        RateCounter(rolling_window_seconds),
            "errors":       RateCounter(rolling_window_seconds),
            "interruptions":RateCounter(rolling_window_seconds),
            "outbound":     RateCounter(rolling_window_seconds),
        }

        # Monotonic totals
        self._totals: Dict[str, int] = defaultdict(int)
        self._totals_lock = threading.Lock()

        # Active session metrics: session_id → SessionMetrics
        self._sessions:      Dict[str, SessionMetrics] = {}
        self._sessions_lock  = threading.Lock()

        # Completed session history (ring buffer)
        self._completed: deque = deque(maxlen=500)

        # Session end callbacks
        self._session_callbacks: List[
            Callable[[SessionMetrics], None]
        ] = []

        # Background reporter
        self._stop_event      = threading.Event()
        self._reporter_thread: Optional[threading.Thread] = None

        # Process start time (for uptime)
        self._start_time = time.time()

        logger.info(
            "MetricsRegistry created: ns=%s window=%ds prom_port=%s",
            namespace, rolling_window_seconds, prometheus_port,
        )

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start Prometheus HTTP server and background reporter."""
        # Prometheus metrics HTTP server
        if self._prom_port and _PROMETHEUS_AVAILABLE:
            try:
                start_http_server(self._prom_port)
                logger.info(
                    "Prometheus metrics: http://0.0.0.0:%d/metrics",
                    self._prom_port,
                )
            except Exception as e:
                logger.error(
                    "Failed to start Prometheus server on :%d — %s",
                    self._prom_port, e,
                )

        # Background log reporter
        self._reporter_thread = threading.Thread(
            target=self._reporter_loop,
            name="metrics-reporter",
            daemon=True,
        )
        self._reporter_thread.start()
        logger.info(
            "Metrics reporter started (interval=%ds)",
            self._report_interval,
        )

    def stop(self) -> None:
        """Stop the background reporter."""
        self._stop_event.set()
        if self._reporter_thread and self._reporter_thread.is_alive():
            self._reporter_thread.join(timeout=5)
        logger.info("MetricsRegistry stopped")

    # -----------------------------------------------------------------------
    # Session lifecycle
    # -----------------------------------------------------------------------

    def session_start(
        self, session_id: str, call_uuid: str
    ) -> SessionMetrics:
        """
        Create and register a SessionMetrics for a new call.
        Called by session.py on session creation.
        """
        sm = SessionMetrics(
            session_id=session_id,
            call_uuid=call_uuid,
        )
        with self._sessions_lock:
            self._sessions[session_id] = sm

        self._rates["calls"].increment()
        with self._totals_lock:
            self._totals["calls_total"] += 1

        logger.debug("Session metrics started: %s", session_id[:8])
        return sm

    def session_end(
        self,
        session_id:  str,
        final_state: str = "completed",
        direction:   str = "inbound",
    ) -> Optional[SessionMetrics]:
        """
        Finalize a session's metrics.
        Called by session.py on session termination.
        """
        with self._sessions_lock:
            sm = self._sessions.pop(session_id, None)

        if sm is None:
            logger.warning(
                "session_end: unknown session %s", session_id[:8]
            )
            return None

        sm.finalize(final_state)

        # Update global rolling stats
        dur = sm.duration_seconds
        self._rolling["call_duration_s"].record(dur)

        for ms in sm.e2e_latencies_ms:
            self._rolling["e2e_latency_ms"].record(ms)
        for ms in sm.llm_latencies_ms:
            self._rolling["llm_latency_ms"].record(ms)

        # Prometheus
        result = (
            "success"
            if final_state in ("completed", "normal")
            else final_state
        )
        self.prometheus.call_ended(direction, result, dur)

        # Totals
        with self._totals_lock:
            self._totals["interruptions_total"] += sm.interruptions
            self._totals["errors_total"]        += sm.errors

        # Archive
        self._completed.append(sm)

        # Callbacks
        for cb in self._session_callbacks:
            try:
                cb(sm)
            except Exception as e:
                logger.error("Session callback error: %s", e)

        logger.info(
            "Session ended [%s]: "
            "duration=%.1fs state=%s "
            "e2e_p95=%.0fms interruptions=%d",
            session_id[:8],
            dur,
            final_state,
            self._rolling["e2e_latency_ms"].p95(),
            sm.interruptions,
        )
        return sm

    def get_session(self, session_id: str) -> Optional[SessionMetrics]:
        with self._sessions_lock:
            return self._sessions.get(session_id)

    def add_session_callback(
        self, cb: Callable[[SessionMetrics], None]
    ) -> None:
        self._session_callbacks.append(cb)

    # -----------------------------------------------------------------------
    # Global observation helpers
    # -----------------------------------------------------------------------

    def observe_e2e(self, latency_ms: float) -> None:
        self._rolling["e2e_latency_ms"].record(latency_ms)
        self.prometheus.observe_e2e(latency_ms)

    def observe_llm(
        self,
        latency_ms: float,
        provider:   str  = "minicpm",
        model:      str  = "minicpmo-4.5",
        success:    bool = True,
    ) -> None:
        self._rolling["llm_latency_ms"].record(latency_ms)
        self.prometheus.observe_llm(latency_ms, provider, model)
        if not success:
            self._rates["errors"].increment()
            self.prometheus.record_error("llm", "request_failed")

    def observe_ring(self, duration_s: float) -> None:
        self._rolling["ring_duration_s"].record(duration_s * 1000)
        self.prometheus.observe_ring(duration_s)

    def record_interruption(self) -> None:
        self._rates["interruptions"].increment()
        self.prometheus.record_interruption()
        with self._totals_lock:
            self._totals["interruptions_total"] += 1

    def record_error(
        self,
        component: str,
        err_type:  str = "unknown",
    ) -> None:
        self._rates["errors"].increment()
        self.prometheus.record_error(component, err_type)
        with self._totals_lock:
            self._totals["errors_total"] += 1

    # ---- Outbound specific ----

    def outbound_call_started(self, direction: str = "outbound") -> None:
        self._rates["outbound"].increment()
        self.prometheus.call_started(direction)
        with self._totals_lock:
            self._totals["outbound_total"] += 1

    def outbound_call_ended(
        self,
        result:     str,
        duration_s: float,
        ring_s:     float = 0.0,
    ) -> None:
        self.prometheus.outbound_result(result)
        self.prometheus.call_ended("outbound", result, duration_s)
        if ring_s > 0:
            self.observe_ring(ring_s)
        self._rolling["call_duration_s"].record(duration_s)

    def outbound_retry(self) -> None:
        self.prometheus.outbound_retry()
        with self._totals_lock:
            self._totals["outbound_retries"] += 1

    # -----------------------------------------------------------------------
    # Summary / reporting
    # -----------------------------------------------------------------------

    def get_summary(self) -> Dict:
        with self._sessions_lock:
            active_sessions = len(self._sessions)

        with self._totals_lock:
            totals = dict(self._totals)

        uptime = time.time() - self._start_time

        return {
            "timestamp":       datetime.utcnow().isoformat() + "Z",
            "uptime_seconds":  round(uptime, 1),
            "active_sessions": active_sessions,
            "totals":          totals,
            "rates_per_second": {
                name: round(rc.rate(), 4)
                for name, rc in self._rates.items()
            },
            "latency_stats": {
                name: rs.to_dict()
                for name, rs in self._rolling.items()
            },
        }

    def get_completed_sessions(
        self, limit: int = 20
    ) -> List[Dict]:
        snaps = list(self._completed)[-limit:]
        return [s.to_dict() for s in snaps]

    # -----------------------------------------------------------------------
    # Background reporter
    # -----------------------------------------------------------------------

    def _reporter_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._report_interval):
            try:
                self._emit_log_report()
                self.prometheus.update_process_stats()
            except Exception as e:
                logger.error("Metrics reporter error: %s", e)

    def _emit_log_report(self) -> None:
        summary = self.get_summary()
        e2e = summary["latency_stats"].get("e2e_latency_ms", {})
        llm = summary["latency_stats"].get("llm_latency_ms", {})
        dur = summary["latency_stats"].get("call_duration_s", {})

        logger.info(
            "METRICS | uptime=%.0fs active=%d "
            "calls/s=%.2f err/s=%.2f | "
            "e2e p50=%.0f p95=%.0f p99=%.0f ms | "
            "llm p95=%.0f ms | "
            "avg_duration=%.0fs",
            summary["uptime_seconds"],
            summary["active_sessions"],
            summary["rates_per_second"].get("calls", 0),
            summary["rates_per_second"].get("errors", 0),
            e2e.get("median", 0),
            e2e.get("p95", 0),
            e2e.get("p99", 0),
            llm.get("p95", 0),
            dur.get("mean", 0),
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def _totals_lock(self) -> threading.Lock:
        if not hasattr(self, "__totals_lock"):
            self.__totals_lock = threading.Lock()
        return self.__totals_lock

    @property
    def active_session_count(self) -> int:
        with self._sessions_lock:
            return len(self._sessions)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time


# ---------------------------------------------------------------------------
# Health checker
# ---------------------------------------------------------------------------

@dataclass
class HealthStatus:
    healthy:   bool
    checks:    Dict[str, bool]
    details:   Dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )

    def to_dict(self) -> Dict:
        return {
            "healthy":   self.healthy,
            "checks":    self.checks,
            "details":   self.details,
            "timestamp": self.timestamp,
        }


class HealthChecker:
    """
    Runs registered health checks and returns aggregated HealthStatus.

    Usage:
        checker = HealthChecker(registry)
        checker.register("esl", lambda: (esl.is_connected, {"host": "..."}))
        status = checker.run()
    """

    def __init__(self, registry: MetricsRegistry):
        self._registry = registry
        self._checks:  Dict[str, Callable[[], Tuple[bool, Any]]] = {}

    def register(
        self,
        name:     str,
        check_fn: Callable[[], Tuple[bool, Any]],
    ) -> None:
        self._checks[name] = check_fn

    def run(self) -> HealthStatus:
        results: Dict[str, bool] = {}
        details: Dict[str, Any]  = {}

        for name, fn in self._checks.items():
            try:
                ok, detail = fn()
                results[name] = ok
                details[name] = detail
            except Exception as e:
                results[name] = False
                details[name] = str(e)

        details["metrics"] = self._registry.get_summary()
        overall = all(results.values()) if results else True

        return HealthStatus(
            healthy=overall,
            checks=results,
            details=details,
        )


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

_global_registry: Optional[MetricsRegistry] = None
_registry_lock    = threading.Lock()


def init_metrics(
    namespace:              str          = "voicebot",
    rolling_window_seconds: int          = 60,
    prometheus_port:        Optional[int]= None,
    log_report_interval_seconds: int     = 60,
) -> MetricsRegistry:
    """
    Create, start, and cache the global MetricsRegistry.
    Safe to call multiple times (idempotent).
    """
    global _global_registry
    with _registry_lock:
        if _global_registry is None:
            _global_registry = MetricsRegistry(
                namespace=namespace,
                rolling_window_seconds=rolling_window_seconds,
                prometheus_port=prometheus_port,
                log_report_interval_s=log_report_interval_seconds,
            )
            _global_registry.start()
        return _global_registry


def get_metrics() -> MetricsRegistry:
    """Return the global MetricsRegistry. Call init_metrics() first."""
    if _global_registry is None:
        raise RuntimeError(
            "MetricsRegistry not initialized. "
            "Call init_metrics() first."
        )
    return _global_registry
