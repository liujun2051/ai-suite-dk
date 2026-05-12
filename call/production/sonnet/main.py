"""
main.py - VoiceBot application entry point

Responsibilities:
  - Parse CLI arguments
  - Load configuration (config.py)
  - Initialize all components in correct order
  - Wire up inter-component dependencies
  - Start all async services concurrently
  - Handle OS signals (SIGINT, SIGTERM, SIGHUP)
  - Graceful shutdown sequence
  - Health monitoring loop
  - Exit code management

Startup sequence:
  1. Parse args + load config
  2. Setup logging
  3. Initialize metrics registry
  4. Connect ESL client
  5. Start AudioServer (WebSocket for mod_audio_stream)
  6. Create SessionManager
  7. Start API server
  8. Register signal handlers
  9. Run forever (event loop)

Shutdown sequence (reverse order):
  1. Stop accepting new calls (AudioServer stops)
  2. Terminate all active sessions gracefully
  3. Stop API server
  4. Disconnect ESL
  5. Stop metrics reporter
  6. Exit
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, List

# ---------------------------------------------------------------------------
# Bootstrap logging before anything else
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local imports (after basicConfig so import errors are visible)
# ---------------------------------------------------------------------------
try:
    from config import (
        init_config, reload_config, get_config,
        AppConfig, setup_logging,
    )
    from metrics import init_metrics, MetricsRegistry
    from esl import ESLClient, create_esl_client
    from audio import AudioServer, create_audio_server
    from session import SessionManager
    from api import APIServer, create_api_server
except ImportError as e:
    logger.critical("Import error: %s", e, exc_info=True)
    logger.critical(
        "Make sure all dependencies are installed:\n"
        "  pip install websockets aiohttp webrtcvad numpy scipy"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_ENV_PATH    = ".env"
SHUTDOWN_TIMEOUT    = 15.0      # seconds to wait for graceful shutdown
ESL_CONNECT_RETRIES = 5         # attempts before giving up at startup
ESL_CONNECT_DELAY   = 3.0       # seconds between ESL connect attempts
HEALTH_CHECK_INTERVAL = 30.0    # seconds between internal health checks


# ---------------------------------------------------------------------------
# Application container
# ---------------------------------------------------------------------------

class VoiceBot:
    """
    Top-level application container.

    Owns all long-lived components and orchestrates their lifecycle.
    This is the only class that knows about all components simultaneously.
    """

    def __init__(self, cfg: AppConfig):
        self._cfg = cfg

        # Components (initialized in start())
        self._metrics:  Optional[MetricsRegistry] = None
        self._esl:      Optional[ESLClient]        = None
        self._audio:    Optional[AudioServer]      = None
        self._sessions: Optional[SessionManager]   = None
        self._api:      Optional[APIServer]         = None

        # Runtime state
        self._running        = False
        self._shutdown_event = asyncio.Event()
        self._start_time     = time.time()

        # Background tasks
        self._health_task:   Optional[asyncio.Task] = None
        self._esl_monitor_task: Optional[asyncio.Task] = None

        logger.info(
            "VoiceBot initializing: %s v%s [%s]",
            cfg.app_name, cfg.app_version, cfg.environment,
        )

    # -----------------------------------------------------------------------
    # Startup
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """
        Initialize and start all components.
        Raises on any fatal initialization error.
        """
        logger.info("=" * 60)
        logger.info(" VoiceBot Starting Up")
        logger.info("=" * 60)
        logger.info("\n%s", self._cfg.summary())

        try:
            await self._init_metrics()
            await self._init_esl()
            await self._init_audio_server()
            await self._init_session_manager()
            await self._init_api_server()
            await self._start_background_tasks()
            self._running = True

        except Exception as e:
            logger.critical(
                "Startup failed: %s", e, exc_info=True
            )
            await self.stop(exit_code=1)
            raise

        logger.info("=" * 60)
        logger.info(" VoiceBot Ready")
        logger.info(
            " Audio WS : ws://%s:%d",
            self._cfg.server.host,
            self._cfg.server.audio_ws_port,
        )
        logger.info(
            " API      : http://%s:%d",
            self._cfg.server.host,
            self._cfg.server.api_port,
        )
        if self._cfg.metrics.prometheus_enabled:
            logger.info(
                " Prometheus: http://%s:%d",
                self._cfg.server.host,
                self._cfg.metrics.prometheus_port,
            )
        logger.info("=" * 60)

    async def _init_metrics(self) -> None:
        """Initialize metrics registry and start Prometheus server."""
        logger.info("Initializing metrics...")
        m_cfg = self._cfg.metrics
        self._metrics = init_metrics(
            namespace=m_cfg.namespace,
            rolling_window_seconds=m_cfg.rolling_window_seconds,
            prometheus_port=(
                m_cfg.prometheus_port if m_cfg.prometheus_enabled else None
            ),
            log_report_interval_seconds=m_cfg.log_report_interval_seconds,
        )
        logger.info("Metrics initialized")

    async def _init_esl(self) -> None:
        """
        Create ESL client and connect to FreeSWITCH.
        Retries several times before giving up.
        """
        logger.info(
            "Connecting to FreeSWITCH ESL %s:%d ...",
            self._cfg.esl.host, self._cfg.esl.port,
        )

        self._esl = create_esl_client(self._cfg)

        # Register lifecycle callbacks
        self._esl._on_connected    = self._on_esl_connected
        self._esl._on_disconnected = self._on_esl_disconnected

        # Retry loop
        last_error: Optional[Exception] = None
        for attempt in range(1, ESL_CONNECT_RETRIES + 1):
            try:
                await self._esl.connect()
                logger.info(
                    "ESL connected (attempt %d)", attempt
                )
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    "ESL connect attempt %d/%d failed: %s",
                    attempt, ESL_CONNECT_RETRIES, e,
                )
                if attempt < ESL_CONNECT_RETRIES:
                    logger.info(
                        "Retrying in %.0fs...", ESL_CONNECT_DELAY
                    )
                    await asyncio.sleep(ESL_CONNECT_DELAY)

        # All retries exhausted
        raise ConnectionError(
            f"Cannot connect to FreeSWITCH ESL after "
            f"{ESL_CONNECT_RETRIES} attempts: {last_error}"
        )

    async def _init_audio_server(self) -> None:
        """Create and start the WebSocket audio server."""
        logger.info(
            "Starting audio WebSocket server on %s:%d ...",
            self._cfg.server.host,
            self._cfg.server.audio_ws_port,
        )
        self._audio = create_audio_server(self._cfg)
        await self._audio.start()
        logger.info("Audio server started")

    async def _init_session_manager(self) -> None:
        """Create the session manager and wire up callbacks."""
        logger.info("Initializing session manager...")
        self._sessions = SessionManager(
            audio_server=self._audio,
            esl_client=self._esl,
            cfg=self._cfg,
            metrics=self._metrics,
            session_cfg_factory=self._build_session_config,
        )

        # Register ESL event handlers now that manager is ready
        if self._esl.is_connected:
            self._sessions._register_esl_handlers()

        logger.info("Session manager initialized")

    async def _init_api_server(self) -> None:
        """Create and start the REST API server."""
        if not self._cfg.api.enabled:
            logger.info("API server disabled in config")
            return

        logger.info(
            "Starting API server on %s:%d ...",
            self._cfg.api.host, self._cfg.api.port,
        )
        self._api = create_api_server(
            cfg=self._cfg,
            session_manager=self._sessions,
            metrics=self._metrics,
            esl_client=self._esl,
            audio_server=self._audio,
        )
        await self._api.start()
        logger.info("API server started")

    async def _start_background_tasks(self) -> None:
        """Start internal health check and ESL monitor tasks."""
        self._health_task = asyncio.ensure_future(
            self._health_loop()
        )
        self._esl_monitor_task = asyncio.ensure_future(
            self._esl_monitor_loop()
        )
        logger.info("Background tasks started")

    def _build_session_config(self, stream: "Any") -> "Any":
        """
        Per-call session config factory.
        Called by SessionManager for each new call.
        Can be overridden to implement routing logic:
          - Different prompts for different DIDs
          - Different languages based on caller country
          - A/B testing different AI configurations
        """
        from session import SessionConfig
        m   = self._cfg.minicpm
        i   = self._cfg.interruption
        meta = stream.metadata

        # Example: route to different prompt based on destination number
        system_prompt = m.system_prompt
        greeting_text = ""

        if meta:
            dest = meta.destination
            # Could look up routing table here
            # e.g. if dest == "8001": system_prompt = SALES_PROMPT
            pass

        return SessionConfig(
            system_prompt=system_prompt,
            language=m.language,
            temperature=m.temperature,
            voice_id=m.voice_id,
            greeting_text=greeting_text,
            interruption_enabled=i.enabled,
            interruption_cooldown_s=i.cooldown_ms / 1000.0,
            grace_period_ms=float(i.grace_period_ms),
            direction=meta.direction if meta else "inbound",
        )

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------

    async def stop(self, exit_code: int = 0) -> None:
        """
        Graceful shutdown sequence.
        Stops components in reverse startup order.
        """
        if not self._running and exit_code == 0:
            return

        logger.info("=" * 60)
        logger.info(" VoiceBot Shutting Down (exit_code=%d)", exit_code)
        logger.info("=" * 60)

        self._running = False
        self._shutdown_event.set()

        # Step 1: Stop health checks
        for task in [self._health_task, self._esl_monitor_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        # Step 2: Stop accepting new audio connections
        if self._audio:
            logger.info("Stopping audio server...")
            try:
                await asyncio.wait_for(
                    self._audio.stop(), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Audio server stop timed out")
            except Exception as e:
                logger.error("Audio server stop error: %s", e)

        # Step 3: Terminate all active sessions
        if self._sessions:
            active = self._sessions.active_count
            if active > 0:
                logger.info(
                    "Terminating %d active sessions...", active
                )
                try:
                    await asyncio.wait_for(
                        self._sessions.terminate_all("shutdown"),
                        timeout=SHUTDOWN_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Session termination timed out after %.0fs",
                        SHUTDOWN_TIMEOUT,
                    )
                except Exception as e:
                    logger.error(
                        "Session termination error: %s", e
                    )

        # Step 4: Stop API server
        if self._api:
            logger.info("Stopping API server...")
            try:
                await asyncio.wait_for(
                    self._api.stop(), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("API server stop timed out")
            except Exception as e:
                logger.error("API server stop error: %s", e)

        # Step 5: Disconnect ESL
        if self._esl:
            logger.info("Disconnecting ESL...")
            try:
                await asyncio.wait_for(
                    self._esl.disconnect(), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("ESL disconnect timed out")
            except Exception as e:
                logger.error("ESL disconnect error: %s", e)

        # Step 6: Stop metrics
        if self._metrics:
            logger.info("Stopping metrics reporter...")
            try:
                self._metrics.stop()
            except Exception as e:
                logger.error("Metrics stop error: %s", e)

        duration = time.time() - self._start_time
        logger.info(
            "VoiceBot stopped. Total uptime: %.0fs", duration
        )

    # -----------------------------------------------------------------------
    # Signal handlers
    # -----------------------------------------------------------------------

    def setup_signal_handlers(
        self, loop: asyncio.AbstractEventLoop
    ) -> None:
        """
        Register OS signal handlers.

        SIGTERM / SIGINT : graceful shutdown
        SIGHUP           : reload config (Unix only)
        SIGUSR1          : dump debug info to log
        """
        # SIGTERM: sent by systemd/k8s on shutdown
        loop.add_signal_handler(
            signal.SIGTERM,
            lambda: asyncio.ensure_future(
                self._handle_signal(signal.SIGTERM)
            ),
        )

        # SIGINT: Ctrl+C
        loop.add_signal_handler(
            signal.SIGINT,
            lambda: asyncio.ensure_future(
                self._handle_signal(signal.SIGINT)
            ),
        )

        # SIGHUP: reload config (Unix only)
        if hasattr(signal, "SIGHUP"):
            loop.add_signal_handler(
                signal.SIGHUP,
                lambda: asyncio.ensure_future(
                    self._handle_signal(signal.SIGHUP)
                ),
            )

        # SIGUSR1: dump debug state
        if hasattr(signal, "SIGUSR1"):
            loop.add_signal_handler(
                signal.SIGUSR1,
                lambda: asyncio.ensure_future(
                    self._handle_signal(signal.SIGUSR1)
                ),
            )

        logger.info("Signal handlers registered")

    async def _handle_signal(self, sig: int) -> None:
        """Handle OS signals."""
        sig_name = signal.Signals(sig).name

        if sig in (signal.SIGTERM, signal.SIGINT):
            logger.info(
                "Received %s — initiating graceful shutdown", sig_name
            )
            await self.stop(exit_code=0)

        elif hasattr(signal, "SIGHUP") and sig == signal.SIGHUP:
            logger.info("Received SIGHUP — reloading config")
            await self._reload_config()

        elif hasattr(signal, "SIGUSR1") and sig == signal.SIGUSR1:
            logger.info("Received SIGUSR1 — dumping debug state")
            self._dump_debug_state()

    async def _reload_config(self) -> None:
        """
        Hot-reload configuration on SIGHUP.
        Only reloads fields that are safe to change at runtime.
        (Cannot change ports, ESL host, etc. without restart.)
        """
        logger.info("Reloading configuration...")
        try:
            new_cfg = reload_config()
            # Apply safe runtime changes
            if self._metrics:
                # Update log level
                setup_logging(new_cfg.log)
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error("Config reload failed: %s", e)

    def _dump_debug_state(self) -> None:
        """
        Dump current system state to logs.
        Triggered by SIGUSR1.
        """
        logger.info("=== DEBUG STATE DUMP ===")

        if self._sessions:
            stats = self._sessions.get_stats()
            logger.info(
                "Sessions: active=%d completed=%d",
                stats["active_sessions"],
                stats["completed_sessions"],
            )
            for s in stats["sessions"]:
                logger.info("  Session: %s", s)

        if self._metrics:
            summary = self._metrics.get_summary()
            logger.info("Metrics: %s", summary)

        if self._esl:
            info = self._esl.connection_info
            logger.info("ESL: %s", info.to_dict())

        if self._audio:
            audio_stats = self._audio.get_all_stats()
            logger.info("Audio: %s", audio_stats["server"])

        logger.info("=== END DEBUG DUMP ===")

    # -----------------------------------------------------------------------
    # ESL event callbacks
    # -----------------------------------------------------------------------

    async def _on_esl_connected(self) -> None:
        """Called when ESL connects or reconnects."""
        logger.info("ESL connected to FreeSWITCH")

        if self._metrics:
            self._metrics.prometheus.set_esl_connections(1)

        # Re-register ESL handlers after reconnect
        if self._sessions:
            self._sessions._register_esl_handlers()

    async def _on_esl_disconnected(self) -> None:
        """Called when ESL disconnects."""
        logger.warning("ESL disconnected from FreeSWITCH")

        if self._metrics:
            self._metrics.prometheus.set_esl_connections(0)

    # -----------------------------------------------------------------------
    # Background loops
    # -----------------------------------------------------------------------

    async def _health_loop(self) -> None:
        """
        Periodic internal health check.
        Logs warnings if components are degraded.
        Updates Prometheus process stats.
        """
        logger.debug("Health check loop started")

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

                if self._shutdown_event.is_set():
                    break

                issues: List[str] = []

                # Check ESL
                if self._esl and not self._esl.is_connected:
                    issues.append("ESL disconnected")

                # Check active sessions vs limit
                if self._sessions:
                    active = self._sessions.active_count
                    limit  = self._cfg.server.max_connections
                    if active >= limit:
                        issues.append(
                            f"At connection limit ({active}/{limit})"
                        )

                # Check audio server
                if self._audio:
                    pending = self._audio.pending_count
                    if pending > 3:
                        issues.append(
                            f"High pending connections: {pending}"
                        )

                # Update process metrics
                if self._metrics:
                    self._metrics.prometheus.update_process_stats()

                if issues:
                    logger.warning(
                        "Health check issues: %s", "; ".join(issues)
                    )
                else:
                    logger.debug(
                        "Health check OK: "
                        "sessions=%d esl=%s uptime=%.0fs",
                        self._sessions.active_count if self._sessions else 0,
                        self._esl.is_connected if self._esl else False,
                        time.time() - self._start_time,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health loop error: %s", e)

        logger.debug("Health check loop ended")

    async def _esl_monitor_loop(self) -> None:
        """
        Monitor ESL connection and update metrics.
        Separate from health loop to allow different intervals.
        """
        logger.debug("ESL monitor loop started")

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60.0)   # every minute

                if not self._esl:
                    continue

                info = self._esl.connection_info
                logger.info(
                    "ESL status: state=%s "
                    "events_rx=%d cmds_sent=%d "
                    "reconnects=%d",
                    info.state.name,
                    info.events_received,
                    info.commands_sent,
                    info.reconnect_attempts,
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("ESL monitor error: %s", e)

        logger.debug("ESL monitor loop ended")

    # -----------------------------------------------------------------------
    # Run forever
    # -----------------------------------------------------------------------

    async def run_forever(self) -> None:
        """
        Block until shutdown is triggered.
        Called after start() to keep the event loop running.
        """
        logger.info("VoiceBot running. Press Ctrl+C to stop.")
        await self._shutdown_event.wait()
        logger.info("Shutdown event received")

    @property
    def is_running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="voicebot",
        description="VoiceBot — MiniCPM-o 4.5 full-duplex voice AI for FreeSWITCH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default config
  python main.py

  # Start with custom config file
  python main.py --config /etc/voicebot/config.yaml

  # Override specific settings
  python main.py --log-level DEBUG --api-port 9090

  # Check config and exit
  python main.py --check-config

  # Run without ESL (audio-only mode for testing)
  python main.py --no-esl

Environment variables:
  MINICPM_API_KEY           MiniCPM-o API key (required)
  VOICEBOT__ESL__PASSWORD   FreeSWITCH ESL password
  VOICEBOT__DEBUG           Enable debug mode (true/false)
  VOICEBOT__LOG__LEVEL      Log level (DEBUG/INFO/WARNING/ERROR)
        """,
    )

    parser.add_argument(
        "--config", "-c",
        default=DEFAULT_CONFIG_PATH,
        metavar="PATH",
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_PATH,
        metavar="PATH",
        help=f"Path to .env file (default: {DEFAULT_ENV_PATH})",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Override log level from config",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration and exit without starting",
    )
    parser.add_argument(
        "--no-esl",
        action="store_true",
        help="Start without ESL connection (testing mode)",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable REST API server",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=None,
        help="Override REST API port",
    )
    parser.add_argument(
        "--audio-port",
        type=int,
        default=None,
        help="Override audio WebSocket port",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize all components then exit immediately",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def check_dependencies() -> List[str]:
    """
    Check that all required Python packages are installed.
    Returns list of missing package names.
    """
    required = {
        "websockets":  "websockets",
        "aiohttp":     "aiohttp",
    }
    optional = {
        "webrtcvad":   "webrtcvad (VAD quality will be reduced)",
        "numpy":       "numpy (resampling quality will be reduced)",
        "scipy":       "scipy (resampling quality will be reduced)",
        "prometheus_client": "prometheus_client (Prometheus metrics disabled)",
        "psutil":      "psutil (process metrics disabled)",
    }

    missing_required = []
    for pkg, name in required.items():
        try:
            __import__(pkg)
        except ImportError:
            missing_required.append(name)

    for pkg, name in optional.items():
        try:
            __import__(pkg)
        except ImportError:
            logger.warning("Optional package not installed: %s", name)

    return missing_required


def check_environment(cfg: AppConfig) -> List[str]:
    """
    Run pre-flight environment checks.
    Returns list of warning strings.
    """
    warnings: List[str] = []

    # API key
    if not cfg.minicpm.api_key:
        warnings.append(
            "MINICPM_API_KEY not set — AI will not work"
        )

    # Ports available (basic check)
    import socket
    ports_to_check = [
        ("audio WebSocket", cfg.server.audio_ws_port),
        ("API server",      cfg.server.api_port),
    ]
    if cfg.metrics.prometheus_enabled:
        ports_to_check.append(
            ("Prometheus", cfg.metrics.prometheus_port)
        )

    for name, port in ports_to_check:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            sock.close()
        except OSError as e:
            warnings.append(
                f"Port {port} ({name}) may not be available: {e}"
            )

    # Log directory
    if cfg.log.file_enabled:
        log_dir = Path(cfg.log.file_path).parent
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Created log directory: %s", log_dir)
            except OSError as e:
                warnings.append(
                    f"Cannot create log directory {log_dir}: {e}"
                )

    return warnings


# ---------------------------------------------------------------------------
# Async main runner
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> int:
    """
    Main async entry point.
    Returns exit code.
    """
    # ---- Load config ----
    try:
        cfg = init_config(
            yaml_path=args.config if Path(args.config).exists() else None,
            env_file=args.env_file if Path(args.env_file).exists() else None,
            validate=True,
        )
    except ValueError as e:
        # Validation error
        logger.critical("Configuration error:\n%s", e)
        return 2
    except FileNotFoundError as e:
        logger.critical("Config file not found: %s", e)
        return 2
    except Exception as e:
        logger.critical("Config load failed: %s", e, exc_info=True)
        return 2

    # ---- Apply CLI overrides ----
    if args.log_level:
        from config import LogLevel
        cfg.log.level = LogLevel(args.log_level)
        setup_logging(cfg.log)

    if args.api_port:
        cfg.api.port         = args.api_port
        cfg.server.api_port  = args.api_port

    if args.audio_port:
        cfg.server.audio_ws_port = args.audio_port

    if args.no_api:
        cfg.api.enabled = False

    if args.no_esl:
        logger.warning(
            "Running without ESL — call control disabled"
        )

    # ---- Check config only ----
    if args.check_config:
        errors = cfg.validate()
        if errors:
            print("Configuration INVALID:")
            for e in errors:
                print(f"  ✗ {e}")
            return 2
        print("Configuration VALID")
        print(cfg.summary())
        return 0

    # ---- Version ----
    if args.version:
        print(f"{cfg.app_name} v{cfg.app_version}")
        return 0

    # ---- Dependency check ----
    missing = check_dependencies()
    if missing:
        logger.critical(
            "Missing required packages: %s\n"
            "Install with: pip install %s",
            ", ".join(missing),
            " ".join(missing),
        )
        return 1

    # ---- Pre-flight environment checks ----
    warnings = check_environment(cfg)
    for w in warnings:
        logger.warning("Pre-flight: %s", w)

    # ---- Create and start VoiceBot ----
    bot = VoiceBot(cfg)

    loop = asyncio.get_event_loop()
    bot.setup_signal_handlers(loop)

    try:
        await bot.start()
    except Exception as e:
        logger.critical("Failed to start VoiceBot: %s", e, exc_info=True)
        return 1

    # ---- Dry run: start then immediately stop ----
    if args.dry_run:
        logger.info("Dry run complete — stopping")
        await bot.stop(exit_code=0)
        return 0

    # ---- Run forever (blocks until shutdown signal) ----
    try:
        await bot.run_forever()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception as e:
        logger.critical(
            "Unexpected error in main loop: %s",
            e, exc_info=True,
        )
        await bot.stop(exit_code=1)
        return 1

    await bot.stop(exit_code=0)
    return 0


# ---------------------------------------------------------------------------
# Synchronous entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """
    Synchronous entry point.
    Sets up the event loop and runs async_main().
    Returns exit code.
    """
    args = parse_args(argv)

    # ---- Setup event loop ----
    if sys.platform == "win32":
        # Windows: use ProactorEventLoop for better async I/O
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
    else:
        # Unix: use default SelectorEventLoop
        # uvloop would be a drop-in replacement for better performance:
        #   pip install uvloop && import uvloop; uvloop.install()
        try:
            import uvloop
            uvloop.install()
            logger.info("uvloop installed for better performance")
        except ImportError:
            pass
        loop = asyncio.get_event_loop()

    # ---- Configure loop ----
    loop.set_debug(
        os.environ.get("PYTHONASYNCIODEBUG", "0") == "1"
    )

    # ---- Exception handler for unhandled async exceptions ----
    def handle_exception(
        loop: asyncio.AbstractEventLoop,
        context: dict,
    ) -> None:
        msg = context.get("exception", context.get("message", ""))
        if "exception" in context:
            exc = context["exception"]
            # Ignore benign exceptions
            if isinstance(exc, (
                ConnectionResetError,
                BrokenPipeError,
                asyncio.CancelledError,
            )):
                return
            logger.error(
                "Unhandled async exception: %s\n%s",
                exc,
                "".join(traceback.format_exception(
                    type(exc), exc, exc.__traceback__
                )),
            )
        else:
            logger.error("Async loop error: %s", msg)

    loop.set_exception_handler(handle_exception)

    # ---- Run ----
    try:
        exit_code = loop.run_until_complete(async_main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted")
        exit_code = 0
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        logger.critical(
            "Fatal error: %s", e, exc_info=True
        )
        exit_code = 1
    finally:
        # Clean up remaining tasks
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                logger.info(
                    "Cancelling %d pending tasks...", len(pending)
                )
                for task in pending:
                    task.cancel()
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        except Exception:
            pass
        finally:
            loop.close()
            logger.info("Event loop closed")

    return exit_code


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
