"""
config.py - Centralized configuration management for VoiceBot
Supports: environment variables, .env file, YAML file, dataclass validation
"""

import os
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try optional deps
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LogLevel(str, Enum):
    DEBUG    = "DEBUG"
    INFO     = "INFO"
    WARNING  = "WARNING"
    ERROR    = "ERROR"
    CRITICAL = "CRITICAL"


class AudioSampleRate(int, Enum):
    RATE_8000  = 8000
    RATE_16000 = 16000
    RATE_24000 = 24000
    RATE_48000 = 48000


class ESLMode(str, Enum):
    INBOUND  = "inbound"   # we connect to FreeSWITCH
    OUTBOUND = "outbound"  # FreeSWITCH connects to us


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    """
    Our application's own HTTP/WebSocket server settings.
    mod_audio_stream on FreeSWITCH will connect here.
    """
    host: str = "0.0.0.0"
    # Port for mod_audio_stream WebSocket connections (audio)
    audio_ws_port: int = 8765
    # Port for REST management API
    api_port: int = 8080
    # Max concurrent WebSocket audio connections
    max_connections: int = 10
    # WebSocket ping interval (seconds)
    ws_ping_interval: float = 20.0
    # WebSocket ping timeout (seconds)
    ws_ping_timeout: float = 10.0
    # Read/write buffer size for WebSocket (bytes)
    ws_max_message_size: int = 1024 * 1024  # 1 MB
    # Graceful shutdown timeout
    shutdown_timeout_seconds: float = 15.0


@dataclass
class ESLConfig:
    """
    FreeSWITCH ESL (Event Socket Library) connection settings.
    Used for sending control commands (uuid_break, uuid_setvar, etc.)
    """
    mode: ESLMode = ESLMode.INBOUND
    host: str = "127.0.0.1"
    port: int = 8021
    password: str = "ClueCon"
    # How long to wait for ESL connection (seconds)
    connect_timeout: float = 10.0
    # How long to wait for ESL command response
    command_timeout: float = 5.0
    # Reconnect settings
    reconnect_enabled: bool = True
    reconnect_interval_seconds: float = 3.0
    reconnect_max_attempts: int = 0  # 0 = unlimited
    # Heartbeat interval to detect dead connections
    heartbeat_interval_seconds: float = 30.0
    # ESL outbound server port (only used when mode=OUTBOUND)
    outbound_port: int = 8084
    # Events to subscribe to
    subscribe_events: List[str] = field(default_factory=lambda: [
        "CHANNEL_CREATE",
        "CHANNEL_ANSWER",
        "CHANNEL_HANGUP",
        "CHANNEL_HANGUP_COMPLETE",
        "CHANNEL_BRIDGE",
        "CHANNEL_UNBRIDGE",
        "CHANNEL_EXECUTE",
        "CHANNEL_EXECUTE_COMPLETE",
        "DTMF",
        "DETECTED_SPEECH",
        "PLAYBACK_START",
        "PLAYBACK_STOP",
        "RECORD_START",
        "RECORD_STOP",
        "BACKGROUND_JOB",
        "HEARTBEAT",
    ])


@dataclass
class AudioConfig:
    """
    Audio pipeline configuration.
    FreeSWITCH mod_audio_stream sends/receives PCM audio.
    MiniCPM-o requires 16kHz 16bit mono PCM.
    """
    # FreeSWITCH side sample rate (what FS sends us)
    fs_sample_rate: AudioSampleRate = AudioSampleRate.RATE_16000
    # MiniCPM-o required sample rate
    model_sample_rate: AudioSampleRate = AudioSampleRate.RATE_16000
    # Channels (always mono for telephony)
    channels: int = 1
    # Bit depth
    sample_width_bytes: int = 2  # 16-bit PCM
    # Audio chunk size sent from FreeSWITCH (ms)
    fs_chunk_ms: int = 20
    # Audio chunk size we send to MiniCPM (ms)
    model_chunk_ms: int = 100
    # Playback buffer size (ms) before we start sending to FS
    playback_buffer_ms: int = 0   # 0 = send immediately (lowest latency)
    # Max audio queue size (chunks) before we drop old audio
    max_queue_chunks: int = 500
    # Enable resampling (needed if fs_sample_rate != model_sample_rate)
    enable_resampling: bool = True
    # Enable audio normalization before sending to model
    enable_normalization: bool = True
    # Normalization target level (dBFS)
    normalization_target_dbfs: float = -20.0

    @property
    def fs_chunk_samples(self) -> int:
        return int(self.fs_sample_rate * self.fs_chunk_ms / 1000)

    @property
    def fs_chunk_bytes(self) -> int:
        return self.fs_chunk_samples * self.sample_width_bytes * self.channels

    @property
    def model_chunk_samples(self) -> int:
        return int(self.model_sample_rate * self.model_chunk_ms / 1000)

    @property
    def model_chunk_bytes(self) -> int:
        return self.model_chunk_samples * self.sample_width_bytes * self.channels


@dataclass
class VADConfig:
    """
    Voice Activity Detection configuration.
    Used for interruption detection.
    """
    # Energy threshold (dBFS) - below this is silence
    energy_threshold_dbfs: float = -40.0
    # Minimum speech duration to consider as real speech (ms)
    min_speech_duration_ms: int = 200
    # Silence duration after speech to consider utterance complete (ms)
    silence_duration_ms: int = 700
    # Silence duration during bot speech to trigger interruption (ms)
    # Shorter than normal because we want fast interruption response
    interruption_silence_ms: int = 200
    # Minimum speech duration to trigger interruption (ms)
    # Prevents brief noise from triggering interruption
    interruption_min_speech_ms: int = 250
    # Use WebRTC VAD (requires webrtcvad package) vs simple energy VAD
    use_webrtc_vad: bool = True
    # WebRTC VAD aggressiveness (0=least, 3=most aggressive)
    webrtc_vad_aggressiveness: int = 2
    # Frame duration for WebRTC VAD (ms) - must be 10, 20, or 30
    webrtc_frame_ms: int = 20
    # Smoothing: number of frames to look back when making VAD decision
    smoothing_frames: int = 5
    # How many of smoothing_frames must be speech to declare speech
    smoothing_threshold: int = 3


@dataclass
class MiniCPMConfig:
    """
    MiniCPM-o 4.5 API configuration.
    Full-duplex voice model accessed via official WebSocket API.
    """
    # API base URL
    api_base_url: str = "wss://api.minimaxi.chat/v1/realtime"
    # API key (load from env)
    api_key: str = ""
    # Model name
    model: str = "MiniCPMo-4.5"
    # WebSocket connection timeout (seconds)
    connect_timeout: float = 15.0
    # Maximum session duration (seconds) before we force reconnect
    max_session_duration_seconds: float = 3600.0
    # Reconnect on error
    reconnect_on_error: bool = True
    reconnect_delay_seconds: float = 1.0
    max_reconnect_attempts: int = 5
    # System prompt / instructions for the AI
    system_prompt: str = (
        "You are a helpful voice assistant. "
        "Keep responses concise and conversational. "
        "You are speaking with a caller on the phone."
    )
    # Voice/speaker ID (if the API supports selecting voice)
    voice_id: str = "default"
    # Language hint
    language: str = "zh-CN"
    # Temperature for generation
    temperature: float = 0.7
    # Audio output format requested from model
    audio_output_format: str = "pcm"  # pcm | wav | mp3
    # Expected output sample rate from model
    output_sample_rate: int = 16000
    # How long to wait for first audio chunk before considering it a timeout (ms)
    first_audio_timeout_ms: int = 5000
    # How long silence from model before we consider response complete (ms)
    model_silence_timeout_ms: int = 2000
    # Send audio in this interval when streaming to model (ms)
    send_interval_ms: int = 100
    # Extra headers for WebSocket connection
    extra_headers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Allow override from environment
        if not self.api_key:
            self.api_key = os.environ.get("MINICPM_API_KEY", "")


@dataclass
class InterruptionConfig:
    """
    Interruption handling configuration.
    Controls how and when user speech interrupts bot playback.
    """
    # Enable interruption handling
    enabled: bool = True
    # Strategy: "vad" = energy/webrtc VAD, "always" = any audio interrupts
    strategy: str = "vad"
    # Grace period after bot starts speaking before interruptions are accepted (ms)
    # Prevents the bot's own audio from being picked up and triggering interruption
    grace_period_ms: int = 500
    # After interruption, how long to suppress new interruptions (ms)
    # Prevents rapid back-and-forth interruption loops
    cooldown_ms: int = 1000
    # Send uuid_break to FreeSWITCH on interruption
    send_uuid_break: bool = True
    # Clear the model's audio output queue on interruption
    clear_model_audio_queue: bool = True
    # Send interrupt signal to MiniCPM WebSocket
    send_model_interrupt: bool = True
    # Max interruptions per session before we disable interruption
    max_interruptions_per_session: int = 50


@dataclass
class MetricsConfig:
    """Metrics and monitoring configuration."""
    # Enable Prometheus metrics endpoint
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    # Prometheus metrics namespace
    namespace: str = "voicebot"
    # Rolling window for internal stats (seconds)
    rolling_window_seconds: int = 60
    # Log metrics report every N seconds
    log_report_interval_seconds: int = 60
    # Enable per-session detailed logging
    log_session_details: bool = True
    # Latency alert thresholds (ms) - will log WARNING if exceeded
    alert_e2e_latency_ms: float = 2000.0
    alert_stt_latency_ms: float = 1000.0
    alert_llm_latency_ms: float = 3000.0
    alert_tts_latency_ms: float = 1000.0


@dataclass
class APIConfig:
    """REST management API configuration."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    # Enable authentication for management API
    auth_enabled: bool = False
    auth_token: str = ""
    # CORS origins
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    # Request timeout
    request_timeout_seconds: float = 10.0


@dataclass
class LogConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    # Log format
    format: str = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    # Log to file
    file_enabled: bool = False
    file_path: str = "logs/voicebot.log"
    file_max_bytes: int = 50 * 1024 * 1024   # 50 MB
    file_backup_count: int = 5
    # JSON logging (for log aggregation systems like ELK)
    json_enabled: bool = False
    # Log caller info (file, line number) - useful for debug
    log_caller: bool = False


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """
    Root application configuration.
    All sub-configs are nested here.
    """
    # Application info
    app_name: str = "VoiceBot"
    app_version: str = "1.0.0"
    environment: str = "production"   # development | staging | production
    debug: bool = False

    # Sub-configs
    server: ServerConfig        = field(default_factory=ServerConfig)
    esl: ESLConfig              = field(default_factory=ESLConfig)
    audio: AudioConfig          = field(default_factory=AudioConfig)
    vad: VADConfig              = field(default_factory=VADConfig)
    minicpm: MiniCPMConfig      = field(default_factory=MiniCPMConfig)
    interruption: InterruptionConfig = field(default_factory=InterruptionConfig)
    metrics: MetricsConfig      = field(default_factory=MetricsConfig)
    api: APIConfig              = field(default_factory=APIConfig)
    log: LogConfig              = field(default_factory=LogConfig)

    def __post_init__(self):
        # If debug mode, drop log level
        if self.debug:
            self.log.level = LogLevel.DEBUG

    def validate(self) -> List[str]:
        """
        Validate configuration. Returns list of error strings.
        Empty list means config is valid.
        """
        errors: List[str] = []

        # MiniCPM API key
        if not self.minicpm.api_key:
            errors.append(
                "minicpm.api_key is required. "
                "Set MINICPM_API_KEY environment variable."
            )

        # ESL password
        if not self.esl.password:
            errors.append("esl.password is required.")

        # Port conflicts
        ports = {
            "audio_ws": self.server.audio_ws_port,
            "api": self.server.api_port,
            "prometheus": self.metrics.prometheus_port,
            "esl_outbound": self.esl.outbound_port,
        }
        seen: Dict[int, str] = {}
        for name, port in ports.items():
            if port in seen:
                errors.append(
                    f"Port conflict: {name} and {seen[port]} both use port {port}"
                )
            seen[port] = name

        # Audio config
        if self.audio.channels not in (1, 2):
            errors.append("audio.channels must be 1 (mono) or 2 (stereo)")
        if self.audio.sample_width_bytes not in (1, 2, 4):
            errors.append("audio.sample_width_bytes must be 1, 2, or 4")
        if self.audio.fs_chunk_ms <= 0:
            errors.append("audio.fs_chunk_ms must be > 0")
        if self.audio.model_chunk_ms <= 0:
            errors.append("audio.model_chunk_ms must be > 0")

        # VAD
        if self.vad.webrtc_frame_ms not in (10, 20, 30):
            errors.append("vad.webrtc_frame_ms must be 10, 20, or 30")
        if not (0 <= self.vad.webrtc_vad_aggressiveness <= 3):
            errors.append("vad.webrtc_vad_aggressiveness must be 0-3")
        if self.vad.smoothing_threshold > self.vad.smoothing_frames:
            errors.append(
                "vad.smoothing_threshold must be <= vad.smoothing_frames"
            )

        # Server
        if not (1024 <= self.server.audio_ws_port <= 65535):
            errors.append("server.audio_ws_port must be 1024-65535")
        if not (1024 <= self.server.api_port <= 65535):
            errors.append("server.api_port must be 1024-65535")

        # Interruption
        if self.interruption.strategy not in ("vad", "always"):
            errors.append("interruption.strategy must be 'vad' or 'always'")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict (masks sensitive fields)."""
        import dataclasses
        d = dataclasses.asdict(self)
        # Mask sensitive values
        if "minicpm" in d and "api_key" in d["minicpm"]:
            key = d["minicpm"]["api_key"]
            d["minicpm"]["api_key"] = (
                key[:6] + "***" + key[-4:] if len(key) > 10 else "***"
            )
        if "esl" in d and "password" in d["esl"]:
            d["esl"]["password"] = "***"
        if "api" in d and "auth_token" in d["api"]:
            d["api"]["auth_token"] = "***" if d["api"]["auth_token"] else ""
        return d

    def summary(self) -> str:
        """Human-readable config summary."""
        return (
            f"{self.app_name} v{self.app_version} [{self.environment}]\n"
            f"  Audio WS Server : {self.server.host}:{self.server.audio_ws_port}\n"
            f"  API Server      : {self.server.host}:{self.server.api_port}\n"
            f"  ESL             : {self.esl.host}:{self.esl.port} ({self.esl.mode.value})\n"
            f"  MiniCPM API     : {self.minicpm.api_base_url}\n"
            f"  Model           : {self.minicpm.model}\n"
            f"  Audio FS rate   : {self.audio.fs_sample_rate}Hz\n"
            f"  Audio Model rate: {self.audio.model_sample_rate}Hz\n"
            f"  VAD             : webrtc={self.vad.use_webrtc_vad} "
            f"agg={self.vad.webrtc_vad_aggressiveness}\n"
            f"  Interruption    : {self.interruption.enabled} "
            f"strategy={self.interruption.strategy}\n"
            f"  Prometheus      : :{self.metrics.prometheus_port}\n"
            f"  Log level       : {self.log.level.value}\n"
            f"  Debug           : {self.debug}\n"
        )


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def _load_dotenv(env_file: Optional[str] = None) -> None:
    """Load .env file if dotenv is available."""
    if not _DOTENV_AVAILABLE:
        return
    path = env_file or ".env"
    if Path(path).exists():
        load_dotenv(path)
        logger.debug("Loaded .env from: %s", path)
    else:
        logger.debug(".env file not found at %s, skipping", path)


def _load_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    if not _YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to load YAML config. "
            "Install with: pip install pyyaml"
        )
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _apply_env_overrides(config: AppConfig) -> None:
    """
    Apply environment variable overrides to config.
    Convention: VOICEBOT__SECTION__KEY=value
    e.g. VOICEBOT__ESL__HOST=192.168.1.10
         VOICEBOT__MINICPM__API_KEY=sk-xxx
         VOICEBOT__DEBUG=true
    """
    prefix = "VOICEBOT__"

    # Direct top-level overrides
    _env_str("VOICEBOT__APP_NAME",    lambda v: setattr(config, "app_name", v))
    _env_str("VOICEBOT__ENVIRONMENT", lambda v: setattr(config, "environment", v))
    _env_bool("VOICEBOT__DEBUG",      lambda v: setattr(config, "debug", v))

    # Server
    _env_str("VOICEBOT__SERVER__HOST",
             lambda v: setattr(config.server, "host", v))
    _env_int("VOICEBOT__SERVER__AUDIO_WS_PORT",
             lambda v: setattr(config.server, "audio_ws_port", v))
    _env_int("VOICEBOT__SERVER__API_PORT",
             lambda v: setattr(config.server, "api_port", v))
    _env_int("VOICEBOT__SERVER__MAX_CONNECTIONS",
             lambda v: setattr(config.server, "max_connections", v))

    # ESL
    _env_str("VOICEBOT__ESL__HOST",
             lambda v: setattr(config.esl, "host", v))
    _env_int("VOICEBOT__ESL__PORT",
             lambda v: setattr(config.esl, "port", v))
    _env_str("VOICEBOT__ESL__PASSWORD",
             lambda v: setattr(config.esl, "password", v))

    # MiniCPM
    _env_str("MINICPM_API_KEY",
             lambda v: setattr(config.minicpm, "api_key", v))
    _env_str("VOICEBOT__MINICPM__API_BASE_URL",
             lambda v: setattr(config.minicpm, "api_base_url", v))
    _env_str("VOICEBOT__MINICPM__MODEL",
             lambda v: setattr(config.minicpm, "model", v))
    _env_str("VOICEBOT__MINICPM__SYSTEM_PROMPT",
             lambda v: setattr(config.minicpm, "system_prompt", v))
    _env_str("VOICEBOT__MINICPM__LANGUAGE",
             lambda v: setattr(config.minicpm, "language", v))
    _env_float("VOICEBOT__MINICPM__TEMPERATURE",
               lambda v: setattr(config.minicpm, "temperature", v))

    # Audio
    _env_int("VOICEBOT__AUDIO__FS_SAMPLE_RATE",
             lambda v: setattr(config.audio, "fs_sample_rate", AudioSampleRate(v)))
    _env_int("VOICEBOT__AUDIO__MODEL_SAMPLE_RATE",
             lambda v: setattr(config.audio, "model_sample_rate", AudioSampleRate(v)))
    _env_int("VOICEBOT__AUDIO__FS_CHUNK_MS",
             lambda v: setattr(config.audio, "fs_chunk_ms", v))

    # VAD
    _env_bool("VOICEBOT__VAD__USE_WEBRTC_VAD",
              lambda v: setattr(config.vad, "use_webrtc_vad", v))
    _env_int("VOICEBOT__VAD__WEBRTC_VAD_AGGRESSIVENESS",
             lambda v: setattr(config.vad, "webrtc_vad_aggressiveness", v))
    _env_float("VOICEBOT__VAD__ENERGY_THRESHOLD_DBFS",
               lambda v: setattr(config.vad, "energy_threshold_dbfs", v))

    # Interruption
    _env_bool("VOICEBOT__INTERRUPTION__ENABLED",
              lambda v: setattr(config.interruption, "enabled", v))

    # Metrics
    _env_bool("VOICEBOT__METRICS__PROMETHEUS_ENABLED",
              lambda v: setattr(config.metrics, "prometheus_enabled", v))
    _env_int("VOICEBOT__METRICS__PROMETHEUS_PORT",
             lambda v: setattr(config.metrics, "prometheus_port", v))

    # API
    _env_str("VOICEBOT__API__AUTH_TOKEN",
             lambda v: setattr(config.api, "auth_token", v))
    _env_bool("VOICEBOT__API__AUTH_ENABLED",
              lambda v: setattr(config.api, "auth_enabled", v))

    # Log
    _env_str("VOICEBOT__LOG__LEVEL",
             lambda v: setattr(config.log, "level", LogLevel(v.upper())))
    _env_bool("VOICEBOT__LOG__JSON_ENABLED",
              lambda v: setattr(config.log, "json_enabled", v))
    _env_bool("VOICEBOT__LOG__FILE_ENABLED",
              lambda v: setattr(config.log, "file_enabled", v))
    _env_str("VOICEBOT__LOG__FILE_PATH",
             lambda v: setattr(config.log, "file_path", v))


# ---------------------------------------------------------------------------
# Env var helper functions
# ---------------------------------------------------------------------------

def _env_str(key: str, setter: Any) -> None:
    val = os.environ.get(key)
    if val is not None:
        try:
            setter(val.strip())
        except Exception as e:
            logger.warning("Config env override failed %s=%r: %s", key, val, e)


def _env_int(key: str, setter: Any) -> None:
    val = os.environ.get(key)
    if val is not None:
        try:
            setter(int(val.strip()))
        except Exception as e:
            logger.warning("Config env override failed %s=%r: %s", key, val, e)


def _env_float(key: str, setter: Any) -> None:
    val = os.environ.get(key)
    if val is not None:
        try:
            setter(float(val.strip()))
        except Exception as e:
            logger.warning("Config env override failed %s=%r: %s", key, val, e)


def _env_bool(key: str, setter: Any) -> None:
    val = os.environ.get(key)
    if val is not None:
        try:
            setter(val.strip().lower() in ("1", "true", "yes", "on"))
        except Exception as e:
            logger.warning("Config env override failed %s=%r: %s", key, val, e)


def _apply_yaml_dict(config: AppConfig, data: Dict[str, Any]) -> None:
    """
    Recursively apply a yaml dict onto AppConfig.
    Keys map directly to dataclass field names.
    """
    def _set(obj: Any, d: Dict[str, Any]) -> None:
        for k, v in d.items():
            if not hasattr(obj, k):
                logger.warning("Unknown config key: %s", k)
                continue
            current = getattr(obj, k)
            if isinstance(v, dict) and hasattr(current, "__dataclass_fields__"):
                _set(current, v)
            else:
                try:
                    # Handle enum fields
                    field_type = type(current)
                    if issubclass(field_type, Enum):
                        setattr(obj, k, field_type(v))
                    else:
                        setattr(obj, k, v)
                except Exception as e:
                    logger.warning(
                        "Failed to set config %s=%r: %s", k, v, e
                    )

    _set(config, data)


# ---------------------------------------------------------------------------
# Logging setup (applied from LogConfig)
# ---------------------------------------------------------------------------

def setup_logging(log_cfg: LogConfig) -> None:
    """Configure root logger from LogConfig."""
    handlers: List[logging.Handler] = []

    # Console handler
    console = logging.StreamHandler()
    if log_cfg.json_enabled:
        try:
            import json as _json

            class JsonFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:
                    log_entry = {
                        "timestamp": self.formatTime(record, self.datefmt),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                    }
                    if record.exc_info:
                        log_entry["exception"] = self.formatException(
                            record.exc_info
                        )
                    if log_cfg.log_caller:
                        log_entry["caller"] = (
                            f"{record.filename}:{record.lineno}"
                        )
                    return _json.dumps(log_entry, ensure_ascii=False)

            console.setFormatter(JsonFormatter(datefmt=log_cfg.date_format))
        except Exception:
            console.setFormatter(
                logging.Formatter(log_cfg.format, datefmt=log_cfg.date_format)
            )
    else:
        fmt = log_cfg.format
        if log_cfg.log_caller:
            fmt = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s"
        console.setFormatter(
            logging.Formatter(fmt, datefmt=log_cfg.date_format)
        )
    handlers.append(console)

    # File handler
    if log_cfg.file_enabled:
        from logging.handlers import RotatingFileHandler
        log_dir = Path(log_cfg.file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_cfg.file_path,
            maxBytes=log_cfg.file_max_bytes,
            backupCount=log_cfg.file_backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(
            logging.Formatter(log_cfg.format, datefmt=log_cfg.date_format)
        )
        handlers.append(fh)

    # Apply to root logger
    root = logging.getLogger()
    root.setLevel(log_cfg.level.value)
    # Remove existing handlers
    root.handlers.clear()
    for h in handlers:
        h.setLevel(log_cfg.level.value)
        root.addHandler(h)

    # Silence noisy third-party loggers
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    logger.debug("Logging configured: level=%s json=%s file=%s",
                 log_cfg.level.value, log_cfg.json_enabled, log_cfg.file_enabled)


# ---------------------------------------------------------------------------
# Main factory function
# ---------------------------------------------------------------------------

def load_config(
    yaml_path: Optional[str] = None,
    env_file: Optional[str] = None,
    validate: bool = True,
) -> AppConfig:
    """
    Load and return AppConfig.

    Priority (highest to lowest):
      1. Environment variables  (VOICEBOT__* or MINICPM_API_KEY)
      2. YAML config file       (if yaml_path provided or config.yaml exists)
      3. .env file              (if present)
      4. Dataclass defaults

    Args:
        yaml_path:  Path to YAML config file. If None, looks for config.yaml
        env_file:   Path to .env file. If None, looks for .env
        validate:   If True, raises ValueError on invalid config

    Returns:
        AppConfig instance, fully populated and validated

    Raises:
        ValueError: If validate=True and config has errors
        FileNotFoundError: If yaml_path is specified but doesn't exist
    """
    # Step 1: Load .env (lowest priority, sets env vars)
    _load_dotenv(env_file)

    # Step 2: Start with defaults
    config = AppConfig()

    # Step 3: Apply YAML overrides
    _yaml_path = yaml_path or "config.yaml"
    if yaml_path and not Path(yaml_path).exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    if Path(_yaml_path).exists():
        try:
            yaml_data = _load_yaml(_yaml_path)
            _apply_yaml_dict(config, yaml_data)
            logger.debug("Loaded YAML config from: %s", _yaml_path)
        except Exception as e:
            logger.error("Failed to load YAML config %s: %s", _yaml_path, e)
            raise

    # Step 4: Apply environment variable overrides (highest priority)
    _apply_env_overrides(config)

    # Step 5: Validate
    if validate:
        errors = config.validate()
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise ValueError(error_msg)

    return config


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_config_lock = threading.Lock()
_global_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Return the global AppConfig instance. Must call init_config() first."""
    if _global_config is None:
        raise RuntimeError(
            "Config not initialized. Call init_config() or load_config() first."
        )
    return _global_config


def init_config(
    yaml_path: Optional[str] = None,
    env_file: Optional[str] = None,
    validate: bool = True,
) -> AppConfig:
    """
    Initialize and cache the global AppConfig singleton.
    Safe to call multiple times (idempotent after first call).
    """
    global _global_config
    with _config_lock:
        if _global_config is None:
            _global_config = load_config(
                yaml_path=yaml_path,
                env_file=env_file,
                validate=validate,
            )
            setup_logging(_global_config.log)
            logger.info("Configuration loaded successfully")
            logger.debug("\n%s", _global_config.summary())
        return _global_config


def reload_config(
    yaml_path: Optional[str] = None,
    env_file: Optional[str] = None,
    validate: bool = True,
) -> AppConfig:
    """
    Force reload config (e.g. on SIGHUP).
    Thread-safe.
    """
    global _global_config
    with _config_lock:
        new_cfg = load_config(
            yaml_path=yaml_path,
            env_file=env_file,
            validate=validate,
        )
        _global_config = new_cfg
        setup_logging(new_cfg.log)
        logger.info("Configuration reloaded")
        logger.debug("\n%s", new_cfg.summary())
        return new_cfg
