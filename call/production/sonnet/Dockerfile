# ============================================================
# VoiceBot Dockerfile
# Multi-stage build for minimal production image
# ============================================================

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile webrtcvad + numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into /install (isolated from system)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

LABEL maintainer="voicebot"
LABEL description="VoiceBot - MiniCPM-o 4.5 full-duplex voice AI for FreeSWITCH"

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY config.py       .
COPY audio_utils.py  .
COPY metrics.py      .
COPY esl.py          .
COPY audio.py        .
COPY ai_client.py    .
COPY session.py      .
COPY api.py          .
COPY main.py         .

# Copy default config (can be overridden by volume mount)
COPY config.yaml     .

# Create non-root user for security
RUN groupadd -r voicebot && \
    useradd -r -g voicebot -d /app -s /sbin/nologin voicebot && \
    mkdir -p /app/logs /var/lib/voicebot/recordings && \
    chown -R voicebot:voicebot /app /var/lib/voicebot

USER voicebot

# ---- Ports ----
# 8765  - mod_audio_stream WebSocket (audio)
# 8080  - REST management API
# 9090  - Prometheus metrics
EXPOSE 8765 8080 9090

# ---- Health check ----
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

# ---- Volumes ----
# /app/config.yaml          - config file
# /app/logs                 - log files
# /var/lib/voicebot/recordings - call recordings
VOLUME ["/app/logs", "/var/lib/voicebot/recordings"]

# ---- Entry point ----
ENTRYPOINT ["python", "main.py"]
CMD ["--config", "/app/config.yaml"]
