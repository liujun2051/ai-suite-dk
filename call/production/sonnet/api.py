"""
api.py - REST management API for VoiceBot

Provides:
  - Session management endpoints (list, get, terminate, inject)
  - System health and readiness checks
  - Metrics snapshot endpoint
  - Real-time stats via Server-Sent Events (SSE)
  - Call transfer and hold endpoints
  - Authentication (Bearer token, optional)
  - Rate limiting
  - Request logging middleware
  - CORS support

Built with aiohttp (lightweight, async-native, no framework overhead).

Endpoints:
  GET  /health                      - Liveness probe
  GET  /ready                       - Readiness probe
  GET  /metrics                     - Internal metrics snapshot (JSON)
  GET  /sessions                    - List all active sessions
  GET  /sessions/{session_id}       - Get session details
  DELETE /sessions/{session_id}     - Terminate a session
  POST /sessions/{session_id}/inject       - Inject text message
  POST /sessions/{session_id}/transfer     - Transfer call
  POST /sessions/{session_id}/hold         - Hold call
  POST /sessions/{session_id}/unhold       - Unhold call
  POST /sessions/{session_id}/record/start - Start recording
  POST /sessions/{session_id}/record/stop  - Stop recording
  GET  /sessions/completed          - Recently completed sessions
  GET  /stats                       - System-wide stats
  GET  /stream/stats                - SSE real-time stats stream
  GET  /version                     - App version info
"""

import asyncio
import json
import logging
import time
import hashlib
import hmac
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

try:
    from aiohttp import web
    from aiohttp.web import (
        Request, Response, StreamResponse,
        middleware, HTTPException,
    )
    import aiohttp_cors
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_VERSION         = "v1"
DEFAULT_PAGE_SIZE   = 20
MAX_PAGE_SIZE       = 100
SSE_PING_INTERVAL   = 15.0     # seconds between SSE keepalive pings
SSE_STATS_INTERVAL  = 2.0      # seconds between SSE stats pushes


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Simple in-memory token bucket rate limiter.
    Keyed by IP address.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst: int = 10,
    ):
        self._rpm        = requests_per_minute
        self._burst      = burst
        self._buckets:   Dict[str, List[float]] = defaultdict(list)
        self._lock       = asyncio.Lock()

    async def is_allowed(self, key: str) -> bool:
        """
        Check if request from key is within rate limit.
        Uses sliding window (last 60 seconds).
        """
        async with self._lock:
            now    = time.monotonic()
            window = now - 60.0
            times  = self._buckets[key]

            # Evict old entries
            while times and times[0] < window:
                times.pop(0)

            if len(times) >= self._rpm:
                return False

            times.append(now)
            return True

    async def cleanup(self) -> None:
        """Remove stale entries (call periodically)."""
        async with self._lock:
            now    = time.monotonic()
            window = now - 60.0
            stale  = [
                k for k, v in self._buckets.items()
                if not v or v[-1] < window
            ]
            for k in stale:
                del self._buckets[k]


# ---------------------------------------------------------------------------
# Request/response helpers
# ---------------------------------------------------------------------------

def _json_response(
    data:   Any,
    status: int = 200,
) -> "Response":
    """Return a JSON response."""
    return web.Response(
        status=status,
        content_type="application/json",
        text=json.dumps(data, ensure_ascii=False, default=str),
    )


def _error_response(
    message: str,
    status:  int = 400,
    code:    str = "error",
) -> "Response":
    """Return a standardized JSON error response."""
    return _json_response(
        {"error": {"code": code, "message": message}},
        status=status,
    )


def _ok_response(message: str = "ok", **kwargs) -> "Response":
    """Return a simple success response."""
    return _json_response({"status": "ok", "message": message, **kwargs})


def _get_client_ip(request: "Request") -> str:
    """Extract client IP, respecting X-Forwarded-For."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote or "unknown"


def _get_bearer_token(request: "Request") -> str:
    """Extract Bearer token from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return ""


def _constant_time_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison for token validation."""
    return hmac.compare_digest(
        a.encode("utf-8"),
        b.encode("utf-8"),
    )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

def make_auth_middleware(token: str, enabled: bool):
    """
    Authentication middleware.
    Skips auth for /health and /ready endpoints.
    """
    @middleware
    async def auth_middleware(request: "Request", handler):
        # Always allow health/ready probes (for k8s/load balancer)
        if request.path in ("/health", "/ready", "/version"):
            return await handler(request)

        if not enabled:
            return await handler(request)

        provided = _get_bearer_token(request)
        if not provided or not _constant_time_compare(provided, token):
            return _error_response(
                "Unauthorized: invalid or missing Bearer token",
                status=401,
                code="unauthorized",
            )
        return await handler(request)

    return auth_middleware


def make_rate_limit_middleware(limiter: RateLimiter, enabled: bool):
    """Rate limiting middleware keyed by client IP."""
    @middleware
    async def rate_limit_middleware(request: "Request", handler):
        if not enabled:
            return await handler(request)

        ip = _get_client_ip(request)
        allowed = await limiter.is_allowed(ip)
        if not allowed:
            return _error_response(
                "Rate limit exceeded. Please slow down.",
                status=429,
                code="rate_limited",
            )
        return await handler(request)

    return rate_limit_middleware


def make_logging_middleware():
    """Request/response logging middleware."""
    @middleware
    async def logging_middleware(request: "Request", handler):
        start   = time.monotonic()
        method  = request.method
        path    = request.path
        ip      = _get_client_ip(request)

        try:
            response = await handler(request)
            elapsed  = (time.monotonic() - start) * 1000
            logger.info(
                "API %s %s %d %.1fms [%s]",
                method, path, response.status, elapsed, ip,
            )
            return response

        except HTTPException as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.warning(
                "API %s %s %d %.1fms [%s]",
                method, path, e.status, elapsed, ip,
            )
            raise

        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.error(
                "API %s %s 500 %.1fms [%s] error=%s",
                method, path, elapsed, ip, e, exc_info=True,
            )
            return _error_response(
                f"Internal server error: {e}",
                status=500,
                code="internal_error",
            )

    return logging_middleware


def make_cors_middleware(origins: List[str]):
    """
    Simple CORS middleware.
    For full CORS support with aiohttp-cors, see setup_cors().
    """
    @middleware
    async def cors_middleware(request: "Request", handler):
        # Handle preflight
        if request.method == "OPTIONS":
            resp = web.Response()
            resp.headers["Access-Control-Allow-Origin"]  = (
                ", ".join(origins) if origins != ["*"] else "*"
            )
            resp.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, DELETE, OPTIONS"
            )
            resp.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization"
            )
            resp.headers["Access-Control-Max-Age"] = "86400"
            return resp

        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = (
            ", ".join(origins) if origins != ["*"] else "*"
        )
        return response

    return cors_middleware


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

class APIHandlers:
    """
    All REST endpoint handlers.
    Receives dependencies via constructor (session manager, metrics, config).
    """

    def __init__(
        self,
        session_manager: "Any",    # SessionManager from session.py
        metrics:         "Any",    # MetricsRegistry from metrics.py
        esl_client:      "Any",    # ESLClient from esl.py
        audio_server:    "Any",    # AudioServer from audio.py
        cfg:             "Any",    # AppConfig from config.py
    ):
        self._sm      = session_manager
        self._metrics = metrics
        self._esl     = esl_client
        self._audio   = audio_server
        self._cfg     = cfg

        # SSE subscriber set (for /stream/stats)
        self._sse_subscribers: Set[StreamResponse] = set()
        self._sse_lock = asyncio.Lock()

    # 外呼
    
    # 在 APIHandlers 中新增以下方法
    # 在 APIServer._add_routes() 中注册
    
    # -----------------------------------------------------------------------
    # Outbound: single call
    # -----------------------------------------------------------------------
    
    async def handle_place_call(self, request: "Request") -> "Response":
        """
        POST /calls
        Place a single outbound call.
    
        Body JSON:
          destination      (str, required) : phone number to dial
          caller_id_num    (str)           : CLI number shown to callee
          caller_id_name   (str)           : CLI name shown to callee
          gateway          (str)           : FreeSWITCH SIP gateway name
          ring_timeout     (int)           : seconds to ring before no-answer
          max_retries      (int)           : retry attempts on no-answer/busy
          retry_delay      (float)         : seconds between retries
          system_prompt    (str)           : AI system prompt for this call
          greeting_text    (str)           : first thing AI says when answered
          language         (str)           : language code e.g. zh-CN
          webhook_url      (str)           : HTTP endpoint for call events
          webhook_secret   (str)           : HMAC secret for webhook signing
          scheduled_at     (float)         : Unix timestamp to dial (future)
          metadata         (dict)          : custom key/value (passed to webhook)
        """
        from outbound import OutboundCallConfig, Priority
    
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON body", 400, "bad_request")
    
        destination = body.get("destination", "").strip()
        if not destination:
            return _error_response(
                "Field 'destination' is required",
                400, "validation_error",
            )
    
        # Validate destination format
        digits = "".join(c for c in destination if c.isdigit() or c == "+")
        if len(digits) < 3:
            return _error_response(
                "Invalid destination number",
                400, "validation_error",
            )
    
        try:
            call_cfg = OutboundCallConfig(
                destination=destination,
                caller_id_num=body.get(
                    "caller_id_num",
                    self._cfg.minicpm.system_prompt and "8000" or "8000",
                ),
                caller_id_name=body.get("caller_id_name", "VoiceBot"),
                gateway=body.get("gateway", "default"),
                ring_timeout=int(body.get("ring_timeout", 30)),
                max_retries=int(body.get("max_retries", 2)),
                retry_delay=float(body.get("retry_delay", 60.0)),
                system_prompt=body.get(
                    "system_prompt", self._cfg.minicpm.system_prompt
                ),
                greeting_text=body.get("greeting_text", ""),
                language=body.get("language", self._cfg.minicpm.language),
                temperature=float(body.get(
                    "temperature", self._cfg.minicpm.temperature
                )),
                voice_id=body.get("voice_id", self._cfg.minicpm.voice_id),
                webhook_url=body.get("webhook_url"),
                webhook_secret=body.get("webhook_secret"),
                scheduled_at=body.get("scheduled_at"),
                metadata=body.get("metadata", {}),
            )
        except (ValueError, TypeError) as e:
            return _error_response(
                f"Invalid parameter: {e}", 400, "validation_error"
            )
    
        call = await self._outbound.place_call(call_cfg)
    
        return _json_response(call.to_dict(), status=202)
    
    
    async def handle_list_calls(self, request: "Request") -> "Response":
        """
        GET /calls
        List active outbound calls.
    
        Query params:
          state (str): filter by state
          limit (int): max results
        """
        from outbound import CallState
    
        state_str = request.rel_url.query.get("state", "")
        state_filter = None
        if state_str:
            try:
                state_filter = CallState(state_str)
            except ValueError:
                return _error_response(
                    f"Invalid state: {state_str}", 400, "validation_error"
                )
    
        try:
            limit = min(
                int(request.rel_url.query.get("limit", 100)),
                MAX_PAGE_SIZE,
            )
        except (ValueError, TypeError):
            limit = 100
    
        calls = self._outbound.list_calls(
            state_filter=state_filter, limit=limit
        )
        return _json_response({
            "count": len(calls),
            "calls": calls,
        })
    
    
    async def handle_get_call(self, request: "Request") -> "Response":
        """GET /calls/{outbound_id}"""
        outbound_id = request.match_info["outbound_id"]
        call = self._outbound.get_call(outbound_id)
    
        if call is None:
            # Check completed archive
            completed = self._outbound.list_completed(limit=1000)
            for r in completed:
                if r["outbound_id"] == outbound_id:
                    return _json_response(r)
            return _error_response(
                f"Call not found: {outbound_id}",
                404, "not_found",
            )
    
        return _json_response(call.to_dict())
    
    
    async def handle_cancel_call(self, request: "Request") -> "Response":
        """DELETE /calls/{outbound_id}"""
        outbound_id = request.match_info["outbound_id"]
        ok = await self._outbound.cancel_call(outbound_id)
        if not ok:
            return _error_response(
                f"Call not found: {outbound_id}",
                404, "not_found",
            )
        return _ok_response("Call cancelled", outbound_id=outbound_id)
    
    
    async def handle_completed_calls(
        self, request: "Request"
    ) -> "Response":
        """
        GET /calls/completed
        List recently completed outbound calls.
        """
        try:
            limit = min(
                int(request.rel_url.query.get("limit", DEFAULT_PAGE_SIZE)),
                MAX_PAGE_SIZE,
            )
        except (ValueError, TypeError):
            limit = DEFAULT_PAGE_SIZE
    
        completed = self._outbound.list_completed(limit=limit)
        return _json_response({
            "count": len(completed),
            "calls": completed,
        })
    
    
    # -----------------------------------------------------------------------
    # Outbound: campaigns
    # -----------------------------------------------------------------------
    
    async def handle_create_campaign(
        self, request: "Request"
    ) -> "Response":
        """
        POST /campaigns
        Create a new outbound campaign.
    
        Body JSON:
          name                (str, required)   : campaign name
          calls               (list, required)  : list of call configs
            - destination     (str, required)
            - caller_id_num   (str)
            - system_prompt   (str)
            - greeting_text   (str)
            - metadata        (dict)
          max_concurrent      (int)             : simultaneous calls (default 5)
          calls_per_minute    (float)           : rate limit
          default_caller_id_num  (str)
          default_caller_id_name (str)
          default_gateway        (str)
          default_ring_timeout   (int)
          default_max_retries    (int)
          default_system_prompt  (str)
          default_greeting_text  (str)
          business_hours_start   (str)          : "09:00"
          business_hours_end     (str)          : "18:00"
          business_days          (list[int])    : [0,1,2,3,4] (Mon-Fri)
          webhook_url            (str)
          auto_start             (bool)         : start immediately (default false)
        """
        from outbound import (
            CampaignConfig, OutboundCallConfig, Priority
        )
    
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON body", 400, "bad_request")
    
        name = body.get("name", "").strip()
        if not name:
            return _error_response(
                "Field 'name' is required", 400, "validation_error"
            )
    
        raw_calls = body.get("calls", [])
        if not raw_calls:
            return _error_response(
                "Field 'calls' must be a non-empty list",
                400, "validation_error",
            )
    
        if len(raw_calls) > MAX_CAMPAIGN_SIZE:
            return _error_response(
                f"Campaign too large: {len(raw_calls)} calls "
                f"(max {MAX_CAMPAIGN_SIZE})",
                400, "validation_error",
            )
    
        # Build call configs
        call_cfgs: List[OutboundCallConfig] = []
        errors = []
        for i, raw in enumerate(raw_calls):
            dest = raw.get("destination", "").strip()
            if not dest:
                errors.append(f"calls[{i}]: destination required")
                continue
            try:
                call_cfgs.append(OutboundCallConfig(
                    destination=dest,
                    caller_id_num=raw.get("caller_id_num", ""),
                    caller_id_name=raw.get("caller_id_name", ""),
                    gateway=raw.get("gateway", ""),
                    ring_timeout=int(raw.get("ring_timeout",
                        DEFAULT_RING_TIMEOUT)),
                    max_retries=int(raw.get("max_retries",
                        DEFAULT_MAX_RETRIES)),
                    retry_delay=float(raw.get("retry_delay",
                        DEFAULT_RETRY_DELAY)),
                    system_prompt=raw.get("system_prompt", ""),
                    greeting_text=raw.get("greeting_text", ""),
                    language=raw.get("language",
                        self._cfg.minicpm.language),
                    webhook_url=body.get("webhook_url"),
                    metadata=raw.get("metadata", {}),
                    priority=Priority(
                        int(raw.get("priority", Priority.NORMAL.value))
                    ),
                    scheduled_at=raw.get("scheduled_at"),
                ))
            except Exception as e:
                errors.append(f"calls[{i}]: {e}")
    
        if errors:
            return _error_response(
                f"Validation errors: {'; '.join(errors)}",
                400, "validation_error",
            )
    
        # Parse business hours
        bh_start = None
        bh_end   = None
        if body.get("business_hours_start"):
            try:
                from datetime import time as dt_time
                h, m = body["business_hours_start"].split(":")
                bh_start = dt_time(int(h), int(m))
                h, m = body["business_hours_end"].split(":")
                bh_end = dt_time(int(h), int(m))
            except Exception as e:
                return _error_response(
                    f"Invalid business hours format: {e}",
                    400, "validation_error",
                )
    
        from outbound import MAX_CAMPAIGN_SIZE
        try:
            campaign_cfg = CampaignConfig(
                name=name,
                calls=call_cfgs,
                max_concurrent=min(
                    int(body.get("max_concurrent", 5)),
                    self._cfg.server.max_connections,
                ),
                calls_per_minute=float(
                    body.get("calls_per_minute", 30.0)
                ),
                default_caller_id_num=body.get(
                    "default_caller_id_num", "8000"
                ),
                default_caller_id_name=body.get(
                    "default_caller_id_name", "VoiceBot"
                ),
                default_gateway=body.get("default_gateway", "default"),
                default_ring_timeout=int(
                    body.get("default_ring_timeout", 30)
                ),
                default_max_retries=int(
                    body.get("default_max_retries", 2)
                ),
                default_retry_delay=float(
                    body.get("default_retry_delay", 60.0)
                ),
                default_system_prompt=body.get(
                    "default_system_prompt",
                    self._cfg.minicpm.system_prompt,
                ),
                default_greeting_text=body.get(
                    "default_greeting_text", ""
                ),
                business_hours_start=bh_start,
                business_hours_end=bh_end,
                business_days=body.get(
                    "business_days", [0,1,2,3,4]
                ),
                webhook_url=body.get("webhook_url"),
            )
        except (ValueError, TypeError) as e:
            return _error_response(
                f"Campaign config error: {e}", 400, "validation_error"
            )
    
        campaign = await self._outbound.create_campaign(campaign_cfg)
    
        # Auto-start?
        if body.get("auto_start", False):
            await self._outbound.start_campaign(campaign.campaign_id)
    
        return _json_response(campaign.stats, status=201)
    
    
    async def handle_list_campaigns(
        self, request: "Request"
    ) -> "Response":
        """GET /campaigns"""
        campaigns = self._outbound.list_campaigns()
        return _json_response({
            "count": len(campaigns),
            "campaigns": campaigns,
        })
    
    
    async def handle_get_campaign(
        self, request: "Request"
    ) -> "Response":
        """GET /campaigns/{campaign_id}"""
        campaign_id = request.match_info["campaign_id"]
        campaign    = self._outbound.get_campaign(campaign_id)
        if campaign is None:
            return _error_response(
                f"Campaign not found: {campaign_id}",
                404, "not_found",
            )
        return _json_response(campaign.stats)
    
    
    async def handle_campaign_action(
        self, request: "Request"
    ) -> "Response":
        """
        POST /campaigns/{campaign_id}/action
        Perform an action on a campaign.
    
        Body JSON:
          action (str): start | pause | resume | cancel
        """
        campaign_id = request.match_info["campaign_id"]
    
        try:
            body   = await request.json()
            action = body.get("action", "").strip().lower()
        except Exception:
            return _error_response("Invalid JSON body", 400, "bad_request")
    
        if action not in ("start", "pause", "resume", "cancel"):
            return _error_response(
                "action must be: start | pause | resume | cancel",
                400, "validation_error",
            )
    
        handlers = {
            "start":  self._outbound.start_campaign,
            "pause":  self._outbound.pause_campaign,
            "resume": self._outbound.resume_campaign,
            "cancel": self._outbound.cancel_campaign,
        }
    
        ok = await handlers[action](campaign_id)
        if not ok:
            return _error_response(
                f"Campaign not found: {campaign_id}",
                404, "not_found",
            )
    
        return _ok_response(
            f"Campaign {action} successful",
            campaign_id=campaign_id,
            action=action,
        )
    
    
    async def handle_get_campaign_calls(
        self, request: "Request"
    ) -> "Response":
        """
        GET /campaigns/{campaign_id}/calls
        List calls within a campaign.
        """
        from outbound import CallState
    
        campaign_id = request.match_info["campaign_id"]
        campaign    = self._outbound.get_campaign(campaign_id)
        if campaign is None:
            return _error_response(
                f"Campaign not found: {campaign_id}",
                404, "not_found",
            )
    
        state_str = request.rel_url.query.get("state", "")
        state_filter = None
        if state_str:
            try:
                state_filter = CallState(state_str)
            except ValueError:
                pass
    
        try:
            limit = min(
                int(request.rel_url.query.get("limit", 100)),
                MAX_PAGE_SIZE,
            )
        except (ValueError, TypeError):
            limit = 100
    
        calls = campaign.get_calls(state_filter=state_filter, limit=limit)
        return _json_response({
            "campaign_id": campaign_id,
            "count":       len(calls),
            "calls":       calls,
        })
    
    
    # -----------------------------------------------------------------------
    # DNC list
    # -----------------------------------------------------------------------
    
    async def handle_dnc_add(self, request: "Request") -> "Response":
        """
        POST /dnc
        Add number(s) to DNC list.
    
        Body JSON:
          numbers (list[str]): phone numbers to add
        """
        try:
            body    = await request.json()
            numbers = body.get("numbers", [])
            if isinstance(numbers, str):
                numbers = [numbers]
        except Exception:
            return _error_response("Invalid JSON body", 400, "bad_request")
    
        if not numbers:
            return _error_response(
                "Field 'numbers' must be a non-empty list",
                400, "validation_error",
            )
    
        added = 0
        for num in numbers:
            if isinstance(num, str) and num.strip():
                await self._outbound.dnc.add(num.strip())
                added += 1
    
        return _ok_response(
            f"Added {added} numbers to DNC list",
            added=added,
            dnc_size=self._outbound.dnc.size,
        )
    
    
    async def handle_dnc_check(self, request: "Request") -> "Response":
        """
        GET /dnc/{number}
        Check if a number is on the DNC list.
        """
        number = request.match_info["number"]
        on_dnc = await self._outbound.dnc.check(number)
        return _json_response({
            "number": number,
            "on_dnc": on_dnc,
        })
    
    
    async def handle_outbound_stats(
        self, request: "Request"
    ) -> "Response":
        """GET /outbound/stats"""
        return _json_response(self._outbound.get_stats())  

    # -----------------------------------------------------------------------
    # Health & readiness
    # -----------------------------------------------------------------------

    async def handle_health(self, request: "Request") -> "Response":
        """
        GET /health
        Liveness probe — returns 200 if process is alive.
        Used by k8s/docker healthcheck.
        """
        return _json_response({
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._start_time,
        })

    async def handle_ready(self, request: "Request") -> "Response":
        """
        GET /ready
        Readiness probe — returns 200 only if system is ready to handle calls.
        Checks: ESL connected, AudioServer listening, AI credentials present.
        """
        checks: Dict[str, bool] = {}
        details: Dict[str, Any] = {}

        # ESL connection
        checks["esl_connected"] = self._esl.is_connected
        details["esl"] = self._esl.connection_info.to_dict()

        # AudioServer
        checks["audio_server"] = self._audio is not None
        details["audio_server"] = {
            "active_connections": self._audio.active_count,
            "pending_connections": self._audio.pending_count,
        }

        # AI credentials
        checks["ai_configured"] = bool(self._cfg.minicpm.api_key)

        # Active sessions within limit
        active = self._sm.active_count
        checks["within_capacity"] = active < self._cfg.server.max_connections
        details["sessions"] = {
            "active": active,
            "max": self._cfg.server.max_connections,
        }

        all_ready = all(checks.values())
        status    = 200 if all_ready else 503

        return _json_response(
            {
                "ready":   all_ready,
                "checks":  checks,
                "details": details,
                "timestamp": time.time(),
            },
            status=status,
        )

    async def handle_version(self, request: "Request") -> "Response":
        """GET /version"""
        return _json_response({
            "app":         self._cfg.app_name,
            "version":     self._cfg.app_version,
            "environment": self._cfg.environment,
            "api_version": API_VERSION,
        })

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    async def handle_metrics(self, request: "Request") -> "Response":
        """
        GET /metrics
        Returns internal metrics snapshot as JSON.
        For Prometheus scraping use the dedicated Prometheus port.
        """
        summary = self._metrics.get_summary()
        summary["audio_server"] = self._audio.get_all_stats()["server"]
        summary["esl"]          = self._esl.connection_info.to_dict()
        return _json_response(summary)

    async def handle_stats(self, request: "Request") -> "Response":
        """
        GET /stats
        System-wide statistics snapshot.
        """
        return _json_response({
            "sessions":     self._sm.get_stats(),
            "metrics":      self._metrics.get_summary(),
            "audio_server": self._audio.get_all_stats()["server"],
            "esl":          self._esl.connection_info.to_dict(),
            "timestamp":    time.time(),
        })

    # -----------------------------------------------------------------------
    # Session CRUD
    # -----------------------------------------------------------------------

    async def handle_list_sessions(self, request: "Request") -> "Response":
        """
        GET /sessions
        List all active sessions with summary info.

        Query params:
          limit (int): max sessions to return (default 20, max 100)
        """
        try:
            limit = min(
                int(request.rel_url.query.get("limit", DEFAULT_PAGE_SIZE)),
                MAX_PAGE_SIZE,
            )
        except (ValueError, TypeError):
            limit = DEFAULT_PAGE_SIZE

        sessions = self._sm.list_sessions()
        return _json_response({
            "count":    len(sessions),
            "sessions": [s.to_dict() for s in sessions[:limit]],
        })

    async def handle_get_session(self, request: "Request") -> "Response":
        """
        GET /sessions/{session_id}
        Get detailed info for a specific session.
        """
        session_id = request.match_info["session_id"]
        session    = self._sm.get_session(session_id)

        if session is None:
            return _error_response(
                f"Session not found: {session_id}",
                status=404,
                code="not_found",
            )

        info = session.get_info().to_dict()

        # Add audio stream stats
        stream_stats = self._audio.get_stream_stats(session.call_uuid)
        if stream_stats:
            info["audio_stream"] = stream_stats

        # Add AI client info
        if session._ai:
            info["ai_client"] = session._ai.get_info()

        return _json_response(info)

    async def handle_terminate_session(
        self, request: "Request"
    ) -> "Response":
        """
        DELETE /sessions/{session_id}
        Terminate an active session (hang up the call).

        Body (optional JSON):
          reason (str): termination reason for logging
        """
        session_id = request.match_info["session_id"]

        # Parse optional body
        reason = "api_request"
        try:
            if request.content_length and request.content_length > 0:
                body   = await request.json()
                reason = body.get("reason", "api_request")[:64]
        except Exception:
            pass

        ok = await self._sm.terminate_session(session_id, reason=reason)
        if not ok:
            return _error_response(
                f"Session not found: {session_id}",
                status=404,
                code="not_found",
            )

        return _ok_response(
            f"Session {session_id[:8]} terminating",
            session_id=session_id,
        )

    async def handle_inject_message(
        self, request: "Request"
    ) -> "Response":
        """
        POST /sessions/{session_id}/inject
        Inject a text message into the AI conversation mid-call.
        Useful for supervisor whisper or context injection.

        Body JSON:
          text (str, required): message to inject
        """
        session_id = request.match_info["session_id"]

        try:
            body = await request.json()
        except Exception:
            return _error_response(
                "Invalid JSON body", status=400, code="bad_request"
            )

        text = body.get("text", "").strip()
        if not text:
            return _error_response(
                "Field 'text' is required and must be non-empty",
                status=400,
                code="validation_error",
            )
        if len(text) > 2000:
            return _error_response(
                "Field 'text' must be ≤ 2000 characters",
                status=400,
                code="validation_error",
            )

        session = self._sm.get_session(session_id)
        if session is None:
            return _error_response(
                f"Session not found: {session_id}",
                status=404,
                code="not_found",
            )

        ok = await self._sm.inject_message(session_id, text)
        if not ok:
            return _error_response(
                "Failed to inject message (session may be terminating)",
                status=409,
                code="conflict",
            )

        return _ok_response(
            "Message injected",
            session_id=session_id,
            text_preview=text[:80],
        )

    async def handle_transfer(self, request: "Request") -> "Response":
        """
        POST /sessions/{session_id}/transfer
        Transfer a call to another extension or SIP URI.

        Body JSON:
          destination (str, required): extension or SIP URI
          context     (str, optional): dialplan context (default: "default")
        """
        session_id = request.match_info["session_id"]

        try:
            body = await request.json()
        except Exception:
            return _error_response(
                "Invalid JSON body", status=400, code="bad_request"
            )

        destination = body.get("destination", "").strip()
        context     = body.get("context", "default").strip()

        if not destination:
            return _error_response(
                "Field 'destination' is required",
                status=400,
                code="validation_error",
            )

        session = self._sm.get_session(session_id)
        if session is None:
            return _error_response(
                f"Session not found: {session_id}",
                status=404,
                code="not_found",
            )

        ok = await self._sm.transfer_call(session_id, destination, context)
        if not ok:
            return _error_response(
                f"Transfer failed for session {session_id}",
                status=500,
                code="transfer_failed",
            )

        return _ok_response(
            "Call transferred",
            session_id=session_id,
            destination=destination,
        )

    async def handle_hold(self, request: "Request") -> "Response":
        """
        POST /sessions/{session_id}/hold
        Put a call on hold.
        """
        session_id = request.match_info["session_id"]
        session    = self._sm.get_session(session_id)

        if session is None:
            return _error_response(
                f"Session not found: {session_id}",
                status=404,
                code="not_found",
            )

        ok = await session.hold()
        if not ok:
            return _error_response(
                "Hold failed",
                status=500,
                code="hold_failed",
            )
        return _ok_response("Call on hold", session_id=session_id)

    async def handle_unhold(self, request: "Request") -> "Response":
        """
        POST /sessions/{session_id}/unhold
        Resume a held call.
        """
        session_id = request.match_info["session_id"]
        session    = self._sm.get_session(session_id)

        if session is None:
            return _error_response(
                f"Session not found: {session_id}",
                status=404,
                code="not_found",
            )

        ok = await session.unhold()
        if not ok:
            return _error_response(
                "Unhold failed",
                status=500,
                code="unhold_failed",
            )
        return _ok_response("Call resumed", session_id=session_id)

    async def handle_record_start(self, request: "Request") -> "Response":
        """
        POST /sessions/{session_id}/record/start
        Start recording a call.

        Body JSON:
          file_path (str, required): recording file path on FreeSWITCH server
        """
        session_id = request.match_info["session_id"]
        session    = self._sm.get_session(session_id)

        if session is None:
            return _error_response(
                f"Session not found: {session_id}",
                status=404,
                code="not_found",
            )

        try:
            body      = await request.json()
            file_path = body.get("file_path", "").strip()
        except Exception:
            file_path = ""

        if not file_path:
            # Auto-generate filename based on call UUID and timestamp
            file_path = (
                f"/var/lib/freeswitch/recordings/"
                f"{session.call_uuid}_{int(time.time())}.wav"
            )

        ok = await session.start_recording(file_path)
        if not ok:
            return _error_response(
                "Failed to start recording",
                status=500,
                code="record_failed",
            )

        return _ok_response(
            "Recording started",
            session_id=session_id,
            file_path=file_path,
        )

    async def handle_record_stop(self, request: "Request") -> "Response":
        """
        POST /sessions/{session_id}/record/stop
        Stop call recording.

        Body JSON:
          file_path (str, required): same path used in record/start
        """
        session_id = request.match_info["session_id"]
        session    = self._sm.get_session(session_id)

        if session is None:
            return _error_response(
                f"Session not found: {session_id}",
                status=404,
                code="not_found",
            )

        try:
            body      = await request.json()
            file_path = body.get("file_path", "").strip()
        except Exception:
            file_path = ""

        if not file_path:
            return _error_response(
                "Field 'file_path' is required",
                status=400,
                code="validation_error",
            )

        ok = await session.stop_recording(file_path)
        if not ok:
            return _error_response(
                "Failed to stop recording",
                status=500,
                code="record_failed",
            )

        return _ok_response(
            "Recording stopped",
            session_id=session_id,
            file_path=file_path,
        )

    async def handle_update_prompt(self, request: "Request") -> "Response":
        """
        POST /sessions/{session_id}/prompt
        Update system prompt mid-call.

        Body JSON:
          prompt (str, required): new system prompt
        """
        session_id = request.match_info["session_id"]
        session    = self._sm.get_session(session_id)

        if session is None:
            return _error_response(
                f"Session not found: {session_id}",
                status=404,
                code="not_found",
            )

        try:
            body   = await request.json()
            prompt = body.get("prompt", "").strip()
        except Exception:
            return _error_response(
                "Invalid JSON body", status=400, code="bad_request"
            )

        if not prompt:
            return _error_response(
                "Field 'prompt' is required",
                status=400,
                code="validation_error",
            )

        ok = await session.update_system_prompt(prompt)
        if not ok:
            return _error_response(
                "Failed to update prompt",
                status=500,
                code="update_failed",
            )

        return _ok_response(
            "System prompt updated",
            session_id=session_id,
        )

    async def handle_completed_sessions(
        self, request: "Request"
    ) -> "Response":
        """
        GET /sessions/completed
        List recently completed sessions (last N).

        Query params:
          limit (int): max results (default 20)
        """
        try:
            limit = min(
                int(request.rel_url.query.get("limit", DEFAULT_PAGE_SIZE)),
                MAX_PAGE_SIZE,
            )
        except (ValueError, TypeError):
            limit = DEFAULT_PAGE_SIZE

        completed = self._sm.list_completed(limit=limit)
        return _json_response({
            "count":    len(completed),
            "sessions": completed,
        })

    async def handle_get_session_metrics(
        self, request: "Request"
    ) -> "Response":
        """
        GET /sessions/{session_id}/metrics
        Get detailed metrics for a specific session.
        """
        session_id = request.match_info["session_id"]
        session    = self._sm.get_session(session_id)

        if session is None:
            return _error_response(
                f"Session not found: {session_id}",
                status=404,
                code="not_found",
            )

        sm = self._metrics.get_session(session_id)
        if sm is None:
            return _error_response(
                "Session metrics not available",
                status=404,
                code="not_found",
            )

        snap = sm.snapshot()
        return _json_response(snap.to_dict())

    # -----------------------------------------------------------------------
    # Server-Sent Events: real-time stats stream
    # -----------------------------------------------------------------------

    async def handle_sse_stats(self, request: "Request") -> "StreamResponse":
        """
        GET /stream/stats
        Real-time stats stream using Server-Sent Events.

        Clients receive a JSON stats update every SSE_STATS_INTERVAL seconds.
        Connection stays open until client disconnects.

        Example client (JavaScript):
          const es = new EventSource('/stream/stats');
          es.onmessage = e => console.log(JSON.parse(e.data));
        """
        response = web.StreamResponse(
            headers={
                "Content-Type":  "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection":    "keep-alive",
                "X-Accel-Buffering": "no",   # disable nginx buffering
            }
        )
        await response.prepare(request)

        # Register subscriber
        async with self._sse_lock:
            self._sse_subscribers.add(response)

        client_ip = _get_client_ip(request)
        logger.info("SSE client connected: %s", client_ip)

        try:
            while True:
                # Check if client is still connected
                if response.task is not None and response.task.done():
                    break

                # Build stats payload
                stats = {
                    "type":         "stats",
                    "timestamp":    time.time(),
                    "sessions": {
                        "active":     self._sm.active_count,
                        "sessions":   [
                            s.to_dict()
                            for s in self._sm.list_sessions()
                        ],
                    },
                    "metrics":      self._metrics.get_summary(),
                    "audio_server": self._audio.get_all_stats()["server"],
                    "esl":          self._esl.connection_info.to_dict(),
                }

                # SSE format: "data: <json>\n\n"
                payload = (
                    f"data: {json.dumps(stats, default=str)}\n\n"
                )
                try:
                    await response.write(payload.encode("utf-8"))
                except Exception:
                    break

                await asyncio.sleep(SSE_STATS_INTERVAL)

            # Send periodic pings
            # (handled by the sleep above — ping is just an empty event)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("SSE error [%s]: %s", client_ip, e)
        finally:
            async with self._sse_lock:
                self._sse_subscribers.discard(response)
            logger.info(
                "SSE client disconnected: %s (remaining=%d)",
                client_ip, len(self._sse_subscribers),
            )

        return response

    async def broadcast_sse_event(
        self,
        event_type: str,
        data:       Dict,
    ) -> None:
        """
        Broadcast an event to all SSE subscribers.
        Called internally when significant events occur.
        """
        payload = (
            f"event: {event_type}\n"
            f"data: {json.dumps(data, default=str)}\n\n"
        )
        encoded = payload.encode("utf-8")

        async with self._sse_lock:
            dead = set()
            for sub in self._sse_subscribers:
                try:
                    await sub.write(encoded)
                except Exception:
                    dead.add(sub)
            for sub in dead:
                self._sse_subscribers.discard(sub)

    # -----------------------------------------------------------------------
    # Internal state
    # -----------------------------------------------------------------------

    @property
    def _start_time(self) -> float:
        """Process start time (used for uptime calculation)."""
        if not hasattr(self, "_process_start_time"):
            self._process_start_time = time.time()
        return self._process_start_time


# ---------------------------------------------------------------------------
# API Server
# ---------------------------------------------------------------------------

class APIServer:
    """
    aiohttp-based REST API server.

    Wires up routes, middleware, and starts the server.
    Designed to run alongside the AudioServer in the same event loop.
    """

    def __init__(
        self,
        handlers:    APIHandlers,
        host:        str   = "0.0.0.0",
        port:        int   = 8080,
        auth_enabled:bool  = False,
        auth_token:  str   = "",
        cors_origins:List[str] = None,
        rate_limit_enabled:    bool = True,
        rate_limit_per_minute: int  = 60,
    ):
        if not _AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for the API server. "
                "Install with: pip install aiohttp"
            )

        self._handlers     = handlers
        self.host          = host
        self.port          = port
        self._auth_enabled = auth_enabled
        self._auth_token   = auth_token
        self._cors_origins = cors_origins or ["*"]
        self._rate_limit_enabled = rate_limit_enabled
        self._rate_limit_per_minute = rate_limit_per_minute

        self._app:    Optional[web.Application] = None
        self._runner: Optional[web.AppRunner]   = None
        self._site:   Optional[web.TCPSite]     = None
        self._rate_limiter = RateLimiter(
            requests_per_minute=rate_limit_per_minute
        )

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

    def _build_app(self) -> web.Application:
        """Build the aiohttp Application with all middleware and routes."""
        middlewares = [
            make_logging_middleware(),
            make_cors_middleware(self._cors_origins),
            make_auth_middleware(self._auth_token, self._auth_enabled),
            make_rate_limit_middleware(
                self._rate_limiter, self._rate_limit_enabled
            ),
        ]

        app = web.Application(middlewares=middlewares)
        self._add_routes(app)
        return app

    def _add_routes(self, app: web.Application) -> None:
      
        """Register all URL routes."""
      
        h = self._handlers

        # ------------------------------------
        #
        # ---- 单次外呼 ------------------------
      
        app.router.add_route("POST",   "/calls",
                             h.handle_place_call)
        app.router.add_route("GET",    "/calls",
                             h.handle_list_calls)
        app.router.add_route("GET",    "/calls/completed",
                             h.handle_completed_calls)
        app.router.add_route("GET",    "/calls/{outbound_id}",
                             h.handle_get_call)
        app.router.add_route("DELETE", "/calls/{outbound_id}",
                             h.handle_cancel_call)
    
        # ---- 批量外呼 Campaign ----
        app.router.add_route("POST",   "/campaigns",
                             h.handle_create_campaign)
        app.router.add_route("GET",    "/campaigns",
                             h.handle_list_campaigns)
        app.router.add_route("GET",    "/campaigns/{campaign_id}",
                             h.handle_get_campaign)
        app.router.add_route("POST",
                             "/campaigns/{campaign_id}/action",
                             h.handle_campaign_action)
        app.router.add_route("GET",
                             "/campaigns/{campaign_id}/calls",
                             h.handle_get_campaign_calls)
    
        # ---- DNC ----
        app.router.add_route("POST",   "/dnc",
                             h.handle_dnc_add)
        app.router.add_route("GET",    "/dnc/{number}",
                             h.handle_dnc_check)
    
        # ---- 外呼统计 ----
        app.router.add_route("GET",    "/outbound/stats",
                             h.handle_outbound_stats)      

      
        app.router.add_route("GET",    "/health",   h.handle_health)
        app.router.add_route("GET",    "/ready",    h.handle_ready)
        app.router.add_route("GET",    "/version",  h.handle_version)
        app.router.add_route("GET",    "/metrics",  h.handle_metrics)
        app.router.add_route("GET",    "/stats",    h.handle_stats)

        # Sessions
        app.router.add_route(
            "GET",    "/sessions",
            h.handle_list_sessions,
        )
        app.router.add_route(
            "GET",    "/sessions/completed",
            h.handle_completed_sessions,
        )
        app.router.add_route(
            "GET",    f"/sessions/{{session_id}}",
            h.handle_get_session,
        )
        app.router.add_route(
            "DELETE", f"/sessions/{{session_id}}",
            h.handle_terminate_session,
        )
        app.router.add_route(
            "GET",    f"/sessions/{{session_id}}/metrics",
            h.handle_get_session_metrics,
        )
        app.router.add_route(
            "POST",   f"/sessions/{{session_id}}/inject",
            h.handle_inject_message,
        )
        app.router.add_route(
            "POST",   f"/sessions/{{session_id}}/transfer",
            h.handle_transfer,
        )
        app.router.add_route(
            "POST",   f"/sessions/{{session_id}}/hold",
            h.handle_hold,
        )
        app.router.add_route(
            "POST",   f"/sessions/{{session_id}}/unhold",
            h.handle_unhold,
        )
        app.router.add_route(
            "POST",   f"/sessions/{{session_id}}/record/start",
            h.handle_record_start,
        )
        app.router.add_route(
            "POST",   f"/sessions/{{session_id}}/record/stop",
            h.handle_record_stop,
        )
        app.router.add_route(
            "POST",   f"/sessions/{{session_id}}/prompt",
            h.handle_update_prompt,
        )

        # SSE
        app.router.add_route(
            "GET",    "/stream/stats",
            h.handle_sse_stats,
        )

        # OPTIONS for CORS preflight on all routes
        app.router.add_route("OPTIONS", "/{path_info:.*}", _handle_options)

        logger.info(
            "API routes registered: %d routes",
            len(app.router.routes()),
        )

    async def start(self) -> None:
        """Start the API server."""
        self._app    = self._build_app()
        self._runner = web.AppRunner(
            self._app,
            access_log=None,   # We handle logging in middleware
        )
        await self._runner.setup()

        self._site = web.TCPSite(
            self._runner,
            self.host,
            self.port,
            reuse_address=True,
            reuse_port=True,
        )
        await self._site.start()

        # Start periodic rate limiter cleanup
        self._cleanup_task = asyncio.ensure_future(
            self._cleanup_loop()
        )

        logger.info(
            "API server listening on http://%s:%d",
            self.host, self.port,
        )
        self._log_routes()

    async def stop(self) -> None:
        """Gracefully stop the API server."""
        logger.info("API server stopping...")

        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

        if self._runner:
            await self._runner.cleanup()

        logger.info("API server stopped")

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of rate limiter state."""
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            try:
                await self._rate_limiter.cleanup()
            except Exception as e:
                logger.error("Rate limiter cleanup error: %s", e)

    def _log_routes(self) -> None:
        """Log all registered routes at startup."""
        if not self._app:
            return
        lines = []
        for resource in self._app.router.resources():
            for route in resource:
                lines.append(
                    f"  {route.method:8s} {resource.canonical}"
                )
        logger.info("Registered API routes:\n%s", "\n".join(lines))


async def _handle_options(request: "Request") -> "Response":
    """Handle OPTIONS preflight for CORS."""
    return web.Response(status=204)


# ---------------------------------------------------------------------------
# Factory from AppConfig
# ---------------------------------------------------------------------------

def create_api_server(
    cfg:             "Any",   # AppConfig
    session_manager: "Any",   # SessionManager
    metrics:         "Any",   # MetricsRegistry
    esl_client:      "Any",   # ESLClient
    audio_server:    "Any",   # AudioServer
) -> APIServer:
    """
    Create the API server from AppConfig.

    Args:
        cfg:             AppConfig instance
        session_manager: SessionManager instance
        metrics:         MetricsRegistry instance
        esl_client:      ESLClient instance
        audio_server:    AudioServer instance

    Returns:
        Configured APIServer (not yet started)
    """
    api_cfg = cfg.api

    handlers = APIHandlers(
        session_manager=session_manager,
        metrics=metrics,
        esl_client=esl_client,
        audio_server=audio_server,
        cfg=cfg,
    )

    return APIServer(
        handlers=handlers,
        host=api_cfg.host,
        port=api_cfg.port,
        auth_enabled=api_cfg.auth_enabled,
        auth_token=api_cfg.auth_token,
        cors_origins=api_cfg.cors_origins,
        rate_limit_enabled=api_cfg.rate_limit_enabled,
        rate_limit_per_minute=api_cfg.rate_limit_per_minute,
    )
