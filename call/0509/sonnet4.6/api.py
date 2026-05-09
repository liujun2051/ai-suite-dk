"""
FastAPI 路由模块
"""
import logging
import uuid as uuid_mod

from fastapi import FastAPI, HTTPException

import config

logger = logging.getLogger("API")

api_app = FastAPI(title="MiniCPM-o Gateway v2.0")

# 由 main.py 注入
_session_mgr = None
_esl_conn = None


def init(session_mgr, esl_conn):
    global _session_mgr, _esl_conn
    _session_mgr = session_mgr
    _esl_conn = esl_conn


@api_app.get("/")
async def root():
    return {
        "status": "running",
        "version": "2.0",
        "ai_mode": config.AI_MODE
    }


@api_app.get("/health")
async def health():
    return {
        "esl_connected": _esl_conn.connected if _esl_conn else False,
        "active_sessions": len(_session_mgr.sessions) if _session_mgr else 0,
        "ai_mode": config.AI_MODE,
    }


@api_app.get("/sessions")
async def list_sessions():
    if not _session_mgr:
        return {"count": 0, "sessions": []}
    return {
        "count": len(_session_mgr.sessions),
        "sessions": _session_mgr.list_active()
    }


@api_app.post("/call")
async def make_call(request: dict):
    phone = request.get("phone_number", "").strip()
    if not phone:
        raise HTTPException(status_code=400, detail="phone_number required")

    gateway = request.get("gateway", config.FXO_GATEWAY)
    caller_id = request.get("caller_id", config.OUTBOUND_CID)
    call_uuid = str(uuid_mod.uuid4())

    originate_str = (
        f"originate {{"
        f"origination_uuid={call_uuid},"
        f"origination_caller_id_number={caller_id},"
        f"origination_caller_id_name=AI,"
        f"ignore_early_media=true"
        f"}} "
        f"sofia/gateway/{gateway}/{phone} &park"
    )

    try:
        result = _esl_conn.send_command(originate_str)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ESL error: {e}")

    return {
        "status": "initiated",
        "call_uuid": call_uuid,
        "phone_number": phone,
        "fs_result": result
    }


@api_app.post("/call/{call_uuid}/hangup")
async def hangup_call(call_uuid: str):
    try:
        _esl_conn.send_bgapi_nowait(f"uuid_kill {call_uuid}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ESL error: {e}")
    return {"status": "hangup_sent"}


@api_app.post("/call/{call_uuid}/barge_in")
async def barge_in(call_uuid: str):
    session = _session_mgr.get(call_uuid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.barge_in()
    return {"status": "barge_in_triggered"}


@api_app.post("/call/{call_uuid}/say")
async def say_text(call_uuid: str, request: dict):
    """向指定通话注入文本，让 AI 说出来。"""
    session = _session_mgr.get(call_uuid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    text = request.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    if session.ai_client and session._loop:
        import asyncio
        asyncio.run_coroutine_threadsafe(
            session.ai_client.send_text(text),
            session._loop
        )
    return {"status": "sent"}
