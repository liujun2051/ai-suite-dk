"""
MiniCPM-o 网关 v2.0 入口
核心设计：
  - 一个 asyncio 事件循环（uvicorn 的），通过 asyncio.get_event_loop() 获取引用
  - ESL 事件在独立线程中处理，通过 run_coroutine_threadsafe 提交协程
  - 无 nest_asyncio，无全局裸变量
"""
import asyncio
import logging
import os
import threading
import time
from typing import Dict, List, Optional

import uvicorn

import config
from esl import ESLConnection
from session import CallSession, FIFO_OPEN_TIMEOUT
from api import api_app, init as api_init

logger = logging.getLogger("Main")


# ======================================================================
#  会话管理器
# ======================================================================

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, CallSession] = {}
        self._lock = threading.RLock()

    def create(self, uuid: str, direction: str,
               caller: str, callee: str) -> CallSession:
        with self._lock:
            if uuid in self.sessions:
                return self.sessions[uuid]
            s = CallSession(
                uuid=uuid,
                direction=direction,
                caller_number=caller,
                callee_number=callee
            )
            self.sessions[uuid] = s
            return s

    def get(self, uuid: str) -> Optional[CallSession]:
        with self._lock:
            return self.sessions.get(uuid)

    def remove(self, uuid: str):
        with self._lock:
            self.sessions.pop(uuid, None)

    def list_active(self) -> List[dict]:
        with self._lock:
            result = []
            for s in self.sessions.values():
                result.append({
                    "uuid": s.uuid,
                    "direction": s.direction,
                    "caller": s.caller_number,
                    "callee": s.callee_number,
                    "duration": round(time.time() - s.created_at, 1),
                    "is_ai_speaking": s.is_ai_speaking,
                    "ai_connected": (
                        s.ai_client.connected if s.ai_client else False
                    ),
                    "audio_in_bytes": s.audio_in_bytes,
                    "audio_out_bytes": s.audio_out_bytes,
                })
            return result


# ======================================================================
#  ESL 事件处理
# ======================================================================

# 允许进入 AI 处理的上下文（防止误处理内部通道）
ALLOWED_CONTEXTS = frozenset({"public", "default"})


class GatewayEventHandler:
    """将 ESL 事件路由到对应的 CallSession。"""

    def __init__(self, session_mgr: SessionManager,
                 esl: ESLConnection,
                 loop: asyncio.AbstractEventLoop):
        self._mgr = session_mgr
        self._esl = esl
        self._loop = loop

    def on_channel_create(self, event: dict):
        h = event["headers"]
        call_uuid = h.get("Unique-ID", "")
        direction = h.get("Call-Direction", "unknown")
        caller = h.get("Caller-Caller-ID-Number", "unknown")
        callee = h.get("Caller-Destination-Number", "unknown")
        context = h.get("Caller-Context", "")

        if context not in ALLOWED_CONTEXTS:
            logger.debug(f"Ignoring channel in context '{context}'")
            return

        self._mgr.create(call_uuid, direction, caller, callee)
        logger.info(f"[ESL] CHANNEL_CREATE {call_uuid} "
                    f"({direction}) {caller} → {callee}")

    def on_channel_answer(self, event: dict):
        h = event["headers"]
        call_uuid = h.get("Unique-ID", "")
        session = self._mgr.get(call_uuid)
        if not session:
            return

        session.on_answered()

        # 创建 FIFO
        fifo_path = f"/tmp/minicpm_{call_uuid}.pcma"
        try:
            os.mkfifo(fifo_path)
        except FileExistsError:
            pass
        except OSError as e:
            logger.error(f"mkfifo failed: {e}")
            return

        session._fifo_path = fifo_path

        # 通知 FS 开始录音到 FIFO
        try:
            self._esl.send_bgapi_nowait(
                f"uuid_record {call_uuid} start {fifo_path} 1800"
            )
        except Exception as e:
            logger.error(f"uuid_record failed: {e}")
            return

        # 启动会话（FIFO 读取 + AI 协程 + 播放线程）
        session.start(self._loop, self._esl)
        logger.info(f"[ESL] CHANNEL_ANSWER {call_uuid} - session started")

    def on_channel_hangup(self, event: dict):
        h = event["headers"]
        call_uuid = h.get("Unique-ID", "")
        cause = h.get("Hangup-Cause", "UNKNOWN")
        session = self._mgr.get(call_uuid)
        if session:
            session.hangup(cause)
            self._mgr.remove(call_uuid)
        logger.info(f"[ESL] CHANNEL_HANGUP {call_uuid} cause={cause}")

    def on_dtmf(self, event: dict):
        h = event["headers"]
        call_uuid = h.get("Unique-ID", "")
        digit = h.get("DTMF-Digit", "")
        session = self._mgr.get(call_uuid)
        if session and digit:
            session.on_dtmf(digit)


# ======================================================================
#  主入口
# ======================================================================

async def async_main():
    """
    在 uvicorn 的事件循环中运行。
    获取运行中的 loop 引用，初始化 ESL 和会话管理器。
    """
    loop = asyncio.get_running_loop()

    # 初始化 ESL
    esl = ESLConnection(
        config.FS_HOST,
        config.FS_ESL_PORT,
        config.FS_ESL_PASS
    )
    if not esl.connect():
        logger.critical("Cannot connect to FreeSWITCH ESL, exiting.")
        raise SystemExit(1)

    # 会话管理器
    mgr = SessionManager()

    # 事件处理器
    handler = GatewayEventHandler(mgr, esl, loop)

    # 注册事件回调
    esl.on("CHANNEL_CREATE", handler.on_channel_create)
    esl.on("CHANNEL_ANSWER", handler.on_channel_answer)
    esl.on("CHANNEL_HANGUP", handler.on_channel_hangup)
    esl.on("DTMF", handler.on_dtmf)

    # 订阅事件（必须在 start() 之前，否则事件循环未启动）
    esl.subscribe(
        "CHANNEL_CREATE",
        "CHANNEL_ANSWER",
        "CHANNEL_HANGUP",
        "DTMF",
        "BACKGROUND_JOB",   # bgapi 结果路由需要
    )

    # 启动 ESL 后台读取线程
    esl.start()

    # 注入 API 依赖
    api_init(mgr, esl)

    logger.info("=" * 50)
    logger.info("MiniCPM-o Gateway v2.0 ready")
    logger.info(f"AI mode : {config.AI_MODE}")
    logger.info(f"ESL     : {config.FS_HOST}:{config.FS_ESL_PORT}")
    logger.info(f"Listen  : 0.0.0.0:8080")
    logger.info("=" * 50)


def main():
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 使用 uvicorn 的 Server API，在其事件循环启动后再初始化业务逻辑
    uv_config = uvicorn.Config(
        api_app,
        host="0.0.0.0",
        port=8080,
        log_level="warning",
        # 不指定 loop，让 uvicorn 自己创建并管理
    )
    server = uvicorn.Server(uv_config)

    # 在 uvicorn 启动时执行初始化
    original_startup = server.startup

    async def patched_startup():
        await original_startup()
        await async_main()          # 在同一个事件循环中初始化

    server.startup = patched_startup

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
