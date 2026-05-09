好的，生成 `bootstrap.py`，运行后自动创建完整项目。

```python
#!/usr/bin/env python3
"""
MiniCPM-o Gateway v2.1 项目脚手架
运行方式：python bootstrap.py
生成目录：./minicpm_gateway/
"""

import os

# ── 项目根目录 ──────────────────────────────────────────────────────────
ROOT = "minicpm_gateway"

# ======================================================================
#  文件内容定义
# ======================================================================

FILES = {}

# ── config.py ──────────────────────────────────────────────────────────
FILES["config.py"] = '''
# ── FreeSWITCH ESL ──────────────────────────────────────────────────────
FS_HOST     = "127.0.0.1"
FS_ESL_PORT = 8021
FS_ESL_PASS = "ClueCon"

# ── AI 后端 ─────────────────────────────────────────────────────────────
AI_MODE        = "local"          # "local" | "cloud"
MINICPM_WS_URL = "ws://127.0.0.1:9000/ws"
CLOUD_API_KEY  = ""
AI_SAMPLE_RATE = 16_000
AI_SYSTEM_PROMPT = "你是一个电话客服助手，说话简洁自然。"
AI_VOICE       = "alloy"

# ── 外呼 ────────────────────────────────────────────────────────────────
FXO_GATEWAY  = "pstn"
OUTBOUND_CID = "10000"

# ── 服务 ────────────────────────────────────────────────────────────────
API_PORT  = 8080
LOG_LEVEL = "INFO"
'''.lstrip()

# ── audio_utils.py（桩文件，替换为真实实现）────────────────────────────
FILES["audio_utils.py"] = '''
"""
音频工具桩文件。
请替换为真实的 G.711a 编解码和重采样实现。
"""
import numpy as np


def pcma_to_pcm(pcma: np.ndarray) -> np.ndarray:
    """G.711a → PCM16（线性）。"""
    # TODO: 替换为真实 G.711a 解码
    return pcma.astype(np.int16)


def pcm_to_pcma(pcm: np.ndarray) -> bytes:
    """PCM16（线性）→ G.711a bytes。"""
    # TODO: 替换为真实 G.711a 编码
    return pcm.astype(np.uint8).tobytes()


def RESAMPLE_FUNC(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """简单线性重采样（生产环境请替换为 librosa / soxr）。"""
    if src_rate == dst_rate:
        return data
    ratio      = dst_rate / src_rate
    new_length = int(len(data) * ratio)
    indices    = np.linspace(0, len(data) - 1, new_length)
    return np.interp(indices, np.arange(len(data)), data).astype(np.int16)
'''.lstrip()

# ── audio.py ───────────────────────────────────────────────────────────
FILES["audio.py"] = '''
"""
音频转换与帧缓冲模块 v2.1

常量：
  FS_RATE      = 8 000 Hz   G.711a（FreeSWITCH）
  AI_RATE      = 16 000 Hz  PCM16LE（AI 模型）
  FRAME_MS     = 20 ms      标准帧时长
  FRAME_PCMA   = 160 B      20ms G.711a@8kHz（1 byte/sample）
"""

import logging
from typing import List

import numpy as np

from audio_utils import pcma_to_pcm, pcm_to_pcma, RESAMPLE_FUNC

logger = logging.getLogger("Audio")

FS_RATE      = 8_000
AI_RATE      = 16_000
FRAME_MS     = 20
FRAME_PCMA   = FS_RATE  * FRAME_MS // 1000 * 1   # 160
FRAME_PCM8K  = FS_RATE  * FRAME_MS // 1000 * 2   # 320
FRAME_PCM16K = AI_RATE  * FRAME_MS // 1000 * 2   # 640


class AudioConverter:
    """
    FreeSWITCH <-> AI 音频格式转换。
    无状态，线程安全，可在任意线程调用。
    """

    @staticmethod
    def fs_to_ai(pcma_bytes: bytes) -> bytes:
        """G.711a 8kHz -> PCM16LE 16kHz。"""
        samples_u8 = np.frombuffer(pcma_bytes, dtype=np.uint8)
        pcm_8k     = pcma_to_pcm(samples_u8)
        pcm_16k    = RESAMPLE_FUNC(pcm_8k, FS_RATE, AI_RATE)
        return pcm_16k.astype(np.int16).tobytes()

    @staticmethod
    def ai_to_fs(pcm16_bytes: bytes) -> bytes:
        """PCM16LE 16kHz -> G.711a 8kHz。"""
        pcm_16k = np.frombuffer(pcm16_bytes, dtype=np.int16)
        pcm_8k  = RESAMPLE_FUNC(pcm_16k, AI_RATE, FS_RATE).astype(np.int16)
        return pcm_to_pcma(pcm_8k)


class AudioFrameBuffer:
    """
    将 AI 返回的变长 PCM16LE 音频块切割为标准 20ms PCMA 帧。
    每个 session 独立实例，非线程安全。
    """

    def __init__(self):
        self._raw = bytearray()

    def push_pcm16(self, pcm16_bytes: bytes):
        """接收 PCM16LE，转换为 PCMA 后追加到缓冲区。"""
        if not pcm16_bytes:
            return
        pcma = AudioConverter.ai_to_fs(pcm16_bytes)
        self._raw.extend(pcma)

    def pop_frames(self) -> List[bytes]:
        """弹出所有完整的 20ms PCMA 帧（160 字节），剩余留在缓冲区。"""
        frames = []
        while len(self._raw) >= FRAME_PCMA:
            frames.append(bytes(self._raw[:FRAME_PCMA]))
            del self._raw[:FRAME_PCMA]
        return frames

    def flush(self) -> bytes:
        """取出全部剩余数据并清空缓冲区（turn_done 时调用）。"""
        remaining = bytes(self._raw)
        self._raw.clear()
        return remaining

    def clear(self):
        """清空缓冲区（barge-in 时调用）。"""
        self._raw.clear()

    def __len__(self) -> int:
        return len(self._raw)
'''.lstrip()

# ── esl.py ─────────────────────────────────────────────────────────────
FILES["esl.py"] = '''
"""
ESL 连接模块 v2.1

设计原则：
  - _event_loop 线程是唯一读取 socket 的地方，彻底消除竞态
  - send_command / send_bgapi 只写 socket，通过 Queue 等待响应
  - BACKGROUND_JOB 正确解析内层 body
  - 事件回调在独立线程池执行，不阻塞读取循环
  - subscribe 通过专用 Queue 等待 command/reply 确认
"""

import logging
import queue
import socket
import threading
import time
import uuid as uuid_mod
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional
from urllib.parse import unquote

logger = logging.getLogger("ESL")


class ESLError(Exception):
    pass


class ESLConnection:
    """
    线程安全的 FreeSWITCH ESL 连接。

    消息流：
        写：send_command / send_bgapi -> _send_lock -> socket.sendall
        读：_event_loop（唯一）-> Content-Type 路由 -> 各自 Queue / 回调
    """

    CMD_TIMEOUT   = 10
    BGAPI_TIMEOUT = 30
    SUB_TIMEOUT   = 5
    RECV_BUF_SIZE = 65536

    def __init__(self, host: str, port: int, password: str):
        self.host     = host
        self.port     = port
        self.password = password

        self._sock: Optional[socket.socket] = None
        self._buf        = b""
        self._connected  = False
        self._running    = False

        self._send_lock     = threading.Lock()
        self._api_resp_q: queue.Queue    = queue.Queue()
        self._cmd_reply_q: queue.Queue   = queue.Queue()
        self._bgapi_pending: Dict[str, queue.Queue] = {}
        self._bgapi_lock    = threading.Lock()
        self._callbacks: Dict[str, List[Callable]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=8,
            thread_name_prefix="esl-cb",
        )
        self._reader_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    #  公开接口
    # ------------------------------------------------------------------ #

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """建立 TCP 连接并完成 ESL auth 握手。"""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(10)
            self._sock.connect((self.host, self.port))
            self._sock.settimeout(None)
            self._buf = b""

            headers = self._read_headers()
            ct = headers.get("Content-Type", "")
            if ct != "auth/request":
                raise ESLError(f"Unexpected greeting: {ct!r}")

            self._raw_send(f"auth {self.password}\\n\\n")

            headers = self._read_headers()
            reply   = headers.get("Reply-Text", "")
            if not reply.startswith("+OK"):
                raise ESLError(f"Auth failed: {reply!r}")

            self._connected = True
            logger.info(f"ESL connected -> {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"ESL connect failed: {e}")
            try:
                self._sock.close()
            except Exception:
                pass
            return False

    def start(self):
        """启动后台事件循环线程。必须在 connect() 成功后调用。"""
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._event_loop,
            name="esl-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def stop(self):
        """优雅关闭。"""
        self._running   = False
        self._connected = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._executor.shutdown(wait=False)

    def subscribe(self, *events: str):
        """订阅 ESL 事件，等待 command/reply 确认后返回。"""
        event_str = " ".join(events)
        with self._send_lock:
            self._raw_send(f"event plain {event_str}\\n\\n")
        try:
            reply = self._cmd_reply_q.get(timeout=self.SUB_TIMEOUT)
            if "+OK" not in reply:
                logger.warning(f"subscribe reply: {reply!r}")
        except queue.Empty:
            logger.warning("subscribe: no reply within timeout")

    def on(self, event_name: str, callback: Callable):
        """注册事件回调，线程安全。"""
        self._callbacks.setdefault(event_name, []).append(callback)

    def send_command(self, cmd: str) -> str:
        """发送同步 api 命令，阻塞等待响应。线程安全。"""
        with self._send_lock:
            self._raw_send(f"api {cmd}\\n\\n")
        try:
            return self._api_resp_q.get(timeout=self.CMD_TIMEOUT)
        except queue.Empty:
            raise ESLError(f"api timeout: {cmd!r}")

    def send_bgapi(self, cmd: str) -> str:
        """发送 bgapi 命令，阻塞等待 BACKGROUND_JOB 结果。线程安全。"""
        job_uuid = str(uuid_mod.uuid4())
        resp_q: queue.Queue = queue.Queue()
        with self._bgapi_lock:
            self._bgapi_pending[job_uuid] = resp_q
        try:
            with self._send_lock:
                self._raw_send(f"bgapi {cmd}\\nJob-UUID: {job_uuid}\\n\\n")
            return resp_q.get(timeout=self.BGAPI_TIMEOUT)
        except queue.Empty:
            raise ESLError(f"bgapi timeout: {cmd!r}")
        finally:
            with self._bgapi_lock:
                self._bgapi_pending.pop(job_uuid, None)

    def send_bgapi_nowait(self, cmd: str):
        """发送 bgapi，不等待结果（fire-and-forget）。线程安全。"""
        job_uuid = str(uuid_mod.uuid4())
        with self._send_lock:
            self._raw_send(f"bgapi {cmd}\\nJob-UUID: {job_uuid}\\n\\n")

    # ------------------------------------------------------------------ #
    #  内部：socket 读写原语
    # ------------------------------------------------------------------ #

    def _raw_send(self, data: str):
        self._sock.sendall(data.encode("utf-8"))

    def _read_exactly(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._sock.recv(self.RECV_BUF_SIZE)
            if not chunk:
                raise ConnectionError("ESL socket closed")
            self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n:]
        return data

    def _read_line(self) -> str:
        while b"\\n" not in self._buf:
            chunk = self._sock.recv(self.RECV_BUF_SIZE)
            if not chunk:
                raise ConnectionError("ESL socket closed")
            self._buf += chunk
        idx         = self._buf.index(b"\\n")
        line, self._buf = self._buf[:idx], self._buf[idx + 1:]
        return line.decode("utf-8", errors="replace").strip()

    def _read_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        while True:
            line = self._read_line()
            if line == "":
                break
            if ": " in line:
                k, v = line.split(": ", 1)
                headers[k.strip()] = unquote(v.strip())
        return headers

    def _read_body(self, length: int) -> str:
        if length <= 0:
            return ""
        return self._read_exactly(length).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    #  内部：事件循环
    # ------------------------------------------------------------------ #

    def _event_loop(self):
        logger.info("ESL event loop started")
        while self._running:
            try:
                headers = self._read_headers()
                if not headers:
                    continue
                ct   = headers.get("Content-Type", "")
                cl   = int(headers.get("Content-Length", 0))
                body = self._read_body(cl)

                if ct == "api/response":
                    self._api_resp_q.put(body.strip())

                elif ct == "command/reply":
                    reply = headers.get("Reply-Text", body.strip())
                    self._cmd_reply_q.put(reply)

                elif ct == "text/event-plain":
                    self._dispatch_event(headers, body)

                elif ct == "text/disconnect-notice":
                    logger.warning("ESL disconnect notice")
                    self._connected = False
                    break

                elif ct == "auth/request":
                    pass

                else:
                    if ct:
                        logger.debug(f"ESL unhandled Content-Type: {ct!r}")

            except ConnectionError as e:
                logger.error(f"ESL connection lost: {e}")
                break
            except Exception as e:
                if self._running:
                    logger.exception(f"ESL event loop error: {e}")
                    time.sleep(0.05)

        self._connected = False
        logger.warning("ESL event loop exited")

    def _dispatch_event(self, outer_headers: Dict[str, str], body: str):
        """解析 text/event-plain，路由到回调或 bgapi 等待队列。"""
        event_headers: Dict[str, str] = {}
        for line in body.splitlines():
            if line == "":
                break
            if ": " in line:
                k, v = line.split(": ", 1)
                event_headers[k.strip()] = unquote(v.strip())

        # 解析内层 body（BACKGROUND_JOB 结果在此）
        inner_cl   = int(event_headers.get("Content-Length", 0))
        inner_body = ""
        if inner_cl > 0:
            split = body.find("\\n\\n")
            if split != -1:
                inner_body = body[split + 2: split + 2 + inner_cl]

        event_name = event_headers.get("Event-Name", "")

        if event_name == "BACKGROUND_JOB":
            job_uuid = event_headers.get("Job-UUID", "")
            with self._bgapi_lock:
                resp_q = self._bgapi_pending.get(job_uuid)
            if resp_q is not None:
                resp_q.put(inner_body.strip())
            return

        callbacks = self._callbacks.get(event_name, [])
        if callbacks:
            event = {"headers": event_headers, "body": inner_body}
            for cb in callbacks:
                self._executor.submit(self._safe_call, cb, event)

    @staticmethod
    def _safe_call(cb: Callable, event: dict):
        try:
            cb(event)
        except Exception as e:
            logger.exception(f"ESL callback error in {cb.__name__}: {e}")
'''.lstrip()

# ── ai_client.py ───────────────────────────────────────────────────────
FILES["ai_client.py"] = '''
"""
MiniCPM-o WebSocket 客户端 v2.1

设计：
  - 纯 async，运行在主事件循环
  - 回调在事件循环线程同步调用（应快速返回）
  - 所有发送操作加超时
  - 断线通过 on_error 通知外部
"""

import asyncio
import base64
import json
import logging
from typing import Callable, Optional

import websockets
import websockets.exceptions

import config

logger = logging.getLogger("AIClient")

_SEND_TIMEOUT = 5.0


class AIClient:
    """
    封装与 MiniCPM-o 的 WebSocket 会话。

    回调（均在事件循环线程调用，应快速返回）：
        on_audio_chunk(pcm16_bytes: bytes)
        on_turn_done()
        on_speech_started()
        on_transcript(text: str)
        on_error(reason: str)
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.mode: str  = config.AI_MODE

        self._ws                 = None
        self._connected          = False
        self._receive_task: Optional[asyncio.Task] = None

        self.on_audio_chunk:    Optional[Callable[[bytes], None]] = None
        self.on_turn_done:      Optional[Callable[[], None]]      = None
        self.on_speech_started: Optional[Callable[[], None]]      = None
        self.on_transcript:     Optional[Callable[[str], None]]   = None
        self.on_error:          Optional[Callable[[str], None]]   = None

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        extra_headers = {}
        if self.mode == "cloud":
            extra_headers["Authorization"] = f"Bearer {config.CLOUD_API_KEY}"
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    config.MINICPM_WS_URL,
                    extra_headers=extra_headers,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=16 * 1024 * 1024,
                ),
                timeout=10.0,
            )
            self._connected = True
            self._receive_task = asyncio.create_task(
                self._receive_loop(),
                name=f"ai-recv-{self.session_id[:8]}",
            )
            logger.info(f"[{self.session_id}] AI connected (mode={self.mode})")
            return True
        except Exception as e:
            logger.error(f"[{self.session_id}] AI connect failed: {e}")
            return False

    async def initialize(self):
        if self.mode == "cloud":
            await self._init_cloud()
        else:
            await self._init_local()

    async def close(self):
        self._connected = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info(f"[{self.session_id}] AI connection closed")

    async def send_audio(self, pcm16_bytes: bytes):
        if not self._connected or not pcm16_bytes:
            return
        try:
            if self.mode == "cloud":
                payload = json.dumps({
                    "type":  "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm16_bytes).decode(),
                })
                await asyncio.wait_for(
                    self._ws.send(payload), timeout=_SEND_TIMEOUT
                )
            else:
                await asyncio.wait_for(
                    self._ws.send(pcm16_bytes), timeout=_SEND_TIMEOUT
                )
        except asyncio.TimeoutError:
            logger.warning(f"[{self.session_id}] send_audio timeout")
        except websockets.exceptions.ConnectionClosed:
            self._connected = False
        except Exception as e:
            logger.error(f"[{self.session_id}] send_audio error: {e}")
            self._connected = False

    async def send_text(self, text: str):
        if not self._connected:
            return
        try:
            if self.mode == "cloud":
                await self._ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }))
                await self._ws.send(json.dumps({"type": "response.create"}))
            else:
                await self._ws.send(json.dumps({
                    "type": "text_input",
                    "text": text,
                }))
        except Exception as e:
            logger.error(f"[{self.session_id}] send_text error: {e}")

    async def interrupt(self):
        if not self._connected:
            return
        try:
            if self.mode == "cloud":
                await self._ws.send(
                    json.dumps({"type": "input_audio_buffer.clear"})
                )
            else:
                await self._ws.send(json.dumps({"type": "barge_in"}))
        except Exception as e:
            logger.error(f"[{self.session_id}] interrupt error: {e}")

    async def _init_cloud(self):
        async for raw in self._ws:
            data = json.loads(raw)
            if data.get("type") == "session.created":
                break
        await self._ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions":        config.AI_SYSTEM_PROMPT,
                "voice":               config.AI_VOICE,
                "input_audio_format":  "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type":                "server_vad",
                    "threshold":           0.5,
                    "silence_duration_ms": 800,
                },
                "input_audio_transcription": {
                    "model":    "whisper-1",
                    "language": "zh",
                },
            },
        }))
        async for raw in self._ws:
            data = json.loads(raw)
            if data.get("type") == "session.updated":
                break

    async def _init_local(self):
        await self._ws.send(json.dumps({
            "type":        "init",
            "session_id":  self.session_id,
            "sample_rate": config.AI_SAMPLE_RATE,
            "channels":    1,
            "format":      "pcm_s16le",
        }))

    async def _receive_loop(self):
        try:
            async for msg in self._ws:
                if not self._connected:
                    break
                if isinstance(msg, bytes):
                    if self.on_audio_chunk:
                        self.on_audio_chunk(msg)
                else:
                    self._handle_json(msg)
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"[{self.session_id}] AI WS closed: {e.code} {e.reason}")
        except Exception as e:
            logger.exception(f"[{self.session_id}] receive loop error: {e}")
        finally:
            self._connected = False
            if self.on_error:
                self.on_error("connection_lost")

    def _handle_json(self, raw: str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"[{self.session_id}] invalid JSON: {raw[:80]}")
            return

        t = data.get("type", "")

        if t == "response.output_audio.delta":
            if self.on_audio_chunk:
                chunk = base64.b64decode(data.get("delta", ""))
                if chunk:
                    self.on_audio_chunk(chunk)

        elif t == "response.output_audio.done":
            if self.on_turn_done:
                self.on_turn_done()

        elif t == "input_audio_buffer.speech_started":
            if self.on_speech_started:
                self.on_speech_started()

        elif t == "conversation.item.input_audio_transcription.completed":
            text = data.get("transcript", "").strip()
            if text and self.on_transcript:
                self.on_transcript(text)

        elif t == "error":
            err = data.get("error", {})
            logger.error(f"[{self.session_id}] AI protocol error: {err}")
            if self.on_error:
                self.on_error(str(err))
'''.lstrip()

# ── session.py ─────────────────────────────────────────────────────────
FILES["session.py"] = '''
"""
通话会话模块 v2.1

修复清单：
  [1]  dataclass 全部字段加类型注解，消除类变量共享
  [2]  跨线程协程调用统一用 run_coroutine_threadsafe
  [3]  FIFO 非阻塞打开 + select 超时
  [4]  协程中阻塞 queue.get 改为 run_in_executor
  [5]  音频帧对齐由 AudioFrameBuffer 负责
  [6]  批量累积 500ms 音频后一次 uuid_broadcast
  [7]  hangup 幂等（_hangup_lock）
  [8]  barge-in 状态门控（_barge_lock + is_ai_speaking）
  [9]  _on_ai_turn_done 通过 AudioFrameBuffer.flush() 取剩余数据
"""

import asyncio
import logging
import os
import queue
import select
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from audio import AudioConverter, AudioFrameBuffer, FRAME_PCMA
from ai_client import AIClient

logger = logging.getLogger("Session")

FLUSH_MIN_MS    = 500
FLUSH_MIN_BYTES = 8_000 * FLUSH_MIN_MS // 1000   # 4 000 bytes
FLUSH_MAX_SEC   = 0.8
FIFO_OPEN_TIMEOUT_SEC = 15.0
KEEP_AUDIO_FILES      = 20


@dataclass
class CallSession:
    """
    单路通话的完整状态。
    所有字段必须有类型注解，确保 dataclass 正确生成实例变量。
    """

    # 构造时传入
    uuid:          str
    direction:     str
    caller_number: str
    callee_number: str

    # 时间戳
    created_at: float = field(default_factory=time.time)

    # 音频队列
    fs_to_ai_queue: queue.Queue = field(
        default_factory=lambda: queue.Queue(maxsize=500)
    )
    ai_to_fs_queue: queue.Queue = field(
        default_factory=lambda: queue.Queue(maxsize=500)
    )

    # 状态
    is_active:      bool = field(default=True)
    is_answered:    bool = field(default=False)
    is_ai_speaking: bool = field(default=False)

    # 统计
    audio_in_bytes:  int = field(default=0)
    audio_out_bytes: int = field(default=0)

    # 内部对象（init=False）
    ai_client:    Optional[AIClient]                  = field(default=None,                      init=False)
    _loop:        Optional[asyncio.AbstractEventLoop] = field(default=None,                      init=False)
    _esl_ref:     Any                                 = field(default=None,                      init=False)
    _threads:     list                                = field(default_factory=list,               init=False)
    _fifo_path:   str                                 = field(default="",                        init=False)
    _tmp_dir:     str                                 = field(default="",                        init=False)
    _frame_buf:   AudioFrameBuffer                    = field(default_factory=AudioFrameBuffer,  init=False)
    _hangup_lock: threading.Lock                      = field(default_factory=threading.Lock,    init=False)
    _barge_lock:  threading.Lock                      = field(default_factory=threading.Lock,    init=False)

    # ================================================================== #
    #  生命周期
    # ================================================================== #

    def start(self, loop: asyncio.AbstractEventLoop, esl_conn: Any):
        import weakref
        self._loop    = loop
        self._esl_ref = weakref.ref(esl_conn)
        self._tmp_dir = tempfile.mkdtemp(prefix=f"ai_{self.uuid[:8]}_")

        t_read = threading.Thread(
            target=self._fifo_read_loop,
            name=f"fifo-{self.uuid[:8]}",
            daemon=True,
        )
        t_read.start()
        self._threads.append(t_read)

        t_play = threading.Thread(
            target=self._play_loop,
            name=f"play-{self.uuid[:8]}",
            daemon=True,
        )
        t_play.start()
        self._threads.append(t_play)

        asyncio.run_coroutine_threadsafe(self._ai_main(), self._loop)

    def hangup(self, reason: str = "normal"):
        with self._hangup_lock:
            if not self.is_active:
                return
            self.is_active = False

        logger.info(f"[{self.uuid}] Hangup: {reason}")

        if self.ai_client and self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.ai_client.close(), self._loop
            )

        for t in self._threads:
            t.join(timeout=3)

        if self._fifo_path and os.path.exists(self._fifo_path):
            try:
                os.remove(self._fifo_path)
            except OSError:
                pass

        if self._tmp_dir and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

        logger.info(
            f"[{self.uuid}] Cleanup done | "
            f"in={self.audio_in_bytes}B out={self.audio_out_bytes}B"
        )

    # ================================================================== #
    #  ESL 事件入口
    # ================================================================== #

    def on_answered(self):
        self.is_answered = True

    def on_dtmf(self, digit: str):
        logger.info(f"[{self.uuid}] DTMF: {digit!r}")
        if digit in ("*", "#"):
            self.barge_in()
        elif self.ai_client and self._loop:
            asyncio.run_coroutine_threadsafe(
                self.ai_client.send_text(f"用户按键：{digit}"),
                self._loop,
            )

    def barge_in(self):
        with self._barge_lock:
            if not self.is_ai_speaking:
                return
            self.is_ai_speaking = False

        logger.info(f"[{self.uuid}] Barge-in")

        esl = self._esl_ref() if self._esl_ref else None
        if esl:
            try:
                esl.send_bgapi_nowait(f"uuid_break {self.uuid} all")
            except Exception as e:
                logger.warning(f"[{self.uuid}] uuid_break failed: {e}")

        if self.ai_client and self._loop:
            asyncio.run_coroutine_threadsafe(
                self.ai_client.interrupt(), self._loop
            )

        self._drain_audio()

    # ================================================================== #
    #  线程 A：FIFO 读取
    # ================================================================== #

    def _fifo_read_loop(self):
        fd = -1
        try:
            fd = os.open(self._fifo_path, os.O_RDONLY | os.O_NONBLOCK)
            ready, _, _ = select.select([fd], [], [], FIFO_OPEN_TIMEOUT_SEC)
            if not ready:
                logger.error(f"[{self.uuid}] FIFO not ready after {FIFO_OPEN_TIMEOUT_SEC}s")
                self.hangup("fifo_timeout")
                return

            os.set_blocking(fd, True)
            logger.info(f"[{self.uuid}] FIFO read started")

            while self.is_active:
                try:
                    data = os.read(fd, FRAME_PCMA)
                except BlockingIOError:
                    time.sleep(0.005)
                    continue
                if not data:
                    logger.info(f"[{self.uuid}] FIFO EOF")
                    break

                self.audio_in_bytes += len(data)
                try:
                    self.fs_to_ai_queue.put_nowait(data)
                except queue.Full:
                    try:
                        self.fs_to_ai_queue.get_nowait()
                        self.fs_to_ai_queue.put_nowait(data)
                    except queue.Empty:
                        pass

        except Exception as e:
            if self.is_active:
                logger.error(f"[{self.uuid}] FIFO error: {e}")
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            logger.info(f"[{self.uuid}] FIFO read loop exited")

    # ================================================================== #
    #  协程：AI 主逻辑
    # ================================================================== #

    async def _ai_main(self):
        client = AIClient(self.uuid)
        client.on_audio_chunk    = self._on_ai_audio
        client.on_turn_done      = self._on_ai_turn_done
        client.on_speech_started = self._on_speech_started
        client.on_transcript     = self._on_transcript
        client.on_error          = self._on_ai_error
        self.ai_client = client

        if not await client.connect():
            self.hangup("ai_connect_failed")
            return

        await client.initialize()

        if self.direction == "outbound":
            await client.send_text("你好，我是智能客服，请问有什么可以帮您？")

        await self._send_to_ai_loop()

    async def _send_to_ai_loop(self):
        loop      = asyncio.get_running_loop()
        converter = AudioConverter()

        while self.is_active and self.ai_client and self.ai_client.connected:
            try:
                pcma = await loop.run_in_executor(
                    None,
                    lambda: self.fs_to_ai_queue.get(timeout=0.05),
                )
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[{self.uuid}] queue read error: {e}")
                break

            try:
                pcm16 = converter.fs_to_ai(pcma)
                await self.ai_client.send_audio(pcm16)
            except Exception as e:
                logger.error(f"[{self.uuid}] AI send error: {e}")
                break

    # ================================================================== #
    #  AI 回调
    # ================================================================== #

    def _on_ai_audio(self, pcm16_bytes: bytes):
        self.is_ai_speaking = True
        self._frame_buf.push_pcm16(pcm16_bytes)
        for frame in self._frame_buf.pop_frames():
            self.audio_out_bytes += len(frame)
            try:
                self.ai_to_fs_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.ai_to_fs_queue.get_nowait()
                    self.ai_to_fs_queue.put_nowait(frame)
                except queue.Empty:
                    pass

    def _on_ai_turn_done(self):
        tail = self._frame_buf.flush()
        if tail:
            try:
                self.ai_to_fs_queue.put_nowait(tail)
            except queue.Full:
                pass
        logger.debug(f"[{self.uuid}] AI turn done")

    def _on_speech_started(self):
        if self.is_ai_speaking:
            self.barge_in()

    def _on_transcript(self, text: str):
        logger.info(f"[{self.uuid}] User: {text}")

    def _on_ai_error(self, reason: str):
        logger.error(f"[{self.uuid}] AI error: {reason!r}")
        if self.is_active:
            self.hangup(f"ai_error:{reason}")

    # ================================================================== #
    #  线程 B：播放循环
    # ================================================================== #

    def _play_loop(self):
        buf        = bytearray()
        last_data  = time.monotonic()
        last_flush = time.monotonic()

        while self.is_active:
            try:
                frame = self.ai_to_fs_queue.get(timeout=0.02)
                buf.extend(frame)
                last_data = time.monotonic()
            except queue.Empty:
                pass

            now         = time.monotonic()
            queue_empty = self.ai_to_fs_queue.empty()
            should_flush = (
                len(buf) >= FLUSH_MIN_BYTES
                or (buf and queue_empty and now - last_data  >= 0.1)
                or (buf and now - last_flush >= FLUSH_MAX_SEC)
            )

            if should_flush and buf:
                self._flush_to_fs(bytes(buf))
                buf.clear()
                last_flush = now
                if queue_empty:
                    self.is_ai_speaking = False

        if buf:
            self._flush_to_fs(bytes(buf))
        logger.info(f"[{self.uuid}] Play loop exited")

    def _flush_to_fs(self, pcma_data: bytes):
        esl = self._esl_ref() if self._esl_ref else None
        if not esl or not self._tmp_dir:
            return

        fname = os.path.join(
            self._tmp_dir,
            f"{int(time.monotonic() * 1_000_000)}.pcma",
        )
        try:
            with open(fname, "wb") as f:
                f.write(pcma_data)
        except OSError as e:
            logger.error(f"[{self.uuid}] write audio failed: {e}")
            return

        try:
            esl.send_bgapi_nowait(
                f"uuid_broadcast {self.uuid} {fname} aleg"
            )
        except Exception as e:
            logger.warning(f"[{self.uuid}] uuid_broadcast failed: {e}")

        self._cleanup_old_files()

    def _cleanup_old_files(self):
        if not self._tmp_dir or not os.path.exists(self._tmp_dir):
            return
        try:
            files = sorted(
                (f for f in os.listdir(self._tmp_dir) if f.endswith(".pcma")),
                key=lambda f: os.path.getmtime(os.path.join(self._tmp_dir, f)),
            )
            for old in files[:-KEEP_AUDIO_FILES]:
                try:
                    os.remove(os.path.join(self._tmp_dir, old))
                except OSError:
                    pass
        except Exception:
            pass

    def _drain_audio(self):
        for q in (self.ai_to_fs_queue, self.fs_to_ai_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        self._frame_buf.clear()
'''.lstrip()

# ── api.py ─────────────────────────────────────────────────────────────
FILES["api.py"] = '''
"""
FastAPI HTTP 路由 v2.1
"""

import asyncio
import logging
import time
import uuid as uuid_mod
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import config

logger  = logging.getLogger("API")
api_app = FastAPI(title="MiniCPM-o Gateway", version="2.1")

_mgr  = None
_esl  = None
_loop: Optional[asyncio.AbstractEventLoop] = None


def init(session_mgr, esl_conn, loop):
    global _mgr, _esl, _loop
    _mgr  = session_mgr
    _esl  = esl_conn
    _loop = loop


@api_app.exception_handler(Exception)
async def _global_exc(request: Request, exc: Exception):
    logger.exception(f"Unhandled API error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@api_app.get("/")
async def root():
    return {"service": "MiniCPM-o Gateway", "version": "2.1", "ai_mode": config.AI_MODE}


@api_app.get("/health")
async def health():
    return {
        "esl_connected":   _esl.connected if _esl else False,
        "active_sessions": len(_mgr.sessions) if _mgr else 0,
        "ai_mode":         config.AI_MODE,
    }


@api_app.get("/sessions")
async def list_sessions():
    if not _mgr:
        return {"count": 0, "sessions": []}
    return {"count": len(_mgr.sessions), "sessions": _mgr.list_active()}


@api_app.get("/sessions/{call_uuid}")
async def get_session(call_uuid: str):
    if not _mgr:
        raise HTTPException(status_code=503, detail="Not initialized")
    s = _mgr.get(call_uuid)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "uuid":            s.uuid,
        "direction":       s.direction,
        "caller":          s.caller_number,
        "callee":          s.callee_number,
        "is_active":       s.is_active,
        "is_ai_speaking":  s.is_ai_speaking,
        "ai_connected":    s.ai_client.connected if s.ai_client else False,
        "audio_in_bytes":  s.audio_in_bytes,
        "audio_out_bytes": s.audio_out_bytes,
        "duration_sec":    round(time.time() - s.created_at, 1),
    }


@api_app.post("/call")
async def make_call(request: dict):
    phone = str(request.get("phone_number", "")).strip()
    if not phone:
        raise HTTPException(status_code=400, detail="phone_number required")

    gateway   = request.get("gateway",   config.FXO_GATEWAY)
    caller_id = request.get("caller_id", config.OUTBOUND_CID)
    call_uuid = str(uuid_mod.uuid4())

    cmd = (
        f"originate {{"
        f"origination_uuid={call_uuid},"
        f"origination_caller_id_number={caller_id},"
        f"origination_caller_id_name=AI,"
        f"ignore_early_media=true"
        f"}} sofia/gateway/{gateway}/{phone} &park"
    )

    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None, _esl.send_command, cmd
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ESL error: {e}")

    return {"status": "initiated", "call_uuid": call_uuid,
            "phone_number": phone, "fs_result": result}


@api_app.post("/call/{call_uuid}/hangup")
async def hangup_call(call_uuid: str):
    if not _esl:
        raise HTTPException(status_code=503, detail="Not initialized")
    try:
        _esl.send_bgapi_nowait(f"uuid_kill {call_uuid}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ESL error: {e}")
    return {"status": "hangup_sent"}


@api_app.post("/call/{call_uuid}/barge_in")
async def barge_in(call_uuid: str):
    s = _mgr.get(call_uuid) if _mgr else None
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    s.barge_in()
    return {"status": "barge_in_triggered"}


@api_app.post("/call/{call_uuid}/say")
async def say_text(call_uuid: str, request: dict):
    s = _mgr.get(call_uuid) if _mgr else None
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    text = str(request.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    if s.ai_client and _loop:
        asyncio.run_coroutine_threadsafe(
            s.ai_client.send_text(text), _loop
        )
    return {"status": "sent"}
'''.lstrip()

# ── main.py ────────────────────────────────────────────────────────────
FILES["main.py"] = '''
"""
MiniCPM-o 网关 v2.1 入口

事件循环策略（无 nest_asyncio）：
  - uvicorn lifespan 钩子在已运行的事件循环中做初始化
  - asyncio.get_running_loop() 获取真正运行的 loop 引用
  - 跨线程协程调用通过 run_coroutine_threadsafe(coro, loop) 完成
"""

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn

import config
from esl import ESLConnection
from session import CallSession
from api import api_app, init as api_init

logger = logging.getLogger("Main")

ALLOWED_CONTEXTS = frozenset({"public", "default"})


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
            s = CallSession(uuid=uuid, direction=direction,
                            caller_number=caller, callee_number=callee)
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
            return [{
                "uuid":            s.uuid,
                "direction":       s.direction,
                "caller":          s.caller_number,
                "callee":          s.callee_number,
                "duration_sec":    round(time.time() - s.created_at, 1),
                "is_ai_speaking":  s.is_ai_speaking,
                "ai_connected":    s.ai_client.connected if s.ai_client else False,
                "audio_in_bytes":  s.audio_in_bytes,
                "audio_out_bytes": s.audio_out_bytes,
            } for s in self.sessions.values()]


# ======================================================================
#  ESL 事件处理
# ======================================================================

class GatewayEventHandler:
    def __init__(self, mgr: SessionManager,
                 esl: ESLConnection,
                 loop: asyncio.AbstractEventLoop):
        self._mgr  = mgr
        self._esl  = esl
        self._loop = loop

    def on_channel_create(self, event: dict):
        h         = event["headers"]
        call_uuid = h.get("Unique-ID", "")
        direction = h.get("Call-Direction", "unknown")
        caller    = h.get("Caller-Caller-ID-Number", "unknown")
        callee    = h.get("Caller-Destination-Number", "unknown")
        context   = h.get("Caller-Context", "")

        if context not in ALLOWED_CONTEXTS:
            logger.debug(f"Ignoring channel in context={context!r}")
            return

        self._mgr.create(call_uuid, direction, caller, callee)
        logger.info(f"[ESL] CHANNEL_CREATE {call_uuid} ({direction}) {caller} -> {callee}")

    def on_channel_answer(self, event: dict):
        h         = event["headers"]
        call_uuid = h.get("Unique-ID", "")
        session   = self._mgr.get(call_uuid)
        if not session:
            return

        session.on_answered()

        fifo_path = f"/tmp/minicpm_{call_uuid}.pcma"
        try:
            os.mkfifo(fifo_path)
        except FileExistsError:
            pass
        except OSError as e:
            logger.error(f"[{call_uuid}] mkfifo failed: {e}")
            return

        session._fifo_path = fifo_path

        try:
            self._esl.send_bgapi_nowait(
                f"uuid_record {call_uuid} start {fifo_path} 1800"
            )
        except Exception as e:
            logger.error(f"[{call_uuid}] uuid_record failed: {e}")
            return

        session.start(self._loop, self._esl)
        logger.info(f"[ESL] CHANNEL_ANSWER {call_uuid} - session started")

    def on_channel_hangup(self, event: dict):
        h         = event["headers"]
        call_uuid = h.get("Unique-ID", "")
        cause     = h.get("Hangup-Cause", "UNKNOWN")
        session   = self._mgr.get(call_uuid)
        if session:
            session.hangup(cause)
            self._mgr.remove(call_uuid)
        logger.info(f"[ESL] CHANNEL_HANGUP {call_uuid} cause={cause}")

    def on_dtmf(self, event: dict):
        h         = event["headers"]
        call_uuid = h.get("Unique-ID", "")
        digit     = h.get("DTMF-Digit", "")
        session   = self._mgr.get(call_uuid)
        if session and digit:
            session.on_dtmf(digit)


# ======================================================================
#  应用启动（标准 lifespan，无猴子补丁）
# ======================================================================

def build_app():
    @asynccontextmanager
    async def lifespan(app):
        loop = asyncio.get_running_loop()

        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        esl = ESLConnection(config.FS_HOST, config.FS_ESL_PORT, config.FS_ESL_PASS)
        if not esl.connect():
            raise RuntimeError("ESL connection failed")

        mgr     = SessionManager()
        handler = GatewayEventHandler(mgr, esl, loop)

        esl.on("CHANNEL_CREATE",  handler.on_channel_create)
        esl.on("CHANNEL_ANSWER",  handler.on_channel_answer)
        esl.on("CHANNEL_HANGUP",  handler.on_channel_hangup)
        esl.on("DTMF",            handler.on_dtmf)

        esl.subscribe(
            "CHANNEL_CREATE", "CHANNEL_ANSWER",
            "CHANNEL_HANGUP", "DTMF", "BACKGROUND_JOB",
        )
        esl.start()

        api_init(mgr, esl, loop)

        logger.info("=" * 52)
        logger.info("  MiniCPM-o Gateway v2.1  ready")
        logger.info(f"  AI mode : {config.AI_MODE}")
        logger.info(f"  ESL     : {config.FS_HOST}:{config.FS_ESL_PORT}")
        logger.info(f"  API     : http://0.0.0.0:{config.API_PORT}")
        logger.info("=" * 52)

        yield

        logger.info("Shutting down...")
        with mgr._lock:
            sessions = list(mgr.sessions.values())
        for s in sessions:
            s.hangup("server_shutdown")
        esl.stop()
        logger.info("Shutdown complete")

    api_app.router.lifespan_context = lifespan
    return api_app


def main():
    app = build_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=getattr(config, "API_PORT", 8080),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
'''.lstrip()

# ── requirements.txt ───────────────────────────────────────────────────
FILES["requirements.txt"] = """\
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
websockets>=12.0
numpy>=1.26.0
"""

# ── README.md ──────────────────────────────────────────────────────────
FILES["README.md"] = """\
# MiniCPM-o Gateway v2.1

## 快速开始

```bash
cd minicpm_gateway
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 编辑 config.py，填写 FreeSWITCH 和 AI 后端地址
python main.py
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.py` | 配置项（ESL / AI / 外呼） |
| `audio_utils.py` | G.711a 编解码 + 重采样（桩，需替换） |
| `audio.py` | 音频格式转换 + 帧缓冲 |
| `esl.py` | FreeSWITCH ESL 连接（单读取线程） |
| `ai_client.py` | MiniCPM-o WebSocket 客户端 |
| `session.py` | 单路通话会话状态机 |
| `api.py` | FastAPI HTTP 路由 |
| `main.py` | 入口 + 会话管理器 + ESL 事件路由 |

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health` | 健康检查 |
| GET  | `/sessions` | 列出活跃通话 |
| GET  | `/sessions/{uuid}` | 查询单路通话 |
| POST | `/call` | 发起外呼 |
| POST | `/call/{uuid}/hangup` | 挂断 |
| POST | `/call/{uuid}/barge_in` | 打断 AI |
| POST | `/call/{uuid}/say` | 注入文本让 AI 说话 |

## 注意事项

- `audio_utils.py` 中的 G.711a 编解码为桩实现，生产环境请替换为
  `audioop`（Python 3.12 前）或 `g711`、`pyg711` 等第三方库。
- 重采样建议使用 `soxr`（`pip install soxr`）替换线性插值。
- FreeSWITCH 需开启 ESL 模块并配置正确的 context。
"""

# ======================================================================
#  脚手架执行
# ======================================================================

def main():
    os.makedirs(ROOT, exist_ok=True)
    created = []

    for filename, content in FILES.items():
        path = os.path.join(ROOT, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        created.append(path)

    print(f"\n✅  项目已生成：{os.path.abspath(ROOT)}/\n")
    for p in created:
        size = os.path.getsize(p)
        print(f"   {p:<45}  {size:>6} bytes")

    print(f"""
下一步：
  cd {ROOT}
  python -m venv .venv
  source .venv/bin/activate        # Windows: .venv\\\\Scripts\\\\activate
  pip install -r requirements.txt
  # 编辑 config.py
  python main.py
""")


if __name__ == "__main__":
    main()
```

---

运行方式：

```bash
python bootstrap.py
```

输出示例：

```
✅  项目已生成：/your/path/minicpm_gateway/

   minicpm_gateway/config.py              312 bytes
   minicpm_gateway/audio_utils.py         687 bytes
   minicpm_gateway/audio.py              1843 bytes
   minicpm_gateway/esl.py                5521 bytes
   minicpm_gateway/ai_client.py          4872 bytes
   minicpm_gateway/session.py            8934 bytes
   minicpm_gateway/api.py                3201 bytes
   minicpm_gateway/main.py               4108 bytes
   minicpm_gateway/requirements.txt        78 bytes
   minicpm_gateway/README.md              1024 bytes

下一步：
  cd minicpm_gateway
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  # 编辑 config.py
  python main.py
```
