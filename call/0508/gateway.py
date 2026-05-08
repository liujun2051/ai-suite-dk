#!/usr/bin/env python3
"""
MiniCPM-o 4.1 全双工语音网关
纯 Python 实现，通过 ESL 控制 FreeSWITCH
"""

import asyncio
import json
import logging
import os
import queue
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import websockets
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# 本地模块
import config
from audio_utils import pcma_to_pcm, pcm_to_pcma, RESAMPLE_FUNC, get_frame_size_ms

# ========== 日志 ==========
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MiniCPM-Gateway")


# ========== ESL 客户端 ==========

class ESLConnection:
    """FreeSWITCH Event Socket Library 客户端"""

    def __init__(self, host: str, port: int, password: str):
        self.host = host
        self.port = port
        self.password = password
        self.sock: Optional[socket.socket] = None
        self.connected = False
        self._lock = threading.Lock()
        self._event_callbacks: Dict[str, list] = {}
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))

            header = self._recv_line()
            if "auth/request" not in header:
                logger.error(f"Unexpected ESL greeting: {header}")
                return False

            self._send(f"auth {self.password}\n\n")
            response = self._recv_line()

            if "ok" not in response.lower():
                logger.error(f"ESL auth failed: {response}")
                return False

            self.connected = True
            logger.info("ESL connected and authenticated")
            return True

        except Exception as e:
            logger.error(f"ESL connection failed: {e}")
            return False

    def _send(self, data: str):
        with self._lock:
            self.sock.sendall(data.encode('utf-8'))

    def _recv_line(self) -> str:
        buffer = b""
        while True:
            chunk = self.sock.recv(1)
            if not chunk:
                raise ConnectionError("ESL connection closed")
            buffer += chunk
            if buffer.endswith(b"\n"):
                return buffer.decode('utf-8').strip()

    def _recv_event(self) -> Dict:
        headers = {}
        body = ""

        while True:
            line = self._recv_line()
            if line == "":
                break
            if ": " in line:
                key, val = line.split(": ", 1)
                headers[key] = val

        content_length = int(headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.sock.recv(content_length).decode('utf-8')

        return {"headers": headers, "body": body}

    def send_command(self, cmd: str) -> str:
        self._send(f"api {cmd}\n\n")
        response = ""
        while True:
            line = self._recv_line()
            if line == "":
                break
            response += line + "\n"
        return response.strip()

    def send_bgapi(self, cmd: str) -> str:
        self._send(f"bgapi {cmd}\n\n")
        response = ""
        while True:
            line = self._recv_line()
            if line == "":
                break
            response += line + "\n"
        return response.strip()

    def subscribe(self, events: str):
        self._send(f"event plain {events}\n\n")
        self._recv_line()

    def on(self, event_name: str, callback):
        if event_name not in self._event_callbacks:
            self._event_callbacks[event_name] = []
        self._event_callbacks[event_name].append(callback)

    def start_event_loop(self):
        self._running = True
        self._reader_thread = threading.Thread(target=self._event_loop)
        self._reader_thread.daemon = True
        self._reader_thread.start()

    def _event_loop(self):
        while self._running:
            try:
                event = self._recv_event()
                event_name = event["headers"].get("Event-Name", "UNKNOWN")

                callbacks = self._event_callbacks.get(event_name, [])
                for cb in callbacks:
                    try:
                        cb(event)
                    except Exception as e:
                        logger.error(f"Event callback error: {e}")

            except Exception as e:
                if self._running:
                    logger.error(f"Event loop error: {e}")
                    time.sleep(1)

    def stop(self):
        self._running = False
        if self.sock:
            self.sock.close()


# ========== 通话会话 ==========

@dataclass
class CallSession:
    uuid: str
    direction: str
    caller_number: str
    callee_number: str
    created_at: float = field(default_factory=time.time)

    audio_port: int = 0
    fs_audio_socket: Optional[socket.socket] = None

    fs_to_ai_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=300))
    ai_to_fs_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=300))

    minicpm_ws: Optional[websockets.WebSocketClientProtocol] = None
    minicpm_connected: bool = False

    is_active: bool = True
    is_answered: bool = False
    is_ai_speaking: bool = False
    barge_in_requested: bool = False

    audio_in_bytes: int = 0
    audio_out_bytes: int = 0

    _threads: list = field(default_factory=list)
    _tasks: list = field(default_factory=list)

    def __post_init__(self):
        self.audio_port = self._allocate_port()
        logger.info(f"[{self.uuid}] Session created, audio port: {self.audio_port}")

    def _allocate_port(self) -> int:
        return config.AUDIO_PORT_BASE + (hash(self.uuid) % (config.AUDIO_PORT_MAX - config.AUDIO_PORT_BASE))

    def start(self):
        t1 = threading.Thread(target=self._receive_fs_audio)
        t1.daemon = True
        t1.start()
        self._threads.append(t1)

        asyncio.create_task(self._minicpm_main())

        t2 = threading.Thread(target=self._send_audio_to_fs)
        t2.daemon = True
        t2.start()
        self._threads.append(t2)

    def _receive_fs_audio(self):
        self.fs_audio_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.fs_audio_socket.bind(("0.0.0.0", self.audio_port))
        self.fs_audio_socket.settimeout(2.0)

        logger.info(f"[{self.uuid}] Listening for FS audio on port {self.audio_port}")

        frame_size_8k = get_frame_size_ms(config.FS_SAMPLE_RATE, config.FRAME_DURATION_MS)
        buffer = bytearray()

        while self.is_active:
            try:
                data, addr = self.fs_audio_socket.recvfrom(2048)
                if not data:
                    continue

                buffer.extend(data)
                self.audio_in_bytes += len(data)

                while len(buffer) >= frame_size_8k:
                    frame_8k = bytes(buffer[:frame_size_8k])
                    buffer = buffer[frame_size_8k:]

                    pcm_8k = pcma_to_pcm(frame_8k)
                    pcm_16k = RESAMPLE_FUNC(pcm_8k, config.FS_SAMPLE_RATE, config.MINICPM_SAMPLE_RATE)

                    try:
                        self.fs_to_ai_queue.put(pcm_16k.tobytes(), timeout=0.01)
                    except queue.Full:
                        try:
                            self.fs_to_ai_queue.get_nowait()
                            self.fs_to_ai_queue.put(pcm_16k.tobytes(), timeout=0.01)
                        except queue.Empty:
                            pass

            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"[{self.uuid}] FS audio receive error: {e}")

        self.fs_audio_socket.close()
        logger.info(f"[{self.uuid}] FS audio receiver stopped")

    async def _minicpm_main(self):
        try:
            logger.info(f"[{self.uuid}] Connecting to MiniCPM-o at {config.MINICPM_WS_URL}")

            async with websockets.connect(
                config.MINICPM_WS_URL,
                ping_interval=20,
                ping_timeout=10
            ) as ws:
                self.minicpm_ws = ws
                self.minicpm_connected = True

                init_msg = {
                    "type": "init",
                    "session_id": self.uuid,
                    "sample_rate": config.MINICPM_SAMPLE_RATE,
                    "channels": 1,
                    "format": "pcm_s16le"
                }
                await ws.send(json.dumps(init_msg))
                logger.info(f"[{self.uuid}] MiniCPM-o connected")

                send_task = asyncio.create_task(self._send_to_minicpm(ws))
                recv_task = asyncio.create_task(self._recv_from_minicpm(ws))

                await asyncio.gather(send_task, recv_task)

        except Exception as e:
            logger.error(f"[{self.uuid}] MiniCPM-o connection error: {e}")
            self.minicpm_connected = False
            await asyncio.sleep(5)
            if self.is_active:
                asyncio.create_task(self._minicpm_main())

    async def _send_to_minicpm(self, ws):
        frame_size = get_frame_size_ms(config.MINICPM_SAMPLE_RATE, config.FRAME_DURATION_MS) * 2

        while self.is_active and self.minicpm_connected:
            try:
                frames = []
                total_len = 0

                while total_len < frame_size:
                    try:
                        chunk = self.fs_to_ai_queue.get(timeout=0.02)
                        frames.append(chunk)
                        total_len += len(chunk)
                    except queue.Empty:
                        break

                if frames:
                    data = b"".join(frames)
                    await ws.send(data)
                else:
                    silence = b'\x00' * frame_size
                    await ws.send(silence)
                    await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"[{self.uuid}] Send to MiniCPM-o error: {e}")
                break

    async def _recv_from_minicpm(self, ws):
        frame_size_8k = get_frame_size_ms(config.FS_SAMPLE_RATE, config.FRAME_DURATION_MS)

        while self.is_active and self.minicpm_connected:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=5.0)

                if isinstance(message, str):
                    await self._handle_control_message(message)
                else:
                    pcm_16k = np.frombuffer(message, dtype=np.int16)
                    pcm_8k = RESAMPLE_FUNC(pcm_16k, config.MINICPM_SAMPLE_RATE, config.FS_SAMPLE_RATE)

                    target_len = frame_size_8k
                    if len(pcm_8k) > target_len:
                        pcm_8k = pcm_8k[:target_len]
                    elif len(pcm_8k) < target_len:
                        pcm_8k = np.pad(pcm_8k, (0, target_len - len(pcm_8k)))

                    pcma = pcm_to_pcma(pcm_8k)

                    try:
                        self.ai_to_fs_queue.put(pcma, timeout=0.01)
                        self.audio_out_bytes += len(pcma)
                    except queue.Full:
                        pass

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"[{self.uuid}] Recv from MiniCPM-o error: {e}")
                break

    async def _handle_control_message(self, msg: str):
        try:
            data = json.loads(msg)
            msg_type = data.get("type", "")

            if msg_type == "speech_start":
                self.is_ai_speaking = True
                logger.debug(f"[{self.uuid}] AI started speaking")
            elif msg_type == "speech_end":
                self.is_ai_speaking = False
                logger.debug(f"[{self.uuid}] AI stopped speaking")
            elif msg_type == "barge_in_detected":
                logger.info(f"[{self.uuid}] Barge-in detected by MiniCPM-o")
                self._handle_barge_in()
            elif msg_type == "error":
                logger.error(f"[{self.uuid}] MiniCPM-o error: {data.get('message')}")

        except json.JSONDecodeError:
            logger.warning(f"[{self.uuid}] Invalid control message: {msg[:100]}")

    def _send_audio_to_fs(self):
        tmp_dir = tempfile.mkdtemp(prefix=f"ai_{self.uuid}_")

        while self.is_active:
            try:
                pcma = self.ai_to_fs_queue.get(timeout=0.05)

                tmp_file = os.path.join(tmp_dir, f"{int(time.time()*1000)}.pcma")
                with open(tmp_file, "wb") as f:
                    f.write(pcma)

                if hasattr(self, '_esl_conn_ref'):
                    esl = self._esl_conn_ref()
                    if esl:
                        esl.send_bgapi(f"uuid_broadcast {self.uuid} {tmp_file} both")

                self._cleanup_old_files(tmp_dir, keep=10)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[{self.uuid}] Send audio to FS error: {e}")

        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def _cleanup_old_files(self, directory: str, keep: int = 10):
        try:
            files = sorted(
                [f for f in os.listdir(directory) if f.endswith('.pcma')],
                key=lambda x: os.path.getmtime(os.path.join(directory, x))
            )
            for old_file in files[:-keep]:
                os.remove(os.path.join(directory, old_file))
        except Exception:
            pass

    def handle_dtmf(self, digit: str):
        logger.info(f"[{self.uuid}] DTMF received: {digit}")

        if digit == "*" or digit == "#":
            self._handle_barge_in()
        else:
            asyncio.create_task(self._send_text_to_minicpm(f"用户按键：{digit}"))

    def _handle_barge_in(self):
        if not self.barge_in_requested:
            self.barge_in_requested = True
            logger.info(f"[{self.uuid}] Barge-in executed")

            if hasattr(self, '_esl_conn_ref'):
                esl = self._esl_conn_ref()
                if esl:
                    esl.send_command(f"uuid_break {self.uuid}")

            asyncio.create_task(self._send_control_to_minicpm({"type": "barge_in"}))

            def reset():
                time.sleep(1)
                self.barge_in_requested = False
            threading.Thread(target=reset, daemon=True).start()

    async def _send_text_to_minicpm(self, text: str):
        if self.minicpm_ws and self.minicpm_connected:
            await self.minicpm_ws.send(json.dumps({
                "type": "text_input",
                "text": text
            }))

    async def _send_control_to_minicpm(self, control: dict):
        if self.minicpm_ws and self.minicpm_connected:
            await self.minicpm_ws.send(json.dumps(control))

    def set_esl_ref(self, esl_conn):
        import weakref
        self._esl_conn_ref = weakref.ref(esl_conn)

    def on_answered(self):
        self.is_answered = True
        logger.info(f"[{self.uuid}] Call answered")

    def hangup(self, reason: str = "normal"):
        logger.info(f"[{self.uuid}] Hanging up: {reason}")
        self.is_active = False

        if self.minicpm_ws:
            asyncio.create_task(self.minicpm_ws.close())

        if self.fs_audio_socket:
            self.fs_audio_socket.close()

        for t in self._threads:
            t.join(timeout=2)

        logger.info(f"[{self.uuid}] Session destroyed. "
                   f"In: {self.audio_in_bytes} bytes, Out: {self.audio_out_bytes} bytes")


# ========== 会话管理器 ==========

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, CallSession] = {}
        self._lock = threading.RLock()

    def create(self, uuid: str, direction: str, caller: str, callee: str) -> CallSession:
        with self._lock:
            if uuid in self.sessions:
                return self.sessions[uuid]

            session = CallSession(
                uuid=uuid,
                direction=direction,
                caller_number=caller,
                callee_number=callee
            )
            self.sessions[uuid] = session
            return session

    def get(self, uuid: str) -> Optional[CallSession]:
        with self._lock:
            return self.sessions.get(uuid)

    def remove(self, uuid: str):
        with self._lock:
            if uuid in self.sessions:
                del self.sessions[uuid]

    def list_active(self) -> list:
        with self._lock:
            return [
                {
                    "uuid": s.uuid,
                    "direction": s.direction,
                    "caller": s.caller_number,
                    "duration": time.time() - s.created_at,
                    "ai_speaking": s.is_ai_speaking,
                    "minicpm_connected": s.minicpm_connected
                }
                for s in self.sessions.values()
            ]


session_mgr = SessionManager()


# ========== FreeSWITCH 事件处理 ==========

esl_conn: Optional[ESLConnection] = None


def on_channel_create(event: Dict):
    h = event["headers"]
    uuid = h.get("Unique-ID")
    direction = h.get("Call-Direction", "unknown")
    caller = h.get("Caller-Caller-ID-Number", "unknown")
    callee = h.get("Caller-Destination-Number", "unknown")

    context = h.get("Caller-Context", "")
    if context not in ["public", "default"]:
        return

    session = session_mgr.create(uuid, direction, caller, callee)
    session.set_esl_ref(esl_conn)
    logger.info(f"[ESL] Channel create: {uuid} ({direction}) {caller} -> {callee}")


def on_channel_answer(event: Dict):
    h = event["headers"]
    uuid = h.get("Unique-ID")

    session = session_mgr.get(uuid)
    if not session:
        return

    session.on_answered()

    fifo_path = f"/tmp/minicpm_fifo_{uuid}.pcma"
    os.system(f"mkfifo {fifo_path} 2>/dev/null")

    esl_conn.send_bgapi(f"uuid_record {uuid} start {fifo_path} 1800")

    def read_fifo():
        try:
            with open(fifo_path, "rb") as f:
                while session.is_active:
                    data = f.read(160)
                    if not data:
                        break
                    try:
                        session.fs_to_ai_queue.put(data, timeout=0.01)
                    except queue.Full:
                        pass
        except Exception as e:
            logger.error(f"[{uuid}] FIFO read error: {e}")
        finally:
            os.system(f"rm -f {fifo_path}")

    t = threading.Thread(target=read_fifo)
    t.daemon = True
    t.start()
    session._threads.append(t)

    session.start()


def on_channel_hangup(event: Dict):
    h = event["headers"]
    uuid = h.get("Unique-ID")
    cause = h.get("Hangup-Cause", "UNKNOWN")

    session = session_mgr.get(uuid)
    if session:
        session.hangup(cause)
        session_mgr.remove(uuid)

    logger.info(f"[ESL] Channel hangup: {uuid}, cause: {cause}")


def on_dtmf(event: Dict):
    h = event["headers"]
    uuid = h.get("Unique-ID")
    digit = h.get("DTMF-Digit")

    session = session_mgr.get(uuid)
    if session:
        session.handle_dtmf(digit)


# ========== HTTP API ==========

api_app = FastAPI(title="MiniCPM-o Gateway")


@api_app.get("/")
async def root():
    return {"status": "running", "service": "MiniCPM-o Gateway"}


@api_app.get("/sessions")
async def list_sessions():
    return {
        "count": len(session_mgr.sessions),
        "sessions": session_mgr.list_active()
    }


@api_app.post("/call")
async def make_call(request: dict):
    phone = request.get("phone_number")
    if not phone:
        raise HTTPException(status_code=400, detail="phone_number required")

    gateway = request.get("gateway", config.FXO_GATEWAY)
    caller_id = request.get("caller_id", config.OUTBOUND_CID)

    call_uuid = str(uuid.uuid4())

    originate_str = (
        f"originate "
        f"{{origination_uuid={call_uuid},"
        f"origination_caller_id_number={caller_id},"
        f"origination_caller_id_name=AI,"
        f"ignore_early_media=true,"
        f"return_ring_ready=false}} "
        f"sofia/gateway/{gateway}/{phone} "
        f"&park"
    )

    result = esl_conn.send_command(originate_str)

    return {
        "status": "initiated",
        "call_uuid": call_uuid,
        "phone_number": phone,
        "fs_result": result
    }


@api_app.post("/call/{call_uuid}/hangup")
async def hangup_call(call_uuid: str):
    session = session_mgr.get(call_uuid)
    if session:
        esl_conn.send_command(f"uuid_kill {call_uuid}")
        return {"status": "hangup_requested"}

    esl_conn.send_command(f"uuid_kill {call_uuid}")
    return {"status": "hangup_sent"}


@api_app.post("/call/{call_uuid}/barge_in")
async def barge_in(call_uuid: str):
    session = session_mgr.get(call_uuid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session._handle_barge_in()
    return {"status": "barge_in_triggered"}


@api_app.get("/health")
async def health():
    return {
        "esl_connected": esl_conn.connected if esl_conn else False,
        "active_sessions": len(session_mgr.sessions),
        "minicpm_url": config.MINICPM_WS_URL
    }


# ========== 主程序 ==========

def start_esl():
    global esl_conn

    esl_conn = ESLConnection(config.FS_HOST, config.FS_ESL_PORT, config.FS_ESL_PASS)

    if not esl_conn.connect():
        logger.error("Failed to connect to FreeSWITCH ESL")
        return False

    esl_conn.on("CHANNEL_CREATE", on_channel_create)
    esl_conn.on("CHANNEL_ANSWER", on_channel_answer)
    esl_conn.on("CHANNEL_HANGUP", on_channel_hangup)
    esl_conn.on("DTMF", on_dtmf)

    esl_conn.subscribe("CHANNEL_CREATE CHANNEL_ANSWER CHANNEL_HANGUP DTMF")
    esl_conn.start_event_loop()

    logger.info("ESL event loop started")
    return True


def main():
    logger.info("=" * 50)
    logger.info("MiniCPM-o 4.1 Gateway Starting...")
    logger.info("=" * 50)

    if not start_esl():
        logger.error("Cannot start without FreeSWITCH connection")
        return 1

    logger.info("Starting HTTP API on port 8080")

    import nest_asyncio
    nest_asyncio.apply()

    uvicorn.run(api_app, host="0.0.0.0", port=8080, log_level="warning")

    return 0


if __name__ == "__main__":
    exit(main())
