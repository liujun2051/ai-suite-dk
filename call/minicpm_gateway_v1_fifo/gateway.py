#!/usr/bin/env python3
"""
MiniCPM-o 网关 v1.0 - FIFO 版本
FreeSWITCH ESL + uuid_record/FIFO
特点：稳定可靠，无需额外模块
"""

import asyncio
import base64
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

import numpy as np
import websockets
from fastapi import FastAPI, HTTPException
import uvicorn

import config
from audio_utils import pcma_to_pcm, pcm_to_pcma, RESAMPLE_FUNC, get_frame_size_ms

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MiniCPM-Gateway")


class ESLConnection:
    def __init__(self, host: str, port: int, password: str):
        self.host = host
        self.port = port
        self.password = password
        self.sock = None
        self.connected = False
        self._lock = threading.Lock()
        self._event_callbacks = {}
        self._running = False
        self._reader_thread = None

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
            header = self._recv_line()
            if "auth/request" not in header:
                return False
            self._send(f"auth {self.password}\n\n")
            response = self._recv_line()
            if "ok" not in response.lower():
                return False
            self.connected = True
            logger.info("ESL connected")
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
                raise ConnectionError("ESL closed")
            buffer += chunk
            if buffer.endswith(b"\n"):
                return buffer.decode('utf-8').strip()

    def _recv_event(self):
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
                for cb in self._event_callbacks.get(event_name, []):
                    try:
                        cb(event)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            except Exception as e:
                if self._running:
                    time.sleep(1)

    def stop(self):
        self._running = False
        if self.sock:
            self.sock.close()


class AudioFormatConverter:
    @staticmethod
    def fs_to_ai(pcma_8k: bytes) -> str:
        pcm_8k = pcma_to_pcm(pcma_8k)
        pcm_16k = RESAMPLE_FUNC(pcm_8k, 8000, 16000)
        return base64.b64encode(pcm_16k.astype(np.int16).tobytes()).decode()

    @staticmethod
    def ai_to_fs(b64_audio: str) -> bytes:
        pcm_16k = np.frombuffer(base64.b64decode(b64_audio), dtype=np.int16)
        pcm_8k = RESAMPLE_FUNC(pcm_16k, 16000, 8000)
        target_len = get_frame_size_ms(8000, 20)
        if len(pcm_8k) > target_len:
            pcm_8k = pcm_8k[:target_len]
        elif len(pcm_8k) < target_len:
            pcm_8k = np.pad(pcm_8k, (0, target_len - len(pcm_8k)))
        return pcm_to_pcma(pcm_8k)


class MiniCPMClient:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.ws = None
        self.connected = False
        self.mode = config.AI_MODE
        self.on_audio_delta = None
        self.on_audio_done = None
        self.on_transcript = None
        self.on_speech_started = None

    async def connect(self) -> bool:
        headers = {}
        if self.mode == "cloud":
            headers["Authorization"] = f"Bearer {config.CLOUD_API_KEY}"
        try:
            self.ws = await websockets.connect(
                config.MINICPM_WS_URL,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            self.connected = True
            asyncio.create_task(self._receive_loop())
            return True
        except Exception as e:
            logger.error(f"[{self.session_id}] Connection failed: {e}")
            return False

    async def initialize(self):
        if self.mode == "cloud":
            await self._init_cloud()
        else:
            await self._init_local()

    async def _init_cloud(self):
        async for msg in self.ws:
            data = json.loads(msg)
            if data.get("type") == "session.created":
                break
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": "你是一个电话客服助手，说话简洁自然。",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1", "language": "zh"}
            }
        }))
        async for msg in self.ws:
            data = json.loads(msg)
            if data.get("type") == "session.updated":
                return

    async def _init_local(self):
        await self.ws.send(json.dumps({
            "type": "init",
            "session_id": self.session_id,
            "sample_rate": config.AI_SAMPLE_RATE,
            "channels": 1,
            "format": "pcm_s16le"
        }))

    async def send_audio(self, b64_audio: str):
        if self.mode == "cloud":
            await self.ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64_audio}))
            await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        else:
            pcm = np.frombuffer(base64.b64decode(b64_audio), dtype=np.int16)
            await self.ws.send(pcm.tobytes())

    async def send_text(self, text: str):
        if self.mode == "cloud":
            await self.ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
            }))
            await self.ws.send(json.dumps({"type": "response.create"}))
        else:
            await self.ws.send(json.dumps({"type": "text_input", "text": text}))

    async def interrupt(self):
        if self.mode == "cloud":
            await self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        else:
            await self.ws.send(json.dumps({"type": "barge_in"}))

    async def _receive_loop(self):
        try:
            async for msg in self.ws:
                if not self.connected:
                    break
                if isinstance(msg, str):
                    await self._handle_text(msg)
                else:
                    if self.mode == "local" and self.on_audio_delta:
                        self.on_audio_delta(base64.b64encode(msg).decode())
        except Exception as e:
            logger.error(f"Receive error: {e}")
        finally:
            self.connected = False

    async def _handle_text(self, msg: str):
        try:
            data = json.loads(msg)
            msg_type = data.get("type", "")
            if msg_type == "response.output_audio.delta" and self.on_audio_delta:
                self.on_audio_delta(data.get("delta", ""))
            elif msg_type == "response.output_audio.done" and self.on_audio_done:
                self.on_audio_done()
            elif msg_type == "input_audio_buffer.speech_started" and self.on_speech_started:
                self.on_speech_started()
            elif msg_type == "conversation.item.input_audio_transcription.completed" and self.on_transcript:
                self.on_transcript(data.get("transcript", ""))
        except:
            pass

    async def close(self):
        self.connected = False
        if self.ws:
            await self.ws.close()


@dataclass
class CallSession:
    uuid: str
    direction: str
    caller_number: str
    callee_number: str
    created_at: float = field(default_factory=time.time)
    fs_to_ai_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=300))
    ai_to_fs_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=300))
    ai_client = None
    is_active = True
    is_answered = False
    is_ai_speaking = False
    barge_in_requested = False
    audio_in_bytes = 0
    audio_out_bytes = 0
    _threads = field(default_factory=list)
    _fifo_path = ""

    def start(self):
        t1 = threading.Thread(target=self._fifo_read_loop)
        t1.daemon = True
        t1.start()
        self._threads.append(t1)
        asyncio.create_task(self._ai_main())
        t2 = threading.Thread(target=self._send_audio_to_fs)
        t2.daemon = True
        t2.start()
        self._threads.append(t2)

    def _fifo_read_loop(self):
        try:
            with open(self._fifo_path, "rb") as f:
                while self.is_active:
                    data = f.read(160)
                    if not data:
                        break
                    self.audio_in_bytes += len(data)
                    try:
                        self.fs_to_ai_queue.put(data, timeout=0.01)
                    except queue.Full:
                        try:
                            self.fs_to_ai_queue.get_nowait()
                            self.fs_to_ai_queue.put(data, timeout=0.01)
                        except:
                            pass
        except Exception as e:
            logger.error(f"[{self.uuid}] FIFO error: {e}")
        finally:
            if os.path.exists(self._fifo_path):
                os.remove(self._fifo_path)

    async def _ai_main(self):
        self.ai_client = MiniCPMClient(self.uuid)
        self.ai_client.on_audio_delta = self._on_ai_audio
        self.ai_client.on_audio_done = self._on_ai_done
        self.ai_client.on_speech_started = self._on_cloud_speech
        self.ai_client.on_transcript = self._on_transcript
        if not await self.ai_client.connect():
            self.hangup("ai_connect_failed")
            return
        await self.ai_client.initialize()
        if self.direction == "outbound":
            await self.ai_client.send_text("你好，我是智能客服，请问有什么可以帮您？")
        await self._send_to_ai_loop()

    async def _send_to_ai_loop(self):
        converter = AudioFormatConverter()
        while self.is_active and self.ai_client and self.ai_client.connected:
            try:
                audio_data = self.fs_to_ai_queue.get(timeout=0.02)
                b64_audio = converter.fs_to_ai(audio_data)
                await self.ai_client.send_audio(b64_audio)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[{self.uuid}] Send error: {e}")
                break

    def _on_ai_audio(self, b64_audio: str):
        converter = AudioFormatConverter()
        pcma = converter.ai_to_fs(b64_audio)
        try:
            self.ai_to_fs_queue.put(pcma, timeout=0.01)
            self.audio_out_bytes += len(pcma)
        except queue.Full:
            pass

    def _on_ai_done(self):
        self.is_ai_speaking = False

    def _on_cloud_speech(self):
        if self.is_ai_speaking:
            self._execute_barge_in()

    def _on_transcript(self, text: str):
        logger.info(f"[{self.uuid}] User: {text}")

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
                files = sorted([f for f in os.listdir(tmp_dir) if f.endswith('.pcma')],
                              key=lambda x: os.path.getmtime(os.path.join(tmp_dir, x)))
                for old in files[:-5]:
                    os.remove(os.path.join(tmp_dir, old))
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[{self.uuid}] Send error: {e}")
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def handle_dtmf(self, digit: str):
        logger.info(f"[{self.uuid}] DTMF: {digit}")
        if digit in ["*", "#"]:
            self._execute_barge_in()
        else:
            if self.ai_client:
                asyncio.create_task(self.ai_client.send_text(f"用户按键：{digit}"))

    def _execute_barge_in(self):
        if self.barge_in_requested:
            return
        self.barge_in_requested = True
        if hasattr(self, '_esl_conn_ref'):
            esl = self._esl_conn_ref()
            if esl:
                esl.send_command(f"uuid_break {self.uuid}")
        if self.ai_client:
            asyncio.create_task(self.ai_client.interrupt())
        while not self.ai_to_fs_queue.empty():
            try:
                self.ai_to_fs_queue.get_nowait()
            except:
                break
        def reset():
            time.sleep(0.5)
            self.barge_in_requested = False
            self.is_ai_speaking = False
        threading.Thread(target=reset, daemon=True).start()

    def set_esl_ref(self, esl_conn):
        import weakref
        self._esl_conn_ref = weakref.ref(esl_conn)

    def on_answered(self):
        self.is_answered = True

    def hangup(self, reason: str = "normal"):
        logger.info(f"[{self.uuid}] Hanging up: {reason}")
        self.is_active = False
        if self.ai_client:
            asyncio.create_task(self.ai_client.close())
        for t in self._threads:
            t.join(timeout=2)
        if os.path.exists(self._fifo_path):
            os.remove(self._fifo_path)


class SessionManager:
    def __init__(self):
        self.sessions = {}
        self._lock = threading.RLock()

    def create(self, uuid, direction, caller, callee):
        with self._lock:
            if uuid in self.sessions:
                return self.sessions[uuid]
            session = CallSession(uuid=uuid, direction=direction, caller_number=caller, callee_number=callee)
            self.sessions[uuid] = session
            return session

    def get(self, uuid):
        with self._lock:
            return self.sessions.get(uuid)

    def remove(self, uuid):
        with self._lock:
            if uuid in self.sessions:
                del self.sessions[uuid]

    def list_active(self):
        with self._lock:
            return [{"uuid": s.uuid, "direction": s.direction, "caller": s.caller_number,
                     "duration": time.time() - s.created_at, "ai_speaking": s.is_ai_speaking,
                     "ai_connected": s.ai_client.connected if s.ai_client else False}
                    for s in self.sessions.values()]


session_mgr = SessionManager()
esl_conn = None


def on_channel_create(event):
    h = event["headers"]
    uuid = h.get("Unique-ID")
    direction = h.get("Call-Direction", "unknown")
    caller = h.get("Caller-Caller-ID-Number", "unknown")
    callee = h.get("Caller-Destination-Number", "unknown")
    if h.get("Caller-Context", "") not in ["public", "default"]:
        return
    session = session_mgr.create(uuid, direction, caller, callee)
    session.set_esl_ref(esl_conn)
    logger.info(f"[ESL] Create: {uuid} ({direction}) {caller} -> {callee}")


def on_channel_answer(event):
    h = event["headers"]
    uuid = h.get("Unique-ID")
    session = session_mgr.get(uuid)
    if not session:
        return
    session.on_answered()
    fifo_path = f"/tmp/minicpm_fifo_{uuid}.pcma"
    os.system(f"mkfifo {fifo_path} 2>/dev/null")
    session._fifo_path = fifo_path
    esl_conn.send_bgapi(f"uuid_record {uuid} start {fifo_path} 1800")
    session.start()


def on_channel_hangup(event):
    h = event["headers"]
    uuid = h.get("Unique-ID")
    cause = h.get("Hangup-Cause", "UNKNOWN")
    session = session_mgr.get(uuid)
    if session:
        session.hangup(cause)
        session_mgr.remove(uuid)
    logger.info(f"[ESL] Hangup: {uuid}, cause: {cause}")


def on_dtmf(event):
    h = event["headers"]
    uuid = h.get("Unique-ID")
    digit = h.get("DTMF-Digit")
    session = session_mgr.get(uuid)
    if session:
        session.handle_dtmf(digit)


api_app = FastAPI(title="MiniCPM-o Gateway v1.0 FIFO")


@api_app.get("/")
async def root():
    return {"status": "running", "version": "1.0-fifo", "ai_mode": config.AI_MODE}


@api_app.get("/sessions")
async def list_sessions():
    return {"count": len(session_mgr.sessions), "sessions": session_mgr.list_active()}


@api_app.post("/call")
async def make_call(request: dict):
    phone = request.get("phone_number")
    if not phone:
        raise HTTPException(status_code=400, detail="phone_number required")
    gateway = request.get("gateway", config.FXO_GATEWAY)
    caller_id = request.get("caller_id", config.OUTBOUND_CID)
    call_uuid = str(uuid.uuid4())
    originate_str = (f"originate {{origination_uuid={call_uuid},origination_caller_id_number={caller_id},"
                      f"origination_caller_id_name=AI,ignore_early_media=true}} "
                      f"sofia/gateway/{gateway}/{phone} &park")
    result = esl_conn.send_command(originate_str)
    return {"status": "initiated", "call_uuid": call_uuid, "phone_number": phone, "fs_result": result}


@api_app.post("/call/{call_uuid}/hangup")
async def hangup_call(call_uuid: str):
    esl_conn.send_command(f"uuid_kill {call_uuid}")
    return {"status": "hangup_sent"}


@api_app.post("/call/{call_uuid}/barge_in")
async def barge_in(call_uuid: str):
    session = session_mgr.get(call_uuid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session._execute_barge_in()
    return {"status": "barge_in_triggered"}


@api_app.get("/health")
async def health():
    return {"esl_connected": esl_conn.connected if esl_conn else False,
            "active_sessions": len(session_mgr.sessions), "ai_mode": config.AI_MODE, "audio_backend": "fifo"}


def start_esl():
    global esl_conn
    esl_conn = ESLConnection(config.FS_HOST, config.FS_ESL_PORT, config.FS_ESL_PASS)
    if not esl_conn.connect():
        return False
    esl_conn.on("CHANNEL_CREATE", on_channel_create)
    esl_conn.on("CHANNEL_ANSWER", on_channel_answer)
    esl_conn.on("CHANNEL_HANGUP", on_channel_hangup)
    esl_conn.on("DTMF", on_dtmf)
    esl_conn.subscribe("CHANNEL_CREATE CHANNEL_ANSWER CHANNEL_HANGUP DTMF")
    esl_conn.start_event_loop()
    return True


def main():
    logger.info("=" * 50)
    logger.info("MiniCPM-o Gateway v1.0 [FIFO]")
    logger.info(f"Mode: {config.AI_MODE}")
    logger.info("=" * 50)
    if not start_esl():
        return 1
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(api_app, host="0.0.0.0", port=8080, log_level="warning")
    return 0


if __name__ == "__main__":
    exit(main())
