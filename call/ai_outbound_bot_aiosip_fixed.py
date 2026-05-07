#!/usr/bin/env python3
"""
AI Outbound Call Bot - aiosip + asyncio RTP + MiniCPM-o 4.5
Fixed: audio format conversion, SDP IP, dynamic RTP ports, interruption, cleanup.

Dependencies:
    pip install aiosip websockets numpy webrtcvad

FreeSWITCH Setup:
    1. Create SIP user in /etc/freeswitch/directory/default/aibot.xml
    2. Add dialplan to route calls to user/aibot@${domain_name}
    3. Or use ESL: originate sofia/gateway/mygw/13800138000 &bridge(user/aibot@192.168.1.100)
"""

import asyncio
import json
import base64
import logging
import struct
import time
import socket
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import websockets
import aiosip

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False
    print("WARNING: pip install webrtcvad for better VAD")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- 配置 ----------
@dataclass
class Config:
    local_ip: str = "0.0.0.0"           # SIP 监听地址
    local_sip_port: int = 5066          # SIP 端口
    freeswitch_ip: str = "192.168.1.100" # FreeSWITCH IP（用于获取本机 IP）
    sip_username: str = "aibot"
    sip_password: str = "bot_password"
    minicpmo_api_key: str = "你的API_Key"
    minicpmo_ws_url: str = "wss://api.modelbest.cn/v1/realtime?mode=audio"
    fs_sample_rate: int = 8000          # FreeSWITCH 采样率
    ai_sample_rate: int = 16000         # MiniCPM-o 采样率
    frame_duration_ms: int = 20         # 帧时长
    silence_timeout_ms: int = 800      # 停顿检测
    max_call_duration_sec: int = 600   # 通话超时

CONFIG = Config()

# ---------- 音频工具 ----------
class AudioUtils:
    """纯 numpy 音频处理"""

    @staticmethod
    def pcmu_decode(data: bytes) -> np.ndarray:
        """PCMU 解码为 float32 [-1, 1]"""
        bias = 33
        exp_lut = [0, 132, 396, 924, 1980, 4092, 8316, 16764]
        out = np.zeros(len(data), dtype=np.float32)
        for i, b in enumerate(data):
            b = ~b & 0xFF
            sign = (b & 0x80) != 0
            exp = (b >> 4) & 0x07
            mant = b & 0x0F
            sample = exp_lut[exp] + (mant << (exp + 3))
            if sign:
                sample = -sample
            out[i] = (sample - bias) / 32768.0
        return out

    @staticmethod
    def pcmu_encode(data: np.ndarray) -> bytes:
        """float32 [-1, 1] 编码为 PCMU"""
        data = np.clip(data, -1.0, 1.0)
        samples = (data * 32768).astype(np.int16)
        out = bytearray()
        for s in samples:
            sign = 0x80 if s < 0 else 0
            s = abs(int(s))
            if s < 256:
                enc = s >> 4
            elif s < 512:
                enc = 0x10 | ((s - 256) >> 5)
            elif s < 1024:
                enc = 0x20 | ((s - 512) >> 6)
            elif s < 2048:
                enc = 0x30 | ((s - 1024) >> 7)
            elif s < 4096:
                enc = 0x40 | ((s - 2048) >> 8)
            elif s < 8192:
                enc = 0x50 | ((s - 4096) >> 9)
            elif s < 16384:
                enc = 0x60 | ((s - 8192) >> 10)
            else:
                enc = 0x70 | ((s - 16384) >> 11)
            out.append(~(sign | enc) & 0xFF)
        return bytes(out)

    @staticmethod
    def resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """线性插值重采样"""
        if orig_sr == target_sr:
            return data
        ratio = target_sr / orig_sr
        new_len = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, new_len)
        return np.interp(indices, np.arange(len(data)), data).astype(np.float32)

    @staticmethod
    def pcmu_to_int16(data: bytes) -> np.ndarray:
        """PCMU 转 int16（用于 webrtcvad）"""
        return (AudioUtils.pcmu_decode(data) * 32767).astype(np.int16)

# ---------- RTP 收发 ----------
class RTPEndpoint:
    """RTP 端点：收发音频"""

    def __init__(self, port: int):
        self.port = port
        self.remote_addr: Optional[Tuple[str, int]] = None
        self.rx_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.tx_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.seq = 0
        self.ts = 0
        self.ssrc = int(time.time() * 1000) & 0xFFFFFFFF
        self.transport = None
        self.running = False

    async def start(self):
        loop = asyncio.get_event_loop()
        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: RTPProtocol(self),
            local_addr=('0.0.0.0', self.port)
        )
        self.running = True
        asyncio.create_task(self._send_loop())
        logger.info(f"RTP listening on port {self.port}")

    def set_remote(self, addr: Tuple[str, int]):
        self.remote_addr = addr
        logger.info(f"RTP remote set to {addr}")

    async def _send_loop(self):
        while self.running:
            try:
                chunk = await asyncio.wait_for(self.tx_queue.get(), timeout=0.1)
                if self.remote_addr:
                    self.seq = (self.seq + 1) & 0xFFFF
                    self.ts += int(8000 * 0.02)
                    header = struct.pack('!BBHII', 0x80, 0, self.seq, self.ts, self.ssrc)
                    self.transport.sendto(header + chunk, self.remote_addr)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"RTP send error: {e}")

    def send_audio(self, pcmu: bytes):
        """发送 PCMU 音频（自动分 20ms 帧）"""
        frame_size = 160  # 20ms @ 8kHz
        for i in range(0, len(pcmu), frame_size):
            chunk = pcmu[i:i + frame_size]
            if len(chunk) < frame_size:
                chunk = chunk.ljust(frame_size, b'\x00')
            try:
                self.tx_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass

    def stop(self):
        self.running = False
        if self.transport:
            self.transport.close()

class RTPProtocol(asyncio.DatagramProtocol):
    def __init__(self, endpoint: RTPEndpoint):
        self.endpoint = endpoint

    def datagram_received(self, data, addr):
        if not self.endpoint.remote_addr:
            self.endpoint.set_remote(addr)
        if len(data) > 12:
            try:
                self.endpoint.rx_queue.put_nowait(data[12:])
            except asyncio.QueueFull:
                pass

# ---------- VAD ----------
class VAD:
    """语音活动检测"""

    def __init__(self):
        self.vad = webrtcvad.Vad(2) if HAS_VAD else None
        self.buf = []
        self.silence = 0
        self.max_silence = int(CONFIG.silence_timeout_ms / CONFIG.frame_duration_ms)
        self.speech = False

    def process(self, frame: bytes) -> Tuple[bool, bytes]:
        """
        处理一帧 PCMU 音频
        返回: (is_sentence_end, audio_bytes)
        """
        if not self.vad:
            # 无 webrtcvad 时简单能量检测
            pcm = AudioUtils.pcmu_decode(frame)
            energy = np.sqrt(np.mean(pcm ** 2))
            is_speech = energy > 0.02
        else:
            pcm = AudioUtils.pcmu_to_int16(frame)
            is_speech = self.vad.is_speech(pcm.tobytes(), 8000)

        if is_speech:
            self.silence = 0
            if not self.speech:
                self.speech = True
                self.buf = []
            self.buf.append(frame)
            return False, b''
        else:
            if self.speech:
                self.silence += 1
                self.buf.append(frame)
                if self.silence >= self.max_silence:
                    self.speech = False
                    audio = b''.join(self.buf)
                    self.buf = []
                    return True, audio
            return False, b''

    def reset(self):
        self.speech = False
        self.silence = 0
        self.buf = []

# ---------- 通话会话 ----------
@dataclass
class CallSession:
    uuid: str
    dialog: aiosip.Dialog
    rtp: RTPEndpoint
    ws: Optional[websockets.WebSocketClientProtocol] = None
    is_active: bool = False
    is_ai_speaking: bool = False
    is_user_speaking: bool = False
    vad: VAD = field(default_factory=VAD)
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

# ---------- SIP 应用 ----------
class AIBot:
    def __init__(self):
        self.app = aiosip.Application()
        self.sessions: Dict[str, CallSession] = {}
        self.running = False

    def _get_local_ip(self) -> str:
        """获取本机 IP（用于 SDP）"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((CONFIG.freeswitch_ip, 1))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def _get_free_port(self) -> int:
        """获取空闲 UDP 端口"""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        return port

    async def start(self):
        self.app.router.add_route('INVITE', '/', self.on_invite)
        self.app.router.add_route('BYE', '/', self.on_bye)
        self.app.router.add_route('ACK', '/', self.on_ack)
        self.app.router.add_route('CANCEL', '/', self.on_cancel)

        await self.app.start(local_addr=(CONFIG.local_ip, CONFIG.local_sip_port))
        self.running = True
        logger.info(f"SIP listening on {CONFIG.local_ip}:{CONFIG.local_sip_port}")

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass

    async def on_invite(self, msg: aiosip.Message):
        """处理 INVITE"""
        logger.info(f"Received INVITE from {msg.from_details}")

        # 解析 SDP
        sdp = msg.payload
        remote_ip = None
        remote_port = None
        for line in sdp.split('\n'):
            line = line.strip()
            if line.startswith('c=IN IP4 '):
                remote_ip = line.split()[-1]
            elif line.startswith('m=audio '):
                remote_port = int(line.split()[1])

        if not remote_ip or not remote_port:
            logger.error("Invalid SDP")
            await msg.create_response(400).send()
            return

        # 动态分配 RTP 端口
        local_rtp_port = self._get_free_port()

        # 创建 RTP 端点
        rtp = RTPEndpoint(local_rtp_port)
        await rtp.start()
        rtp.set_remote((remote_ip, remote_port))

        # 创建 Dialog
        dialog = self.app.dialog_from_request(msg)

        # 生成本地 SDP
        local_ip = self._get_local_ip()
        local_sdp = f"""v=0
o=- 0 0 IN IP4 {local_ip}
s=-
c=IN IP4 {local_ip}
t=0 0
m=audio {local_rtp_port} RTP/AVP 0
a=rtpmap:0 PCMU/8000
a=sendrecv
"""
        await dialog.reply(msg, 200, payload=local_sdp)

        call_id = f"call_{int(time.time() * 1000)}"
        logger.info(f"Call {call_id} accepted, RTP -> {remote_ip}:{remote_port}")

        # 创建会话
        session = CallSession(uuid=call_id, dialog=dialog, rtp=rtp)
        session.is_active = True
        self.sessions[call_id] = session

        # 启动 AI 处理
        asyncio.create_task(self.handle_call(session))

    async def on_bye(self, msg: aiosip.Message):
        """处理 BYE"""
        logger.info("Received BYE")
        dialog = self.app.dialog_from_request(msg)
        await dialog.reply(msg, 200)

        # 查找并结束会话
        for uuid, session in list(self.sessions.items()):
            if session.dialog == dialog:
                await self.end_session(session)
                break

    async def on_ack(self, msg: aiosip.Message):
        logger.info("Received ACK")

    async def on_cancel(self, msg: aiosip.Message):
        logger.info("Received CANCEL")
        dialog = self.app.dialog_from_request(msg)
        await dialog.reply(msg, 200)

    async def handle_call(self, session: CallSession):
        """处理通话：连接 MiniCPM-o，启动音频处理"""
        try:
            # 连接 MiniCPM-o
            ws = await self._connect_minicpmo()
            if not ws:
                logger.error("Failed to connect MiniCPM-o")
                await self.end_session(session)
                return

            session.ws = ws

            # 初始化会话
            if not await self._init_minicpmo_session(ws):
                await self.end_session(session)
                return

            # 发送问候语
            await self._send_text_to_ai(session, "你好，我是XX公司的AI客服，请问有什么可以帮您？")

            # 启动三个协程
            await asyncio.gather(
                self._read_loop(session),
                self._write_loop(session),
                self._ws_handler(session),
                self._monitor(session)
            )

        except Exception as e:
            logger.error(f"Call handling error: {e}", exc_info=True)
        finally:
            await self.end_session(session)

    async def _connect_minicpmo(self, max_retries: int = 3):
        """连接 MiniCPM-o WebSocket"""
        for attempt in range(max_retries):
            try:
                ws = await websockets.connect(
                    CONFIG.minicpmo_ws_url,
                    extra_headers={"Authorization": f"Bearer {CONFIG.minicpmo_api_key}"},
                    ping_interval=20,
                    ping_timeout=10
                )
                logger.info("MiniCPM-o connected")
                return ws
            except Exception as e:
                logger.warning(f"Connection failed ({attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        return None

    async def _init_minicpmo_session(self, ws) -> bool:
        """初始化 MiniCPM-o 会话"""
        try:
            # 等待 session.created
            async for msg in ws:
                data = json.loads(msg)
                if data.get("type") == "session.created":
                    break
                elif data.get("type") == "error":
                    logger.error(f"Session error: {data}")
                    return False

            # 发送配置
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": KNOWLEDGE_BASE,
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "language": "zh"
                    }
                }
            }))

            # 等待确认
            async for msg in ws:
                data = json.loads(msg)
                if data.get("type") == "session.updated":
                    logger.info("MiniCPM-o session initialized")
                    return True
                elif data.get("type") == "error":
                    logger.error(f"Update error: {data}")
                    return False

        except Exception as e:
            logger.error(f"Init error: {e}")
        return False

    async def _read_loop(self, session: CallSession):
        """从 RTP 读取音频 → VAD → 发送给 MiniCPM-o"""
        while session.is_active:
            try:
                frame = await asyncio.wait_for(session.rtp.rx_queue.get(), timeout=0.1)

                # VAD 检测
                end, audio = session.vad.process(frame)

                if audio:
                    # 打断检测
                    if session.is_ai_speaking:
                        await self._interrupt_ai(session)
                        session.is_user_speaking = True

                    # 转换音频格式：PCMU 8kHz → PCM 16kHz
                    await self._send_pcmu_to_ai(session, audio)
                    session.last_activity = time.time()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Read loop error: {e}")
                await asyncio.sleep(0.01)

    async def _send_pcmu_to_ai(self, session: CallSession, pcmu: bytes):
        """将 PCMU 音频转换为 PCM 16kHz 发送给 MiniCPM-o"""
        if not session.ws or not session.ws.open:
            return

        # PCMU (8kHz) → float32 → 重采样 16kHz → int16
        pcm_float_8k = AudioUtils.pcmu_decode(pcmu)
        pcm_float_16k = AudioUtils.resample(pcm_float_8k, 8000, 16000)
        pcm_int16 = (pcm_float_16k * 32767).astype(np.int16)

        # base64 编码
        audio_b64 = base64.b64encode(pcm_int16.tobytes()).decode()

        await session.ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }))
        await session.ws.send(json.dumps({
            "type": "input_audio_buffer.commit"
        }))

        logger.debug(f"Sent {len(pcm_int16)} samples to AI")

    async def _write_loop(self, session: CallSession):
        """从 MiniCPM-o 接收音频 → 处理 → 发送 RTP"""
        if not hasattr(session, '_tts_queue'):
            session._tts_queue = asyncio.Queue()

        while session.is_active:
            try:
                audio_b64 = await asyncio.wait_for(
                    session._tts_queue.get(), timeout=0.1
                )

                if not audio_b64:
                    continue

                # 如果用户正在说话，跳过播放
                if session.is_user_speaking:
                    logger.debug("User speaking, skipping TTS")
                    continue

                # 解码：PCM 16kHz int16 → float32
                pcm_int16 = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                pcm_float_16k = pcm_int16.astype(np.float32) / 32768.0

                # 重采样 16kHz → 8kHz
                pcm_float_8k = AudioUtils.resample(pcm_float_16k, 16000, 8000)

                # 编码为 PCMU
                pcmu = AudioUtils.pcmu_encode(pcm_float_8k)

                # 发送 RTP
                session.rtp.send_audio(pcmu)

                session.is_ai_speaking = True
                await asyncio.sleep(len(pcm_float_8k) / 8000)
                session.is_ai_speaking = False

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Write loop error: {e}")
                session.is_ai_speaking = False

    async def _ws_handler(self, session: CallSession):
        """处理 MiniCPM-o WebSocket 消息"""
        ws = session.ws

        if not hasattr(session, '_tts_queue'):
            session._tts_queue = asyncio.Queue()

        try:
            async for msg in ws:
                if not session.is_active:
                    break

                data = json.loads(msg)
                msg_type = data.get("type", "")

                if msg_type == "response.output_audio.delta":
                    audio_b64 = data.get("delta", "")
                    if audio_b64:
                        await session._tts_queue.put(audio_b64)

                elif msg_type == "response.output_audio.done":
                    session.is_ai_speaking = False

                elif msg_type == "response.text.delta":
                    logger.info(f"AI: {data.get('delta', '')}")

                elif msg_type == "input_audio_buffer.speech_started":
                    if session.is_ai_speaking:
                        await self._interrupt_ai(session)

                elif msg_type == "conversation.item.input_audio_transcription.completed":
                    logger.info(f"User: {data.get('transcript', '')}")

                elif msg_type == "error":
                    logger.error(f"MiniCPM-o error: {data}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("MiniCPM-o connection closed")
        except Exception as e:
            logger.error(f"WS handler error: {e}")

    async def _monitor(self, session: CallSession):
        """监控通话状态"""
        while session.is_active:
            await asyncio.sleep(5)

            duration = time.time() - session.start_time
            if duration > CONFIG.max_call_duration_sec:
                logger.info("Call timeout")
                await self._send_text_to_ai(session, "感谢您的来电，再见")
                await asyncio.sleep(3)
                await self.end_session(session)
                break

            if time.time() - session.last_activity > 30:
                logger.info("No activity for 30s")
                await self._send_text_to_ai(session, "请问您还在吗？")
                session.last_activity = time.time()

    async def _send_text_to_ai(self, session: CallSession, text: str):
        """发送文本给 MiniCPM-o"""
        if not session.ws or not session.ws.open:
            return

        await session.ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        }))
        await session.ws.send(json.dumps({"type": "response.create"}))

    async def _interrupt_ai(self, session: CallSession):
        """打断 AI 说话"""
        if not session.is_ai_speaking:
            return

        logger.info("Interrupting AI")
        session.is_ai_speaking = False
        session.is_user_speaking = True

        # 清空 RTP 发送队列
        while not session.rtp.tx_queue.empty():
            try:
                session.rtp.tx_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # 通知 MiniCPM-o
        if session.ws and session.ws.open:
            await session.ws.send(json.dumps({
                "type": "input_audio_buffer.clear"
            }))

        # 200ms 后恢复
        await asyncio.sleep(0.2)
        session.is_user_speaking = False

    async def end_session(self, session: CallSession):
        """结束会话"""
        if not session.is_active:
            return

        logger.info(f"Ending session {session.uuid}")
        session.is_active = False

        # 停止 RTP
        session.rtp.stop()

        # 关闭 WebSocket
        if session.ws:
            try:
                await session.ws.close()
            except:
                pass

        # 从字典移除
        if session.uuid in self.sessions:
            del self.sessions[session.uuid]

    async def stop(self):
        """停止机器人"""
        logger.info("Stopping bot...")
        self.running = False

        for session in list(self.sessions.values()):
            await self.end_session(session)

        if self.app:
            await self.app.stop()

        logger.info("Bot stopped")


# ---------- 启动 ----------
if __name__ == '__main__':
    bot = AIBot()
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        asyncio.run(bot.stop())
