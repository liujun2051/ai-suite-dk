#!/usr/bin/env python3
"""
AI Outbound Call Bot - aiosip + asyncio RTP + MiniCPM-o 4.5
Production-ready pure Python implementation

Dependencies:
    pip install aiosip websockets numpy webrtcvad

FreeSWITCH Setup:
    1. Configure a SIP user for the bot in directory
    2. Create dialplan to route calls to the bot
    3. Or use ESL originate to bridge calls to the bot
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
from enum import Enum, auto

import numpy as np
import websockets

try:
    import aiosip
except ImportError:
    raise ImportError("pip install aiosip")

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False
    print("警告: pip install webrtcvad 以获得更好的 VAD 效果")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== 配置 ==========
@dataclass
class Config:
    # SIP
    local_ip: str = "0.0.0.0"
    local_sip_port: int = 5066
    local_rtp_port: int = 5004
    freeswitch_ip: str = "192.168.1.100"
    freeswitch_sip_port: int = 5060
    sip_username: str = "aibot"
    sip_password: str = "bot_password"

    # MiniCPM-o
    minicpmo_api_key: str = "你的_MiniCPM-o_API_Key"
    minicpmo_ws_url: str = "wss://api.modelbest.cn/v1/realtime?mode=audio"

    # 音频
    fs_sample_rate: int = 8000
    ai_sample_rate: int = 16000
    frame_duration_ms: int = 20
    vad_aggressiveness: int = 2
    silence_timeout_ms: int = 800

    # 业务
    max_call_duration_sec: int = 600

CONFIG = Config()

# ========== 知识库 ==========
KNOWLEDGE_BASE = """
你是XX公司的专业客服代表，请严格遵循以下信息：

【公司介绍】
我们专注于企业级AI语音解决方案，核心产品MiniCPM-o系列全双工语音模型。

【产品优势】
1. 原生全双工：用户可随时打断，实时响应
2. 声音克隆：任意音色克隆，低于0.6秒延迟
3. 中文表现：语音识别错误率仅0.86%

【常见问题】
- 价格：基础版月费XXX元，企业版请咨询销售
- 技术参数：支持WebSocket API，8192上下文
- 售后：7x24小时技术支持

【话术要求】
- 语气亲切专业，像真人一样自然
- 客户问不知道的问题，诚实告知并记录
- 不得虚构产品功能
- 每次回复控制在30秒以内
- 如果客户说"不需要"、"挂了"、"拜拜"等，礼貌结束对话
"""

# ========== 音频工具类 ==========
class AudioUtils:
    """纯 numpy 音频处理，零外部依赖"""

    MULAW_BIAS = 33
    EXP_LUT = [0, 132, 396, 924, 1980, 4092, 8316, 16764]

    @classmethod
    def pcmu_decode(cls, data: bytes) -> np.ndarray:
        """PCMU 解码为 float32 [-1, 1]"""
        result = np.zeros(len(data), dtype=np.float32)
        for i, byte in enumerate(data):
            byte = ~byte & 0xFF
            sign = (byte & 0x80)
            exponent = (byte >> 4) & 0x07
            mantissa = byte & 0x0F
            sample = cls.EXP_LUT[exponent] + (mantissa << (exponent + 3))
            if sign:
                sample = -sample
            result[i] = (sample - cls.MULAW_BIAS) / 32768.0
        return result

    @classmethod
    def pcmu_encode(cls, data: np.ndarray) -> bytes:
        """float32 [-1, 1] 编码为 PCMU"""
        data = np.clip(data, -1.0, 1.0)
        samples = (data * 32768).astype(np.int16)
        result = bytearray(len(samples))
        for i, sample in enumerate(samples):
            sign = 0x80 if sample < 0 else 0
            sample = abs(int(sample))
            if sample < 256:
                encoded = sample >> 4
            elif sample < 512:
                encoded = 0x10 | ((sample - 256) >> 5)
            elif sample < 1024:
                encoded = 0x20 | ((sample - 512) >> 6)
            elif sample < 2048:
                encoded = 0x30 | ((sample - 1024) >> 7)
            elif sample < 4096:
                encoded = 0x40 | ((sample - 2048) >> 8)
            elif sample < 8192:
                encoded = 0x50 | ((sample - 4096) >> 9)
            elif sample < 16384:
                encoded = 0x60 | ((sample - 8192) >> 10)
            else:
                encoded = 0x70 | ((sample - 16384) >> 11)
            result[i] = ~(sign | encoded) & 0xFF
        return bytes(result)

    @classmethod
    def resample(cls, data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """线性插值重采样"""
        if orig_sr == target_sr:
            return data
        ratio = target_sr / orig_sr
        new_length = int(len(data) * ratio)
        old_indices = np.linspace(0, len(data) - 1, new_length)
        indices = old_indices.astype(np.int32)
        frac = old_indices - indices
        result = np.zeros(new_length, dtype=np.float32)
        for i in range(new_length):
            idx = min(indices[i], len(data) - 2)
            result[i] = data[idx] * (1 - frac[i]) + data[idx + 1] * frac[i]
        return result

    @classmethod
    def float_to_int16(cls, data: np.ndarray) -> np.ndarray:
        return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)

    @classmethod
    def int16_to_float(cls, data: np.ndarray) -> np.ndarray:
        return data.astype(np.float32) / 32768.0

    @classmethod
    def pcmu_to_int16(cls, pcmu: bytes) -> np.ndarray:
        """PCMU 直接转 int16（用于 webrtcvad）"""
        return (cls.pcmu_decode(pcmu) * 32767).astype(np.int16)

# ========== VAD 状态机 ==========
class VADState(Enum):
    SILENCE = auto()
    SPEECH = auto()
    SPEECH_END = auto()

class VADDetector:
    """VAD 检测器，支持 webrtcvad + 能量阈值"""

    def __init__(self, sample_rate: int = 8000, frame_duration_ms: int = 20):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        self.vad = None
        if WEBRTC_VAD_AVAILABLE and sample_rate in (8000, 16000, 32000, 48000):
            self.vad = webrtcvad.Vad(CONFIG.vad_aggressiveness)

        self.energy_threshold = 0.015
        self.state = VADState.SILENCE
        self.silence_frames = 0
        self.speech_frames = 0
        self.max_silence_frames = int(CONFIG.silence_timeout_ms / frame_duration_ms)
        self.min_speech_frames = 3

        self.audio_buffer: deque = deque()
        self.pre_buffer: deque = deque(maxlen=int(300 / frame_duration_ms))

    def process(self, pcmu_frame: bytes) -> tuple:
        """
        处理 PCMU 音频帧
        返回: (state, is_speech, is_sentence_end)
        """
        # 解码为 int16 用于 VAD
        pcm_int16 = AudioUtils.pcmu_to_int16(pcmu_frame)

        # 确保帧大小正确（webrtcvad 需要特定帧大小）
        if len(pcm_int16) < self.frame_size:
            pcm_int16 = np.pad(pcm_int16, (0, self.frame_size - len(pcm_int16)))

        is_speech = False

        # webrtcvad 检测
        if self.vad:
            try:
                # webrtcvad 需要 10/20/30ms 帧
                frame_bytes = pcm_int16[:self.frame_size].tobytes()
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            except Exception as e:
                logger.debug(f"VAD error: {e}")

        # 能量阈值备用
        if not is_speech:
            pcm_float = AudioUtils.pcmu_decode(pcmu_frame)
            energy = np.sqrt(np.mean(pcm_float ** 2))
            is_speech = energy > self.energy_threshold

        # 状态机
        if is_speech:
            self.silence_frames = 0
            self.speech_frames += 1

            if self.state == VADState.SILENCE and self.speech_frames >= self.min_speech_frames:
                self.state = VADState.SPEECH
                self.audio_buffer.extend(self.pre_buffer)
                self.pre_buffer.clear()
                return VADState.SPEECH, True, False

            if self.state == VADState.SPEECH:
                self.audio_buffer.append(pcmu_frame)
                return VADState.SPEECH, True, False

            self.pre_buffer.append(pcmu_frame)
            return VADState.SILENCE, False, False

        else:
            self.speech_frames = 0
            self.silence_frames += 1

            if self.state == VADState.SPEECH:
                self.audio_buffer.append(pcmu_frame)
                if self.silence_frames >= self.max_silence_frames:
                    self.state = VADState.SPEECH_END
                    return VADState.SPEECH_END, False, True
                return VADState.SPEECH, False, False

            if len(self.pre_buffer) < self.pre_buffer.maxlen:
                self.pre_buffer.append(pcmu_frame)

            self.state = VADState.SILENCE
            return VADState.SILENCE, False, False

    def get_buffered_audio(self) -> bytes:
        """获取缓冲的 PCMU 音频并清空"""
        if not self.audio_buffer:
            return b''
        result = b''.join(self.audio_buffer)
        self.audio_buffer.clear()
        self.state = VADState.SILENCE
        self.silence_frames = 0
        return result

    def reset(self):
        self.state = VADState.SILENCE
        self.silence_frames = 0
        self.speech_frames = 0
        self.audio_buffer.clear()
        self.pre_buffer.clear()

# ========== RTP 协议实现 ==========
class RTPProtocol(asyncio.DatagramProtocol):
    """RTP 接收协议"""

    def __init__(self, rx_queue: asyncio.Queue, on_remote_addr):
        self.rx_queue = rx_queue
        self.on_remote_addr = on_remote_addr
        self.remote_addr = None

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        if not self.remote_addr:
            self.remote_addr = addr
            self.on_remote_addr(addr)

        if len(data) < 12:
            return

        # 解析 RTP 头
        payload = data[12:]
        try:
            self.rx_queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # 丢弃旧帧

class RTPSession:
    """完整的 RTP 收发会话"""

    def __init__(self, local_port: int):
        self.local_port = local_port
        self.remote_addr: Optional[Tuple[str, int]] = None
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.protocol: Optional[RTPProtocol] = None

        # RTP 序列号和时间戳
        self.sequence = 0
        self.timestamp = 0
        self.ssrc = int(time.time() * 1000) & 0xFFFFFFFF

        # 队列
        self.rx_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1000)
        self.tx_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1000)

        self.running = False
        self._tx_task = None

    def _on_remote_addr(self, addr: Tuple[str, int]):
        """发现远端 RTP 地址"""
        self.remote_addr = addr
        logger.info(f"RTP remote address discovered: {addr}")

    async def start(self):
        """启动 RTP 会话"""
        loop = asyncio.get_event_loop()
        self.transport, self.protocol = await loop.create_datagram_endpoint(
            lambda: RTPProtocol(self.rx_queue, self._on_remote_addr),
            local_addr=(CONFIG.local_ip, self.local_port)
        )
        self.running = True
        self._tx_task = asyncio.create_task(self._transmit_loop())
        logger.info(f"RTP session started on port {self.local_port}")

    async def _transmit_loop(self):
        """发送 RTP 包"""
        while self.running:
            try:
                pcmu_chunk = await asyncio.wait_for(self.tx_queue.get(), timeout=0.1)

                if not self.remote_addr:
                    continue

                # 构建 RTP 头
                self.sequence = (self.sequence + 1) & 0xFFFF
                self.timestamp += int(CONFIG.fs_sample_rate * CONFIG.frame_duration_ms / 1000)

                header = struct.pack('!BBHII',
                    0x80,           # V=2, P=0, X=0, CC=0
                    0x00,           # M=0, PT=0 (PCMU)
                    self.sequence,
                    self.timestamp,
                    self.ssrc
                )

                self.transport.sendto(header + pcmu_chunk, self.remote_addr)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"RTP transmit error: {e}")
                await asyncio.sleep(0.01)

    async def send_audio(self, pcmu_data: bytes):
        """发送音频（分帧）"""
        frame_size = int(CONFIG.fs_sample_rate * CONFIG.frame_duration_ms / 1000)

        for i in range(0, len(pcmu_data), frame_size):
            chunk = pcmu_data[i:i + frame_size]
            if len(chunk) < frame_size:
                chunk += b'\x00' * (frame_size - len(chunk))

            try:
                self.tx_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass

    def stop(self):
        """停止 RTP 会话"""
        self.running = False
        if self._tx_task:
            self._tx_task.cancel()
        if self.transport:
            self.transport.close()

# ========== SDP 工具 ==========
class SDPUtils:
    """SDP 解析和生成工具"""

    @staticmethod
    def parse_sdp(sdp: str) -> Dict[str, Any]:
        """解析 SDP，提取关键信息"""
        result = {
            'ip': None,
            'rtp_port': None,
            'codec': None,
            'sample_rate': None
        }

        for line in sdp.strip().split('\n'):
            line = line.strip()
            if line.startswith('c=IN IP4 '):
                result['ip'] = line.split()[-1]
            elif line.startswith('m=audio '):
                parts = line.split()
                result['rtp_port'] = int(parts[1])
                result['codec'] = parts[3] if len(parts) > 3 else '0'
            elif line.startswith('a=rtpmap:0 '):
                # PCMU/8000
                parts = line.split('/')
                if len(parts) > 1:
                    result['sample_rate'] = int(parts[1])

        return result

    @staticmethod
    def generate_sdp(local_ip: str, rtp_port: int, codec: str = "PCMU", 
                     sample_rate: int = 8000) -> str:
        """生成 SDP"""
        return f"""v=0
o=- 0 0 IN IP4 {local_ip}
s=AI Bot
c=IN IP4 {local_ip}
t=0 0
m=audio {rtp_port} RTP/AVP 0
a=rtpmap:0 {codec}/{sample_rate}
a=sendrecv
"""

# ========== 通话会话 ==========
@dataclass
class CallSession:
    uuid: str
    phone_number: str
    rtp_session: RTPSession
    dialog: aiosip.Dialog
    ws: Optional[websockets.WebSocketClientProtocol] = None
    is_active: bool = False
    is_ai_speaking: bool = False
    is_user_speaking: bool = False
    vad: VADDetector = field(default_factory=lambda: VADDetector(sample_rate=8000))
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

# ========== 主应用类 ==========
class AIOutboundBot:
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.app: Optional[aiosip.Application] = None
        self.sessions: Dict[str, CallSession] = {}
        self.running = False

    async def start(self):
        """启动 SIP 服务器"""
        self.app = aiosip.Application()

        # 注册路由
        self.app.router.add_route('INVITE', '/', self._on_invite)
        self.app.router.add_route('BYE', '/', self._on_bye)
        self.app.router.add_route('ACK', '/', self._on_ack)
        self.app.router.add_route('CANCEL', '/', self._on_cancel)

        # 启动 UDP 监听
        await self.app.start(
            local_addr=(self.config.local_ip, self.config.local_sip_port)
        )

        self.running = True
        logger.info(f"SIP bot listening on {self.config.local_ip}:{self.config.local_sip_port}")

        # 保持运行
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def _on_invite(self, request: aiosip.Request):
        """处理 INVITE 请求"""
        logger.info(f"Received INVITE from {request.from_details}")

        # 解析 SDP 获取远端 RTP 信息
        sdp_info = SDPUtils.parse_sdp(request.payload)
        remote_rtp_addr = (sdp_info['ip'], sdp_info['rtp_port']) if sdp_info['ip'] else None

        # 创建 RTP 会话
        rtp_session = RTPSession(self.config.local_rtp_port)
        if remote_rtp_addr:
            rtp_session.remote_addr = remote_rtp_addr

        # 创建 Dialog 并发送 200 OK
        dialog = self.app.dialog_from_request(request)

        local_sdp = SDPUtils.generate_sdp(
            self._get_local_ip(),
            self.config.local_rtp_port
        )

        await dialog.reply(request, status_code=200, payload=local_sdp)

        # 创建会话
        call_id = f"call_{int(time.time() * 1000)}"
        session = CallSession(
            uuid=call_id,
            phone_number=str(request.from_details),
            rtp_session=rtp_session,
            dialog=dialog
        )
        session.is_active = True
        self.sessions[call_id] = session

        # 启动处理
        asyncio.create_task(self._handle_call(session))

    async def _on_bye(self, request: aiosip.Request):
        """处理 BYE 请求"""
        logger.info("Received BYE")
        dialog = self.app.dialog_from_request(request)
        await dialog.reply(request, status_code=200)

        # 查找并结束会话
        for session in list(self.sessions.values()):
            if session.dialog == dialog:
                await self._end_session(session)
                break

    async def _on_ack(self, request: aiosip.Request):
        """处理 ACK"""
        logger.info("Received ACK")

    async def _on_cancel(self, request: aiosip.Request):
        """处理 CANCEL"""
        logger.info("Received CANCEL")
        dialog = self.app.dialog_from_request(request)
        await dialog.reply(request, status_code=200)

    def _get_local_ip(self) -> str:
        """获取本机 IP"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((self.config.freeswitch_ip, 1))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    async def _handle_call(self, session: CallSession):
        """处理通话"""
        try:
            # 启动 RTP
            await session.rtp_session.start()

            # 连接 MiniCPM-o
            ws = await self._connect_minicpmo()
            if not ws:
                logger.error("Failed to connect MiniCPM-o")
                await self._end_session(session)
                return

            session.ws = ws

            # 初始化 MiniCPM-o 会话
            if not await self._init_minicpmo_session(ws):
                await self._end_session(session)
                return

            # 发送问候语
            await self._send_text_to_ai(session, "你好，我是XX公司的AI客服，请问有什么可以帮您？")

            # 启动三个协程
            await asyncio.gather(
                self._audio_reader(session),
                self._audio_writer(session),
                self._ws_handler(session),
                self._monitor(session)
            )

        except Exception as e:
            logger.error(f"Call handling error: {e}", exc_info=True)
        finally:
            await self._end_session(session)

    async def _connect_minicpmo(self, max_retries: int = 3):
        """连接 MiniCPM-o"""
        for attempt in range(max_retries):
            try:
                ws = await websockets.connect(
                    self.config.minicpmo_ws_url,
                    extra_headers={"Authorization": f"Bearer {self.config.minicpmo_api_key}"},
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
            async for msg in ws:
                data = json.loads(msg)
                if data.get("type") == "session.created":
                    break
                elif data.get("type") == "error":
                    logger.error(f"Session error: {data}")
                    return False

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

    async def _audio_reader(self, session: CallSession):
        """从 RTP 读取音频 → VAD → 发送给 MiniCPM-o"""
        rtp = session.rtp_session

        while session.is_active and self.running:
            try:
                # 从 RTP 接收队列读取（100ms 超时）
                pcmu_frame = await asyncio.wait_for(rtp.rx_queue.get(), timeout=0.1)

                # VAD 检测
                vad_state, is_speech, is_sentence_end = session.vad.process(pcmu_frame)

                # 打断检测
                if is_speech and session.is_ai_speaking:
                    logger.info("User interrupt detected")
                    await self._interrupt_ai(session)
                    session.is_user_speaking = True

                # 句子结束，发送给 AI
                if is_sentence_end:
                    sentence_pcmu = session.vad.get_buffered_audio()
                    if sentence_pcmu:
                        await self._send_pcmu_to_ai(session, sentence_pcmu)

                # 更新活动时间
                if is_speech:
                    session.last_activity = time.time()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Audio reader error: {e}")
                await asyncio.sleep(0.01)

    async def _send_pcmu_to_ai(self, session: CallSession, pcmu_data: bytes):
        """将 PCMU 音频发送给 MiniCPM-o"""
        if not session.ws or not session.ws.open:
            return

        # PCMU (8kHz) → float32 → 重采样 16kHz → int16
        pcm_float_8k = AudioUtils.pcmu_decode(pcmu_data)
        pcm_float_16k = AudioUtils.resample(pcm_float_8k, 8000, 16000)
        pcm_int16 = AudioUtils.float_to_int16(pcm_float_16k)

        # base64 编码
        audio_b64 = base64.b64encode(pcm_int16.tobytes()).decode()

        # 发送
        await session.ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }))
        await session.ws.send(json.dumps({
            "type": "input_audio_buffer.commit"
        }))

        logger.debug(f"Sent {len(pcm_int16)} samples to AI")

    async def _audio_writer(self, session: CallSession):
        """从 MiniCPM-o 接收音频 → 处理 → 发送 RTP"""
        rtp = session.rtp_session

        if not hasattr(session, '_tts_queue'):
            session._tts_queue = asyncio.Queue()

        while session.is_active and self.running:
            try:
                # 从 TTS 队列获取音频
                audio_b64 = await asyncio.wait_for(
                    session._tts_queue.get(), timeout=0.1
                )

                if not audio_b64:
                    continue

                # 如果用户正在说话，跳过播放
                if session.is_user_speaking:
                    logger.debug("User speaking, skipping TTS")
                    continue

                # 解码 base64 → int16 PCM (16kHz)
                pcm_int16 = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                pcm_float_16k = AudioUtils.int16_to_float(pcm_int16)

                # 重采样 16kHz → 8kHz
                pcm_float_8k = AudioUtils.resample(pcm_float_16k, 16000, 8000)

                # 编码为 PCMU
                pcmu_data = AudioUtils.pcmu_encode(pcm_float_8k)

                # 发送 RTP
                await rtp.send_audio(pcmu_data)

                session.is_ai_speaking = True
                await asyncio.sleep(len(pcm_float_8k) / 8000)  # 等待播放完成
                session.is_ai_speaking = False

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Audio writer error: {e}")
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
        while session.is_active and self.running:
            await asyncio.sleep(5)

            duration = time.time() - session.start_time
            if duration > self.config.max_call_duration_sec:
                logger.info("Call timeout")
                await self._send_text_to_ai(session, "感谢您的来电，再见")
                await asyncio.sleep(3)
                await self._end_session(session)
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

        # 清空 RTP 发送队列（立即停止出声）
        while not session.rtp_session.tx_queue.empty():
            try:
                session.rtp_session.tx_queue.get_nowait()
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

    async def _end_session(self, session: CallSession):
        """结束会话"""
        logger.info(f"Ending session {session.uuid}")
        session.is_active = False

        if session.rtp_session:
            session.rtp_session.stop()

        if session.ws:
            try:
                await session.ws.close()
            except:
                pass

        if session.uuid in self.sessions:
            del self.sessions[session.uuid]

    async def stop(self):
        """停止机器人"""
        logger.info("Stopping bot...")
        self.running = False

        for session in list(self.sessions.values()):
            await self._end_session(session)

        if self.app:
            await self.app.stop()

        logger.info("Bot stopped")


# ========== 启动 ==========
async def main():
    bot = AIOutboundBot()

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
