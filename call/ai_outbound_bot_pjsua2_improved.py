#!/usr/bin/env python3
"""
Complete AI Outbound Call Bot with FreeSWITCH, PJSUA2, and MiniCPM-o 4.5
Production-ready version with audio format conversion, VAD, and interruption handling.

Dependencies:
    pip install websockets numpy webrtcvad
    # PJSUA2 must be compiled and installed separately

FreeSWITCH Setup:
    1. Configure a gateway in /etc/freeswitch/sip_profiles/external/your_gateway.xml
    2. Create a dialplan extension to bridge to the bot's SIP URI
    3. Or use ESL originate to connect the call to the bot
"""

import asyncio
import json
import base64
import logging
import threading
import queue
import time
import os
from typing import Optional, Deque
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto

import numpy as np
import websockets

# 导入 PJSUA2
try:
    import pjsua2 as pj
except ImportError:
    raise ImportError(
        "PJSUA2 not found. Please install it first:\n"
        "  git clone https://github.com/pjsip/pjproject.git\n"
        "  cd pjproject && ./configure --enable-shared && make && sudo make install\n"
        "  cd pjsip-apps/src/swig/python && sudo make install"
    )

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== 配置 ==========
@dataclass
class Config:
    """全局配置"""
    # MiniCPM-o
    minicpmo_api_key: str = "你的_MiniCPM-o_API_Key"
    minicpmo_ws_url: str = "wss://api.modelbest.cn/v1/realtime?mode=audio"

    # FreeSWITCH
    freeswitch_ip: str = "192.168.1.100"
    fs_sip_port: int = 5060
    fs_esl_port: int = 8021
    fs_esl_pass: str = "ClueCon"

    # 本机 SIP
    sip_bot_port: int = 5066
    sip_bot_user: str = "aibot"
    sip_bot_domain: str = "192.168.1.100"  # 通常与 FS IP 相同

    # 音频
    fs_sample_rate: int = 8000       # FreeSWITCH/PJSUA2 默认
    ai_sample_rate: int = 16000      # MiniCPM-o 要求
    frame_duration_ms: int = 20      # 20ms 帧
    vad_aggressiveness: int = 2      # webrtcvad 激进程度

    # 业务
    max_call_duration_sec: int = 600
    silence_timeout_ms: int = 800    # 停顿检测

    # 路径
    tts_dir: str = "/tmp/fs_tts"

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
    """音频编解码和重采样工具"""

    # PCMU (u-law) 编解码表
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

# ========== VAD 状态机 ==========
class VADState(Enum):
    SILENCE = auto()
    SPEECH = auto()
    SPEECH_END = auto()

class VADDetector:
    """VAD 检测器"""

    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30):
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

        self.audio_buffer: Deque[np.ndarray] = deque()
        self.pre_buffer: Deque[np.ndarray] = deque(
            maxlen=int(300 / frame_duration_ms)  # 300ms 预缓冲
        )

    def process(self, audio_frame: np.ndarray) -> tuple:
        """返回: (state, is_speech, is_sentence_end)"""
        if len(audio_frame) < self.frame_size:
            audio_frame = np.pad(audio_frame, (0, self.frame_size - len(audio_frame)))

        pcm_int16 = AudioUtils.float_to_int16(audio_frame)
        is_speech = False

        if self.vad:
            try:
                is_speech = self.vad.is_speech(pcm_int16.tobytes(), self.sample_rate)
            except:
                pass

        if not is_speech:
            energy = np.sqrt(np.mean(audio_frame ** 2))
            is_speech = energy > self.energy_threshold

        if is_speech:
            self.silence_frames = 0
            self.speech_frames += 1

            if self.state == VADState.SILENCE and self.speech_frames >= self.min_speech_frames:
                self.state = VADState.SPEECH
                self.audio_buffer.extend(self.pre_buffer)
                self.pre_buffer.clear()
                return VADState.SPEECH, True, False

            if self.state == VADState.SPEECH:
                self.audio_buffer.append(audio_frame)
                return VADState.SPEECH, True, False

            self.pre_buffer.append(audio_frame)
            return VADState.SILENCE, False, False

        else:
            self.speech_frames = 0
            self.silence_frames += 1

            if self.state == VADState.SPEECH:
                self.audio_buffer.append(audio_frame)
                if self.silence_frames >= self.max_silence_frames:
                    self.state = VADState.SPEECH_END
                    return VADState.SPEECH_END, False, True
                return VADState.SPEECH, False, False

            if len(self.pre_buffer) < self.pre_buffer.maxlen:
                self.pre_buffer.append(audio_frame)

            self.state = VADState.SILENCE
            return VADState.SILENCE, False, False

    def get_buffered_audio(self) -> np.ndarray:
        if not self.audio_buffer:
            return np.array([], dtype=np.float32)
        result = np.concatenate(list(self.audio_buffer))
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

# ========== 自定义音频媒体端口 ==========
class BotAudioMediaPort(pj.AudioMediaPort):
    """
    自定义音频端口，用于：
    - onFrameReceived: 从通话接收音频 → 放入 rx_queue
    - onFrameRequested: 向通话发送音频 ← 从 tx_queue 取
    """

    def __init__(self):
        super().__init__()
        self.name = "BotAudioPort"
        self.rx_queue = queue.Queue(maxsize=1000)  # 防止内存溢出
        self.tx_queue = queue.Queue(maxsize=1000)
        self._is_running = True
        self._frame_count = 0

    def onFrameReceived(self, frame: pj.MediaFrame) -> None:
        if frame.buf and self._is_running:
            try:
                self.rx_queue.put_nowait(bytes(frame.buf))
            except queue.Full:
                pass  # 丢弃旧帧

    def onFrameRequested(self, frame: pj.MediaMediaFrame) -> None:
        if not self._is_running:
            frame.buf = b'\x00' * frame.size
            return

        try:
            audio_bytes = self.tx_queue.get_nowait()
            if len(audio_bytes) <= frame.size:
                frame.buf = audio_bytes
                # 如果数据不够，补静音
                if len(audio_bytes) < frame.size:
                    frame.buf += b'\x00' * (frame.size - len(audio_bytes))
            else:
                frame.buf = audio_bytes[:frame.size]
                # 剩余数据放回队列头部
                self.tx_queue.put(audio_bytes[frame.size:])
        except queue.Empty:
            frame.buf = b'\x00' * frame.size

    def stop(self):
        self._is_running = False
        # 清空队列
        while not self.rx_queue.empty():
            try:
                self.rx_queue.get_nowait()
            except queue.Empty:
                break
        while not self.tx_queue.empty():
            try:
                self.tx_queue.get_nowait()
            except queue.Empty:
                break

# ========== PJSUA2 账户和通话处理 ==========
class BotAccount(pj.Account):
    def __init__(self, app):
        super().__init__()
        self.app = app

    def onRegState(self, prm):
        logger.info(f"SIP Registration: {prm.code} - {prm.reason}")

    def onIncomingCall(self, prm):
        logger.info(f"Incoming call from {prm.callId}")
        call = BotCall(self.app, self, prm.callId)
        call.answer()
        self.app.current_call = call

class BotCall(pj.Call):
    def __init__(self, app, acc, call_id):
        super().__init__(acc, call_id)
        self.app = app
        self.custom_media_port: Optional[BotAudioMediaPort] = None

    def onCallMediaState(self, prm):
        logger.info("Call media state changed")
        call_info = self.getInfo()

        for media_idx, media in enumerate(call_info.media):
            if media.type == pj.PJMEDIA_TYPE_AUDIO and media.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                try:
                    call_aud_med = self.getAudioMedia(media_idx)

                    self.custom_media_port = BotAudioMediaPort()
                    media_format = call_aud_med.getFormat()
                    self.custom_media_port.createPort("bot_audio", media_format)

                    # 双向连接
                    call_aud_med.startTransmit(self.custom_media_port)
                    self.custom_media_port.startTransmit(call_aud_med)

                    logger.info(f"Audio connected: {media_format.clockRate}Hz, "
                              f"{media_format.channelCount}ch")

                    # 通知主应用
                    self.app.on_call_established(self.custom_media_port, media_format)
                    break

                except Exception as e:
                    logger.error(f"Media connection failed: {e}")

    def onCallState(self, prm):
        call_info = self.getInfo()
        logger.info(f"Call state: {call_info.stateText}")

        if call_info.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            logger.info("Call disconnected")
            if self.custom_media_port:
                self.custom_media_port.stop()
            self.app.on_call_ended()

# ========== 通话会话 ==========
@dataclass
class CallSession:
    uuid: str
    phone_number: str
    media_port: BotAudioMediaPort
    media_format: pj.MediaFormat
    ws: Optional[Any] = None
    is_active: bool = False
    is_ai_speaking: bool = False
    is_user_speaking: bool = False
    vad: VADDetector = field(default_factory=lambda: VADDetector(sample_rate=16000))
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

# ========== 主应用类 ==========
class AIOutboundBot:
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.lib: Optional[pj.Lib] = None
        self.acc: Optional[BotAccount] = None
        self.current_call: Optional[BotCall] = None
        self.sessions: Dict[str, CallSession] = {}
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        os.makedirs(config.tts_dir, exist_ok=True)

    # --- PJSUA2 初始化 ---
    def init_pjsua(self):
        logger.info("Initializing PJSUA2...")

        ep_cfg = pj.EpConfig()
        ep_cfg.uaConfig.threadCnt = 2
        ep_cfg.uaConfig.mainThreadOnly = False

        self.lib = pj.Lib()
        self.lib.init(ep_cfg)

        # UDP 传输
        transport_cfg = pj.TransportConfig()
        transport_cfg.port = self.config.sip_bot_port
        self.lib.createTransport(pj.PJSIP_TRANSPORT_UDP, transport_cfg)

        self.lib.start()
        logger.info(f"PJSUA2 started on port {self.config.sip_bot_port}")

        # 创建账户（注册到 FreeSWITCH）
        acc_cfg = pj.AccountConfig()
        acc_cfg.idUri = f"sip:{self.config.sip_bot_user}@{self.config.sip_bot_domain}"
        acc_cfg.regConfig.registrarUri = f"sip:{self.config.freeswitch_ip}:{self.config.fs_sip_port}"

        # 如果 FreeSWITCH 需要认证
        # acc_cfg.sipConfig.authCreds.append(pj.AuthCredInfo("digest", "*", "user", 0, "pass"))

        self.acc = BotAccount(self)
        self.acc.create(acc_cfg)
        logger.info(f"Account created: {acc_cfg.idUri}")

    # --- MiniCPM-o 连接 ---
    async def connect_minicpmo(self, max_retries: int = 3) -> Optional[websockets.WebSocketClientProtocol]:
        for attempt in range(max_retries):
            try:
                ws = await websockets.connect(
                    self.config.minicpmo_ws_url,
                    extra_headers={"Authorization": f"Bearer {self.config.minicpmo_api_key}"},
                    ping_interval=20,
                    ping_timeout=10
                )
                logger.info("MiniCPM-o WebSocket connected")
                return ws
            except Exception as e:
                logger.warning(f"Connection failed ({attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        return None

    async def initialize_minicpmo_session(self, ws) -> bool:
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
            logger.error(f"Initialize error: {e}")
        return False

    # --- 通话处理 ---
    def on_call_established(self, media_port: BotAudioMediaPort, media_format: pj.MediaFormat):
        """通话建立回调（PJSUA2 线程）"""
        call_id = f"call_{int(time.time() * 1000)}"

        session = CallSession(
            uuid=call_id,
            phone_number="unknown",
            media_port=media_port,
            media_format=media_format
        )
        session.is_active = True
        self.sessions[call_id] = session

        # 投递到 asyncio 事件循环
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._start_call_processing(session),
                self.loop
            )

    async def _start_call_processing(self, session: CallSession):
        """启动通话处理（asyncio 线程）"""
        try:
            # 连接 MiniCPM-o
            ws = await self.connect_minicpmo()
            if not ws:
                logger.error("Failed to connect MiniCPM-o")
                await self._end_call(session)
                return

            session.ws = ws

            if not await self.initialize_minicpmo_session(ws):
                await self._end_call(session)
                return

            # 启动三个协程
            await asyncio.gather(
                self._audio_reader(session),
                self._audio_writer(session),
                self._ws_handler(session),
                self._monitor(session)
            )

        except Exception as e:
            logger.error(f"Call processing error: {e}", exc_info=True)
        finally:
            await self._cleanup_session(session)

    async def _audio_reader(self, session: CallSession):
        """从 PJSUA2 读取音频 → 处理 → 发送给 MiniCPM-o"""
        media_port = session.media_port
        sample_rate = session.media_format.clockRate

        # 计算帧大小：20ms @ 8kHz = 160 bytes (PCMU)
        frame_size = int(sample_rate * self.config.frame_duration_ms / 1000)

        logger.info(f"Audio reader started: {sample_rate}Hz, frame={frame_size}")

        while session.is_active and self.running:
            try:
                # 从队列读取 PCMU 帧（阻塞，但带超时）
                try:
                    pcmu_frame = media_port.rx_queue.get(timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                # 解码 PCMU → float32
                pcm_float_8k = AudioUtils.pcmu_decode(pcmu_frame)

                # 重采样 8kHz → 16kHz
                pcm_float_16k = AudioUtils.resample(
                    pcm_float_8k, sample_rate, self.config.ai_sample_rate
                )

                # VAD 检测
                vad_state, is_speech, is_sentence_end = session.vad.process(pcm_float_16k)

                # 打断检测
                if is_speech and session.is_ai_speaking:
                    logger.info("User interrupt detected")
                    await self._interrupt_ai(session)
                    session.is_user_speaking = True

                # 句子结束，发送给 AI
                if is_sentence_end:
                    sentence_audio = session.vad.get_buffered_audio()
                    if len(sentence_audio) > 0:
                        await self._send_audio_to_ai(session, sentence_audio)

                # 更新活动时间
                if is_speech:
                    session.last_activity = time.time()

            except Exception as e:
                logger.error(f"Audio reader error: {e}")
                await asyncio.sleep(0.01)

    async def _audio_writer(self, session: CallSession):
        """从 MiniCPM-o 接收音频 → 处理 → 写入 PJSUA2"""
        media_port = session.media_port
        sample_rate = session.media_format.clockRate

        # 帧大小：20ms @ 8kHz
        frame_size = int(sample_rate * self.config.frame_duration_ms / 1000)

        logger.info(f"Audio writer started: {sample_rate}Hz, frame={frame_size}")

        while session.is_active and self.running:
            try:
                # 从 TTS 队列获取音频（由 _ws_handler 填充）
                if not hasattr(session, '_tts_queue'):
                    await asyncio.sleep(0.01)
                    continue

                try:
                    audio_b64 = await asyncio.wait_for(
                        session._tts_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                if not audio_b64:
                    continue

                # 如果用户正在说话，跳过播放
                if session.is_user_speaking:
                    logger.debug("User speaking, skipping TTS")
                    continue

                # 解码 base64 → int16 PCM
                pcm_int16 = np.frombuffer(
                    base64.b64decode(audio_b64), dtype=np.int16
                )
                pcm_float_16k = AudioUtils.int16_to_float(pcm_int16)

                # 重采样 16kHz → 8kHz
                pcm_float_8k = AudioUtils.resample(
                    pcm_float_16k,
                    self.config.ai_sample_rate,
                    sample_rate
                )

                # 编码为 PCMU
                pcmu_data = AudioUtils.pcmu_encode(pcm_float_8k)

                # 分帧写入 PJSUA2 发送队列
                for i in range(0, len(pcmu_data), frame_size):
                    chunk = pcmu_data[i:i + frame_size]
                    if len(chunk) < frame_size:
                        chunk += b'\x00' * (frame_size - len(chunk))

                    try:
                        media_port.tx_queue.put_nowait(chunk)
                    except queue.Full:
                        pass

                session.is_ai_speaking = True
                await asyncio.sleep(len(pcm_float_8k) / sample_rate)  # 等待播放完成
                session.is_ai_speaking = False

            except Exception as e:
                logger.error(f"Audio writer error: {e}")
                session.is_ai_speaking = False
                await asyncio.sleep(0.01)

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
                    text = data.get("delta", "")
                    logger.info(f"AI: {text}")

                elif msg_type == "input_audio_buffer.speech_started":
                    if session.is_ai_speaking:
                        await self._interrupt_ai(session)

                elif msg_type == "conversation.item.input_audio_transcription.completed":
                    transcript = data.get("transcript", "")
                    logger.info(f"User: {transcript}")

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
                await self._end_call(session)
                break

            if time.time() - session.last_activity > 30:
                logger.info("No activity for 30s")
                await self._send_text_to_ai(session, "请问您还在吗？")
                session.last_activity = time.time()

    async def _send_audio_to_ai(self, session: CallSession, audio: np.ndarray):
        """发送音频给 MiniCPM-o"""
        if not session.ws or not session.ws.open:
            return

        pcm_int16 = AudioUtils.float_to_int16(audio)
        audio_b64 = base64.b64encode(pcm_int16.tobytes()).decode()

        await session.ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }))
        await session.ws.send(json.dumps({
            "type": "input_audio_buffer.commit"
        }))

        logger.debug(f"Sent {len(audio)} samples to AI")

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
        """打断 AI"""
        if not session.is_ai_speaking:
            return

        logger.info("Interrupting AI")
        session.is_ai_speaking = False
        session.is_user_speaking = True

        # 清空 PJSUA2 播放队列
        while not session.media_port.tx_queue.empty():
            try:
                session.media_port.tx_queue.get_nowait()
            except queue.Empty:
                break

        # 通知 MiniCPM-o
        if session.ws and session.ws.open:
            await session.ws.send(json.dumps({
                "type": "input_audio_buffer.clear"
            }))

        # 200ms 后恢复用户说话状态
        await asyncio.sleep(0.2)
        session.is_user_speaking = False

    async def _end_call(self, session: CallSession):
        """结束通话"""
        session.is_active = False
        if self.current_call:
            try:
                self.current_call.hangup()
            except:
                pass

    async def _cleanup_session(self, session: CallSession):
        """清理会话"""
        logger.info(f"Cleaning up session {session.uuid}")

        session.is_active = False

        if session.media_port:
            session.media_port.stop()

        if session.ws:
            try:
                await session.ws.close()
            except:
                pass

        if session.uuid in self.sessions:
            del self.sessions[session.uuid]

    def on_call_ended(self):
        """通话结束回调（PJSUA2 线程）"""
        logger.info("Call ended callback")
        if self.current_call:
            self.current_call = None

    # --- 外呼 ---
    def make_outbound_call(self, phone_number: str):
        """发起外呼（同步方法，可从其他线程调用）"""
        if not self.acc:
            logger.error("Account not ready")
            return

        # 通过 FreeSWITCH 网关外呼
        # 方法1：直接呼叫 FreeSWITCH，由 FS 路由到网关
        target_uri = f"sip:{phone_number}@{self.config.freeswitch_ip}"

        # 方法2：如果 FreeSWITCH 配置了拨号规则，可以呼叫特定分机
        # target_uri = f"sip:gateway+{self.config.fs_gateway_name}+{phone_number}@{self.config.freeswitch_ip}"

        call = BotCall(self, self.acc, pj.CallOpParam())
        prm = pj.CallOpParam()
        prm.opt.audioCount = 1
        prm.opt.videoCount = 0

        try:
            call.makeCall(target_uri, prm)
            self.current_call = call
            logger.info(f"Outbound call to {phone_number}")
        except Exception as e:
            logger.error(f"Call failed: {e}")

    # --- 主运行 ---
    def run(self):
        """运行主程序"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # 初始化 PJSUA2
        self.init_pjsua()
        self.running = True

        async def main_async():
            logger.info("Bot ready. Waiting for calls...")

            # 保持运行，处理用户输入
            while self.running:
                try:
                    cmd = await asyncio.to_thread(
                        input, 
                        "Commands: 'call <number>', 'status', 'quit'\n> "
                    )
                    cmd = cmd.strip()

                    if cmd.startswith('call '):
                        _, number = cmd.split(' ', 1)
                        # 在 PJSUA2 线程中执行
                        await asyncio.to_thread(self.make_outbound_call, number)

                    elif cmd == 'status':
                        logger.info(f"Active sessions: {len(self.sessions)}")
                        for uuid, s in self.sessions.items():
                            logger.info(f"  {uuid}: active={s.is_active}, "
                                      f"ai_speaking={s.is_ai_speaking}")

                    elif cmd == 'quit':
                        break

                except Exception as e:
                    logger.error(f"Command error: {e}")

        try:
            self.loop.run_until_complete(main_async())
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.running = False

            # 清理所有会话
            for session in list(self.sessions.values()):
                asyncio.run_coroutine_threadsafe(
                    self._cleanup_session(session), self.loop
                )

            self.loop.run_until_complete(asyncio.sleep(1))
            self.loop.close()

            if self.lib:
                self.lib.destroy()

            logger.info("Bot stopped")

# ========== 启动 ==========
if __name__ == "__main__":
    bot = AIOutboundBot()
    bot.run()
