#!/usr/bin/env python3
"""
========================================================================
                   智能外呼机器人 - FreeSWITCH + MiniCPM-o 4.5
========================================================================

协议: MiniCPM-o Audio Full-Duplex Protocol
文档: https://github.com/OpenBMB/MiniCPM-o-Demo/blob/realtime-protocol/docs/audio-duplex-protocol.md

关键协议要求:
- 上行音频: 16kHz, float32 PCM, base64, 每秒发送 1 秒(16000 samples)
- 下行音频: 24kHz, float32 PCM, base64
- 必须等待 session.queue_done 后才能发 session.update
- 打断使用 force_listen=true
- 不需要 input_audio_buffer.commit

依赖:
    pip install websockets numpy webrtcvad
    # FreeSWITCH 需要 mod_sndfile 支持 wav 格式
    # 需要 ESL 模块
"""
import asyncio
import json
import base64
import os
import time
import threading
import concurrent.futures
import logging
import struct
from typing import Optional, Dict, Any, Callable, Deque
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import numpy as np
import websockets


try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False
    print("警告: webrtcvad 未安装，将使用能量阈值 VAD。安装: pip install webrtcvad")
try:
    from ESL import ESLconnection
except ImportError:
    raise ImportError(
        "ESL 模块未安装。请执行：\n"
        "cd /usr/local/src/freeswitch/libs/esl/python\n"
        "sudo python3 setup.py install"
    )


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
    # FreeSWITCH
    fs_host: str = "127.0.0.1"
    fs_port: int = 8021
    fs_pass: str = "ClueCon"
    fs_sample_rate: int = 8000      # FreeSWITCH 默认 8kHz

    # MiniCPM-o (请替换为你的真实 API 地址和 Key)
    # minicpmo_ws_url: str = "wss://api.modelbest.cn/v1/realtime?mode=audio"
    
    minicpmo_ws_url: str = "wss://minicpmo45.modelbest.cn/v1/realtime?mode=audio"
    minicpmo_api_key: str = ""
    minicpmo_sample_rate_in: int = 16000   # 上行: 16kHz float32
    minicpmo_sample_rate_out: int = 24000  # 下行: 24kHz float32

    # 音频积累参数
    chunk_duration_ms: int = 90         # FreeSWITCH 录音间隔
    send_interval_ms: int = 1000        # 每秒发送 1 秒音频给 MiniCPM-o
    target_samples_per_send: int = 16000  # 16kHz * 1s = 16000 samples
    max_sentence_silence_ms: int = 800
    pre_buffer_ms: int = 300

    # VAD
    vad_aggressiveness: int = 2
    energy_threshold: float = 0.015

    # 路径
    shm_dir: str = "/dev/shm/fs_bot"
    tts_dir: str = "/dev/shm/fs_tts"

    # 业务 (请替换为真实号码和网关)
    caller_id: str = "你的外显号码"
    gateway: str = "dinstar_dag1000"
    max_call_duration_sec: int = 600

CONFIG = Config()

# ========== 知识库 ==========
KNOWLEDGE_BASE = """
你是XX公司的专业客服代表，请严格遵循以下信息：
【公司介绍】
我们专注于企业级AI语音解决方案，核心产品MiniCPM-o系列全双工语音模型。
【话术要求】
- 语气亲切专业，像真人一样自然
- 每次回复控制在30秒以内
- 如果客户说"不需要"、"挂了"、"拜拜"等，礼貌结束对话
"""

# ========== 音频工具 ==========
class AudioUtils:
    """音频处理工具"""

    @staticmethod
    def pcmu_decode(data: bytes) -> np.ndarray:
        """PCMU 解码为 float32 [-1, 1] (仅用于读取 FS 录音)"""
        MULAW_BIAS = 33
        exp_lut = [0, 132, 396, 924, 1980, 4092, 8316, 16764]
        result = np.zeros(len(data), dtype=np.float32)
        for i, byte in enumerate(data):
            byte = ~byte & 0xFF
            sign = (byte & 0x80)
            exponent = (byte >> 4) & 0x07
            mantissa = byte & 0x0F
            sample = exp_lut[exponent] + (mantissa << (exponent + 3))
            if sign:
                sample = -sample
            result[i] = (sample - MULAW_BIAS) / 32768.0
        return result

    @staticmethod
    def resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """线性插值重采样"""
        if orig_sr == target_sr:
            return data
        ratio = target_sr / orig_sr
        new_length = int(len(data) * ratio)
        if new_length == 0:
            return np.array([], dtype=np.float32)
        old_indices = np.linspace(0, len(data) - 1, new_length)
        indices = old_indices.astype(np.int32)
        frac = old_indices - indices
        result = np.zeros(new_length, dtype=np.float32)
        for i in range(new_length):
            idx = min(indices[i], len(data) - 2)
            result[i] = data[idx] * (1 - frac[i]) + data[idx + 1] * frac[i]
        return result

    @staticmethod
    def float_to_int16(data: np.ndarray) -> np.ndarray:
        return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)

    @staticmethod
    def int16_to_float(data: np.ndarray) -> np.ndarray:
        return data.astype(np.float32) / 32768.0

    @staticmethod
    def create_wav_header(data_len: int, sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
        """创建标准 PCM WAV 文件头"""
        byte_rate = sample_rate * channels * bits // 8
        block_align = channels * bits // 8
        header = b'RIFF'
        header += (36 + data_len).to_bytes(4, 'little')
        header += b'WAVE'
        header += b'fmt '
        header += (16).to_bytes(4, 'little')
        header += (1).to_bytes(2, 'little')    # AudioFormat = 1 (PCM)
        header += channels.to_bytes(2, 'little')
        header += sample_rate.to_bytes(4, 'little')
        header += byte_rate.to_bytes(4, 'little')
        header += block_align.to_bytes(2, 'little')
        header += bits.to_bytes(2, 'little')
        header += b'data'
        header += data_len.to_bytes(4, 'little')
        return header

# ========== VAD 检测器 ==========
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
        self.energy_threshold = CONFIG.energy_threshold
        self.state = VADState.SILENCE
        self.silence_frames = 0
        self.speech_frames = 0
        self.max_silence_frames = int(CONFIG.max_sentence_silence_ms / frame_duration_ms)
        self.min_speech_frames = 3
        self.audio_buffer: Deque[np.ndarray] = deque()
        self.pre_buffer: Deque[np.ndarray] = deque(maxlen=int(CONFIG.pre_buffer_ms / frame_duration_ms))

    def _process_single_frame(self, audio_frame: np.ndarray) -> tuple:
        if len(audio_frame) < self.frame_size:
            audio_frame = np.pad(audio_frame, (0, self.frame_size - len(audio_frame)))
        pcm_int16 = AudioUtils.float_to_int16(audio_frame)
        is_speech = False
        if self.vad:
            try:
                is_speech = self.vad.is_speech(pcm_int16.tobytes(), self.sample_rate)
            except:
                is_speech = False
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
            if self.state == VADState.SPEECH:
                self.audio_buffer.append(audio_frame)
            else:
                self.pre_buffer.append(audio_frame)
        else:
            self.speech_frames = 0
            self.silence_frames += 1
            if self.state == VADState.SPEECH:
                self.audio_buffer.append(audio_frame)
                if self.silence_frames >= self.max_silence_frames:
                    self.state = VADState.SPEECH_END
                    return VADState.SPEECH_END, False, True
            else:
                if len(self.pre_buffer) < self.pre_buffer.maxlen:
                    self.pre_buffer.append(audio_frame)
                self.state = VADState.SILENCE
        return self.state, is_speech, False

    def process(self, audio_chunk: np.ndarray) -> tuple:
        final_state = self.state
        has_speech = False
        is_end = False
        for i in range(0, len(audio_chunk), self.frame_size):
            frame = audio_chunk[i:i+self.frame_size]
            if len(frame) == 0:
                continue
            s, speech, end = self._process_single_frame(frame)
            final_state = s
            if speech:
                has_speech = True
            if end:
                is_end = True
        return final_state, has_speech, is_end

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

# ========== 异步 ESL 封装 ==========
class AsyncESL:
    def __init__(self, host: str, port: int, password: str, loop: asyncio.AbstractEventLoop):
        self.host = host
        self.port = port
        self.password = password
        self.conn: Optional[ESLconnection] = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._loop = loop
        self._connected = False
        self._event_callbacks: Dict[str, list] = {}
        self._event_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    async def connect(self) -> bool:
        def _connect():
            conn = ESLconnection(self.host, self.port, self.password)
            if conn.connected():
                conn.events("plain", "ALL")
                return conn
            return None
        self.conn = await self._loop.run_in_executor(self.executor, _connect)
        if self.conn:
            self._connected = True
            self._event_thread = threading.Thread(target=self._event_loop, daemon=True)
            self._event_thread.start()
            logger.info("FreeSWITCH ESL 连接成功")
            return True
        return False

    def _event_loop(self):
        while not self._stop_event.is_set() and self.conn:
            try:
                event = self.conn.recvEventTimed(100)
                if event:
                    event_name = event.getHeader("Event-Name")
                    if event_name in self._event_callbacks:
                        for callback in self._event_callbacks[event_name]:
                            try:
                                callback(event)
                            except Exception as e:
                                logger.error(f"事件回调错误: {e}")
            except Exception as e:
                logger.error(f"事件接收错误: {e}")
                break

    def on_event(self, event_name: str, callback: Callable):
        if event_name not in self._event_callbacks:
            self._event_callbacks[event_name] = []
        self._event_callbacks[event_name].append(callback)

    async def api(self, command: str, arg: str = "") -> Optional[Any]:
        if not self.conn:
            return None
        return await self._loop.run_in_executor(self.executor, self.conn.api, command, arg)

    async def send(self, command: str) -> bool:
        if not self.conn:
            return False
        return await self._loop.run_in_executor(self.executor, self.conn.send, command)

    async def disconnect(self):
        self._stop_event.set()
        if self._event_thread:
            self._event_thread.join(timeout=2)
        if self.conn:
            self.conn.disconnect()
            self.conn = None
        self.executor.shutdown(wait=False)

# ========== 通话会话 ==========
@dataclass
class CallSession:
    uuid: str
    phone_number: str
    ws: Optional[Any] = None
    is_active: bool = False
    is_ai_speaking: bool = False
    is_user_speaking: bool = False
    vad: VADDetector = field(default_factory=lambda: VADDetector(sample_rate=16000))
    sentence_audio: Deque[np.ndarray] = field(default_factory=deque)
    pending_sentence: bool = False
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    total_turns: int = 0
    ai_talk_time: float = 0.0
    user_talk_time: float = 0.0
    tts_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # 音频积累器: 积累 FreeSWITCH 录音到 1 秒再发送
    audio_accumulator: Deque[np.ndarray] = field(default_factory=deque)
    accumulated_samples: int = 0

# ========== 主机器人类 ==========
class OutboundCallBot:
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.esl: Optional[AsyncESL] = None
        self.sessions: Dict[str, CallSession] = {}
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        os.makedirs(config.shm_dir, exist_ok=True)
        os.makedirs(config.tts_dir, exist_ok=True)

    async def initialize(self) -> bool:
        self.loop = asyncio.get_event_loop()
        self.esl = AsyncESL(self.config.fs_host, self.config.fs_port, self.config.fs_pass, self.loop)
        if not await self.esl.connect():
            logger.error("无法连接 FreeSWITCH")
            return False
        self.esl.on_event("CHANNEL_ANSWER", self._on_channel_answer)
        self.esl.on_event("CHANNEL_HANGUP", self._on_channel_hangup)
        self.running = True
        logger.info("机器人初始化完成")
        return True

    def _on_channel_answer(self, event):
        uuid = event.getHeader("Unique-ID")
        phone = event.getHeader("Caller-Destination-Number")
        logger.info(f"用户接听: UUID={uuid}, 号码={phone}")
        if uuid not in self.sessions:
            self.sessions[uuid] = CallSession(uuid=uuid, phone_number=phone)
        self.sessions[uuid].is_active = True
        asyncio.run_coroutine_threadsafe(self._handle_call(uuid), self.loop)

    def _on_channel_hangup(self, event):
        uuid = event.getHeader("Unique-ID")
        cause = event.getHeader("Hangup-Cause")
        logger.info(f"用户挂断: UUID={uuid}, 原因={cause}")
        if uuid in self.sessions:
            self.sessions[uuid].is_active = False

    async def originate_call(self, phone_number: str) -> Optional[str]:
        cmd = (
            f"{{origination_caller_id_number={self.config.caller_id}}}"
            f"sofia/gateway/{self.config.gateway}/{phone_number} "
            f"&park()"
        )
        res = await self.esl.api("originate", cmd)
        if res:
            reply = res.getBody() if hasattr(res, 'getBody') else str(res)
            if "+OK" in reply:
                return reply.strip().split()[-1]
        return None

    async def _handle_call(self, uuid: str):
        session = self.sessions.get(uuid)
        if not session:
            return
        try:
            ws = await self._connect_minicpmo()
            if not ws:
                await self._hangup(uuid, "CONNECT_FAILED")
                return
            session.ws = ws
            if not await self._initialize_minicpmo_session(ws):
                await self._hangup(uuid, "SESSION_FAILED")
                return
            await self._send_greeting(session)
            await self._main_loop(session)
        except Exception as e:
            logger.error(f"通话处理异常: {e}", exc_info=True)
        finally:
            await self._cleanup_session(uuid)

    async def _connect_minicpmo(self, max_retries: int = 3) -> Optional[Any]:
        for attempt in range(max_retries):
            try:
                ws = await websockets.connect(
                    self.config.minicpmo_ws_url,
                    extra_headers={"Authorization": f"Bearer {self.config.minicpmo_api_key}"},
                    ping_interval=20, ping_timeout=10
                )
                return ws
            except Exception as e:
                logger.warning(f"连接失败 ({attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(2 ** attempt)
        return None

    async def _initialize_minicpmo_session(self, ws) -> bool:
        """初始化会话: 必须等待 session.queue_done 后才能发 session.update"""
        try:
            # Phase 1: 等待 queue_done (必达事件)
            async for message in ws:
                data = json.loads(message)
                msg_type = data.get("type", "")
                if msg_type == "session.queue_done":
                    logger.info("MiniCPM-o Worker 分配完成")
                    break
                elif msg_type == "session.queued":
                    logger.info(f"MiniCPM-o 排队中: {data}")
                elif msg_type == "session.queue_update":
                    logger.info(f"MiniCPM-o 排队更新: {data}")
                elif msg_type == "error":
                    logger.error(f"MiniCPM-o 错误: {data}")
                    return False

            # Phase 2: 发送 session.update
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": KNOWLEDGE_BASE
                    # voice/ref_audio/tts_ref_audio 按需添加
                }
            }))

            # Phase 3: 等待 session.created
            async for message in ws:
                data = json.loads(message)
                msg_type = data.get("type", "")
                if msg_type == "session.created":
                    session_id = data.get("session_id", "unknown")
                    logger.info(f"MiniCPM-o 会话创建成功: {session_id}")
                    return True
                elif msg_type == "error":
                    logger.error(f"MiniCPM-o 会话创建错误: {data}")
                    return False
        except Exception as e:
            logger.error(f"初始化异常: {e}")
        return False

    async def _send_greeting(self, session: CallSession):
        if not session.ws:
            return
        # 通过文本触发开场白
        await session.ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "请向客户打招呼并自我介绍"}]
            }
        }))
        await session.ws.send(json.dumps({"type": "response.create"}))

    async def _read_audio_file(self, path: str) -> tuple:
        """读取 WAV 文件，返回 (音频裸流, 格式代码)"""
        def _read():
            try:
                with open(path, 'rb') as f:
                    data = f.read()
                    if len(data) <= 44:
                        return None, None
                    fmt_code = struct.unpack_from('<H', data, 20)[0]
                    audio_data = data[44:]
                    return audio_data, fmt_code
            except:
                return None, None
        return await asyncio.get_event_loop().run_in_executor(None, _read)

    async def _main_loop(self, session: CallSession):
        uuid = session.uuid
        chunk_ms = self.config.chunk_duration_ms
        tts_task = asyncio.create_task(self._tts_player(session))
        ws_task = asyncio.create_task(self._ws_message_handler(session))
        chunk_idx = 0

        while session.is_active and self.running:
            try:
                chunk_idx += 1
                chunk_path = os.path.join(self.config.shm_dir, f"{uuid}_{chunk_idx}.wav")

                # 录音
                record_cmd = f"uuid_record {uuid} start {chunk_path} {chunk_ms / 1000.0}"
                await self.esl.send(record_cmd)
                await asyncio.sleep(chunk_ms / 1000.0 + 0.03)
                await self.esl.send(f"uuid_record {uuid} stop {chunk_path}")

                audio_8k, fmt_code = await self._read_audio_file(chunk_path)
                try:
                    os.remove(chunk_path)
                except:
                    pass

                if audio_8k is None or len(audio_8k) == 0:
                    continue

                # 解码为 float32
                if fmt_code == 7:  # PCMU
                    pcm_float_8k = AudioUtils.pcmu_decode(audio_8k)
                elif fmt_code == 1:  # PCM16
                    pcm_int16 = np.frombuffer(audio_8k, dtype=np.int16)
                    pcm_float_8k = AudioUtils.int16_to_float(pcm_int16)
                elif fmt_code == 6:  # PCMA
                    logger.warning("暂不支持 PCMA 解码")
                    continue
                else:
                    logger.error(f"不支持的录音格式代码: {fmt_code}")
                    continue

                # 重采样到 16kHz (MiniCPM-o 输入要求)
                pcm_float_16k = AudioUtils.resample(pcm_float_8k, self.config.fs_sample_rate, self.config.minicpmo_sample_rate_in)

                # VAD 检测
                vad_state, is_speech, is_sentence_end = session.vad.process(pcm_float_16k)

                # 积累音频到 1 秒再发送
                session.audio_accumulator.append(pcm_float_16k)
                session.accumulated_samples += len(pcm_float_16k)

                # 检查是否积累到 1 秒 (16000 samples)
                if session.accumulated_samples >= self.config.target_samples_per_send:
                    # 拼接并截取 1 秒
                    accumulated = np.concatenate(list(session.audio_accumulator))
                    one_second = accumulated[:self.config.target_samples_per_send]
                    remainder = accumulated[self.config.target_samples_per_send:]

                    # 更新积累器
                    session.audio_accumulator.clear()
                    if len(remainder) > 0:
                        session.audio_accumulator.append(remainder)
                    session.accumulated_samples = len(remainder)

                    # 发送给 MiniCPM-o (float32 PCM, base64)
                    await self._send_audio_to_ai(session, one_second)

                # 打断检测
                if is_speech and session.is_ai_speaking:
                    logger.info("检测到用户打断 AI")
                    await self._interrupt_ai(session)
                    session.is_user_speaking = True
                elif is_sentence_end:
                    logger.info("检测到用户停顿，句子结束")
                    sentence_audio = session.vad.get_buffered_audio()
                    if len(sentence_audio) > 0:
                        # 句子结束也发送积累的音频
                        await self._send_audio_to_ai(session, sentence_audio)
                        session.total_turns += 1

                if is_speech:
                    session.last_activity = time.time()
                    session.is_user_speaking = True
                else:
                    session.is_user_speaking = False

                # 超时检查
                if time.time() - session.start_time > self.config.max_call_duration_sec:
                    await self._hangup(uuid, "TIMEOUT")
                    break

            except Exception as e:
                logger.error(f"轮询异常: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        tts_task.cancel()
        ws_task.cancel()

    async def _send_audio_to_ai(self, session: CallSession, audio: np.ndarray, force_listen: bool = False):
        """发送音频给 MiniCPM-o: 16kHz float32 PCM, base64"""
        if not session.ws or not session.ws.open:
            return
        # 确保是 float32
        audio_float32 = audio.astype(np.float32)
        audio_b64 = base64.b64encode(audio_float32.tobytes()).decode()
        await session.ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
            "force_listen": force_listen
        }))

    async def _interrupt_ai(self, session: CallSession):
        """打断 AI: 使用 force_listen=true"""
        if not session.is_ai_speaking:
            return
        session.is_ai_speaking = False
        await self.esl.api("uuid_break", session.uuid)
        # 清空 TTS 队列
        while not session.tts_queue.empty():
            session.tts_queue.get_nowait()
        # 发送 force_listen 打断模型
        if session.ws and session.ws.open:
            empty_audio = np.zeros(self.config.target_samples_per_send, dtype=np.float32)
            await self._send_audio_to_ai(session, empty_audio, force_listen=True)

    async def _tts_player(self, session: CallSession):
        """播放 MiniCPM-o 返回的音频: 24kHz float32 -> 8kHz int16 -> WAV -> uuid_displace"""
        while session.is_active and self.running:
            try:
                # 收集单次回复的所有音频片段
                audio_chunks = []
                while True:
                    try:
                        item = await asyncio.wait_for(session.tts_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        break
                    audio_b64, is_last = item
                    if audio_b64:
                        # 解码 float32 PCM, 24kHz
                        pcm_float = np.frombuffer(base64.b64decode(audio_b64), dtype=np.float32)
                        audio_chunks.append(pcm_float)
                    if is_last:
                        break

                if not audio_chunks or not session.is_active:
                    continue

                # 拼接
                pcm_float_24k = np.concatenate(audio_chunks)
                # 重采样到 8kHz (FreeSWITCH 播放)
                pcm_float_8k = AudioUtils.resample(pcm_float_24k, self.config.minicpmo_sample_rate_out, self.config.fs_sample_rate)
                # 转为 int16
                pcm_int16_8k = AudioUtils.float_to_int16(pcm_float_8k)
                pcm_bytes_8k = pcm_int16_8k.tobytes()

                # 写入 WAV
                wav_data = AudioUtils.create_wav_header(len(pcm_bytes_8k), self.config.fs_sample_rate) + pcm_bytes_8k
                tmp_file = os.path.join(self.config.tts_dir, f"tts_{session.uuid}_{int(time.time()*1000)}.wav")
                with open(tmp_file, 'wb') as f:
                    f.write(wav_data)

                # 播放
                session.is_ai_speaking = True
                await self.esl.api("uuid_displace", f"{session.uuid} start {tmp_file} 0 mux")
                duration = len(pcm_float_8k) / self.config.fs_sample_rate
                await asyncio.sleep(duration + 0.1)
                session.is_ai_speaking = False
                try:
                    os.remove(tmp_file)
                except:
                    pass

            except Exception as e:
                logger.error(f"TTS 播放异常: {e}")
                session.is_ai_speaking = False

    async def _ws_message_handler(self, session: CallSession):
        """处理 MiniCPM-o 返回的消息"""
        ws = session.ws
        try:
            async for message in ws:
                if not session.is_active:
                    break
                data = json.loads(message)
                msg_type = data.get("type", "")

                if msg_type == "response.output_audio.delta":
                    audio_b64 = data.get("audio", "")
                    if audio_b64:
                        await session.tts_queue.put((audio_b64, False))

                elif msg_type == "response.output_audio.done":
                    # 注意: 协议中没有这个事件，用 end_of_turn 判断
                    pass

                elif msg_type == "response.output_audio.delta" and data.get("end_of_turn", False):
                    # 本轮生成结束
                    await session.tts_queue.put((None, True))

                elif msg_type == "response.text.delta":
                    logger.info(f"AI: {data.get('text', '')}")

                elif msg_type == "response.listen":
                    # 模型进入 listen 状态，停止播放残留音频
                    logger.info("MiniCPM-o 进入 listen 状态")
                    # 可以在这里清空 TTS 队列
                    while not session.tts_queue.empty():
                        session.tts_queue.get_nowait()

                elif msg_type == "conversation.item.input_audio_transcription.completed":
                    logger.info(f"用户: {data.get('transcript', '')}")

                elif msg_type == "session.closed":
                    reason = data.get("reason", "unknown")
                    logger.info(f"MiniCPM-o 会话关闭: {reason}")
                    break

                elif msg_type == "error":
                    logger.error(f"MiniCPM-o API Error: {data}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket 连接关闭")
        except Exception as e:
            logger.error(f"WS 异常: {e}")

    async def _hangup(self, uuid: str, cause: str = "NORMAL_CLEARING"):
        if uuid in self.sessions:
            self.sessions[uuid].is_active = False
        await self.esl.api("uuid_kill", uuid)

    async def _cleanup_session(self, uuid: str):
        session = self.sessions.get(uuid)
        if not session:
            return
        if session.ws:
            try:
                await session.ws.close()
            except:
                pass
        for d in [self.config.shm_dir, self.config.tts_dir]:
            for f in os.listdir(d):
                if uuid in f:
                    try:
                        os.remove(os.path.join(d, f))
                    except:
                        pass
        if uuid in self.sessions:
            del self.sessions[uuid]

    async def shutdown(self):
        self.running = False
        for uuid in list(self.sessions.keys()):
            await self._hangup(uuid)
            await self._cleanup_session(uuid)
        if self.esl:
            await self.esl.disconnect()


async def main():
    bot = OutboundCallBot()
    if not await bot.initialize():
        return
    try:
        phone_number = input("请输入要呼叫的号码 (或输入 'listen' 只监听来电): ").strip()
        if phone_number.lower() == 'listen':
            while bot.running:
                await asyncio.sleep(1)
        else:
            uuid = await bot.originate_call(phone_number)
            if uuid:
                while uuid in bot.sessions and bot.sessions[uuid].is_active:
                    await asyncio.sleep(1)
            else:
                logger.error("外呼失败")
    except KeyboardInterrupt:
        pass
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
