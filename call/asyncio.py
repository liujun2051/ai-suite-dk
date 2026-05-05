#!/usr/bin/env python3
"""
AI Outbound Call Bot - aiosip + asyncio RTP + MiniCPM-o 4.5
轻量级纯 Python 方案，适合快速原型验证
"""

import asyncio
import json
import base64
import logging
import socket
import struct
import time
from typing import Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 配置 ==========
@dataclass
class Config:
    freeswitch_ip: str = "192.168.1.100"
    freeswitch_sip_port: int = 5060
    local_ip: str = "0.0.0.0"
    local_sip_port: int = 5066
    local_rtp_port: int = 5004
    
    minicpmo_api_key: str = "你的API_Key"
    minicpmo_ws_url: str = "wss://api.modelbest.cn/v1/realtime?mode=audio"
    
    fs_sample_rate: int = 8000
    ai_sample_rate: int = 16000
    frame_duration_ms: int = 20
    
    max_call_duration_sec: int = 600

CONFIG = Config()

# ========== 知识库 ==========
KNOWLEDGE_BASE = """
你是XX公司的专业客服代表...
"""

# ========== 音频工具（同上） ==========
class AudioUtils:
    MULAW_BIAS = 33
    EXP_LUT = [0, 132, 396, 924, 1980, 4092, 8316, 16764]
    
    @classmethod
    def pcmu_decode(cls, data: bytes) -> np.ndarray:
        result = np.zeros(len(data), dtype=np.float32)
        for i, byte in enumerate(data):
            byte = ~byte & 0xFF
            sign = (byte & 0x80)
            exponent = (byte >> 4) & 0x07
            mantissa = byte & 0x0F
            sample = cls.EXP_LUT[exponent] + (mantissa << (exponent + 3))
            if sign: sample = -sample
            result[i] = (sample - cls.MULAW_BIAS) / 32768.0
        return result
    
    @classmethod
    def pcmu_encode(cls, data: np.ndarray) -> bytes:
        data = np.clip(data, -1.0, 1.0)
        samples = (data * 32768).astype(np.int16)
        result = bytearray(len(samples))
        for i, sample in enumerate(samples):
            sign = 0x80 if sample < 0 else 0
            sample = abs(int(sample))
            if sample < 256: encoded = sample >> 4
            elif sample < 512: encoded = 0x10 | ((sample - 256) >> 5)
            elif sample < 1024: encoded = 0x20 | ((sample - 512) >> 6)
            elif sample < 2048: encoded = 0x30 | ((sample - 1024) >> 7)
            elif sample < 4096: encoded = 0x40 | ((sample - 2048) >> 8)
            elif sample < 8192: encoded = 0x50 | ((sample - 4096) >> 9)
            elif sample < 16384: encoded = 0x60 | ((sample - 8192) >> 10)
            else: encoded = 0x70 | ((sample - 16384) >> 11)
            result[i] = ~(sign | encoded) & 0xFF
        return bytes(result)
    
    @classmethod
    def resample(cls, data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr: return data
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


# ========== 极简 RTP 实现 ==========
class RTPSession:
    """
    极简 RTP 会话，处理 UDP 收发
    生产环境建议用 aiortc.RtpReceiver/RtpSender
    """
    
    def __init__(self, local_port: int, remote_addr: Optional[Tuple[str, int]] = None):
        self.local_port = local_port
        self.remote_addr = remote_addr
        self.sock: Optional[socket.socket] = None
        self.sequence = 0
        self.timestamp = 0
        self.ssrc = int(time.time()) & 0xFFFFFFFF
        
        # 接收队列
        self.rx_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1000)
        self.tx_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1000)
        
        self.running = False
    
    async def start(self):
        """启动 RTP 会话"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((CONFIG.local_ip, self.local_port))
        self.sock.setblocking(False)
        self.running = True
        
        # 启动收发任务
        await asyncio.gather(
            self._receive_loop(),
            self._transmit_loop()
        )
    
    async def _receive_loop(self):
        """接收 RTP 包"""
        loop = asyncio.get_event_loop()
        while self.running:
            try:
                data, addr = await loop.sock_recvfrom(self.sock, 2048)
                if len(data) < 12:
                    continue  # 不是有效的 RTP 包
                
                # 解析 RTP 头（简化版）
                payload = data[12:]  # 跳过 12 字节 RTP 头
                await self.rx_queue.put(payload)
                
                # 记录远端地址（用于发送）
                if not self.remote_addr:
                    self.remote_addr = addr
                
            except Exception as e:
                logger.error(f"RTP receive error: {e}")
                await asyncio.sleep(0.01)
    
    async def _transmit_loop(self):
        """发送 RTP 包"""
        loop = asyncio.get_event_loop()
        while self.running:
            try:
                payload = await asyncio.wait_for(self.tx_queue.get(), timeout=0.1)
                
                if not self.remote_addr:
                    continue
                
                # 构建 RTP 头
                self.sequence = (self.sequence + 1) & 0xFFFF
                # 时间戳增量 = 采样率 * 帧时长
                self.timestamp += int(CONFIG.fs_sample_rate * CONFIG.frame_duration_ms / 1000)
                
                rtp_header = struct.pack('!BBHII',
                    0x80,           # V=2, P=0, X=0, CC=0
                    0x00,           # M=0, PT=0 (PCMU)
                    self.sequence,
                    self.timestamp,
                    self.ssrc
                )
                
                packet = rtp_header + payload
                await loop.sock_sendto(self.sock, packet, self.remote_addr)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"RTP transmit error: {e}")
                await asyncio.sleep(0.01)
    
    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()


# ========== SIP 处理 ==========
class SIPCallHandler:
    """处理 SIP 信令"""
    
    def __init__(self, app):
        self.app = app
        self.dialog: Optional[aiosip.Dialog] = None
        self.rtp_session: Optional[RTPSession] = None
    
    async def handle_invite(self, request: aiosip.Request):
        """处理 INVITE 请求（来电）"""
        logger.info(f"Incoming INVITE from {request.from_details['uri']}")
        
        # 解析 SDP 获取远端 RTP 地址
        sdp = request.payload
        remote_rtp_addr = self._parse_sdp(sdp)
        
        # 创建 RTP 会话
        self.rtp_session = RTPSession(
            local_port=CONFIG.local_rtp_port,
            remote_addr=remote_rtp_addr
        )
        
        # 发送 200 OK 带 SDP
        local_sdp = self._generate_sdp()
        await request.reply(200, payload=local_sdp)
        
        # 启动 RTP
        asyncio.create_task(self.rtp_session.start())
        
        # 通知应用通话建立
        await self.app.on_call_established(self.rtp_session)
    
    async def handle_bye(self, request: aiosip.Request):
        """处理 BYE 请求（挂断）"""
        logger.info("Received BYE")
        if self.rtp_session:
            self.rtp_session.stop()
        await request.reply(200)
        await self.app.on_call_ended()
    
    def _parse_sdp(self, sdp: str) -> Optional[Tuple[str, int]]:
        """从 SDP 解析 RTP 地址"""
        for line in sdp.split('\n'):
            if line.startswith('c=IN IP4 '):
                ip = line.split()[-1]
            if line.startswith('m=audio '):
                port = int(line.split()[1])
                return (ip, port)
        return None
    
    def _generate_sdp(self) -> str:
        """生成本地 SDP"""
        return f"""v=0
o=- 0 0 IN IP4 {CONFIG.local_ip}
s=AI Bot
c=IN IP4 {CONFIG.local_ip}
t=0 0
m=audio {CONFIG.local_rtp_port} RTP/AVP 0
a=rtpmap:0 PCMU/8000
a=sendrecv
"""


# ========== 主应用类 ==========
class AIOutboundBot:
    def __init__(self):
        self.config = CONFIG
        self.sip_app: Optional[aiosip.Application] = None
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.rtp_session: Optional[RTPSession] = None
        self.running = False
        
        # 音频处理
        self.vad_buffer = bytearray()
        self.is_ai_speaking = False
    
    async def start(self):
        """启动 SIP 服务器"""
        self.sip_app = aiosip.Application()
        
        # 注册路由
        self.sip_app.router.add_route('INVITE', '/', self._on_invite)
        self.sip_app.router.add_route('BYE', '/', self._on_bye)
        self.sip_app.router.add_route('ACK', '/', self._on_ack)
        
        # 启动 UDP 监听
        await self.sip_app.run(
            protocol=aiosip.UDP,
            host=self.config.local_ip,
            port=self.config.local_sip_port
        )
        
        self.running = True
        logger.info(f"SIP server listening on {self.config.local_ip}:{self.config.local_sip_port}")
        
        # 保持运行
        while self.running:
            await asyncio.sleep(1)
    
    async def _on_invite(self, request: aiosip.Request):
        handler = SIPCallHandler(self)
        await handler.handle_invite(request)
    
    async def _on_bye(self, request: aiosip.Request):
        if self.rtp_session:
            self.rtp_session.stop()
        await request.reply(200)
    
    async def _on_ack(self, request: aiosip.Request):
        logger.info("ACK received")
    
    async def on_call_established(self, rtp_session: RTPSession):
        """通话建立回调"""
        self.rtp_session = rtp_session
        logger.info("Call established, starting audio processing")
        
        # 连接 MiniCPM-o
        self.ws = await self._connect_minicpmo()
        if not self.ws:
            logger.error("Failed to connect MiniCPM-o")
            return
        
        # 初始化会话
        await self._init_minicpmo_session()
        
        # 启动音频处理
        await asyncio.gather(
            self._audio_reader(rtp_session),
            self._audio_writer(rtp_session),
            self._ws_handler()
        )
    
    async def _connect_minicpmo(self):
        """连接 MiniCPM-o"""
        try:
            ws = await websockets.connect(
                self.config.minicpmo_ws_url,
                extra_headers={"Authorization": f"Bearer {self.config.minicpmo_api_key}"}
            )
            logger.info("MiniCPM-o connected")
            return ws
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return None
    
    async def _init_minicpmo_session(self):
        """初始化 MiniCPM-o 会话"""
        async for msg in self.ws:
            data = json.loads(msg)
            if data.get("type") == "session.created":
                break
        
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": KNOWLEDGE_BASE,
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16"
            }
        }))
        
        async for msg in self.ws:
            data = json.loads(msg)
            if data.get("type") == "session.updated":
                logger.info("Session initialized")
                break
    
    async def _audio_reader(self, rtp_session: RTPSession):
        """从 RTP 读取音频 → 处理 → 发送给 MiniCPM-o"""
        frame_size = int(self.config.fs_sample_rate * self.config.frame_duration_ms / 1000)
        
        while self.running:
            try:
                # 从 RTP 接收队列读取
                pcmu_frame = await asyncio.wait_for(rtp_session.rx_queue.get(), timeout=0.1)
                
                # 解码 PCMU → float32
                pcm_float_8k = AudioUtils.pcmu_decode(pcmu_frame)
                
                # 重采样 8→16kHz
                pcm_float_16k = AudioUtils.resample(
                    pcm_float_8k, self.config.fs_sample_rate, self.config.ai_sample_rate
                )
                
                # 转 int16
                pcm_int16 = AudioUtils.float_to_int16(pcm_float_16k)
                
                # 发送给 MiniCPM-o
                audio_b64 = base64.b64encode(pcm_int16.tobytes()).decode()
                await self.ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Audio reader error: {e}")
    
    async def _audio_writer(self, rtp_session: RTPSession):
        """从 MiniCPM-o 接收音频 → 处理 → 写入 RTP"""
        frame_size = int(self.config.fs_sample_rate * self.config.frame_duration_ms / 1000)
        
        while self.running:
            try:
                # 这里需要从 ws_handler 获取音频
                # 简化：直接等待 ws 消息
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Audio writer error: {e}")
    
    async def _ws_handler(self):
        """处理 MiniCPM-o WebSocket 消息"""
        try:
            async for msg in self.ws:
                data = json.loads(msg)
                msg_type = data.get("type", "")
                
                if msg_type == "response.output_audio.delta":
                    audio_b64 = data.get("delta", "")
                    if audio_b64 and self.rtp_session:
                        # 解码 → 重采样 → 编码 → 发送 RTP
                        pcm_int16 = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                        pcm_float_16k = AudioUtils.int16_to_float(pcm_int16)
                        pcm_float_8k = AudioUtils.resample(
                            pcm_float_16k, self.config.ai_sample_rate, self.config.fs_sample_rate
                        )
                        pcmu_data = AudioUtils.pcmu_encode(pcm_float_8k)
                        
                        # 分帧发送
                        for i in range(0, len(pcmu_data), frame_size):
                            chunk = pcmu_data[i:i + frame_size]
                            if len(chunk) < frame_size:
                                chunk += b'\x00' * (frame_size - len(chunk))
                            await self.rtp_session.tx_queue.put(chunk)
                
                elif msg_type == "response.text.delta":
                    logger.info(f"AI: {data.get('delta', '')}")
                    
        except Exception as e:
            logger.error(f"WS handler error: {e}")
    
    async def on_call_ended(self):
        """通话结束"""
        logger.info("Call ended")
        self.rtp_session = None
        if self.ws:
            await self.ws.close()
    
    async def make_outbound_call(self, phone_number: str):
        """
        发起外呼
        需要通过 FreeSWITCH 的 originate + bridge
        或者直接用 aiosip 发送 INVITE 到 FreeSWITCH
        """
        # 简化：这里只是示例，实际需要构造完整 SIP INVITE
        logger.info(f"Outbound call to {phone_number} - not fully implemented in this POC")


# ========== 启动 ==========
async def main():
    bot = AIOutboundBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())
