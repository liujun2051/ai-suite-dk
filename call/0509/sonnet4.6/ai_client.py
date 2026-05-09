"""
MiniCPM-o WebSocket 客户端
设计：纯 async，运行在主事件循环中，通过回调通知外部。
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


class AIClient:
    """
    封装与 MiniCPM-o（本地或云端）的 WebSocket 通信。

    回调（在事件循环线程中调用）：
        on_audio_chunk(pcm16_bytes: bytes)   AI 产生音频块
        on_turn_done()                        AI 本轮回复结束
        on_speech_started()                   云端检测到用户开始说话
        on_transcript(text: str)              用户语音转文字结果
        on_error(reason: str)                 不可恢复的错误
    """

    CONNECT_TIMEOUT = 10
    SEND_TIMEOUT = 5

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.mode: str = config.AI_MODE
        self._ws = None
        self._connected = False
        self._receive_task: Optional[asyncio.Task] = None

        # 回调
        self.on_audio_chunk: Optional[Callable[[bytes], None]] = None
        self.on_turn_done: Optional[Callable[[], None]] = None
        self.on_speech_started: Optional[Callable[[], None]] = None
        self.on_transcript: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        headers = {}
        if self.mode == "cloud":
            headers["Authorization"] = f"Bearer {config.CLOUD_API_KEY}"
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    config.MINICPM_WS_URL,
                    extra_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=10 * 1024 * 1024,  # 10MB，防止大音频包被截断
                ),
                timeout=self.CONNECT_TIMEOUT
            )
            self._connected = True
            self._receive_task = asyncio.create_task(
                self._receive_loop(),
                name=f"ai-recv-{self.session_id}"
            )
            logger.info(f"[{self.session_id}] AI connected ({self.mode})")
            return True
        except Exception as e:
            logger.error(f"[{self.session_id}] AI connect failed: {e}")
            return False

    async def initialize(self):
        """发送会话初始化配置。"""
        if self.mode == "cloud":
            await self._init_cloud()
        else:
            await self._init_local()

    async def send_audio(self, pcm16_bytes: bytes):
        """发送用户音频（PCM16 bytes）。"""
        if not self._connected:
            return
        b64 = base64.b64encode(pcm16_bytes).decode()
        try:
            if self.mode == "cloud":
                await asyncio.wait_for(
                    self._ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": b64
                    })),
                    timeout=self.SEND_TIMEOUT
                )
            else:
                # 本地模式：直接发送二进制
                await asyncio.wait_for(
                    self._ws.send(pcm16_bytes),
                    timeout=self.SEND_TIMEOUT
                )
        except asyncio.TimeoutError:
            logger.warning(f"[{self.session_id}] send_audio timeout")
        except Exception as e:
            logger.error(f"[{self.session_id}] send_audio error: {e}")
            self._connected = False

    async def send_text(self, text: str):
        """发送文本消息（如 DTMF 提示）。"""
        if not self._connected:
            return
        try:
            if self.mode == "cloud":
                await self._ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}]
                    }
                }))
                await self._ws.send(json.dumps({"type": "response.create"}))
            else:
                await self._ws.send(json.dumps({
                    "type": "text_input",
                    "text": text
                }))
        except Exception as e:
            logger.error(f"[{self.session_id}] send_text error: {e}")

    async def interrupt(self):
        """通知 AI 打断当前输出。"""
        if not self._connected:
            return
        try:
            if self.mode == "cloud":
                await self._ws.send(json.dumps({
                    "type": "input_audio_buffer.clear"
                }))
            else:
                await self._ws.send(json.dumps({"type": "barge_in"}))
        except Exception as e:
            logger.error(f"[{self.session_id}] interrupt error: {e}")

    async def close(self):
        """关闭连接，幂等。"""
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
        logger.info(f"[{self.session_id}] AI connection closed")

    # ------------------------------------------------------------------ #
    #  内部
    # ------------------------------------------------------------------ #

    async def _init_cloud(self):
        # 等待 session.created
        async for msg in self._ws:
            data = json.loads(msg)
            if data.get("type") == "session.created":
                break

        await self._ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": config.AI_SYSTEM_PROMPT,
                "voice": config.AI_VOICE,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 800,
                },
                "input_audio_transcription": {
                    "model": "whisper-1",
                    "language": "zh"
                }
            }
        }))

        # 等待 session.updated
        async for msg in self._ws:
            data = json.loads(msg)
            if data.get("type") == "session.updated":
                break

    async def _init_local(self):
        await self._ws.send(json.dumps({
            "type": "init",
            "session_id": self.session_id,
            "sample_rate": config.AI_SAMPLE_RATE,
            "channels": 1,
            "format": "pcm_s16le"
        }))

    async def _receive_loop(self):
        try:
            async for msg in self._ws:
                if not self._connected:
                    break
                if isinstance(msg, bytes):
                    # 本地模式：二进制音频
                    if self.on_audio_chunk:
                        self.on_audio_chunk(msg)
                else:
                    await self._handle_json(msg)
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"[{self.session_id}] AI WS closed: {e}")
        except Exception as e:
            logger.exception(f"[{self.session_id}] receive loop error: {e}")
        finally:
            self._connected = False
            if self.on_error:
                self.on_error("connection_lost")

    async def _handle_json(self, raw: str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
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
            if self.on_transcript:
                transcript = data.get("transcript", "").strip()
                if transcript:
                    self.on_transcript(transcript)

        elif t == "error":
            err = data.get("error", {})
            logger.error(f"[{self.session_id}] AI error: {err}")
            if self.on_error:
                self.on_error(str(err))
