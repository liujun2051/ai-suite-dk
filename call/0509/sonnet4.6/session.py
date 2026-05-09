"""
通话会话模块
核心修复：
  1. dataclass 字段全部加类型注解，消除类变量共享
  2. 跨线程协程调用统一用 run_coroutine_threadsafe
  3. 音频帧缓冲区确保 20ms 对齐
  4. uuid_broadcast 批量播放，减少 ESL 调用频率
  5. FIFO 打开加超时保护
  6. 生命周期管理：hangup 幂等，资源清理完整
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
from typing import Any, Callable, Optional

import numpy as np

from audio import AudioConverter, AudioFrameBuffer, FRAME_BYTES_PCMA
from ai_client import AIClient

logger = logging.getLogger("Session")

# 批量播放：累积多少 ms 的音频后触发一次 uuid_broadcast
PLAY_FLUSH_MS = 500
PLAY_FLUSH_BYTES = int(8000 * PLAY_FLUSH_MS / 1000)   # 4000 bytes PCMA
FIFO_OPEN_TIMEOUT = 15      # 等待 FS 开始写入 FIFO 的最大秒数


@dataclass
class CallSession:
    """
    单路通话的完整状态。
    所有字段必须有类型注解，确保 dataclass 正确创建实例变量。
    """
    # 基本信息（构造时传入）
    uuid: str
    direction: str
    caller_number: str
    callee_number: str

    # 时间戳
    created_at: float = field(default_factory=time.time)

    # 音频队列：线程 ↔ 协程之间用普通 queue.Queue 桥接
    # fs→ai: FIFO 读取线程 → 协程
    fs_to_ai_queue: queue.Queue = field(
        default_factory=lambda: queue.Queue(maxsize=500)
    )
    # ai→fs: 协程回调 → 播放线程
    ai_to_fs_queue: queue.Queue = field(
        default_factory=lambda: queue.Queue(maxsize=500)
    )

    # 状态标志
    is_active: bool = field(default=True)
    is_answered: bool = field(default=False)
    is_ai_speaking: bool = field(default=False)

    # 统计
    audio_in_bytes: int = field(default=0)
    audio_out_bytes: int = field(default=0)

    # 内部对象（不参与 __init__ 但作为实例变量）
    ai_client: Optional[AIClient] = field(default=None, init=False)
    _threads: list = field(default_factory=list, init=False)
    _fifo_path: str = field(default="", init=False)
    _tmp_dir: str = field(default="", init=False)
    _frame_buffer: AudioFrameBuffer = field(
        default_factory=AudioFrameBuffer, init=False
    )
    _hangup_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False
    )
    _barge_in_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False
    )
    _esl_ref: Any = field(default=None, init=False)   # weakref to ESLConnection
    _loop: Optional[asyncio.AbstractEventLoop] = field(default=None, init=False)

    # ------------------------------------------------------------------ #
    #  生命周期
    # ------------------------------------------------------------------ #

    def start(self, loop: asyncio.AbstractEventLoop, esl_conn):
        """
        启动会话。
        必须在 CHANNEL_ANSWER 之后、FIFO 创建之后调用。
        """
        import weakref
        self._loop = loop
        self._esl_ref = weakref.ref(esl_conn)
        self._tmp_dir = tempfile.mkdtemp(prefix=f"ai_{self.uuid}_")

        # 线程1：从 FIFO 读取 FS 音频
        t_fifo = threading.Thread(
            target=self._fifo_read_loop,
            name=f"fifo-{self.uuid[:8]}",
            daemon=True
        )
        t_fifo.start()
        self._threads.append(t_fifo)

        # 线程2：将 AI 音频播放到 FS
        t_play = threading.Thread(
            target=self._play_loop,
            name=f"play-{self.uuid[:8]}",
            daemon=True
        )
        t_play.start()
        self._threads.append(t_play)

        # 协程：AI 主逻辑（在主事件循环中运行）
        asyncio.run_coroutine_threadsafe(self._ai_main(), self._loop)

    def hangup(self, reason: str = "normal"):
        """
        挂断并清理所有资源。幂等：多次调用安全。
        """
        with self._hangup_lock:
            if not self.is_active:
                return
            self.is_active = False

        logger.info(f"[{self.uuid}] Hangup: {reason}")

        # 关闭 AI 连接
        if self.ai_client and self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.ai_client.close(), self._loop
            )

        # 等待子线程退出
        for t in self._threads:
            t.join(timeout=3)

        # 清理 FIFO
        if self._fifo_path and os.path.exists(self._fifo_path):
            try:
                os.remove(self._fifo_path)
            except OSError:
                pass

        # 清理临时目录
        if self._tmp_dir and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

        logger.info(f"[{self.uuid}] Cleanup done. "
                    f"in={self.audio_in_bytes}B out={self.audio_out_bytes}B")

    # ------------------------------------------------------------------ #
    #  事件处理（从 ESL 事件线程调用）
    # ------------------------------------------------------------------ #

    def on_answered(self):
        self.is_answered = True

    def on_dtmf(self, digit: str):
        logger.info(f"[{self.uuid}] DTMF: {digit}")
        if digit in ("*", "#"):
            self.barge_in()
        else:
            if self.ai_client and self._loop:
                asyncio.run_coroutine_threadsafe(
                    self.ai_client.send_text(f"用户按键：{digit}"),
                    self._loop
                )

    def barge_in(self):
        """打断 AI 当前输出。线程安全，幂等（短时间内多次调用只执行一次）。"""
        with self._barge_in_lock:
            if not self.is_ai_speaking:
                return
            self.is_ai_speaking = False

        logger.info(f"[{self.uuid}] Barge-in triggered")

        # 1. 通知 FS 停止当前播放
        esl = self._esl_ref() if self._esl_ref else None
        if esl:
            try:
                esl.send_bgapi_nowait(f"uuid_break {self.uuid} all")
            except Exception as e:
                logger.warning(f"[{self.uuid}] uuid_break failed: {e}")

        # 2. 通知 AI 打断
        if self.ai_client and self._loop:
            asyncio.run_coroutine_threadsafe(
                self.ai_client.interrupt(), self._loop
            )

        # 3. 清空待播放队列和帧缓冲区
        self._drain_queues()

    # ------------------------------------------------------------------ #
    #  FIFO 读取线程
    # ------------------------------------------------------------------ #

    def _fifo_read_loop(self):
        """
        以阻塞方式读取 FS 写入 FIFO 的 G.711a 音频。
        使用 select 实现打开超时，防止 FS 未启动录音时永久挂起。
        """
        fifo = self._fifo_path
        fd = -1
        try:
            # 非阻塞打开 FIFO，等待写入方（FS）就绪
            fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
            ready, _, _ = select.select([fd], [], [], FIFO_OPEN_TIMEOUT)
            if not ready:
                logger.error(f"[{self.uuid}] FIFO open timeout ({FIFO_OPEN_TIMEOUT}s)")
                return

            # 切回阻塞模式，进入读取循环
            os.set_blocking(fd, True)
            logger.info(f"[{self.uuid}] FIFO reading started")

            while self.is_active:
                try:
                    # 读取 160 字节 = 20ms G.711a@8kHz
                    data = os.read(fd, FRAME_BYTES_PCMA)
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
                    # 丢弃最旧的帧，保持实时性
                    try:
                        self.fs_to_ai_queue.get_nowait()
                        self.fs_to_ai_queue.put_nowait(data)
                    except queue.Empty:
                        pass

        except Exception as e:
            logger.error(f"[{self.uuid}] FIFO read error: {e}")
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            logger.info(f"[{self.uuid}] FIFO read loop exited")

    # ------------------------------------------------------------------ #
    #  AI 主协程（在主事件循环中运行）
    # ------------------------------------------------------------------ #

    async def _ai_main(self):
        """连接 AI，设置回调，启动音频发送循环。"""
        client = AIClient(self.uuid)
        client.on_audio_chunk = self._on_ai_audio       # 在事件循环线程调用
        client.on_turn_done = self._on_ai_turn_done
        client.on_speech_started = self._on_speech_started
        client.on_transcript = self._on_transcript
        client.on_error = self._on_ai_error
        self.ai_client = client

        if not await client.connect():
            self.hangup("ai_connect_failed")
            return

        await client.initialize()

        # 外呼：AI 先说第一句话
        if self.direction == "outbound":
            await client.send_text(
                "你好，我是智能客服，请问有什么可以帮您？"
            )

        # 持续将 FS 音频发送给 AI
        await self._send_to_ai_loop()

    async def _send_to_ai_loop(self):
        """
        从 fs_to_ai_queue 读取 G.711a 帧，转换后发给 AI。
        使用 run_in_executor 避免阻塞事件循环。
        """
        loop = asyncio.get_running_loop()
        converter = AudioConverter()

        while self.is_active and self.ai_client and self.ai_client.connected:
            try:
                # 非阻塞方式从同步队列取数据
                pcma_frame = await loop.run_in_executor(
                    None,
                    lambda: self.fs_to_ai_queue.get(timeout=0.05)
                )
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[{self.uuid}] queue read error: {e}")
                break

            try:
                pcm16 = converter.fs_to_ai(pcma_frame)
                await self.ai_client.send_audio(pcm16)
            except Exception as e:
                logger.error(f"[{self.uuid}] AI send error: {e}")
                break

    # ------------------------------------------------------------------ #
    #  AI 回调（在事件循环线程中调用）
    # ------------------------------------------------------------------ #

    def _on_ai_audio(self, pcm16_bytes: bytes):
        """AI 产生音频块，送入帧缓冲区并转发到播放队列。"""
        self.is_ai_speaking = True
        self._frame_buffer.push_pcm16(pcm16_bytes)
        for frame in self._frame_buffer.pop_frames():
            self.audio_out_bytes += len(frame)
            try:
                self.ai_to_fs_queue.put_nowait(frame)
            except queue.Full:
                # 播放队列满：说明 _play_loop 处理不过来，丢弃旧帧
                try:
                    self.ai_to_fs_queue.get_nowait()
                    self.ai_to_fs_queue.put_nowait(frame)
                except queue.Empty:
                    pass

    def _on_ai_turn_done(self):
        """AI 本轮回复结束，等待剩余帧播放完毕后重置状态。"""
        # 将缓冲区剩余数据（不足一帧的尾部）也送出去
        remaining = bytes(self._frame_buffer._buf)
        if remaining:
            self._frame_buffer.clear()
            try:
                self.ai_to_fs_queue.put_nowait(remaining)
            except queue.Full:
                pass
        # 注意：is_ai_speaking 在播放线程队列清空后由 _play_loop 重置
        logger.debug(f"[{self.uuid}] AI turn done")

    def _on_speech_started(self):
        """云端 VAD 检测到用户开口，触发打断。"""
        if self.is_ai_speaking:
            self.barge_in()

    def _on_transcript(self, text: str):
        logger.info(f"[{self.uuid}] User said: {text}")

    def _on_ai_error(self, reason: str):
        logger.error(f"[{self.uuid}] AI error: {reason}")
        if self.is_active:
            self.hangup(f"ai_error:{reason}")

    # ------------------------------------------------------------------ #
    #  播放线程
    # ------------------------------------------------------------------ #

    def _play_loop(self):
        """
        从 ai_to_fs_queue 读取 PCMA 帧，累积后批量通过
        uuid_broadcast 播放给 FS 通话方。
        批量播放策略：累积 PLAY_FLUSH_MS 毫秒或检测到队列空时触发一次播放。
        """
        buf = bytearray()
        last_flush = time.monotonic()

        while self.is_active:
            try:
                frame = self.ai_to_fs_queue.get(timeout=0.02)
                buf.extend(frame)
            except queue.Empty:
                # 队列暂时为空：如果有积累的数据，立即 flush
                pass

            now = time.monotonic()
            should_flush = (
                len(buf) >= PLAY_FLUSH_BYTES or
                (buf and now - last_flush >= PLAY_FLUSH_MS / 1000)
            )

            if should_flush and buf:
                self._flush_to_fs(bytes(buf))
                buf.clear()
                last_flush = now

                # 如果队列已空，说明本轮 AI 回复播放完毕
                if self.ai_to_fs_queue.empty():
                    self.is_ai_speaking = False

        # 退出前 flush 剩余数据
        if buf:
            self._flush_to_fs(bytes(buf))

    def _flush_to_fs(self, pcma_data: bytes):
        """将一段 PCMA 音频写入临时文件并通过 uuid_broadcast 播放。"""
        esl = self._esl_ref() if self._esl_ref else None
        if not esl or not self._tmp_dir:
            return

        # 写入临时文件
        filename = os.path.join(
            self._tmp_dir,
            f"{int(time.monotonic() * 1000)}.pcma"
        )
        try:
            with open(filename, "wb") as f:
                f.write(pcma_data)
        except OSError as e:
            logger.error(f"[{self.uuid}] Write audio file failed: {e}")
            return

        # 广播给 FS（aleg = 播放给主叫方）
        try:
            esl.send_bgapi_nowait(
                f"uuid_broadcast {self.uuid} {filename} aleg"
            )
        except Exception as e:
            logger.warning(f"[{self.uuid}] uuid_broadcast failed: {e}")

        # 延迟清理旧文件（保留最近 10 个，防止 FS 异步读取时文件已删除）
        self._cleanup_old_files(keep=10)

    def _cleanup_old_files(self, keep: int = 10):
        """清理临时目录中较旧的音频文件。"""
        if not self._tmp_dir or not os.path.exists(self._tmp_dir):
            return
        try:
            files = sorted(
                [f for f in os.listdir(self._tmp_dir) if f.endswith(".pcma")],
                key=lambda f: os.path.getmtime(
                    os.path.join(self._tmp_dir, f)
                )
            )
            for old in files[:-keep]:
                try:
                    os.remove(os.path.join(self._tmp_dir, old))
                except OSError:
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  工具
    # ------------------------------------------------------------------ #

    def _drain_queues(self):
        """清空音频队列和帧缓冲区（barge-in 时使用）。"""
        while not self.ai_to_fs_queue.empty():
            try:
                self.ai_to_fs_queue.get_nowait()
            except queue.Empty:
                break
        while not self.fs_to_ai_queue.empty():
            try:
                self.fs_to_ai_queue.get_nowait()
            except queue.Empty:
                break
        self._frame_buffer.clear()
