"""
音频格式转换模块
设计：无状态静态方法，线程安全，可在任意线程调用。
"""
import base64
import logging

import numpy as np

from audio_utils import pcma_to_pcm, pcm_to_pcma, RESAMPLE_FUNC

logger = logging.getLogger("Audio")

# 常量
FS_SAMPLE_RATE = 8000       # FreeSWITCH: 8kHz G.711
AI_SAMPLE_RATE = 16000      # AI 模型: 16kHz PCM
FRAME_MS = 20               # 每帧时长
FRAME_SAMPLES_8K = int(FS_SAMPLE_RATE * FRAME_MS / 1000)   # 160 samples
FRAME_BYTES_8K = FRAME_SAMPLES_8K * 2                       # 320 bytes PCM
FRAME_BYTES_PCMA = FRAME_SAMPLES_8K                         # 160 bytes G.711


class AudioConverter:
    """FreeSWITCH ↔ AI 音频格式转换。"""

    @staticmethod
    def fs_to_ai(pcma_8k: bytes) -> bytes:
        """
        G.711a 8kHz → PCM16 16kHz
        返回 PCM16LE bytes，直接可用于 base64 编码后发给 AI。
        """
        pcm_8k = pcma_to_pcm(np.frombuffer(pcma_8k, dtype=np.uint8))
        pcm_16k = RESAMPLE_FUNC(pcm_8k, FS_SAMPLE_RATE, AI_SAMPLE_RATE)
        return pcm_16k.astype(np.int16).tobytes()

    @staticmethod
    def ai_to_fs(pcm_16k_bytes: bytes) -> bytes:
        """
        PCM16 16kHz → G.711a 8kHz
        返回 PCMA bytes，可直接写入 FIFO 或文件。
        """
        pcm_16k = np.frombuffer(pcm_16k_bytes, dtype=np.int16)
        pcm_8k = RESAMPLE_FUNC(pcm_16k, AI_SAMPLE_RATE, FS_SAMPLE_RATE)
        return pcm_to_pcma(pcm_8k.astype(np.int16))


class AudioFrameBuffer:
    """
    AI 输出音频帧缓冲区。
    将 AI 返回的变长音频块切割为标准 20ms PCMA 帧。
    非线程安全，每个 session 独立实例。
    """

    def __init__(self):
        self._buf = bytearray()

    def push_pcm16(self, pcm16_bytes: bytes):
        """推入 PCM16 数据，内部转换为 PCMA 后进入缓冲区。"""
        pcma = AudioConverter.ai_to_fs(pcm16_bytes)
        self._buf.extend(pcma)

    def pop_frames(self) -> list[bytes]:
        """弹出所有完整的 20ms PCMA 帧。"""
        frames = []
        while len(self._buf) >= FRAME_BYTES_PCMA:
            frame = bytes(self._buf[:FRAME_BYTES_PCMA])
            del self._buf[:FRAME_BYTES_PCMA]
            frames.append(frame)
        return frames

    def clear(self):
        self._buf.clear()

    def __len__(self):
        return len(self._buf)
