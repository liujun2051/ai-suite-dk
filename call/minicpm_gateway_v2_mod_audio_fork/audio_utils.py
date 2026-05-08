"""音频处理：PCMA 编解码 + 重采样"""

import numpy as np

_ALAW_ENCODE_TABLE = np.zeros(65536, dtype=np.uint8)
_ALAW_DECODE_TABLE = np.zeros(256, dtype=np.int16)


def _init_alaw_tables():
    for i in range(65536):
        pcm = i - 32768
        _ALAW_ENCODE_TABLE[i] = _pcm_to_alaw_single(pcm)
    for i in range(256):
        _ALAW_DECODE_TABLE[i] = _alaw_to_pcm_single(i)


def _pcm_to_alaw_single(pcm: int) -> int:
    mask = 0xD5 if pcm < 0 else 0x55
    pcm = -pcm - 1 if pcm < 0 else pcm
    pcm = min(pcm, 32767)
    if pcm >= 256:
        seg = 7
        if pcm < 16384: seg = 6
        if pcm < 8192:  seg = 5
        if pcm < 4096:  seg = 4
        if pcm < 2048:  seg = 3
        if pcm < 1024:  seg = 2
        if pcm < 512:   seg = 1
        aval = seg << 4 | ((pcm >> (seg + 3)) & 0x0F)
    else:
        aval = pcm >> 4
    return aval ^ mask


def _alaw_to_pcm_single(aval: int) -> int:
    aval ^= 0x55
    sign = -1 if aval & 0x80 else 1
    seg = (aval >> 4) & 0x07
    if seg:
        pcm = ((aval & 0x0F) | 0x10) << (seg + 3)
    else:
        pcm = (aval & 0x0F) << 4
    return sign * (pcm + 8)


_init_alaw_tables()


def pcma_to_pcm(pcma_data: bytes) -> np.ndarray:
    indices = np.frombuffer(pcma_data, dtype=np.uint8)
    return _ALAW_DECODE_TABLE[indices].copy()


def pcm_to_pcma(pcm_data: np.ndarray) -> bytes:
    clipped = np.clip(pcm_data, -32768, 32767).astype(np.int32) + 32768
    return _ALAW_ENCODE_TABLE[clipped].tobytes()


def resample_linear(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return pcm.copy()
    ratio = dst_rate / src_rate
    new_len = int(len(pcm) * ratio)
    if new_len == 0:
        return np.array([], dtype=np.int16)
    old_indices = np.linspace(0, len(pcm) - 1, new_len)
    indices = old_indices.astype(np.int32)
    fractions = old_indices - indices
    next_indices = np.minimum(indices + 1, len(pcm) - 1)
    result = pcm[indices].astype(np.float32) * (1 - fractions) + pcm[next_indices].astype(np.float32) * fractions
    return result.astype(np.int16)


try:
    from scipy import signal
    def resample_scipy(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate:
            return pcm.copy()
        new_len = int(len(pcm) * dst_rate / src_rate)
        return signal.resample(pcm, new_len).astype(np.int16)
    RESAMPLE_FUNC = resample_scipy
except ImportError:
    RESAMPLE_FUNC = resample_linear


def get_frame_size_ms(sample_rate: int, duration_ms: int = 20) -> int:
    return int(sample_rate * duration_ms / 1000)
