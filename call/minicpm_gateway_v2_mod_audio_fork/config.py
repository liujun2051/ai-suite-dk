"""配置：支持云端 API 和本地模型切换"""

# ========== FreeSWITCH ESL ==========
FS_HOST = "127.0.0.1"
FS_ESL_PORT = 8021
FS_ESL_PASS = "ClueCon"

# ========== 音频参数 ==========
FS_SAMPLE_RATE = 8000
AI_SAMPLE_RATE = 16000
FRAME_DURATION_MS = 20

# ========== 网络 ==========
AUDIO_PORT_BASE = 35000
AUDIO_PORT_MAX = 45000

# ========== MiniCPM-o 模式切换 ==========
AI_MODE = "cloud"

# 云端配置（当前）
CLOUD_API_KEY = "你的API_Key"
CLOUD_WS_URL = "wss://api.modelbest.cn/v1/realtime?mode=audio"

# 本地配置（未来）
LOCAL_WS_URL = "ws://127.0.0.1:8080/audio_stream"

# 实际使用的 URL
MINICPM_WS_URL = CLOUD_WS_URL if AI_MODE == "cloud" else LOCAL_WS_URL

# ========== 外呼 ==========
OUTBOUND_CID = "02112345678"
FXO_GATEWAY = "newrock_fxo"

# ========== 日志 ==========
LOG_LEVEL = "INFO"
