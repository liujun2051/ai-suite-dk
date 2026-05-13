# VoiceBot

**MiniCPM-o 4.5 全双工语音AI + FreeSWITCH mod_audio_stream**

---

## 架构概览

电话用户
│ SIP
▼
FreeSWITCH
│ mod_audio_stream (raw binary PCM WebSocket)
▼
audio.py (AudioServer :8765)
│ PCM 16kHz 16bit mono
▼
audio_utils.py (VAD + UtteranceDetector)
│ SpeechSegment / 打断信号
▼
session.py (Session 状态机)
│ PCM chunks
▼
ai_client.py (MiniCPMClient)
│ 全双工 WebSocket
▼
MiniCPM-o 4.5 云端API


---

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

### 2. 配置
cp .env.example .env
# 编辑 .env，填入 MINICPM_API_KEY
nano .env

### 3. 启动
# 验证配置
python main.py --check-config

# 启动
python main.py

# 调试模式
python main.py --log-level DEBUG

### 4. Docker 启动
# 构建并启动全部服务
docker-compose up -d

# 查看日志
docker-compose logs -f voicebot

# 停止
docker-compose down

