# MiniCPM-o 网关 v2.0 (mod_audio_fork + VAD)

## 特点
- **音频后端**: mod_audio_fork (UDP 直连)
- **延迟**: ~10-20ms
- **打断**: 本地 webrtcvad (< 30ms) + DTMF + 云端事件
- **需安装**: FreeSWITCH mod_audio_fork 模块

## 安装 FreeSWITCH 模块

```bash
cd /usr/src/freeswitch
git clone https://github.com/thehunmonkgroup/mod_audio_fork.git
cd mod_audio_fork
make
make install

echo "<load module=\"mod_audio_fork\"/>" >> /etc/freeswitch/autoload_configs/modules.conf.xml
fs_cli -x "reload mod_audio_fork"
```

## Python 依赖

```bash
pip install -r requirements.txt
```

## FreeSWITCH 配置

### SIP 网关
`conf/sip_profiles/external/newrock.xml`：

```xml
<include>
  <gateway name="newrock_fxo">
    <param name="realm" value="192.168.1.200"/>
    <param name="proxy" value="192.168.1.200"/>
    <param name="port" value="5060"/>
    <param name="register" value="false"/>
    <param name="username" value="fxo"/>
    <param name="password" value="fxo"/>
    <param name="context" value="public"/>
    <param name="auth-calls" value="false"/>
    <param name="codec-prefs" value="PCMA"/>
  </gateway>
</include>
```

### 呼入 Dialplan
`conf/dialplan/public/01_park.xml`：

```xml
<extension name="ai_inbound">
    <condition field="destination_number" expression="^(\d+)$">
        <action application="answer"/>
        <action application="sleep" data="100"/>
        <action application="park"/>
    </condition>
</extension>
```

## 配置

编辑 `config.py`：

```python
# 云端模式
AI_MODE = "cloud"
CLOUD_API_KEY = "你的API_Key"

# 或本地模式
AI_MODE = "local"
LOCAL_WS_URL = "ws://127.0.0.1:8080/audio_stream"
```

## 启动

```bash
python gateway.py
```

## API

```bash
# 外呼
curl -X POST http://127.0.0.1:8080/call -H "Content-Type: application/json" -d '{"phone_number":"13800138000"}'

# 状态
curl http://127.0.0.1:8080/sessions
curl http://127.0.0.1:8080/health
```

## 打断方式（三重保障）

1. **本地 VAD** (webrtcvad): 延迟 < 30ms，最可靠
2. **DTMF `*`/`#` 键**: 用户主动打断
3. **云端 speech_started**: 备用检测

## 与 v1.0 FIFO 版本对比

| 特性 | v1.0 FIFO | v2.0 mod_audio_fork |
|------|-----------|---------------------|
| 延迟 | ~50-100ms | ~10-20ms |
| 打断延迟 | 300-800ms | < 30ms |
| FreeSWITCH 模块 | 无需额外 | 需 mod_audio_fork |
| 稳定性 | 高 | 高 |
| 适用阶段 | 快速验证 | 生产部署 |
