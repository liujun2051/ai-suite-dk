# MiniCPM-o 网关 v1.0 (FIFO 版本)

## 特点
- **音频后端**: uuid_record + FIFO 命名管道
- **延迟**: ~50-100ms
- **稳定性**: 高，无需额外 FreeSWITCH 模块
- **打断**: DTMF `*`/`#` + 云端 speech_started 事件

## 安装

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

## 注意事项

- FIFO 文件在 `/tmp/minicpm_fifo_*.pcma`
- 异常退出后手动清理：`rm -f /tmp/minicpm_fifo_*`
