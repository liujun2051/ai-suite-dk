# MiniCPM-o 4.1 全双工语音网关

纯 Python 实现的 FreeSWITCH ↔ MiniCPM-o 桥接网关，支持 AI 外呼 + AI 接听。

## 架构

```
[用户手机] ←→ [迅时 FXO 网关] ←→ [FreeSWITCH] ←→ [本网关] ←→ [MiniCPM-o 4.1]
                                              (Python + ESL)
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `gateway.py` | 主程序：ESL 连接、会话管理、HTTP API |
| `audio_utils.py` | 音频编解码（PCMA）和重采样 |
| `config.py` | 配置参数 |
| `requirements.txt` | Python 依赖 |

## 安装

```bash
pip install -r requirements.txt
```

## FreeSWITCH 配置

### 1. 确保模块加载

编辑 `/etc/freeswitch/autoload_configs/modules.conf.xml`：

```xml
<load module="mod_event_socket"/>
<load module="mod_dptools"/>
<load module="mod_commands"/>
```

### 2. 网关配置

创建 `conf/sip_profiles/external/newrock.xml`：

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
    <param name="ping" value="30"/>
  </gateway>
</include>
```

### 3. 呼入 Dialplan

创建 `conf/dialplan/public/01_park.xml`：

```xml
<extension name="ai_inbound">
    <condition field="destination_number" expression="^(\d+)$">
        <action application="answer"/>
        <action application="sleep" data="100"/>
        <action application="park"/>
    </condition>
</extension>
```

## 启动

### 1. 启动 MiniCPM-o 推理服务

```bash
# 根据你的实际部署方式
cd /path/to/minicpm-o
python serve.py --model MiniCPM-o-4_1
```

### 2. 启动网关

```bash
python gateway.py
```

## API 使用

### 发起外呼

```bash
curl -X POST http://127.0.0.1:8080/call \
  -H "Content-Type: application/json" \
  -d '{"phone_number":"13800138000"}'
```

### 查看活跃会话

```bash
curl http://127.0.0.1:8080/sessions
```

### 手动打断

```bash
curl -X POST http://127.0.0.1:8080/call/{call_uuid}/barge_in
```

### 挂断

```bash
curl -X POST http://127.0.0.1:8080/call/{call_uuid}/hangup
```

### 健康检查

```bash
curl http://127.0.0.1:8080/health
```

## 打断方式

1. **用户按 `*` 或 `#` 键**
2. **MiniCPM-o 内部检测到用户说话**（模型级打断）
3. **HTTP API 手动触发**

## 注意事项

- 确保 FreeSWITCH ESL 端口 `8021` 可访问
- 确保 MiniCPM-o WebSocket 服务在 `ws://127.0.0.1:8080/audio_stream` 监听
- 根据实际模型调整 `config.py` 中的 `MINICPM_SAMPLE_RATE`（可能是 16000 或 24000）
- 安装 `scipy` 可获得更好的重采样质量（可选）

## 调试

```bash
# 查看 FreeSWITCH 日志
fs_cli -x "console loglevel debug"

# 查看网关日志
# 日志直接输出到终端，级别由 config.LOG_LEVEL 控制
```
