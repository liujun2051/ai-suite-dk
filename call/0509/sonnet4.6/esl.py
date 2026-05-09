"""
ESL 连接模块
核心设计：单一读取线程，所有响应通过队列路由，彻底消除 socket 竞态。
"""
import logging
import queue
import socket
import threading
import time
import uuid as uuid_mod
from typing import Callable, Dict, List, Optional
from urllib.parse import unquote

logger = logging.getLogger("ESL")


class ESLError(Exception):
    pass


class ESLConnection:
    """
    线程安全的 FreeSWITCH ESL 连接。

    架构：
        - _event_loop 线程是唯一读取 socket 的地方
        - send_command / send_bgapi 只写 socket，通过 Queue 等待响应
        - 所有事件回调在独立线程池中执行，不阻塞读取循环
    """

    RECONNECT_INTERVAL = 5      # 断线重连间隔（秒）
    CMD_TIMEOUT = 10            # 命令响应超时（秒）
    BGAPI_TIMEOUT = 30          # bgapi 响应超时（秒）
    RECV_CHUNK = 4096           # socket 读取块大小

    def __init__(self, host: str, port: int, password: str):
        self.host = host
        self.port = port
        self.password = password

        self._sock: Optional[socket.socket] = None
        self._connected = False
        self._running = False

        # 发送锁：保证同一时刻只有一个命令在发送
        self._send_lock = threading.Lock()

        # 同步 api 命令的响应队列（同一时刻只有一个 api 在飞）
        self._api_response_queue: queue.Queue = queue.Queue()

        # bgapi 异步结果路由：job_uuid → Queue
        self._pending_bgapi: Dict[str, queue.Queue] = {}
        self._pending_lock = threading.Lock()

        # 事件回调注册表
        self._event_callbacks: Dict[str, List[Callable]] = {}
        self._callback_executor = ThreadPoolExecutor(max_workers=4,
                                                     thread_name_prefix="esl-cb")

        self._reader_thread: Optional[threading.Thread] = None
        self._buf = b""         # socket 读取缓冲区

    # ------------------------------------------------------------------ #
    #  公开接口
    # ------------------------------------------------------------------ #

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """建立连接并完成 ESL 握手，成功返回 True。"""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(10)
            self._sock.connect((self.host, self.port))
            self._sock.settimeout(None)         # 切换为阻塞模式
            self._buf = b""

            # 握手：等待 auth/request
            header = self._read_line()
            if "auth/request" not in header:
                raise ESLError(f"Unexpected greeting: {header}")

            # 发送密码（此时事件循环尚未启动，直接读写安全）
            self._raw_send(f"auth {self.password}\n\n")
            reply = self._read_line()
            if "ok" not in reply.lower():
                raise ESLError(f"Auth failed: {reply}")

            self._connected = True
            logger.info(f"ESL connected to {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"ESL connect failed: {e}")
            self._connected = False
            return False

    def start(self):
        """启动后台事件循环线程。必须在 connect() 成功后调用。"""
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._event_loop,
            name="esl-reader",
            daemon=True
        )
        self._reader_thread.start()

    def stop(self):
        """优雅关闭。"""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._callback_executor.shutdown(wait=False)

    def subscribe(self, *events: str):
        """订阅事件，例如 subscribe('CHANNEL_ANSWER', 'DTMF')。"""
        event_str = " ".join(events)
        # 此时事件循环已启动，用 send_command 发送
        self._send_command_raw(f"event plain {event_str}")

    def on(self, event_name: str, callback: Callable):
        """注册事件回调，线程安全。"""
        if event_name not in self._event_callbacks:
            self._event_callbacks[event_name] = []
        self._event_callbacks[event_name].append(callback)

    def send_command(self, cmd: str) -> str:
        """
        发送同步 api 命令，阻塞直到收到响应。
        线程安全，可从任意线程调用。
        """
        with self._send_lock:
            self._raw_send(f"api {cmd}\n\n")
            try:
                return self._api_response_queue.get(timeout=self.CMD_TIMEOUT)
            except queue.Empty:
                raise ESLError(f"api command timeout: {cmd}")

    def send_bgapi(self, cmd: str) -> str:
        """
        发送异步 bgapi 命令，阻塞直到 BACKGROUND_JOB 事件返回结果。
        线程安全，可从任意线程调用。

        注意：仍然是阻塞调用，适合在线程中使用。
        如果不需要结果，用 send_bgapi_nowait。
        """
        job_uuid = str(uuid_mod.uuid4())
        resp_q: queue.Queue = queue.Queue()

        with self._pending_lock:
            self._pending_bgapi[job_uuid] = resp_q

        try:
            with self._send_lock:
                self._raw_send(f"bgapi {cmd}\nJob-UUID: {job_uuid}\n\n")
            return resp_q.get(timeout=self.BGAPI_TIMEOUT)
        except queue.Empty:
            raise ESLError(f"bgapi timeout: {cmd}")
        finally:
            with self._pending_lock:
                self._pending_bgapi.pop(job_uuid, None)

    def send_bgapi_nowait(self, cmd: str):
        """发送 bgapi 但不等待结果，适合 fire-and-forget 场景。"""
        job_uuid = str(uuid_mod.uuid4())
        with self._send_lock:
            self._raw_send(f"bgapi {cmd}\nJob-UUID: {job_uuid}\n\n")

    # ------------------------------------------------------------------ #
    #  内部：socket 读写
    # ------------------------------------------------------------------ #

    def _raw_send(self, data: str):
        """直接写 socket，调用方负责持有 _send_lock。"""
        self._sock.sendall(data.encode("utf-8"))

    def _send_command_raw(self, cmd: str) -> str:
        """用于 subscribe 等内部命令，不走 api 响应队列。"""
        with self._send_lock:
            self._raw_send(f"{cmd}\n\n")
        # 读取 command/reply，由事件循环处理，这里短暂等待
        time.sleep(0.05)

    def _read_bytes(self, n: int) -> bytes:
        """从缓冲区读取精确 n 字节。"""
        while len(self._buf) < n:
            chunk = self._sock.recv(self.RECV_CHUNK)
            if not chunk:
                raise ConnectionError("ESL socket closed")
            self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n:]
        return data

    def _read_line(self) -> str:
        """从缓冲区读取一行（以 \n 结尾），返回去除首尾空白的字符串。"""
        while b"\n" not in self._buf:
            chunk = self._sock.recv(self.RECV_CHUNK)
            if not chunk:
                raise ConnectionError("ESL socket closed")
            self._buf += chunk
        idx = self._buf.index(b"\n")
        line, self._buf = self._buf[:idx], self._buf[idx + 1:]
        return line.decode("utf-8").strip()

    def _read_headers(self) -> Dict[str, str]:
        """读取 ESL 头部块（以空行结束），返回 key→value 字典。"""
        headers: Dict[str, str] = {}
        while True:
            line = self._read_line()
            if line == "":
                break
            if ": " in line:
                key, val = line.split(": ", 1)
                headers[key.strip()] = unquote(val.strip())
        return headers

    def _read_body(self, headers: Dict[str, str]) -> str:
        """根据 Content-Length 读取消息体。"""
        length = int(headers.get("Content-Length", 0))
        if length <= 0:
            return ""
        return self._read_bytes(length).decode("utf-8")

    # ------------------------------------------------------------------ #
    #  内部：事件循环（唯一读取 socket 的线程）
    # ------------------------------------------------------------------ #

    def _event_loop(self):
        """
        永久循环读取 ESL 消息，按 Content-Type 路由：
          - auth/request       → 忽略（已在 connect 处理）
          - command/reply      → 忽略（subscribe 的确认）
          - api/response       → 填入 _api_response_queue
          - text/event-plain   → 解析事件，分发回调
          - text/disconnect-notice → 触发重连
        """
        while self._running:
            try:
                headers = self._read_headers()
                content_type = headers.get("Content-Type", "")
                body = self._read_body(headers)

                if content_type == "api/response":
                    self._api_response_queue.put(body.strip())

                elif content_type == "command/reply":
                    # subscribe / auth 的确认，忽略即可
                    pass

                elif content_type == "text/event-plain":
                    self._dispatch_event(body)

                elif content_type == "text/disconnect-notice":
                    logger.warning("ESL disconnect notice received")
                    self._connected = False
                    break

                else:
                    # 未知类型，记录但不崩溃
                    if content_type:
                        logger.debug(f"Unhandled Content-Type: {content_type}")

            except ConnectionError as e:
                logger.error(f"ESL read error: {e}")
                self._connected = False
                break
            except Exception as e:
                if self._running:
                    logger.exception(f"ESL event loop error: {e}")
                    time.sleep(0.1)

        logger.warning("ESL event loop exited")

    def _dispatch_event(self, body: str):
        """解析 text/event-plain 消息体，路由到已注册的回调。"""
        # 事件体本身也是 key: value 格式
        event_headers: Dict[str, str] = {}
        for line in body.splitlines():
            if ": " in line:
                k, v = line.split(": ", 1)
                event_headers[k.strip()] = unquote(v.strip())

        event_name = event_headers.get("Event-Name", "")

        # BACKGROUND_JOB：路由给等待的 send_bgapi 调用
        if event_name == "BACKGROUND_JOB":
            job_uuid = event_headers.get("Job-UUID", "")
            job_body = event_headers.get("_body", "").strip()
            with self._pending_lock:
                q = self._pending_bgapi.get(job_uuid)
            if q:
                q.put(job_body)
            return

        # 普通事件：分发到已注册的回调（在线程池中执行，不阻塞读取）
        callbacks = self._event_callbacks.get(event_name, [])
        if callbacks:
            event = {"headers": event_headers, "body": body}
            for cb in callbacks:
                self._callback_executor.submit(self._safe_callback, cb, event)

    @staticmethod
    def _safe_callback(cb: Callable, event: dict):
        try:
            cb(event)
        except Exception as e:
            logger.exception(f"ESL callback error: {e}")


# 延迟导入，避免循环
from concurrent.futures import ThreadPoolExecutor
