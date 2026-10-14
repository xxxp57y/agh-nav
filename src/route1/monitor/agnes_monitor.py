#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agnes 视觉大模型语义 Monitor（路线① 第 5 个失败信号来源）。

判决口径对齐 AVERT-VLN 的 ``instruction-execution consistency``：
**只**依据「自然语言导航指令 + 按时间排序的第一人称观测帧」判断导航智能体
此刻是否仍在执行指令所描述的子目标，输出 PASS / LOST。

刻意不使用任何几何量（距离、位移、漂移、碰撞、朝向角、步数、耗时……）：
本项目要论证「几何判据 vs 语义判据」的差异，因此语义 Monitor 绝不能偷看几何信号。

设计要点
--------
* 纯标准库（``urllib.request`` + ``json`` + ``base64`` + ``hashlib``），Python 3.9 兼容。
* key 来源：环境变量 ``AGNES_API_KEY``，否则 ``/srv/agh-demo/.secrets/agnes-key.txt``。
  任何路径都不打印 key 内容（只报告是否配置）。
* 端点 ``https://api.agnes-ai.cn/v1/chat/completions``（OpenAI 兼容）。
  主模型 ``agnes-3.0-flash``（视觉），备选 ``agnes-2.5-flash``。
* **单请求最多 4 张图**（该上限跨历史 + 工具结果累计）。接口允许最多 8 帧输入，
  发送前会等距下采样到 ≤4 张，且**始终保留最后一张（当前观测）**。
* 强制 JSON 输出：优先 ``response_format={"type":"json_object"}``；若网关报 400 不支持，
  自动降级为「只输出 JSON」的 prompt + 稳健解析；解析失败抛 ``MonitorParseError`` 并重试一次。
* 磁盘缓存：key = sha256(模型名 + prompt 版本 + instruction + 各帧文件内容 sha1)。
* 每次调用记录 latency_ms / usage / model / cache_hit，append 到 ``monitor_calls.jsonl``。

CLI
---
    python3 agnes_monitor.py --instruction "..." --frames a.jpg b.jpg ...
    python3 agnes_monitor.py --selftest
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 常量与路径
# --------------------------------------------------------------------------

PROMPT_VERSION = "v1"

API_URL = "https://api.agnes-ai.cn/v1/chat/completions"
DEFAULT_MODEL = "agnes-3.0-flash"
FALLBACK_MODEL = "agnes-2.5-flash"

MAX_FRAMES = 8            # 接口允许的输入帧数上限
MAX_IMAGES_PER_REQUEST = 4  # Agnes 单请求图片数硬上限（跨历史 + 工具结果累计）

ROOT = os.environ.get("ROUTE1_ROOT", "/srv/agh-demo")
SECRETS_FILE = os.environ.get("AGNES_KEY_FILE", os.path.join(ROOT, ".secrets", "agnes-key.txt"))
CACHE_DIR = os.environ.get("AGNES_CACHE_DIR", os.path.join(ROOT, "logs", "route1", "monitor_cache"))
CALLS_LOG = os.environ.get("AGNES_CALLS_LOG", os.path.join(ROOT, "logs", "route1", "monitor_calls.jsonl"))

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_TOKENS = 1024
VERDICTS = ("PASS", "LOST")

_DEBUG = bool(os.environ.get("AGNES_MONITOR_DEBUG"))


def _log(msg: str) -> None:
    """调试日志走 stderr；绝不打印 key。"""
    if _DEBUG:
        sys.stderr.write("[agnes_monitor] %s\n" % msg)
        sys.stderr.flush()


# --------------------------------------------------------------------------
# 异常
# --------------------------------------------------------------------------


class MonitorError(Exception):
    """Monitor 基类异常。"""


class MonitorConfigError(MonitorError):
    """配置/密钥缺失。"""


class MonitorAPIError(MonitorError):
    """HTTP 层错误（含 401/403/429/5xx/超时）。"""

    def __init__(self, message: str, status: Optional[int] = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class MonitorParseError(MonitorError):
    """模型返回内容无法解析为符合契约的 JSON。"""

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


# --------------------------------------------------------------------------
# 密钥
# --------------------------------------------------------------------------


def load_api_key(explicit: Optional[str] = None) -> str:
    """按 显式参数 → 环境变量 → key 文件 的顺序取 key。不打印内容。"""
    if explicit:
        return explicit.strip()
    env = os.environ.get("AGNES_API_KEY")
    if env and env.strip():
        return env.strip()
    if os.path.exists(SECRETS_FILE):
        with open(SECRETS_FILE, "r", encoding="utf-8") as fh:
            key = fh.read().strip()
        if key:
            return key
    raise MonitorConfigError(
        "找不到 Agnes API key：既无环境变量 AGNES_API_KEY，也无可用文件 %s" % SECRETS_FILE
    )


def describe_key(key: str) -> str:
    """只暴露长度与前缀，用于自检输出。"""
    if not key:
        return "configured=False"
    return "configured=%s" % bool(key)


# --------------------------------------------------------------------------
# 图像编码
# --------------------------------------------------------------------------

_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def guess_mime(path: str) -> str:
    return _MIME_BY_EXT.get(os.path.splitext(path)[1].lower(), "image/jpeg")


def read_image(path: str) -> Tuple[str, str, int]:
    """读图 → (base64_str, content_sha1_hex, n_bytes)。"""
    if not os.path.isfile(path):
        raise MonitorConfigError("帧文件不存在：%s" % path)
    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw:
        raise MonitorConfigError("帧文件为空：%s" % path)
    return base64.b64encode(raw).decode("ascii"), hashlib.sha1(raw).hexdigest(), len(raw)


def select_images(n_frames: int, max_images: int = MAX_IMAGES_PER_REQUEST) -> List[int]:
    """从 n_frames 帧中选出要真正发送的下标：等距采样，且**必含最后一帧**。

    Agnes 单请求 ≤4 张图（跨历史 + 工具结果累计），所以超过上限必须下采样。
    返回升序且去重的下标列表。
    """
    if n_frames <= 0:
        return []
    if n_frames <= max_images:
        return list(range(n_frames))
    if max_images == 1:
        return [n_frames - 1]
    idx = []
    for i in range(max_images):
        # linspace(0, n-1, max_images)
        pos = i * (n_frames - 1) / float(max_images - 1)
        idx.append(int(round(pos)))
    idx[-1] = n_frames - 1  # 强制保留当前观测
    out: List[int] = []
    for i in idx:
        if i not in out:
            out.append(i)
    return sorted(out)


# --------------------------------------------------------------------------
# 缓存
# --------------------------------------------------------------------------


def compute_cache_key(model: str, prompt_version: str, instruction: str,
                      frame_sha1s: Sequence[str]) -> str:
    """key = sha256(模型名 + prompt 版本 + instruction + 各帧文件内容的 sha1)。"""
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt_version.encode("utf-8"))
    h.update(b"\x00")
    h.update(instruction.encode("utf-8"))
    for s in frame_sha1s:
        h.update(b"\x00")
        h.update(s.encode("utf-8"))
    return h.hexdigest()


class DiskCache(object):
    """极简磁盘缓存：``<CACHE_DIR>/<key>.json``。"""

    def __init__(self, cache_dir: str = CACHE_DIR) -> None:
        self.cache_dir = cache_dir

    def path_for(self, key: str) -> str:
        return os.path.join(self.cache_dir, key + ".json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self.path_for(key)
        if not os.path.isfile(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception as exc:  # 缓存损坏不应让主流程失败
            _log("缓存损坏，忽略：%s (%r)" % (p, exc))
            return None
        if not isinstance(obj, dict):
            return None
        return obj

    def put(self, key: str, payload: Dict[str, Any]) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        p = self.path_for(key)
        tmp = p + ".tmp.%d" % os.getpid()
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, p)
        return p


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是机器人导航任务中的【语义 Monitor】，只做 instruction-execution consistency 判断。\n"
    "你只依据【自然语言导航指令】与【按时间顺序排列的第一人称观测帧】做判断。"
    "最后一张图 = 当前观测。\n"
    "\n"
    "硬性规则（违反即视为无效判断）：\n"
    "1. 严禁使用或推测任何几何量：距离、位移、漂移、碰撞、朝向角、速度、里程计、步数、耗时、"
    "画面运动幅度等一律不作为证据。\n"
    "2. 不要因为「看起来移动很慢 / 场景变化不大 / 似乎停了很久」就判 LOST；只比较视觉语义与指令是否一致。\n"
    "3. 若图像信息不足以判断，请给出较低的 confidence（可以 <0.5），但 verdict 仍必须是 PASS 或 LOST。\n"
    "\n"
    "判决口径：\n"
    "- PASS：当前观测的语义场景与朝向仍与指令中「此刻应处于的子目标」一致。\n"
    "- LOST：当前观测与指令明显矛盾，其偏航性质属于以下之一：走错方向 / 走过头 / 漏掉转折 / 停在错误位置。\n"
    "\n"
    "输出要求：**只输出一个 JSON 对象**，不要 markdown 代码块，不要任何解释文字。字段如下：\n"
    "{\n"
    '  "verdict": "PASS" 或 "LOST",\n'
    '  "confidence": 0.0 到 1.0 之间的数字,\n'
    '  "reason_l1": "≤40字，一句话说明智能体此刻是否仍在执行指令所描述的子目标",\n'
    '  "evidence": ["≤3 条具体视觉证据，必须带方位/物体名"],\n'
    '  "l2_diagnosis": "两段式：① 偏航性质（走错方向/走过头/漏掉转折/停在错误位置）'
    ' ② 建议恢复动作类型（重新规划/局部脱离/重新解析指令/请求澄清）；PASS 也要写，可写「①无显著偏航 ②维持当前规划」",\n'
    '  "subgoal_guess": "当前应处于的指令子目标片段（从指令原文摘录）"\n'
    "}"
)

JSON_ONLY_SUFFIX = "\n\n再次强调：只输出一个 JSON 对象，不要 markdown 代码块，不要前后缀说明文字。"


def build_user_prompt(instruction: str, n_frames: int, n_images: int,
                      history: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    """构造文本部分。图片以 image_url 片段追加在后面。"""
    parts: List[str] = []
    parts.append("== 导航指令（原文，勿改写）==")
    parts.append(instruction.strip())
    parts.append("")
    parts.append("== 观测帧 ==")
    parts.append(
        "共提供 %d 帧第一人称观测，按时间升序排列（最早 → 最新）。"
        "其中最后 1 帧是**当前观测**。" % n_frames
    )
    if n_images < n_frames:
        parts.append(
            "注意：受单请求图片数上限约束，仅附带其中 %d 张代表帧，"
            "且**必含最后一张当前观测**；中间帧被跳过不代表时间连续。" % n_images
        )
    parts.append("")
    if history:
        parts.append("== 此前 Monitor 判决（仅作连续性参考，不是几何证据，也不是图像）==")
        for i, h in enumerate(history):
            if not isinstance(h, dict):
                parts.append("- #%d %s" % (i + 1, str(h)))
                continue
            parts.append(
                "- #%d verdict=%s confidence=%s subgoal=%s reason=%s" % (
                    i + 1,
                    h.get("verdict", "?"),
                    h.get("confidence", "?"),
                    str(h.get("subgoal_guess", ""))[:80],
                    str(h.get("reason_l1", ""))[:80],
                )
            )
        parts.append("")
    parts.append("请按上述 JSON 契约输出，判断智能体此刻是否仍在执行指令所描述的子目标。")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# 稳健 JSON 解析
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def robust_json_parse(text: str) -> Dict[str, Any]:
    """从模型回复中抠出 JSON 对象。

    依次尝试：整体解析 → ``` 代码块 → 第一个平衡花括号块。
    """
    if text is None:
        raise MonitorParseError("模型返回内容为空", "")
    s = text.strip()
    if not s:
        raise MonitorParseError("模型返回内容为空字符串", text)

    candidates: List[str] = [s]
    for m in _FENCE_RE.finditer(s):
        candidates.append(m.group(1).strip())
    cand = extract_first_json_object(s)
    if cand:
        candidates.append(cand)

    last_err: Optional[Exception] = None
    for c in candidates:
        if not c:
            continue
        try:
            obj = json.loads(c)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
        if isinstance(obj, dict):
            return obj
        last_err = ValueError("顶层不是 JSON 对象，而是 %s" % type(obj).__name__)
    raise MonitorParseError(
        "无法从模型回复中解析出 JSON 对象（最后一次错误：%r）" % (last_err,), text
    )


def extract_first_json_object(s: str) -> Optional[str]:
    """扫描出第一个花括号平衡的 JSON 对象子串（跳过字符串内的括号）。"""
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        start = s.find("{", start + 1)
    return None


# --------------------------------------------------------------------------
# 契约校验
# --------------------------------------------------------------------------


def _coerce_confidence(v: Any) -> float:
    if isinstance(v, bool):
        raise MonitorParseError("confidence 是 bool，不是数字", repr(v))
    if isinstance(v, (int, float)):
        c = float(v)
    elif isinstance(v, str):
        try:
            c = float(v.strip().rstrip("%"))
        except Exception:
            raise MonitorParseError("confidence 无法解析为数字：%r" % (v,), repr(v))
    else:
        raise MonitorParseError("confidence 类型非法：%r" % (v,), repr(v))
    # 容忍百分数写法（85 -> 0.85）；直接把 85 夹到 1.0 会虚高置信度
    if 1.0 < c <= 100.0:
        c = c / 100.0
    return max(0.0, min(1.0, c))


def validate_verdict(obj: Dict[str, Any]) -> Dict[str, Any]:
    """校验并规范化模型输出；不合契约直接抛 MonitorParseError。"""
    if not isinstance(obj, dict):
        raise MonitorParseError("顶层不是 JSON 对象", repr(obj))

    verdict = obj.get("verdict")
    if not isinstance(verdict, str):
        raise MonitorParseError("缺少 verdict 字段或类型非法：%r" % (verdict,), json.dumps(obj, ensure_ascii=False))
    verdict = verdict.strip().upper()
    if verdict not in VERDICTS:
        # 容忍 "VERDICT: PASS" / "pass." 之类
        for v in VERDICTS:
            if v in verdict:
                verdict = v
                break
        else:
            raise MonitorParseError("verdict 非法：%r" % (obj.get("verdict"),), json.dumps(obj, ensure_ascii=False))

    if "confidence" not in obj:
        raise MonitorParseError("缺少 confidence 字段", json.dumps(obj, ensure_ascii=False))
    confidence = _coerce_confidence(obj["confidence"])

    def _str_field(name: str, required: bool = True) -> str:
        v = obj.get(name)
        if v is None:
            if required:
                raise MonitorParseError("缺少 %s 字段" % name, json.dumps(obj, ensure_ascii=False))
            return ""
        if isinstance(v, (list, tuple)):
            v = " ".join(str(x) for x in v)
        if not isinstance(v, str):
            v = str(v)
        return v.strip()

    reason_l1 = _str_field("reason_l1")
    if len(reason_l1) > 80:  # 契约是 ≤40 字，超一点容忍但截断
        reason_l1 = reason_l1[:80]

    evidence_raw = obj.get("evidence")
    if evidence_raw is None:
        evidence: List[str] = []
    elif isinstance(evidence_raw, str):
        evidence = [evidence_raw.strip()] if evidence_raw.strip() else []
    elif isinstance(evidence_raw, (list, tuple)):
        evidence = [str(x).strip() for x in evidence_raw if str(x).strip()]
    else:
        raise MonitorParseError("evidence 类型非法：%r" % (type(evidence_raw).__name__,),
                                json.dumps(obj, ensure_ascii=False))
    evidence = evidence[:3]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason_l1": reason_l1,
        "evidence": evidence,
        "l2_diagnosis": _str_field("l2_diagnosis", required=False),
        "subgoal_guess": _str_field("subgoal_guess", required=False),
    }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def post_chat(payload: Dict[str, Any], api_key: str,
              timeout: float = DEFAULT_TIMEOUT_S) -> Tuple[Dict[str, Any], int, int]:
    """POST 到 Agnes 网关。返回 (响应 JSON, http_status, elapsed_ms)。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % api_key,
            "Accept": "application/json",
        },
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read() if hasattr(exc, "read") else b""
        text = raw.decode("utf-8", "replace")
        elapsed = int((time.time() - t0) * 1000)
        raise MonitorAPIError(
            "HTTP %s from %s (%d ms): %s" % (exc.code, API_URL, elapsed, text[:800]),
            status=getattr(exc, "code", None),
            body=text,
        )
    except urllib.error.URLError as exc:
        elapsed = int((time.time() - t0) * 1000)
        raise MonitorAPIError("网络错误（%d ms）：%r" % (elapsed, exc.reason))
    except Exception as exc:  # noqa: BLE001
        elapsed = int((time.time() - t0) * 1000)
        raise MonitorAPIError("请求异常（%d ms）：%r" % (elapsed, exc))

    elapsed = int((time.time() - t0) * 1000)
    try:
        obj = json.loads(raw.decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001
        raise MonitorAPIError("响应不是合法 JSON（HTTP %s, %d ms）：%r" % (status, elapsed, exc))
    if not isinstance(obj, dict):
        raise MonitorAPIError("响应顶层不是对象（HTTP %s）：%s" % (status, type(obj).__name__))
    return obj, status, elapsed


def extract_content(resp: Dict[str, Any]) -> str:
    """从 OpenAI 兼容响应里取正文。"""
    if "error" in resp and resp["error"]:
        raise MonitorAPIError("网关返回 error 字段：%s" % json.dumps(resp["error"], ensure_ascii=False)[:600])
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        raise MonitorAPIError("响应缺少 choices：%s" % json.dumps(resp, ensure_ascii=False)[:600])
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        raise MonitorAPIError("choices[0].message 非法：%s" % json.dumps(choices[0], ensure_ascii=False)[:600])
    content = msg.get("content")
    if isinstance(content, list):
        # 少数实现返回 content 分段数组
        buf = []
        for seg in content:
            if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                buf.append(seg["text"])
        content = "".join(buf)
    if not isinstance(content, str):
        raise MonitorAPIError("message.content 非字符串：%r" % (type(content).__name__,))
    return content


# --------------------------------------------------------------------------
# 调用日志
# --------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def append_call_log(record: Dict[str, Any], calls_log: str = CALLS_LOG) -> None:
    """append 一行 JSONL。写日志失败不应让主流程失败。"""
    try:
        d = os.path.dirname(calls_log)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(calls_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001
        _log("写调用日志失败：%r" % (exc,))


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


def monitor(instruction: str,
            frames: Sequence[str],
            history: Optional[Sequence[Dict[str, Any]]] = None,
            model: str = DEFAULT_MODEL,
            temperature: Optional[float] = 0.0,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            timeout: float = DEFAULT_TIMEOUT_S,
            use_cache: bool = True,
            api_key: Optional[str] = None,
            force_response_format: Optional[bool] = None,
            cache_dir: str = CACHE_DIR,
            calls_log: str = CALLS_LOG,
            extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """调用 Agnes 视觉模型做一次语义 Monitor 判决。

    参数
    ----
    instruction : 自然语言导航指令。
    frames      : 按时间升序的图片路径，最多 ``MAX_FRAMES`` 张，最后一张 = 当前观测。
    history     : 此前若干次 Monitor 判决（dict 列表），仅作文本上下文。
    force_response_format : True 强制带 response_format；False 不带；None = 自动（带，失败降级）。

    返回
    ----
    dict：契约 6 字段 + ``model`` / ``latency_ms`` / ``usage`` / ``cache_hit`` /
    ``n_frames`` / ``n_images_sent`` / ``cache_key`` / ``response_format_used`` / ``attempts``。
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise MonitorConfigError("instruction 必须是非空字符串")
    frames = [str(f) for f in (frames or [])]
    if not frames:
        raise MonitorConfigError("frames 至少需要 1 张图")
    if len(frames) > MAX_FRAMES:
        raise MonitorConfigError(
            "frames 最多 %d 张，收到 %d 张" % (MAX_FRAMES, len(frames))
        )

    key = load_api_key(api_key)
    sent_idx = select_images(len(frames))
    n_images = len(sent_idx)

    frame_shas: List[str] = []
    b64_by_idx: Dict[int, str] = {}
    nbytes_total = 0
    for i in sent_idx:
        b64, sha1, nbytes = read_image(frames[i])
        b64_by_idx[i] = b64
        frame_shas.append(sha1)
        nbytes_total += nbytes
    # 缓存 key 必须覆盖全部输入帧（而不只是发送的子集），否则换掉被跳过的帧会误命中
    all_shas: List[str] = []
    for f in frames:
        if os.path.isfile(f):
            with open(f, "rb") as fh:
                all_shas.append(hashlib.sha1(fh.read()).hexdigest())
        else:
            all_shas.append("missing:" + f)
    cache_key = compute_cache_key(model, PROMPT_VERSION, instruction, all_shas)

    cache = DiskCache(cache_dir)
    if use_cache:
        hit = cache.get(cache_key)
        if hit is not None and hit.get("result"):
            res = dict(hit["result"])
            res["cache_hit"] = True
            res["latency_ms"] = 0
            res["attempts"] = 0
            res["n_frames"] = len(frames)
            res["n_images_sent"] = n_images
            res["cache_key"] = cache_key
            append_call_log({
                "ts": now_iso(),
                "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16],
                "n_frames": len(frames),
                "n_images_sent": n_images,
                "verdict": res.get("verdict"),
                "confidence": res.get("confidence"),
                "latency_ms": 0,
                "usage": None,
                "model": model,
                "cache_hit": True,
                "error": None,
            }, calls_log)
            return res

    user_text = build_user_prompt(instruction, len(frames), n_images, history)

    def _payload(with_rf: bool, json_only: bool) -> Dict[str, Any]:
        text = user_text + (JSON_ONLY_SUFFIX if json_only else "")
        c: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for i in sent_idx:
            c.append({
                "type": "image_url",
                "image_url": {"url": "data:%s;base64,%s" % (guess_mime(frames[i]), b64_by_idx[i])},
            })
        p: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": c},
            ],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            p["temperature"] = temperature
        if with_rf:
            p["response_format"] = {"type": "json_object"}
        if extra:
            p.update(extra)
        return p

    attempts: List[Dict[str, Any]] = []
    last_err: Optional[Exception] = None
    t_start = time.time()

    # with_rf 计划：True（若允许）→ False（降级）
    if force_response_format is True:
        rf_plan = [True]
    elif force_response_format is False:
        rf_plan = [False]
    else:
        rf_plan = [True, False]

    for phase, with_rf in enumerate(rf_plan):
        # 每个 response_format 形态下：最多尝试 2 次（第 2 次 = 解析失败重试）
        for attempt in range(2):
            json_only = (attempt == 1) or (not with_rf)
            payload = _payload(with_rf, json_only)
            t0 = time.time()
            try:
                resp, status, elapsed = post_chat(payload, key, timeout=timeout)
                text = extract_content(resp)
                obj = robust_json_parse(text)
                result = validate_verdict(obj)
                usage = resp.get("usage")
                latency_ms = int((time.time() - t_start) * 1000)
                attempts.append({
                    "response_format": with_rf, "attempt": attempt + 1,
                    "ok": True, "latency_ms": elapsed, "status": status,
                })
                out = dict(result)
                out.update({
                    "model": model,
                    "latency_ms": latency_ms,
                    "api_latency_ms": elapsed,
                    "http_status": status,
                    "usage": usage,
                    "cache_hit": False,
                    "n_frames": len(frames),
                    "n_images_sent": n_images,
                    "sent_frame_indices": sent_idx,
                    "cache_key": cache_key,
                    "response_format_used": with_rf,
                    "attempts": len(attempts),
                    "attempt_trace": attempts,
                    "prompt_version": PROMPT_VERSION,
                    "n_frames_bytes": nbytes_total,
                })
                # 只有真正成功的判决才写缓存
                cache.put(cache_key, {"result": result, "model": model,
                                      "prompt_version": PROMPT_VERSION,
                                      "ts": now_iso(), "usage": usage})
                append_call_log({
                    "ts": now_iso(),
                    "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16],
                    "n_frames": len(frames),
                    "n_images_sent": n_images,
                    # 帧身份：让"同一输入不同判决"这类结论可以被逐条核对
                    "frame_names": [os.path.basename(f) for f in frames],
                    "frame_sha1_16": [s[:16] for s in all_shas],
                    "sent_indices": sent_idx,
                    "verdict": out["verdict"],
                    "confidence": out["confidence"],
                    "latency_ms": latency_ms,
                    "api_latency_ms": elapsed,
                    "usage": usage,
                    "model": model,
                    "cache_hit": False,
                    "response_format_used": with_rf,
                    "attempts": len(attempts),
                    "http_status": status,
                    "error": None,
                }, calls_log)
                return out
            except MonitorParseError as exc:
                last_err = exc
                attempts.append({
                    "response_format": with_rf, "attempt": attempt + 1,
                    "ok": False, "error": "parse: %s" % exc,
                    "raw_head": (exc.raw_text or "")[:300],
                })
                _log("解析失败（rf=%s, attempt=%d）：%s" % (with_rf, attempt + 1, exc))
                continue  # 重试一次
            except MonitorAPIError as exc:
                last_err = exc
                attempts.append({
                    "response_format": with_rf, "attempt": attempt + 1,
                    "ok": False, "error": "api: %s" % exc,
                    "status": exc.status,
                })
                _log("API 错误（rf=%s, attempt=%d, status=%s）：%s" % (with_rf, attempt + 1, exc.status, exc))
                # 400 且疑似 response_format 不支持 → 跳到下一个 rf 形态
                body = (exc.body or "").lower()
                if exc.status == 400 and ("response_format" in body or "json_object" in body):
                    break
                if exc.status in (401, 403):
                    # 鉴权问题重试无意义
                    break
                continue
        else:
            continue
        # break 到这里：如果是因为 rf 不支持，继续下一个 rf 形态
        if force_response_format is not None:
            break

    latency_ms = int((time.time() - t_start) * 1000)
    err_msg = str(last_err) if last_err else "未知错误"
    append_call_log({
        "ts": now_iso(),
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16],
        "n_frames": len(frames),
        "n_images_sent": n_images,
        "verdict": None,
        "confidence": None,
        "latency_ms": latency_ms,
        "usage": None,
        "model": model,
        "cache_hit": False,
        "error": err_msg[:600],
    }, calls_log)
    if isinstance(last_err, MonitorAPIError):
        raise last_err
    raise MonitorParseError("Monitor 调用失败（最终错误：%s）" % err_msg)


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------


def _make_synthetic_frames(out_dir: str) -> List[str]:
    """用 PIL 造 3 张示意房间图（无 PIL 时退回手写最小 PNG）。"""
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    specs = [
        ("living_room", (210, 180, 140), (139, 69, 19), "sofa"),
        ("corridor", (160, 160, 170), (100, 100, 110), "door"),
        ("kitchen", (200, 210, 220), (70, 70, 80), "counter"),
    ]
    try:
        from PIL import Image, ImageDraw  # type: ignore

        for name, wall, floor, label in specs:
            im = Image.new("RGB", (640, 480), wall)
            d = ImageDraw.Draw(im)
            d.rectangle([0, 320, 640, 480], fill=floor)          # 地板
            d.rectangle([480, 120, 620, 400], fill=(240, 240, 245))  # 门/窗
            d.rectangle([80, 200, 320, 380], fill=(120, 60, 40))     # 家具
            d.text((20, 20), "%s / %s" % (name, label), fill=(20, 20, 20))
            p = os.path.join(out_dir, "synth_%s.jpg" % name)
            im.save(p, quality=85)
            paths.append(p)
    except Exception as exc:  # noqa: BLE001
        _log("PIL 不可用（%r），退回手写 PNG" % (exc,))
        for name, wall, _floor, _label in specs:
            p = os.path.join(out_dir, "synth_%s.png" % name)
            _write_solid_png(p, 8, 8, wall)
            paths.append(p)
    return paths


def _write_solid_png(path: str, w: int, h: int, rgb: Tuple[int, int, int]) -> None:
    import struct
    import zlib

    raw = b""
    row = bytes(rgb) * w
    for _ in range(h):
        raw += b"\x00" + row

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)


def selftest(online: bool = True, workdir: Optional[str] = None) -> int:
    """内置最小自检。返回 0 = 全部通过。"""
    ok = True
    workdir = workdir or os.path.join(ROOT, "out", "route1", "monitor_selftest")
    print("== agnes_monitor selftest ==")
    print("python:", sys.version.split()[0])
    print("PROMPT_VERSION:", PROMPT_VERSION)
    print("API_URL:", API_URL)
    print("DEFAULT_MODEL:", DEFAULT_MODEL)
    print("MAX_FRAMES:", MAX_FRAMES, " MAX_IMAGES_PER_REQUEST:", MAX_IMAGES_PER_REQUEST)
    print("CACHE_DIR:", CACHE_DIR)
    print("CALLS_LOG:", CALLS_LOG)

    # 1) key
    try:
        key = load_api_key()
        print("[PASS] key: %s (source=%s)" % (
            describe_key(key),
            "env" if os.environ.get("AGNES_API_KEY") else SECRETS_FILE))
    except MonitorConfigError as exc:
        print("[FAIL] key: %s" % exc)
        ok = False

    # 2) 图像下采样
    cases = [(1, [0]), (4, [0, 1, 2, 3]), (5, [0, 1, 3, 4]), (8, [0, 2, 5, 7])]
    for n, want in cases:
        got = select_images(n)
        flag = "PASS" if got == want else "FAIL"
        if got != want:
            ok = False
        print("[%s] select_images(%d) = %s (want %s)" % (flag, n, got, want))

    # 3) 稳健解析
    samples = [
        ('{"verdict":"PASS","confidence":0.9,"reason_l1":"x","evidence":["a"],"l2_diagnosis":"y","subgoal_guess":"z"}',
         "PASS", 0.9),
        ('```json\n{"verdict":"LOST","confidence":"0.7","reason_l1":"x","evidence":"only one",'
         '"l2_diagnosis":"y","subgoal_guess":"z"}\n```', "LOST", 0.7),
        ('好的，结果如下：\n{"verdict": "lost", "confidence": 85, "reason_l1": "x", "evidence": ["a","b"],'
         ' "l2_diagnosis": "y", "subgoal_guess": "z"}\n以上。', "LOST", 0.85),
        ('{"note": "no verdict here"}', None, None),
    ]
    for raw, want_v, want_c in samples:
        try:
            got = validate_verdict(robust_json_parse(raw))
            flag = "PASS" if (got["verdict"] == want_v and (want_c is None or abs(got["confidence"] - want_c) < 1e-9)) else "FAIL"
            if flag == "FAIL":
                ok = False
            print("[%s] parse -> %s %.2f" % (flag, got["verdict"], got["confidence"]))
        except MonitorParseError as exc:
            if want_v is None:
                print("[PASS] parse correctly rejected: %s" % str(exc)[:60])
            else:
                ok = False
                print("[FAIL] parse raised unexpectedly: %s" % exc)

    # 4) cache key 稳定性
    k1 = compute_cache_key("m", "v1", "ins", ["a", "b"])
    k2 = compute_cache_key("m", "v1", "ins", ["a", "b"])
    k3 = compute_cache_key("m", "v1", "ins", ["a", "c"])
    if k1 == k2 and k1 != k3 and len(k1) == 64:
        print("[PASS] cache key 稳定且对帧内容敏感")
    else:
        ok = False
        print("[FAIL] cache key: %s %s %s" % (k1[:12], k2[:12], k3[:12]))

    # 5) 磁盘缓存往返
    try:
        c = DiskCache(os.path.join(workdir, "cache"))
        c.put("deadbeef", {"result": {"verdict": "PASS"}})
        back = c.get("deadbeef")
        if back and back["result"]["verdict"] == "PASS":
            print("[PASS] DiskCache 往返")
        else:
            ok = False
            print("[FAIL] DiskCache 往返：%r" % (back,))
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("[FAIL] DiskCache：%r" % (exc,))

    # 6) 合成帧
    try:
        frames = _make_synthetic_frames(workdir)
        print("[PASS] 合成帧 %d 张：%s" % (len(frames), ", ".join(os.path.basename(f) for f in frames)))
    except Exception as exc:  # noqa: BLE001
        ok = False
        frames = []
        print("[FAIL] 合成帧：%r" % (exc,))

    # 7) prompt 构造
    try:
        p = build_user_prompt("go to the kitchen", 8, 4, [{"verdict": "PASS", "confidence": 0.8}])
        assert "go to the kitchen" in p and "8 帧" in p and "仅附带其中 4 张" in p
        print("[PASS] prompt 构造（含下采样提示与 history）")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("[FAIL] prompt 构造：%r" % (exc,))

    # 8) 在线连通
    if online and frames:
        print("-- 在线调用（1 帧，合成图）--")
        try:
            r = monitor("Walk into the living room and stop.",
                        [frames[0]], use_cache=False)
            print("[PASS] 在线调用 verdict=%s confidence=%.2f latency_ms=%s" % (
                r["verdict"], r["confidence"], r["latency_ms"]))
            print("       usage=%s" % json.dumps(r.get("usage"), ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            ok = False
            print("[FAIL] 在线调用：%r" % (exc,))

    print("== selftest %s ==" % ("ALL PASS" if ok else "HAS FAILURES"))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Agnes 视觉大模型语义 Monitor（instruction-execution consistency）")
    ap.add_argument("--instruction", "-i", help="自然语言导航指令")
    ap.add_argument("--frames", "-f", nargs="*", default=[], help="图片路径，时间升序，最后一张=当前观测")
    ap.add_argument("--history", help="此前 Monitor 判决的 JSON 文件（list[dict] 或 JSONL）")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--no-cache", action="store_true", help="跳过磁盘缓存")
    ap.add_argument("--no-response-format", action="store_true", help="不发送 response_format")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--offline", action="store_true", help="selftest 时跳过在线调用")
    ap.add_argument("--pretty", action="store_true", help="缩进输出 JSON")
    ap.add_argument("--trace", action="store_true", help="输出 attempt_trace 等调试字段")
    return ap


def _load_history(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        txt = fh.read().strip()
    if not txt:
        return None
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:  # noqa: BLE001
        pass
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.selftest:
        return selftest(online=not args.offline)
    if not args.instruction:
        build_argparser().print_help()
        return 2
    try:
        res = monitor(
            instruction=args.instruction,
            frames=args.frames,
            history=_load_history(args.history),
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            use_cache=not args.no_cache,
            force_response_format=False if args.no_response_format else None,
        )
    except MonitorError as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 1
    if not args.trace:
        res = {k: v for k, v in res.items() if k != "attempt_trace"}
    print(json.dumps(res, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
