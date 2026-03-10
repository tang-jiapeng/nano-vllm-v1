#!/usr/bin/env python3
"""
nano-vllm 交互式聊天客户端
=========================

连接到 nano-vllm API Server，提供终端交互式对话体验，
支持流式输出、多轮对话历史、思考过程过滤。

前置条件：
  先启动服务端：
    python -m nanovllm.entrypoints.serve --model models/Qwen3-0.6B --enforce-eager

用法：
  python chat_client.py
  python chat_client.py --url http://localhost:8000 --max-tokens 200
  python chat_client.py --no-stream          # 非流式模式
  python chat_client.py --show-thinking      # 显示模型思考过程

快捷命令：
  /clear   清除对话历史
  /system  设置系统提示词
  /help    显示帮助
  /quit    退出

依赖：
  pip install httpx   # 或 pip install requests
"""

import argparse
import json
import sys
import time

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    import urllib.error
    import urllib.request

# ── 终端颜色 ──
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def _post_stream_httpx(url, payload):
    """使用 httpx 发送流式 POST 请求，yield 每行 SSE 数据。"""
    with httpx.stream("POST", url, json=payload, timeout=120.0) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    return
                yield json.loads(data)


def _post_stream_urllib(url, payload):
    """使用 stdlib urllib 发送流式 POST 请求（无外部依赖备选）。"""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    return
                yield json.loads(data)


def _post_json(url, payload):
    """发送非流式 POST 请求，返回 JSON。"""
    if HAS_HTTPX:
        resp = httpx.post(url, json=payload, timeout=120.0)
        resp.raise_for_status()
        return resp.json()
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())


def _post_stream(url, payload):
    if HAS_HTTPX:
        return _post_stream_httpx(url, payload)
    return _post_stream_urllib(url, payload)


def _filter_thinking(text: str) -> str:
    """移除 <think>...</think> 标签中的思考内容。"""
    import re

    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def chat_loop(args):
    base_url = args.url.rstrip("/")
    chat_url = f"{base_url}/v1/chat/completions"

    # 检查服务是否可用
    try:
        health = _post_json(f"{base_url}/health", None) if False else None
        if HAS_HTTPX:
            httpx.get(f"{base_url}/health", timeout=5).raise_for_status()
        else:
            urllib.request.urlopen(f"{base_url}/health", timeout=5)
    except Exception:
        print(f"{RED}✗ 无法连接到 {base_url}，请确认服务已启动。{RESET}")
        print(
            f"  启动命令: python -m nanovllm.entrypoints.serve --model <模型路径> --enforce-eager"
        )
        sys.exit(1)

    # 获取模型名称
    try:
        if HAS_HTTPX:
            models_resp = httpx.get(f"{base_url}/v1/models", timeout=5).json()
        else:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as r:
                models_resp = json.loads(r.read().decode())
        model_name = models_resp["data"][0]["id"]
    except Exception:
        model_name = "unknown"

    messages = []
    system_prompt = None

    print(f"\n{BOLD}╭─────────────────────────────────────────╮{RESET}")
    print(f"{BOLD}│  nano-vllm Chat Client                  │{RESET}")
    print(f"{BOLD}│  Model: {CYAN}{model_name:<32s}{RESET}{BOLD}│{RESET}")
    print(f"{BOLD}│  Stream: {'on' if args.stream else 'off':<32s}│{RESET}")
    print(f"{BOLD}╰─────────────────────────────────────────╯{RESET}")
    print(f"{DIM}  输入 /help 查看命令，/quit 或 Ctrl+C 退出{RESET}\n")

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Bye!{RESET}")
            break

        if not user_input:
            continue

        # ── 快捷命令 ──
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd in ("/quit", "/exit", "/q"):
                print(f"{DIM}Bye!{RESET}")
                break
            elif cmd == "/clear":
                messages.clear()
                print(f"{DIM}✓ 对话历史已清除{RESET}\n")
                continue
            elif cmd == "/system":
                system_prompt = user_input[len("/system") :].strip()
                if not system_prompt:
                    system_prompt = None
                    print(f"{DIM}✓ 系统提示词已清除{RESET}\n")
                else:
                    print(f"{DIM}✓ 系统提示词已设置: {system_prompt[:50]}...{RESET}\n")
                continue
            elif cmd == "/history":
                if not messages:
                    print(f"{DIM}  (空){RESET}\n")
                else:
                    for m in messages:
                        role_color = GREEN if m["role"] == "user" else CYAN
                        print(
                            f"  {role_color}{m['role']}{RESET}: {m['content'][:80]}..."
                        )
                    print()
                continue
            elif cmd == "/help":
                print(
                    f"""
  {BOLD}快捷命令:{RESET}
    /clear       清除对话历史
    /system <p>  设置系统提示词
    /history     查看对话历史
    /help        显示此帮助
    /quit        退出
"""
                )
                continue
            else:
                print(f"{DIM}  未知命令: {cmd}，输入 /help 查看帮助{RESET}\n")
                continue

        # ── 构建消息列表 ──
        request_messages = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(messages)
        request_messages.append({"role": "user", "content": user_input})

        payload = {
            "model": model_name,
            "messages": request_messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": args.stream,
        }

        print(f"{CYAN}{BOLD}Assistant > {RESET}", end="", flush=True)

        t0 = time.time()
        full_response = ""
        token_count = 0
        first_token_time = None

        try:
            if args.stream:
                # ── 流式模式 ──
                in_thinking = False
                for chunk in _post_stream(chat_url, payload):
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if not content:
                        continue

                    token_count += 1
                    if first_token_time is None:
                        first_token_time = time.time()

                    full_response += content

                    # 处理 <think> 标签
                    if "<think>" in content:
                        in_thinking = True
                        if args.show_thinking:
                            print(f"{DIM}", end="", flush=True)
                    if "</think>" in content:
                        in_thinking = False
                        if args.show_thinking:
                            print(f"{RESET}", end="", flush=True)
                        continue

                    if in_thinking and not args.show_thinking:
                        continue

                    print(content, end="", flush=True)
            else:
                # ── 非流式模式 ──
                resp = _post_json(chat_url, payload)
                first_token_time = time.time()
                raw_text = resp["choices"][0]["message"]["content"]
                token_count = resp.get("usage", {}).get("completion_tokens", 0)
                full_response = raw_text

                if args.show_thinking:
                    print(raw_text, end="")
                else:
                    print(_filter_thinking(raw_text), end="")

        except Exception as e:
            print(f"\n{RED}✗ 请求失败: {e}{RESET}\n")
            continue

        elapsed = time.time() - t0
        ttft = (first_token_time - t0) * 1000 if first_token_time else 0

        # 统计信息
        clean_response = _filter_thinking(full_response)
        print(
            f"\n{DIM}  [{token_count} tokens, {elapsed:.1f}s, TTFT {ttft:.0f}ms]{RESET}\n"
        )

        # 保存到多轮历史（只保存过滤后的内容）
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": clean_response})

        # 限制历史长度
        if len(messages) > 20:
            messages = messages[-20:]


def main():
    parser = argparse.ArgumentParser(
        description="nano-vllm 交互式聊天客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API 服务地址 (default: http://localhost:8000)",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument(
        "--no-stream", dest="stream", action="store_false", help="禁用流式输出"
    )
    parser.add_argument(
        "--show-thinking", action="store_true", help="显示模型思考过程 (<think> 标签)"
    )
    parser.set_defaults(stream=True)
    args = parser.parse_args()

    chat_loop(args)


if __name__ == "__main__":
    main()
