"""
OpenAI 兼容的请求/响应数据结构。

支持的 API：
    /v1/completions         - 文本补全（流式/非流式）
    /v1/chat/completions    - 对话补全（流式/非流式）
"""

import time
from dataclasses import dataclass, field
from typing import Literal

# ══════════════════════════════════════════════════
# /v1/completions
# ══════════════════════════════════════════════════


@dataclass
class CompletionRequest:
    """OpenAI /v1/completions 请求格式。"""

    model: str = ""
    prompt: str | list[int] = ""
    max_tokens: int = 64
    temperature: float = 1.0
    stream: bool = False
    stop: list[str] | None = None
    n: int = 1


@dataclass
class CompletionChoice:
    index: int
    text: str
    finish_reason: str | None = None


@dataclass
class CompletionResponse:
    id: str
    object: str = "text_completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[CompletionChoice] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


@dataclass
class CompletionStreamChoice:
    index: int
    text: str
    finish_reason: str | None = None


@dataclass
class CompletionStreamResponse:
    id: str
    object: str = "text_completion.chunk"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[CompletionStreamChoice] = field(default_factory=list)


# ══════════════════════════════════════════════════
# /v1/chat/completions
# ══════════════════════════════════════════════════


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str = ""


@dataclass
class ChatCompletionRequest:
    """OpenAI /v1/chat/completions 请求格式。"""

    model: str = ""
    messages: list[ChatMessage] = field(default_factory=list)
    max_tokens: int = 64
    temperature: float = 1.0
    stream: bool = False
    stop: list[str] | None = None


@dataclass
class ChatCompletionChoice:
    index: int
    message: ChatMessage
    finish_reason: str | None = None


@dataclass
class ChatCompletionResponse:
    id: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


@dataclass
class ChatCompletionStreamDelta:
    role: str | None = None
    content: str | None = None


@dataclass
class ChatCompletionStreamChoice:
    index: int
    delta: ChatCompletionStreamDelta
    finish_reason: str | None = None


@dataclass
class ChatCompletionStreamResponse:
    id: str
    object: str = "chat.completion.chunk"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionStreamChoice] = field(default_factory=list)
