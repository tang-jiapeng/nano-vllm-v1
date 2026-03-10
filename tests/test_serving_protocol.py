"""
OpenAI Protocol 数据结构测试。

不需要 GPU，纯 Python 验证序列化行为。
"""

import json
from dataclasses import asdict

import pytest

from nanovllm.serving.openai_protocol import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionChoice,
    CompletionResponse,
    CompletionStreamChoice,
    CompletionStreamResponse,
)


class TestCompletionProtocol:
    def test_completion_response_serialization(self):
        resp = CompletionResponse(
            id="cmpl-test123",
            model="Qwen3-0.6B",
            choices=[CompletionChoice(index=0, text="Hello!", finish_reason="stop")],
            usage={"completion_tokens": 5},
        )
        d = asdict(resp)
        j = json.dumps(d)
        parsed = json.loads(j)

        assert parsed["id"] == "cmpl-test123"
        assert parsed["object"] == "text_completion"
        assert parsed["model"] == "Qwen3-0.6B"
        assert len(parsed["choices"]) == 1
        assert parsed["choices"][0]["text"] == "Hello!"
        assert parsed["choices"][0]["finish_reason"] == "stop"

    def test_completion_stream_response(self):
        resp = CompletionStreamResponse(
            id="cmpl-stream",
            model="Qwen3-0.6B",
            choices=[CompletionStreamChoice(index=0, text="Hi", finish_reason=None)],
        )
        d = asdict(resp)
        assert d["object"] == "text_completion.chunk"
        assert d["choices"][0]["finish_reason"] is None


class TestChatCompletionProtocol:
    def test_chat_completion_response(self):
        resp = ChatCompletionResponse(
            id="chatcmpl-test",
            model="Qwen3-0.6B",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage={"completion_tokens": 3},
        )
        d = asdict(resp)
        j = json.dumps(d)
        parsed = json.loads(j)

        assert parsed["object"] == "chat.completion"
        assert parsed["choices"][0]["message"]["role"] == "assistant"
        assert parsed["choices"][0]["message"]["content"] == "Hello!"

    def test_chat_stream_response_role_chunk(self):
        """流式第一个 chunk：发送 role。"""
        resp = ChatCompletionStreamResponse(
            id="chatcmpl-stream",
            model="Qwen3-0.6B",
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatCompletionStreamDelta(role="assistant"),
                )
            ],
        )
        d = asdict(resp)
        assert d["object"] == "chat.completion.chunk"
        assert d["choices"][0]["delta"]["role"] == "assistant"
        assert d["choices"][0]["delta"]["content"] is None

    def test_chat_stream_response_content_chunk(self):
        """流式后续 chunk：发送 content。"""
        resp = ChatCompletionStreamResponse(
            id="chatcmpl-stream",
            model="Qwen3-0.6B",
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatCompletionStreamDelta(content="世界"),
                    finish_reason=None,
                )
            ],
        )
        d = asdict(resp)
        assert d["choices"][0]["delta"]["content"] == "世界"
        assert d["choices"][0]["finish_reason"] is None

    def test_sse_format(self):
        """验证 SSE 格式拼接。"""
        resp = CompletionStreamResponse(
            id="cmpl-sse",
            model="test",
            choices=[CompletionStreamChoice(index=0, text="token")],
        )
        sse_line = f"data: {json.dumps(asdict(resp))}\n\n"
        assert sse_line.startswith("data: {")
        assert sse_line.endswith("}\n\n")


class TestChatMessage:
    def test_roles(self):
        for role in ["system", "user", "assistant"]:
            msg = ChatMessage(role=role, content="test")
            assert msg.role == role


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
