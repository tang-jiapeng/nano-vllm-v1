"""
OpenAI 兼容的 HTTP API 服务器。

用法：
    python -m nanovllm.entrypoints.serve --model models/Qwen3-0.6B --port 8000

API 端点：
    POST /v1/completions          - 文本补全
    POST /v1/chat/completions     - 对话补全
    GET  /v1/models               - 模型列表
    GET  /v1/metrics              - JSON 格式指标快照
    GET  /health                  - 健康检查
    GET  /metrics                 - Prometheus 拉取端点
"""

import json
import uuid
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from nanovllm.engine.async_llm_engine import AsyncLLMEngine
from nanovllm.engine.prometheus_metrics import HAS_PROMETHEUS, PrometheusExporter
from nanovllm.sampling_params import SamplingParams
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


def _strip_none(obj):
    """递归移除 dict 中值为 None 的键，使 JSON 输出符合 OpenAI 协议规范。"""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(item) for item in obj]
    return obj

# 全局引擎实例（在 create_server() / entrypoints 中初始化）
engine: AsyncLLMEngine = None
model_name: str = ""
prometheus_exporter: PrometheusExporter | None = None

# 背压控制
MAX_PENDING = 1000


def create_app() -> FastAPI:
    """创建 FastAPI 应用，注册路由和中间件。"""
    app = FastAPI(title="nano-vllm API Server")

    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── 健康检查 ─────────────────────────────────────────────
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ─── 模型列表 ─────────────────────────────────────────────
    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "nano-vllm",
                }
            ],
        }

    # ─── /v1/completions ──────────────────────────────────────
    @app.post("/v1/completions")
    async def completions(request: Request):
        if len(engine._pending_requests) >= MAX_PENDING:
            raise HTTPException(status_code=503, detail="Server overloaded")

        body = await request.json()
        prompt = body.get("prompt", "")
        temperature = body.get("temperature", 1.0)
        max_tokens = body.get("max_tokens", 64)
        stream = body.get("stream", False)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"

        if stream:
            return StreamingResponse(
                _stream_completions(prompt, sampling_params, request_id),
                media_type="text/event-stream",
            )

        # 非流式：等待完整结果
        output = await engine.generate(prompt, sampling_params, request_id)
        resp = CompletionResponse(
            id=request_id,
            model=model_name,
            choices=[
                CompletionChoice(
                    index=0,
                    text=output.cumulative_text,
                    finish_reason=output.finish_reason,
                )
            ],
            usage={
                "completion_tokens": len(output.cumulative_token_ids),
            },
        )
        return JSONResponse(asdict(resp))

    async def _stream_completions(prompt, sampling_params, request_id):
        """SSE 流式补全生成器。"""
        async for output in engine.stream_generate(prompt, sampling_params, request_id):
            chunk = CompletionStreamResponse(
                id=request_id,
                model=model_name,
                choices=[
                    CompletionStreamChoice(
                        index=0,
                        text=output.text,
                        finish_reason=output.finish_reason if output.finished else None,
                    )
                ],
            )
            yield f"data: {json.dumps(_strip_none(asdict(chunk)))}\n\n"
        yield "data: [DONE]\n\n"

    # ─── /v1/chat/completions ─────────────────────────────────
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if len(engine._pending_requests) >= MAX_PENDING:
            raise HTTPException(status_code=503, detail="Server overloaded")

        body = await request.json()
        messages = body.get("messages", [])
        temperature = body.get("temperature", 1.0)
        max_tokens = body.get("max_tokens", 64)
        stream = body.get("stream", False)

        # 使用 tokenizer 的 chat template 转换
        prompt = engine.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        if stream:
            return StreamingResponse(
                _stream_chat(prompt, sampling_params, request_id),
                media_type="text/event-stream",
            )

        output = await engine.generate(prompt, sampling_params, request_id)
        resp = ChatCompletionResponse(
            id=request_id,
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant", content=output.cumulative_text
                    ),
                    finish_reason=output.finish_reason,
                )
            ],
            usage={
                "completion_tokens": len(output.cumulative_token_ids),
            },
        )
        return JSONResponse(asdict(resp))

    async def _stream_chat(prompt, sampling_params, request_id):
        """SSE 流式聊天生成器。"""
        # 第一个 chunk：发送 role
        first_chunk = ChatCompletionStreamResponse(
            id=request_id,
            model=model_name,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatCompletionStreamDelta(role="assistant"),
                )
            ],
        )
        yield f"data: {json.dumps(_strip_none(asdict(first_chunk)))}\n\n"

        # 后续 chunk：逐 token 发送 content
        async for output in engine.stream_generate(prompt, sampling_params, request_id):
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                model=model_name,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=ChatCompletionStreamDelta(content=output.text),
                        finish_reason=(
                            output.finish_reason if output.finished else None
                        ),
                    )
                ],
            )
            yield f"data: {json.dumps(_strip_none(asdict(chunk)))}\n\n"
        yield "data: [DONE]\n\n"

    # ─── /v1/metrics (JSON) ───────────────────────────────────
    @app.get("/v1/metrics")
    async def json_metrics():
        """JSON 格式的指标快照（无需 Prometheus）。"""
        snapshot = engine.engine.metrics.snapshot()
        return JSONResponse(snapshot)

    # ─── /metrics (Prometheus) ────────────────────────────────
    @app.get("/metrics")
    async def prom_metrics():
        """Prometheus 拉取端点。"""
        if prometheus_exporter is not None and prometheus_exporter.available:
            prometheus_exporter.export(engine.engine.metrics)
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )
        # 降级到 JSON
        snapshot = engine.engine.metrics.snapshot()
        return JSONResponse(snapshot)

    return app
