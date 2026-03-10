"""
异步推理引擎，在后台线程中运行同步 LLMEngine 的 step() 循环，
通过 asyncio.Queue 与 FastAPI 协程通信。

核心设计：
    - GPU 推理（step()）在后台独立线程中执行，不阻塞 asyncio 事件循环
    - 每个请求关联一个 asyncio.Queue 作为输出通道，支持流式返回
    - 通过 _seq_to_request 映射 Sequence.seq_id → request_id

用法：
    engine = AsyncLLMEngine("models/Qwen3-0.6B", enforce_eager=True)
    engine.start()

    # 非流式
    output = await engine.generate("Hello", SamplingParams())

    # 流式
    async for output in engine.stream_generate("Hello", SamplingParams()):
        print(output.text, end="")
"""

import asyncio
import time
from dataclasses import dataclass, field
from threading import Thread
from typing import AsyncGenerator

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


@dataclass
class RequestOutput:
    """单次 step 产生的增量输出。"""

    request_id: str
    text: str  # 增量文本（本次新 token decode 后）
    token_ids: list[int]  # 增量 token id
    cumulative_text: str  # 累计文本
    cumulative_token_ids: list[int]  # 累计 token id
    finished: bool = False
    finish_reason: str | None = None  # "stop" | "length"


@dataclass
class PendingRequest:
    """一个待处理的异步请求。"""

    request_id: str
    prompt: str | list[int]
    sampling_params: SamplingParams
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    arrival_time: float = field(default_factory=time.time)
    # 累计跟踪（用于流式输出中构建 cumulative text）
    cumulative_token_ids: list[int] = field(default_factory=list)


class AsyncLLMEngine:
    """
    异步推理引擎：在后台线程中运行同步 LLMEngine 的 step() 循环，
    通过 asyncio.Queue 与 FastAPI 协程通信。
    """

    def __init__(self, model: str, **kwargs):
        # 同步引擎（现有 LLMEngine，不做任何修改核心逻辑）
        self.engine = LLMEngine(model, **kwargs)
        self.tokenizer = self.engine.tokenizer

        # 请求管理
        self._pending_requests: dict[str, PendingRequest] = {}
        self._new_requests: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._seq_to_request: dict[int, str] = {}  # seq_id → request_id

        # 后台推理线程
        self._loop: asyncio.AbstractEventLoop | None = None
        self._background_thread: Thread | None = None
        self._is_running = False

    def start(self):
        """启动后台推理线程。"""
        self._is_running = True
        self._background_thread = Thread(
            target=self._run_engine_loop, daemon=True, name="engine-loop"
        )
        self._background_thread.start()

    def stop(self):
        """停止后台推理线程并清理资源。"""
        self._is_running = False
        if self._background_thread:
            self._background_thread.join(timeout=10)
        self.engine.exit()

    def _run_engine_loop(self):
        """
        后台推理主循环（在独立线程中运行）。

        流程：
          1. 从 _new_requests 队列取出新请求，调用 engine.add_request()
          2. 调用 engine.step() 执行一次推理
          3. 将完成/增量结果推送到对应请求的 output_queue
          4. 循环直到 _is_running=False 且无活跃请求
        """
        while self._is_running or self._pending_requests:
            # ── 1. 吸收新请求 ──
            added = False
            while True:
                try:
                    req = self._new_requests.get_nowait()
                except asyncio.queues.QueueEmpty:
                    break
                seq_id = self.engine.add_request(req.prompt, req.sampling_params)
                self._seq_to_request[seq_id] = req.request_id
                self._pending_requests[req.request_id] = req
                added = True

            # ── 2. 如果没有活跃请求，短暂睡眠避免空转 ──
            if self.engine.is_finished() and not added:
                time.sleep(0.001)
                continue

            # ── 3. 执行一步推理 ──
            try:
                finished_outputs, incremental_outputs, _, _ = self.engine.step()
            except Exception as e:
                # 将错误推送给所有等待中的请求
                for req in self._pending_requests.values():
                    self._push_to_queue(req.output_queue, e)
                self._pending_requests.clear()
                self._seq_to_request.clear()
                continue

            # ── 4. 处理增量输出（流式） ──
            for seq_id, new_token_id in incremental_outputs:
                request_id = self._seq_to_request.get(seq_id)
                if request_id is None:
                    continue
                req = self._pending_requests.get(request_id)
                if req is None:
                    continue

                req.cumulative_token_ids.append(new_token_id)
                # 增量 decode：只 decode 新 token（可能包含子词碎片）
                new_text = self.tokenizer.decode(
                    [new_token_id], skip_special_tokens=True
                )
                output = RequestOutput(
                    request_id=request_id,
                    text=new_text,
                    token_ids=[new_token_id],
                    cumulative_text=self.tokenizer.decode(
                        req.cumulative_token_ids, skip_special_tokens=True
                    ),
                    cumulative_token_ids=list(req.cumulative_token_ids),
                    finished=False,
                )
                self._push_to_queue(req.output_queue, output)

            # ── 5. 处理完成的序列 ──
            for seq_id, completion_token_ids in finished_outputs:
                request_id = self._seq_to_request.pop(seq_id, None)
                if request_id is None:
                    continue
                req = self._pending_requests.pop(request_id, None)
                if req is None:
                    continue

                cumulative_text = self.tokenizer.decode(
                    completion_token_ids, skip_special_tokens=True
                )
                # 最后一个 token 的增量文本
                last_token_text = ""
                if completion_token_ids:
                    last_token_id = completion_token_ids[-1]
                    last_token_text = self.tokenizer.decode(
                        [last_token_id], skip_special_tokens=True
                    )

                output = RequestOutput(
                    request_id=request_id,
                    text=last_token_text,
                    token_ids=list(completion_token_ids),
                    cumulative_text=cumulative_text,
                    cumulative_token_ids=list(completion_token_ids),
                    finished=True,
                    finish_reason="stop",
                )
                self._push_to_queue(req.output_queue, output)

    def _push_to_queue(self, queue: asyncio.Queue, item):
        """线程安全地向 asyncio.Queue 推送数据。"""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(queue.put_nowait, item)

    # ══════════════════════════════════════════════════════════════
    # 公开异步 API
    # ══════════════════════════════════════════════════════════════

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> RequestOutput:
        """
        非流式生成：等待完整结果返回。

        用法：
            output = await engine.generate("Hello", SamplingParams())
            print(output.cumulative_text)
        """
        final_output = None
        async for output in self.stream_generate(prompt, sampling_params, request_id):
            final_output = output
        return final_output

    async def stream_generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        流式生成：逐 token yield RequestOutput。

        用法：
            async for output in engine.stream_generate("Hello", params):
                print(output.text, end="", flush=True)
        """
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        if request_id is None:
            request_id = f"req-{id(object())}"

        req = PendingRequest(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
        )

        await self._new_requests.put(req)

        # 等待输出
        while True:
            output = await req.output_queue.get()
            if isinstance(output, Exception):
                raise output
            yield output
            if output.finished:
                break
