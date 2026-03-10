"""
推理引擎可观测性指标收集器。

核心类：
    SequenceMetrics - 单个序列的度量数据（TTFT / TBT / E2E）
    InferenceMetrics - 全局度量收集器，由引擎在关键路径埋入钩子

事件钩子：
    on_request_arrival()     → LLMEngine.add_request() 中调用
    on_token_generated()     → LLMEngine.step() postprocess 后调用
    on_request_finished()    → LLMEngine.step() 检测到 finished 后调用
    on_step()                → LLMEngine.step() 末尾调用
"""

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class SequenceMetrics:
    """单个序列的度量数据。"""

    seq_id: int
    arrival_time: float = 0.0  # 请求到达时间
    first_token_time: float = 0.0  # 首 token 生成时间
    last_token_time: float = 0.0  # 上一个 token 生成时间
    finish_time: float = 0.0  # 完成时间
    num_prompt_tokens: int = 0  # prompt 长度
    num_completion_tokens: int = 0  # 生成 token 数

    @property
    def ttft(self) -> float:
        """Time To First Token (秒)。"""
        if self.first_token_time > 0 and self.arrival_time > 0:
            return self.first_token_time - self.arrival_time
        return 0.0

    @property
    def e2e_latency(self) -> float:
        """端到端延迟 (秒)。"""
        if self.finish_time > 0 and self.arrival_time > 0:
            return self.finish_time - self.arrival_time
        return 0.0


class InferenceMetrics:
    """
    推理引擎的全局度量收集器。

    职责：
    1. 跟踪每个序列的延迟指标（TTFT / TBT / E2E）
    2. 计算全局吞吐量（tokens/s）
    3. 收集调度器和 KV-cache 状态
    4. 提供 JSON 快照 & 向 Prometheus 导出
    """

    def __init__(self, window_size: int = 100):
        # ── 序列级度量 ──
        self._seq_metrics: dict[int, SequenceMetrics] = {}

        # ── 滑动窗口（最近 N 条记录） ──
        self._window_size = window_size
        self._recent_tbt: deque[float] = deque(maxlen=window_size)
        self._recent_ttft: deque[float] = deque(maxlen=window_size)
        self._recent_e2e: deque[float] = deque(maxlen=window_size)

        # ── 累计计数 ──
        self.total_prompt_tokens: int = 0
        self.total_generation_tokens: int = 0
        self.total_requests: int = 0

        # ── 实时状态（由 on_step 更新） ──
        self.num_waiting: int = 0
        self.num_running: int = 0
        self.kv_cache_usage: float = 0.0  # 0.0 ~ 1.0
        self.prefix_cache_hit_rate: float = 0.0

        # ── 吞吐量滑动窗口 ──
        self._throughput_window_start: float = time.time()
        self._throughput_window_tokens: int = 0

    # ══════════════════════════════════════════════════════════════
    # 事件钩子（由引擎在关键点调用）
    # ══════════════════════════════════════════════════════════════

    def on_request_arrival(self, seq_id: int, num_prompt_tokens: int):
        """请求到达。在 add_request() 中调用。"""
        self._seq_metrics[seq_id] = SequenceMetrics(
            seq_id=seq_id,
            arrival_time=time.time(),
            num_prompt_tokens=num_prompt_tokens,
        )
        self.total_requests += 1
        self.total_prompt_tokens += num_prompt_tokens

    def on_token_generated(self, seq_id: int):
        """生成一个 token。在 step() postprocess 后调用。"""
        now = time.time()
        sm = self._seq_metrics.get(seq_id)
        if sm is None:
            return

        sm.num_completion_tokens += 1
        self.total_generation_tokens += 1
        self._throughput_window_tokens += 1

        if sm.first_token_time == 0.0:
            # 首 token → 记录 TTFT
            sm.first_token_time = now
            self._recent_ttft.append(sm.ttft)
        else:
            # 非首 token → 记录 TBT
            tbt = now - sm.last_token_time
            self._recent_tbt.append(tbt)

        sm.last_token_time = now

    def on_request_finished(self, seq_id: int):
        """请求完成。在 step() 检测到 finished 时调用。"""
        sm = self._seq_metrics.get(seq_id)
        if sm is None:
            return
        sm.finish_time = time.time()
        self._recent_e2e.append(sm.e2e_latency)
        # 清理过老的度量数据，避免无限增长
        if len(self._seq_metrics) > self._window_size * 2:
            oldest_ids = sorted(self._seq_metrics.keys())[: self._window_size]
            for sid in oldest_ids:
                self._seq_metrics.pop(sid, None)

    def on_step(
        self,
        num_waiting: int,
        num_running: int,
        kv_cache_usage: float,
        num_prefill: int,
        num_decode: int,
    ):
        """每步推理后更新状态。在 step() 末尾调用。"""
        self.num_waiting = num_waiting
        self.num_running = num_running
        self.kv_cache_usage = kv_cache_usage

    # ══════════════════════════════════════════════════════════════
    # 查询接口
    # ══════════════════════════════════════════════════════════════

    @property
    def avg_ttft(self) -> float:
        """平均 TTFT (秒)。"""
        return (
            sum(self._recent_ttft) / len(self._recent_ttft)
            if self._recent_ttft
            else 0.0
        )

    @property
    def avg_tbt(self) -> float:
        """平均 TBT (秒)。"""
        return (
            sum(self._recent_tbt) / len(self._recent_tbt) if self._recent_tbt else 0.0
        )

    @property
    def avg_e2e(self) -> float:
        """平均端到端延迟 (秒)。"""
        return (
            sum(self._recent_e2e) / len(self._recent_e2e) if self._recent_e2e else 0.0
        )

    @property
    def p99_ttft(self) -> float:
        """P99 TTFT (秒)。"""
        return _percentile(list(self._recent_ttft), 99)

    @property
    def p99_tbt(self) -> float:
        """P99 TBT (秒)。"""
        return _percentile(list(self._recent_tbt), 99)

    @property
    def throughput_tokens_per_sec(self) -> float:
        """实时吞吐量 (tokens/s)，基于滑动窗口。"""
        elapsed = time.time() - self._throughput_window_start
        if elapsed < 0.001:
            return 0.0
        tps = self._throughput_window_tokens / elapsed
        # 每 10 秒重置窗口
        if elapsed > 10.0:
            self._throughput_window_start = time.time()
            self._throughput_window_tokens = 0
        return tps

    def snapshot(self) -> dict:
        """返回当前所有指标的快照（JSON 序列化友好）。"""
        return {
            "latency": {
                "avg_ttft_ms": round(self.avg_ttft * 1000, 2),
                "avg_tbt_ms": round(self.avg_tbt * 1000, 2),
                "avg_e2e_ms": round(self.avg_e2e * 1000, 2),
                "p99_ttft_ms": round(self.p99_ttft * 1000, 2),
                "p99_tbt_ms": round(self.p99_tbt * 1000, 2),
            },
            "throughput": {
                "tokens_per_sec": round(self.throughput_tokens_per_sec, 1),
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_generation_tokens": self.total_generation_tokens,
            },
            "queue": {
                "num_waiting": self.num_waiting,
                "num_running": self.num_running,
            },
            "resource": {
                "kv_cache_usage_percent": round(self.kv_cache_usage * 100, 1),
                "prefix_cache_hit_rate": round(self.prefix_cache_hit_rate * 100, 1),
            },
            "requests": {
                "total": self.total_requests,
                "active": len(
                    [s for s in self._seq_metrics.values() if s.finish_time == 0.0]
                ),
            },
        }


def _percentile(data: list[float], p: float) -> float:
    """计算 P 百分位数。"""
    if not data:
        return 0.0
    data.sort()
    idx = int(len(data) * p / 100)
    idx = min(idx, len(data) - 1)
    return data[idx]
