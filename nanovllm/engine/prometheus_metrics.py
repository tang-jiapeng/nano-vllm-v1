"""
Prometheus 格式的指标导出。

使用 prometheus_client 库暴露以下指标类型：
    Histogram : TTFT / TBT / E2E 延迟、batch size、prefill tokens
    Counter   : 累计 prompt/generation tokens、请求数
    Gauge     : waiting/running 队列深度、KV-cache 使用率、吞吐量

在 API Server 中通过 GET /metrics 暴露，可接入 Grafana 监控面板。
"""

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


def _create_metrics():
    """创建并返回所有 Prometheus 指标对象（惰性初始化）。"""
    if not HAS_PROMETHEUS:
        return None

    metrics = {}

    # ── 信息 ──
    metrics["engine_info"] = Info(
        "nanovllm_engine",
        "推理引擎基本信息",
    )

    # ── 延迟指标（Histogram） ──
    metrics["ttft"] = Histogram(
        "nanovllm_ttft_seconds",
        "Time To First Token (首 token 延迟)",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    metrics["tbt"] = Histogram(
        "nanovllm_tbt_seconds",
        "Time Between Tokens (token 间延迟)",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
    )
    metrics["e2e"] = Histogram(
        "nanovllm_e2e_latency_seconds",
        "端到端请求延迟",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    metrics["batch_size"] = Histogram(
        "nanovllm_batch_size",
        "每步调度的 batch size",
        buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    )
    metrics["prefill_tokens"] = Histogram(
        "nanovllm_prefill_tokens_per_step",
        "每步 prefill token 数量",
        buckets=(1, 16, 64, 256, 1024, 4096, 16384),
    )

    # ── 计数器（Counter） ──
    metrics["prompt_tokens"] = Counter(
        "nanovllm_prompt_tokens_total",
        "累计处理的 prompt token 数",
    )
    metrics["generation_tokens"] = Counter(
        "nanovllm_generation_tokens_total",
        "累计生成的 token 数",
    )
    metrics["requests"] = Counter(
        "nanovllm_requests_total",
        "累计请求数",
    )

    # ── 仪表盘（Gauge） ──
    metrics["waiting"] = Gauge(
        "nanovllm_num_waiting_requests",
        "waiting 队列深度",
    )
    metrics["running"] = Gauge(
        "nanovllm_num_running_requests",
        "running 队列深度",
    )
    metrics["kv_cache"] = Gauge(
        "nanovllm_kv_cache_usage_percent",
        "KV-cache 块使用率 (%)",
    )
    metrics["throughput"] = Gauge(
        "nanovllm_token_throughput",
        "实时吞吐量 (tokens/s)",
    )
    metrics["gpu_memory"] = Gauge(
        "nanovllm_gpu_memory_used_bytes",
        "GPU 显存占用 (bytes)",
    )
    metrics["prefix_cache_hit"] = Gauge(
        "nanovllm_prefix_cache_hit_rate",
        "前缀缓存命中率 (%)",
    )

    return metrics


class PrometheusExporter:
    """
    将 InferenceMetrics 的数据推送到 Prometheus 指标对象。

    调用时机：
    - 在 API Server 的 /metrics 端点被拉取时调用 export()
    - 或者在 engine loop 中每 N 步调用一次
    """

    def __init__(self, model_name: str = ""):
        self._prom = _create_metrics()
        self._last_prompt_tokens = 0
        self._last_generation_tokens = 0
        self._last_requests = 0
        # 记录上次导出后的 deque 偏移
        self._last_ttft_len = 0
        self._last_tbt_len = 0
        self._last_e2e_len = 0

        if self._prom is not None:
            self._prom["engine_info"].info({"model": model_name, "version": "0.2.0"})

    @property
    def available(self) -> bool:
        return self._prom is not None

    def export(self, metrics) -> None:
        """
        从 InferenceMetrics 实例读取最新数据，更新 Prometheus 指标。

        Args:
            metrics: InferenceMetrics 实例
        """
        if self._prom is None:
            return

        p = self._prom

        # ── 延迟 Histogram：只 observe 新增的值 ──
        ttft_list = list(metrics._recent_ttft)
        for v in ttft_list[self._last_ttft_len :]:
            p["ttft"].observe(v)
        self._last_ttft_len = len(ttft_list)

        tbt_list = list(metrics._recent_tbt)
        for v in tbt_list[self._last_tbt_len :]:
            p["tbt"].observe(v)
        self._last_tbt_len = len(tbt_list)

        e2e_list = list(metrics._recent_e2e)
        for v in e2e_list[self._last_e2e_len :]:
            p["e2e"].observe(v)
        self._last_e2e_len = len(e2e_list)

        # ── Counter：增加差值 ──
        delta_prompt = metrics.total_prompt_tokens - self._last_prompt_tokens
        if delta_prompt > 0:
            p["prompt_tokens"].inc(delta_prompt)
        self._last_prompt_tokens = metrics.total_prompt_tokens

        delta_gen = metrics.total_generation_tokens - self._last_generation_tokens
        if delta_gen > 0:
            p["generation_tokens"].inc(delta_gen)
        self._last_generation_tokens = metrics.total_generation_tokens

        delta_reqs = metrics.total_requests - self._last_requests
        if delta_reqs > 0:
            p["requests"].inc(delta_reqs)
        self._last_requests = metrics.total_requests

        # ── Gauge ──
        p["waiting"].set(metrics.num_waiting)
        p["running"].set(metrics.num_running)
        p["kv_cache"].set(metrics.kv_cache_usage * 100)
        p["throughput"].set(metrics.throughput_tokens_per_sec)
        p["prefix_cache_hit"].set(metrics.prefix_cache_hit_rate * 100)

        # GPU 显存
        try:
            import torch

            if torch.cuda.is_available():
                p["gpu_memory"].set(torch.cuda.memory_allocated())
        except Exception:
            pass
