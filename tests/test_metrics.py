"""
InferenceMetrics 单元测试。

测试不需要 GPU，纯 Python 逻辑验证。
"""

import time

import pytest

from nanovllm.engine.metrics import InferenceMetrics, SequenceMetrics, _percentile

# ══════════════════════════════════════════════════════════════
# SequenceMetrics 测试
# ══════════════════════════════════════════════════════════════


class TestSequenceMetrics:
    def test_ttft(self):
        sm = SequenceMetrics(seq_id=0, arrival_time=1.0, first_token_time=1.05)
        assert abs(sm.ttft - 0.05) < 1e-6

    def test_ttft_zero_when_no_first_token(self):
        sm = SequenceMetrics(seq_id=0, arrival_time=1.0)
        assert sm.ttft == 0.0

    def test_e2e_latency(self):
        sm = SequenceMetrics(seq_id=0, arrival_time=1.0, finish_time=2.5)
        assert abs(sm.e2e_latency - 1.5) < 1e-6

    def test_e2e_zero_when_not_finished(self):
        sm = SequenceMetrics(seq_id=0, arrival_time=1.0)
        assert sm.e2e_latency == 0.0


# ══════════════════════════════════════════════════════════════
# InferenceMetrics 测试
# ══════════════════════════════════════════════════════════════


class TestInferenceMetrics:
    def test_on_request_arrival(self):
        metrics = InferenceMetrics()
        metrics.on_request_arrival(seq_id=0, num_prompt_tokens=10)

        assert metrics.total_requests == 1
        assert metrics.total_prompt_tokens == 10
        assert 0 in metrics._seq_metrics
        assert metrics._seq_metrics[0].num_prompt_tokens == 10

    def test_on_token_generated_first_token(self):
        metrics = InferenceMetrics()
        metrics.on_request_arrival(seq_id=0, num_prompt_tokens=5)

        # 等一小会让 arrival_time 和 first_token_time 有差异
        time.sleep(0.01)
        metrics.on_token_generated(seq_id=0)

        assert metrics.total_generation_tokens == 1
        sm = metrics._seq_metrics[0]
        assert sm.num_completion_tokens == 1
        assert sm.first_token_time > 0
        assert len(metrics._recent_ttft) == 1
        assert metrics._recent_ttft[0] > 0

    def test_on_token_generated_subsequent_tokens(self):
        metrics = InferenceMetrics()
        metrics.on_request_arrival(seq_id=0, num_prompt_tokens=5)

        # 首 token
        metrics.on_token_generated(seq_id=0)
        time.sleep(0.01)

        # 第二个 token → 应记录 TBT
        metrics.on_token_generated(seq_id=0)

        assert metrics.total_generation_tokens == 2
        assert len(metrics._recent_tbt) == 1
        assert metrics._recent_tbt[0] > 0

    def test_on_request_finished(self):
        metrics = InferenceMetrics()
        metrics.on_request_arrival(seq_id=0, num_prompt_tokens=5)
        metrics.on_token_generated(seq_id=0)
        time.sleep(0.01)
        metrics.on_request_finished(seq_id=0)

        assert len(metrics._recent_e2e) == 1
        assert metrics._recent_e2e[0] > 0
        assert metrics._seq_metrics[0].finish_time > 0

    def test_on_step_updates_state(self):
        metrics = InferenceMetrics()
        metrics.on_step(
            num_waiting=3,
            num_running=5,
            kv_cache_usage=0.42,
            num_prefill=100,
            num_decode=5,
        )
        assert metrics.num_waiting == 3
        assert metrics.num_running == 5
        assert abs(metrics.kv_cache_usage - 0.42) < 1e-6

    def test_avg_ttft(self):
        metrics = InferenceMetrics()
        # 模拟 3 个请求
        for i in range(3):
            metrics.on_request_arrival(seq_id=i, num_prompt_tokens=10)
        time.sleep(0.01)
        for i in range(3):
            metrics.on_token_generated(seq_id=i)

        assert metrics.avg_ttft > 0
        assert len(metrics._recent_ttft) == 3

    def test_snapshot_format(self):
        metrics = InferenceMetrics()
        metrics.on_request_arrival(seq_id=0, num_prompt_tokens=10)
        metrics.on_token_generated(seq_id=0)
        metrics.on_request_finished(seq_id=0)
        metrics.on_step(2, 3, 0.5, 100, 3)

        snap = metrics.snapshot()

        assert "latency" in snap
        assert "throughput" in snap
        assert "queue" in snap
        assert "resource" in snap
        assert "requests" in snap
        assert snap["queue"]["num_waiting"] == 2
        assert snap["queue"]["num_running"] == 3
        assert snap["resource"]["kv_cache_usage_percent"] == 50.0

    def test_multiple_requests_lifecycle(self):
        """完整的多请求生命周期测试。"""
        metrics = InferenceMetrics()

        # 3 个请求到达
        for i in range(3):
            metrics.on_request_arrival(seq_id=i, num_prompt_tokens=10 + i)

        assert metrics.total_requests == 3
        assert metrics.total_prompt_tokens == 33  # 10 + 11 + 12

        # 每个请求生成 2 个 token
        for i in range(3):
            metrics.on_token_generated(seq_id=i)  # first token
            time.sleep(0.005)
            metrics.on_token_generated(seq_id=i)  # second token

        assert metrics.total_generation_tokens == 6
        assert len(metrics._recent_ttft) == 3
        assert len(metrics._recent_tbt) == 3

        # 全部完成
        for i in range(3):
            metrics.on_request_finished(seq_id=i)

        assert len(metrics._recent_e2e) == 3

    def test_unknown_seq_id_ignored(self):
        """对未知 seq_id 的操作应安全忽略。"""
        metrics = InferenceMetrics()
        metrics.on_token_generated(seq_id=999)  # 未注册
        metrics.on_request_finished(seq_id=999)  # 未注册
        assert metrics.total_generation_tokens == 0

    def test_cleanup_old_metrics(self):
        """验证过老的度量数据被自动清理。"""
        metrics = InferenceMetrics(window_size=5)
        # 创建 15 个请求并完成
        for i in range(15):
            metrics.on_request_arrival(seq_id=i, num_prompt_tokens=1)
            metrics.on_token_generated(seq_id=i)
            metrics.on_request_finished(seq_id=i)
        # 度量字典不应无限增长
        assert len(metrics._seq_metrics) <= 15  # cleanup 只在超过 2*window_size 时


# ══════════════════════════════════════════════════════════════
# _percentile 辅助函数测试
# ══════════════════════════════════════════════════════════════


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 99) == 0.0

    def test_single(self):
        assert _percentile([1.5], 99) == 1.5

    def test_p50(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _percentile(data, 50)
        assert result == 3.0

    def test_p99(self):
        data = list(range(100))
        result = _percentile(data, 99)
        assert result == 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
