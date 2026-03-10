"""
API Server 端到端测试。

需要 GPU 和模型文件才能运行。使用 pytest -m e2e 选择性运行。

用法：
    MODEL=./models/Qwen3-0.6B pytest tests/test_e2e_serve.py -v -s
"""

import asyncio
import json
import os
import time

import pytest

# 需要 GPU 环境
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.environ.get("MODEL", "./models/Qwen3-0.6B")),
    reason="MODEL directory not found, set MODEL env var",
)


@pytest.fixture(scope="module")
def model_path():
    return os.environ.get("MODEL", "./models/Qwen3-0.6B")


@pytest.fixture(scope="module")
def engine_and_app(model_path):
    """创建 AsyncLLMEngine + FastAPI app（module 级 fixture）。"""
    from nanovllm.engine.async_llm_engine import AsyncLLMEngine
    from nanovllm.engine.prometheus_metrics import PrometheusExporter
    from nanovllm.serving import api_server

    engine = AsyncLLMEngine(
        model_path,
        max_model_len=512,
        max_num_seqs=16,
        enforce_eager=True,
    )
    engine.start()

    api_server.engine = engine
    api_server.model_name = model_path.rstrip("/").split("/")[-1]
    api_server.prometheus_exporter = PrometheusExporter(
        model_name=api_server.model_name
    )

    app = api_server.create_app()
    yield engine, app

    engine.stop()


@pytest.fixture(scope="module")
def client(engine_and_app):
    """FastAPI TestClient（同步包装器，适合 pytest）。"""
    from fastapi.testclient import TestClient

    _, app = engine_and_app
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestModels:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1


class TestCompletions:
    def test_non_streaming(self, client):
        resp = client.post(
            "/v1/completions",
            json={
                "prompt": "Hello, who are you?",
                "max_tokens": 20,
                "temperature": 0.7,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) == 1
        assert len(data["choices"][0]["text"]) > 0
        assert data["choices"][0]["finish_reason"] is not None

    def test_streaming(self, client):
        with client.stream(
            "POST",
            "/v1/completions",
            json={
                "prompt": "Count from 1 to 5:",
                "max_tokens": 30,
                "temperature": 0.7,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            chunks = []
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    chunks.append(chunk)
            assert len(chunks) > 0
            # 最后一个 chunk 应有 finish_reason
            last = chunks[-1]
            assert last["choices"][0]["finish_reason"] is not None


class TestChatCompletions:
    def test_non_streaming(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 20,
                "temperature": 0.7,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0

    def test_streaming(self, client):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 20,
                "temperature": 0.7,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            chunks = []
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    chunks.append(chunk)
            # 第一个 chunk 应有 role
            assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
            assert len(chunks) > 1


class TestMetrics:
    def test_json_metrics(self, client):
        # 先发送一个请求，让引擎有数据
        client.post(
            "/v1/completions",
            json={"prompt": "test", "max_tokens": 5, "temperature": 0.7},
        )
        # 等一下让 metrics 更新
        time.sleep(0.5)

        resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "latency" in data
        assert "throughput" in data
        assert "queue" in data
        assert "resource" in data
        assert "requests" in data

    def test_prometheus_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
