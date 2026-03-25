"""
nano-vllm 在线推理服务入口。

用法：
    python -m nanovllm.entrypoints.serve \
        --model models/Qwen3-0.6B \
        --host 0.0.0.0 \
        --port 8000 \
        --max-model-len 4096 \
        --enforce-eager
"""

import argparse

import uvicorn

from nanovllm.engine.async_llm_engine import AsyncLLMEngine
from nanovllm.engine.prometheus_metrics import PrometheusExporter
from nanovllm.serving import api_server


def main():
    parser = argparse.ArgumentParser(description="nano-vllm API Server")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enable-kv-offload", action="store_true")
    parser.add_argument("--cpu-offload-gb", type=str, default="0")
    parser.add_argument("--cpu-offload-safety-margin-gb", type=float, default=4.0)
    parser.add_argument("--cpu-offload-watermark-blocks", type=int, default=0)
    parser.add_argument("--speculative-method", choices=["ngram"], default=None)
    parser.add_argument("--num-speculative-tokens", type=int, default=0)
    parser.add_argument("--ngram-prompt-lookup-min", type=int, default=1)
    parser.add_argument("--ngram-prompt-lookup-max", type=int, default=4)
    args = parser.parse_args()

    # 初始化异步引擎
    engine_instance = AsyncLLMEngine(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        chunked_prefill=args.chunked_prefill,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_kv_offload=args.enable_kv_offload,
        cpu_offload_gb=(
            "auto"
            if args.cpu_offload_gb == "auto"
            else float(args.cpu_offload_gb)
        ),
        cpu_offload_safety_margin_gb=args.cpu_offload_safety_margin_gb,
        cpu_offload_watermark_blocks=args.cpu_offload_watermark_blocks,
        speculative_method=args.speculative_method,
        num_speculative_tokens=args.num_speculative_tokens,
        ngram_prompt_lookup_min=args.ngram_prompt_lookup_min,
        ngram_prompt_lookup_max=args.ngram_prompt_lookup_max,
    )
    engine_instance.start()

    # 注入全局引擎和模型名称
    api_server.engine = engine_instance
    api_server.model_name = args.model.rstrip("/").split("/")[-1]

    # 初始化 Prometheus exporter（可选）
    api_server.prometheus_exporter = PrometheusExporter(
        model_name=api_server.model_name
    )

    app = api_server.create_app()

    print(f"\n{'='*60}")
    print(f"  nano-vllm API Server")
    print(f"  Model: {api_server.model_name}")
    print(f"  Listening: http://{args.host}:{args.port}")
    print(f"  Endpoints:")
    print(f"    POST /v1/completions")
    print(f"    POST /v1/chat/completions")
    print(f"    GET  /v1/models")
    print(f"    GET  /v1/metrics")
    print(f"    GET  /metrics")
    print(f"    GET  /health")
    print(f"{'='*60}\n")

    # 启动 uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
