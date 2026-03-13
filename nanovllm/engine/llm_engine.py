import atexit
import gc
from dataclasses import fields
from time import perf_counter

import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.metrics import InferenceMetrics
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class LLMEngine:
    """
    推理引擎核心类，协调 Scheduler、ModelRunner 和 Tokenizer，
    支持 tensor parallel 多进程推理。
    """

    def __init__(self, model, **kwargs):
        """
        初始化推理引擎。

        Args:
            model: 本地模型目录路径
            **kwargs: 传递给 Config 的参数

        流程：解析配置 -> 启动 tensor parallel 子进程 -> 创建 ModelRunner ->
              加载 Tokenizer -> 创建 Scheduler -> 注册 atexit 清理
        """
        # 提取 Config 相关参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.ps = []
        self.events = []

        ctx = mp.get_context("spawn")

        # 为 rank 1..N-1 启动子进程
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # rank 0 在主进程运行
        self.model_runner = ModelRunner(config, 0, self.events)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.scheduler = Scheduler(config)
        self.metrics = InferenceMetrics()

        atexit.register(self.exit)

    def exit(self):
        """发送 exit 指令给所有 ModelRunner，等待子进程结束，释放资源。"""
        if not hasattr(self, "model_runner"):
            return  # 初始化失败时安全退出
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
        # 强制 Python GC 回收引用，再清理 CUDA 缓存，确保后续实例可获得足够显存
        gc.collect()
        torch.cuda.empty_cache()

    def add_request(
        self, prompt: str | list[int], sampling_params: SamplingParams
    ) -> int:
        """将 prompt（字符串或 token ID 列表）封装为 Sequence 并加入调度器。返回 seq_id。"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        self.metrics.on_request_arrival(seq.seq_id, len(prompt))
        return seq.seq_id

    def step(self):
        """
        执行一次统一推理步骤（v1 风格）。

        返回值：
        - finished_outputs: 已完成序列的 (seq_id, completion_token_ids) 列表
        - incremental_outputs: 未完成但有新 token 的 (seq_id, last_token_id) 列表
        - num_prefill_tokens: 本次 prefill 处理的 token 数
        - num_decode_tokens: 本次 decode 处理的序列数
        """
        seqs = self.scheduler.schedule()
        if not seqs:
            return [], [], 0, 0

        # 在模型推理前计算指标（postprocess 会重置 num_new_tokens）
        num_prefill_tokens = sum(s.num_new_tokens for s in seqs if s.num_new_tokens > 1)
        num_decode_tokens = sum(1 for s in seqs if s.num_new_tokens == 1)

        token_ids, seq_need_compute_logits = self.model_runner.call("run", seqs)
        # 记录本步获得新 token 的序列索引
        new_token_indices = set(seq_need_compute_logits)
        self.scheduler.postprocess(seqs, token_ids, seq_need_compute_logits)

        finished_outputs = []
        incremental_outputs = []
        for idx, seq in enumerate(seqs):
            if idx in new_token_indices:
                # 该序列本步获得了新 token
                self.metrics.on_token_generated(seq.seq_id)
                if seq.is_finished:
                    finished_outputs.append(
                        (
                            seq.seq_id,
                            seq.completion_token_ids,
                            seq.num_speculative_proposed_total,
                            seq.num_speculative_accepted_total,
                        )
                    )
                    self.metrics.on_request_finished(seq.seq_id)
                else:
                    incremental_outputs.append((seq.seq_id, seq.last_token))

        # 更新全局状态指标
        self.metrics.on_step(
            num_waiting=len(self.scheduler.waiting),
            num_running=len(self.scheduler.running),
            kv_cache_usage=self.scheduler.block_manager.usage_ratio,
            num_prefill=num_prefill_tokens,
            num_decode=num_decode_tokens,
        )

        return (
            finished_outputs,
            incremental_outputs,
            num_prefill_tokens,
            num_decode_tokens,
        )

    def is_finished(self):
        """检查所有请求是否已完成。"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量文本生成主接口。

        Args:
            prompts: 字符串列表或 token ID 列表
            sampling_params: 单个或与 prompts 等长的采样参数
            use_tqdm: 是否显示进度条

        Returns:
            每个元素包含 {"text": str, "token_ids": list[int]}
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.0

        while not self.is_finished():
            t = perf_counter()
            output, _incremental, num_prefill, num_decode = self.step()
            elapsed = perf_counter() - t

            if use_tqdm:
                if num_prefill > 0:
                    prefill_throughput = num_prefill / elapsed
                if num_decode > 0:
                    decode_throughput = num_decode / elapsed
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )

            for seq_id, token_ids, proposed, accepted in output:
                outputs[seq_id] = {
                    "token_ids": token_ids,
                    "proposed": proposed,
                    "accepted": accepted,
                }
                if use_tqdm:
                    pbar.update(1)

        # 按 seq_id 排序并 decode
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {
                "text": self.tokenizer.decode(item["token_ids"]),
                "token_ids": item["token_ids"],
                "proposed": item["proposed"],
                "accepted": item["accepted"],
            }
            for item in outputs
        ]

        if use_tqdm:
            pbar.close()

        return outputs
