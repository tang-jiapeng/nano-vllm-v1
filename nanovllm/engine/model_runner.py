"""
模型运行器：负责模型加载、KV-cache 分配、CUDA Graph 捕获及多进程通信。

vLLM v1 风格：统一 prepare_model_input 替代分离的 prepare_prefill/prepare_decode，
通过 cu_seqlens_q/k 和 seq_need_compute_logits 支持混合 prefill+decode 批次。
"""

import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.quantization.awq_config import AWQConfig
from nanovllm.layers.sampler import Sampler
from nanovllm.models.models import model_dict
from nanovllm.utils.context import get_context, reset_context, set_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型运行器，负责模型加载、KV-cache 分配、CUDA Graph 捕获及多进程通信。
    rank 0 主进程负责任务分发和采样，rank > 0 子进程仅负责前向计算。
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 1. 初始化 NCCL 分布式环境
        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)

        # 2. 加载模型
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        model_type = hf_config.model_type
        if model_type not in model_dict:
            raise ValueError(
                f"不支持的模型类型: {model_type!r}。"
                f"当前支持: {list(model_dict.keys())}"
            )
        model_cls = model_dict[model_type]
        hf_config._awq_config = config.awq_config
        self.model = model_cls(hf_config)
        load_model(self.model, config.model)

        self.speculative_decoding = (
            config.speculative_model is not None and config.num_speculative_tokens > 0
        )
        self.num_speculative_tokens = config.num_speculative_tokens
        self.speculative_model = None
        self.speculative_hf_config = None
        if self.speculative_decoding:
            self.speculative_hf_config = AutoConfig.from_pretrained(
                config.speculative_model
            )
            draft_model_type = self.speculative_hf_config.model_type
            if draft_model_type not in model_dict:
                raise ValueError(
                    f"不支持的 speculative 模型类型: {draft_model_type!r}。"
                    f"当前支持: {list(model_dict.keys())}"
                )
            draft_model_cls = model_dict[draft_model_type]
            self.speculative_hf_config._awq_config = AWQConfig.from_json(
                config.speculative_model
            )
            self.speculative_model = draft_model_cls(self.speculative_hf_config)
            load_model(self.speculative_model, config.speculative_model)

        self.sampler = Sampler()

        # 3. Warmup
        self.warmup_model()

        # 4. 分配 KV-cache
        self.allocate_kv_cache()

        # 5. 捕获 CUDA Graph
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 6. 初始化 SharedMemory 通信
        if self.world_size > 1:
            if rank == 0:
                try:
                    existing_shm = SharedMemory(name="nanovllm", create=False)
                    existing_shm.close()
                    existing_shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"Error cleaning up shared memory: {e}")
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        """清理资源，显式释放 GPU 显存。"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if self.speculative_decoding:
                if hasattr(self, "draft_graphs"):
                    del self.draft_graphs, self.draft_graph_pool
                if hasattr(self, "verify_graphs"):
                    del self.verify_graphs, self.verify_graph_pool
        # 显式释放 KV cache 和模型权重占用的 GPU 显存
        if hasattr(self, "kv_cache"):
            del self.kv_cache
        if hasattr(self, "draft_kv_cache"):
            del self.draft_kv_cache
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "speculative_model") and self.speculative_model is not None:
            del self.speculative_model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.destroy_process_group()

    def loop(self):
        """子进程消息循环。"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """子进程从 SharedMemory 读取指令。"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """主进程向 SharedMemory 写入指令。"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """统一调用接口。"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """模型预热：用虚拟序列执行一次前向，触发 CUDA kernel JIT 编译。"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = max(
            min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs), 1
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_new_tokens = max_model_len
        self.run(seqs)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据剩余显存分配 KV-cache，支持 target + draft 双模型。"""
        config = self.config
        hf_config = config.hf_config

        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        available_bytes = int(
            total * config.gpu_memory_utilization - used - peak + current
        )

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )

        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.torch_dtype.itemsize
        )

        # 按比例分配主模型和 draft 模型的 KV-cache 显存
        target_split = 1.0
        if self.speculative_decoding:
            sconf = self.speculative_hf_config
            target_kv = (
                hf_config.num_hidden_layers
                * hf_config.num_key_value_heads
                * head_dim
                * hf_config.torch_dtype.itemsize
            )
            draft_kv = (
                sconf.num_hidden_layers
                * sconf.num_key_value_heads
                * getattr(
                    sconf,
                    "head_dim",
                    sconf.hidden_size // sconf.num_attention_heads,
                )
                * sconf.torch_dtype.itemsize
            )
            split_ratio = target_kv / draft_kv
            target_split = split_ratio / (1 + split_ratio)

        main_budget = int(available_bytes * target_split)
        config.num_kvcache_blocks = main_budget // block_bytes
        assert config.num_kvcache_blocks > 0

        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        # Draft 模型 KV-cache
        self.draft_kv_cache = None
        config.num_draft_kvcache_blocks = 0
        if self.speculative_decoding:
            sconf = self.speculative_hf_config
            s_num_kv_heads = sconf.num_key_value_heads // self.world_size
            s_head_dim = getattr(
                sconf,
                "head_dim",
                sconf.hidden_size // sconf.num_attention_heads,
            )
            s_block_bytes = (
                2
                * sconf.num_hidden_layers
                * self.block_size
                * s_num_kv_heads
                * s_head_dim
                * sconf.torch_dtype.itemsize
            )
            spec_budget = available_bytes - main_budget
            config.num_draft_kvcache_blocks = spec_budget // s_block_bytes
            assert config.num_draft_kvcache_blocks > 0

            self.draft_kv_cache = torch.zeros(
                2,
                sconf.num_hidden_layers,
                config.num_draft_kvcache_blocks,
                self.block_size,
                s_num_kv_heads,
                s_head_dim,
                dtype=sconf.torch_dtype,
            )
            layer_id = 0
            for module in self.speculative_model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.draft_kv_cache[0, layer_id]
                    module.v_cache = self.draft_kv_cache[1, layer_id]
                    layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence], use_draft: bool = False):
        """构建 [num_seqs, max_num_blocks] 的 block table tensor。"""
        if use_draft:
            tables = [seq.draft_block_table for seq in seqs]
        else:
            tables = [seq.block_table for seq in seqs]
        max_len = max(len(t) for t in tables)
        block_tables = [t + [-1] * (max_len - len(t)) for t in tables]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_model_input(self, seqs: list[Sequence]):
        """
        统一输入准备：处理 prefill、decode 和混合批次。

        每个序列按 num_cached_tokens 和 num_new_tokens 确定:
        - 输入 token 范围: [num_cached_tokens, num_cached_tokens + num_new_tokens)
        - Q 长度 = num_new_tokens, K 长度 = num_cached_tokens + num_new_tokens
        - 是否需要 logits: 当所有 token 都已处理且有 block_table 时
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        context_lens = []
        seq_need_compute_logits = []

        for seq_index, seq in enumerate(seqs):
            # 判断是否需要计算 logits
            # 条件：该序列所有 token 都已覆盖（cached + new = total），且有 block_table
            if (
                len(seq) == seq.num_cached_tokens + seq.num_new_tokens
                and seq.block_table
            ):
                seq_need_compute_logits.append(seq_index)

            context_lens.append(seq.num_context_tokens)

            # 输入 token：从 cached 位置到 context 末尾
            input_ids.extend(seq[seq.num_cached_tokens : seq.num_context_tokens])
            positions.extend(range(seq.num_cached_tokens, seq.num_context_tokens))

            seqlen_q = seq.num_new_tokens
            seqlen_k = seq.num_context_tokens
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # Slot mapping：严格按本轮实际输入 token 逐个映射，避免把 speculative 预留块算进来
            if not seq.block_table:  # warmup
                continue
            for pos in range(seq.num_cached_tokens, seq.num_context_tokens):
                block_idx = pos // self.block_size
                offset = pos % self.block_size
                block_id = seq.block_table[block_idx]
                slot_mapping.append(block_id * self.block_size + offset)

        # 当 K 长度总和 > Q 长度总和时，说明有 prefix cache 或 decode
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # 转为 GPU tensor
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        seq_need_compute_logits = torch.tensor(
            seq_need_compute_logits, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        set_context(
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            context_lens,
            block_tables,
            seq_need_compute_logits,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """提取温度参数，按 seq_need_compute_logits 过滤。仅 rank 0 调用。"""
        context = get_context()
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        if context.seq_need_compute_logits.numel():
            temperatures = temperatures[context.seq_need_compute_logits]
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """
        执行 target 模型前向推理。

        纯 decode 且序列数 ≤ 512 时使用 CUDA Graph 加速；
        其余情况（prefill、混合批次、大 batch）使用 eager 模式。
        speculative verify 使用专用 verify CUDA Graph（以序列数而非总 token 数判断）。
        """
        context = get_context()
        is_prefill = context.is_prefill
        # verify 阶段每个序列有 K+1 个 token，判断 CUDA Graph 可用性时应用序列数
        # 而非总 token 数（否则 B=256, K=5 → 1536 tokens 会错误地跳过 CUDA Graph）。
        if context.is_speculative:
            effective_bs = input_ids.size(0) // (self.num_speculative_tokens + 1)
        else:
            effective_bs = input_ids.size(0)
        use_cuda_graph = (
            not self.enforce_eager and not is_prefill and effective_bs <= 512
        )

        if not use_cuda_graph:
            return self.model.compute_logits(self.model(input_ids, positions))

        bs = input_ids.size(0)

        if context.is_speculative:
            K1 = self.num_speculative_tokens + 1
            B = bs // K1
            graph = self.verify_graphs[next(x for x in self.graph_bs if x >= B)]
            graph_vars = self.verify_graph_vars
            graph_vars["input_ids"].zero_()
            graph_vars["positions"].zero_()
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["context_lens"].zero_()
            graph_vars["block_tables"].fill_(-1)
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:B] = context.context_lens
            graph_vars["block_tables"][
                :B, : context.block_tables.size(1)
            ] = context.block_tables
        else:
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"].zero_()
            graph_vars["positions"].zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"].fill_(-1)
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables

        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run_draft_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """执行 draft 模型前向推理。"""
        context = get_context()
        is_prefill = context.is_prefill
        use_cuda_graph = (
            not self.enforce_eager and not is_prefill and input_ids.size(0) <= 512
        )

        if not use_cuda_graph:
            return self.speculative_model.compute_logits(
                self.speculative_model(input_ids, positions)
            )

        bs = input_ids.size(0)
        graph = self.draft_graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.draft_graph_vars
        graph_vars["input_ids"].zero_()
        graph_vars["positions"].zero_()
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["context_lens"].zero_()
        graph_vars["block_tables"].fill_(-1)
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][
            :bs, : context.block_tables.size(1)
        ] = context.block_tables
        graph.replay()
        return self.speculative_model.compute_logits(graph_vars["outputs"][:bs])

    # ── Speculative decode 输入准备 ──

    def prepare_draft_prefill(self, seqs: list[Sequence]):
        """为 draft 模型的首次 prefill 准备输入（补齐 draft cache 落后的 token）。"""
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []

        for seq in seqs:
            start = seq.draft_num_cached_tokens
            end = len(seq)
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            seqlen_q = end - start
            seqlen_k = end
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            for pos in range(start, end):
                block_idx = pos // self.block_size
                offset = pos % self.block_size
                block_id = seq.draft_block_table[block_idx]
                slot_mapping.append(block_id * self.block_size + offset)

        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs, use_draft=True)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        set_context(
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
            None,
        )
        return input_ids, positions

    def prepare_draft_decode(self, seqs: list[Sequence]):
        """为 draft 模型的 decode（逐 token 生成）准备输入。"""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            block_id = seq.draft_block_table[-1]
            slot_mapping.append(
                block_id * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs, use_draft=True)
        set_context(
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_verify_decode(self, seqs: list[Sequence]):
        """为 target 模型 verify 阶段准备输入：每个 seq 取 K+1 个 token。"""
        K = self.num_speculative_tokens
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            start_idx = max(0, len(seq) - K - 1)
            end_idx = len(seq) - 1
            input_ids.extend(seq[start_idx : end_idx + 1])
            positions.extend(range(start_idx, end_idx + 1))
            for pos in range(start_idx, end_idx + 1):
                block_idx = pos // self.block_size
                offset = pos % self.block_size
                block_id = seq.block_table[block_idx]
                slot_mapping.append(block_id * self.block_size + offset)
            context_lens.append(len(seq))

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            is_speculative=True,
            num_speculative_tokens=self.num_speculative_tokens,
        )
        return input_ids, positions

    def prepare_sample_flat(self, seqs: list[Sequence]):
        """提取温度参数（不按 seq_need_compute_logits 过滤）。"""
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(
            non_blocking=True
        )

    # ── 推测解码核心 ──

    @torch.inference_mode()
    def generate_draft_tokens(self, seqs, temps, device, dtype):
        """用 draft 模型循环生成 K 个候选 token。"""
        B = len(seqs)
        K = self.num_speculative_tokens
        vocab_size = self.model.lm_head.num_embeddings

        draft_tokens = torch.empty((B, K), dtype=torch.int64, device=device)
        # 概率分布用于 rejection sampling，保持 float32 避免 bf16/fp16 精度损失。
        draft_probs = torch.empty(
            (B, K, vocab_size), dtype=torch.float32, device=device
        )

        # 判断 draft 模型是否需要先做一次 prefill（同步落后的 token）
        needs_prefill = any(seq.draft_num_cached_tokens < len(seq) - 1 for seq in seqs)

        for t in range(K):
            if t == 0 and needs_prefill:
                input_ids, positions = self.prepare_draft_prefill(seqs)
                logits = self.run_draft_model(input_ids, positions)
                # prefill 可能已在 lm_head 中归约为 [B, V]（每个 seq 一个 logit）；
                # 仅当输出仍是逐 token 形状时再按 last_indices 取最后位置。
                if logits.size(0) != B:
                    context = get_context()
                    last_indices = context.cu_seqlens_q[1:] - 1
                    logits = logits[last_indices]
            else:
                input_ids, positions = self.prepare_draft_decode(seqs)
                logits = self.run_draft_model(input_ids, positions)

            reset_context()
            next_tokens, probs = self.sampler(logits, temps, return_probs=True)
            draft_tokens[:, t] = next_tokens
            draft_probs[:, t] = probs

            for i, seq in enumerate(seqs):
                token = int(next_tokens[i].item())
                seq.append_token(token)
                # 更新 draft cache 计数
                if t == 0 and needs_prefill:
                    seq.draft_num_cached_tokens = len(seq) - 1
                else:
                    seq.draft_num_cached_tokens += 1

        return draft_tokens, draft_probs

    @torch.inference_mode()
    def verify_draft_tokens(self, seqs, draft_tokens, draft_probs, temps):
        """用 target 模型一次前向验证 K 个 draft token，执行 rejection sampling。"""
        K = self.num_speculative_tokens
        B = len(seqs)

        for seq in seqs:
            seq.is_speculative = True

        input_ids, positions = self.prepare_verify_decode(seqs)
        logits = self.run_model(input_ids, positions)
        reset_context()

        # 计算 target 概率分布
        temps_rep = temps.repeat_interleave(K + 1)
        probs = self.sampler.compute_temperature_scaled_probs(logits, temps_rep)
        probs = probs.reshape(B, K + 1, -1)
        target_probs = probs[:, :K, :]

        # Rejection sampling
        indices = draft_tokens.unsqueeze(-1)
        p_draft = torch.gather(draft_probs, 2, indices).squeeze(-1)
        p_target = torch.gather(target_probs, 2, indices).squeeze(-1)
        accept_ratio = torch.exp(
            torch.log(p_target.clamp_min(1e-10)) - torch.log(p_draft.clamp_min(1e-10))
        )
        accept_probs = torch.min(torch.ones_like(accept_ratio), accept_ratio)
        accepted = torch.rand_like(accept_probs) < accept_probs
        valid_mask = torch.cumprod(accepted.long(), dim=1).bool()
        num_accepted = valid_mask.sum(dim=1)

        # 计算最终 token
        if (num_accepted < K).any():
            rej_pos = torch.clamp(num_accepted, max=K - 1)
            batch_idx = torch.arange(B, device=rej_pos.device)
            target_at_rej = target_probs[batch_idx, rej_pos]
            draft_at_rej = draft_probs[batch_idx, rej_pos]
            adjusted = torch.clamp(target_at_rej - draft_at_rej, min=0)
            norm = adjusted.sum(dim=-1, keepdim=True)
            adjusted = adjusted / torch.clamp(norm, min=1e-8)
            all_accepted_mask = num_accepted == K
            last_probs = probs[:, -1, :]
            final_probs = torch.where(
                all_accepted_mask.unsqueeze(-1), last_probs, adjusted
            )
        else:
            final_probs = probs[:, -1, :]

        # final_probs 已是概率分布，不能再次走 logits->softmax 采样。
        greedy_tokens = final_probs.argmax(dim=-1)
        sampled_tokens = torch.multinomial(final_probs, num_samples=1).squeeze(1)
        final_tokens = torch.where(temps == 0, greedy_tokens, sampled_tokens)

        for i, seq in enumerate(seqs):
            ac = num_accepted[i].item()
            seq.num_speculative_proposed_total += K
            seq.num_speculative_accepted_total += ac
            seq.pending_accepted_tokens = draft_tokens[i, :ac].tolist()

        return final_tokens.tolist()

    def run_speculative_decode(self, seqs, temperatures):
        """完整推测解码流程：draft → verify → rejection sampling。"""
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        draft_tokens, draft_probs = self.generate_draft_tokens(
            seqs, temperatures, device, dtype
        )
        final_token_ids = self.verify_draft_tokens(
            seqs, draft_tokens, draft_probs, temperatures
        )
        return final_token_ids

    def run(self, seqs: list[Sequence]):
        """
        完整推理流程：数据准备 -> 前向计算 -> 采样 -> 清理。
        返回 (token_ids, seq_need_compute_logits)。

        在 decode 阶段如果启用了 speculative decoding 则走推测解码路径。
        """
        # 判断是否为纯 decode 阶段（所有 seq 只有 1 个 new token）
        is_pure_decode = all(seq.num_new_tokens == 1 for seq in seqs) and all(
            seq.block_table for seq in seqs
        )

        if is_pure_decode and self.speculative_decoding and self.rank == 0:
            temperatures = self.prepare_sample_flat(seqs)
            token_ids = self.run_speculative_decode(seqs, temperatures)
            # speculative 路径：所有 seq 都产出 token
            seq_need_compute_logits = list(range(len(seqs)))
            return token_ids, seq_need_compute_logits

        # 常规路径（prefill / 混合 / 非 speculative decode）
        input_ids, positions = self.prepare_model_input(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions)
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        seq_need_compute_logits = get_context().seq_need_compute_logits.tolist()
        reset_context()
        return token_ids, seq_need_compute_logits

    @torch.inference_mode()
    def capture_cudagraph(self):
        """为预定义 batch sizes 捕获 CUDA Graph，加速 decode 推理。"""
        config = self.config
        hf_config = config.hf_config

        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        # 捕获 draft 模型和 verify 图
        if self.speculative_decoding:
            self._capture_draft_cudagraph()
            self._capture_verify_cudagraph()

    @torch.inference_mode()
    def _capture_draft_cudagraph(self):
        """为 draft 模型捕获 CUDA Graph。"""
        sconf = self.speculative_hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (
            self.config.max_model_len + self.block_size - 1
        ) // self.block_size

        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, sconf.hidden_size)

        self.draft_graphs = {}
        self.draft_graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.speculative_model(
                input_ids[:bs], positions[:bs]
            )  # warmup
            with torch.cuda.graph(graph, self.draft_graph_pool):
                outputs[:bs] = self.speculative_model(
                    input_ids[:bs], positions[:bs]
                )  # capture
            if self.draft_graph_pool is None:
                self.draft_graph_pool = graph.pool()
            self.draft_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.draft_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def _capture_verify_cudagraph(self):
        """为 target 模型 verify（K+1 tokens/seq）捕获 CUDA Graph。"""
        config = self.config
        hf_config = config.hf_config
        K1 = self.num_speculative_tokens + 1
        max_B = min(config.max_num_seqs, 512)
        max_tokens = max_B * K1
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(max_tokens, dtype=torch.int64)
        positions = torch.zeros(max_tokens, dtype=torch.int64)
        slot_mapping = torch.zeros(max_tokens, dtype=torch.int32)
        context_lens = torch.zeros(max_B, dtype=torch.int32)
        block_tables = torch.zeros(max_B, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_tokens, hf_config.hidden_size)

        self.verify_graphs = {}
        self.verify_graph_pool = None

        for B in reversed(self.graph_bs):
            if B > max_B:
                continue
            n = B * K1
            graph = torch.cuda.CUDAGraph()
            set_context(
                slot_mapping=slot_mapping[:n],
                context_lens=context_lens[:B],
                block_tables=block_tables[:B],
                is_speculative=True,
                num_speculative_tokens=self.num_speculative_tokens,
            )
            outputs[:n] = self.model(input_ids[:n], positions[:n])  # warmup
            with torch.cuda.graph(graph, self.verify_graph_pool):
                outputs[:n] = self.model(input_ids[:n], positions[:n])  # capture
            if self.verify_graph_pool is None:
                self.verify_graph_pool = graph.pool()
            self.verify_graphs[B] = graph
            torch.cuda.synchronize()
            reset_context()

        self.verify_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
