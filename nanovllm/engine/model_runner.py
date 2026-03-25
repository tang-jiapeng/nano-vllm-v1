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

from nanovllm.config import Config
from nanovllm.engine.ngram_proposer import NgramProposer
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.models import model_dict
from nanovllm.utils.context import get_context, reset_context, set_context
from nanovllm.utils.loader import load_model


def _get_model_dtype(hf_config):
    dtype = getattr(hf_config, "dtype", None)
    if dtype is None:
        dtype = getattr(hf_config, "torch_dtype")
    return dtype


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
        self.enable_kv_offload = config.enable_kv_offload
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
        torch.set_default_dtype(_get_model_dtype(hf_config))
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
            config.speculative_method == "ngram" and config.num_speculative_tokens > 0
        )
        self.num_speculative_tokens = config.num_speculative_tokens
        self.speculative_method = config.speculative_method
        self.ngram_proposer = None
        if self.speculative_decoding:
            self.ngram_proposer = NgramProposer(
                min_ngram=config.ngram_prompt_lookup_min,
                max_ngram=config.ngram_prompt_lookup_max,
            )
        self.sampler = Sampler()
        if self.enable_kv_offload:
            self.swap_stream = torch.cuda.Stream(device=rank)
            self.swap_event = torch.cuda.Event()

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
        # 显式释放 KV cache 和模型权重占用的 GPU 显存
        if hasattr(self, "gpu_kv_cache"):
            del self.gpu_kv_cache
        if hasattr(self, "cpu_kv_cache"):
            del self.cpu_kv_cache
        if hasattr(self, "model"):
            del self.model
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
        """根据剩余显存分配 KV-cache，布局: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]。"""
        config = self.config
        hf_config = config.hf_config

        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

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
            * _get_model_dtype(hf_config).itemsize
        )

        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0

        self.gpu_kv_cache = torch.empty(
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
                module.k_cache = self.gpu_kv_cache[0, layer_id]
                module.v_cache = self.gpu_kv_cache[1, layer_id]
                layer_id += 1

        if self.enable_kv_offload:
            cpu_offload_gb = config.cpu_offload_gb
            if isinstance(cpu_offload_gb, str):
                if cpu_offload_gb != "auto":
                    raise ValueError(
                        "cpu_offload_gb must be a non-negative float or 'auto'"
                    )
                try:
                    import psutil
                except ImportError as exc:
                    raise ImportError(
                        "CPU KV offload requires psutil when cpu_offload_gb='auto'."
                    ) from exc
                available_gb = psutil.virtual_memory().available / (1024**3)
                allocated_cpu_gb = available_gb - config.cpu_offload_safety_margin_gb
                if allocated_cpu_gb <= 0:
                    raise RuntimeError(
                        "Not enough CPU memory for KV offload after reserving "
                        f"{config.cpu_offload_safety_margin_gb:.2f} GB safety margin."
                    )
            else:
                allocated_cpu_gb = float(cpu_offload_gb)
                if allocated_cpu_gb <= 0:
                    raise ValueError(
                        "CPU KV offload is enabled but cpu_offload_gb is not positive."
                    )

            config.num_cpu_kvcache_blocks = int((allocated_cpu_gb * 1024**3) // block_bytes)
            assert config.num_cpu_kvcache_blocks > 0, (
                "Not enough CPU memory to hold at least one KV cache block."
            )
            self.cpu_kv_cache = torch.empty(
                2,
                hf_config.num_hidden_layers,
                config.num_cpu_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                device="cpu",
                pin_memory=True,
            )
        else:
            config.num_cpu_kvcache_blocks = 0
            self.cpu_kv_cache = None

    def execute_swap(
        self,
        swap_in_map: dict[int, int] | None = None,
        swap_out_map: dict[int, int] | None = None,
    ) -> bool:
        """
        执行 GPU <-> CPU KV block 迁移。

        Args:
            swap_in_map: {cpu_block_id: gpu_block_id}
            swap_out_map: {gpu_block_id: cpu_block_id}

        Returns:
            是否实际提交了异步 copy。
        """
        if not self.enable_kv_offload:
            return False

        swap_in_map = swap_in_map or {}
        swap_out_map = swap_out_map or {}
        if not swap_in_map and not swap_out_map:
            return False

        with torch.cuda.stream(self.swap_stream):
            for gpu_block_id, cpu_block_id in swap_out_map.items():
                self.cpu_kv_cache[:, :, cpu_block_id].copy_(
                    self.gpu_kv_cache[:, :, gpu_block_id], non_blocking=True
                )
            for cpu_block_id, gpu_block_id in swap_in_map.items():
                self.gpu_kv_cache[:, :, gpu_block_id].copy_(
                    self.cpu_kv_cache[:, :, cpu_block_id], non_blocking=True
                )
        self.swap_event.record(self.swap_stream)
        return True

    def prepare_block_tables(self, seqs: list[Sequence]):
        """构建 [num_seqs, max_num_blocks] 的 block table tensor。"""
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
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

            # Slot mapping：严格按本轮实际输入 token 逐个映射，
            # 避免 speculative 预留块被错误计入本轮写入范围。
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
        执行模型前向推理。

        纯 decode 且 batch ≤ 512 时使用 CUDA Graph 加速；
        其余情况（prefill、混合批次、speculative verify、大 batch）使用 eager 模式。
        """
        context = get_context()
        # 仅纯 decode（所有序列 q_len=1）且小 batch 时使用 CUDA Graph
        use_cuda_graph = (
            not self.enforce_eager
            and input_ids.size(0) <= 512
            and ((context.max_seqlen_q <= 1) or context.is_speculative)
        )

        if not use_cuda_graph:
            return self.model.compute_logits(self.model(input_ids, positions))
        elif context.is_speculative:
            proposal_len = context.num_speculative_tokens
            group_size = proposal_len + 1
            bs = input_ids.size(0) // group_size
            graph = self.verify_graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.verify_graph_vars

            graph_vars["input_ids"].zero_()
            graph_vars["positions"].zero_()
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["context_lens"].zero_()
            graph_vars["block_tables"].fill_(-1)
            graph_vars["input_ids"][: input_ids.size(0)] = input_ids
            graph_vars["positions"][: input_ids.size(0)] = positions
            graph_vars["slot_mapping"][: input_ids.size(0)] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables

            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][: input_ids.size(0)])
        else:
            bs = input_ids.size(0)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars

            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables

            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def prepare_sample_flat(self, seqs: list[Sequence]):
        """提取温度参数（不按 seq_need_compute_logits 过滤）。"""
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(
            non_blocking=True
        )

    def prepare_verify_decode(self, seqs: list[Sequence], proposal_len: int):
        """为 target 模型 verify 阶段准备输入：每个 seq 取 K+1 个 token。"""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            start_idx = max(0, len(seq) - proposal_len - 1)
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
            num_speculative_tokens=proposal_len,
        )
        return input_ids, positions

    def propose_ngram_tokens(self, seqs: list[Sequence]) -> list[list[int]]:
        """为每个序列基于 token 历史生成 N-gram draft token。"""
        return [
            self.ngram_proposer.propose(seq.token_ids, self.num_speculative_tokens)
            for seq in seqs
        ]

    @torch.inference_mode()
    def run_standard_decode(
        self, seqs: list[Sequence], temperatures: torch.Tensor | None
    ):
        """对给定 decode 子批次执行常规单 token 解码。"""
        input_ids, positions = self.prepare_model_input(seqs)
        logits = self.run_model(input_ids, positions)
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        seq_need_compute_logits = get_context().seq_need_compute_logits.tolist()
        reset_context()
        return token_ids, seq_need_compute_logits

    @torch.inference_mode()
    def verify_ngram_proposals(
        self,
        seqs: list[Sequence],
        proposals: list[list[int]],
        temperatures: torch.Tensor | None,
    ) -> list[int] | None:
        """验证 full-K N-gram proposals。"""
        if not seqs:
            return [] if self.rank == 0 else None

        proposal_len = len(proposals[0])
        token_ids_by_index = [None] * len(seqs) if self.rank == 0 else None

        for seq, draft in zip(seqs, proposals):
            seq.is_speculative = True
            seq.speculative_draft_tokens = list(draft)
            for token_id in draft:
                seq.append_token(token_id)

        input_ids, positions = self.prepare_verify_decode(seqs, proposal_len)
        logits = self.run_model(input_ids, positions)
        reset_context()

        if self.rank != 0:
            return None

        sampled = self.sampler(
            logits, temperatures.repeat_interleave(proposal_len + 1)
        ).view(len(seqs), proposal_len + 1)

        for row, (seq, draft) in enumerate(zip(seqs, proposals)):
            accepted = 0
            while accepted < proposal_len and int(sampled[row, accepted]) == draft[accepted]:
                accepted += 1

            seq.num_speculative_proposed_total += proposal_len
            seq.num_speculative_accepted_total += accepted
            seq.pending_accepted_tokens = draft[:accepted]

            if accepted < proposal_len:
                token_ids_by_index[row] = int(sampled[row, accepted].item())
            else:
                token_ids_by_index[row] = int(sampled[row, proposal_len].item())

        return token_ids_by_index

    def run_speculative_decode(self, seqs: list[Sequence], temperatures: torch.Tensor):
        """N-gram speculative decode：propose -> target verify -> prefix accept."""
        proposals = self.propose_ngram_tokens(seqs)
        token_ids = [None] * len(seqs) if self.rank == 0 else None

        full_k = self.num_speculative_tokens
        regular_indices = [
            idx for idx, draft in enumerate(proposals) if len(draft) != full_k
        ]
        speculative_indices = [
            idx for idx, draft in enumerate(proposals) if len(draft) == full_k
        ]

        if regular_indices:
            regular_seqs = [seqs[idx] for idx in regular_indices]
            regular_temps = (
                temperatures[
                    torch.tensor(regular_indices, device=temperatures.device)
                ]
                if self.rank == 0
                else None
            )
            regular_token_ids, _ = self.run_standard_decode(regular_seqs, regular_temps)
            if self.rank == 0:
                for idx, token_id in zip(regular_indices, regular_token_ids):
                    token_ids[idx] = token_id

        if speculative_indices:
            spec_seqs = [seqs[idx] for idx in speculative_indices]
            spec_proposals = [proposals[idx] for idx in speculative_indices]
            spec_temps = (
                temperatures[
                    torch.tensor(speculative_indices, device=temperatures.device)
                ]
                if self.rank == 0
                else None
            )
            speculative_token_ids = self.verify_ngram_proposals(
                spec_seqs, spec_proposals, spec_temps
            )
            if self.rank == 0:
                for local_idx, token_id in enumerate(speculative_token_ids):
                    token_ids[speculative_indices[local_idx]] = token_id

        return token_ids

    def run(
        self,
        seqs: list[Sequence],
        swap_in_map: dict[int, int] | None = None,
        swap_out_map: dict[int, int] | None = None,
    ):
        """
        完整推理流程：数据准备 -> 前向计算 -> 采样 -> 清理。
        返回 (token_ids, seq_need_compute_logits)。
        """
        has_swaps = self.execute_swap(swap_in_map, swap_out_map)
        if not seqs:
            if has_swaps:
                self.swap_event.wait(torch.cuda.current_stream())
            return [], []

        is_pure_decode = all(seq.num_new_tokens == 1 for seq in seqs) and all(
            seq.block_table for seq in seqs
        )

        if is_pure_decode and self.speculative_decoding:
            temperatures = self.prepare_sample_flat(seqs) if self.rank == 0 else None
            token_ids = self.run_speculative_decode(seqs, temperatures)
            seq_need_compute_logits = list(range(len(seqs)))
            return token_ids, seq_need_compute_logits

        input_ids, positions = self.prepare_model_input(seqs)
        if has_swaps:
            self.swap_event.wait(torch.cuda.current_stream())
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
            # CUDA Graph 以 decode 模式捕获（不设置 cu_seqlens_q → is_prefill=False）
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

        if self.speculative_decoding:
            self._capture_verify_cudagraph()

    @torch.inference_mode()
    def _capture_verify_cudagraph(self):
        """为 target 模型 verify（K+1 tokens/seq）捕获 CUDA Graph。"""
        config = self.config
        hf_config = config.hf_config
        proposal_len = self.num_speculative_tokens
        group_size = proposal_len + 1
        max_bs = min(config.max_num_seqs, 512)
        max_tokens = max_bs * group_size
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(max_tokens, dtype=torch.int64)
        positions = torch.zeros(max_tokens, dtype=torch.int64)
        slot_mapping = torch.zeros(max_tokens, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_tokens, hf_config.hidden_size)

        self.verify_graphs = {}
        self.verify_graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            n = bs * group_size
            set_context(
                slot_mapping=slot_mapping[:n],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                is_speculative=True,
                num_speculative_tokens=proposal_len,
            )
            outputs[:n] = self.model(input_ids[:n], positions[:n])
            with torch.cuda.graph(graph, self.verify_graph_pool):
                outputs[:n] = self.model(input_ids[:n], positions[:n])
            if self.verify_graph_pool is None:
                self.verify_graph_pool = graph.pool()
            self.verify_graphs[bs] = graph
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
