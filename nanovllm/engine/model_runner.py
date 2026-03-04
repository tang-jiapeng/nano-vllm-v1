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
from nanovllm.engine.sequence import Sequence
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
        self.model = model_cls(hf_config)
        load_model(self.model, config.model)
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
        # 显式释放 KV cache 和模型权重占用的 GPU 显存
        if hasattr(self, "kv_cache"):
            del self.kv_cache
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
            * hf_config.torch_dtype.itemsize
        )

        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
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

            # Slot mapping：只为新块中的 token 分配
            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_cached_blocks, len(seq.block_table)):
                if i == seq.num_cached_blocks:
                    start = (
                        seq.block_table[i] * self.block_size
                        + seq.num_cached_tokens % self.block_size
                    )
                else:
                    start = seq.block_table[i] * self.block_size
                if i == len(seq.block_table) - 1:
                    end = (
                        seq.block_table[i] * self.block_size
                        + seq.num_context_tokens % self.block_size
                        if seq.num_context_tokens % self.block_size != 0
                        else (seq.block_table[i] + 1) * self.block_size
                    )
                else:
                    end = (seq.block_table[i] + 1) * self.block_size
                slot_mapping.extend(list(range(start, end)))

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
        其余情况（prefill、混合批次、大 batch）使用 eager 模式。
        """
        context = get_context()
        # 仅纯 decode（所有序列 q_len=1）且小 batch 时使用 CUDA Graph
        use_cuda_graph = (
            not self.enforce_eager
            and context.max_seqlen_q <= 1
            and input_ids.size(0) <= 512
        )

        if not use_cuda_graph:
            return self.model.compute_logits(self.model(input_ids, positions))
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

    def run(self, seqs: list[Sequence]):
        """
        完整推理流程：数据准备 -> 前向计算 -> 采样 -> 清理。
        返回 (token_ids, seq_need_compute_logits)。
        """
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
