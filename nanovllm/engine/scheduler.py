"""
vLLM v1 风格统一调度器，支持 chunked prefill。

核心设计：
- 统一 running 队列（无独立 prefill/decode 阶段）
- 基于 token budget 的灵活调度（decode 优先，保证低延迟）
- Chunked prefill：长 prompt 分块处理，与 decode 混合执行
- 抢占机制：KV-cache 满时从 running 队列尾部抢占

调度流程：
  1. Phase 1: 调度 running 队列（decode token + chunked prefill 续接）
  2. Phase 2: 调度 waiting 队列（新请求的首个 chunk）
     - 仅在 Phase 1 无抢占时执行
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """统一调度器，基于 token budget 调度 prefill 和 decode 混合批次。"""

    def __init__(self, config: Config):
        self.enable_chunked = config.chunked_prefill
        self.max_model_len = config.max_model_len
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """所有队列均为空时返回 True。"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """将新序列追加到 waiting 队列。"""
        assert (
            len(seq) <= self.max_model_len - 1
        ), f"Sequence length {len(seq)} exceeds max_model_len {self.max_model_len}"
        self.waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        """
        执行一次统一调度决策。

        返回设置了 num_new_tokens 的已调度序列列表。
        在混合批次中，decode 序列（num_new_tokens=1）和
        prefill 块（num_new_tokens>1）可以共存。
        """
        scheduled_running_seqs = []
        scheduled_new_reqs = []
        preempted_seqs = []
        token_budget = self.max_num_batched_tokens

        # ═══ Phase 1: 调度 running 队列 ═══
        # running 中的序列可能是 decode（1 个新 token）或
        # chunked prefill 的后续块（多个新 token）
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            seq = self.running[req_index]

            # 计算本次需处理的 token 数
            num_new_tokens = len(seq) - seq.num_cached_tokens
            if self.enable_chunked:
                num_new_tokens = min(num_new_tokens, token_budget)
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - seq.num_cached_tokens
            )
            assert num_new_tokens > 0

            # 尝试分配块，不足时从尾部抢占
            while True:
                if self.block_manager.can_append(seq, num_new_tokens):
                    seq.num_new_tokens = num_new_tokens
                    self.block_manager.may_append(seq)
                    break
                # 抢占 running 队列尾部的序列
                preempted_seq = self.running.pop()
                self.preempt(preempted_seq)
                preempted_seqs.append(preempted_seq)
                if len(self.running) == req_index:
                    break  # 无法再抢占（当前序列已是最后一个）

            if len(self.running) == req_index:
                break  # 当前序列被自身抢占

            scheduled_running_seqs.append(seq)
            token_budget -= seq.num_new_tokens
            req_index += 1

        # ═══ Phase 2: 调度 waiting 队列（仅在无抢占时执行）═══
        if not preempted_seqs:
            while (
                self.waiting
                and token_budget > 0
                and len(self.running) < self.max_num_seqs
            ):
                seq = self.waiting[0]
                assert not seq.block_table

                # 分析 prefix cache 命中情况
                num_computed_in_used, num_computed_in_cached, num_new_tokens = (
                    self.block_manager.get_token_layout(seq)
                )

                if self.enable_chunked:
                    num_new_tokens = min(num_new_tokens, token_budget)

                assert num_new_tokens > 0

                # 检查资源：token budget + block 可分配性
                if (
                    num_new_tokens > token_budget
                    or not self.block_manager.can_allocate(
                        num_computed_in_cached + num_new_tokens
                    )
                ):
                    break

                # 调度该序列
                seq.num_new_tokens = num_new_tokens
                self.block_manager.allocate(seq)
                assert (
                    seq.num_cached_tokens
                    == num_computed_in_used + num_computed_in_cached
                )

                token_budget -= num_new_tokens
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_new_reqs.append(seq)

        scheduled_seqs = scheduled_running_seqs + scheduled_new_reqs
        assert scheduled_seqs, "No sequences could be scheduled (possible OOM)"
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        """抢占序列：释放其 KV-cache 块，状态置为 WAITING 并插入 waiting 队列头部。"""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self, seqs: list[Sequence], token_ids: list[int], seq_need_compute_logits
    ):
        """
        推理后处理：
        1. 为需要 logits 的序列追加生成的 token
        2. 检查终止条件（EOS / max_tokens / max_model_len）
        3. 更新所有未完成序列的 num_cached_tokens
        """
        assert len(token_ids) == len(seq_need_compute_logits)

        for seq_index, token_id in zip(seq_need_compute_logits, token_ids):
            seq = seqs[seq_index]
            seq.append_token(token_id)

            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
                or len(seq) >= self.max_model_len
            ):
                if len(seq) >= self.max_model_len:
                    print(
                        f"Sequence {seq.seq_id} reached max_model_len {self.max_model_len}."
                    )
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

        # 更新所有未完成序列的缓存计数
        for seq in seqs:
            if seq.status != SequenceStatus.FINISHED:
                seq.num_cached_tokens = seq.num_cached_tokens + seq.num_new_tokens
                seq.num_new_tokens = 0
