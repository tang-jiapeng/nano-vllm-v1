"""
vLLM v1 风格统一调度器，支持 chunked prefill 与 CPU KV offload。

核心设计：
- 统一 running / waiting 队列（无独立 prefill/decode 阶段）
- 基于 token budget 的灵活调度（decode 优先，保证低延迟）
- Chunked prefill：长 prompt 分块处理，与 decode 混合执行
- 抢占机制：KV-cache 满时从 running 队列尾部抢占
- CPU offload：被抢占的序列仍回到 waiting，通过 residency 优先恢复

调度流程：
  1. Phase 1: 调度 running 队列（decode token + chunked prefill 续接）
  2. Phase 2: 调度 waiting 队列（offloaded 请求优先恢复，其次新请求）
     - 仅在 Phase 1 无抢占时执行
"""

from collections import deque
from dataclasses import dataclass, field

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import CacheResidency, Sequence, SequenceStatus


@dataclass
class SchedulerStep:
    """单步调度结果，携带本轮执行序列与 swap 元数据。"""

    seqs: list[Sequence] = field(default_factory=list)
    swap_in_map: dict[int, int] = field(default_factory=dict)
    swap_out_map: dict[int, int] = field(default_factory=dict)


class Scheduler:
    """统一调度器，基于 token budget 调度 prefill 和 decode 混合批次。"""

    def __init__(self, config: Config):
        self.enable_chunked = config.chunked_prefill
        self.enable_kv_offload = config.enable_kv_offload
        speculative_method = getattr(config, "speculative_method", None)
        num_speculative_tokens = getattr(config, "num_speculative_tokens", 0)
        self.speculative_decoding = (
            speculative_method is not None and num_speculative_tokens > 0
        )
        self.num_speculative_tokens = num_speculative_tokens
        self.max_model_len = config.max_model_len
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            speculative_decoding=self.speculative_decoding,
            num_speculative_tokens=self.num_speculative_tokens,
            num_cpu_blocks=config.num_cpu_kvcache_blocks,
            cpu_offload_watermark_blocks=config.cpu_offload_watermark_blocks,
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

    def _pop_next_waiting(self) -> Sequence | None:
        """优先选择 CPU-offloaded 的等待序列，其次选择普通 waiting 序列。"""
        cpu_idx = next(
            (
                i
                for i, seq in enumerate(self.waiting)
                if seq.residency == CacheResidency.CPU
            ),
            None,
        )
        if cpu_idx is None:
            return self.waiting.popleft() if self.waiting else None

        self.waiting.rotate(-cpu_idx)
        seq = self.waiting.popleft()
        self.waiting.rotate(cpu_idx)
        return seq

    def _peek_next_offloaded_waiting(self) -> Sequence | None:
        """查看 waiting 中下一个需要优先恢复的 CPU-offloaded 序列。"""
        for seq in self.waiting:
            if seq.residency == CacheResidency.CPU:
                return seq
        return None

    def _compute_num_new_tokens(self, seq: Sequence, token_budget: int) -> int:
        """计算序列本轮需处理的 token 数。"""
        num_new_tokens = len(seq) - seq.num_cached_tokens
        if self.enable_chunked:
            num_new_tokens = min(num_new_tokens, token_budget)
        num_new_tokens = min(
            num_new_tokens, self.max_model_len - 1 - seq.num_cached_tokens
        )
        return num_new_tokens

    def _preempt(self, seq: Sequence, step: SchedulerStep):
        """抢占序列：优先 offload 到 CPU，否则回退为 recompute。"""
        seq.status = SequenceStatus.WAITING
        if self.enable_kv_offload and self.block_manager.can_swap_out(seq):
            step.swap_out_map.update(self.block_manager.swap_out(seq))
            self._append_waiting_offloaded(seq)
        else:
            self.block_manager.deallocate(seq)
            self.waiting.appendleft(seq)

    def _append_waiting_offloaded(self, seq: Sequence):
        """将 CPU-offloaded 序列按 FIFO 顺序插入 waiting 的 offloaded 区域。"""
        cpu_count = sum(1 for waiting_seq in self.waiting if waiting_seq.residency == CacheResidency.CPU)
        self.waiting.rotate(-cpu_count)
        self.waiting.appendleft(seq)
        self.waiting.rotate(cpu_count)

    def _ensure_resume_capacity(
        self, step: SchedulerStep, token_budget: int
    ) -> list[Sequence]:
        """
        为 waiting 中的 CPU-offloaded 序列主动腾出恢复空间。

        设计思路与官方 swapped 请求优先的精神保持一致：
        当系统中已经存在 offloaded 请求时，允许主动从 running 队尾抢占，
        避免 waiting 中的 offloaded 请求长时间得不到恢复机会。
        """
        proactively_preempted = []
        if not self.enable_kv_offload:
            return proactively_preempted

        while self.running:
            seq = self._peek_next_offloaded_waiting()
            if seq is None:
                break

            num_new_tokens = self._compute_num_new_tokens(seq, token_budget)
            if num_new_tokens <= 0 or self.block_manager.can_swap_in(
                seq, num_new_tokens
            ):
                break

            preempted_seq = self.running.pop()
            self._preempt(preempted_seq, step)
            proactively_preempted.append(preempted_seq)

        return proactively_preempted

    def schedule(self) -> SchedulerStep:
        """
        执行一次统一调度决策。

        返回设置了 num_new_tokens 的已调度序列列表，以及本轮需要执行的
        swap in / swap out 映射。
        """
        step = SchedulerStep()
        scheduled_running_seqs = []
        scheduled_waiting_seqs = []
        preempted_seqs = []
        token_budget = self.max_num_batched_tokens

        # 若 waiting 中已有 CPU-offloaded 请求，先主动为恢复请求腾挪 GPU 块。
        preempted_seqs.extend(self._ensure_resume_capacity(step, token_budget))

        # ═══ Phase 1: 调度 running 队列 ═══
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            seq = self.running[req_index]
            num_new_tokens = self._compute_num_new_tokens(seq, token_budget)
            assert num_new_tokens > 0

            while True:
                if self.block_manager.can_append(seq, num_new_tokens):
                    seq.num_new_tokens = num_new_tokens
                    self.block_manager.may_append(seq)
                    break

                preempted_seq = self.running.pop()
                self._preempt(preempted_seq, step)
                preempted_seqs.append(preempted_seq)
                if len(self.running) == req_index:
                    break

            if len(self.running) == req_index:
                break

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
                seq = self._pop_next_waiting()
                if seq is None:
                    break

                num_new_tokens = self._compute_num_new_tokens(seq, token_budget)
                assert num_new_tokens > 0

                if seq.residency == CacheResidency.CPU:
                    if not self.block_manager.can_swap_in(seq, num_new_tokens):
                        self.waiting.appendleft(seq)
                        break

                    seq.num_new_tokens = num_new_tokens
                    step.swap_in_map.update(self.block_manager.swap_in(seq))
                    assert self.block_manager.can_append(seq, num_new_tokens), (
                        "swap_in succeeded but append still failed; "
                        "can_swap_in should reserve enough blocks."
                    )
                    self.block_manager.may_append(seq)
                else:
                    assert not seq.block_table
                    num_computed_in_used, num_computed_in_cached, num_new_tokens = (
                        self.block_manager.get_token_layout(seq)
                    )
                    if self.enable_chunked:
                        num_new_tokens = min(num_new_tokens, token_budget)

                    if (
                        num_new_tokens <= 0
                        or num_new_tokens > token_budget
                        or not self.block_manager.can_allocate(
                            num_computed_in_cached + num_new_tokens
                        )
                    ):
                        self.waiting.appendleft(seq)
                        break

                    seq.num_new_tokens = num_new_tokens
                    self.block_manager.allocate(seq)
                    assert (
                        seq.num_cached_tokens
                        == num_computed_in_used + num_computed_in_cached
                    )

                token_budget -= seq.num_new_tokens
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_waiting_seqs.append(seq)

        step.seqs = scheduled_running_seqs + scheduled_waiting_seqs
        return step

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

            if seq.is_speculative:
                proposed_count = len(seq.speculative_draft_tokens)
                accepted_count = len(seq.pending_accepted_tokens)

                if accepted_count > 0:
                    end_index = -1
                    for i, t in enumerate(seq.pending_accepted_tokens):
                        if (not seq.ignore_eos and t == self.eos) or (
                            seq.num_completion_tokens - proposed_count + i + 1
                            == seq.max_tokens
                        ):
                            end_index = i
                            break

                    if end_index != -1:
                        to_pop = proposed_count - (end_index + 1)
                        if to_pop > 0:
                            seq.pop_last_n_tokens(to_pop)
                        seq.speculative_draft_tokens.clear()
                        seq.pending_accepted_tokens.clear()
                        seq.status = SequenceStatus.FINISHED
                        self.block_manager.deallocate(seq)
                        if seq in self.running:
                            self.running.remove(seq)
                        continue

                if accepted_count < proposed_count:
                    to_pop = proposed_count - accepted_count
                    seq.pop_last_n_tokens(to_pop)

                seq.speculative_draft_tokens.clear()
                seq.pending_accepted_tokens.clear()
                seq.append_token(token_id)

                seq.num_cached_tokens = seq.num_cached_tokens + accepted_count + 1
                seq.num_new_tokens = 0
                seq.is_speculative = False

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
                    if seq in self.running:
                        self.running.remove(seq)
                continue

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

        for seq in seqs:
            if seq.status != SequenceStatus.FINISHED and (not seq.is_speculative):
                seq.num_cached_tokens = seq.num_cached_tokens + seq.num_new_tokens
                seq.num_new_tokens = 0
