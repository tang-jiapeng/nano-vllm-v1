from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """
    调度器，管理 waiting/running 双队列，实现 prefill/decode 两级调度，
    处理 KV-cache 分配和内存不足时的 preemption。
    """

    def __init__(self, config: Config):
        """初始化调度器，创建 BlockManager 及 waiting/running 队列。"""
        # 最大并发序列数（同时运行的序列数量）
        self.max_num_seqs = config.max_num_seqs

        # 批处理的最大token数（一次处理的token总数）
        self.max_num_batched_tokens = config.max_num_batched_tokens

        # 结束符ID，用于判断序列是否结束
        self.eos = config.eos

        # 创建KV-cache块管理器
        # 负责分配、释放和管理KV-cache块，支持prefix caching
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            enable_prefix_caching=config.enable_prefix_caching,
        )

        # 等待队列：包含所有已提交但未开始处理的序列
        self.waiting: deque[Sequence] = deque()

        # 运行队列：包含所有正在处理的序列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """所有队列均为空时返回 True。"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """将新序列追加到 waiting 队列尾部。"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        执行一次调度决策。

        Returns:
            (scheduled_seqs, is_prefill):
            - is_prefill=True: 从 waiting 队列批量取出序列做 prefill
            - is_prefill=False: 对 running 队列中的序列做 decode，
              内存不足时执行 preemption
        """
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        # ===== Prefill阶段：处理等待队列 =====
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            # 检查是否超出批处理token数限制
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                break

            # 检查KV-cache块是否足够分配
            if not self.block_manager.can_allocate(seq):
                break

            # 选择该序列进行处理
            num_seqs += 1
            self.block_manager.allocate(seq)  # 分配KV-cache块

            # 统计新处理的token数（排除缓存的token）
            num_batched_tokens += len(seq) - seq.num_cached_tokens

            # 更新序列状态并移动队列
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()  # 从等待队列移除
            self.running.append(seq)  # 加入运行队列
            scheduled_seqs.append(seq)

        # 如果有prefill序列，直接返回
        if scheduled_seqs:
            return scheduled_seqs, True

        # ===== Decode阶段：处理运行队列 =====
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()

            # 检查是否可以追加新token（KV-cache空间检查）
            while not self.block_manager.can_append(seq):
                # 内存不足，执行抢占
                if self.running:
                    # 抢占队列中最长的序列
                    self.preempt(self.running.pop())
                else:
                    # 只能抢占当前序列
                    self.preempt(seq)
                    break  # 跳出内层while循环
            else:
                # 内存充足，可以处理
                num_seqs += 1
                self.block_manager.may_append(seq)  # 可能需要分配新块
                scheduled_seqs.append(seq)

        # 保证队列顺序：将decode的序列放回队列前端（reverse twice preserves order）
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占序列：释放其 KV-cache 块，状态置为 WAITING 并插入 waiting 队列头部。
        - seq: 要抢占的序列

        抢占策略：
        1. 优先抢占最长的序列（释放更多KV-cache）
        2. 将序列状态从RUNNING改为WAITING
        3. 释放所有KV-cache块
        4. 重新加入等待队列头部（高优先级重新调度）

        使用场景：
        1. KV-cache空间不足，无法处理新token
        2. 内存压力过大，需要回收资源
        3. 负载均衡，避免某个序列占用过多资源

        注意：
        抢占会导致序列重新处理（recompute），但保证了系统的稳定运行
        """
        # 1. 更新序列状态
        seq.status = SequenceStatus.WAITING

        # 2. 释放KV-cache块
        self.block_manager.deallocate(seq)

        # 3. 重新加入等待队列头部（优先重新调度）
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        推理后处理：将生成的 token 追加到序列，
        检查 EOS / max_tokens 终止条件，完成的序列释放资源并移出 running 队列。
        """
        finished_flags = []

        for seq, token_id in zip(seqs, token_ids):
            # 1. 将生成的token添加到序列中
            seq.append_token(token_id)

            # 2. 检查是否应该结束
            should_finish = False

            # 情况1：遇到EOS token且未忽略EOS
            if not seq.ignore_eos and token_id == self.eos:
                should_finish = True

            # 情况2：达到最大生成token数
            elif seq.num_completion_tokens == seq.max_tokens:
                should_finish = True

            # 3. 如果应该结束，更新状态并释放资源
            if should_finish:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # 释放KV-cache
                self.running.remove(seq)  # 从运行队列移除

            finished_flags.append(should_finish)

        return finished_flags
