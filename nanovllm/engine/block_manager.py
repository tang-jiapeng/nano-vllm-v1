"""
KV-cache 块管理器，支持 prefix caching、LRU eviction、chunked prefill
以及 CPU KV offload。

GPU 块生命周期：
    free_block_ids → used_block_ids → cached_blocks (LRU) → 被驱逐 → free_block_ids

CPU 块生命周期：
    free_cpu_block_ids → used_cpu_block_ids → free_cpu_block_ids

设计边界：
    - prefix caching 仅发生在 GPU resident blocks 上
    - CPU 块只作为 KV offload backing store，不参与 prefix cache 查找
"""

from collections import OrderedDict, deque

import numpy as np
import xxhash

from nanovllm.engine.sequence import CacheResidency, Sequence


class Block:
    """KV-cache 块，通过引用计数管理生命周期，通过 hash 支持 prefix caching。"""

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """更新块的 hash 和 token 数据。"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态（分配时调用）：设 ref_count=1，清空 hash 和 token。"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    管理所有 KV-cache 块的分配/释放，支持：
    - 基于 xxhash 的 GPU prefix caching（O(1) 查找复用）
    - GPU LRU eviction（空闲块用完时驱逐最久未使用的缓存块）
    - chunked prefill（一次分配多个新块）
    - CPU KV offload（GPU <-> CPU block 元数据与物理块映射）
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        speculative_decoding: bool = False,
        num_speculative_tokens: int = 0,
        num_cpu_blocks: int = 0,
        cpu_offload_watermark_blocks: int = 0,
    ):
        self.block_size = block_size

        # GPU 块池
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        # LRU 缓存：hash → block_id，最近使用的在末尾
        self.cached_blocks: OrderedDict[int, int] = OrderedDict()

        # CPU 块池：只作为 offload backing store，不参与 prefix cache
        self.cpu_blocks: list[Block] = [Block(i) for i in range(num_cpu_blocks)]
        self.free_cpu_block_ids: deque[int] = deque(range(num_cpu_blocks))
        self.used_cpu_block_ids: set[int] = set()
        self.cpu_offload_watermark_blocks = cpu_offload_watermark_blocks

        self.speculative_decoding = speculative_decoding
        self.num_speculative_tokens = num_speculative_tokens

    @property
    def num_free_blocks(self):
        """总可用 GPU 块数（真正空闲 + 可驱逐的 LRU 缓存块）。"""
        return len(self.free_block_ids) + len(self.cached_blocks)

    @property
    def num_free_cpu_blocks(self):
        """总可用 CPU 块数。"""
        return len(self.free_cpu_block_ids)

    @property
    def total_blocks(self) -> int:
        """总 GPU KV-cache 块数。"""
        return len(self.blocks)

    @property
    def total_cpu_blocks(self) -> int:
        """总 CPU KV-cache 块数。"""
        return len(self.cpu_blocks)

    @property
    def used_blocks(self) -> int:
        """已使用的 GPU KV-cache 块数。"""
        return len(self.used_block_ids)

    @property
    def used_cpu_blocks(self) -> int:
        """已使用的 CPU KV-cache 块数。"""
        return len(self.used_cpu_block_ids)

    @property
    def usage_ratio(self) -> float:
        """GPU KV-cache 使用率 (0.0 ~ 1.0)。"""
        if self.total_blocks == 0:
            return 0.0
        return self.used_blocks / self.total_blocks

    @property
    def cpu_usage_ratio(self) -> float:
        """CPU KV-cache 使用率 (0.0 ~ 1.0)。"""
        if self.total_cpu_blocks == 0:
            return 0.0
        return self.used_cpu_blocks / self.total_cpu_blocks

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算 token 序列的链式 hash（xxhash64），用于 prefix caching 块匹配。"""
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # ── 内部块管理方法 ──

    def _find_gpu_block(self, h: int, token_ids: list[int]) -> int:
        """按 hash + token_ids 精确匹配 GPU 块，避免 hash 冲突。"""
        if h == -1:
            return -1
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1:
            return -1
        if self.blocks[block_id].token_ids != token_ids:
            return -1
        return block_id

    def _get_new_block_id(self) -> int:
        """
        获取一个可用 GPU 块 ID：优先从真正空闲块取，
        不够时按 LRU 顺序驱逐缓存块。返回 -1 表示 OOM。
        """
        if self.free_block_ids:
            block_id = self.free_block_ids.popleft()
        elif self.cached_blocks:
            # LRU 驱逐：移除最久未使用的缓存块
            evicted_hash, block_id = self.cached_blocks.popitem(last=False)
            if self.hash_to_block_id.get(evicted_hash) == block_id:
                del self.hash_to_block_id[evicted_hash]
        else:
            return -1

        block = self.blocks[block_id]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _get_new_cpu_block_id(self) -> int:
        """获取一个可用 CPU 块 ID。返回 -1 表示 CPU offload 空间不足。"""
        if not self.free_cpu_block_ids:
            return -1
        block_id = self.free_cpu_block_ids.popleft()
        block = self.cpu_blocks[block_id]
        block.reset()
        self.used_cpu_block_ids.add(block_id)
        return block_id

    def _revive_cached_block(self, block_id: int):
        """将 LRU 缓存中的 GPU 块恢复为活跃使用状态。"""
        block = self.blocks[block_id]
        if (
            block.hash in self.cached_blocks
            and self.cached_blocks.get(block.hash) == block_id
        ):
            del self.cached_blocks[block.hash]
        block.ref_count = 1
        self.used_block_ids.add(block_id)

    def _release_block(self, block_id: int):
        """释放 GPU 块（ref_count 已为 0）。有 hash 的移入 LRU 缓存，否则归还空闲池。"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.used_block_ids.discard(block_id)
        if block.hash != -1:
            self.cached_blocks[block.hash] = block_id
            self.cached_blocks.move_to_end(block.hash)
        else:
            self.free_block_ids.append(block_id)

    def _release_swapped_block(self, block_id: int):
        """释放已被 swap out 的 GPU 块，直接回空闲池，不进入 LRU 缓存。"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.used_block_ids.discard(block_id)
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.hash = -1
        block.token_ids = []
        self.free_block_ids.append(block_id)

    def _release_cpu_block(self, block_id: int):
        """释放 CPU 块，直接归还空闲池。"""
        block = self.cpu_blocks[block_id]
        assert block.ref_count == 0
        self.used_cpu_block_ids.discard(block_id)
        block.hash = -1
        block.token_ids = []
        self.free_cpu_block_ids.append(block_id)

    # ── Waiting 队列操作 ──

    def get_token_layout(self, seq: Sequence):
        """
        分析 waiting 队列新序列的 prefix cache 命中情况。

        Returns:
            (num_computed_in_used, num_computed_in_cached, num_new_tokens)
            - num_computed_in_used:  命中活跃块的 token 数（ref_count > 0）
            - num_computed_in_cached: 命中 LRU 缓存块的 token 数（需要恢复）
            - num_new_tokens: 需要计算的新 token 数
        """
        assert not seq.block_table
        assert seq.residency != CacheResidency.CPU, (
            "CPU-offloaded sequences should be resumed via swap_in, "
            "not via get_token_layout."
        )
        num_new_tokens = 0
        num_computed_in_used = 0
        num_computed_in_cached = 0
        h = -1
        cache_miss = False

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self._find_gpu_block(h, token_ids)

            if block_id == -1 or i == seq.num_blocks - 1:
                cache_miss = True

            if cache_miss:
                num_new_tokens += len(token_ids)
            else:
                if block_id in self.used_block_ids:
                    num_computed_in_used += len(token_ids)
                else:
                    num_computed_in_cached += len(token_ids)

        return num_computed_in_used, num_computed_in_cached, num_new_tokens

    def can_allocate(self, num_tokens: int) -> bool:
        """检查可用 GPU 块是否足够分配给定数量的新 token。"""
        needed = (num_tokens + self.block_size - 1) // self.block_size
        return self.num_free_blocks >= needed

    def allocate(self, seq: Sequence):
        """
        为 waiting 队列新序列分配 GPU KV-cache 块。
        第一阶段：复用 prefix cache 命中的块。
        第二阶段：为剩余 token 分配新块。
        """
        assert not seq.block_table
        h = -1

        # 阶段1：匹配 GPU prefix cache
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self._find_gpu_block(h, token_ids)

            if block_id == -1 or i == seq.num_blocks - 1:
                break

            seq.num_cached_tokens += self.block_size

            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                self._revive_cached_block(block_id)
                block = self.blocks[block_id]

            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

        # 阶段2：为剩余 token 分配新块
        start = seq.num_cached_tokens
        end = seq.num_cached_tokens + seq.num_new_tokens
        for i in range(start, end, self.block_size):
            token_ids = seq[i : min(i + self.block_size, end)]
            if i == start:
                if len(token_ids) != self.block_size:
                    h = -1
            else:
                h = (
                    self.compute_hash(token_ids, h)
                    if len(token_ids) == self.block_size
                    else -1
                )
            block_id = self._get_new_block_id()
            assert block_id != -1, "OOM: no free blocks available"
            block = self.blocks[block_id]
            block.update(h, token_ids)
            if h != -1:
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

        seq.residency = CacheResidency.GPU

    def deallocate(self, seq: Sequence):
        """释放序列所有 GPU/CPU 块，并清空调度相关状态。"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._release_block(block_id)

        for block_id in reversed(seq.cpu_block_table):
            block = self.cpu_blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._release_cpu_block(block_id)

        seq.num_cached_tokens = 0
        seq.num_new_tokens = 0
        seq.block_table.clear()
        seq.cpu_block_table.clear()
        seq.residency = CacheResidency.NONE

    # ── CPU offload 操作 ──

    def can_swap_out(self, seq: Sequence) -> bool:
        """检查是否有足够 CPU block 可用于 swap out。"""
        return bool(seq.block_table) and len(self.free_cpu_block_ids) >= len(seq.block_table)

    def can_swap_in(self, seq: Sequence, num_new_tokens: int) -> bool:
        """
        检查能否将 CPU-offloaded 序列恢复到 GPU。

        需要同时考虑：
        - 恢复已有 CPU block 需要占用的 GPU 块
        - 本轮为新 token 追加时可能新增的 GPU 块
        - 可选 GPU watermark，减少来回震荡
        """
        if not seq.cpu_block_table:
            return False

        needed = 0
        for cpu_block_id in seq.cpu_block_table:
            cpu_block = self.cpu_blocks[cpu_block_id]
            block_id = self._find_gpu_block(cpu_block.hash, cpu_block.token_ids)
            if block_id == -1:
                needed += 1

        current_capacity = len(seq.cpu_block_table) * self.block_size
        required_capacity = seq.num_cached_tokens + num_new_tokens
        append_needed = max(
            0, (required_capacity - current_capacity + self.block_size - 1) // self.block_size
        )
        needed += append_needed

        return self.num_free_blocks >= needed + self.cpu_offload_watermark_blocks

    def swap_out(self, seq: Sequence) -> dict[int, int]:
        """
        将序列的 GPU block 映射到 CPU block。

        返回:
            {gpu_block_id: cpu_block_id}
        """
        assert self.can_swap_out(seq), "Not enough CPU blocks for swap out"
        mapping: dict[int, int] = {}

        for gpu_block_id in seq.block_table:
            cpu_block_id = self._get_new_cpu_block_id()
            assert cpu_block_id != -1

            gpu_block = self.blocks[gpu_block_id]
            cpu_block = self.cpu_blocks[cpu_block_id]
            cpu_block.update(gpu_block.hash, list(gpu_block.token_ids))

            mapping[gpu_block_id] = cpu_block_id
            seq.cpu_block_table.append(cpu_block_id)

            gpu_block.ref_count -= 1
            if gpu_block.ref_count == 0:
                self._release_swapped_block(gpu_block_id)

        seq.block_table.clear()
        seq.num_new_tokens = 0
        seq.residency = CacheResidency.CPU
        return mapping

    def swap_in(self, seq: Sequence) -> dict[int, int]:
        """
        将序列的 CPU block 恢复为 GPU block。

        返回:
            {cpu_block_id: gpu_block_id}
            仅包含真正需要执行 CPU -> GPU copy 的块。
        """
        assert seq.cpu_block_table, "Sequence has no CPU blocks to swap in"
        mapping: dict[int, int] = {}

        for cpu_block_id in seq.cpu_block_table:
            cpu_block = self.cpu_blocks[cpu_block_id]
            block_id = self._find_gpu_block(cpu_block.hash, cpu_block.token_ids)

            if block_id != -1 and block_id in self.used_block_ids:
                gpu_block = self.blocks[block_id]
                gpu_block.ref_count += 1
            elif block_id != -1:
                self._revive_cached_block(block_id)
                gpu_block = self.blocks[block_id]
            else:
                block_id = self._get_new_block_id()
                assert block_id != -1, "No free GPU blocks available during swap in"
                gpu_block = self.blocks[block_id]
                gpu_block.update(cpu_block.hash, list(cpu_block.token_ids))
                if cpu_block.hash != -1:
                    self.hash_to_block_id[cpu_block.hash] = block_id
                mapping[cpu_block_id] = block_id

            seq.block_table.append(gpu_block.block_id)

            cpu_block.ref_count -= 1
            if cpu_block.ref_count == 0:
                self._release_cpu_block(cpu_block_id)

        seq.cpu_block_table.clear()
        seq.residency = CacheResidency.GPU
        return mapping

    # ── Running 队列操作 ──

    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool:
        """
        检查能否为 running 队列序列追加 num_new_tokens 个 token 的块。
        支持 decode（1 个 token）和 chunked prefill（多个 token）。
        """
        if not seq.block_table:
            return False

        current_capacity = len(seq.block_table) * self.block_size
        required_capacity = seq.num_cached_tokens + num_new_tokens
        needed = max(
            0, (required_capacity - current_capacity + self.block_size - 1) // self.block_size
        )

        if self.speculative_decoding:
            target_tokens = (
                seq.num_cached_tokens + num_new_tokens + self.num_speculative_tokens
            )
            target_blocks = (target_tokens + self.block_size - 1) // self.block_size
            needed = max(needed, max(0, target_blocks - len(seq.block_table)))
        return self.num_free_blocks >= needed

    def may_append(self, seq: Sequence):
        """
        为 running 队列序列的新 token 分配/更新块。
        同时处理 decode（1 个新 token）和 chunked prefill（多个新 token）。
        满块时计算 hash 以支持后续 GPU prefix caching。
        """
        start = seq.num_cached_blocks * self.block_size
        end = seq.num_cached_tokens + seq.num_new_tokens

        for i in range(start, end, self.block_size):
            token_ids = seq[i : min(i + self.block_size, end)]
            block_idx = i // self.block_size
            current_block_id = (
                seq.block_table[block_idx] if block_idx < len(seq.block_table) else -1
            )

            if current_block_id != -1:
                current_block = self.blocks[current_block_id]
                assert current_block.hash == -1

            if len(token_ids) % self.block_size == 0:
                prev_block_id = seq.block_table[block_idx - 1] if block_idx > 0 else -1
                prefix = self.blocks[prev_block_id].hash if prev_block_id != -1 else -1
                h = self.compute_hash(token_ids, prefix)

                if current_block_id == -1:
                    block_id = self._get_new_block_id()
                    assert block_id != -1
                    current_block = self.blocks[block_id]
                    seq.block_table.append(block_id)

                current_block.update(h, token_ids)
                self.hash_to_block_id[h] = current_block.block_id
            elif current_block_id == -1:
                block_id = self._get_new_block_id()
                assert block_id != -1
                current_block = self.blocks[block_id]
                current_block.update(-1, token_ids)
                seq.block_table.append(block_id)
            else:
                current_block.update(-1, token_ids)

        if self.speculative_decoding:
            target_tokens = (
                seq.num_cached_tokens + seq.num_new_tokens + self.num_speculative_tokens
            )
            target_blocks = (target_tokens + self.block_size - 1) // self.block_size
            while len(seq.block_table) < target_blocks:
                block_id = self._get_new_block_id()
                assert block_id != -1
                seq.block_table.append(block_id)

        seq.residency = CacheResidency.GPU
