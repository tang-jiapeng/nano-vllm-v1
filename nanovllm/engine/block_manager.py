"""
KV-cache 块管理器，支持 prefix caching、LRU eviction 和 chunked prefill。

块生命周期：
    free_block_ids → used_block_ids → cached_blocks (LRU) → 被驱逐 → free_block_ids

三个块池：
    free_block_ids:  真正空闲的块（无有效数据）
    used_block_ids:  活跃使用的块（ref_count > 0）
    cached_blocks:   LRU 缓存的最近释放块（ref_count=0，有效 hash）

块内存布局示意：
    ──────────────────────────────────────────────────────────
    | < computed (cached) > | < new tokens to compute >      |
    ──────────────────────────────────────────────────────────
    | < prefix-cached >     | < to be computed & allocated > |
    ──────────────────────────────────────────────────────────
"""

from collections import OrderedDict, deque

import numpy as np
import xxhash

from nanovllm.engine.sequence import Sequence


class Block:
    """KV-cache 块，通过引用计数管理生命周期，通过 hash 支持 prefix caching。"""

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """更新块的 hash 和 token 数据，用于建立 prefix caching 索引。"""
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
    - 基于 xxhash 的 prefix caching（O(1) 查找复用）
    - LRU eviction（空闲块用完时驱逐最久未使用的缓存块）
    - chunked prefill（一次分配多个新块）
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        # LRU 缓存：hash → block_id，最近使用的在末尾
        self.cached_blocks: OrderedDict[int, int] = OrderedDict()

    @property
    def num_free_blocks(self):
        """总可用块数（真正空闲 + 可驱逐的 LRU 缓存块）。"""
        return len(self.free_block_ids) + len(self.cached_blocks)

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算 token 序列的链式 hash（xxhash64），用于 prefix caching 块匹配。"""
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # ── 内部块管理方法 ──

    def _get_new_block_id(self) -> int:
        """
        获取一个可用块 ID：优先从真正空闲块取，
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
            return -1  # OOM

        block = self.blocks[block_id]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _revive_cached_block(self, block_id: int):
        """将 LRU 缓存中的块恢复为活跃使用状态。"""
        block = self.blocks[block_id]
        if (
            block.hash in self.cached_blocks
            and self.cached_blocks.get(block.hash) == block_id
        ):
            del self.cached_blocks[block.hash]
        block.ref_count = 1
        self.used_block_ids.add(block_id)

    def _release_block(self, block_id: int):
        """释放块（ref_count 已为 0）。有 hash 的移入 LRU 缓存，否则归还空闲池。"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.used_block_ids.discard(block_id)
        if block.hash != -1:
            # 有效 hash：保留在 LRU 缓存中以备复用
            self.cached_blocks[block.hash] = block_id
            self.cached_blocks.move_to_end(block.hash)
        else:
            # 无 hash（不完整块）：直接归还空闲池
            self.free_block_ids.append(block_id)

    # ── Waiting 队列操作 ──

    def get_token_layout(self, seq: Sequence):
        """
        分析 waiting 队列序列的 prefix cache 命中情况。

        Returns:
            (num_computed_in_used, num_computed_in_cached, num_new_tokens)
            - num_computed_in_used:  命中活跃块的 token 数（ref_count > 0）
            - num_computed_in_cached: 命中 LRU 缓存块的 token 数（需要恢复）
            - num_new_tokens: 需要计算的新 token 数
        """
        assert not seq.block_table
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
            block_id = self.hash_to_block_id.get(h, -1)

            if (
                block_id == -1
                or self.blocks[block_id].token_ids != token_ids
                or i == seq.num_blocks - 1
            ):
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
        """检查可用块是否足够分配给定数量的新 token（waiting 队列用）。"""
        return (
            self.num_free_blocks
            >= (num_tokens + self.block_size - 1) // self.block_size
        )

    def allocate(self, seq: Sequence):
        """
        为 waiting 队列序列分配 KV-cache 块。
        第一阶段：复用 prefix cache 命中的块。
        第二阶段：为剩余 token 分配新块。
        """
        assert not seq.block_table
        h = -1

        # 阶段1：匹配 prefix cache
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self.hash_to_block_id.get(h, -1)

            if (
                block_id == -1
                or self.blocks[block_id].token_ids != token_ids
                or i == seq.num_blocks - 1
            ):
                break  # Cache miss

            seq.num_cached_tokens += self.block_size

            if block_id in self.used_block_ids:
                # 活跃块：增加引用计数
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                # LRU 缓存块：恢复为活跃使用
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
                # h 继承自阶段1，仅当本块完整时才有效；
                # chunked prefill 可能只部分填充该块，须置 h = -1
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
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """释放序列所有块。块根据 hash 状态移入 LRU 缓存或空闲池。"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._release_block(block_id)
        seq.num_cached_tokens = 0
        seq.num_new_tokens = 0
        seq.block_table.clear()

    # ── Running 队列操作 ──

    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool:
        """
        检查能否为 running 队列序列追加 num_new_tokens 个 token 的块。
        支持 decode（1 个 token）和 chunked prefill（多个 token）。
        """
        last_block_capacity = self.block_size - (
            seq.num_cached_tokens % self.block_size
        )
        if last_block_capacity == self.block_size:
            last_block_capacity = 0
        needed = max(
            0,
            (num_new_tokens - last_block_capacity + self.block_size - 1)
            // self.block_size,
        )
        return self.num_free_blocks >= needed

    def may_append(self, seq: Sequence):
        """
        为 running 队列序列的新 token 分配/更新块。
        同时处理 decode（1 个新 token）和 chunked prefill（多个新 token）。
        满块时计算 hash 以支持后续 prefix caching。
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
                assert current_block.hash == -1  # 不完整块不应有 hash

            if len(token_ids) % self.block_size == 0:
                # 块已满：计算 hash 用于 prefix caching
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
                # 新的不完整块
                block_id = self._get_new_block_id()
                assert block_id != -1
                seq.block_table.append(block_id)
