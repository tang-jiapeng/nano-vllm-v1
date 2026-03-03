from collections import deque

import numpy as np
import xxhash

from nanovllm.engine.sequence import Sequence


class Block:
    """KV-cache 块，通过引用计数管理生命周期，通过 hash 支持 prefix caching。"""

    def __init__(self, block_id):
        """初始化为空闲状态 (ref_count=0, hash=-1)。"""
        self.block_id = block_id
        self.ref_count = 0  # 引用计数，0表示空闲
        self.hash = -1  # 块的哈希值，-1表示无效
        self.token_ids = []  # 块内的token列表

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
    """管理所有 KV-cache 块的分配/释放，维护空闲队列、引用计数和 prefix caching 哈希表。"""

    def __init__(
        self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True
    ):
        """初始化块管理器，创建所有 block 实例并初始化空闲队列。"""
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        # 创建所有块
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # 哈希到block ID的映射（用于prefix caching）
        self.hash_to_block_id: dict[int, int] = dict()

        # 空闲块ID的双端队列（高效pop from left）
        self.free_block_ids: deque[int] = deque(range(num_blocks))

        # 已使用块的集合
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算 token 序列的链式 hash（xxhash），用于 prefix caching 块匹配。"""
        h = xxhash.xxh64()

        # 如果有前缀，先添加前缀哈希
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))

        # 添加token数据
        h.update(np.array(token_ids).tobytes())

        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """分配指定 ID 的块：重置状态，从空闲队列移除，加入已使用集合。"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """释放指定 ID 的块：从已使用集合移除，回收到空闲队列。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """检查空闲块是否足够分配给该序列。"""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """为序列分配 KV-cache 块，优先复用 prefix caching 命中的块。"""
        assert not seq.block_table

        h = -1
        cache_miss = not self.enable_prefix_caching  # 未启用时视为全部 miss

        # 遍历序列的每个block
        for i in range(seq.num_blocks):
            # 获取该block的token
            token_ids = seq.block(i)

            # 计算哈希（仅对满的block计算，且启用 prefix caching 时才有意义）
            h = (
                self.compute_hash(token_ids, h)
                if self.enable_prefix_caching and len(token_ids) == self.block_size
                else -1
            )

            # 查找缓存的块
            block_id = self.hash_to_block_id.get(h, -1)

            # 检查是否真正命中缓存（哈希冲突检查）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # 缓存未命中

            if cache_miss:
                # 缓存未命中：从空闲池分配新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中：可能共享已有块
                seq.num_cached_tokens += self.block_size  # 增加缓存token计数

                if block_id in self.used_block_ids:
                    # 块正在使用中，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块是空闲的，分配并初始化
                    block = self._allocate_block(block_id)

            # 更新块的哈希和索引
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            # 记录到序列的block table
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有块，递减引用计数，归零时回收到空闲池。"""
        # 逆序遍历block table
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1  # 减少引用计数

            # 引用计数为0时可以释放
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        # 清空序列的缓存信息
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查是否可以向序列追加新 token（当前 block 未满或有空闲块）。"""
        # 需要新块的条件：当前长度 % block_size == 1
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """为序列追加 token 做准备：block 刚满时分配新块，填满时计算 hash。"""
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        # 情况1：当前block刚满，需要新块
        # 例如：block_size=256，当前长度=256*k + 1
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1

            # 分配新块
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        # 情况2：当前block刚填满，可以计算哈希
        # 例如：block_size=256，当前长度=256*k
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1

            # 获取刚填满的block的token
            token_ids = seq.block(seq.num_blocks - 1)

            # 计算前缀哈希
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1

            # 计算当前block的哈希
            h = self.compute_hash(token_ids, prefix)

            # 更新块的哈希和索引
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id

        # 情况3：当前block未满，无需操作
        else:
            assert last_block.hash == -1
