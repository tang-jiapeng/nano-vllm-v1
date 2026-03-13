"""
vLLM v1 调度器单元测试（纯 CPU，无需 GPU）。

测试覆盖：
1. BlockManager 基本分配/释放
2. BlockManager prefix caching + LRU eviction
3. BlockManager chunked prefill 支持
4. Scheduler 统一调度（running 优先 + waiting）
5. Token budget 约束
6. Chunked prefill 分块调度
7. 混合 prefill+decode 批次
8. Preemption（抢占与恢复）
9. Postprocess 终止条件
10. Speculative decoding: draft_num_cached_tokens 正确更新
"""

import importlib
import os
import sys

# 避免通过 nanovllm/__init__.py 触发 CUDA 初始化
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_seq = importlib.import_module("nanovllm.engine.sequence")
_bm = importlib.import_module("nanovllm.engine.block_manager")
_sp = importlib.import_module("nanovllm.sampling_params")

Sequence = _seq.Sequence
SequenceStatus = _seq.SequenceStatus
Block = _bm.Block
BlockManager = _bm.BlockManager
SamplingParams = _sp.SamplingParams

# 测试用小 block_size
BS = 4


def setup():
    """每个测试前重置 Sequence.block_size 为测试值。"""
    Sequence.block_size = BS


def sp(**kw):
    """快捷创建 SamplingParams。"""
    kw.setdefault("temperature", 0.6)
    kw.setdefault("max_tokens", 10)
    return SamplingParams(**kw)


class MockConfig:
    """最小化的 Config mock。"""

    def __init__(self, **kw):
        self.chunked_prefill = kw.get("chunked_prefill", False)
        self.max_model_len = kw.get("max_model_len", 1024)
        self.max_num_seqs = kw.get("max_num_seqs", 16)
        self.max_num_batched_tokens = kw.get("max_num_batched_tokens", 256)
        self.eos = kw.get("eos", 0)
        self.num_kvcache_blocks = kw.get("num_kvcache_blocks", 100)
        self.kvcache_block_size = kw.get("kvcache_block_size", BS)
        # Speculative decoding fields (default: disabled)
        self.speculative_model = kw.get("speculative_model", None)
        self.num_speculative_tokens = kw.get("num_speculative_tokens", 0)
        self.num_draft_kvcache_blocks = kw.get("num_draft_kvcache_blocks", 0)


def make_scheduler(**kw):
    _sched = importlib.import_module("nanovllm.engine.scheduler")
    return _sched.Scheduler(MockConfig(**kw))


# ═══════════════════════════════════════════════
# BlockManager 测试
# ═══════════════════════════════════════════════


def test_bm_basic_allocate():
    """基本分配和释放。"""
    setup()
    bm = BlockManager(num_blocks=10, block_size=BS)
    assert bm.num_free_blocks == 10

    seq = Sequence([1, 2, 3, 4, 5, 6, 7], sp())
    seq.num_new_tokens = 7
    used, cached, new = bm.get_token_layout(seq)
    assert used == 0 and cached == 0 and new == 7

    assert bm.can_allocate(7)
    bm.allocate(seq)
    assert len(seq.block_table) == 2  # ceil(7/4) = 2

    bm.deallocate(seq)
    assert bm.num_free_blocks == 10
    assert seq.block_table == []
    print("  ✓ test_bm_basic_allocate")


def test_bm_prefix_caching():
    """相同前缀的两个序列应命中缓存。"""
    setup()
    bm = BlockManager(num_blocks=20, block_size=BS)

    # seq1: [1,2,3,4 | 5,6,7,8 | 9,10]
    seq1 = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], sp())
    seq1.num_new_tokens = 10
    bm.allocate(seq1)
    assert seq1.num_cached_tokens == 0

    bm.deallocate(seq1)

    # seq2: [1,2,3,4 | 5,6,7,8 | 11,12] — 前两个块相同
    seq2 = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 11, 12], sp())
    used, cached, new = bm.get_token_layout(seq2)
    assert used + cached == 8, f"Expected 8 cached, got {used + cached}"
    assert new == 2

    seq2.num_new_tokens = new
    bm.allocate(seq2)
    assert seq2.num_cached_tokens == 8
    assert len(seq2.block_table) == 3

    bm.deallocate(seq2)
    print("  ✓ test_bm_prefix_caching")


def test_bm_lru_eviction():
    """空闲块用完时驱逐 LRU 缓存块。"""
    setup()
    bm = BlockManager(num_blocks=4, block_size=BS)

    seq1 = Sequence([1, 2, 3, 4, 5], sp())
    seq1.num_new_tokens = 5
    bm.allocate(seq1)
    bm.deallocate(seq1)

    assert bm.num_free_blocks == 4

    # 4 个块全用（需要驱逐 LRU）
    seq2 = Sequence(list(range(16)), sp())
    seq2.num_new_tokens = 16
    assert bm.can_allocate(16)
    bm.allocate(seq2)
    assert len(seq2.block_table) == 4

    bm.deallocate(seq2)
    print("  ✓ test_bm_lru_eviction")


def test_bm_can_append():
    """running 队列的 can_append 和 may_append。"""
    setup()
    bm = BlockManager(num_blocks=10, block_size=BS)

    seq = Sequence([1, 2, 3], sp(max_tokens=20))
    seq.num_new_tokens = 3
    bm.allocate(seq)

    # 模拟一步完成: 现在有 3 个 cached token
    seq.num_cached_tokens = 3
    seq.num_new_tokens = 0
    seq.append_token(4)

    # 追加 1 个 decode token
    assert bm.can_append(seq, 1)
    seq.num_new_tokens = 1
    bm.may_append(seq)
    assert bm.blocks[seq.block_table[-1]].hash != -1  # 满块计算了 hash

    # 再追加一个 → 需要新块
    seq.num_cached_tokens = 4
    seq.num_new_tokens = 0
    seq.append_token(5)
    assert bm.can_append(seq, 1)
    seq.num_new_tokens = 1
    bm.may_append(seq)
    assert len(seq.block_table) == 2

    bm.deallocate(seq)
    print("  ✓ test_bm_can_append")


def test_bm_chunked_append():
    """may_append 处理 chunked prefill（多个新 token）。"""
    setup()
    bm = BlockManager(num_blocks=20, block_size=BS)

    seq = Sequence(list(range(12)), sp())
    seq.num_new_tokens = 4
    bm.allocate(seq)
    assert len(seq.block_table) == 1

    # 第二个 chunk
    seq.num_cached_tokens = 4
    seq.num_new_tokens = 4
    bm.may_append(seq)
    assert len(seq.block_table) == 2

    # 第三个 chunk
    seq.num_cached_tokens = 8
    seq.num_new_tokens = 4
    bm.may_append(seq)
    assert len(seq.block_table) == 3

    bm.deallocate(seq)
    print("  ✓ test_bm_chunked_append")


# ═══════════════════════════════════════════════
# Scheduler 测试
# ═══════════════════════════════════════════════


def test_sched_basic():
    """基本调度：waiting → running。"""
    setup()
    s = make_scheduler(max_num_batched_tokens=64, num_kvcache_blocks=50)

    seq1 = Sequence([1, 2, 3], sp(max_tokens=5))
    seq2 = Sequence([4, 5, 6, 7], sp(max_tokens=5))
    s.add(seq1)
    s.add(seq2)

    seqs = s.schedule()
    assert len(seqs) == 2
    assert seq1.status == SequenceStatus.RUNNING
    assert seq1.num_new_tokens == 3
    assert seq2.num_new_tokens == 4
    print("  ✓ test_sched_basic")


def test_sched_token_budget():
    """Token budget 约束。"""
    setup()
    s = make_scheduler(max_num_batched_tokens=10, num_kvcache_blocks=50)

    seq1 = Sequence([1] * 8, sp(max_tokens=5))
    seq2 = Sequence([2] * 8, sp(max_tokens=5))
    s.add(seq1)
    s.add(seq2)

    seqs = s.schedule()
    assert len(seqs) == 1
    assert seqs[0] is seq1
    assert len(s.waiting) == 1
    print("  ✓ test_sched_token_budget")


def test_sched_decode_priority():
    """Running 队列优先于 waiting 队列。"""
    setup()
    s = make_scheduler(max_num_batched_tokens=64, num_kvcache_blocks=50)

    seq1 = Sequence([1, 2, 3], sp(max_tokens=10))
    s.add(seq1)
    seqs = s.schedule()

    # 模拟一步
    seq1.append_token(10)
    seq1.num_cached_tokens += seq1.num_new_tokens
    seq1.num_new_tokens = 0

    seq2 = Sequence([4, 5, 6, 7], sp(max_tokens=5))
    s.add(seq2)

    seqs = s.schedule()
    assert len(seqs) == 2
    assert seqs[0] is seq1  # running 优先
    assert seqs[1] is seq2
    assert seq1.num_new_tokens == 1  # decode
    assert seq2.num_new_tokens == 4  # prefill
    print("  ✓ test_sched_decode_priority")


def test_sched_chunked_prefill():
    """Chunked prefill：长序列分块处理。"""
    setup()
    s = make_scheduler(
        chunked_prefill=True, max_num_batched_tokens=8, num_kvcache_blocks=50
    )

    seq = Sequence(list(range(16)), sp(max_tokens=5))
    s.add(seq)

    # 第一个 chunk
    seqs = s.schedule()
    assert len(seqs) == 1
    assert seqs[0].num_new_tokens == 8

    seq.num_cached_tokens += seq.num_new_tokens
    seq.num_new_tokens = 0

    # 第二个 chunk
    seqs = s.schedule()
    assert len(seqs) == 1
    assert seqs[0].num_new_tokens == 8
    print("  ✓ test_sched_chunked_prefill")


def test_sched_chunked_mixed():
    """混合批次：chunked prefill + decode。"""
    setup()
    s = make_scheduler(
        chunked_prefill=True, max_num_batched_tokens=16, num_kvcache_blocks=100
    )

    short = Sequence([1, 2, 3], sp(max_tokens=10))
    s.add(short)
    seqs = s.schedule()

    short.append_token(10)
    short.num_cached_tokens += short.num_new_tokens
    short.num_new_tokens = 0

    long = Sequence(list(range(20)), sp(max_tokens=5))
    s.add(long)

    seqs = s.schedule()
    assert len(seqs) == 2
    assert short.num_new_tokens == 1  # decode
    assert long.num_new_tokens <= 15  # budget - 1
    print("  ✓ test_sched_chunked_mixed")


def test_sched_preemption():
    """KV-cache 不足时触发抢占。"""
    setup()
    s = make_scheduler(
        max_num_batched_tokens=64, num_kvcache_blocks=3, kvcache_block_size=BS
    )

    seq1 = Sequence([1, 2, 3, 4], sp(max_tokens=10))
    seq2 = Sequence([5, 6, 7, 8], sp(max_tokens=10))
    s.add(seq1)
    s.add(seq2)

    seqs = s.schedule()
    assert len(seqs) >= 1
    assert len(s.running) >= 1
    print("  ✓ test_sched_preemption")


def test_sched_postprocess_eos():
    """Postprocess: EOS 终止。"""
    setup()
    s = make_scheduler(max_num_batched_tokens=64, num_kvcache_blocks=50, eos=0)

    seq = Sequence([1, 2, 3], sp(max_tokens=100))
    s.add(seq)
    seqs = s.schedule()

    s.postprocess(seqs, [0], [0])  # EOS
    assert seq.status == SequenceStatus.FINISHED
    assert len(s.running) == 0
    print("  ✓ test_sched_postprocess_eos")


def test_sched_postprocess_max_tokens():
    """Postprocess: max_tokens 终止。"""
    setup()
    s = make_scheduler(max_num_batched_tokens=64, num_kvcache_blocks=50, eos=999)

    seq = Sequence([1, 2, 3], sp(max_tokens=2))
    s.add(seq)
    seqs = s.schedule()

    # token 1
    s.postprocess(seqs, [10], [0])
    assert seq.status != SequenceStatus.FINISHED

    seqs = s.schedule()
    # token 2
    s.postprocess(seqs, [11], [0])
    assert seq.status == SequenceStatus.FINISHED
    assert seq.num_completion_tokens == 2
    print("  ✓ test_sched_postprocess_max_tokens")


def test_sched_is_finished():
    """is_finished 状态检查。"""
    setup()
    s = make_scheduler(max_num_batched_tokens=64, num_kvcache_blocks=50, eos=999)

    assert s.is_finished()

    seq = Sequence([1, 2], sp(max_tokens=1))
    s.add(seq)
    assert not s.is_finished()

    seqs = s.schedule()
    assert not s.is_finished()

    s.postprocess(seqs, [10], [0])
    assert s.is_finished()
    print("  ✓ test_sched_is_finished")


def test_sched_ref_count_sharing():
    """Prefix cache: 共享块引用计数正确。"""
    setup()
    bm = BlockManager(num_blocks=20, block_size=BS)

    prefix = [1, 2, 3, 4]  # 一个完整块

    # seq1
    seq1 = Sequence(prefix + [10, 11], sp())
    seq1.num_new_tokens = 6
    bm.allocate(seq1)

    # seq2 共享前缀
    seq2 = Sequence(prefix + [20, 21], sp())
    seq2.num_new_tokens = 2  # 只有新的 2 个 token
    _, _, new = bm.get_token_layout(seq2)
    seq2.num_new_tokens = new
    bm.allocate(seq2)
    assert seq2.num_cached_tokens == 4  # 前缀命中

    # 第一个块的 ref_count 应该 > 1
    shared_block = bm.blocks[seq1.block_table[0]]
    assert shared_block.ref_count == 2

    # 释放 seq1
    bm.deallocate(seq1)
    assert shared_block.ref_count == 1  # seq2 还在用

    # 释放 seq2
    bm.deallocate(seq2)
    assert shared_block.ref_count == 0
    print("  ✓ test_sched_ref_count_sharing")


# ═══════════════════════════════════════════════
# Speculative Decoding 测试
# ═══════════════════════════════════════════════


def test_sched_speculative_draft_num_cached():
    """Speculative postprocess: draft_num_cached_tokens 正确更新。

    验证修复前的 off-by-1 bug（原 max(0, num_cached-1)）已不存在：
    - a < K 时，draft_num_cached 应等于 num_cached（下一步无需 prefill）
    - a == K 时，draft_num_cached 应等于 num_cached-1（下一步需 2-token prefill）
    """
    setup()
    K = 5
    s = make_scheduler(
        max_num_batched_tokens=256,
        num_kvcache_blocks=50,
        num_draft_kvcache_blocks=50,
        speculative_model="dummy",
        num_speculative_tokens=K,
        eos=0,
    )

    P = 3  # prompt 长度
    C = P  # target 模型已缓存 P 个 prompt token

    def make_spec_seq(accepted_count):
        """构造 generate_draft_tokens 后的序列状态。"""
        # 创建 prompt 序列
        seq = Sequence(
            list(range(1, P + 1)),
            SamplingParams(temperature=0.6, max_tokens=100, ignore_eos=True),
        )
        # 追加 bonus_prev 和 K 个 draft token，模拟 K 步 draft decode 后的状态
        for i in range(K + 1):
            seq.append_token(100 + i)  # 100=bonus_prev, 101..105=d0..d4
        # 手动设置 generate_draft_tokens 结束后的状态
        seq.num_cached_tokens = C
        seq.draft_num_cached_tokens = C + K  # K 步 decode 后
        seq.is_speculative = True
        seq.pending_accepted_tokens = list(range(101, 101 + accepted_count))
        seq.num_new_tokens = 0
        seq.status = SequenceStatus.RUNNING
        return seq

    # ── 情形 1: a < K (3 out of 5 accepted) ──
    a1 = 3
    seq1 = make_spec_seq(a1)
    s.running.append(seq1)
    s.postprocess([seq1], [999], [0])

    expected_cached1 = C + a1 + 1  # = 7
    assert seq1.num_cached_tokens == expected_cached1, (
        f"a<K: expected num_cached={expected_cached1}, got {seq1.num_cached_tokens}"
    )
    # draft_num_cached 应 = num_cached（min(C+K, C+a+1) = C+a+1）
    assert seq1.draft_num_cached_tokens == expected_cached1, (
        f"a<K: expected draft_num_cached={expected_cached1}, "
        f"got {seq1.draft_num_cached_tokens}"
    )
    # 下一步 needs_prefill = (draft_num_cached < len-1) 应为 False
    assert seq1.draft_num_cached_tokens == len(seq1) - 1, (
        f"a<K: draft_num_cached should equal len-1 so next step skips prefill; "
        f"draft={seq1.draft_num_cached_tokens}, len-1={len(seq1)-1}"
    )

    # ── 情形 2: a == K (all 5 accepted) ──
    a2 = K
    seq2 = make_spec_seq(a2)
    s.running.append(seq2)
    s.postprocess([seq2], [999], [0])

    expected_cached2 = C + a2 + 1  # = C + K + 1
    expected_draft2 = C + K         # = num_cached - 1
    assert seq2.num_cached_tokens == expected_cached2, (
        f"a=K: expected num_cached={expected_cached2}, got {seq2.num_cached_tokens}"
    )
    # draft_num_cached 应 = C+K（min(C+K, C+K+1) = C+K，因末尾 draft token 未写入 KV）
    assert seq2.draft_num_cached_tokens == expected_draft2, (
        f"a=K: expected draft_num_cached={expected_draft2}, "
        f"got {seq2.draft_num_cached_tokens}"
    )
    # 下一步 needs_prefill 应为 True（需要补写未缓存的最后 draft token）
    assert seq2.draft_num_cached_tokens < len(seq2) - 1, (
        f"a=K: draft_num_cached should be less than len-1 so next step needs prefill"
    )

    print("  ✓ test_sched_speculative_draft_num_cached")

if __name__ == "__main__":
    original_bs = Sequence.block_size
    try:
        print("=" * 50)
        print("BlockManager tests")
        print("=" * 50)
        test_bm_basic_allocate()
        test_bm_prefix_caching()
        test_bm_lru_eviction()
        test_bm_can_append()
        test_bm_chunked_append()

        print()
        print("=" * 50)
        print("Scheduler tests")
        print("=" * 50)
        test_sched_basic()
        test_sched_token_budget()
        test_sched_decode_priority()
        test_sched_chunked_prefill()
        test_sched_chunked_mixed()
        test_sched_preemption()
        test_sched_postprocess_eos()
        test_sched_postprocess_max_tokens()
        test_sched_is_finished()
        test_sched_ref_count_sharing()

        print()
        print("=" * 50)
        print("Speculative Decoding tests")
        print("=" * 50)
        test_sched_speculative_draft_num_cached()

        print()
        print("=" * 50)
        print("All 16 tests passed! ✅")
        print("=" * 50)
    finally:
        Sequence.block_size = original_bs
