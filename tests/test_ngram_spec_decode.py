"""
N-gram speculative decoding unit tests.

These tests stay on CPU and focus on:
1. N-gram proposer correctness
2. Speculative scheduler postprocess state transitions
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_seq = importlib.import_module("nanovllm.engine.sequence")
_sched = importlib.import_module("nanovllm.engine.scheduler")
_sp = importlib.import_module("nanovllm.sampling_params")
_ngram = importlib.import_module("nanovllm.engine.ngram_proposer")

Sequence = _seq.Sequence
SequenceStatus = _seq.SequenceStatus
Scheduler = _sched.Scheduler
SamplingParams = _sp.SamplingParams
NgramProposer = _ngram.NgramProposer

BS = 4


def setup():
    Sequence.block_size = BS


def sp(**kw):
    kw.setdefault("temperature", 0.6)
    kw.setdefault("max_tokens", 10)
    return SamplingParams(**kw)


class MockConfig:
    def __init__(self, **kw):
        self.chunked_prefill = kw.get("chunked_prefill", False)
        self.speculative_method = kw.get("speculative_method", None)
        self.num_speculative_tokens = kw.get("num_speculative_tokens", 0)
        self.max_model_len = kw.get("max_model_len", 1024)
        self.max_num_seqs = kw.get("max_num_seqs", 16)
        self.max_num_batched_tokens = kw.get("max_num_batched_tokens", 256)
        self.eos = kw.get("eos", 0)
        self.num_kvcache_blocks = kw.get("num_kvcache_blocks", 100)
        self.enable_kv_offload = kw.get("enable_kv_offload", False)
        self.num_cpu_kvcache_blocks = kw.get("num_cpu_kvcache_blocks", 0)
        self.cpu_offload_watermark_blocks = kw.get(
            "cpu_offload_watermark_blocks", 0
        )
        self.kvcache_block_size = kw.get("kvcache_block_size", BS)


def make_scheduler(**kw):
    return Scheduler(MockConfig(**kw))


def test_ngram_proposer_prefers_longest_match():
    proposer = NgramProposer(min_ngram=2, max_ngram=4)
    token_ids = [1, 2, 3, 4, 2, 3, 4]
    # suffix [2,3,4] matches earlier occurrence starting at 1, so proposal starts at 4.
    assert proposer.propose(token_ids, 3) == [2, 3, 4]


def test_ngram_proposer_returns_empty_when_no_match():
    proposer = NgramProposer(min_ngram=2, max_ngram=4)
    token_ids = [1, 2, 3, 5, 6, 7]
    assert proposer.propose(token_ids, 4) == []


def test_scheduler_postprocess_speculative_partial_accept():
    setup()
    scheduler = make_scheduler(speculative_method="ngram", num_speculative_tokens=4)

    seq = Sequence([10, 11], sp(max_tokens=8))
    seq.status = SequenceStatus.RUNNING
    seq.num_cached_tokens = 2
    seq.num_new_tokens = 1
    seq.is_speculative = True
    seq.speculative_draft_tokens = [20, 21, 22]
    seq.pending_accepted_tokens = [20]
    seq.append_token(20)
    seq.append_token(21)
    seq.append_token(22)

    scheduler.running.append(seq)
    scheduler.postprocess([seq], [30], [0])

    assert seq.token_ids == [10, 11, 20, 30]
    assert seq.num_cached_tokens == 4
    assert seq.num_new_tokens == 0
    assert seq.is_speculative is False
    assert seq.pending_accepted_tokens == []
    assert seq.speculative_draft_tokens == []
    assert seq.last_token == 30
    assert seq.status == SequenceStatus.RUNNING
