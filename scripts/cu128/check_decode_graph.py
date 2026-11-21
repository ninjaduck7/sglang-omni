"""Capture one paged decode; change Q, KV writes and page lengths across replays."""

import os
import sys

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", sys.prefix)
os.environ.setdefault("MAX_JOBS", "2")

import flashinfer
import torch


@torch.inference_mode()
def main():
    torch.manual_seed(5090)
    f16 = {"device": "cuda", "dtype": torch.float16}
    i32 = {"device": "cuda", "dtype": torch.int32}
    # Two non-contiguous physical pages, plus one unused guard page.
    pages = [2, 0]
    cache = torch.zeros(3, 2, 16, 2, 64, **f16)
    expected_cache = cache.clone()
    q = torch.randn(1, 2, 64, **f16)
    kv = torch.randn(1, 2, 2, 64, **f16)
    page = torch.full((1,), pages[0], device="cuda", dtype=torch.long)
    slot = torch.zeros_like(page)
    out = torch.empty_like(q)
    wrapper = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(32 << 20, dtype=torch.uint8, device="cuda"),
        torch.empty(2, **i32), torch.empty(2, **i32), torch.empty(1, **i32),
    )

    def plan(n):
        count = (n + 15) // 16
        wrapper.plan(*(torch.tensor(v, dtype=torch.int32) for v in (
            [0, count], pages[:count], [(n - 1) % 16 + 1]
        )), 2, 2, 64, 16, non_blocking=False)

    # The write and attention are both captured, but the wrapper plan is not.
    def decode():
        cache[page, :, slot] = kv
        wrapper.run(q, cache, out=out)

    plan(1)
    decode()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode()
    cache.zero_()
    keys, values = [], []
    for t in range(20):
        plan(t + 1)
        q.normal_()
        kv.normal_()
        page.fill_(pages[t // 16])
        slot.fill_(t % 16)
        graph.replay()
        keys.append(kv[0, 0].clone())
        values.append(kv[0, 1].clone())
        k = torch.stack(keys).transpose(0, 1).float()
        v = torch.stack(values).transpose(0, 1).float()
        scores = q.transpose(0, 1).float() @ k.transpose(1, 2) / 8
        ref = (scores.softmax(-1) @ v).transpose(0, 1)
        torch.testing.assert_close(out, ref.half(), atol=0.003, rtol=0.003)
        expected_cache[pages[t // 16], :, t % 16] = kv[0]
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    print("PASS: 20 replays, page boundary + full KV guard, FP32 reference")


if __name__ == "__main__":
    main()
