"""Offline RTX 5090 environment check. No pretrained configuration or weights."""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", sys.prefix)
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(sys.prefix) / ".cache/triton"))
os.environ.setdefault("MAX_JOBS", "2")


def main():
    import torch
    from torch.nn import functional as F

    assert torch.version.cuda == "12.8", (torch.__version__, torch.version.cuda)
    assert torch.cuda.get_device_capability() == (12, 0)
    x = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(x @ x, torch.full_like(x, 32))
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | BF16 OK", flush=True)

    import flashinfer
    import sgl_kernel
    from flash_attn import flash_attn_func

    torch.manual_seed(42)
    x = torch.randn((8, 256), device="cuda", dtype=torch.float16)
    w = torch.ones_like(x[0])
    expected = F.rms_norm(x.float(), (256,), w.float(), eps=1e-6).half()
    for norm in (sgl_kernel.rmsnorm, flashinfer.rmsnorm):
        torch.testing.assert_close(norm(x, w), expected, atol=0.002, rtol=0.002)
        print(f"{norm.__module__}.rmsnorm OK", flush=True)

    from sglang.srt.layers.attention.vision import VisionAttention, VisionTritonAttention
    from sglang.srt.server_args import set_global_server_args_for_scheduler
    from transformers import Qwen3Config

    from sglang_omni.models.qwen3_asr.sglang_model import Qwen3ASRForConditionalGeneration
    from sglang_omni.scheduling.sglang_backend import build_sglang_server_args

    # ServerArgs reads config.json even when only inspecting the backend choice.
    with TemporaryDirectory() as tmp:
        Qwen3Config(architectures=["Qwen3ForCausalLM"]).save_pretrained(tmp)
        args = build_sglang_server_args(tmp, context_length=128)
        set_global_server_args_for_scheduler(args)
        backend = VisionAttention._determine_attention_backend(None, None)
        assert backend == "triton_attn", backend
        q, k, v = torch.randn((3, 32, 2, 64), device="cuda", dtype=torch.float16)
        scores = q.float().transpose(0, 1) @ k.float().permute(1, 2, 0) / 8
        expected = (scores.softmax(dim=-1) @ v.float().transpose(0, 1)).transpose(0, 1)
        actual = VisionTritonAttention(use_data_parallel=True)(q, k, v, None, 1, 32)
        torch.testing.assert_close(actual, expected.half(), atol=0.003, rtol=0.003)
        print(f"{Qwen3ASRForConditionalGeneration.__name__} import | {backend} OK")
        assert args.attention_backend == "flashinfer", args.attention_backend
        prefill = flashinfer.single_prefill_with_kv_cache(q, k, v)
        decode = flashinfer.single_decode_with_kv_cache(q[-1], k, v)
        fa2 = flash_attn_func(q[None], k[None], v[None])[0]
        for output, reference in ((prefill, expected), (decode, expected[-1]), (fa2, expected)):
            torch.testing.assert_close(output, reference.half(), atol=0.003, rtol=0.003)
        print("FlashInfer prefill/decode | FlashAttention 2 OK")


if __name__ == "__main__":
    main()
