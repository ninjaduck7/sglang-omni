# CUDA 12.8 feature matrix

Target: one RTX 5090 (SM120), driver 570.172.08, Torch 2.9.1+cu128,
SGLang 0.5.8; source baseline b83910176. Upstream comparison: efe971926.
Scope: selected upstream features for the two already qualified speech models.

2026-11-22 full EN/ZH ASR qualification: see the [measured results](docs/benchmarks/cu128-5090/REPORT.md)
and [reproduction instructions](scripts/cu128/SEEDTTS.md). The fixed-language sweep
covered all 3,108 SeedTTS rows at 1/2/4/8 clients, three times. A separate mixed
pass correctly identified 3,103/3,108 language labels. No request errors across
40,404 measured requests. This coverage does not extend to TTS or other languages.

| Feature | Upstream PR | Status | Acceptance / reason |
| --- | --- | --- | --- |
| Qwen3-ASR canonical language hints | [1325](https://github.com/sgl-project/sglang-omni/pull/1325) | Verified on GPU | English, ZH-cn and fr produced the expected short transcripts. Invalid hint returned HTTP 400; the next valid request succeeded. Other mapped languages were not evaluated on speech. |
| Qwen3-ASR automatic language detection | [1403](https://github.com/sgl-project/sglang-omni/pull/1403), [1781](https://github.com/sgl-project/sglang-omni/pull/1781) | Verified on GPU | Without hints, verbose_json returned English, Chinese and French with the expected text. Plain JSON still returned only text. |
| Qwen3-TTS non-streaming finish reason | [1185](https://github.com/sgl-project/sglang-omni/pull/1185), [2714](https://github.com/sgl-project/sglang-omni/pull/2714) | Verified on GPU | Budget 4: X-Finish-Reason=length, 4 tokens. Budget 128: stop after 31 tokens. Both bodies decoded as 24 kHz WAV. Checked the single-request, non-streaming speech endpoint only. |
| Qwen3-ASR long audio | [1347](https://github.com/sgl-project/sglang-omni/pull/1347) | Deferred | Needs separate chunk-boundary correctness and memory tests. |
| Qwen3-ASR streaming | [1359](https://github.com/sgl-project/sglang-omni/pull/1359) | Deferred | Needs a separate contract for partial results, cancellation and the last chunk. |
| Qwen3-ASR SM120 FP8 | [3659](https://github.com/sgl-project/sglang-omni/pull/3659) | Deferred | New kernels; no parity or speed evidence in this stack. |

Evidence (2026-11-21): real loopback HTTP requests to local checkpoints
Qwen3-ASR-1.7B and Qwen3-TTS-12Hz-0.6B-CustomVoice. See the
[base runbook](scripts/cu128/README.md) for model revisions, downloads and
the guarded launch commands. Start one service at a time on port 18591, then run
`python scripts/cu128/check_features.py asr` or `python scripts/cu128/check_features.py tts`
from the worktree with its venv active. The independently checked French clip is
`../../runs/sglang-omni/cu128-5090/feature-fr.wav` (eSpeak NG, "Bonjour, comment allez-vous ?").
Post it with `response_format=verbose_json`, once with `language=fr` and once without.

Minimum sampled free GPU memory: ASR 15045 MiB, TTS 16129 MiB (whole GPU).
Both services stopped. Peak temporary checkpoint storage was one ASR model
(4.4 GiB); neither model was retained. No change to environment pins.
No claims here about long-form accuracy, arbitrary languages, or workloads beyond the measured run.
