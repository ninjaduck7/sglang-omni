# RTX 5090 local qualification, 2026-11-21

Run commands from the cu128 worktree with `source .venv/bin/activate`.
The base environment is described in [CU128.md](../../CU128.md).
Only the downloaded model directories and small WAVs live outside this repo.
Runtime requests are local; weights are fetched explicitly from ModelScope, pinned
by revision and verified per file with SHA256. ASR downloads 4.4 GiB and TTS 2.3 GiB.
The qualification weights were removed after testing; re-fetch only the model needed.

## Measured results

| Check | Observed result on driver 570.172.08 |
| --- | --- |
| FlashInfer 0.6.1 decode | One captured graph, 20 replays, 2 heads, FP16. Crossed a 16-token page boundary with non-contiguous page IDs. Every output matched an FP32 reference and the whole KV buffer matched exactly after each write. |
| Qwen3-ASR-1.7B BF16 | Real HTTP transcriptions for English and Mandarin. Six successful requests (two sequential plus four concurrent); both texts matched their references ignoring punctuation. |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | Vivian, English and Mandarin: mono PCM16, 24 kHz, 2.40 s and 2.16 s. Re-transcribing both generated WAVs with Qwen3-ASR returned the original text. The English request used 30 of 128 allowed tokens. |

Sampled free GPU memory (MiB from nvidia-smi, whole GPU; not per-process peaks):

| Service | Before | Lowest sampled | After exit |
| --- | ---: | ---: | ---: |
| ASR, single + concurrent requests | 21607 | 15033 | 21607 |
| TTS, sequential requests | 21607 | 16870 | 21607 |
| ASR, TTS round-trip | 21607 | 15752 | 21607 |

These are short-input checks, not a corpus accuracy, throughput, or soak benchmark.
The graphs stay enabled for both models. TTS uses ordinary decoder layers with
torch.compile disabled. The original code failed at graph capture when it assumed
compiled layers existed. The optional TTS overlay leaves the base Transformers
version untouched. A missing system SoX binary did not affect these two CustomVoice requests.

## Run one service at a time

```bash
python scripts/cu128/check_decode_graph.py

MODEL_ROOT="$(realpath ../../models)"
python scripts/cu128/fetch_model.py asr "$MODEL_ROOT"
python scripts/cu128/guard.py sgl-omni serve --host 127.0.0.1 --port 18590 \
  --config examples/configs/cu128_asr_5090.yaml \
  --model-path "$MODEL_ROOT/asr/qwen/huggingface/Qwen3-ASR-1.7B"
```

The guard checks physical GPU 0 before launch (14 GiB free required),
locks this environment to one guarded service, polls free memory every 0.5 s
and stops the service if a sample is below 4 GiB. This is not a hard allocation
limit. ASR limits KV capacity to 4096 tokens and concurrency to four. Stop
with Ctrl-C and check the final memory line before launching another service.
If the configured network proxy is unavailable, run just the download command
with `NO_PROXY='*' no_proxy='*'` on this host.

In another terminal, generate the English input and send a request:

```bash
out="$(realpath ../../runs)/sglang-omni/cu128-5090"
mkdir -p "$out"
ffmpeg -v error -f lavfi -i "flite=text='The weather is sunny today.':voice=slt" \
  -ar 16000 "$out/english.wav"
curl --noproxy '*' -fsS -F "file=@$out/english.wav" -F model=Qwen3-ASR-1.7B \
  http://127.0.0.1:18590/v1/audio/transcriptions
```

Mandarin input: [SeedTTS mini prompt audio](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/0df0b57b45253d5a9fa0b0e98846642f235f828e/zh/prompt-wavs/10002287-00000094.wav);
the prompt transcript is the second field of the corresponding `zh/meta.lst` row.
ASR returned “在此奉劝大家，别乱打美白针。”.

For TTS, install the small overlay once (skip installation if already present).
There is no second CUDA or PyTorch installation.

```bash
uv pip install --no-cache --python "$VIRTUAL_ENV/bin/python" --target "$VIRTUAL_ENV/tts-overlay" \
  --no-deps --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  qwen-tts==0.1.1 transformers==4.57.3 accelerate==1.12.0 sox==1.5.0
python scripts/cu128/fetch_model.py tts "$MODEL_ROOT"
PYTHONPATH="$VIRTUAL_ENV/tts-overlay" python scripts/cu128/guard.py \
  sgl-omni serve --host 127.0.0.1 --port 18590 \
  --config examples/configs/cu128_tts_5090.yaml \
  --model-path "$MODEL_ROOT/tts/qwen/huggingface/Qwen3-TTS-12Hz-0.6B-CustomVoice"
```

TTS limits concurrency to one and KV capacity to 8192 tokens. In the client
terminal, use a fresh output path for each synthesis:

```bash
curl --noproxy '*' -fsS http://127.0.0.1:18590/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"The weather is sunny today.","voice":"Vivian","language":"English","max_new_tokens":128,"response_format":"wav"}' \
  -o "$out/new-en.wav"
```

For the Mandarin request set `input` to “今天天气很好。” and `language` to
`Chinese`. To repeat the round-trip, stop TTS, start ASR, and post the new WAVs
to the transcription endpoint. Original local outputs are `vivian-en.wav` and
`vivian-zh.wav` under the run directory above.
