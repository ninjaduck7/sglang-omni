# Reproduce the RTX 5090 short-form ASR measurements

Use the cu128 worktree, its existing environment and the single-GPU profile.
The benchmark is local; no model, audio or report is published by the commands
below. In addition to the existing environment, budget **8 GiB free**. Measured
input footprint: about 4.4 GiB of weights and 669.3 MiB for **both** Parquet
splits. No separate Hugging Face cache, extracted WAV tree, or second model copy
is required. Dataset revision and per-file SHA256 checks are enforced by the
runner. Use a fresh evidence directory per run, never reuse a partial sweep.

```bash
source .venv/bin/activate
inputs="$(mktemp -d "$VIRTUAL_ENV/publish-asr.XXXXXX")"
output="$(realpath ../../runs)/sglang-omni/asr-cu128-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$output"
dataset=https://hf-mirror.com/datasets/zhaochenyang20/seed-tts-eval-arrow/resolve/27f4c1adee83b5b29b7c4b375f6b976324bda308/data
for split in en zh; do
  curl -fLsS --retry 2 -o "$inputs/$split.parquet" "$dataset/$split-00000-of-00001.parquet" || exit 1
done
python scripts/cu128/fetch_model.py asr "$inputs"
uv pip install --no-cache --no-deps --python "$VIRTUAL_ENV/bin/python" \
  --target "$VIRTUAL_ENV/asr-eval-tools" zhon==2.1.1
model="$inputs/asr/qwen/huggingface/Qwen3-ASR-1.7B"
python scripts/cu128/run_seedtts.py "$model" "$inputs" "$output"
PYTHONPATH="$PWD:$VIRTUAL_ENV/asr-eval-tools" python scripts/cu128/audit_seedtts.py "$inputs" "$output"
python scripts/cu128/render_seedtts.py "$output"
```

On this host, use `NO_PROXY='*' no_proxy='*'` for the download commands if
the local proxy is unavailable. Our run fetched the small evaluation-only
package from `https://pypi.tuna.tsinghua.edu.cn/simple`. After a successful
audit, remove only the disposable `$inputs` directory created above.

The supervisor binds the service to loopback, runs it under the 4 GiB
free-memory guard, and records **its descendants'** NVML compute memory
separately from the rest of the system. The 0.5 s guard and approximately
2 s telemetry are sampled, not hard allocation limits. When all measured cells
are complete it checks health, stops the entire service group, and checks port
closure and GPU process cleanup.

For each split, `ref_audio` is the input and `ref_text` the ground truth,
including repeated recordings that belong to distinct dataset rows. Do not use
`target_text`. The output reports **corpus-weighted** errors. Each failing
request remains in the denominator as an empty hypothesis. Report HTTP errors
separately. A finite aggregate from an incomplete sweep is not a pass.

Twenty-four measured cells use the complete splits, hints and three repeats.
The final mixed cell uses the complete corpus with automatic language detection,
four clients and one repeat. No live retries or omitted score outliers. The
published latency is end-to-end **after warmup**; it excludes initial asset
download, startup, the per-cell warmup and post-request scoring. Eight HTTP
clients have at most four resident engine requests. Sampling temperature uses
the service's zero-to-0.01 behavior. No exclusive-GPU, cross-version speedup,
TTS quality or long-audio claim should be made from this experiment.

For independent verification, retain `*.jsonl.gz`, `cells.jsonl`,
`dataset.json`, `environment.json` and `gpu.csv`. Re-running the audit needs
only the pinned Parquets and the evaluation dependencies, not the model or a GPU.
It checks every ID, reference unit and edit count using jiwer, rechecks P95,
and writes `metrics.csv` and `audit.json`.
