# Local qualification protocol

The endpoints and fixtures below are local. Preserve each run under a fresh directory.
Do not change the resident limit, reuse another benchmark's private cache or mix
historical serving revisions. Compare results to the criteria below without
reclassifying client disconnects or a terminal HTTP response as server cancellation.

| Path | Criterion set before the soak |
| --- | --- |
| Historical short-form quality | Account for all IDs, repeat cells, reference units, errors and edits. Report observed ranges. No new retrospective H100 parity threshold. |
| Mixed stability | Four closed-loop cells of at least 450 s with the same server identity. Count all ordinary completions and all probe outcomes. Zero unexpected timeouts/5xx on valid traffic. Score fixed control transcripts separately. |
| Error isolation | Empty file, malformed audio and invalid parameters return 4xx; new independent fixed transcript and health succeed. 5xx fails the rejection criterion even if recovery succeeds. |
| Disconnect | Start from idle, observe exactly one pending request with no competing traffic. Disconnect it, sample pending state, then issue a new independent request. Claim backend cancellation only with request-specific abort evidence. |
| Streaming | Request a stream; require incremental text events and a terminal result for streaming/first-token timing. Terminal JSON alone does not qualify. |
| Boundary | Place known speech immediately before 30 s and wholly after 30 s with an identical prefix. Check for its words. Probe a much longer input or oversized context; distinguish HTTP acceptance, feature truncation and an actual decoder-context rejection. |
| Active shutdown | From idle start a request, observe it pending, signal the real server PID. Record the client outcome, port, all observed descendants and surviving service GPU PIDs. Rebind same port and recheck transcript. Distinguish guard group cleanup from server-only graceful shutdown. |
| Startup and memory | Disk-resident verified weights. First run with empty private JIT paths, then same paths with a new process. Record spawn-to-health and spawn-to-correct-output. Record board memory separately from descendants and other compute PIDs; warm and cooldown must use the same process. A single cooldown is not proof of leak freedom. |
| Resource guard | One service; existing guard requires 14 GiB free to launch and samples for a 4 GiB floor. Require 8 GiB free temporary headroom before launch. Stop only own processes. |

Raw downloads, private compiler cache and model files are transient. The local evidence directory
contains per-request records, probe outcomes, sampled resources and package/model manifests.
The existing full-corpus sweep is not repeated.


## Reproduce locally

Run from the cu128 worktree with its existing venv active. Leave 8 GiB
additional space. Inspect `nvidia-smi`, `df -h .` and `ss -ltn` first.
The supervisor always uses the existing single-GPU guard and loopback port
18593. Do not start if another service uses that port. The guard launches only
at >=14 GiB free and stops its own group below 4 GiB sampled free memory.
The fixed control clips are the already qualified `vivian-en.wav` and
`vivian-zh.wav` in `../../runs/sglang-omni/cu128-5090`.

```bash
source .venv/bin/activate
run_dir=$(mktemp -d "$(realpath ../../runs)/sglang-omni/asr-1120-XXXXXXXX")
mkdir "$run_dir/private"
old_dir=$(realpath ../../runs/sglang-omni/asr-cu128-seedtts-20261122)
model_dir="$run_dir/private/asr/qwen/huggingface/Qwen3-ASR-1.7B"
data_url=https://hf-mirror.com/datasets/zhaochenyang20/seed-tts-eval-arrow/resolve/27f4c1adee83b5b29b7c4b375f6b976324bda308/data
for lang in en zh; do
  curl -fLsS "$data_url/$lang-00000-of-00001.parquet" -o "$run_dir/private/$lang.parquet" || exit 1
done
python scripts/cu128/fetch_model.py asr "$run_dir/private"
curl -fLsS 'https://hf-mirror.com/api/models/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5?recursive=true&expand=false' -o "$run_dir/hf-tree.json"
python scripts/cu128/issue1120_inventory.py "$run_dir"
```

On this host the download commands used `NO_PROXY='*' no_proxy='*'`.
For each service start, run the supervisor in one terminal, and the probes in
a second one from the same worktree with the same paths. Wait for the specific
`<name>-ready.json` (two correct requests) before sending probes. Wait for
`<name>-exit.json`, port closure and no surviving service PIDs before starting
the next instance. A supervisor error is a failed launch; do not switch ports
or silently reuse partial evidence. Use fresh names/directories for a retry.

```bash
python scripts/cu128/issue1120_service.py "$run_dir" "$model_dir" cold
# Second terminal: after cold-ready.json exists:
touch "$run_dir/cold.stop"

python scripts/cu128/issue1120_service.py "$run_dir" "$model_dir" soak
# Second terminal, after soak-ready.json and at least 8 idle seconds:
python scripts/cu128/issue1120_load.py "$run_dir" mixed
sleep 20
python scripts/cu128/issue1120_load.py "$run_dir" edges
python scripts/cu128/issue1120_load.py "$run_dir" signal
# Record whether the real server drains/exits; guard cleanup is a separate outcome.
# If it does not exit, after 8 seconds request guard cleanup with this file:
touch "$run_dir/soak.stop"

# Start the log marker observer before launching the restart on the same port:
python scripts/cu128/issue1120_checkpoints.py "$run_dir" restart &
python scripts/cu128/issue1120_service.py "$run_dir" "$model_dir" restart
# Second terminal, after restart-ready.json:
touch "$run_dir/restart.stop"
```

Score after inference stops. Reuse the existing evaluation-only zhon installation
documented in [SEEDTTS.md](SEEDTTS.md). For the historical audit, create a new
`reaudit` directory with symlinks to the historical `*.jsonl.gz` and
`cells.jsonl`; the audit writes only to the new directory.

```bash
export PYTHONPATH="$PWD:$VIRTUAL_ENV/asr-eval-tools"
mkdir "$run_dir/reaudit"
ln -s "$old_dir"/*.jsonl.gz "$old_dir/cells.jsonl" "$run_dir/reaudit/"
python scripts/cu128/audit_seedtts.py "$run_dir/private" "$run_dir/reaudit"
cmp "$run_dir/reaudit/metrics.csv" "$old_dir/metrics.csv"
python scripts/cu128/issue1120_metrics.py "$run_dir" "$old_dir"
python scripts/cu128/issue1120_resources.py "$run_dir"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python scripts/cu128/issue1120_window.py "$run_dir"
```

The ordinary soak draws the first 32 rows of each split in alternating order;
it is a stability workload with repetitions, not another full corpus sample.
Extra control, health and invalid requests may overlap clients during the
injection burst; isolated disconnects and active shutdown start from idle.
End-to-end HTTP latency includes upload, queue and response time. RTF is the
mean per-request ratio of latency to input duration; audio s/s sums successful
input duration over cell wall time. Neither metric is the reciprocal of the other
under concurrency. The timer includes draining last submissions. The four-cell
soak uses no server restart between concurrency cells.

The meter samples NVML and CPU about every 0.5 s. Per-PID GPU memory is NVML
compute memory; board used also includes graphics/driver allocations. CPU sums
user+system deltas for the guard/service tree, where 100% is one logical CPU;
first-seen or short-lived processes can be undercounted. Board watts and
utilization are shared. Stage markers are progress-log observation times;
the reported signed sample offsets are not allocator timestamps.

When hashes, receipts and exit checks are complete, remove only this fresh
run's `private` directory. Never clean the venv, shared cache or historical run.
The raw evidence stays local. See the [report](../../docs/benchmarks/cu128-5090/ISSUE-1120-BENCHMARK.md)
for the results and the subset included with the repository.
