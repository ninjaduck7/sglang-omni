# Evidence index

Read the [four-part report](../ISSUE-1120-BENCHMARK.md) alongside the
[protocol, start/stop commands and metric definitions](../../../../scripts/cu128/ISSUE1120.md).
This directory contains summaries; it does not contain audio, weights or the full
per-request transcripts. [SHA256SUMS](SHA256SUMS) covers the report and these
attachments. From this directory run `sha256sum -c SHA256SUMS`.

- [Inventory](inventory.json): exact package versions, host, GPU, power limit and hashes for every downloaded checkpoint file. The inventory was captured during the cold-launch revision; lifecycle receipts give the SHA for each server process.
- [Historical metrics](historical-summary.csv), [original 25 cells](../metrics.csv) and [five auto-language anomalies](auto-exceptions.json): same original serving revision. The old 40,404 saved outputs were independently rescored against the pinned reference transcripts; the audit and CSV were byte-identical to the originals.
- [New mixed metrics](soak-summary.csv): split-specific and pooled ordinary request counts, latency, RTF, audio seconds/s, edit/reference units. Per-language throughput is each language's contribution to the same mixed cell wall time. Do not sum the pooled row with the language rows; no mixed-language error percentage combines words with characters.
- [Lifecycle](lifecycle.json): cold, unsuccessful port attempt, warm, final restart, status codes, correct fixed controls, in-flight disconnects, SIGTERM and duration-probe receipts. An event named `drained` records a snapshot, not an assertion that its count is zero. Here both such snapshots still had one pending completion. The server was explicitly signaled during the in-flight test; guard group cleanup is part of the measured deployment. Exit 241 is consistent with propagating -SIGTERM through the guard, not an unsignaled zero exit. At shutdown the Python resource tracker warned of nine semaphore objects to clean up. No surviving service processes/GPU PIDs were observed; no claim about every IPC object.
- [Memory stages](memory-stages.csv): NVML board, service and other compute metrics, CPU and disk at each phase or marker. NVML whole-board free is the guard metric. Other compute may transiently include former descendants during teardown when their parent exits; it is not a board-minus-service calculation. Restart load progress and graph markers were polled every 50 ms and joined to the nearest roughly 0.5 s sample. Marker snapshots do not isolate a weight, KV or graph allocation. CPU 100% equals one logical CPU. Cold vs warm is for private compiler caches with model files already on disk; it includes neither model download nor a reboot.
- [Window probe](feature-window.json): default CPU feature-extractor output changed for a tail inside 30 seconds, remained bit-identical to the prefix when the tail was wholly after 30 seconds. Actual HTTP transcripts and the 5,000-word prompt probe are in the lifecycle file. The legacy builder uses audio placeholders and a forced-language suffix; multipart `prompt` does not become a decoder prefix. This is not a deliberately over-length decoder token submission.

Local-only raw run, relative to the AIinfer workspace:
`runs/sglang-omni/asr-1120-20261122T043957Z-Mic6fl/`.
It contains `mixed-c{1,2,4,8}.jsonl.gz`, per-launch logs and 0.5 s meter CSVs,
probe events, original failed-port receipts, pre/post fix regression output,
fixture hashes and `evidence.sha256`. Verify from that directory using
`sha256sum -c evidence.sha256`. Original full-split evidence remains in
`runs/sglang-omni/asr-cu128-seedtts-20261122/` with its own checksum manifest.
Recomputing normalized per-request text scores needs the original pinned Parquets.
Upstream reference documents were read locally from the public roadmap, original
issue, and merge PR; no data or comments were posted.

Disk on /mnt/data: before 341,626,736,640 bytes free; minimum across service-meter samples
336,167,645,184; after input cleanup 341,591,658,496 bytes free. These are shared
filesystem readings, not charged exclusively to this experiment. The private
input/cache directory measured 5,425,130,689 bytes (5.05 GiB) before removal.
Budget 8 GiB additional free beyond the existing venv. Before startup the board
reported 7,397 MiB used of 32,607 MiB; after cleanup it reported 7,392 MiB used
and no experiment service GPU processes. Other users' processes and caches were left alone.
