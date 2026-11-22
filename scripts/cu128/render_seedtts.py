"""Render audited full-split measurements; refuse incomplete run directories."""

import argparse
import csv
import json
import statistics as st
from pathlib import Path


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [json.loads(line) for line in (root / "cells.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")
    for ax, key, ylabel in zip(axes, ("success_per_s", "p95_ms"), ("Requests / second", "P95 latency (ms)")):
        for tag, color in (("en", "#196284"), ("zh", "#a34722")):
            points = [[r[key] for r in cells if r["mode"] == tag and r["concurrency"] == c]
                      for c in (1, 2, 4, 8)]
            med = [st.median(p) for p in points]
            spread = [[m - min(p) for m, p in zip(med, points)],
                      [max(p) - m for m, p in zip(med, points)]]
            ax.errorbar([1, 2, 4, 8], med, yerr=spread, marker="o", capsize=3,
                        color=color, label=tag.upper())
        ax.set(xlabel="Concurrent HTTP clients", ylabel=ylabel, xticks=[1, 2, 4, 8], ylim=(0, None))
        ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Qwen3-ASR-1.7B · Shared RTX 5090 / cu128 · Median and range of 3 runs")
    fig.savefig(root / "scaling.png", dpi=180)
    plt.close(fig)


def render(root):
    def read(name):
        return json.loads((root / name).read_text())

    audit = read("audit.json")
    env = read("environment.json")
    dataset = read("dataset.json")["splits"]
    cells = [json.loads(line) for line in (root / "cells.jsonl").read_text().splitlines()]
    assert audit["verified_rows"] == 40404 and audit["cells"] == 25
    telemetry = list(csv.DictReader((root / "gpu.csv").open()))
    table = []
    for lang in ("en", "zh"):
        for concurrency in (1, 2, 4, 8):
            subset = [r for r in cells if r["mode"] == lang and r["concurrency"] == concurrency]
            assert len(subset) == 3
            speed = [r["success_per_s"] for r in subset]
            p95 = [r["p95_ms"] for r in subset]
            error = [r["error_rates"][lang] * 100 for r in subset]
            table.append(
                f"| {lang.upper()} | {concurrency} | {st.median(speed):.2f} "
                f"({min(speed):.2f}-{max(speed):.2f}) | {st.median(p95):.0f} "
                f"({min(p95):.0f}-{max(p95):.0f}) | {min(error):.3f}-{max(error):.3f}% |"
            )
    auto = next(r for r in cells if r["mode"] == "auto")
    initial = cells[0]["start_unix"]
    last = auto["start_unix"] + auto["elapsed_s"]
    aligned = [r for r in telemetry if initial <= float(r["unix_s"]) <= last]
    memory = [int(r["service_mib"]) for r in aligned]
    other = [int(r["other_compute_mib"]) for r in aligned]
    free = [int(r["free_mib"]) for r in aligned]
    return "\n".join([
        "# Qwen3-ASR-1.7B BF16 on one RTX 5090 with CUDA 12.8",
        "",
        f"Tested serving source: `{env['source']}`. Driver {env['driver']}; "
        f"PyTorch {env['packages']['torch']}; SGLang {env['packages']['sglang']}; "
        f"FlashInfer {env['packages']['flashinfer-python']}. Linux, Intel Core Ultra 9 285K. "
        "One shared 32 GB desktop GPU; this is not an exclusive-GPU benchmark.",
        "",
        f"All {audit['verified_rows']:,} measured requests were accounted for; "
        f"{audit['errors']} request errors. Warmups excluded. The fixed-language sweep "
        "ran 37,296 requests in 24 cells. One additional mixed-language pass ran 3,108 requests.",
        "",
        "| Split | Clients | Requests/s: median (range) | P95 ms: median (range) | Corpus WER (EN) / CER (ZH), range |",
        "| --- | ---: | ---: | ---: | ---: |",
        *table, "", "![Throughput and latency by concurrency](scaling.png)", "",
        f"Without language hints (mixed order, four clients, single pass): "
        f"{auto['correct_language']}/{auto['requests']} language labels matched the split "
        f"({auto['correct_language']/auto['requests']:.2%}); English WER "
        f"{auto['error_rates']['en']:.3%}, Mandarin CER {auto['error_rates']['zh']:.3%}.",
        "",
        "## What was measured",
        "",
        f"- SeedTTS revision `{audit['dataset_revision']}`: all 1,088 EN rows and 2,020 ZH rows. "
        f"These reference recordings have {dataset['en']['unique_audio']} and "
        f"{dataset['zh']['unique_audio']} distinct byte hashes, respectively. "
        "All rows have equal standing; the corpus error rate is weighted by reference length. "
        "Ground truth is ref_text for ref_audio, not the TTS target text.",
        "- Closed-loop HTTP on localhost, real multipart uploads; clients reuse connections. "
        "Warm 2 x client concurrency per cell; deterministic shuffled order per repeat. "
        "Explicit language hints for the sweep; different concurrency order in each round. "
        "BF16, FlashInfer decoder, Triton audio attention, CUDA Graph up to batch 4, "
        "torch.compile off, 4 resident requests, 4,096 static KV tokens, mem_fraction_static=0.65. "
        "Eight clients exercise queueing above the resident limit. Default cache policy retained.",
        "- English uses the repository's Whisper English normalizer and word edit distance; "
        "Chinese uses its character normalizer. Failed requests are scored as empty hypotheses. "
        "Latency starts at HTTP submission and includes queueing and response transfer. "
        "Full per-run throughput, mean/P50/P95/P99 latency, RTF, error and unit counts are in metrics.csv. "
        "Per-request IDs and transcripts are retained as compressed JSONL without audio bytes.",
        f"- Measured span {((last-initial)/60):.1f} minutes. NVML sampled about every two seconds: "
        f"ASR descendants peaked at {max(memory)} MiB, minimum free {min(free)} MiB; "
        f"unrelated compute processes occupied {min(other)}-{max(other)} MiB. "
        f"Startup with pre-existing kernel caches: {env['warm_cache_startup_seconds']:.1f} seconds "
        "(one launch). CPU and GPU are shared; power/utilization are board-wide.",
        f"- Shutdown: listener closed={env['port_closed']}; remaining service GPU PIDs="
        f"{env['remaining_owned_gpu_pids']}.",
        "- Dataset bytes: 701,857,722 (669.3 MiB); one model and one corpus copy, about "
        "5.1 GiB total transient inputs. Reserve 8 GiB in addition to the existing environment.",
        "",
        "## Limits",
        "",
        "This qualifies this short-form EN/ZH ASR workload and the stated admission cap. "
        "It does not establish long-audio correctness, streaming behavior, TTS quality, "
        "FP8 throughput, or absence of regressions versus another software revision. "
        "It is not an A/B test against CUDA 13. Upstream issue "
        "[#1212](https://github.com/sgl-project/sglang-omni/issues/1212) used a different "
        "runtime, host and admission limit. Automatic language detection received one mixed "
        "pass, not the three-repeat sweep. No model weights or recordings are redistributed here.",
        "",
    ])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    (args.run / "REPORT.md").write_text(render(args.run))
    plot(args.run)
