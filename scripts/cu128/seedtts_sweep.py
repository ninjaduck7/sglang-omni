"""Measure the full SeedTTS reference-audio workload on a local speech server."""

import argparse
import asyncio
import gzip
import hashlib
import io
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from rapidfuzz.distance import Levenshtein

from benchmarks.tasks.tts import normalize_text

REVISION = "27f4c1adee83b5b29b7c4b375f6b976324bda308"
SPLITS = {
    "en": (1088, "5849b41b49cae996328c06d2c5791717c3bafc369bddfa1ec4f86761bb8bc0ca"),
    "zh": (2020, "5d73be79cf2e84fd0bebf5d47212438a88dd264a591031d092164ba84545f1de"),
}


@dataclass
class Clip:
    id: str
    audio: bytes
    reference: str
    duration: float


def load_split(root, lang):
    path = root / f"{lang}.parquet"
    count, checksum = SPLITS[lang]
    with path.open("rb") as data:
        assert hashlib.file_digest(data, "sha256").hexdigest() == checksum
    clips = []
    recordings = set()
    columns = ["sample_id", "ref_text", "ref_audio"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=columns):
        for row in batch.to_pylist():
            audio = row["ref_audio"]["bytes"]
            duration = sf.info(io.BytesIO(audio)).duration
            assert duration > 0 and row["ref_text"].strip()
            recordings.add(hashlib.sha256(audio).digest())
            clips.append(Clip(row["sample_id"], audio, row["ref_text"], duration))
    assert len(clips) == len({s.id for s in clips}) == count
    return clips, {"rows": count, "unique_audio": len(recordings), "sha256": checksum,
                   "audio_seconds": sum(s.duration for s in clips)}


async def request(session, url, clip, lang, auto):
    form = aiohttp.FormData()
    form.add_field("file", clip.audio, filename="clip.wav", content_type="audio/wav")
    form.add_field("response_format", "verbose_json" if auto else "json")
    if not auto:
        form.add_field("language", lang)
    start = time.perf_counter()
    row = {"id": clip.id, "duration": clip.duration, "lang": lang}
    try:
        async with session.post(url, data=form) as response:
            response.raise_for_status()
            payload = await response.json()
            row["text"] = payload["text"]
            if not isinstance(row["text"], str):
                raise ValueError("Missing transcript")
            if auto:
                row["detected"] = payload.get("language")
    except (aiohttp.ClientError, TimeoutError, KeyError, ValueError) as exc:
        row["error"] = repr(exc)
    row["latency"] = time.perf_counter() - start
    return row


async def measure(session, url, pairs, concurrency, auto):
    queue = iter(pairs)
    rows = []

    async def worker():
        for lang, clip in queue:
            rows.append(await request(session, url, clip, lang, auto))

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return rows, time.perf_counter() - start


async def cell(session, url, pairs, concurrency, repeat, tag, out, auto=False):
    prefix = f"{tag}-c{concurrency}-r{repeat}"
    # Warm each cell at the offered concurrency. Measure every dataset row below.
    warm, _ = await measure(session, url, pairs[:2 * concurrency], concurrency, auto)
    if any("error" in r for r in warm):
        raise RuntimeError(f"Warmup failed: {warm}")
    start_unix = time.time()
    rows, elapsed = await measure(session, url, pairs, concurrency, auto)
    refs = {(lang, c.id): c.reference for lang, c in pairs}
    assert len(rows) == len(refs) == len(pairs)
    counts = {}
    with gzip.open(out / f"{prefix}.jsonl.gz", "wt", encoding="utf-8") as file:
        for row in rows:
            lang = row["lang"]
            units = normalize_text(refs[lang, row["id"]], lang).split()
            hyp = normalize_text(row.get("text", ""), lang).split()
            row["edits"] = Levenshtein.distance(units, hyp)
            row["units"] = len(units)
            totals = counts.setdefault(lang, [0, 0])
            totals[0] += row["edits"]
            totals[1] += row["units"]
            print(json.dumps(row, ensure_ascii=False), file=file)
    ok = [r for r in rows if "error" not in r]
    if not ok:
        raise RuntimeError(f"No successful request in {prefix}")
    latencies = [r["latency"] for r in ok]
    summary = {
        "cell": prefix, "start_unix": start_unix, "concurrency": concurrency,
        "repeat": repeat, "mode": tag, "requests": len(rows), "errors": len(rows)-len(ok),
        "elapsed_s": elapsed, "success_per_s": len(ok)/elapsed,
        "mean_ms": 1000 * statistics.mean(latencies),
        "p50_ms": 1000 * float(np.percentile(latencies, 50)),
        "p95_ms": 1000 * float(np.percentile(latencies, 95)),
        "p99_ms": 1000 * float(np.percentile(latencies, 99)),
        "mean_rtf": statistics.mean(r["latency"]/r["duration"] for r in ok),
        "audio_s_per_wall_s": sum(r["duration"] for r in ok)/elapsed,
        "counts": counts, "error_rates": {k: e/n for k,(e,n) in counts.items()},
    }
    if auto:
        summary["correct_language"] = sum(
            r.get("detected") == {"en":"English","zh":"Chinese"}[r["lang"]] for r in rows
        )
    with (out/"cells.jsonl").open("a") as f:
        print(json.dumps(summary), file=f)
    print(prefix, f'{len(ok)}/{len(rows)} at {len(ok)/elapsed:.2f} req/s', flush=True)


async def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"cells.jsonl").exists():
        raise FileExistsError("Choose a fresh evidence directory")
    data, desc = {}, {}
    for lang in SPLITS:
        data[lang], desc[lang] = load_split(args.data, lang)
    (args.out/"dataset.json").write_text(json.dumps({"revision": REVISION, "splits": desc}, indent=2))
    url = f"http://127.0.0.1:{args.port}/v1/audio/transcriptions"
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for repeat, levels in enumerate(((1,2,4,8), (4,8,1,2), (8,4,2,1)), 1):
            for concurrency in levels:
                for lang in SPLITS:
                    pairs = [(lang, c) for c in data[lang]]
                    random.Random(5090 + repeat).shuffle(pairs)
                    await cell(session, url, pairs, concurrency, repeat, lang, args.out)
        mixed = [(lang, c) for lang, clips in data.items() for c in clips]
        random.Random(570).shuffle(mixed)
        await cell(session, url, mixed, 4, 1, "auto", args.out, auto=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--port", type=int, default=18592)
    asyncio.run(main(parser.parse_args()))
