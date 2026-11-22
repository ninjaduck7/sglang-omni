"""Independently rescore saved responses against the pinned Parquet reference texts."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from jiwer import process_words

from benchmarks.tasks.tts import normalize_text
from scripts.cu128.seedtts_sweep import REVISION, SPLITS


def references(data):
    refs = {}
    for split, (count, digest) in SPLITS.items():
        path = data / f"{split}.parquet"
        with path.open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == digest
        table = pq.read_table(path, columns=["sample_id", "ref_text"])
        refs[split] = {
            row["sample_id"]: normalize_text(row["ref_text"], split)
            for row in table.to_pylist()
        }
        assert len(refs[split]) == count
    return refs


def audit(args):
    refs = references(args.data)
    expected_cells = {
        f"{mode}-c{c}-r{r}"
        for mode in SPLITS for c in (1, 2, 4, 8) for r in (1, 2, 3)
    } | {"auto-c4-r1"}
    cells = [json.loads(line) for line in (args.run / "cells.jsonl").read_text().splitlines()]
    assert len(cells) == len(expected_cells) and {c["cell"] for c in cells} == expected_cells
    csv_rows = []
    detection = {}
    for cell in cells:
        tag = cell["mode"]
        langs = SPLITS if tag == "auto" else [tag]
        expected = {(lang, uid) for lang in langs for uid in refs[lang]}
        seen, edits, units, latency = set(), Counter(), Counter(), []
        errors = 0
        with gzip.open(args.run / (cell["cell"] + ".jsonl.gz"), "rt") as predictions:
            for line in predictions:
                row = json.loads(line)
                lang, uid = row["lang"], row["id"]
                assert (lang, uid) in expected and (lang, uid) not in seen
                seen.add((lang, uid))
                truth = refs[lang][uid]
                hyp = normalize_text(row.get("text", ""), lang)
                scored = process_words(truth, hyp)
                edit_count = scored.substitutions + scored.deletions + scored.insertions
                reference_units = scored.hits + scored.deletions + scored.substitutions
                assert edit_count == row["edits"] and reference_units == row["units"]
                edits[lang] += edit_count
                units[lang] += reference_units
                if "error" in row:
                    errors += 1
                else:
                    latency.append(row["latency"])
                if tag == "auto":
                    key = row.get("detected") or "(none)"
                    counts = detection.setdefault(lang, Counter())
                    counts[key] += 1
        assert seen == expected and errors == cell["errors"]
        assert abs(np.percentile(latency, 95) * 1000 - cell["p95_ms"]) < 1e-6
        for lang in langs:
            assert [edits[lang], units[lang]] == cell["counts"][lang]
        csv_rows.append({
            "cell": cell["cell"], "requests": len(seen), "errors": errors,
            "elapsed_s": cell["elapsed_s"], "success_per_s": cell["success_per_s"],
            "mean_ms": cell["mean_ms"], "p50_ms": cell["p50_ms"],
            "p95_ms": cell["p95_ms"], "p99_ms": cell["p99_ms"],
            "mean_rtf": cell["mean_rtf"], "audio_s_per_wall_s": cell["audio_s_per_wall_s"],
            **{f"{lang}_{key}": values.get(lang, "") for lang in SPLITS
               for key, values in [("edits", edits), ("reference_units", units)]},
        })
    auto_cell = next(c for c in cells if c["mode"] == "auto")
    assert detection["en"]["English"] + detection["zh"]["Chinese"] == auto_cell["correct_language"]
    with (args.run / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    result = dict(dataset_revision=REVISION, cells=len(cells),
                  verified_rows=sum(c["requests"] for c in cells),
                  errors=sum(c["errors"] for c in cells), detection=detection)
    (args.run / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("run", type=Path)
    audit(parser.parse_args())
