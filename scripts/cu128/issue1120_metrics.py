"""Recompute measured metrics for the lifecycle report without running inference."""
import argparse
import csv
import gzip
import io
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from rapidfuzz.distance import Levenshtein

from benchmarks.tasks.tts import normalize_text
from scripts.cu128.audit_seedtts import references


def lines(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def load(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def write(p, rows):
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(vals,q):
    return float(np.percentile(vals,q))


def main(out, historical):
    refs = references(out/"private")
    old = load(historical/"metrics.csv")
    old_summary = []
    for lang in ("en","zh"):
        for c in (1,2,4,8):
            rows=[r for r in old if r["cell"].startswith(f"{lang}-c{c}-")]
            assert len(rows)==3
            rates=[100*int(r[lang+"_edits"])/int(r[lang+"_reference_units"]) for r in rows]
            old_summary.append(dict(lang=lang,c=c,requests=sum(int(r["requests"]) for r in rows),
                errors=sum(int(r["errors"]) for r in rows),quality_min=min(rates),quality_max=max(rates),
                **{k:statistics.median(float(r[k]) for r in rows) for k in
                   ("success_per_s","p95_ms","mean_rtf","audio_s_per_wall_s")}))
    write(out/"historical-summary.csv",old_summary)
    durations={}
    for l in refs:
        batch=next(pq.ParquetFile(out/"private"/f"{l}.parquet").iter_batches(
            batch_size=32,columns=["sample_id","ref_audio"]))
        for row in batch.to_pylist():
            durations[l,row["sample_id"]]=sf.info(io.BytesIO(row["ref_audio"]["bytes"])).duration
    events=lines(out/"mixed-events.jsonl")
    cells={r["c"]:r for r in events if r["kind"]=="cell_end"}
    meters=load(out/"soak-meter.csv")
    assert len(cells)==4 and any(r["kind"]=="soak_finished" for r in events)
    summary=[]
    for c,cell in cells.items():
        wall=cell["elapsed"]
        epoch=next(r["unix"] for r in events if r["kind"]=="cell_start" and r["c"]==c)
        meter=[r for r in meters if epoch<=float(r["unix_s"])<=cell["unix"]]
        with gzip.open(out/f"mixed-c{c}.jsonl.gz","rt") as f:
            data=[json.loads(s) for s in f]
        for lang in ("en","zh","mixed"):
            group=[r for r in data if r["lang"]==lang or lang=="mixed"]
            good=[r for r in group if r["status"]==200 and isinstance(r.get("body"),dict)
                  and isinstance(r["body"].get("text"),str)]
            lat=[r["latency"] for r in good]
            dur=[durations[r["lang"],r["id"]] for r in good]
            edits=units=0
            for r in group:
                l=r["lang"]
                truth=refs[l][r["id"]].split()
                # Failed and malformed responses are empty hypotheses.
                body=r.get("body")
                text=body.get("text","") if isinstance(body,dict) else ""
                hyp=normalize_text(text if isinstance(text,str) and r["status"]==200 else "",l).split()
                edits+=Levenshtein.distance(truth,hyp)
                units+=len(truth)
            summary.append(dict(lang=lang,c=c,rows=len(group),distinct_ids=len({(r["lang"],r["id"]) for r in group}),
                valid=len(good),errors=len(group)-len(good),statuses=json.dumps(Counter(str(r["status"]) for r in group)),
                transport=json.dumps(Counter(r.get("error") for r in group if not r["status"])),
                edits=edits if lang!="mixed" else "",ref_units=units if lang!="mixed" else "",
                corpus_error_pct=100*edits/units if lang!="mixed" else "",
                elapsed_s=wall,req_per_s=len(good)/wall,p50_ms=1000*percentile(lat,50),p95_ms=1000*percentile(lat,95),
                p99_ms=1000*percentile(lat,99),mean_rtf=statistics.mean(t/d for t,d in zip(lat,dur)),
                audio_s_per_wall_s=sum(dur)/wall,mean_tree_cpu_pct=statistics.mean(float(r["tree_cpu_pct"]) for r in meter),
                service_peak_mib=max(int(r["service_mib"]) for r in meter),board_free_min_mib=min(int(r["free_mib"]) for r in meter)))
    write(out/"soak-summary.csv",summary)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("historical",type=Path)
    a=p.parse_args()
    main(a.out,a.historical)
