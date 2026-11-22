"""Join measured wall-clock intervals to NVML and process-tree CPU samples."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def obj(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def main(out):
    events=lines(out/"mixed-events.jsonl")
    ends=[e for e in events if e["kind"]=="cell_end"]
    starts=[e for e in events if e["kind"]=="cell_start"]
    samples={}
    for name in ("cold","soak","restart"):
        with (out/f"{name}-meter.csv").open() as f:
            samples[name]=[{k:float(v) if k!="phase" else v for k,v in r.items()} for r in csv.DictReader(f)]
    rows=[]
    def add(name,tag,group,offset=None):
        if not group:
            return
        def med(k):
            return statistics.median(p[k] for p in group)
        rows.append(dict(launch=name,stage=tag,n=len(group),
            first_unix=group[0]["unix_s"],last_unix=group[-1]["unix_s"],
            marker_offset_s=offset if offset is not None else "",
            median_service_mib=med("service_mib"),peak_service_mib=max(p["service_mib"] for p in group),
            median_other_compute_mib=med("other_compute_mib"),
            median_board_used_mib=med("total_mib")-med("free_mib"),
            min_board_free_mib=min(p["free_mib"] for p in group),
            mean_tree_cpu_pct=statistics.mean(p["tree_cpu_pct"] for p in group),
            peak_board_w=max(p["power_w"] for p in group),min_disk_free_bytes=min(p["disk_free_bytes"] for p in group)))
    for name,meter in samples.items():
        for tag in ("before","startup","first_request","after"):
            add(name,tag,[p for p in meter if p["phase"]==tag])
    def interval(label,t0,t1):
        add("soak",label,[p for p in samples["soak"] if t0<=p["unix_s"]<t1])
    interval("warm_idle_last_5s",starts[0]["unix"]-5,starts[0]["unix"])
    for s,e in zip(starts,ends,strict=True):
        assert s["c"]==e["c"]
        interval("c"+str(s["c"]),s["unix"],e["unix"])
    interval("cooldown_seconds_5_to_20",ends[-1]["unix"]+5,ends[-1]["unix"]+20)
    markers=lines(out/"restart-markers.jsonl")
    for m in markers:
        sample=min(samples["restart"],key=lambda x:abs(x["unix_s"]-m["unix"]))
        add("restart",m["marker"],[sample],round(sample["unix_s"]-m["unix"],3))
    with (out/"memory-stages.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    main(p.parse_args().out)
