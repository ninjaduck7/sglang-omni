"""Timestamp visible startup progress markers to align with independently sampled NVML."""
import argparse
import json
from pathlib import Path
import re
import time

MARKERS = {
    "weights_first_shard": rb"Loading safetensors checkpoint shards:\s+50%",
    "weights_progress_100": rb"Loading safetensors checkpoint shards:\s+100%",
    "capture_start": rb"Capturing batches \(bs=4 ",
    "capture_progress_100": rb"Capturing batches \(bs=1 [^\r\n]+?100%",
    "http_listening": rb"Uvicorn running on",
}


def main(out, name):
    path = out/f"{name}-server.log"
    with (out/f"{name}-markers.jsonl").open("x",buffering=1) as f:
        while not path.exists():
            time.sleep(.25)
        with path.open("rb") as log:
            tail = b""
            seen = set()
            deadline = time.monotonic()+600
            while time.monotonic()<deadline and not (out/f"{name}-exit.json").exists():
                tail = (tail+log.read())[-8192:]
                for key,pattern in MARKERS.items():
                    if key not in seen and re.search(pattern,tail):
                        print(json.dumps({"unix":time.time(),"marker":key}),file=f)
                        seen.add(key)
                if "http_listening" in seen:
                    return
                time.sleep(.05)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("name")
    a=p.parse_args()
    main(a.out,a.name)
