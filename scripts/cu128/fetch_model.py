"""Fetch one pinned Qwen checkpoint directly to a fresh model directory."""

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

MODELS = {
    "asr": ("Qwen3-ASR-1.7B", "d69410f1c275f2b0fa60cbb9960edfcdb0ae0aec"),
    "tts": ("Qwen3-TTS-12Hz-0.6B-CustomVoice", "3b07e1843e41f3863f213c84e7003ad836a059c6"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=MODELS)
    p.add_argument("models_root", type=Path)
    args = p.parse_args()
    name, rev = MODELS[args.task]
    dest = args.models_root / args.task / "qwen/huggingface" / name
    api = f"https://modelscope.cn/api/v1/models/Qwen/{name}/repo/files?"
    with urllib.request.urlopen(api + urllib.parse.urlencode(
        {"Revision": rev, "Recursive": "true"}
    ), timeout=20) as response:
        files = json.load(response)["Data"]["Files"]
    files = [f for f in files if f["Type"] == "blob" and
             f["Path"].endswith((".json", ".safetensors", ".txt"))]
    if not any(f["Path"].endswith(".safetensors") for f in files):
        raise ValueError("Checkpoint contains no weights")
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(dest).free < sum(f["Size"] for f in files) + (2 << 30):
        raise RuntimeError("Insufficient disk space")
    for f in files:
        path = dest / f["Path"]
        path.resolve().relative_to(dest.resolve())
        path.parent.mkdir(exist_ok=True, parents=True)
        tmp = path.with_name(path.name + ".part")
        if not path.exists():
            url = f"https://modelscope.cn/models/Qwen/{name}/resolve/{rev}/" + urllib.parse.quote(f["Path"])
            subprocess.run(["curl", "-fLsS", "--retry", "3", "--connect-timeout", "10",
                            "--speed-limit", "4096", "--speed-time", "45", "-C", "-",
                            "-o", str(tmp), url], check=True)
        target = path if path.exists() else tmp
        with target.open("rb") as data:
            if target.stat().st_size != f["Size"] or hashlib.file_digest(data, "sha256").hexdigest() != f["Sha256"].lower():
                raise ValueError(f"Checksum mismatch: {target}")
        if target == tmp:
            tmp.rename(path)
        print(f"verified {path.name}: {f['Size']} bytes", flush=True)
    print(dest)


if __name__ == "__main__":
    main()
