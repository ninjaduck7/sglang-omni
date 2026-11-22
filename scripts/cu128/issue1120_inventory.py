"""Record the local stack and compare model files with a pinned HF Git/LFS tree."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import psutil

HF_REV = "7278e1e70fe206f11671096ffdd38061171dd6e5"
MS_REV = "d69410f1c275f2b0fa60cbb9960edfcdb0ae0aec"


def command(*args):
    return subprocess.check_output(args,text=True).strip()


def main(out):
    hf = {f["path"]:f for f in json.loads((out/"hf-tree.json").read_text())}
    model = out/"private/asr/qwen/huggingface/Qwen3-ASR-1.7B"
    files = []
    for p in sorted(model.iterdir()):
        size = p.stat().st_size
        remote = hf.get(p.name,{})
        with p.open("rb") as f:
            digest = hashlib.file_digest(f,"sha256").hexdigest()
        if "lfs" in remote:
            key = "lfs_sha256"
            oid = remote["lfs"]["oid"]
            equal = oid == digest and size == remote["size"]
        elif remote:
            key = "git_blob_sha1"
            oid = remote["oid"]
            equal = hashlib.sha1(f"blob {size}\0".encode()+p.read_bytes()).hexdigest()==oid
        else:
            key = oid = equal = None
        files.append(dict(name=p.name,bytes=size,sha256=digest,hf_id_kind=key,hf_oid=oid,
                          matches_hf=equal))
    info = dict(date=datetime.now(timezone.utc).isoformat(),source=command("git","rev-parse","HEAD"),
        model_local={"repository":"ModelScope Qwen/Qwen3-ASR-1.7B","revision":MS_REV},
        model_comparison={"repository":"Hugging Face Qwen/Qwen3-ASR-1.7B","revision":HF_REV},
        files=files,seedtts_revision="27f4c1adee83b5b29b7c4b375f6b976324bda308",
        linux=platform.platform(),os=platform.freedesktop_os_release()["PRETTY_NAME"],
        python=sys.version,ram_bytes=psutil.virtual_memory().total,
        cpu=next(l.split(":",1)[1].strip() for l in command("lscpu").splitlines() if "Model name:" in l),
        logical_cpus=psutil.cpu_count(),toolkit=command("nvcc","--version"),
        gpu=command("nvidia-smi","--query-gpu=name,driver_version,compute_cap,memory.total,power.limit,power.default_limit,persistence_mode","--format=csv"),
        visible_mapping="CUDA_VISIBLE_DEVICES=0 -> only physical GPU, index 0",
        packages={p:metadata.version(p) for p in ("sglang-omni","torch","torchaudio","sglang","sgl-kernel",
            "nvidia-cuda-runtime-cu12","nvidia-cudnn-cu12","triton","transformers","flashinfer-python","flash-attn",
            "pyarrow","aiohttp","psutil","jiwer","openai-whisper","soundfile")})
    (out/"inventory.json").write_text(json.dumps(info,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    main(p.parse_args().out)
