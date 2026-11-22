"""Verify a fresh guarded ASR service after the sweep has shut down."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pynvml as nv
import requests


def main(args):
    original = json.loads((args.run / "environment.json").read_text())
    assert original["port_closed"] and not original["remaining_owned_gpu_pids"]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.prefix) / "bin") + os.pathsep + env["PATH"]
    health = requests.Session()
    health.trust_env = False
    base = f"http://127.0.0.1:{args.port}"
    evidence = {}
    nv.nvmlInit()
    gpu = nv.nvmlDeviceGetHandleByIndex(0)
    with (args.run / "restart-server.log").open("x") as log:
        start = time.monotonic()
        proc = subprocess.Popen([
            sys.executable, "-u", "scripts/cu128/guard.py", "sgl-omni", "serve",
            "--config", "examples/configs/cu128_asr_5090.yaml", "--model-path",
            str(args.model.resolve()), "--host", "127.0.0.1", "--port", str(args.port),
        ], env=env, stdout=log, stderr=subprocess.STDOUT)
        owned = set()
        try:
            while time.monotonic() - start < 300:
                if proc.poll() is not None:
                    raise RuntimeError("Restart failed")
                try:
                    if health.get(base + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Server health")
            evidence["startup_s"] = time.monotonic() - start
            owned = {p.pid for p in psutil.Process(proc.pid).children(recursive=True)}
            r = health.post(base + "/v1/audio/transcriptions",
                            files={"file": ("empty.wav", b"", "audio/wav")}, timeout=30)
            evidence["empty_audio_status"] = r.status_code
            assert r.status_code == 400
            check = subprocess.run([sys.executable, "scripts/cu128/check_features.py",
                                    "asr", "--port", str(args.port)], env=env,
                                   text=True, stdout=subprocess.PIPE, check=True)
            evidence["feature_probes"] = check.stdout.splitlines()
            evidence["final_health"] = health.get(base + "/health", timeout=5).json()
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=25)
            evidence["remaining_service_gpu_pids"] = [
                p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(gpu) if p.pid in owned]
            with socket.socket() as sock:
                evidence["port_closed"] = sock.connect_ex(("127.0.0.1", args.port)) != 0
            (args.run / "restart.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
    assert evidence["port_closed"] and not evidence["remaining_service_gpu_pids"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--port", type=int, default=18592)
    main(parser.parse_args())
