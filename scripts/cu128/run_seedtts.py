"""Launch the bounded ASR profile, record GPU telemetry, run the sweep and shut down."""

import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import psutil
import pynvml as nv
import requests


def sample(handle, pid):
    try:
        owned = {p.pid for p in psutil.Process(pid).children(recursive=True)}
    except psutil.NoSuchProcess:
        owned = set()
    apps = nv.nvmlDeviceGetComputeRunningProcesses(handle)
    mib = 1024 ** 2
    return [
        round(time.time(), 3),
        nv.nvmlDeviceGetMemoryInfo(handle).free // mib,
        sum(p.usedGpuMemory for p in apps if p.pid in owned) // mib,
        sum(p.usedGpuMemory for p in apps if p.pid not in owned) // mib,
        nv.nvmlDeviceGetUtilizationRates(handle).gpu,
        nv.nvmlDeviceGetPowerUsage(handle) / 1000,
    ]


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(0)
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.prefix) / "bin") + os.pathsep + env["PATH"]
    client_env = env | {"PYTHONPATH": os.pathsep.join([os.getcwd(), str(Path(sys.prefix)/"asr-eval-tools")])}
    endpoint = f"http://127.0.0.1:{args.port}/health"
    health = requests.Session()
    health.trust_env = False
    manifest = {
        "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "machine": platform.platform(), "cpu": subprocess.check_output(["lscpu"], text=True),
        "driver": nv.nvmlSystemGetDriverVersion(), "gpu": nv.nvmlDeviceGetName(handle),
        "profile": Path("examples/configs/cu128_asr_5090.yaml").read_text(),
        "model_revision": "d69410f1c275f2b0fa60cbb9960edfcdb0ae0aec",
        "packages": {p: metadata.version(p) for p in ["torch","sglang","transformers",
                     "flashinfer-python","openai-whisper","rapidfuzz","pyarrow","aiohttp"]},
        "start_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    owned_pids = set()
    with (args.out/"server.log").open("x") as log, (args.out/"gpu.csv").open("x") as meter:
        writer = csv.writer(meter)
        writer.writerow(["unix_s","free_mib","service_mib","other_compute_mib","gpu_pct","board_w"])
        start = time.monotonic()
        server = subprocess.Popen([sys.executable, "-u", "scripts/cu128/guard.py",
            "sgl-omni","serve","--config","examples/configs/cu128_asr_5090.yaml","--model-path",
            str(args.model.resolve()),"--port",str(args.port),"--host","127.0.0.1"], env=env,
            stdout=log, stderr=subprocess.STDOUT)
        benchmark = None
        try:
            while time.monotonic() - start < 600:
                if server.poll() is not None:
                    raise RuntimeError("Server exited during startup")
                writer.writerow(sample(handle, server.pid))
                try:
                    if health.get(endpoint, timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Server startup")
            manifest["warm_cache_startup_seconds"] = time.monotonic()-start
            (args.out/"environment.json").write_text(json.dumps(manifest, indent=2))
            benchmark = subprocess.Popen([sys.executable,"-u","scripts/cu128/seedtts_sweep.py",
                str(args.data),str(args.out),"--port",str(args.port)],env=client_env)
            while benchmark.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError("Service exited during benchmark")
                writer.writerow(sample(handle, server.pid))
                owned_pids.update(p.pid for p in psutil.Process(server.pid).children(recursive=True))
                meter.flush()
                time.sleep(2)
            if benchmark.returncode:
                raise RuntimeError(f"Benchmark exited with code {benchmark.returncode}")
            manifest["final_health"] = health.get(endpoint,timeout=3).json()
            writer.writerow(sample(handle, server.pid))
        finally:
            if benchmark and benchmark.poll() is None:
                benchmark.terminate()
                benchmark.wait(timeout=15)
            if server.poll() is None:
                server.terminate()
            server.wait(timeout=25)
            manifest["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
            manifest["remaining_owned_gpu_pids"] = [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid in owned_pids]
            with socket.socket() as s:
                manifest["port_closed"] = s.connect_ex(("127.0.0.1",args.port)) != 0
            (args.out/"environment.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "data", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--port", type=int, default=18592)
    main(parser.parse_args())
