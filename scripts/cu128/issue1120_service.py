"""Supervise one measured local ASR service through the existing GPU guard."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time

import psutil
import pynvml as nv
import requests
from check_features import units

MIB = 2**20


def dump(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def listening(port):
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main(a):
    out, run = a.out.resolve(), a.name
    cache = out / "private" / "jit"
    if listening(a.port) or shutil.disk_usage(out).free < 8 * 2**30:
        raise RuntimeError("Occupied port or insufficient disk")
    if run == "cold" and cache.exists():
        raise RuntimeError("Cold compiler cache already exists")
    if run != "cold" and not cache.is_dir():
        raise RuntimeError("No private cache to reuse")
    cache.mkdir(exist_ok=True)
    env = os.environ | {"PATH": str(Path(sys.prefix)/"bin") + ":" + os.environ["PATH"]}
    keys = {"FLASHINFER_WORKSPACE_BASE":"flashinfer", "TRITON_CACHE_DIR":"triton",
            "TORCHINDUCTOR_CACHE_DIR":"inductor", "TORCH_EXTENSIONS_DIR":"extensions",
            "CUDA_CACHE_PATH":"driver", "XDG_CACHE_HOME":"xdg", "HF_HOME":"hf",
            "TORCH_HOME":"torch", "NUMBA_CACHE_DIR":"numba"}
    cmd = [sys.executable, "-u", "scripts/cu128/guard.py", "env",
           *(f"{k}={cache/v}" for k,v in keys.items()), "sgl-omni", "serve",
           "--config", "examples/configs/cu128_asr_5090.yaml", "--model-path",
           str(a.model.resolve()), "--port", str(a.port), "--host", "127.0.0.1"]
    start = time.monotonic()
    state = {"launch":cmd, "source":subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True).strip(), "start_unix":time.time()}
    server = None
    owned = {}
    previous = {}
    stop = threading.Event()
    phase = "before"
    nv.nvmlInit()
    dev = nv.nvmlDeviceGetHandleByIndex(0)
    session = requests.Session()
    session.trust_env = False
    endpoint = f"http://127.0.0.1:{a.port}"
    def meter():
        prev_ts = time.monotonic()
        with (out/f"{run}-meter.csv").open("x", buffering=1) as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow("unix_s phase free_mib total_mib service_mib other_compute_mib tree_cpu_pct power_w gpu_pct disk_free_bytes".split())
            while True:
                now = time.monotonic()
                ids, cpu_delta = set(), 0.
                if server is not None:
                    try:
                        parent = psutil.Process(server.pid)
                        tree = [parent] + parent.children(recursive=True)
                    except psutil.NoSuchProcess:
                        tree = []
                    for p in tree:
                        try:
                            ident = (p.pid,p.create_time())
                            cpu = sum(p.cpu_times()[:2])
                            if ident in previous:
                                cpu_delta += max(0., cpu-previous[ident])
                            previous[ident] = cpu
                            owned[ident] = p.name()
                            ids.add(p.pid)
                        except psutil.NoSuchProcess:
                            pass
                mem = nv.nvmlDeviceGetMemoryInfo(dev)
                apps = nv.nvmlDeviceGetComputeRunningProcesses(dev)
                writer.writerow([round(time.time(),3),phase,mem.free//MIB,mem.total//MIB,
                    sum(p.usedGpuMemory for p in apps if p.pid in ids)//MIB,
                    sum(p.usedGpuMemory for p in apps if p.pid not in ids)//MIB,
                    round(100*cpu_delta/max(.001,now-prev_ts),2),
                    nv.nvmlDeviceGetPowerUsage(dev)/1000,
                    nv.nvmlDeviceGetUtilizationRates(dev).gpu,shutil.disk_usage(out).free])
                prev_ts = now
                if stop.wait(.5):
                    return
    monitor = threading.Thread(target=meter)
    monitor.start()
    time.sleep(.7)
    try:
        phase = "startup"
        with (out/f"{run}-server.log").open("x") as log:
            state["spawn_unix"] = time.time()
            t0 = time.monotonic()
            server = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            while time.monotonic()-t0 < 600 and server.poll() is None:
                try:
                    r = session.get(endpoint+"/health",timeout=1)
                    if r.ok and r.json().get("status") == "healthy":
                        break
                except requests.RequestException:
                    pass
                time.sleep(.2)
            else:
                raise RuntimeError("Service failed to become ready")
            state["ready_s"] = time.monotonic()-t0
            phase = "first_request"
            for i in range(2):
                wav = Path("../../runs/sglang-omni/cu128-5090/vivian-en.wav").read_bytes()
                ts = time.monotonic()
                r = session.post(endpoint+"/v1/audio/transcriptions",
                    files={"file":("en.wav",wav,"audio/wav")},data={"language":"en"},timeout=90)
                good = r.status_code == 200 and units(r.json()["text"],"en")==units("The weather is sunny today.","en")
                state[f"probe_{i}"] = dict(correct=good,status=r.status_code,
                        body=r.text,request_s=time.monotonic()-ts,since_spawn_s=time.monotonic()-t0)
                if not good:
                    raise RuntimeError("Startup transcript mismatch")
            service = psutil.Process(server.pid).children()[0]
            state.update(guard_pid=server.pid,service_pid=service.pid,service_birth=service.create_time())
            dump(out/f"{run}-ready.json",state)
            phase = "available"
            while server.poll() is None and not (out/f"{run}.stop").exists():
                if time.monotonic()-start > a.max_seconds:
                    raise TimeoutError("Supervisor wall-clock limit")
                if shutil.disk_usage(out).free < 2*2**30:
                    raise RuntimeError("Disk guard")
                time.sleep(.25)
    finally:
        phase = "shutdown"
        if server is not None:
            if server.poll() is None:
                server.terminate()
            state["guard_rc"] = server.wait(timeout=35)
        time.sleep(1)
        phase = "after"
        time.sleep(1)
        stop.set()
        monitor.join()
        alive = []
        for (pid,birth),name in owned.items():
            try:
                p = psutil.Process(pid)
                if p.create_time()==birth:
                    alive.append([pid,name,p.status()])
            except psutil.NoSuchProcess:
                pass
        state.update(port_closed=not listening(a.port),remaining_tree=alive,
            remaining_service_gpu_pids=[p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(dev)
                if p.pid in {i[0] for i in owned}], observed_tree_pids=sorted({i[0] for i in owned}),
            end_unix=time.time(),cache_bytes=sum(p.stat().st_size for p in cache.rglob("*") if p.is_file()))
        dump(out/f"{run}-exit.json",state)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("model",type=Path)
    p.add_argument("name",choices=("cold","soak","restart"))
    p.add_argument("--port",type=int,default=18593)
    p.add_argument("--max-seconds",type=int,default=2700)
    main(p.parse_args())
