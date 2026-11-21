"""Run one local service on GPU 0; stop its process group if GPU headroom drops."""

import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def free_mib():
    return int(subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=5).strip())


def main():
    envdir = Path(sys.prefix)
    lock = (envdir / ".cu128-gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = minimum = free_mib()
    if initial < 14 * 1024:
        sys.exit(f"GPU 0 has only {initial} MiB free; refusing to start")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["FLASHINFER_WORKSPACE_BASE"] = str(envdir)
    for key, name in [("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor")]:
        os.environ[key] = str(envdir / ".cache" / name)
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    proc = subprocess.Popen(sys.argv[1:], start_new_session=True)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        while proc.poll() is None:
            minimum = min(minimum, free_mib())
            if minimum < 4096:
                sys.exit("GPU headroom below 4096 MiB")
            time.sleep(0.5)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            for _ in range(60):
                proc.poll()
                try:
                    os.killpg(proc.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        print(f"GPU 0 free MiB: start={initial}, sampled_min={minimum}, end={free_mib()}", flush=True)


if __name__ == "__main__":
    main()
