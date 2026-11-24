"""Версии ПО и параметры железа → JSON (GPU, драйвер, CUDA, TensorRT, NVDEC, CPU).

    python scripts/env_info.py [-o env.json]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys

# Число блоков NVDEC: NVML его не отдаёт, поэтому таблица по документации NVIDIA
NVDEC_ENGINES = {"H200": 7, "H100": 7, "GH200": 7, "A100": 5, "A30": 4, "L40S": 3, "L40": 3, "L4": 4, "A10": 2}


def _try(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _gpu() -> dict:
    import pynvml as n

    n.nvmlInit()
    out = {"driver": _try(lambda: n.nvmlSystemGetDriverVersion())}
    cuda = _try(lambda: n.nvmlSystemGetCudaDriverVersion_v2()) or _try(lambda: n.nvmlSystemGetCudaDriverVersion())
    if cuda:
        out["cuda_driver"] = f"{cuda // 1000}.{(cuda % 1000) // 10}"
    gpus = []
    for i in range(n.nvmlDeviceGetCount()):
        h = n.nvmlDeviceGetHandleByIndex(i)
        name = n.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else name
        mem = n.nvmlDeviceGetMemoryInfo(h)
        nvdec = next((v for k, v in sorted(NVDEC_ENGINES.items(), key=lambda kv: -len(kv[0])) if k in name), None)
        gpus.append({
            "index": i,
            "name": name,
            "uuid": _try(lambda: n.nvmlDeviceGetUUID(h)),
            "memory_total_gb": round(mem.total / 2**30, 2),
            "compute_capability": _try(lambda: ".".join(map(str, n.nvmlDeviceGetCudaComputeCapability(h)))),
            "power_limit_w": _try(lambda: n.nvmlDeviceGetPowerManagementLimit(h) / 1000),
            "sm_clock_max_mhz": _try(lambda: n.nvmlDeviceGetMaxClockInfo(h, n.NVML_CLOCK_SM)),
            "mem_clock_max_mhz": _try(lambda: n.nvmlDeviceGetMaxClockInfo(h, n.NVML_CLOCK_MEM)),
            "nvdec_engines": nvdec,
            "mig_mode": _try(lambda: n.nvmlDeviceGetMigMode(h)[0]),
            "persistence_mode": _try(lambda: n.nvmlDeviceGetPersistenceMode(h)),
        })
    out["gpus"] = gpus
    return out


def _versions() -> dict:
    v = {"python": sys.version.split()[0]}
    for mod, attr in (("tensorrt", "__version__"), ("torch", "__version__"), ("torchvision", "__version__"),
                      ("onnxruntime", "__version__"), ("onnx", "__version__"), ("numpy", "__version__")):
        v[mod] = _try(lambda: getattr(__import__(mod), attr))
    v["PyNvVideoCodec"] = _try(lambda: __import__("importlib.metadata").metadata.version("PyNvVideoCodec"))
    torch = _try(lambda: __import__("torch"))
    if torch is not None:
        v["cuda_runtime_torch"] = torch.version.cuda
        v["cudnn"] = _try(lambda: torch.backends.cudnn.version())
    ff = _try(lambda: subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0])
    v["ffmpeg"] = ff
    return v


def collect() -> dict:
    import psutil

    cpu_model = None
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "gpu": _try(_gpu, {"error": "NVML недоступен"}),
        "software": _versions(),
        "host": {
            "os": platform.platform(),
            "cpu_model": cpu_model,
            "cpu_cores_logical": psutil.cpu_count(),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "in_docker": os.path.exists("/.dockerenv"),
            "image": os.environ.get("BENCH_IMAGE"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output")
    args = ap.parse_args()
    data = json.dumps(collect(), indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(data)
    print(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
