"""Изолированный замер моделей (сценарий «model-only», НЕ полный путь кадра): throughput и латентность
каждого engine по батчам. Используется как теоретическая верхняя граница.

    python -m bench.model_only [--backend trt|ort] [--batches 1,8,32,64,128,256] [--iters 200]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

OUT = Path("results/model_only")


def bench_callable(fn, x, iters: int, warmup: int) -> dict:
    import torch

    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn(x)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    t = np.array(times)
    b = x.shape[0]
    return {"batch": b, "lat_ms_mean": float(t.mean()), "lat_ms_p50": float(np.percentile(t, 50)),
            "lat_ms_p99": float(np.percentile(t, 99)), "throughput_img_s": float(b / (t.mean() / 1000))}


class OrtModel:
    """ONNX Runtime CUDA EP с IO binding на torch-тензорах (контрольная точка к TensorRT)."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
        if "CUDAExecutionProvider" not in self.sess.get_providers():
            raise RuntimeError("CUDAExecutionProvider недоступен")
        self.inp = self.sess.get_inputs()[0].name
        self.outs = [o.name for o in self.sess.get_outputs()]

    def __call__(self, x):
        import torch

        x = x.float().contiguous()
        io = self.sess.io_binding()
        io.bind_input(self.inp, "cuda", 0, np.float32, list(x.shape), x.data_ptr())
        for o in self.outs:
            io.bind_output(o, "cuda", 0)
        torch.cuda.synchronize()
        self.sess.run_with_iobinding(io)
        return io


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["trt", "ort"], default="trt")
    ap.add_argument("--batches", default="1,8,32,64,128,256")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--only", help="подстрока ключа engine")
    args = ap.parse_args(argv)

    import torch

    from pipeline.engines import load_manifest

    manifest = load_manifest()
    if not manifest:
        print("нет engine: python -m models.export")
        return 1
    batches = [int(b) for b in args.batches.split(",")]
    results = []
    seen_onnx = set()
    for key, meta in sorted(manifest.items()):
        if args.only and args.only not in key:
            continue
        inp = meta["input"]
        hw = tuple(inp) if isinstance(inp, list) else (inp, inp)
        if args.backend == "trt":
            from pipeline.trt_runtime import TrtModel

            model = TrtModel(meta["path"])
            stream = torch.cuda.Stream()
            fn = lambda x, m=model, s=stream: m(x, s)  # noqa: E731
            max_b, label = model.max_batch, key
        else:
            if meta["onnx"] + str(hw) in seen_onnx:
                continue
            seen_onnx.add(meta["onnx"] + str(hw))
            fn = OrtModel(meta["onnx"])
            max_b, label = max(batches), f"{meta['model']}_{'x'.join(map(str, hw))}_ort_fp32"
        print(f"[{label}]", flush=True)
        for b in [b for b in batches if b <= max_b]:
            x = torch.rand((b, 3, *hw), device="cuda")
            try:
                r = bench_callable(fn, x, args.iters, args.warmup)
            except Exception as e:  # noqa: BLE001
                print(f"  b={b}: ошибка {e}")
                continue
            r.update(engine=label, model=meta["model"], precision=meta["precision"] if args.backend == "trt" else "fp32",
                     backend=args.backend, input=list(hw))
            results.append(r)
            print(f"  b={b:4d}: {r['throughput_img_s']:10.0f} img/s, {r['lat_ms_mean']:.2f} мс", flush=True)

    from scripts.env_info import collect

    run_id = f"model_only_{args.backend}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"run_id": run_id, "scenario": "model-only", "note": "изолированный замер модели, не полный путь кадра",
            "env": collect(), "results": results}
    (OUT / f"{run_id}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    plot(results, OUT / f"{run_id}.png", run_id)
    print(f"→ {OUT / run_id}.json/.png")
    return 0


def plot(results: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for eng in sorted({r["engine"] for r in results}):
        rs = sorted((r for r in results if r["engine"] == eng), key=lambda r: r["batch"])
        ax.plot([r["batch"] for r in rs], [r["throughput_img_s"] for r in rs], marker="o", label=eng)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("батч")
    ax.set_ylabel("изображений/с")
    ax.set_title(f"model-only: {title}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
