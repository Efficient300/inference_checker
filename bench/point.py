"""Прогон одной точки (комбинация матрицы × N камер): потоки → воркеры → прогрев → замер → метрики."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

from bench.config import Combo
from bench.metrics import Sampler, aggregate_point
from bench.streams import StreamManager
from pipeline.engines import engines_for_combo, load_manifest


class PointError(RuntimeError):
    """Ошибка конфигурации/окружения: прогон нельзя продолжать (в отличие от невалидной точки)."""


class PointRunner:
    def __init__(self, cfg: dict, streams: StreamManager | None = None, log=print):
        self.cfg = cfg
        self.streams = streams or StreamManager(cfg)
        self.log = log
        self.manifest = load_manifest()
        self._worker_pids: list[int] = []

    def hw(self) -> dict:
        s = Sampler(self.cfg["pipeline"].get("gpu", 0))
        speeds = [st.speed for name, st in psutil.net_if_stats().items() if st.isup and st.speed > 0]
        return {"vram_total_gb": s.gpu_mem_total_gb, "link_mbps": max(speeds) if speeds else None,
                "cpu_cores": psutil.cpu_count()}

    def run(self, combo: Combo, n: int, warmup_s: float, measure_s: float) -> dict:
        cfg, pcfg = self.cfg, self.cfg["pipeline"]
        engines = engines_for_combo(cfg["models"], combo.to_dict(), self.manifest) if combo.stages else {}
        k = max(1, min(int(pcfg["workers"]), n))
        groups = [[(i, self.streams.url(i)) for i in range(n) if i % k == w] for w in range(k)]
        w, h = (int(x) for x in cfg["source"]["resolution"].split("x"))

        self.streams.start(n, combo.codec)
        tmp = Path(tempfile.mkdtemp(prefix="point_"))
        ctx = mp.get_context("spawn")
        procs, conns = [], []
        sampler = None
        try:
            from pipeline.worker import worker_main

            for wi, cams in enumerate(groups):
                parent, child = ctx.Pipe()
                args = {
                    "worker_idx": wi, "cams": cams, "combo": combo.to_dict(), "pipeline": pcfg,
                    "engines": engines, "tracker": cfg["models"].get("tracker", {}),
                    "source_fps": cfg["source"]["source_fps"], "frame_hw": (h, w), "out_dir": str(tmp),
                }
                p = ctx.Process(target=worker_main, args=(args, child), name=f"worker{wi}")
                p.start()
                procs.append(p)
                conns.append(parent)
            self._worker_pids = [p.pid for p in procs]
            for wi, c in enumerate(conns):
                if not c.poll(600):
                    raise PointError(f"воркер {wi} не инициализировался за 600 с")
                kind, payload = c.recv()
                if kind != "ready":
                    raise PointError(f"воркер {wi}: {payload}")

            sampler = Sampler(pcfg.get("gpu", 0), 1.0, self._proc_groups)
            sampler.start()
            t0 = time.monotonic() + 1.0
            timing = {"t0": t0, "t_meas": t0 + warmup_s, "t_end": t0 + warmup_s + measure_s}
            for c in conns:
                c.send(timing)
            self.log(f"    N={n}: прогрев {warmup_s:.0f} с + замер {measure_s:.0f} с, воркеров {k}")

            results = []
            deadline = timing["t_end"] + float(pcfg.get("drain_s", 3)) + 120
            for wi, c in enumerate(conns):
                if not c.poll(max(1.0, deadline - time.monotonic())):
                    raise PointError(f"воркер {wi} не завершился")
                kind, payload = c.recv()
                if kind != "done":
                    raise PointError(f"воркер {wi}: {payload}")
                results.append(json.loads(Path(payload).read_text()))
            sampler.stop()
            gpu = sampler.summary(timing["t_meas"], timing["t_end"])
            lat = [np.load(tmp / f"lat_{wi}.npy") for wi in range(k)]
            point = aggregate_point(n, results, np.concatenate(lat) if lat else np.zeros(0), gpu,
                                    combo.to_dict(), float(cfg["source"]["source_fps"]), cfg["thresholds"])
            point["workers"] = k
            point["frame_device"] = next((r.get("frame_device") for r in results if r.get("frame_device")), None)
            point["warmup_s"], point["measure_s"] = warmup_s, measure_s
            point["streams_connected_at_end"] = sum(r.get("connected_at_end", 0) for r in results)
            return point
        finally:
            if sampler is not None:
                sampler.stop()
            for p in procs:
                p.join(timeout=10)
                if p.is_alive():
                    p.kill()
            self._worker_pids = []
            self.streams.stop()
            shutil.rmtree(tmp, ignore_errors=True)

    def _proc_groups(self) -> dict[str, list[int]]:
        return {
            "workers": list(self._worker_pids),
            "streamer": self.streams.pids(),
            "bench": [os.getpid()],
        }
