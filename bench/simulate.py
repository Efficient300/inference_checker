"""Имитация прогона точки по модели ёмкости — для тестов поиска/возобновления/отчёта без GPU.

Результаты имитации помечаются simulated=true и не могут попасть в итоговый отчёт как реальные
(синтетика не допускается в итоговых результатах).
"""
from __future__ import annotations

import numpy as np

from bench.config import Combo
from bench.metrics import aggregate_point

# Условные «стоимости» одной камеры при 25 FPS в долях ресурса — только для имитации
_COST = {"S0": 0.004, "S1": 0.006, "S2": 0.008, "S3": 0.012, "S4": 0.02}


class SimulatedPointRunner:
    def __init__(self, cfg: dict, invalid_at: set[tuple[str, int]] | None = None, fail_after: int | None = None):
        self.cfg = cfg
        self.calls: list[tuple[str, int]] = []
        self.invalid_at = invalid_at or set()
        self.fail_after = fail_after          # имитация kill: исключение после стольких точек

    def hw(self) -> dict:
        return {"vram_total_gb": 141.0, "link_mbps": 100000, "cpu_cores": 64}

    def capacity(self, combo: Combo) -> int:
        """Сколько камер «выдерживается» в имитации."""
        if combo.target_fps > self.cfg["source"]["source_fps"]:
            return 0
        if combo.scenario == "S0":
            return 280
        cost = _COST[combo.scenario] * combo.target_fps / 25.0
        if combo.precision == "int8":
            cost *= 0.7
        if combo.id_mode == "new_tracks_only":
            cost *= 0.85
        if combo.detector_input and combo.detector_input > 640:
            cost *= 3
        return min(280, int(1.0 / cost))

    def run(self, combo: Combo, n: int, warmup_s: float, measure_s: float) -> dict:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise KeyboardInterrupt("имитация прерывания")
        self.calls.append((combo.id, n))
        cap = self.capacity(combo)
        load = n / max(cap, 1e-9) if cap else 10.0
        ok = n <= cap
        fps = combo.target_fps if ok else combo.target_fps * min(1.0, cap / n) * 0.9
        cams = {}
        for i in range(n):
            sel = int(fps * measure_s)
            cams[str(i)] = {
                "decoded": int(min(25, combo.target_fps * 2.5) * measure_s), "selected": int(combo.target_fps * measure_s),
                "done": sel, "dropped": 0 if ok else int(0.05 * sel), "persons": sel * 30, "tracks": sel * 25,
                "lag": [(t, 0.0 if ok else t * 0.01) for t in range(int(measure_s))],
                "events": [[1.0, "error", "simulated"]] if (combo.id, n) in self.invalid_at else [],
            }
        worker = {"window_s": measure_s, "t_meas": 0.0, "t_end": float(measure_s), "cams": cams,
                  "counters": {"det_frames": n}, "busy_frac": {"detect": min(1.0, load * 0.6)},
                  "queues": {}, "errors": [], "worker_idx": 0}
        rng = np.random.default_rng(n)
        lat = rng.gamma(4.0, 0.02 * (1 + load ** 3), size=max(1, n * 10))
        gpu = {"sm_util": min(100.0, 100 * load * 0.9), "nvdec_util": min(100.0, 100 * n / 280),
               "vram_gb": 10 + n * 0.1, "cpu_util": min(100.0, 20 * load), "cpu_by_proc": {"workers_max": 50.0},
               "net_mbps": n * 4.0, "power_w": 300 + 300 * min(load, 1), "temp_c": 55.0, "sm_clock": 1980.0,
               "throttle_thermal_frac": 0.0, "power_cap_frac": 0.0}
        point = aggregate_point(n, [worker], lat, gpu, combo.to_dict(), self.cfg["source"]["source_fps"],
                                self.cfg["thresholds"])
        point["simulated"] = True
        return point
