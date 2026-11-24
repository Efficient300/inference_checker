"""Метрики точки: сэмплер NVML/psutil, агрегация по воркерам, критерии прохождения, ограничивающий ресурс."""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np
import psutil

# Биты причин снижения частот NVML (nvmlClocksEventReason*)
_THROTTLE_POWER_CAP = 0x4
_THROTTLE_THERMAL = 0x8 | 0x20 | 0x40 | 0x80   # HW slowdown, SW thermal, HW thermal, power brake


class Sampler(threading.Thread):
    """Раз в interval секунд: GPU (NVML) и CPU/сеть (psutil). Группы процессов задаются функцией."""

    def __init__(self, gpu_index: int = 0, interval: float = 1.0,
                 proc_groups: Callable[[], dict[str, list[int]]] | None = None):
        super().__init__(name="sampler", daemon=True)
        self.interval = interval
        self.proc_groups = proc_groups or (lambda: {})
        self.samples: list[dict] = []
        self._halt = threading.Event()
        self._procs: dict[int, psutil.Process] = {}
        self.nvml = None
        self.gpu_mem_total_gb = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.gpu_mem_total_gb = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total / 2**30
        except Exception:  # noqa: BLE001 — без GPU сэмплер пишет только CPU
            self.nvml = None

    def _gpu(self) -> dict:
        if self.nvml is None:
            return {}
        n, h, out = self.nvml, self.handle, {}

        def safe(key, fn):
            try:
                out[key] = fn()
            except Exception:  # noqa: BLE001
                pass

        safe("sm_util", lambda: n.nvmlDeviceGetUtilizationRates(h).gpu)
        safe("nvdec_util", lambda: n.nvmlDeviceGetDecoderUtilization(h)[0])
        safe("vram_gb", lambda: n.nvmlDeviceGetMemoryInfo(h).used / 2**30)
        safe("power_w", lambda: n.nvmlDeviceGetPowerUsage(h) / 1000.0)
        safe("temp_c", lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU))
        safe("sm_clock", lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM))

        def reasons():
            fn = getattr(n, "nvmlDeviceGetCurrentClocksEventReasons", None) or n.nvmlDeviceGetCurrentClocksThrottleReasons
            return int(fn(h))

        safe("clock_reasons", reasons)
        return out

    def _proc_cpu(self, pids: list[int]) -> tuple[float, float]:
        total, mx = 0.0, 0.0
        for pid in pids:
            p = self._procs.get(pid)
            try:
                if p is None:
                    p = self._procs[pid] = psutil.Process(pid)
                    p.cpu_percent(None)
                    continue
                v = p.cpu_percent(None)
                for ch in p.children(recursive=True):
                    if ch.pid not in self._procs:
                        self._procs[ch.pid] = ch
                        ch.cpu_percent(None)
                    else:
                        v += self._procs[ch.pid].cpu_percent(None)
                total += v
                mx = max(mx, v)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._procs.pop(pid, None)
        return total, mx

    def sample(self) -> dict:
        s = {"t": time.monotonic(), "cpu_util": psutil.cpu_percent(None)}
        net = psutil.net_io_counters()
        s["net_rx_bytes"] = net.bytes_recv
        groups = {}
        for name, pids in self.proc_groups().items():
            total, mx = self._proc_cpu(pids)
            groups[name] = total
            groups[f"{name}_max"] = mx
        s["cpu_by_proc"] = groups
        s.update(self._gpu())
        return s

    def run(self):
        psutil.cpu_percent(None)
        while not self._halt.is_set():
            self.samples.append(self.sample())
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()

    def summary(self, t0: float, t1: float) -> dict:
        ss = [s for s in self.samples if t0 <= s["t"] < t1]
        return summarize_samples(ss)


def summarize_samples(ss: list[dict]) -> dict:
    if not ss:
        return {"n_samples": 0}

    def mean(key):
        v = [s[key] for s in ss if key in s]
        return float(np.mean(v)) if v else None

    def vmax(key):
        v = [s[key] for s in ss if key in s]
        return float(np.max(v)) if v else None

    out = {
        "n_samples": len(ss),
        "sm_util": mean("sm_util"), "sm_util_max": vmax("sm_util"),
        "nvdec_util": mean("nvdec_util"), "nvdec_util_max": vmax("nvdec_util"),
        "vram_gb": vmax("vram_gb"),
        "power_w": mean("power_w"), "temp_c": mean("temp_c"), "temp_c_max": vmax("temp_c"),
        "sm_clock": mean("sm_clock"), "sm_clock_min": (min(s["sm_clock"] for s in ss if "sm_clock" in s)
                                                        if any("sm_clock" in s for s in ss) else None),
        "cpu_util": mean("cpu_util"),
    }
    rs = [s["clock_reasons"] for s in ss if "clock_reasons" in s]
    out["throttle_thermal_frac"] = float(np.mean([(r & _THROTTLE_THERMAL) != 0 for r in rs])) if rs else None
    out["power_cap_frac"] = float(np.mean([(r & _THROTTLE_POWER_CAP) != 0 for r in rs])) if rs else None
    if len(ss) >= 2:
        dt = ss[-1]["t"] - ss[0]["t"]
        out["net_mbps"] = (ss[-1]["net_rx_bytes"] - ss[0]["net_rx_bytes"]) * 8 / 1e6 / dt if dt > 0 else None
    else:
        out["net_mbps"] = None
    keys = set().union(*(s.get("cpu_by_proc", {}).keys() for s in ss))
    out["cpu_by_proc"] = {
        k: float((np.max if k.endswith("_max") else np.mean)([s["cpu_by_proc"].get(k, 0.0) for s in ss]))
        for k in sorted(keys)
    }
    return out


def slope(points: list[tuple[float, float]]) -> float:
    """Наклон МНК y(t) в единицах y за секунду; 0 при нехватке точек."""
    if len(points) < 3:
        return 0.0
    t = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points], dtype=float)
    if np.ptp(t) == 0:
        return 0.0
    return float(np.polyfit(t - t[0], y, 1)[0])


def aggregate_point(n: int, workers: list[dict], latencies: np.ndarray, gpu: dict, combo: dict,
                    source_fps: float, thresholds: dict) -> dict:
    """Сводит сырые данные воркеров и сэмплера в точку результата и проверяет критерии прохождения."""
    cams: dict[str, dict] = {}
    counters: dict[str, int] = {}
    busy: dict[str, float] = {}
    errors, queue_slopes = [], {}
    window = None
    for w in workers:
        window = w["window_s"]
        cams.update(w["cams"])
        for k, v in w["counters"].items():
            counters[k] = counters.get(k, 0) + v
        for k, v in w["busy_frac"].items():
            busy[k] = max(busy.get(k, 0.0), v)
        errors += [{"stage": e[1], "msg": e[2]} for e in w["errors"]]
        for qn, samples in w["queues"].items():
            queue_slopes[f"{qn}@w{w['worker_idx']}"] = slope([tuple(s) for s in samples])
        t_meas, t_end = w["t_meas"], w["t_end"]

    target = float(combo["target_fps"])
    invalid = []
    fps, drops, dec_fps, lag_slopes = [], [], [], []
    persons = tracks = done_total = 0
    for cid, c in sorted(cams.items(), key=lambda kv: int(kv[0])):
        fps.append(c["done"] / window)
        dec_fps.append(c["decoded"] / window)
        drops.append(c["dropped"] / c["selected"] if c["selected"] else 1.0)
        lag_slopes.append(slope([tuple(x) for x in c["lag"]]) * source_fps)
        persons += c["persons"]
        tracks += c["tracks"]
        done_total += c["done"]
        bad = [e for e in c["events"] if t_meas <= e[0] < t_end and e[1] in ("error", "eof", "reconnect", "stall")]
        if bad:
            invalid.append(f"cam{cid}: {bad[0][1]} {bad[0][2]}".strip())
        elif c["decoded"] == 0:
            invalid.append(f"cam{cid}: нет декодированных кадров")

    oom = any(e["msg"] == "cuda_oom" for e in errors)
    other_err = [e for e in errors if e["msg"] != "cuda_oom"]
    if other_err and not oom:
        invalid.append(f"ошибка стадии {other_err[0]['stage']}: {other_err[0]['msg']}")

    lat_ms = latencies * 1000.0 if len(latencies) else np.array([np.nan])
    q_internal = max(queue_slopes.values(), default=0.0)
    q_lag = max(lag_slopes, default=0.0)
    queue_slope = max(q_internal, q_lag)

    point = {
        "n_cams": n,
        "fps_min": float(min(fps)) if fps else 0.0,
        "fps_avg": float(np.mean(fps)) if fps else 0.0,
        "decode_fps_min": float(min(dec_fps)) if dec_fps else 0.0,
        "drop_rate": float(max(drops)) if drops else 1.0,
        "drop_rate_avg": float(np.mean(drops)) if drops else 1.0,
        "lat_p50": float(np.percentile(lat_ms, 50)),
        "lat_p95": float(np.percentile(lat_ms, 95)),
        "lat_p99": float(np.percentile(lat_ms, 99)),
        "queue_slope": float(queue_slope),
        "queue_slope_lag": float(q_lag),
        "queue_slope_internal": float(q_internal),
        "persons_per_frame": persons / done_total if done_total else 0.0,
        "tracks_per_frame": tracks / done_total if done_total else 0.0,
        "model_calls": counters,
        "stage_busy": busy,
        "errors": errors[:20],
        "window_s": window,
    }
    point.update({k: gpu.get(k) for k in (
        "sm_util", "sm_util_max", "nvdec_util", "nvdec_util_max", "vram_gb", "cpu_util", "cpu_by_proc",
        "net_mbps", "power_w", "temp_c", "temp_c_max", "sm_clock", "sm_clock_min",
        "throttle_thermal_frac", "power_cap_frac")})
    point["fps_per_cam"] = [round(f, 3) for f in fps]

    point["invalid_reason"] = "; ".join(invalid[:5]) if invalid else None
    point["valid"] = not invalid
    point["fail_reasons"] = check_pass(point, target, thresholds, oom)
    point["passed"] = point["valid"] and not point["fail_reasons"]
    return point


def check_pass(point: dict, target_fps: float, thr: dict, oom: bool = False) -> list[str]:
    """Критерии прохождения: FPS каждой камеры ≥ 95% цели, пропуски ≤ 1% на камеру, очереди не растут."""
    reasons = []
    if oom:
        reasons.append("cuda_oom")
    if point["fps_min"] < thr["min_fps_ratio"] * target_fps:
        reasons.append(f"fps_min {point['fps_min']:.2f} < {thr['min_fps_ratio'] * target_fps:.2f}")
    if point["drop_rate"] > thr["max_drop"]:
        reasons.append(f"drop {point['drop_rate']:.4f} > {thr['max_drop']}")
    if point["queue_slope"] > thr["max_queue_slope"]:
        reasons.append(f"queue_slope {point['queue_slope']:.3f} > {thr['max_queue_slope']}")
    return reasons


def limiting_resource(point: dict | None, hw: dict) -> tuple[str | None, dict]:
    """Ресурс, ближайший к насыщению на точке N_max+шаг. Возвращает (имя, доли загрузки)."""
    if not point:
        return None, {}
    r: dict[str, float] = {}
    if point.get("nvdec_util") is not None:
        r["nvdec"] = point["nvdec_util"] / 100
    if point.get("sm_util") is not None:
        r["gpu_compute"] = point["sm_util"] / 100
    if point.get("vram_gb") is not None and hw.get("vram_total_gb"):
        r["vram"] = point["vram_gb"] / hw["vram_total_gb"]
    if "cuda_oom" in (point.get("fail_reasons") or []):
        r["vram"] = 1.0
    if point.get("cpu_util") is not None:
        r["cpu_total"] = point["cpu_util"] / 100
    proc = point.get("cpu_by_proc") or {}
    if "workers_max" in proc:
        # один процесс Python ≈ одно ядро под GIL
        r["cpu_worker_gil"] = proc["workers_max"] / 100
    if point.get("net_mbps") and hw.get("link_mbps"):
        r["network"] = point["net_mbps"] / hw["link_mbps"]
    for stage, frac in (point.get("stage_busy") or {}).items():
        r[f"stage_{stage}"] = frac
    if not r:
        return None, {}
    name = max(r, key=r.get)
    return name, {k: round(v, 4) for k, v in sorted(r.items(), key=lambda kv: -kv[1])}
