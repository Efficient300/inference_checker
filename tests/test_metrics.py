import numpy as np

from bench.metrics import aggregate_point, limiting_resource, slope, summarize_samples

THR = {"min_fps_ratio": 0.95, "max_drop": 0.01, "max_queue_slope": 0.05}
COMBO = {"scenario": "S2", "target_fps": 10, "codec": "h264", "precision": "fp16", "id_mode": None, "detector_input": 640}


def worker(cams, window=100.0, events=None, errors=None, idx=0):
    return {
        "window_s": window, "t_meas": 0.0, "t_end": window, "worker_idx": idx,
        "cams": {str(c): {"decoded": 2500, "selected": sel, "done": done, "dropped": dr,
                          "persons": done * 20, "tracks": done * 18, "lag": lag,
                          "events": (events or {}).get(c, [])}
                 for c, (sel, done, dr, lag) in cams.items()},
        "counters": {"det_frames": 10}, "busy_frac": {"detect": 0.5, "track": 0.2},
        "queues": {"detect": [(t, 3) for t in range(100)]}, "errors": errors or [],
    }


GPU = {"sm_util": 80.0, "nvdec_util": 40.0, "vram_gb": 30.0, "cpu_util": 20.0,
       "cpu_by_proc": {"workers": 300.0, "workers_max": 90.0}, "net_mbps": 400.0}


def flat_lag():
    return [(float(t), 0.1) for t in range(100)]


def test_pass():
    w = worker({0: (1000, 1000, 0, flat_lag()), 1: (1000, 990, 5, flat_lag())})
    p = aggregate_point(2, [w], np.full(1000, 0.05), GPU, COMBO, 25, THR)
    assert p["valid"] and p["passed"], p["fail_reasons"]
    assert p["fps_min"] == 9.9
    assert abs(p["lat_p95"] - 50) < 1e-6
    assert p["persons_per_frame"] == 20


def test_fail_on_single_slow_camera():
    w = worker({0: (1000, 1000, 0, flat_lag()), 1: (1000, 900, 0, flat_lag())})
    p = aggregate_point(2, [w], np.full(10, 0.05), GPU, COMBO, 25, THR)
    assert p["valid"] and not p["passed"]
    assert any("fps_min" in r for r in p["fail_reasons"])


def test_fail_on_drop_and_growing_lag():
    growing = [(float(t), 0.01 * t) for t in range(100)]   # +0.01 с/с × 25 fps = 0.25 кадра/с
    w = worker({0: (1000, 1000, 20, growing)})
    p = aggregate_point(1, [w], np.full(10, 0.05), GPU, COMBO, 25, THR)
    assert not p["passed"]
    assert any("drop" in r for r in p["fail_reasons"])
    assert any("queue_slope" in r for r in p["fail_reasons"])
    assert abs(p["queue_slope_lag"] - 0.25) < 1e-6


def test_invalid_on_stream_error_in_window_only():
    ev_in = {0: [[50.0, "reconnect", ""]]}
    p = aggregate_point(1, [worker({0: (1000, 1000, 0, flat_lag())}, events=ev_in)], np.zeros(5), GPU, COMBO, 25, THR)
    assert not p["valid"] and not p["passed"] and "reconnect" in p["invalid_reason"]
    ev_warmup = {0: [[-10.0, "reconnect", ""]]}
    p = aggregate_point(1, [worker({0: (1000, 1000, 0, flat_lag())}, events=ev_warmup)], np.zeros(5), GPU, COMBO, 25, THR)
    assert p["valid"]


def test_oom_is_fail_not_invalid():
    w = worker({0: (1000, 1000, 0, flat_lag())}, errors=[[1.0, "detect", "cuda_oom"]])
    p = aggregate_point(1, [w], np.zeros(5), GPU, COMBO, 25, THR)
    assert p["valid"] and not p["passed"] and "cuda_oom" in p["fail_reasons"]
    name, _ = limiting_resource(p, {"vram_total_gb": 141})
    assert name == "vram"


def test_merge_workers():
    w0 = worker({0: (1000, 1000, 0, flat_lag())}, idx=0)
    w1 = worker({1: (1000, 1000, 0, flat_lag())}, idx=1)
    p = aggregate_point(2, [w0, w1], np.zeros(5), GPU, COMBO, 25, THR)
    assert len(p["fps_per_cam"]) == 2
    assert p["model_calls"]["det_frames"] == 20


def test_slope():
    assert slope([(0, 0), (1, 1), (2, 2), (3, 3)]) == 1.0
    assert slope([(0, 5)]) == 0.0


def test_limiting_resource():
    p = {"nvdec_util": 98.0, "sm_util": 60.0, "vram_gb": 20.0, "cpu_util": 30.0, "stage_busy": {"track": 0.4}}
    name, ratios = limiting_resource(p, {"vram_total_gb": 141})
    assert name == "nvdec"
    assert list(ratios)[0] == "nvdec"
    p["stage_busy"]["track"] = 0.99
    assert limiting_resource(p, {})[0] == "stage_track"


def test_summarize_samples():
    ss = [{"t": float(t), "sm_util": 50 + t, "vram_gb": 10 + t, "cpu_util": 10, "net_rx_bytes": t * 1_000_000,
           "clock_reasons": 0x40 if t == 3 else 0, "cpu_by_proc": {"workers": 100, "workers_max": 60}} for t in range(4)]
    s = summarize_samples(ss)
    assert s["sm_util"] == 51.5 and s["vram_gb"] == 13
    assert s["throttle_thermal_frac"] == 0.25
    assert abs(s["net_mbps"] - 8.0) < 1e-9
