import json

import pytest

from bench.config import Combo, load_config
from bench.run_matrix import main as run_matrix
from bench.search import find_n_max
from bench.simulate import SimulatedPointRunner


def capacity_eval(cap, calls=None):
    def ev(n):
        if calls is not None:
            calls.append(n)
        return {"passed": n <= cap}
    return ev


@pytest.mark.parametrize("cap", [0, 1, 5, 7, 8, 13, 50, 99, 279, 500])
def test_finds_capacity_within_precision(cap):
    r = find_n_max(capacity_eval(cap), start=8, growth=1.5, precision=2, max_n=1000)
    assert r.n_max <= cap and cap - r.n_max <= 2
    assert r.n_fail is not None and r.n_fail - r.n_max <= 2


def test_precision_one_is_exact():
    for cap in (3, 17, 64, 200):
        assert find_n_max(capacity_eval(cap), 8, 1.5, 1, 1000).n_max == cap


def test_unreachable_gives_zero():
    calls = []
    r = find_n_max(capacity_eval(0), 8, 1.5, 1, 1000, )
    r2 = find_n_max(capacity_eval(0, calls), 8, 1.5, 1, 1000)
    assert r.n_max == 0 and r2.n_max == 0 and 1 in calls


def test_capped():
    r = find_n_max(capacity_eval(10_000), 8, 1.5, 2, 100)
    assert r.n_max == 100 and r.capped


def test_geometric_steps():
    calls = []
    find_n_max(capacity_eval(100, calls), 8, 1.5, 2, 1000)
    assert calls[:6] == [8, 12, 18, 27, 41, 62]


def test_refine_bracket():
    calls = []
    r = find_n_max(capacity_eval(40, calls), 8, 1.5, 1, 1000, bracket=(38, 42))
    assert r.n_max == 40
    assert len(calls) <= 5
    calls.clear()
    r = find_n_max(capacity_eval(20, calls), 8, 1.5, 1, 1000, bracket=(38, 42))   # граница сдвинулась вниз
    assert r.n_max == 20
    calls.clear()
    r = find_n_max(capacity_eval(70, calls), 8, 1.5, 1, 1000, bracket=(38, 42))   # и вверх
    assert r.n_max == 70


# ---------- оркестратор на имитации ----------

@pytest.fixture
def workdir(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    shutil.copytree(root / "configs", tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    cfg = (tmp_path / "configs/sim.yaml")
    cfg.write_text("name: sim\nbase: matrix.yaml\nscenarios: [S0, S2]\ntarget_fps: [10, 25, 50]\n"
                   "precision: [fp16]\nextra_runs: []\nsearch: {precision: 2, warmup_s: 0, measure_s: 60}\n"
                   "confirm: {scenarios: []}\n")
    return tmp_path


def test_matrix_simulated_end_to_end(workdir):
    assert run_matrix(["--config", "configs/sim.yaml", "--simulate", "--run-id", "r1"]) == 0
    files = sorted((workdir / "results/r1").glob("S*.json"))
    assert len(files) == 6
    res = {json.loads(f.read_text())["combo_id"]: json.loads(f.read_text()) for f in files}
    # недостижимый FPS (50 > 25 у источника) → N_max = 0
    assert res["S0_h264_fps50"]["n_max"] == 0
    assert res["S2_h264_fps50_fp16_det640"]["n_max"] == 0
    cfg = load_config("configs/sim.yaml")
    sim = SimulatedPointRunner(cfg)
    for cid, d in res.items():
        cap = sim.capacity(Combo.from_dict(d["combo"]))
        assert cap - 2 <= d["n_max"] <= cap, cid
        assert d["limiting_resource"]
        for p in d["points"]:
            for field in ("n_cams", "fps_min", "fps_avg", "drop_rate", "lat_p50", "lat_p95", "lat_p99", "queue_slope",
                          "sm_util", "nvdec_util", "vram_gb", "cpu_util", "cpu_by_proc", "net_mbps", "power_w",
                          "temp_c", "sm_clock", "persons_per_frame", "tracks_per_frame", "passed", "invalid_reason"):
                assert field in p, field
    # S0 при 10/25 FPS упирается в NVDEC
    assert res["S0_h264_fps25"]["limiting_resource"] == "nvdec"


def test_resume_after_kill(workdir, monkeypatch):
    import bench.simulate as simmod

    real_init = simmod.SimulatedPointRunner.__init__

    def init_fail(self, cfg, **kw):
        real_init(self, cfg, fail_after=7)

    monkeypatch.setattr(simmod.SimulatedPointRunner, "__init__", init_fail)
    with pytest.raises(KeyboardInterrupt):
        run_matrix(["--config", "configs/sim.yaml", "--simulate", "--run-id", "r2"])
    saved = sum(len(json.loads(f.read_text())["points"]) for f in (workdir / "results/r2").glob("S*.json"))
    assert saved == 7

    calls = []
    real_run = simmod.SimulatedPointRunner.run

    def run_spy(self, combo, n, w, m):
        calls.append((combo.id, n))
        return real_run(self, combo, n, w, m)

    monkeypatch.setattr(simmod.SimulatedPointRunner, "__init__", real_init)
    monkeypatch.setattr(simmod.SimulatedPointRunner, "run", run_spy)
    assert run_matrix(["--config", "configs/sim.yaml", "--simulate", "--run-id", "r2", "--resume"]) == 0
    # уже измеренные точки не перемеряются
    first = json.loads((workdir / "results/r2/S0_h264_fps10.json").read_text())
    measured_before = {(first["combo_id"], p["n_cams"]) for p in first["points"][:7]}
    assert not (measured_before & set(calls[:1]))
    assert assert_all_done(workdir / "results/r2")


def assert_all_done(d):
    return all(json.loads(f.read_text())["status"] == "done" for f in d.glob("S*.json"))


def test_invalid_point_retried_once(workdir, monkeypatch):
    import bench.simulate as simmod

    real_init = simmod.SimulatedPointRunner.__init__

    def init_inv(self, cfg, **kw):
        real_init(self, cfg, invalid_at={("S0_h264_fps10", 8)})

    monkeypatch.setattr(simmod.SimulatedPointRunner, "__init__", init_inv)
    run_matrix(["--config", "configs/sim.yaml", "--simulate", "--run-id", "r3", "--only", "S0_h264_fps10"])
    d = json.loads((workdir / "results/r3/S0_h264_fps10.json").read_text())
    at8 = [p for p in d["points"] if p["n_cams"] == 8]
    assert len(at8) == 2 and not any(p["valid"] for p in at8)
    assert at8[0]["invalid_reason"]
    # дважды невалидная точка считается непройденной → N_max < 8
    assert d["n_max"] < 8
