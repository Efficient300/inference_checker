"""Проход всей матрицы: для каждой комбинации поиск N_max, ограничивающий ресурс, подтверждение.

    python -m bench.run_matrix --config configs/matrix.yaml --measure-s 120          # грубый проход
    python -m bench.run_matrix --config configs/matrix.yaml --run-id <id> --resume    # продолжить после прерывания
    python -m bench.run_matrix --config configs/matrix.yaml --refine-from <id> --measure-s 300
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from bench.config import Combo, confirm_combos, expand_matrix, load_config
from bench.metrics import limiting_resource
from bench.search import find_n_max

RESULTS = Path("results")


class Log:
    def __init__(self, path: Path | None):
        self.path = path

    def __call__(self, *msg):
        line = " ".join(str(m) for m in msg)
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] {line}", flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(f"[{stamp}] {line}\n")


def _pct(x) -> str:
    return "—" if x is None else f"{x:.0f}%"


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str))
    tmp.replace(path)


def select_combos(combos: list[Combo], only: str | None) -> list[Combo]:
    if not only:
        return combos
    keys = [k.strip() for k in only.split(",") if k.strip()]
    return [c for c in combos if any(k == c.scenario or k in c.id for k in keys)]


@contextlib.contextmanager
def mps_daemon(enabled: bool, log):
    if not enabled:
        yield
        return
    log("запуск CUDA MPS")
    subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
    try:
        yield
    finally:
        subprocess.run(["nvidia-cuda-mps-control"], input=b"quit\n", check=False)
        log("CUDA MPS остановлен")


class MatrixRunner:
    def __init__(self, cfg: dict, runner, out_dir: Path, env: dict, log, warmup_s: float, measure_s: float,
                 phase: str = "coarse", refine_dir: Path | None = None):
        self.cfg = cfg
        self.runner = runner
        self.out_dir = out_dir
        self.env = env
        self.log = log
        self.warmup_s = warmup_s
        self.measure_s = measure_s
        self.phase = phase
        self.refine_dir = refine_dir
        self.hw = runner.hw()
        self.simulated = runner.__class__.__name__.startswith("Simulated")

    def path(self, combo: Combo) -> Path:
        return self.out_dir / f"{combo.id}.json"

    def load_state(self, combo: Combo) -> dict:
        p = self.path(combo)
        if p.exists():
            return json.loads(p.read_text())
        return {
            "run_id": f"{self.out_dir.name}/{combo.id}",
            "matrix_run": self.out_dir.name,
            "combo_id": combo.id,
            "combo": combo.to_dict(),
            "env": self.env,
            "config": {k: v for k, v in self.cfg.items() if not k.startswith("_")},
            "hw": self.hw,
            "simulated": self.simulated,
            "started": now_iso(),
            "status": "running",
            "points": [],
        }

    def evaluate_fn(self, combo: Combo, state: dict, measure_s: float, phase: str):
        """evaluate(n) с кэшем уже измеренных точек и одним повтором невалидной точки."""

        def evaluate(n: int) -> dict:
            attempts = [p for p in state["points"]
                        if p["n_cams"] == n and p.get("phase") == phase and p.get("measure_s") == measure_s]
            for p in attempts:
                if p["valid"]:
                    return p
            while len(attempts) < 2:
                p = self.runner.run(combo, n, self.warmup_s, measure_s)
                p.update(phase=phase, attempt=len(attempts) + 1, measured_at=now_iso(),
                         measure_s=measure_s, warmup_s=self.warmup_s)
                state["points"].append(p)
                save_json(self.path(combo), state)
                attempts.append(p)
                verdict = "OK" if p["passed"] else ("НЕВАЛИДНА" if not p["valid"] else "нет")
                self.log(f"    N={n:4d} fps_min={p['fps_min']:.2f} drop={p['drop_rate']:.4f} "
                         f"p95={p['lat_p95']:.0f}мс sm={_pct(p.get('sm_util'))} nvdec={_pct(p.get('nvdec_util'))}"
                         f" → {verdict}"
                         + (f" ({p['invalid_reason']})" if not p["valid"] else "")
                         + (f" [{'; '.join(p['fail_reasons'])}]" if p["valid"] and not p["passed"] else ""))
                if p["valid"]:
                    return p
                if len(attempts) < 2:
                    self.log(f"    точка N={n} невалидна, повтор")
            return attempts[-1]  # дважды невалидна → считается непройденной

        return evaluate

    def run_combo(self, combo: Combo) -> dict:
        state = self.load_state(combo)
        if state.get("status") == "done":
            self.log(f"  {combo.id}: уже готово, N_max={state['n_max']}")
            return state
        s = self.cfg["search"]
        bracket = None
        if self.refine_dir:
            prev = self.refine_dir / f"{combo.id}.json"
            if prev.exists():
                pj = json.loads(prev.read_text())
                if pj.get("n_max") is not None:
                    bracket = (pj["n_max"], pj.get("n_fail"))
        self.log(f"▶ {combo.id}" + (f" (уточнение около {bracket})" if bracket else ""))
        evaluate = self.evaluate_fn(combo, state, self.measure_s, self.phase)
        res = find_n_max(evaluate, s["start"], s["growth"], s["precision"], s["max_n"], bracket)
        boundary = evaluate(res.n_fail) if res.n_fail else evaluate(res.n_max) if res.n_max else None
        name, ratios = limiting_resource(boundary, self.hw)
        state.update(n_max=res.n_max, n_fail=res.n_fail, capped=res.capped, limiting_resource=name,
                     limiting_ratios=ratios, status="done", finished=now_iso())
        if res.capped:
            state["limiting_resource"] = f"не достигнут (max_n={s['max_n']})"
        save_json(self.path(combo), state)
        self.log(f"  ✔ {combo.id}: N_max={res.n_max}, упор: {state['limiting_resource']}")
        return state

    def confirm(self, combo: Combo, duration_s: float) -> None:
        """Подтверждение: длинный прогон на N_max без троттлинга."""
        state = self.load_state(combo)
        if state.get("confirm") or not state.get("n_max"):
            return
        n = state["n_max"]
        self.log(f"▶ подтверждение {combo.id}: N={n}, {duration_s:.0f} с")
        p = self.runner.run(combo, n, self.warmup_s, duration_s)
        p.update(phase="confirm", attempt=1, measured_at=now_iso(), measure_s=duration_s, warmup_s=self.warmup_s)
        state["points"].append(p)
        thermal = p.get("throttle_thermal_frac") or 0.0
        state["confirm"] = {
            "n": n, "duration_s": duration_s, "passed": p["passed"], "valid": p["valid"],
            "throttle_thermal_frac": thermal, "confirmed": bool(p["passed"] and thermal == 0.0),
            "fail_reasons": p["fail_reasons"], "invalid_reason": p["invalid_reason"],
        }
        save_json(self.path(combo), state)
        self.log(f"  {'✔' if state['confirm']['confirmed'] else '✘'} подтверждение {combo.id}: {state['confirm']}")


def collect_env(simulated: bool) -> dict:
    from pipeline.engines import load_manifest

    env = {"collected_at": now_iso(), "argv": sys.argv}
    if not simulated:
        from scripts.env_info import collect

        env.update(collect())
    manifest = load_manifest()
    env["models"] = {k: {"onnx_sha256": v.get("onnx_sha256"), "engine": v.get("path")} for k, v in manifest.items()}
    return env


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/matrix.yaml")
    ap.add_argument("--run-id")
    ap.add_argument("--resume", action="store_true", help="продолжить прогон --run-id с последней точки")
    ap.add_argument("--measure-s", type=float, help="переопределить search.measure_s")
    ap.add_argument("--warmup-s", type=float, help="переопределить search.warmup_s")
    ap.add_argument("--only", help="сценарии или подстроки id комбинаций через запятую (S1,S4_h264_fps10)")
    ap.add_argument("--refine-from", help="run_id грубого прохода: искать только вокруг найденных границ")
    ap.add_argument("--skip-confirm", action="store_true")
    ap.add_argument("--confirm-only", action="store_true")
    ap.add_argument("--simulate", action="store_true", help="имитация без GPU (для проверки обвязки)")
    ap.add_argument("--list", action="store_true", help="показать комбинации и выйти")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    combos = select_combos(expand_matrix(cfg), args.only)
    if args.list:
        for c in combos:
            print(c.id)
        print(f"всего: {len(combos)}")
        return 0

    run_id = args.run_id or f"{cfg['name']}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    out_dir = RESULTS / run_id
    if out_dir.exists() and not args.resume and not args.confirm_only:
        print(f"{out_dir} уже существует: добавьте --resume или другой --run-id", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Log(out_dir / "log.txt")
    shutil.copy(cfg["_path"], out_dir / "config.yaml")
    (out_dir / "config.resolved.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")},
                                                                 allow_unicode=True, sort_keys=False))

    if args.simulate:
        from bench.simulate import SimulatedPointRunner

        runner = SimulatedPointRunner(cfg)
    else:
        from bench.point import PointRunner

        runner = PointRunner(cfg, log=log)
        runner.streams.check_server()

    s = cfg["search"]
    measure_s = args.measure_s or s["measure_s"]
    warmup_s = args.warmup_s if args.warmup_s is not None else s["warmup_s"]
    phase = "refine" if args.refine_from else "coarse"
    env = collect_env(args.simulate)
    mr = MatrixRunner(cfg, runner, out_dir, env, log, warmup_s, measure_s, phase,
                      RESULTS / args.refine_from if args.refine_from else None)
    log(f"прогон {run_id}: {len(combos)} комбинаций, замер {measure_s:.0f} с, прогрев {warmup_s:.0f} с")

    t_start = time.time()
    with mps_daemon(bool(cfg["pipeline"].get("mps")) and not args.simulate, log):
        if not args.confirm_only:
            for i, c in enumerate(combos, 1):
                log(f"[{i}/{len(combos)}]")
                mr.run_combo(c)
        if not args.skip_confirm:
            for c in confirm_combos(cfg, combos):
                mr.confirm(c, float(cfg["confirm"]["duration_s"]))
    log(f"готово за {(time.time() - t_start) / 3600:.2f} ч → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
