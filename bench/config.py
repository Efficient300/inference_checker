"""Загрузка конфига матрицы и развёртывание комбинаций."""
from __future__ import annotations

import copy
import glob
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

SCENARIOS = ("S0", "S1", "S2", "S3", "S4")
ID_SCENARIOS = {"S3", "S4"}
ID_MODES = ("every_k_5", "new_tracks_only", "every_k_1")
PRECISIONS = ("fp16", "int8")

STAGES = {
    "S0": (),
    "S1": ("detect",),
    "S2": ("detect", "track"),
    "S3": ("detect", "track", "reid"),
    "S4": ("detect", "track", "reid", "face"),
}

DEFAULTS: dict[str, Any] = {
    "streamer": {
        "mode": "local",
        "rtsp_base": "rtsp://127.0.0.1:8554",
        "api": "http://127.0.0.1:9997",
        "cores": "",
        "ready_timeout_s": 60,
    },
    "extra_runs": [],
    "pipeline": {
        "workers": 1,
        "mps": False,
        "batch_timeout_ms": 10,
        "queue_per_cam": 8,
        "stall_timeout_s": 5,
        "drain_s": 3,
        "id_every_k": 5,
        "face_min_person_h": 0,
        "gpu": 0,
    },
    "search": {"start": 8, "growth": 1.5, "precision": 2, "warmup_s": 60, "measure_s": 300, "max_n": 1000},
    "thresholds": {"min_fps_ratio": 0.95, "max_drop": 0.01, "max_queue_slope": 0.05},
    "latency_report_ms": [200, 500],
    "confirm": {"scenarios": [], "precision": [], "target_fps": [], "duration_s": 1800},
    "calibration": {"frames": 1000, "batch": 8},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if "base" in raw:
        base = load_config(path.parent / raw.pop("base"))
        base.pop("_path", None)
        raw = deep_merge(base, raw)
    cfg = deep_merge(DEFAULTS, raw)
    cfg["_path"] = str(path)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    for key in ("source", "scenarios", "target_fps", "precision", "models"):
        if key not in cfg:
            raise ValueError(f"в конфиге нет ключа '{key}'")
    for s in cfg["scenarios"]:
        if s not in SCENARIOS:
            raise ValueError(f"неизвестный сценарий {s}")
    for p in cfg["precision"]:
        if p not in PRECISIONS:
            raise ValueError(f"неизвестная точность {p}")
    for m in cfg.get("id_mode", []):
        if m not in ID_MODES and not m.startswith("every_k_"):
            raise ValueError(f"неизвестный режим идентификации {m}")
    s = cfg["search"]
    if s["growth"] <= 1:
        raise ValueError("search.growth должен быть > 1")


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


@dataclass(frozen=True)
class Combo:
    scenario: str
    target_fps: float
    codec: str
    precision: str | None = None
    id_mode: str | None = None
    detector_input: int | None = None

    @property
    def id(self) -> str:
        parts = [self.scenario, self.codec, f"fps{_fmt(self.target_fps)}"]
        if self.precision:
            parts.append(self.precision)
        if self.detector_input:
            parts.append(f"det{self.detector_input}")
        if self.id_mode:
            parts.append(self.id_mode)
        return "_".join(parts)

    @property
    def stages(self) -> tuple[str, ...]:
        return STAGES[self.scenario]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Combo":
        return Combo(**d)


def expand_matrix(cfg: dict) -> list[Combo]:
    """Все комбинации матрицы: основная + extra_runs, без дублей, в стабильном порядке."""
    base_codec = cfg["source"]["codec"]
    base_det = cfg["models"]["detector"]["input"]
    combos: list[Combo] = []

    def add(scenarios, codec, det_input):
        for sc in scenarios:
            for fps in cfg["target_fps"]:
                if sc == "S0":
                    combos.append(Combo(sc, fps, codec))
                    continue
                for prec in cfg["precision"]:
                    modes = (cfg.get("id_mode") or ["every_k_5"]) if sc in ID_SCENARIOS else [None]
                    for mode in modes:
                        combos.append(Combo(sc, fps, codec, prec, mode, det_input))

    add(cfg["scenarios"], base_codec, base_det)
    for extra in cfg.get("extra_runs") or []:
        ov = extra.get("override", {})
        scen = [s for s in extra["scenarios"] if s in cfg["scenarios"]]
        add(scen, ov.get("codec", base_codec), ov.get("detector_input", base_det))

    seen, out = set(), []
    for c in combos:
        if c.id not in seen:
            seen.add(c.id)
            out.append(c)
    return out


def confirm_combos(cfg: dict, combos: list[Combo]) -> list[Combo]:
    """Комбинации для подтверждающих длинных прогонов."""
    conf = cfg.get("confirm") or {}
    base_codec = cfg["source"]["codec"]
    base_det = cfg["models"]["detector"]["input"]
    return [
        c for c in combos
        if c.scenario in conf.get("scenarios", [])
        and c.precision in conf.get("precision", [])
        and c.target_fps in conf.get("target_fps", [])
        and c.codec == base_codec
        and (c.detector_input in (None, base_det))
    ]


def video_files(cfg: dict, codec: str) -> list[str]:
    pattern = cfg["source"]["video_files"].format(codec=codec)
    return sorted(glob.glob(pattern))


def id_every_k(mode: str | None, default: int = 5) -> int | None:
    if mode and mode.startswith("every_k_"):
        return int(mode.rsplit("_", 1)[1])
    return None if mode else default
