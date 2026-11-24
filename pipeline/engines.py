"""Реестр собранных TensorRT engine (models/engines/manifest.json)."""
from __future__ import annotations

import json
from pathlib import Path

ENGINES_DIR = Path("models/engines")
MANIFEST = ENGINES_DIR / "manifest.json"


def model_input_tag(cfg: dict) -> str:
    inp = cfg["input"]
    return "x".join(str(i) for i in inp) if isinstance(inp, (list, tuple)) else str(inp)


def engine_key(model: str, input_tag: str, precision: str) -> str:
    return f"{model}_{input_tag}_{precision}"


def load_manifest(path: Path = MANIFEST) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_manifest(manifest: dict, path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    tmp.replace(path)


def engines_for_combo(models_cfg: dict, combo: dict, manifest: dict) -> dict[str, dict]:
    """Какие engine нужны сценарию: {model_name: {path, cfg}}. Бросает ошибку, если чего-то нет."""
    from bench.config import STAGES

    need = []
    stages = STAGES[combo["scenario"]]
    if "detect" in stages:
        need.append("detector")
    if "reid" in stages:
        need.append("reid")
    if "face" in stages:
        need += ["face_det", "face_emb"]
    out = {}
    for name in need:
        cfg = dict(models_cfg[name])
        if name == "detector" and combo.get("detector_input"):
            cfg["input"] = combo["detector_input"]
        key = engine_key(name, model_input_tag(cfg), combo["precision"])
        if key not in manifest:
            raise FileNotFoundError(f"нет engine '{key}' — запустите python -m models.export")
        out[name] = {"path": manifest[key]["path"], "cfg": cfg, "key": key}
    return out
