"""Экспорт моделей из конфига: исходник → ONNX (динамический батч) → TensorRT FP16/INT8.

    python -m models.export --config configs/matrix.yaml [--models detector,reid] [--precision fp16,int8] [--force]

Итог — models/engines/*.engine и models/engines/manifest.json (пути, sha256 ONNX, версия TensorRT).
Engine привязан к версии TensorRT и GPU: собирайте в том же контейнере и на той же машине, где мерите.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from functools import partial
from pathlib import Path

MODELS = ("detector", "reid", "face_det", "face_emb")
CACHE = Path("models/cache")
CALIB_DIR = Path("models/calibration")
INSIGHTFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/{pack}.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- ONNX ----------------

def make_dynamic_batch(path: Path) -> None:
    """Делает первую размерность входа символьной (batch); размерности выходов снимаются."""
    import onnx

    m = onnx.load(str(path))
    inits = {i.name for i in m.graph.initializer}
    changed = False
    for inp in m.graph.input:
        if inp.name in inits:
            continue
        d0 = inp.type.tensor_type.shape.dim[0]
        if d0.dim_param != "batch":
            d0.dim_param = "batch"
            changed = True
    for j, out in enumerate(m.graph.output):
        for k, d in enumerate(out.type.tensor_type.shape.dim):
            if d.HasField("dim_value"):
                d.dim_param = f"out{j}_{k}"
                changed = True
    if changed:
        onnx.save(m, str(path))


def onnx_from_ultralytics(weights: str, dst: Path, imgsz: int) -> None:
    from ultralytics import YOLO

    CACHE.mkdir(parents=True, exist_ok=True)
    w = CACHE / weights
    model = YOLO(str(w) if w.exists() else weights)
    out = model.export(format="onnx", dynamic=True, simplify=False, opset=17, imgsz=imgsz, half=False)
    shutil.move(out, dst)
    src_pt = Path(weights)
    if src_pt.exists() and not w.exists():
        shutil.move(str(src_pt), w)


def onnx_from_osnet(variant: str, dst: Path, hw: tuple[int, int], weights: str | None) -> None:
    import torch

    from models.osnet import build

    if not weights:
        print("  ВНИМАНИЕ: веса OSNet не заданы (models.reid.weights) — случайная инициализация. "
              "На скорость не влияет, эмбеддинги бессмысленны.", flush=True)
    m = build(variant, weights)
    x = torch.randn(1, 3, *hw)
    kw = dict(input_names=["input"], output_names=["embedding"],
              dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(m, x, str(dst), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(m, x, str(dst), **kw)


def onnx_from_insightface(spec: str, dst: Path) -> None:
    pack, member = spec.split("/", 1)
    CACHE.mkdir(parents=True, exist_ok=True)
    z = CACHE / f"{pack}.zip"
    if not z.exists():
        url = INSIGHTFACE_URL.format(pack=pack)
        print(f"  скачивание {url}", flush=True)
        urllib.request.urlretrieve(url, z)
    with zipfile.ZipFile(z) as f:
        names = [n for n in f.namelist() if n.endswith("/" + member) or n == member]
        if not names:
            raise FileNotFoundError(f"{member} нет в {z}: {f.namelist()}")
        with f.open(names[0]) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out)
    make_dynamic_batch(dst)


def ensure_onnx(name: str, cfg: dict, force: bool = False) -> Path:
    dst = Path(cfg["onnx"])
    if dst.exists() and not force:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = cfg.get("source", "")
    kind, _, arg = src.partition(":")
    print(f"[{name}] ONNX из {src} → {dst}", flush=True)
    if kind == "ultralytics":
        onnx_from_ultralytics(arg, dst, int(cfg["input"]))
    elif kind == "osnet":
        h, w = cfg["input"]
        onnx_from_osnet(arg or "x1_0", dst, (int(h), int(w)), cfg.get("weights"))
    elif kind == "insightface":
        onnx_from_insightface(arg, dst)
    elif kind == "file":
        shutil.copy(arg, dst)
        make_dynamic_batch(dst)
    elif not dst.exists():
        raise FileNotFoundError(f"[{name}] нет {dst} и не задан source")
    return dst


# ---------------- TensorRT ----------------

def input_chw(name: str, cfg: dict) -> tuple[int, int, int]:
    inp = cfg["input"]
    if isinstance(inp, (list, tuple)):
        return (3, int(inp[0]), int(inp[1]))
    return (3, int(inp), int(inp))


def build_engine(onnx_path: Path, engine_path: Path, chw, max_batch: int, precision: str,
                 calibrator=None, calib_bs: int = 8, workspace_gb: int = 8) -> None:
    import tensorrt as trt

    from pipeline.trt_runtime import LOGGER

    builder = trt.Builder(LOGGER)
    flags = 0
    if int(trt.__version__.split(".")[0]) < 10:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ошибка разбора {onnx_path}: {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(name, (1, *chw), (max_batch, *chw), (max_batch, *chw))
    config.add_optimization_profile(profile)
    config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
        cp = builder.create_optimization_profile()
        cp.set_shape(name, (calib_bs, *chw), (calib_bs, *chw), (calib_bs, *chw))
        config.set_calibration_profile(cp)
    data = builder.build_serialized_network(network, config)
    if data is None:
        raise RuntimeError(f"не удалось собрать {engine_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(data))


def calib_factory(name: str, cfg: dict, models_cfg: dict, frames_fn, built: dict, n: int, bs: int):
    """Фабрика итератора калибровочных батчей для модели name."""
    from models import calib
    from pipeline.detector import YoloDetector
    from pipeline.reid import NORMS
    from pipeline.trt_runtime import TrtModel

    def det_fp16():
        dcfg = models_cfg["detector"]
        return YoloDetector(dcfg, TrtModel(built[f"detector_{dcfg['input']}_fp16"]))

    if name == "detector":
        return lambda: calib.batched(calib.detector_batches(frames_fn(), int(cfg["input"])), bs, n)
    if name == "reid":
        mean, std = NORMS.get(cfg.get("type", "osnet"), NORMS["osnet"])
        hw = tuple(int(x) for x in cfg["input"])
        return lambda: calib.batched(calib.reid_batches(frames_fn(), det_fp16(), hw, mean, std), bs, n)
    if name == "face_det":
        return lambda: calib.batched(calib.face_det_batches(frames_fn(), det_fp16(), int(cfg["input"])), bs, n)
    if name == "face_emb":
        fd = models_cfg["face_det"]

        def it():
            face_det = TrtModel(built[f"face_det_{fd['input']}_fp16"])
            return calib.batched(calib.face_emb_batches(frames_fn(), det_fp16(), face_det, int(fd["input"]),
                                                        int(cfg["input"]), float(fd.get("conf", 0.5))), bs, n)
        return it
    raise KeyError(name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/matrix.yaml")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--precision", help="по умолчанию — из конфига")
    ap.add_argument("--force", action="store_true", help="пересобрать engine и заново получить ONNX")
    ap.add_argument("--onnx-only", action="store_true")
    ap.add_argument("--workspace-gb", type=int, default=8)
    args = ap.parse_args(argv)

    from bench.config import expand_matrix, load_config, video_files
    from pipeline.engines import ENGINES_DIR, engine_key, load_manifest, model_input_tag, save_manifest

    cfg = load_config(args.config)
    mcfg = cfg["models"]
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    precisions = [p.strip() for p in (args.precision or ",".join(cfg["precision"])).split(",")]
    det_inputs = sorted({c.detector_input for c in expand_matrix(cfg) if c.detector_input} | {mcfg["detector"]["input"]})

    onnx = {n: ensure_onnx(n, mcfg[n], args.force) for n in names}
    if args.onnx_only:
        return 0

    import tensorrt as trt

    w, h = (int(x) for x in cfg["source"]["resolution"].split("x"))
    files = video_files(cfg, cfg["source"]["codec"])
    n_calib = int(cfg["calibration"]["frames"])
    bs = int(cfg["calibration"]["batch"])
    from models.calib import iter_frames

    frames_fn = partial(iter_frames, files, n_calib, w, h)

    manifest = load_manifest()
    built = {k: v["path"] for k, v in manifest.items()}
    gpu = None
    try:
        import pynvml

        pynvml.nvmlInit()
        gpu = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        gpu = gpu.decode() if isinstance(gpu, bytes) else gpu
    except Exception:  # noqa: BLE001
        pass

    # порядок важен: калибровка моделей кропов использует FP16-детектор и FP16-детектор лиц
    order = sorted(precisions, key=lambda p: p != "fp16")
    if "int8" in order and "fp16" not in order:
        order = ["fp16"] + order
    for prec in order:
        for name in [m for m in MODELS if m in names]:
            variants = det_inputs if name == "detector" else [mcfg[name]["input"]]
            for inp in variants:
                mc = dict(mcfg[name], input=inp)
                tag = model_input_tag(mc)
                key = engine_key(name, tag, prec)
                mb = int(mc["max_batch"])
                path = ENGINES_DIR / f"{key}_b{mb}.engine"
                digest = sha256(onnx[name])
                old = manifest.get(key)
                if (not args.force and path.exists() and old and old.get("onnx_sha256") == digest
                        and old.get("max_batch") == mb and old.get("trt_version") == trt.__version__):
                    print(f"[{key}] актуален, пропуск", flush=True)
                    built[key] = str(path)
                    continue
                calibrator = None
                if prec == "int8":
                    from models.calib import Calibrator

                    if name != "detector" and f"detector_{mcfg['detector']['input']}_fp16" not in built:
                        raise RuntimeError("для INT8-калибровки моделей кропов нужен FP16-детектор (добавьте detector в --models)")
                    cache = CALIB_DIR / f"{name}_{tag}_{digest[:12]}.cache"
                    calibrator = Calibrator(calib_factory(name, mc, mcfg, frames_fn, built, n_calib, bs), bs, cache)
                print(f"[{key}] сборка (max_batch={mb}) …", flush=True)
                t = dt.datetime.now()
                build_engine(onnx[name], path, input_chw(name, mc), mb, prec, calibrator, bs, args.workspace_gb)
                built[key] = str(path)
                manifest[key] = {
                    "path": str(path), "model": name, "input": inp, "precision": prec, "max_batch": mb,
                    "onnx": str(onnx[name]), "onnx_sha256": digest, "trt_version": trt.__version__, "gpu": gpu,
                    "built_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "calib_samples": calibrator.n if calibrator else None,
                }
                save_manifest(manifest)
                print(f"[{key}] готово за {(dt.datetime.now() - t).seconds} с"
                      + (f", калибровка на {calibrator.n} образцах" if calibrator and calibrator.n else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
