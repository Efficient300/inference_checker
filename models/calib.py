"""INT8-калибровка: источники данных из тестового контента и калибратор TensorRT.

Кадры берутся равномерно из клипов data/videos (по умолчанию 1000). Для моделей кропов
(re-ID, лица) кропы получаются реальным пайплайном: FP16-детектор → боксы людей → кропы.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
import tensorrt as trt

from pipeline.preprocess import align_faces, crop_resize, letterbox, normalize, upper_squares


def iter_frames(files: list[str], n: int, width: int, height: int) -> Iterator[torch.Tensor]:
    """n кадров RGB [3,H,W] uint8 на GPU, равномерно по всем клипам (декодирование ffmpeg на CPU)."""
    if not files:
        raise FileNotFoundError("нет клипов для калибровки (scripts/prepare_videos.py)")
    per_file = math.ceil(n / len(files))
    produced = 0
    frame_bytes = width * height * 3
    for f in files:
        step = max(1, int(300 * 25 / per_file))
        cmd = ["ffmpeg", "-v", "error", "-i", f, "-vf", f"select='not(mod(n\\,{step}))',scale={width}:{height}",
               "-vsync", "vfr", "-frames:v", str(per_file), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
            while produced < n:
                buf = p.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                arr = np.frombuffer(buf, np.uint8).reshape(height, width, 3)
                yield torch.from_numpy(arr.copy()).cuda().permute(2, 0, 1).contiguous()
                produced += 1
            p.kill()
        if produced >= n:
            return


def batched(it: Iterator[torch.Tensor], bs: int, limit: int) -> Iterator[torch.Tensor]:
    """Склеивает поток тензоров [k,...] в батчи ровно по bs (хвост отбрасывается), всего не более limit."""
    buf, total = [], 0
    for x in it:
        buf.append(x)
        if sum(len(b) for b in buf) >= bs:
            cat = torch.cat(buf)
            while len(cat) >= bs and total + bs <= limit:
                yield cat[:bs]
                cat = cat[bs:]
                total += bs
            if total + bs > limit:
                return
            buf = [cat] if len(cat) else []


def detector_batches(frames: Iterator[torch.Tensor], size: int) -> Iterator[torch.Tensor]:
    for f in frames:
        x, _, _ = letterbox(f[None], size)
        yield (x / 255.0).float()


def person_boxes(frames: Iterator[torch.Tensor], detector) -> Iterator[tuple[torch.Tensor, np.ndarray]]:
    for f in frames:
        dets = detector(f[None], torch.cuda.current_stream())[0]
        if len(dets):
            yield f, dets[:, :4]


def reid_batches(frames, detector, hw, mean, std) -> Iterator[torch.Tensor]:
    for f, boxes in person_boxes(frames, detector):
        rois = torch.from_numpy(np.concatenate([np.zeros((len(boxes), 1), np.float32), boxes], 1)).cuda()
        yield normalize(crop_resize(f[None], rois, hw), mean, std).float()


def head_crops(frames, detector, size) -> Iterator[torch.Tensor]:
    for f, boxes in person_boxes(frames, detector):
        reg = upper_squares(boxes, f.shape[-1], f.shape[-2])
        rois = torch.from_numpy(np.concatenate([np.zeros((len(reg), 1), np.float32), reg], 1)).cuda()
        yield crop_resize(f[None], rois, (size, size))


def face_det_batches(frames, detector, size) -> Iterator[torch.Tensor]:
    for c in head_crops(frames, detector, size):
        yield normalize(c, [127.5] * 3, [128.0] * 3).float()


def face_emb_batches(frames, detector, face_det, det_size, emb_size, conf) -> Iterator[torch.Tensor]:
    from pipeline.face import scrfd_top1

    for c in head_crops(frames, detector, det_size):
        outs = face_det(normalize(c, [127.5] * 3, [128.0] * 3), torch.cuda.current_stream())
        score, kps = scrfd_top1(outs, det_size, len(c))
        ok = score > conf
        if ok.any():
            yield normalize(align_faces(c[ok], kps[ok], emb_size), [127.5] * 3, [127.5] * 3).float()


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batches_factory: Callable[[], Iterator[torch.Tensor]], batch_size: int, cache: Path):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.factory = batches_factory
        self.bs = batch_size
        self.cache = Path(cache)
        self._it = None
        self._cur = None
        self.n = 0

    def get_batch_size(self):
        return self.bs

    def get_batch(self, names):
        if self._it is None:
            self._it = iter(self.factory())
        try:
            x = next(self._it)
        except StopIteration:
            return None
        self._cur = x.float().contiguous()
        torch.cuda.synchronize()
        self.n += len(x)
        return [int(self._cur.data_ptr())]

    def read_calibration_cache(self):
        return self.cache.read_bytes() if self.cache.exists() else None

    def write_calibration_cache(self, cache):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache.write_bytes(bytes(cache))
