"""Лица: кропы головы/верха тела → SCRFD (лучшее лицо на кроп) → выравнивание → ArcFace."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.preprocess import align_faces, crop_resize, normalize
from pipeline.reid import build_rois


def scrfd_top1(outs: list[torch.Tensor], input_size: int, batch: int):
    """Декодирует выходы SCRFD и берёт лучшее лицо на изображение.

    Выходы группируются по последней размерности (1 — score, 4 — bbox, 10 — keypoints),
    страйды восстанавливаются по числу якорей. Поддерживаются выходы [B,A,C] и [B*A,C].
    Возвращает (score [B], kps [B,5,2] в пикселях входа).
    """
    groups: dict[int, list[torch.Tensor]] = {1: [], 4: [], 10: []}
    for o in outs:
        c = o.shape[-1]
        if c in groups:
            groups[c].append(o.float().reshape(batch, -1, c))
    scores = sorted(groups[1], key=lambda t: -t.shape[1])
    kpss = sorted(groups[10], key=lambda t: -t.shape[1])
    if not scores or len(kpss) != len(scores):
        raise RuntimeError("модель лиц должна выдавать score и keypoints (SCRFD с kps)")
    strides = [8 * 2 ** i for i in range(len(scores))]
    centers, stride_vec = [], []
    for s, t in zip(strides, scores):
        n = math.ceil(input_size / s)
        n_anchor = t.shape[1] // (n * n)
        ys, xs = torch.meshgrid(torch.arange(n, device=t.device), torch.arange(n, device=t.device), indexing="ij")
        c = (torch.stack([xs, ys], -1).reshape(-1, 2) * s).float()
        c = c.repeat_interleave(n_anchor, dim=0)
        centers.append(c)
        stride_vec.append(torch.full((len(c),), float(s), device=t.device))
    score = torch.cat([t[..., 0] for t in scores], dim=1)          # [B,A]
    kps = torch.cat(kpss, dim=1)                                   # [B,A,10]
    centers = torch.cat(centers)
    stride_vec = torch.cat(stride_vec)
    best = score.argmax(dim=1)
    ar = torch.arange(batch, device=score.device)
    k = kps[ar, best].view(batch, 5, 2)
    pts = centers[best][:, None, :] + k * stride_vec[best][:, None, None]
    return score[ar, best], pts


class FaceStage:
    def __init__(self, det_cfg: dict, det_model, emb_cfg: dict, emb_model):
        self.det = det_model
        self.emb = emb_model
        self.det_size = int(det_cfg["input"])
        self.conf = float(det_cfg.get("conf", 0.5))
        self.emb_size = int(emb_cfg["input"])
        self.stream = torch.cuda.Stream()

    def run(self, frames: list[torch.Tensor], regions: list[np.ndarray]):
        """regions: [k,4] области головы по кадру. Возвращает (найдено лицо [k] bool по кадру, число эмбеддингов)."""
        counts = [len(r) for r in regions]
        total = sum(counts)
        with torch.cuda.stream(self.stream):
            for f in frames:
                f.record_stream(self.stream)
            batch = torch.stack(frames)
            rois = build_rois(regions, batch.device)
            crops = crop_resize(batch, rois, (self.det_size, self.det_size))
            outs = self.det(normalize(crops, [127.5] * 3, [128.0] * 3), self.stream)
            score, kps = scrfd_top1(outs, self.det_size, total)
            found = score > self.conf
            n_found = int(found.sum().item())
            if n_found:
                aligned = align_faces(crops[found], kps[found], self.emb_size)
                emb = self.emb(normalize(aligned, [127.5] * 3, [127.5] * 3), self.stream)[0]
                F.normalize(emb.float().flatten(1), dim=1).cpu()
            found_np = found.cpu().numpy()
        return list(np.split(found_np, np.cumsum(counts)[:-1])), n_found
