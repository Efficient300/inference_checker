"""Батч person re-ID: кропы тел из нескольких кадров → эмбеддинги."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.preprocess import crop_resize, normalize

IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]

NORMS = {
    "osnet": (IMAGENET_MEAN, IMAGENET_STD),
}


def build_rois(frames_boxes: list[np.ndarray], device) -> torch.Tensor:
    rois = []
    for i, b in enumerate(frames_boxes):
        if len(b):
            r = np.empty((len(b), 5), np.float32)
            r[:, 0] = i
            r[:, 1:] = b[:, :4]
            rois.append(r)
    if not rois:
        return torch.zeros((0, 5), device=device)
    return torch.from_numpy(np.concatenate(rois)).to(device, non_blocking=True)


class ReidStage:
    def __init__(self, cfg: dict, model):
        self.model = model
        h, w = cfg["input"]
        self.hw = (int(h), int(w))
        self.mean, self.std = NORMS.get(cfg.get("type", "osnet"), NORMS["osnet"])
        self.stream = torch.cuda.Stream()

    def run(self, frames: list[torch.Tensor], boxes: list[np.ndarray]) -> list[np.ndarray]:
        """frames: RGB [3,H,W] uint8 по кадру; boxes: [k,4] по кадру. Возвращает эмбеддинги по кадрам."""
        counts = [len(b) for b in boxes]
        with torch.cuda.stream(self.stream):
            for f in frames:
                f.record_stream(self.stream)
            batch = torch.stack(frames)
            rois = build_rois(boxes, batch.device)
            crops = crop_resize(batch, rois, self.hw)
            emb = self.model(normalize(crops, self.mean, self.std), self.stream)[0]
            emb = F.normalize(emb.float().flatten(1), dim=1).cpu().numpy()
        return list(np.split(emb, np.cumsum(counts)[:-1]))
