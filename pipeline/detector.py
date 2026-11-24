"""Батч-детекция людей. Адаптеры пре/постпроцессинга выбираются по models.detector.type."""
from __future__ import annotations

import numpy as np
import torch

from pipeline.preprocess import letterbox, nv12_to_rgb


def yolo_postprocess(out: torch.Tensor, ratio: float, pad: tuple[int, int], frame_hw: tuple[int, int],
                     conf: float, iou: float, classes: list[int], max_det: int) -> list[np.ndarray]:
    """Выход YOLOv8/YOLO11 [B, 4+nc, A] (cx,cy,w,h в пикселях входа) → список [M,5] xyxy+score."""
    from torchvision.ops import batched_nms

    out = out.float()
    b = out.shape[0]
    boxes = out[:, :4, :].transpose(1, 2)                         # [B,A,4]
    cls_scores = out[:, 4:, :][:, classes, :]                     # [B,C,A]
    scores, _ = cls_scores.max(dim=1)                             # [B,A]
    keep = scores > conf
    bi, ai = keep.nonzero(as_tuple=True)
    if bi.numel() == 0:
        return [np.zeros((0, 5), np.float32) for _ in range(b)]
    bx = boxes[bi, ai]
    sc = scores[bi, ai]
    xyxy = torch.stack([bx[:, 0] - bx[:, 2] / 2, bx[:, 1] - bx[:, 3] / 2,
                        bx[:, 0] + bx[:, 2] / 2, bx[:, 1] + bx[:, 3] / 2], dim=1)
    k = batched_nms(xyxy, sc, bi, iou)
    xyxy, sc, bi = xyxy[k], sc[k], bi[k]
    px, py = pad
    xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - px) / ratio).clamp(0, frame_hw[1])
    xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - py) / ratio).clamp(0, frame_hw[0])
    res = torch.cat([xyxy, sc[:, None]], dim=1).cpu().numpy()
    bi = bi.cpu().numpy()
    out_list = []
    for i in range(b):
        d = res[bi == i]
        if len(d) > max_det:
            d = d[np.argsort(-d[:, 4])[:max_det]]
        out_list.append(d)
    return out_list


class YoloDetector:
    def __init__(self, cfg: dict, model):
        self.model = model
        self.input = int(cfg["input"])
        self.conf = float(cfg.get("conf", 0.25))
        self.iou = float(cfg.get("iou", 0.45))
        self.classes = list(cfg.get("classes", [0]))
        self.max_det = int(cfg.get("max_det", 300))

    def preprocess(self, rgb: torch.Tensor):
        x, r, pad = letterbox(rgb, self.input)
        return x / 255.0, (r, pad, tuple(rgb.shape[-2:]))

    def __call__(self, rgb: torch.Tensor, stream) -> list[np.ndarray]:
        x, (r, pad, hw) = self.preprocess(rgb)
        out = self.model(x, stream)[0]
        return yolo_postprocess(out, r, pad, hw, self.conf, self.iou, self.classes, self.max_det)


DETECTORS = {"yolo": YoloDetector}


class DetectorStage:
    """NV12-кадры батча → RGB (для последующих кропов) + детекции на CPU."""

    def __init__(self, cfg: dict, model):
        self.impl = DETECTORS[cfg.get("type", "yolo")](cfg, model)
        self.stream = torch.cuda.Stream()

    def run(self, nv12_frames: list[torch.Tensor]) -> tuple[torch.Tensor, list[np.ndarray]]:
        with torch.cuda.stream(self.stream):
            rgb = nv12_to_rgb(torch.stack(nv12_frames))
            dets = self.impl(rgb, self.stream)
        self.stream.synchronize()
        return rgb, dets
