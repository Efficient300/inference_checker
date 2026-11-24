"""GPU-препроцессинг на torch: NV12→RGB, letterbox, кропы, выравнивание лиц."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def nv12_to_rgb(nv12: torch.Tensor) -> torch.Tensor:
    """[B, H*3/2, W] uint8 NV12 → [B, 3, H, W] uint8 RGB (BT.709, limited range)."""
    if nv12.dim() == 2:
        nv12 = nv12[None]
    b, hh, w = nv12.shape
    h = hh * 2 // 3
    y = (nv12[:, :h, :].half() - 16.0) * 1.164383
    uv = nv12[:, h:h + h // 2, :].reshape(b, h // 2, w // 2, 2).permute(0, 3, 1, 2).half() - 128.0
    uv = F.interpolate(uv, size=(h, w), mode="nearest")
    u, v = uv[:, 0], uv[:, 1]
    r = y + 1.792741 * v
    g = y - 0.213249 * u - 0.532909 * v
    bl = y + 2.112402 * u
    return torch.stack((r, g, bl), dim=1).clamp_(0, 255).to(torch.uint8)


def letterbox(rgb: torch.Tensor, size: int, pad_value: float = 114.0):
    """[B,3,H,W] → ([B,3,size,size] half в шкале 0..255, ratio, (pad_x, pad_y))."""
    b, _, h, w = rgb.shape
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    x = F.interpolate(rgb.half(), size=(nh, nw), mode="bilinear", align_corners=False)
    px, py = (size - nw) // 2, (size - nh) // 2
    out = torch.full((b, 3, size, size), pad_value, dtype=torch.half, device=rgb.device)
    out[:, :, py:py + nh, px:px + nw] = x
    return out, r, (px, py)


def normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - m) / s


def crop_resize(frames: torch.Tensor, rois: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Кропы с ресайзом из нескольких кадров за один вызов.

    frames: [F,3,H,W] (uint8 или half); rois: [K,5] (индекс кадра, x1, y1, x2, y2) в пикселях.
    Возвращает [K,3,h,w] half в шкале 0..255.
    """
    from torchvision.ops import roi_align

    if frames.dtype != torch.half:
        frames = frames.half()
    return roi_align(frames, rois.to(frames.dtype), output_size=out_hw, spatial_scale=1.0,
                     sampling_ratio=2, aligned=True)


# Шаблон 5 точек ArcFace для кропа 112×112
ARCFACE_DST = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]]


def similarity_transform(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Подобие (поворот+масштаб+сдвиг) src→dst МНК. src: [N,P,2], dst: [P,2] → [N,2,3]."""
    n, p, _ = src.shape
    x, y = src[..., 0], src[..., 1]
    one, zero = torch.ones_like(x), torch.zeros_like(x)
    a_rows_u = torch.stack([x, -y, one, zero], dim=-1)
    a_rows_v = torch.stack([y, x, zero, one], dim=-1)
    a = torch.cat([a_rows_u, a_rows_v], dim=1)                     # [N,2P,4]
    rhs = torch.cat([dst[:, 0].expand(n, p), dst[:, 1].expand(n, p)], dim=1)[..., None]
    sol = torch.linalg.lstsq(a, rhs).solution[..., 0]               # [N,4]
    aa, bb, tx, ty = sol.unbind(-1)
    return torch.stack([torch.stack([aa, -bb, tx], -1), torch.stack([bb, aa, ty], -1)], dim=1)


def _pix_from_norm(size: int, device) -> torch.Tensor:
    """Матрица: нормированные координаты grid_sample (align_corners=False) → пиксели."""
    return torch.tensor([[size / 2, 0, (size - 1) / 2], [0, size / 2, (size - 1) / 2], [0, 0, 1]],
                        dtype=torch.float32, device=device)


def align_faces(crops: torch.Tensor, kps: torch.Tensor, out_size: int = 112) -> torch.Tensor:
    """Выравнивание лиц по 5 точкам. crops: [N,3,S,S] (0..255), kps: [N,5,2] в пикселях кропа."""
    n, _, sh, sw = crops.shape
    assert sh == sw, "ожидается квадратный кроп"
    dev = crops.device
    dst = torch.tensor(ARCFACE_DST, dtype=torch.float32, device=dev) * (out_size / 112.0)
    m = similarity_transform(kps.float(), dst)                      # src px → dst px
    m3 = torch.cat([m, torch.tensor([0, 0, 1.0], device=dev).expand(n, 1, 3)], dim=1)
    theta = torch.linalg.inv(_pix_from_norm(sh, dev)) @ torch.linalg.inv(m3) @ _pix_from_norm(out_size, dev)
    grid = F.affine_grid(theta[:, :2, :], [n, 3, out_size, out_size], align_corners=False)
    return F.grid_sample(crops.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def upper_squares(boxes, frame_w: int, frame_h: int):
    """Квадратная область головы/верха тела: ширина бокса человека, от верхнего края вниз."""
    import numpy as np

    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    w = b[:, 2] - b[:, 0]
    side = np.minimum(w, b[:, 3] - b[:, 1])
    out = np.stack([b[:, 0], b[:, 1], b[:, 0] + side, b[:, 1] + side], axis=1)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, frame_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, frame_h)
    return out
