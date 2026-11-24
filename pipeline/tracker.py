"""ByteTrack (Zhang et al., 2022) с векторизованным фильтром Калмана.

Состояние треков хранится массивами numpy (structure of arrays), предсказание и
обновление фильтра идут одним батчем на все треки камеры. Один экземпляр на камеру.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

TRACKED, LOST = 1, 2

_W_POS = 1.0 / 20
_W_VEL = 1.0 / 160
_F = np.eye(8)
_F[:4, 4:] = np.eye(4)


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w = b[:, 2] - b[:, 0]
    h = np.maximum(b[:, 3] - b[:, 1], 1e-6)
    return np.stack([b[:, 0] + w / 2, b[:, 1] + h / 2, w / h, h], axis=1)


def xyah_to_xyxy(m: np.ndarray) -> np.ndarray:
    w = m[:, 2] * m[:, 3]
    h = m[:, 3]
    return np.stack([m[:, 0] - w / 2, m[:, 1] - h / 2, m[:, 0] + w / 2, m[:, 1] + h / 2], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def assign(cost: np.ndarray, thresh: float):
    """Венгерский алгоритм с порогом. Возвращает (пары [k,2], несопоставленные строки, столбцы)."""
    if cost.size == 0:
        return np.empty((0, 2), int), np.arange(cost.shape[0]), np.arange(cost.shape[1])
    c = np.where(cost > thresh, thresh + 1e5, cost)
    rows, cols = linear_sum_assignment(c)
    ok = cost[rows, cols] <= thresh
    pairs = np.stack([rows[ok], cols[ok]], axis=1)
    um_r = np.setdiff1d(np.arange(cost.shape[0]), pairs[:, 0])
    um_c = np.setdiff1d(np.arange(cost.shape[1]), pairs[:, 1])
    return pairs, um_r, um_c


class BatchKalman:
    @staticmethod
    def initiate(meas: np.ndarray):
        n = len(meas)
        mean = np.concatenate([meas, np.zeros((n, 4))], axis=1)
        h = meas[:, 3]
        std = np.stack([
            2 * _W_POS * h, 2 * _W_POS * h, np.full(n, 1e-2), 2 * _W_POS * h,
            10 * _W_VEL * h, 10 * _W_VEL * h, np.full(n, 1e-5), 10 * _W_VEL * h,
        ], axis=1)
        cov = np.zeros((n, 8, 8))
        idx = np.arange(8)
        cov[:, idx, idx] = std ** 2
        return mean, cov

    @staticmethod
    def predict(mean: np.ndarray, cov: np.ndarray):
        n = len(mean)
        if n == 0:
            return mean, cov
        h = mean[:, 3]
        std = np.stack([
            _W_POS * h, _W_POS * h, np.full(n, 1e-2), _W_POS * h,
            _W_VEL * h, _W_VEL * h, np.full(n, 1e-5), _W_VEL * h,
        ], axis=1)
        q = np.zeros((n, 8, 8))
        idx = np.arange(8)
        q[:, idx, idx] = std ** 2
        mean = mean @ _F.T
        cov = _F @ cov @ _F.T + q
        return mean, cov

    @staticmethod
    def update(mean: np.ndarray, cov: np.ndarray, meas: np.ndarray):
        n = len(mean)
        if n == 0:
            return mean, cov
        h = mean[:, 3]
        std = np.stack([_W_POS * h, _W_POS * h, np.full(n, 1e-1), _W_POS * h], axis=1)
        idx = np.arange(4)
        s = cov[:, :4, :4].copy()
        s[:, idx, idx] += std ** 2
        pht = cov[:, :, :4]                                  # P Hᵀ
        k = np.linalg.solve(s, np.transpose(pht, (0, 2, 1)))  # (S⁻¹ H P)
        k = np.transpose(k, (0, 2, 1))                       # K = P Hᵀ S⁻¹
        innov = meas - mean[:, :4]
        mean = mean + np.einsum("nij,nj->ni", k, innov)
        cov = cov - k @ s @ np.transpose(k, (0, 2, 1))
        return mean, cov


@dataclass
class TrackOutput:
    ids: np.ndarray        # [T] int64
    boxes: np.ndarray      # [T,4] xyxy
    scores: np.ndarray     # [T]
    is_new: np.ndarray     # [T] bool — трек активирован на этом кадре


class ByteTracker:
    def __init__(self, track_thresh=0.5, low_thresh=0.1, new_track_thresh=0.6, match_thresh=0.8,
                 track_buffer=30, frame_rate=25.0, **_):
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max(1, int(frame_rate / 30.0 * track_buffer))
        self.frame_id = 0
        self._next_id = 1
        self.ids = np.zeros(0, np.int64)
        self.mean = np.zeros((0, 8))
        self.cov = np.zeros((0, 8, 8))
        self.score = np.zeros(0)
        self.state = np.zeros(0, np.int8)
        self.activated = np.zeros(0, bool)
        self.first_active = np.zeros(0, np.int64)
        self.last = np.zeros(0, np.int64)

    def __len__(self):
        return len(self.ids)

    def _boxes(self, idx: np.ndarray) -> np.ndarray:
        return xyah_to_xyxy(self.mean[idx])

    def _apply_matches(self, trk_idx: np.ndarray, det_boxes: np.ndarray, det_scores: np.ndarray):
        if len(trk_idx) == 0:
            return
        m, c = BatchKalman.update(self.mean[trk_idx], self.cov[trk_idx], xyxy_to_xyah(det_boxes))
        self.mean[trk_idx] = m
        self.cov[trk_idx] = c
        self.score[trk_idx] = det_scores
        self.state[trk_idx] = TRACKED
        newly = trk_idx[~self.activated[trk_idx]]
        self.first_active[newly] = self.frame_id
        self.activated[trk_idx] = True
        self.last[trk_idx] = self.frame_id

    def _keep(self, mask: np.ndarray):
        for name in ("ids", "mean", "cov", "score", "state", "activated", "first_active", "last"):
            setattr(self, name, getattr(self, name)[mask])

    def update(self, dets: np.ndarray) -> TrackOutput:
        """dets: [M,5] (x1,y1,x2,y2,score) в пикселях кадра."""
        self.frame_id += 1
        f = self.frame_id
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 5)
        scores = dets[:, 4]
        hi = np.nonzero(scores >= self.track_thresh)[0]
        lo = np.nonzero((scores >= self.low_thresh) & (scores < self.track_thresh))[0]

        if len(self.ids):
            lost = self.state != TRACKED
            self.mean[lost, 7] = 0.0
            self.mean, self.cov = BatchKalman.predict(self.mean, self.cov)

        pool = np.nonzero(self.activated)[0]                      # tracked + lost
        unconfirmed = np.nonzero(~self.activated)[0]
        was_tracked = self.state == TRACKED

        # 1. Активные и потерянные треки ↔ уверенные детекции (IoU × score)
        iou = iou_matrix(self._boxes(pool), dets[hi, :4])
        cost = 1.0 - iou * scores[hi][None, :]
        pairs, um_trk, um_det = assign(cost, self.match_thresh)
        self._apply_matches(pool[pairs[:, 0]], dets[hi[pairs[:, 1]], :4], scores[hi[pairs[:, 1]]])
        rem_hi = hi[um_det]

        # 2. Оставшиеся активные треки ↔ слабые детекции
        r_trk = pool[um_trk]
        r_trk = r_trk[was_tracked[r_trk]]
        cost2 = 1.0 - iou_matrix(self._boxes(r_trk), dets[lo, :4])
        pairs2, um_trk2, _ = assign(cost2, 0.5)
        self._apply_matches(r_trk[pairs2[:, 0]], dets[lo[pairs2[:, 1]], :4], scores[lo[pairs2[:, 1]]])
        self.state[r_trk[um_trk2]] = LOST

        # 3. Неподтверждённые треки ↔ оставшиеся уверенные детекции
        iou3 = iou_matrix(self._boxes(unconfirmed), dets[rem_hi, :4])
        cost3 = 1.0 - iou3 * scores[rem_hi][None, :]
        pairs3, um_unc, um_det3 = assign(cost3, 0.7)
        self._apply_matches(unconfirmed[pairs3[:, 0]], dets[rem_hi[pairs3[:, 1]], :4], scores[rem_hi[pairs3[:, 1]]])
        remove = np.zeros(len(self.ids), bool)
        remove[unconfirmed[um_unc]] = True

        # 4. Удаление давно потерянных
        remove |= (self.state == LOST) & (f - self.last > self.max_time_lost)
        self._keep(~remove)

        # 5. Новые треки
        new_det = rem_hi[um_det3]
        new_det = new_det[scores[new_det] >= self.new_track_thresh]
        if len(new_det):
            m, c = BatchKalman.initiate(xyxy_to_xyah(dets[new_det, :4]))
            n = len(new_det)
            act = f == 1
            self.ids = np.concatenate([self.ids, np.arange(self._next_id, self._next_id + n)])
            self._next_id += n
            self.mean = np.concatenate([self.mean, m])
            self.cov = np.concatenate([self.cov, c])
            self.score = np.concatenate([self.score, scores[new_det]])
            self.state = np.concatenate([self.state, np.full(n, TRACKED, np.int8)])
            self.activated = np.concatenate([self.activated, np.full(n, act)])
            self.first_active = np.concatenate([self.first_active, np.full(n, f if act else -1)])
            self.last = np.concatenate([self.last, np.full(n, f)])

        out = np.nonzero((self.state == TRACKED) & self.activated)[0]
        return TrackOutput(
            ids=self.ids[out].copy(),
            boxes=self._boxes(out),
            scores=self.score[out].copy(),
            is_new=self.first_active[out] == f,
        )
