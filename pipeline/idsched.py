"""Выбор треков для идентификации (re-ID / лица) на текущем кадре.

Режимы:
- every_k_N: трек идентифицируется на первом кадре и затем на каждом N-м анализируемом кадре;
- new_tracks_only: трек идентифицируется, пока не подтверждён (re-ID — получен эмбеддинг,
  лица — найдено лицо). Пока запрос в обработке, повторно трек не выбирается.
"""
from __future__ import annotations

import threading

import numpy as np


class IdScheduler:
    def __init__(self, mode: str, gc_after_frames: int = 300):
        if mode == "new_tracks_only":
            self.every_k = None
        elif mode.startswith("every_k_"):
            self.every_k = int(mode.rsplit("_", 1)[1])
            if self.every_k < 1:
                raise ValueError(mode)
        else:
            raise ValueError(f"неизвестный режим {mode}")
        self.mode = mode
        self._count: dict[int, int] = {}
        self._confirmed: set[int] = set()
        self._pending: set[int] = set()
        self._last_seen: dict[int, int] = {}
        self._frame = 0
        self._gc_after = gc_after_frames
        self._lock = threading.Lock()

    def select(self, track_ids: np.ndarray) -> np.ndarray:
        with self._lock:
            self._frame += 1
            sel = np.zeros(len(track_ids), bool)
            for i, tid in enumerate(track_ids.tolist()):
                self._last_seen[tid] = self._frame
                if self.every_k is not None:
                    c = self._count.get(tid, 0)
                    sel[i] = c % self.every_k == 0
                    self._count[tid] = c + 1
                elif tid not in self._confirmed and tid not in self._pending:
                    sel[i] = True
                    self._pending.add(tid)
            if self._frame % self._gc_after == 0:
                self._gc()
            return sel

    def confirm(self, track_ids, ok) -> None:
        with self._lock:
            for tid, good in zip(np.asarray(track_ids).tolist(), np.broadcast_to(ok, len(track_ids)).tolist()):
                self._pending.discard(tid)
                if good:
                    self._confirmed.add(tid)

    def _gc(self) -> None:
        dead = [t for t, f in self._last_seen.items() if self._frame - f > self._gc_after]
        for t in dead:
            self._last_seen.pop(t, None)
            self._count.pop(t, None)
            self._confirmed.discard(t)
            self._pending.discard(t)
