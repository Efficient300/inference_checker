"""Межкамерный динамический батчер.

Кадры всех камер воркера складываются в одну FIFO-очередь. Батч отдаётся, когда набран
max_batch или с момента появления первого элемента прошло timeout_ms. FIFO сохраняет
порядок кадров каждой камеры. Ёмкость ограничена на камеру: при переполнении новый
кадр отклоняется и считается пропущенным.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Hashable


class DynamicBatcher:
    def __init__(self, max_batch: int, timeout_ms: float, per_cam_capacity: int | None = None, name: str = ""):
        if max_batch < 1:
            raise ValueError("max_batch >= 1")
        self.max_batch = max_batch
        self.timeout = timeout_ms / 1000.0
        self.capacity = per_cam_capacity
        self.name = name
        self._q: deque[tuple[Hashable, float, Any]] = deque()
        self._per_cam: dict[Hashable, int] = {}
        self._cond = threading.Condition()
        self._closed = False
        self.rejected = 0

    def put(self, cam_id: Hashable, item: Any) -> bool:
        """Добавить элемент. False — очередь камеры полна (или батчер закрыт), элемент не принят."""
        with self._cond:
            if self._closed:
                return False
            n = self._per_cam.get(cam_id, 0)
            if self.capacity is not None and n >= self.capacity:
                self.rejected += 1
                return False
            self._per_cam[cam_id] = n + 1
            self._q.append((cam_id, time.monotonic(), item))
            self._cond.notify()
            return True

    def get_batch(self, poll_s: float = 0.1) -> list[tuple[Hashable, Any]] | None:
        """Батч [(cam_id, item), ...] в порядке поступления.

        [] — за poll_s ничего не пришло (вызывающий может проверить флаг остановки);
        None — батчер закрыт и пуст.
        """
        with self._cond:
            poll_deadline = time.monotonic() + poll_s
            while not self._q:
                if self._closed:
                    return None
                remaining = poll_deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._cond.wait(remaining)
            deadline = self._q[0][1] + self.timeout
            while len(self._q) < self.max_batch and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            out = []
            for _ in range(min(self.max_batch, len(self._q))):
                cam, _, item = self._q.popleft()
                self._per_cam[cam] -= 1
                out.append((cam, item))
            return out

    def qsize(self) -> int:
        with self._cond:
            return len(self._q)

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
