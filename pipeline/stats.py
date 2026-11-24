"""Сбор сырых метрик внутри воркера. Учитываются только кадры, принятые в окне замера."""
from __future__ import annotations

import threading
import time
from array import array
from collections import defaultdict


class CamStats:
    __slots__ = ("decoded", "selected", "done", "dropped", "persons", "tracks", "lag", "events")

    def __init__(self):
        self.decoded = 0
        self.selected = 0
        self.done = 0
        self.dropped = 0
        self.persons = 0
        self.tracks = 0
        self.lag: list[tuple[float, float]] = []
        self.events: list[tuple[float, str, str]] = []


class Recorder:
    def __init__(self, cam_ids, t_meas: float, t_end: float):
        self.t_meas = t_meas
        self.t_end = t_end
        self.cams = {c: CamStats() for c in cam_ids}
        self.latencies = array("d")
        self.counters: dict[str, int] = defaultdict(int)
        self.busy: dict[str, float] = defaultdict(float)
        self.queues: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self.errors: list[tuple[float, str, str]] = []
        self._lock = threading.Lock()

    def in_window(self, t: float) -> bool:
        return self.t_meas <= t < self.t_end

    def decoded(self, cam, t: float) -> None:
        if self.in_window(t):
            self.cams[cam].decoded += 1

    def selected(self, cam, t_recv: float) -> None:
        if self.in_window(t_recv):
            self.cams[cam].selected += 1

    def dropped(self, cam, t_recv: float, stage: str) -> None:
        if self.in_window(t_recv):
            with self._lock:
                self.cams[cam].dropped += 1
                self.counters[f"drop_{stage}"] += 1

    def done(self, cam, t_recv: float, t_done: float, persons: int = 0, tracks: int = 0) -> None:
        if self.in_window(t_recv):
            with self._lock:
                s = self.cams[cam]
                s.done += 1
                s.persons += persons
                s.tracks += tracks
                self.latencies.append(t_done - t_recv)

    def lag_sample(self, cam, t: float, lag: float) -> None:
        if self.in_window(t):
            self.cams[cam].lag.append((t, lag))

    def event(self, cam, kind: str, msg: str = "") -> None:
        with self._lock:
            self.cams[cam].events.append((time.monotonic(), kind, msg))

    def error(self, stage: str, msg: str) -> None:
        with self._lock:
            self.errors.append((time.monotonic(), stage, msg))

    def count(self, name: str, n: int = 1) -> None:
        if n:
            with self._lock:
                self.counters[name] += n

    def add_busy(self, stage: str, seconds: float, t: float) -> None:
        if self.in_window(t):
            with self._lock:
                self.busy[stage] += seconds

    def queue_sample(self, name: str, t: float, depth: int) -> None:
        if self.in_window(t):
            self.queues[name].append((t, depth))

    def to_dict(self) -> dict:
        window = self.t_end - self.t_meas
        return {
            "window_s": window,
            "t_meas": self.t_meas,
            "t_end": self.t_end,
            "cams": {
                str(c): {
                    "decoded": s.decoded, "selected": s.selected, "done": s.done, "dropped": s.dropped,
                    "persons": s.persons, "tracks": s.tracks, "lag": s.lag, "events": s.events,
                }
                for c, s in self.cams.items()
            },
            "counters": dict(self.counters),
            "busy_frac": {k: v / window for k, v in self.busy.items()},
            "queues": dict(self.queues),
            "errors": self.errors,
        }
