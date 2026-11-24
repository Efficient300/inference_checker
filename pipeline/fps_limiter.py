"""Равномерное прореживание потока source_fps → target_fps по номеру кадра.

Решение принимается по индексу декодированного кадра, а не по часам: если декодер
не успевает, выходной FPS падает пропорционально, и это видно в метриках.
"""
from __future__ import annotations

from fractions import Fraction


class FpsLimiter:
    def __init__(self, source_fps: float, target_fps: float):
        if source_fps <= 0 or target_fps <= 0:
            raise ValueError("FPS должен быть > 0")
        self.source_fps = Fraction(source_fps).limit_denominator(1001)
        self.target_fps = Fraction(target_fps).limit_denominator(1001)
        # target >= source: пропускаем все кадры (нужно для проверки «недостижимого FPS»)
        self.ratio = min(Fraction(1), self.target_fps / self.source_fps)
        self._i = 0

    def keep(self) -> bool:
        """Вызывается на каждый декодированный кадр; True — кадр идёт в анализ."""
        i = self._i
        self._i += 1
        if i == 0:
            return True
        return (i * self.ratio).__floor__() != ((i - 1) * self.ratio).__floor__()

    def reset(self) -> None:
        self._i = 0
