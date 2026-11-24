"""Поиск N_max: ступенчатый рост (×growth), затем бинарный поиск в последнем интервале.

evaluate(n) возвращает точку с полем passed. Функция детерминирована при одинаковых ответах
evaluate, поэтому возобновление после прерывания — это повтор поиска с кэшем точек.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass
class SearchResult:
    n_max: int                 # наибольшее N, прошедшее критерии (0 — даже минимальное не прошло)
    n_fail: int | None         # наименьшее не прошедшее N над n_max (None — упёрлись в max_n)
    capped: bool = False


def _next_up(n: int, growth: float) -> int:
    return max(n + 1, math.ceil(n * growth))


def find_n_max(evaluate: Callable[[int], dict], start: int, growth: float, precision: int,
               max_n: int, bracket: tuple[int, int | None] | None = None) -> SearchResult:
    passed = lambda n: bool(evaluate(n)["passed"])  # noqa: E731
    lo, hi = 0, None
    max_n = max(1, max_n)

    if bracket is None:
        n = min(max(1, start), max_n)
        while True:
            if passed(n):
                lo = n
                if n >= max_n:
                    return SearchResult(lo, None, capped=True)
                n = min(_next_up(n, growth), max_n)
            else:
                hi = n
                break
    else:
        # уточнение вокруг ранее найденной границы: подтвердить нижнюю, затем найти верхнюю
        g_lo, g_hi = bracket
        n = min(max(1, g_lo), max_n)
        while n >= 1:
            if passed(n):
                lo = n
                break
            hi = n
            n = int(n / growth) if n > 1 else 0
        if hi is None:
            n = min(max(lo + 1, g_hi or _next_up(lo, growth)), max_n)
            while True:
                if passed(n):
                    lo = n
                    if n >= max_n:
                        return SearchResult(lo, None, capped=True)
                    n = min(_next_up(n, growth), max_n)
                else:
                    hi = n
                    break

    while hi - lo > precision:
        mid = (lo + hi) // 2
        if passed(mid):
            lo = mid
        else:
            hi = mid
    return SearchResult(lo, hi)
