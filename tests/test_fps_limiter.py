import pytest

from pipeline.fps_limiter import FpsLimiter


@pytest.mark.parametrize("target", [5, 10, 15, 25])
def test_exact_rate(target):
    lim = FpsLimiter(25, target)
    seconds = 600
    kept = sum(lim.keep() for _ in range(25 * seconds))
    assert abs(kept / seconds - target) <= 0.01 * target


@pytest.mark.parametrize("target", [5, 10, 15])
def test_uniform_gaps(target):
    lim = FpsLimiter(25, target)
    idx = [i for i in range(250) if lim.keep()]
    gaps = {b - a for a, b in zip(idx, idx[1:])}
    ideal = 25 / target
    assert all(abs(g - ideal) < 1 for g in gaps), gaps


@pytest.mark.parametrize("window", [1, 2, 3, 7])
def test_rate_in_every_window(window):
    """В любом окне в целое число секунд FPS не отклоняется больше чем на 1 кадр."""
    lim = FpsLimiter(25, 10)
    keep = [lim.keep() for _ in range(25 * 60)]
    for start in range(0, len(keep) - 25 * window, 13):
        n = sum(keep[start:start + 25 * window])
        assert abs(n - 10 * window) <= 1


def test_target_above_source_keeps_all():
    lim = FpsLimiter(25, 1000)
    assert all(lim.keep() for _ in range(100))


def test_fractional_source():
    lim = FpsLimiter(29.97, 10)
    kept = sum(lim.keep() for _ in range(2997))
    assert abs(kept - 1000) <= 1
