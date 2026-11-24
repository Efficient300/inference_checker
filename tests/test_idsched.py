import numpy as np
import pytest

from pipeline.idsched import IdScheduler


def test_every_k_5():
    s = IdScheduler("every_k_5")
    ids = np.array([1, 2])
    picks = [s.select(ids).copy() for _ in range(20)]
    counts = np.sum(picks, axis=0)
    assert counts.tolist() == [4, 4]
    assert picks[0].all() and picks[5].all() and not picks[1].any()


def test_every_k_new_track_starts_own_phase():
    s = IdScheduler("every_k_5")
    s.select(np.array([1]))
    s.select(np.array([1]))
    sel = s.select(np.array([1, 2]))
    assert sel.tolist() == [False, True]


def test_new_tracks_only_until_confirmed():
    s = IdScheduler("new_tracks_only")
    ids = np.array([1, 2])
    assert s.select(ids).tolist() == [True, True]
    assert s.select(ids).tolist() == [False, False]          # в обработке
    s.confirm(np.array([1, 2]), np.array([True, False]))
    assert s.select(ids).tolist() == [False, True]           # 2 не подтверждён — повтор
    s.confirm(np.array([2]), True)
    for _ in range(5):
        assert not s.select(ids).any()


def test_bad_mode():
    with pytest.raises(ValueError):
        IdScheduler("sometimes")


def test_gc_drops_dead_tracks():
    s = IdScheduler("new_tracks_only", gc_after_frames=10)
    s.select(np.array([1]))
    s.confirm(np.array([1]), True)
    for _ in range(30):
        s.select(np.array([2]))
    assert 1 not in s._confirmed
