import numpy as np

from pipeline.tracker import ByteTracker, iou_matrix


def walk(n_people, n_frames, speed=4.0, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    start = rng.uniform([0, 100], [1500, 700], size=(n_people, 2))
    vel = rng.uniform(-speed, speed, size=(n_people, 2))
    for f in range(n_frames):
        xy = start + vel * f + rng.normal(0, noise, size=(n_people, 2))
        boxes = np.concatenate([xy, xy + [60, 160]], axis=1)
        yield np.concatenate([boxes, np.full((n_people, 1), 0.9)], axis=1)


def test_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0])


def test_ids_stable_for_moving_people():
    tr = ByteTracker(frame_rate=10)
    seen = []
    for dets in walk(12, 60):
        out = tr.update(dets)
        seen.append(out)
    # с 2-го кадра все люди подтверждены, ID не меняются
    final_ids = set(seen[-1].ids.tolist())
    assert len(final_ids) == 12
    for out in seen[2:]:
        assert set(out.ids.tolist()) == final_ids


def test_is_new_flag_once_per_track():
    tr = ByteTracker(frame_rate=10)
    news = 0
    for dets in walk(5, 30):
        news += int(tr.update(dets).is_new.sum())
    assert news == 5


def test_lost_track_recovered_after_gap():
    tr = ByteTracker(frame_rate=25, track_buffer=30)
    frames = list(walk(1, 40, speed=2.0, noise=0.0))
    ids = []
    for i, d in enumerate(frames):
        out = tr.update(d if not (20 <= i < 25) else np.zeros((0, 5)))
        if len(out.ids):
            ids.append(int(out.ids[0]))
    assert len(set(ids)) == 1


def test_low_score_dets_extend_tracks_but_do_not_start():
    tr = ByteTracker(frame_rate=25)
    box = np.array([[100, 100, 160, 260, 0.9]])
    tr.update(box)
    tr.update(box)
    low = box.copy()
    low[0, 4] = 0.3
    out = tr.update(low)
    assert len(out.ids) == 1
    tr2 = ByteTracker(frame_rate=25)
    for _ in range(5):
        assert len(tr2.update(low).ids) == 0


def test_many_tracks_performance():
    import time

    tr = ByteTracker(frame_rate=10)
    gen = walk(150, 50, seed=3)
    t = time.perf_counter()
    for d in gen:
        tr.update(d)
    per_frame_ms = (time.perf_counter() - t) / 50 * 1000
    assert per_frame_ms < 50, per_frame_ms
