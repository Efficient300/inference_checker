import threading
import time

from pipeline.batcher import DynamicBatcher


def test_batch_by_size():
    b = DynamicBatcher(max_batch=4, timeout_ms=10_000)
    for i in range(10):
        b.put(i % 3, i)
    t = time.monotonic()
    batch = b.get_batch()
    assert time.monotonic() - t < 0.5
    assert [x for _, x in batch] == [0, 1, 2, 3]
    assert len(b.get_batch()) == 4
    assert b.qsize() == 2


def test_batch_by_timeout():
    b = DynamicBatcher(max_batch=64, timeout_ms=50)
    b.put(0, "a")
    b.put(1, "b")
    t = time.monotonic()
    batch = b.get_batch()
    dt = time.monotonic() - t
    assert [x for _, x in batch] == ["a", "b"]
    assert 0.03 <= dt < 0.5


def test_timeout_counts_from_first_item():
    b = DynamicBatcher(max_batch=64, timeout_ms=100)
    b.put(0, 1)
    time.sleep(0.08)
    t = time.monotonic()
    b.get_batch()
    assert time.monotonic() - t < 0.06


def test_empty_poll_and_close():
    b = DynamicBatcher(max_batch=8, timeout_ms=10)
    assert b.get_batch(poll_s=0.02) == []
    b.close()
    assert b.get_batch() is None
    assert b.put(0, 1) is False


def test_capacity_per_camera():
    b = DynamicBatcher(max_batch=100, timeout_ms=1, per_cam_capacity=2)
    assert b.put("a", 1) and b.put("a", 2)
    assert not b.put("a", 3)
    assert b.put("b", 1)
    assert b.rejected == 1
    b.get_batch()
    assert b.put("a", 4)


def test_order_preserved_per_camera_concurrent():
    b = DynamicBatcher(max_batch=7, timeout_ms=2)
    n_cams, n_frames = 8, 300
    got = {c: [] for c in range(n_cams)}

    def producer(cam):
        for i in range(n_frames):
            b.put(cam, i)
            if i % 50 == 0:
                time.sleep(0.001)

    def consumer():
        total = 0
        while total < n_cams * n_frames:
            for cam, x in b.get_batch():
                got[cam].append(x)
                total += 1

    ths = [threading.Thread(target=producer, args=(c,)) for c in range(n_cams)]
    cons = threading.Thread(target=consumer)
    cons.start()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    cons.join(timeout=10)
    for c in range(n_cams):
        assert got[c] == list(range(n_frames))
