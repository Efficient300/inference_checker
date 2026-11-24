"""Процесс-воркер: группа камер через полный пайплайн сценария.

Потоки внутри процесса:
  cam*   — CameraReader на каждую камеру (RTSP → NVDEC → прореживание);
  detect — межкамерный батч → NV12→RGB → letterbox → TensorRT → NMS;
  track  — ByteTrack на камеру (CPU);
  reid / face — батчи кропов по выбранным трекам.
Каждая стадия работает на своём CUDA stream. Протокол с родителем — через Pipe:
  воркер → ("ready", pid) | ("error", tb);  родитель → timing;  воркер → ("done", path) | ("error", tb).
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from bench.config import STAGES
from pipeline.batcher import DynamicBatcher
from pipeline.decoder import CameraReader
from pipeline.idsched import IdScheduler
from pipeline.stats import Recorder
from pipeline.tracker import ByteTracker


class FrameJob:
    __slots__ = ("cam", "idx", "t_recv", "nv12", "rgb", "dets", "persons", "tracks", "pending", "dropped")

    def __init__(self, cam, idx, t_recv, nv12):
        self.cam = cam
        self.idx = idx
        self.t_recv = t_recv
        self.nv12 = nv12
        self.rgb = None
        self.dets = None
        self.persons = 0
        self.tracks = 0
        self.pending = 0
        self.dropped = False


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


class Worker:
    def __init__(self, args: dict):
        self.args = args
        self.idx = args["worker_idx"]
        self.combo = args["combo"]
        self.stages = STAGES[self.combo["scenario"]]
        self.cams = [(int(c), u) for c, u in args["cams"]]
        self.pcfg = args["pipeline"]
        self.source_fps = float(args["source_fps"])
        self.target_fps = float(self.combo["target_fps"])
        self.gpu = int(self.pcfg.get("gpu", 0))
        self.models: dict = {}

    # ---------- инициализация ----------
    def init(self) -> None:
        import PyNvVideoCodec  # noqa: F401 — проверка, что декодер доступен

        if not self.stages:
            return
        import torch

        from pipeline.detector import DetectorStage
        from pipeline.trt_runtime import TrtModel

        torch.cuda.set_device(self.gpu)
        eng = self.args["engines"]
        self.det_stage = DetectorStage(eng["detector"]["cfg"], TrtModel(eng["detector"]["path"]))
        h, w = self.args.get("frame_hw", (1080, 1920))
        dummy = torch.zeros((h * 3 // 2, w), dtype=torch.uint8, device="cuda")
        rgb, _ = self.det_stage.run([dummy])
        if "reid" in self.stages:
            from pipeline.reid import ReidStage

            self.reid_stage = ReidStage(eng["reid"]["cfg"], TrtModel(eng["reid"]["path"]))
            self.reid_stage.run([rgb[0]], [np.array([[100, 100, 300, 600]], np.float32)])
        if "face" in self.stages:
            from pipeline.face import FaceStage

            self.face_stage = FaceStage(eng["face_det"]["cfg"], TrtModel(eng["face_det"]["path"]),
                                        eng["face_emb"]["cfg"], TrtModel(eng["face_emb"]["path"]))
            self.face_stage.run([rgb[0]], [np.array([[100, 100, 300, 300]], np.float32)])
        torch.cuda.synchronize()

    # ---------- основной цикл ----------
    def run(self, timing: dict) -> dict:
        t0, t_meas, t_end = timing["t0"], timing["t_meas"], timing["t_end"]
        drain = float(self.pcfg.get("drain_s", 3))
        self.rec = rec = Recorder([c for c, _ in self.cams], t_meas, t_end)
        self.stop = threading.Event()
        self._job_lock = threading.Lock()
        qcap = int(self.pcfg["queue_per_cam"])
        timeout_ms = float(self.pcfg["batch_timeout_ms"])
        id_frames = int(self.pcfg.get("id_frames_per_batch", 32))
        threads: list[threading.Thread] = []
        queues: dict[str, object] = {}

        if "detect" in self.stages:
            det_cfg = self.args["engines"]["detector"]["cfg"]
            self.det_q = DynamicBatcher(int(det_cfg["max_batch"]), timeout_ms, qcap, "detect")
            queues["detect"] = self.det_q
            threads.append(threading.Thread(target=self._detect_loop, name="detect", daemon=True))
        if "track" in self.stages:
            tcfg = {k: v for k, v in self.args["tracker"].items() if k != "type"}
            fr = min(self.target_fps, self.source_fps)
            self.trackers = {c: ByteTracker(frame_rate=fr, **tcfg) for c, _ in self.cams}
            self.track_q: queue.Queue = queue.Queue()
            queues["track"] = self.track_q
            threads.append(threading.Thread(target=self._track_loop, name="track", daemon=True))
        mode = self.combo.get("id_mode")
        if "reid" in self.stages:
            self.reid_sched = {c: IdScheduler(mode) for c, _ in self.cams}
            self.reid_q = DynamicBatcher(id_frames, timeout_ms, qcap, "reid")
            queues["reid"] = self.reid_q
            threads.append(threading.Thread(target=self._reid_loop, name="reid", daemon=True))
        if "face" in self.stages:
            self.face_sched = {c: IdScheduler(mode) for c, _ in self.cams}
            self.face_q = DynamicBatcher(id_frames, timeout_ms, qcap, "face")
            queues["face"] = self.face_q
            threads.append(threading.Thread(target=self._face_loop, name="face", daemon=True))
        for t in threads:
            t.start()

        time.sleep(max(0.0, t0 - time.monotonic()))
        readers = [
            CameraReader(c, url, self.source_fps, self.target_fps, rec, self._on_frame, self.stop,
                         gpu_id=self.gpu, copy_frames=bool(self.stages))
            for c, url in self.cams
        ]
        for r in readers:
            r.start()

        stall_timeout = float(self.pcfg["stall_timeout_s"])
        stalled = set()
        while True:
            now = time.monotonic()
            if now >= t_end + drain:
                break
            for name, q in queues.items():
                rec.queue_sample(name, now, q.qsize())
            if now >= t_meas:
                for r in readers:
                    if now - r.last_frame_t > stall_timeout:
                        if r.cam_id not in stalled:
                            stalled.add(r.cam_id)
                            rec.event(r.cam_id, "stall", f"нет кадров {now - r.last_frame_t:.1f} с")
                    else:
                        stalled.discard(r.cam_id)
            time.sleep(min(1.0, max(0.0, t_end + drain - time.monotonic())))

        self.stop.set()
        for q in queues.values():
            if isinstance(q, DynamicBatcher):
                q.close()
        if "track" in self.stages:
            self.track_q.put(None)
        for t in threads:
            t.join(timeout=5)

        out = rec.to_dict()
        out["worker_idx"] = self.idx
        out["pid"] = os.getpid()
        out["frame_device"] = next((r.device_checked for r in readers if r.device_checked), None)
        out["connected_at_end"] = sum(r.connected for r in readers)
        return out

    # ---------- стадии ----------
    def _on_frame(self, cam, idx, tensor, t_recv):
        if not self.stages:
            self.rec.done(cam, t_recv, time.monotonic())
            return
        if not self.det_q.put(cam, FrameJob(cam, idx, t_recv, tensor)):
            self.rec.dropped(cam, t_recv, "detect")

    def _finish(self, j: FrameJob) -> None:
        j.rgb = None
        if j.dropped:
            self.rec.dropped(j.cam, j.t_recv, "id")
        else:
            self.rec.done(j.cam, j.t_recv, time.monotonic(), j.persons, j.tracks)

    def _stage_done(self, j: FrameJob, ok: bool = True) -> None:
        with self._job_lock:
            j.pending -= 1
            if not ok:
                j.dropped = True
            last = j.pending == 0
        if last:
            self._finish(j)

    def _fail(self, stage: str, e: BaseException) -> None:
        self.rec.error(stage, "cuda_oom" if _is_oom(e) else repr(e)[:300])
        self.rec.count(f"error_{stage}")

    def _detect_loop(self):
        keep_rgb = "reid" in self.stages or "face" in self.stages
        track = "track" in self.stages
        while True:
            batch = self.det_q.get_batch()
            if batch is None:
                return
            if not batch:
                continue
            t = time.monotonic()
            jobs = [j for _, j in batch]
            try:
                rgb, dets = self.det_stage.run([j.nv12 for j in jobs])
            except Exception as e:  # noqa: BLE001
                self._fail("detect", e)
                for j in jobs:
                    j.nv12 = None
                    self.rec.dropped(j.cam, j.t_recv, "detect")
                continue
            now = time.monotonic()
            self.rec.count("det_frames", len(jobs))
            self.rec.count("det_batches")
            for i, j in enumerate(jobs):
                j.nv12 = None
                j.dets = dets[i]
                j.persons = len(dets[i])
                if keep_rgb:
                    j.rgb = rgb[i]
                if track:
                    self.track_q.put(j)
                else:
                    self.rec.done(j.cam, j.t_recv, now, persons=j.persons)
            self.rec.add_busy("detect", time.monotonic() - t, t)

    def _track_loop(self):
        from pipeline.preprocess import upper_squares

        reid = "reid" in self.stages
        face = "face" in self.stages
        min_h = float(self.pcfg.get("face_min_person_h", 0))
        while True:
            j = self.track_q.get()
            if j is None:
                return
            t = time.monotonic()
            tr = self.trackers[j.cam].update(j.dets)
            j.dets = None
            j.tracks = len(tr.ids)
            self.rec.count("new_tracks", int(tr.is_new.sum()))
            reqs = []
            if reid and len(tr.ids):
                sel = self.reid_sched[j.cam].select(tr.ids)
                if sel.any():
                    reqs.append(("reid", tr.boxes[sel], tr.ids[sel]))
            if face and len(tr.ids):
                elig = (tr.boxes[:, 3] - tr.boxes[:, 1]) >= min_h
                ids_e, boxes_e = tr.ids[elig], tr.boxes[elig]
                if len(ids_e):
                    sel = self.face_sched[j.cam].select(ids_e)
                    if sel.any():
                        h, w = j.rgb.shape[-2:]
                        reqs.append(("face", upper_squares(boxes_e[sel], w, h), ids_e[sel]))
            if not reqs:
                self._finish(j)
            else:
                j.pending = len(reqs)
                for kind, boxes, ids in reqs:
                    q = self.reid_q if kind == "reid" else self.face_q
                    if not q.put(j.cam, (j, boxes, ids)):
                        (self.reid_sched if kind == "reid" else self.face_sched)[j.cam].confirm(ids, False)
                        self._stage_done(j, ok=False)
            self.rec.add_busy("track", time.monotonic() - t, t)

    def _id_loop(self, kind: str):
        q = self.reid_q if kind == "reid" else self.face_q
        sched = self.reid_sched if kind == "reid" else self.face_sched
        while True:
            batch = q.get_batch()
            if batch is None:
                return
            if not batch:
                continue
            t = time.monotonic()
            items = [it for _, it in batch]
            frames = [j.rgb for j, _, _ in items]
            boxes = [b for _, b, _ in items]
            try:
                if kind == "reid":
                    self.reid_stage.run(frames, boxes)
                    oks = [True] * len(items)
                    self.rec.count("reid_crops", sum(len(b) for b in boxes))
                else:
                    oks, n_found = self.face_stage.run(frames, boxes)
                    self.rec.count("face_det_crops", sum(len(b) for b in boxes))
                    self.rec.count("face_emb_crops", n_found)
                self.rec.count(f"{kind}_batches")
            except Exception as e:  # noqa: BLE001
                self._fail(kind, e)
                for j, _, ids in items:
                    sched[j.cam].confirm(ids, False)
                    self._stage_done(j, ok=False)
                continue
            for (j, _, ids), ok in zip(items, oks):
                sched[j.cam].confirm(ids, ok)
                self._stage_done(j)
            self.rec.add_busy(kind, time.monotonic() - t, t)

    def _reid_loop(self):
        self._id_loop("reid")

    def _face_loop(self):
        self._id_loop("face")


def worker_main(args: dict, conn) -> None:
    """Точка входа процесса (multiprocessing spawn)."""
    try:
        w = Worker(args)
        w.init()
    except Exception:  # noqa: BLE001
        conn.send(("error", traceback.format_exc()))
        conn.close()
        os._exit(1)
    conn.send(("ready", os.getpid()))
    try:
        timing = conn.recv()
        result = w.run(timing)
        out_dir = Path(args["out_dir"])
        np.save(out_dir / f"lat_{w.idx}.npy", np.frombuffer(w.rec.latencies, dtype=np.float64))
        path = out_dir / f"worker_{w.idx}.json"
        path.write_text(json.dumps(result))
        conn.send(("done", str(path)))
    except Exception:  # noqa: BLE001
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()
        # потоки чтения RTSP могут висеть в блокирующем вызове — не ждём их
        os._exit(0)
