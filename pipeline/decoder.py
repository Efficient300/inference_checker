"""RTSP → NVDEC (PyNvVideoCodec) → кадр NV12 в памяти GPU.

Каждая камера читается своим потоком CameraReader: демультиплексирование, аппаратное
декодирование, прореживание до target_fps, копия выбранного кадра в torch-тензор на GPU.
При обрыве — переподключение с экспоненциальной паузой; все события пишутся в Recorder.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from pipeline.fps_limiter import FpsLimiter


class NvRtspDecoder:
    def __init__(self, url: str, gpu_id: int = 0):
        import PyNvVideoCodec as nvc

        self.nvc = nvc
        self.url = url
        self.demuxer = nvc.CreateDemuxer(filename=url)
        self.decoder = nvc.CreateDecoder(gpuid=gpu_id, codec=self.demuxer.GetNvCodecId(),
                                         cudacontext=0, cudastream=0, usedevicememory=True)

    def frames(self):
        """Генератор (decoded_frame, t_recv). t_recv — время получения пакета, из которого вышел кадр."""
        recv_by_pts: dict[int, float] = {}
        for packet in self.demuxer:
            t_recv = time.monotonic()
            pts = getattr(packet, "pts", None)
            if pts is not None:
                recv_by_pts[pts] = t_recv
                if len(recv_by_pts) > 256:
                    recv_by_pts.pop(next(iter(recv_by_pts)))
            for frame in self.decoder.Decode(packet):
                ts = getattr(frame, "timestamp", None)
                yield frame, recv_by_pts.pop(ts, t_recv) if ts is not None else t_recv


def frame_to_tensor(frame, stream):
    """Копия кадра декодера (буфер переиспользуется декодером) в собственный NV12-тензор [H*3/2, W]."""
    import torch

    with torch.cuda.stream(stream):
        try:
            t = torch.from_dlpack(frame)
        except Exception:
            t = torch.utils.dlpack.from_dlpack(frame.__dlpack__())
        if t.dim() == 3 and t.shape[-1] == 1:
            t = t[..., 0]
        out = t.clone(memory_format=torch.contiguous_format)
    stream.synchronize()
    return out


class CameraReader(threading.Thread):
    def __init__(self, cam_id: int, url: str, source_fps: float, target_fps: float, recorder,
                 on_frame: Callable, stop_event: threading.Event, gpu_id: int = 0, copy_frames: bool = True):
        super().__init__(name=f"cam{cam_id}", daemon=True)
        self.cam_id = cam_id
        self.url = url
        self.source_fps = source_fps
        self.limiter = FpsLimiter(source_fps, target_fps)
        self.rec = recorder
        self.on_frame = on_frame
        self.stop_event = stop_event
        self.gpu_id = gpu_id
        self.copy_frames = copy_frames
        self.last_frame_t = time.monotonic()
        self.connected = False
        self.frame_idx = 0
        self.device_checked = None

    def run(self):
        stream = None
        if self.copy_frames:
            import torch
            stream = torch.cuda.Stream()
        backoff = 0.5
        while not self.stop_event.is_set():
            try:
                dec = NvRtspDecoder(self.url, self.gpu_id)
                self.connected = True
                backoff = 0.5
                t_first, n_dec, last_lag = None, 0, 0.0
                for frame, t_recv in dec.frames():
                    if self.stop_event.is_set():
                        return
                    now = time.monotonic()
                    self.last_frame_t = now
                    if t_first is None:
                        t_first = now
                    n_dec += 1
                    self.rec.decoded(self.cam_id, now)
                    if now - last_lag >= 1.0:
                        # отставание от реального времени: растёт, если декодер не успевает
                        self.rec.lag_sample(self.cam_id, now, (now - t_first) - n_dec / self.source_fps)
                        last_lag = now
                    if not self.limiter.keep():
                        continue
                    self.frame_idx += 1
                    self.rec.selected(self.cam_id, t_recv)
                    tensor = frame_to_tensor(frame, stream) if self.copy_frames else None
                    if tensor is not None and self.device_checked is None:
                        self.device_checked = str(tensor.device)
                    self.on_frame(self.cam_id, self.frame_idx, tensor, t_recv)
                self.rec.event(self.cam_id, "eof", self.url)
            except Exception as e:  # noqa: BLE001 — любой сбой потока = переподключение
                self.rec.event(self.cam_id, "error", repr(e)[:300])
            self.connected = False
            if self.stop_event.wait(backoff):
                return
            self.rec.event(self.cam_id, "reconnect")
            backoff = min(5.0, backoff * 2)
