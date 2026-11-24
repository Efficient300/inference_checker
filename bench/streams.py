"""Управление тестовыми RTSP-потоками (start_streams.sh + проверка готовности через API MediaMTX)."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
STREAMER = ROOT / "streamer"


class StreamManager:
    def __init__(self, cfg: dict):
        s = cfg["streamer"]
        self.mode = s["mode"]
        self.rtsp_base = s["rtsp_base"].rstrip("/")
        self.api = (s.get("api") or "").rstrip("/")
        self.cores = s.get("cores") or ""
        self.ready_timeout = float(s.get("ready_timeout_s", 60))
        self.video_pattern = cfg["source"]["video_files"]
        self.pidfile = STREAMER / ".pids"
        self.running = 0

    def url(self, i: int) -> str:
        return f"{self.rtsp_base}/cam{i}"

    def check_server(self) -> None:
        if self.api:
            try:
                self._paths()
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"MediaMTX API недоступен ({self.api}): docker compose -f streamer/docker-compose.yml up -d") from e

    def _paths(self) -> dict[str, bool]:
        with urllib.request.urlopen(f"{self.api}/v3/paths/list?itemsPerPage=100000", timeout=5) as r:
            data = json.load(r)
        return {it["name"]: bool(it.get("ready")) for it in data.get("items", [])}

    def start(self, n: int, codec: str) -> None:
        if self.mode == "external":
            self.wait_ready(n)
            return
        videos = str(Path(self.video_pattern.format(codec=codec)).parent)
        cmd = ["bash", str(STREAMER / "start_streams.sh"), str(n), "--videos", videos,
               "--rtsp", self.rtsp_base, "--pidfile", str(self.pidfile)]
        if self.cores:
            cmd += ["--cores", self.cores]
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        self.running = n
        self.wait_ready(n)

    def wait_ready(self, n: int) -> None:
        if not self.api:
            time.sleep(5)
            return
        deadline = time.monotonic() + self.ready_timeout
        while True:
            paths = self._paths()
            missing = [i for i in range(n) if not paths.get(f"cam{i}")]
            if not missing:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"не готовы потоки: {['cam%d' % i for i in missing[:10]]} …")
            time.sleep(0.5)

    def stop(self) -> None:
        if self.mode == "external":
            return
        subprocess.run(["bash", str(STREAMER / "stop_streams.sh"), "--pidfile", str(self.pidfile)],
                       cwd=ROOT, stdout=subprocess.DEVNULL, check=False)
        self.running = 0

    def pids(self) -> list[int]:
        """PID ffmpeg-публикаторов и MediaMTX — их CPU пишется отдельно от бенчмарка."""
        out = []
        if self.pidfile.exists():
            out += [int(x) for x in self.pidfile.read_text().split() if x.strip().isdigit()]
        for p in psutil.process_iter(["name"]):
            if (p.info.get("name") or "").startswith("mediamtx"):
                out.append(p.pid)
        return out
