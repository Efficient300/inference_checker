"""Подготовка тестового контента: клипы 1920×1080@25, H.264 и H.265, длиной ≥ 5 минут.

Источники (любой):
  --src DIR            свои записи (любой формат ffmpeg); длинные режутся на клипы, короткие зацикливаются
  --mot20 DIR          распакованный MOT20 (последовательности кадров с seqinfo.ini)
  --download-mot20     скачать MOT20 (~5 ГБ) в data/raw и использовать его
  --smoke-image IMG    только для отладки: панорама по картинке (в итоговых результатах не использовать)

    python scripts/prepare_videos.py --mot20 data/raw/MOT20 --codecs h264,h265
"""
from __future__ import annotations

import argparse
import configparser
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

W, H, FPS = 1920, 1080, 25
MOT20_URL = "https://motchallenge.net/data/MOT20.zip"
VF = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
      f"fps={FPS},format=yuv420p")
ENCODERS = {"h264": "libx264", "h265": "libx265"}
NV_ENCODERS = {"h264": "h264_nvenc", "h265": "hevc_nvenc"}


def run(cmd: list[str]) -> None:
    print(" ".join(cmd[:12]), "…" if len(cmd) > 12 else "", flush=True)
    subprocess.run(cmd, check=True)


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    st = d["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return {"codec": st["codec_name"], "width": st["width"], "height": st["height"],
            "fps": round(int(num) / int(den), 3), "duration_s": round(float(d["format"]["duration"]), 1)}


def encode_args(codec: str, bitrate: str, nvenc: bool) -> list[str]:
    enc = (NV_ENCODERS if nvenc else ENCODERS)[codec]
    args = ["-c:v", enc, "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate, "-g", str(FPS * 2), "-bf", "0"]
    if not nvenc:
        args += ["-preset", "medium"]
        if codec == "h265":
            args += ["-tag:v", "hvc1", "-x265-params", "log-level=error"]
    else:
        args += ["-preset", "p5"]
    return args + ["-an", "-movflags", "+faststart"]


def mot20_sequences(root: Path) -> list[dict]:
    seqs = []
    for ini in sorted(root.rglob("seqinfo.ini")):
        cp = configparser.ConfigParser()
        cp.read(ini)
        s = cp["Sequence"]
        seqs.append({"name": s["name"], "dir": ini.parent / s.get("imDir", "img1"),
                     "fps": float(s["frameRate"]), "ext": s.get("imExt", ".jpg")})
    return seqs


def download_mot20(dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    z = dst / "MOT20.zip"
    if not z.exists():
        print(f"скачивание {MOT20_URL} …", flush=True)
        urllib.request.urlretrieve(MOT20_URL, z)
    out = dst / "MOT20"
    if not out.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(dst)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--src", type=Path)
    src.add_argument("--mot20", type=Path)
    src.add_argument("--download-mot20", action="store_true")
    src.add_argument("--smoke-image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/videos"))
    ap.add_argument("--codecs", default="h264,h265")
    ap.add_argument("--clip-s", type=int, default=300)
    ap.add_argument("--bitrate", default="4M")
    ap.add_argument("--nvenc", action="store_true", help="кодировать на GPU (быстрее)")
    ap.add_argument("--smoke-clips", type=int, default=2)
    args = ap.parse_args()

    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    jobs: list[tuple[str, list[str]]] = []          # (имя клипа, входные аргументы ffmpeg)
    content = "own"
    if args.download_mot20:
        args.mot20 = download_mot20(Path("data/raw"))
    if args.mot20:
        content = "MOT20"
        for s in mot20_sequences(args.mot20):
            jobs.append((s["name"], ["-framerate", str(s["fps"]), "-stream_loop", "-1",
                                     "-i", str(s["dir"] / f"%06d{s['ext']}"), "-t", str(args.clip_s)]))
    elif args.src:
        files = sorted(p for p in args.src.rglob("*") if p.is_file())
        for f in files:
            try:
                dur = ffprobe(f)["duration_s"]
            except Exception:  # noqa: BLE001
                continue
            n = max(1, int(dur // args.clip_s))
            for k in range(n):
                if dur >= args.clip_s:
                    inp = ["-ss", str(k * args.clip_s), "-i", str(f), "-t", str(args.clip_s)]
                else:
                    inp = ["-stream_loop", "-1", "-i", str(f), "-t", str(args.clip_s)]
                jobs.append((f"{f.stem}_{k:02d}", inp))
    else:
        content = "smoke (не для итоговых результатов)"
        for k in range(args.smoke_clips):
            # медленная панорама с разной фазой: люди на картинке движутся в кадре
            pan = f"zoompan=z=1.15:x='(iw-iw/zoom)*(0.5+0.5*sin(on/{60 + 17 * k}))':y='(ih-ih/zoom)/2':d=1:s={W}x{H}:fps={FPS}"
            jobs.append((f"smoke_{k:02d}", ["-loop", "1", "-framerate", str(FPS), "-i", str(args.smoke_image),
                                            "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},{pan}",
                                            "-t", str(args.clip_s)]))
    if not jobs:
        print("не найдено исходных видео", file=sys.stderr)
        return 1

    manifest = {"content": content, "clips": []}
    for codec in codecs:
        d = args.out / codec
        d.mkdir(parents=True, exist_ok=True)
        for name, inp in jobs:
            out = d / f"{name}.mp4"
            if not out.exists():
                vf = [] if "-vf" in inp else ["-vf", VF]
                run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inp, *vf, "-r", str(FPS),
                     *encode_args(codec, args.bitrate, args.nvenc), str(out)])
            info = ffprobe(out)
            info["file"] = str(out)
            ok = (info["width"], info["height"]) == (W, H) and abs(info["fps"] - FPS) < 0.01 \
                and info["duration_s"] >= args.clip_s - 1
            info["ok"] = ok
            manifest["clips"].append(info)
            print(f"  {'OK ' if ok else 'BAD'} {out}: {info}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    bad = [c for c in manifest["clips"] if not c["ok"]]
    per_codec = {c: sum(1 for x in manifest["clips"] if x["ok"] and f"/{c}/" in x["file"]) for c in codecs}
    print(f"клипов по кодекам: {per_codec}; с ошибками: {len(bad)}")
    if any(v < 8 for v in per_codec.values()):
        print("ВНИМАНИЕ: меньше 8 разных клипов на кодек (требование T2)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
