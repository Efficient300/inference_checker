"""Черновик отчёта из results/**/*.json: таблица N_max, N при p95 ≤ порогов, графики.

    python -m bench.report [--runs run_a,run_b] [--out REPORT_DRAFT.md] [--allow-simulated]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path("results")
PHASE_RANK = {"refine": 2, "coarse": 1}


def load_results(runs: list[str] | None, allow_simulated: bool) -> tuple[dict, list[str]]:
    """Последний и самый точный результат по каждой комбинации: {combo_id: state}."""
    best: dict[str, dict] = {}
    skipped = []
    for p in sorted(RESULTS.glob("*/*.json")):
        if p.parent.name == "model_only" or (runs and p.parent.name not in runs):
            continue
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "combo" not in d or d.get("status") != "done":
            continue
        if d.get("simulated") and not allow_simulated:
            skipped.append(d["run_id"])
            continue
        phase = max((PHASE_RANK.get(pt.get("phase"), 0) for pt in d["points"]), default=0)
        key = (phase, d.get("finished", ""))
        cur = best.get(d["combo_id"])
        if cur is None or key > cur[0]:
            best[d["combo_id"]] = (key, d)
    return {k: v[1] for k, v in best.items()}, skipped


def search_points(d: dict) -> list[dict]:
    return [p for p in d["points"] if p.get("phase") in ("coarse", "refine") and p["valid"]]


def n_at_latency(d: dict, limit_ms: float) -> int | None:
    ok = [p["n_cams"] for p in search_points(d) if p["passed"] and p["lat_p95"] <= limit_ms]
    return max(ok) if ok else None


def row_key(c: dict) -> tuple:
    return (c["scenario"], c["codec"], c.get("detector_input") or 0, c.get("id_mode") or "", c.get("precision") or "")


def row_label(c: dict) -> str:
    parts = [c["scenario"], c["codec"]]
    if c.get("detector_input"):
        parts.append(f"det{c['detector_input']}")
    if c.get("precision"):
        parts.append(c["precision"].upper())
    if c.get("id_mode"):
        parts.append(c["id_mode"])
    return " · ".join(parts)


def fmt(x, nd=0):
    if x is None:
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def plots(results: dict, out_dir: Path) -> list[tuple[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    by_scen: dict[tuple, list[dict]] = defaultdict(list)
    for d in results.values():
        c = d["combo"]
        by_scen[(c["scenario"], c["codec"], c.get("detector_input") or 0)].append(d)

    for (scen, codec, det), ds in sorted(by_scen.items()):
        tag = f"{scen}_{codec}" + (f"_det{det}" if det else "")
        # Метрики от N
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for d in sorted(ds, key=lambda d: row_key(d["combo"]) + (d["combo"]["target_fps"],)):
            c = d["combo"]
            pts = sorted(search_points(d), key=lambda p: p["n_cams"])
            if not pts:
                continue
            n = [p["n_cams"] for p in pts]
            lbl = f"{c['target_fps']:g} fps " + " ".join(x for x in (c.get("precision"), c.get("id_mode")) if x)
            axes[0].plot(n, [p["fps_min"] / c["target_fps"] for p in pts], marker=".", label=lbl)
            axes[1].plot(n, [p["lat_p95"] for p in pts], marker=".", label=lbl)
            axes[2].plot(n, [p.get("sm_util") or 0 for p in pts], marker=".", label=lbl)
        axes[0].axhline(0.95, color="grey", ls="--", lw=1)
        axes[0].set_ylabel("min FPS камеры / целевой")
        axes[1].set_ylabel("задержка p95, мс")
        axes[1].set_yscale("log")
        for lim in (200, 500):
            axes[1].axhline(lim, color="grey", ls=":", lw=1)
        axes[2].set_ylabel("GPU util (NVML), %")
        for ax in axes:
            ax.set_xlabel("камер, N")
            ax.grid(alpha=0.3)
        axes[2].legend(fontsize=6, loc="best")
        fig.suptitle(f"{tag}: метрики от числа камер")
        fig.tight_layout()
        p = out_dir / f"{tag}_vs_n.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append((f"{tag}: FPS / задержка / GPU от N", str(p)))

        # Кривая FPS ↔ N_max
        fig, ax = plt.subplots(figsize=(7, 4.8))
        lines: dict[str, list] = defaultdict(list)
        for d in ds:
            c = d["combo"]
            lines[" ".join(x for x in (c.get("precision"), c.get("id_mode")) if x) or "decode"].append(
                (c["target_fps"], d["n_max"]))
        for lbl, xs in sorted(lines.items()):
            xs.sort()
            ax.plot([x for x, _ in xs], [y for _, y in xs], marker="o", label=lbl)
        ax.set_xlabel("FPS на камеру")
        ax.set_ylabel("N_max камер")
        ax.set_title(f"{tag}: FPS ↔ N_max")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / f"{tag}_fps_nmax.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append((f"{tag}: FPS ↔ N_max", str(p)))
    return made


def render(results: dict, skipped: list[str], images: list[tuple[str, str]], lat_limits: list[int]) -> str:
    L = ["# Черновик отчёта: ёмкость GPU по числу камер", ""]
    if not results:
        return "\n".join(L + ["Нет завершённых результатов в results/."])
    sim = any(d.get("simulated") for d in results.values())
    if sim:
        L += ["> **ИМИТАЦИЯ.** Часть данных получена `--simulate` и не является измерением. "
              "Не использовать в итоговых выводах.", ""]
    env = next(iter(results.values())).get("env", {})
    gpu = (env.get("gpu") or {}).get("gpus", [{}])
    sw = env.get("software", {})
    L += ["## Окружение", "",
          f"- GPU: {gpu[0].get('name') if gpu else '—'}, драйвер {(env.get('gpu') or {}).get('driver', '—')}, "
          f"CUDA {(env.get('gpu') or {}).get('cuda_driver', '—')}",
          f"- TensorRT {sw.get('tensorrt', '—')}, PyTorch {sw.get('torch', '—')}, PyNvVideoCodec {sw.get('PyNvVideoCodec', '—')}",
          f"- Образ: {(env.get('host') or {}).get('image', '—')}", ""]

    fps_cols = sorted({d["combo"]["target_fps"] for d in results.values()})
    rows: dict[tuple, dict] = defaultdict(dict)
    labels = {}
    for d in results.values():
        k = row_key(d["combo"])
        rows[k][d["combo"]["target_fps"]] = d
        labels[k] = row_label(d["combo"])

    L += ["## N_max по матрице", "",
          "Камер 1080p, выдержанных при заданном FPS (FPS каждой камеры ≥ 95%, пропуски ≤ 1%, очереди не растут).", "",
          "| Конфигурация | " + " | ".join(f"{f:g} FPS" for f in fps_cols) + " |",
          "|---|" + "---|" * len(fps_cols)]
    for k in sorted(rows):
        cells = []
        for f in fps_cols:
            d = rows[k].get(f)
            if not d:
                cells.append("—")
                continue
            s = str(d["n_max"]) + ("+" if d.get("capped") else "")
            conf = d.get("confirm")
            if conf:
                s += " ✔" if conf["confirmed"] else " ✘"
            cells.append(s)
        L.append(f"| {labels[k]} | " + " | ".join(cells) + " |")
    L += ["", "`+` — упёрлись в search.max_n; ✔/✘ — подтверждение 30-минутным прогоном.", ""]

    L += ["## Задержка: N при p95 ≤ порога", "",
          "| Конфигурация | FPS | " + " | ".join(f"p95 ≤ {x} мс" for x in lat_limits) + " | p95 на N_max, мс |",
          "|---|---|" + "---|" * (len(lat_limits) + 1)]
    for k in sorted(rows):
        for f in fps_cols:
            d = rows[k].get(f)
            if not d:
                continue
            at = next((p for p in search_points(d) if p["n_cams"] == d["n_max"]), None)
            L.append(f"| {labels[k]} | {f:g} | " + " | ".join(fmt(n_at_latency(d, x)) for x in lat_limits)
                     + f" | {fmt(at['lat_p95'] if at else None)} |")
    L += ["", "N берётся из измеренных точек поиска (между точками не интерполируется).", ""]

    L += ["## Ограничивающий ресурс", "",
          "| Конфигурация | FPS | N_max | Упор | Загрузка на N_max+шаг | людей/кадр | run_id |", "|---|---|---|---|---|---|---|"]
    for k in sorted(rows):
        for f in fps_cols:
            d = rows[k].get(f)
            if not d:
                continue
            top = ", ".join(f"{n} {v:.0%}" for n, v in list((d.get("limiting_ratios") or {}).items())[:3])
            at = next((p for p in search_points(d) if p["n_cams"] == d["n_max"]), None)
            L.append(f"| {labels[k]} | {f:g} | {d['n_max']} | {d.get('limiting_resource') or '—'} | {top} | "
                     f"{fmt(at['persons_per_frame'] if at else None, 1)} | `{d['run_id']}` |")
    L.append("")

    confs = [d for d in results.values() if d.get("confirm")]
    if confs:
        L += ["## Подтверждающие прогоны", "",
              "| Конфигурация | FPS | N | Длительность, мин | Прошёл | Терм. троттлинг | Итог |", "|---|---|---|---|---|---|---|"]
        for d in sorted(confs, key=lambda d: row_key(d["combo"])):
            c = d["confirm"]
            L.append(f"| {row_label(d['combo'])} | {d['combo']['target_fps']:g} | {c['n']} | {c['duration_s'] / 60:.0f} | "
                     f"{'да' if c['passed'] else 'нет'} | {c['throttle_thermal_frac']:.1%} | "
                     f"{'подтверждено' if c['confirmed'] else 'НЕ подтверждено'} |")
        L.append("")

    if images:
        L += ["## Графики", ""]
        for title, path in images:
            L += [f"### {title}", "", f"![{title}]({path})", ""]
    if skipped:
        L += [f"_Пропущено имитационных результатов: {len(skipped)} (используйте --allow-simulated)._", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="run_id через запятую (по умолчанию все)")
    ap.add_argument("--out", default="REPORT_DRAFT.md")
    ap.add_argument("--img-dir", default="report")
    ap.add_argument("--allow-simulated", action="store_true")
    args = ap.parse_args(argv)

    runs = [r.strip() for r in args.runs.split(",")] if args.runs else None
    results, skipped = load_results(runs, args.allow_simulated)
    lat = [200, 500]
    if results:
        lat = next(iter(results.values())).get("config", {}).get("latency_report_ms", lat)
    images = plots(results, Path(args.img_dir)) if results else []
    Path(args.out).write_text(render(results, skipped, images, lat))
    print(f"{args.out}: {len(results)} комбинаций, графиков {len(images)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
