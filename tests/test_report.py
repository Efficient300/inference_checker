from tests.test_search import workdir  # noqa: F401 — фикстура

from bench.report import main as report
from bench.run_matrix import main as run_matrix


def test_report_from_simulated(workdir):  # noqa: F811
    cfg = workdir / "configs/sim.yaml"
    cfg.write_text(cfg.read_text().replace("confirm: {scenarios: []}",
                                           "confirm: {scenarios: [S2], precision: [fp16], target_fps: [10], duration_s: 120}"))
    assert run_matrix(["--config", "configs/sim.yaml", "--simulate", "--run-id", "rep"]) == 0
    # без флага имитация в отчёт не попадает
    report(["--out", "R0.md"])
    assert "Нет завершённых" in (workdir / "R0.md").read_text()

    assert report(["--out", "R.md", "--allow-simulated"]) == 0
    md = (workdir / "R.md").read_text()
    assert "ИМИТАЦИЯ" in md
    assert "N_max по матрице" in md and "p95 ≤ 200 мс" in md and "Ограничивающий ресурс" in md
    assert "Подтверждающие прогоны" in md
    assert "`rep/S2_h264_fps10_fp16_det640`" in md
    pngs = list((workdir / "report").glob("*.png"))
    assert len(pngs) == 4          # (S0, S2) × (метрики от N, кривая FPS↔N_max)
