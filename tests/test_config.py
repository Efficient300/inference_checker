from bench.config import Combo, confirm_combos, expand_matrix, load_config


def test_full_matrix():
    cfg = load_config("configs/matrix.yaml")
    combos = expand_matrix(cfg)
    ids = [c.id for c in combos]
    assert len(ids) == len(set(ids))
    by = lambda **kw: [c for c in combos if all(getattr(c, k) == v for k, v in kw.items())]  # noqa: E731
    # h264: S0 4 + S1 8 + S2 8 + S3 16 + S4 16 = 52; h265: S0 4 + S4 16; det1280: S1 8
    assert len(by(codec="h264", detector_input=None)) == 4
    assert len(by(codec="h264", scenario="S3")) == 16
    assert len(by(codec="h265")) == 20
    assert len(by(detector_input=1280)) == 8
    assert len(combos) == 80
    assert all(c.id_mode is None for c in by(scenario="S1"))
    assert all(c.precision is None and c.id_mode is None for c in by(scenario="S0"))


def test_confirm_combos():
    cfg = load_config("configs/matrix.yaml")
    cc = confirm_combos(cfg, expand_matrix(cfg))
    assert len(cc) == 8          # S4 × fp16/int8 × 10/25 × 2 режима идентификации
    assert all(c.codec == "h264" and c.scenario == "S4" for c in cc)


def test_combo_roundtrip():
    c = Combo("S4", 10, "h264", "int8", "every_k_5", 640)
    assert Combo.from_dict(c.to_dict()) == c
    assert c.id == "S4_h264_fps10_int8_det640_every_k_5"


def test_smoke_inherits_base():
    cfg = load_config("configs/smoke.yaml")
    assert cfg["models"]["detector"]["onnx"].endswith("yolo11s.onnx")
    assert cfg["models"]["detector"]["max_batch"] == 16
    assert len(expand_matrix(cfg)) == 5
