"""Model code on tiny untrained networks (no downloads); skipped when PyTorch is not installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from damage_triage import vision
from damage_triage.data import prepare


@pytest.fixture
def prepared(raw_dir: Path, tmp_path: Path) -> Path:
    out = tmp_path / "data"
    out.mkdir()
    prepare(raw_dir, out, seed=3)
    return out


def test_train_predict_and_gradcam(prepared: Path, tmp_path: Path) -> None:
    frame = pd.read_csv(prepared / "splits.csv")
    cfg = vision.TrainConfig(model="resnet18", mode="finetune", epochs=2, batch_size=8, image_size=64,
                             pretrained=False, patience=5)
    summary = vision.train(frame, prepared, tmp_path / "run", cfg, device=torch.device("cpu"), log=lambda s: None)
    assert (tmp_path / "run" / "model.pt").exists() and summary["best_epoch"] >= 1
    model, loaded = vision.load_trained(tmp_path / "run", device=torch.device("cpu"))
    out = vision.write_predictions(model, frame[frame["split"] == "test"], prepared, tmp_path / "p.csv", loaded.image_size)
    preds = pd.read_csv(out)
    assert len(preds) == (frame["split"] == "test").sum() and preds.iloc[:, 1:].to_numpy().max() <= 1
    from PIL import Image

    image = Image.open(prepared / frame["processed"].iloc[0])
    cam = vision.gradcam(model, image, 0, loaded.image_size)
    assert cam.shape == (64, 64) and cam.min() >= 0 and cam.max() <= 1


def test_linear_mode_freezes_backbone() -> None:
    model = vision.build_model("resnet18", pretrained=False)
    groups = vision.set_trainable(model, "linear")
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(groups) == 1 and len(trainable) == len(list(model.get_classifier().parameters()))


def test_pos_weights_upweight_rare_classes() -> None:
    labels = np.array([[1, 0], [1, 0], [1, 1], [0, 0]])
    w = vision.pos_weights(labels)
    assert w[1] > w[0]
