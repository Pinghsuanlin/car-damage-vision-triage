"""Synthetic Supervisely-style dataset built inside the tests: no real photos needed."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from damage_triage.data import DAMAGE_CLASSES

PART_CLASSES = ["Front-door", "Hood", "Fender"]


def _image(path: Path, seed: int, size=(96, 64)) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _dataset(root: Path, name: str, classes: list[str], n: int, seed: int, duplicate_of: dict[int, int] | None = None) -> None:
    rng = random.Random(seed)
    (root / name / "File1" / "img").mkdir(parents=True)
    (root / name / "File1" / "ann").mkdir(parents=True)
    (root / name / "meta.json").write_text(json.dumps({"classes": [{"title": c, "shape": "polygon"} for c in classes]}))
    for i in range(n):
        img = root / name / "File1" / "img" / f"Car damages {i}.png"
        src = (duplicate_of or {}).get(i)
        if src is not None:
            Image.open(root / name / "File1" / "img" / f"Car damages {src}.png").resize((90, 60)).save(img)
        else:
            _image(img, seed * 1000 + i)
        k = rng.randint(1, 3)
        chosen = rng.sample(classes, k) if i % 4 else [classes[i % len(classes)]]
        objects = [{"classTitle": c, "points": {"exterior": [[0, 0], [5, 0], [5, 5]], "interior": []}} for c in chosen]
        (root / name / "File1" / "ann" / f"{img.name}.json").write_text(
            json.dumps({"size": {"width": 96, "height": 64}, "objects": objects, "tags": []}))


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Mimics the public release, including its swapped folder names."""
    root = tmp_path / "raw"
    _dataset(root, "Car_parts_dataset", list(DAMAGE_CLASSES), 40, seed=1, duplicate_of={39: 0})
    _dataset(root, "Car_damages_dataset", PART_CLASSES, 6, seed=2)
    return root
