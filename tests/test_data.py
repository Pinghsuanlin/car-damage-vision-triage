from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from damage_triage.data import (
    DAMAGE_CLASSES,
    find_damage_dir,
    group_ids,
    label_columns,
    load_labels,
    prepare,
    similar_pairs,
    stratified_group_split,
)


def test_finds_damage_folder_despite_swapped_names(raw_dir: Path) -> None:
    found = find_damage_dir(raw_dir)
    assert found.parent.name == "Car_parts_dataset" and found.name == "File1"


def test_labels_are_multi_label_zero_one(raw_dir: Path) -> None:
    frame = load_labels(find_damage_dir(raw_dir))
    cols = label_columns()
    assert len(frame) == 40 and set(np.unique(frame[cols].to_numpy())) <= {0, 1}
    assert (frame[cols].sum(axis=1) >= 1).all()
    assert frame["image"].str.startswith("File1/img/").all()


def test_split_keeps_groups_together_and_covers_classes() -> None:
    rng = np.random.default_rng(0)
    labels = (rng.random((300, 8)) < 0.25).astype(int)
    labels[np.arange(300), rng.integers(0, 8, 300)] = 1
    groups = [i // 2 for i in range(300)]  # pairs of photos of the same car
    split = stratified_group_split(labels, groups, seed=1)
    frame = pd.DataFrame({"group": groups, "split": split})
    assert (frame.groupby("group")["split"].nunique() == 1).all()
    shares = frame["split"].value_counts(normalize=True)
    assert abs(shares["train"] - 0.70) < 0.05 and abs(shares["test"] - 0.15) < 0.05
    for s in ("train", "val", "test"):
        assert labels[np.array(split) == s].sum(axis=0).min() > 0


def test_similar_pairs_and_groups() -> None:
    feats = np.array([[1.0, 0.0], [0.99, 0.05], [0.0, 1.0]])
    pairs = similar_pairs(feats, threshold=0.95)
    assert pairs == [(0, 1)]
    assert group_ids(3, pairs) == [0, 0, 1]


def test_prepare_writes_split_and_groups_duplicates(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "data"
    out.mkdir()
    summary = prepare(raw_dir, out, seed=3)
    frame = pd.read_csv(out / "splits.csv")
    assert summary.n_images == 40 and summary.n_duplicates == 1  # image 39 is a resized copy of image 0
    g = frame.set_index("image")["group"]
    assert g["File1/img/Car damages 0.png"] == g["File1/img/Car damages 39.png"]
    assert set(frame["split"]) == {"train", "val", "test"}
    assert all((out / p).exists() for p in frame["processed"])
    assert list(summary.positives.index) == list(DAMAGE_CLASSES)
