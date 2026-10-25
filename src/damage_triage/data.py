"""Dataset preparation: annotations to labels, near-duplicate groups, a leak-free split, resized images.

Source: the Humans in the Loop "Car parts and car damages" dataset (CC0 1.0), in Supervisely format:
``<dataset>/meta.json`` lists the classes and ``<dataset>/File1/{img,ann}`` hold images and one JSON
annotation per image (polygons with a ``classTitle``). The damage part has 814 images and 8 classes.

Two details of the public release are handled here rather than by hand:

* **The folder names are swapped.** ``Car_parts_dataset`` holds the damage annotations and
  ``Car_damages_dataset`` holds the car-part annotations. The damage folder is therefore found by
  reading each ``meta.json``, not by its name.
* **The same car can appear in several photos.** Many vehicles are shot two or three times from
  slightly different angles. If one shot lands in training and another in test, the model can
  "recognise the car" instead of the damage and the test score is inflated. Photos are grouped two
  ways and each group is kept within one split: a perceptual hash catches near-identical files, and
  (when PyTorch is installed) image embeddings from a pretrained model catch other shots of the same
  vehicle. ``make dup-check`` saves the grouped pairs as an image so the grouping can be checked by eye.

The task is multi-label: a photo can show several damage types at once, so each image gets a 0/1
label per class (does at least one polygon of that class appear?).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

DAMAGE_CLASSES: tuple[str, ...] = (
    "Dent",
    "Scratch",
    "Broken part",
    "Missing part",
    "Paint chip",
    "Flaking",
    "Corrosion",
    "Cracked",
)
SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_columns(classes: Iterable[str] = DAMAGE_CLASSES) -> list[str]:
    """Column names used for the 0/1 labels, e.g. 'y_broken_part'."""
    return ["y_" + c.lower().replace(" ", "_") for c in classes]


# --------------------------------------------------------------------------- locating the data


def find_damage_dir(raw_dir: str | Path, classes: Iterable[str] = DAMAGE_CLASSES) -> Path:
    """Return the folder holding ``ann/`` and ``img/`` for the damage annotations.

    Every sub-folder with a ``meta.json`` is checked; the one whose class list overlaps most with
    ``classes`` wins. Its ``ann``/``img`` folders may sit directly inside it or one level down
    (``File1`` in the public release).
    """
    wanted = set(classes)
    best: Optional[tuple[int, Path]] = None
    for meta in Path(raw_dir).rglob("meta.json"):
        titles = {c.get("title") for c in json.loads(meta.read_text(encoding="utf-8")).get("classes", [])}
        score = len(titles & wanted)
        if score and (best is None or score > best[0]):
            best = (score, meta.parent)
    if best is None:
        raise FileNotFoundError(f"No meta.json with damage classes under {raw_dir}")
    root = best[1]
    for candidate in [root, *sorted(p for p in root.iterdir() if p.is_dir())]:
        if (candidate / "ann").is_dir() and (candidate / "img").is_dir():
            return candidate
    raise FileNotFoundError(f"Found damage meta.json in {root} but no ann/ and img/ folders")


def load_labels(dataset_dir: str | Path, classes: Iterable[str] = DAMAGE_CLASSES) -> pd.DataFrame:
    """One row per image: relative path, size, polygon count and a 0/1 column per damage class.

    Images without any polygon of the listed classes are kept (all zeros) and flagged, because in
    this dataset every image should show damage; an all-zero row usually means a labelling gap.
    """
    dataset_dir = Path(dataset_dir)
    classes = tuple(classes)
    cols = label_columns(classes)
    rows = []
    for ann_path in sorted((dataset_dir / "ann").glob("*.json")):
        image_name = ann_path.name[: -len(".json")]
        image_path = dataset_dir / "img" / image_name
        if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        titles = [o.get("classTitle") for o in ann.get("objects", [])]
        row = {
            "image": f"{dataset_dir.name}/img/{image_name}",
            "width": ann.get("size", {}).get("width"),
            "height": ann.get("size", {}).get("height"),
            "n_polygons": sum(t in classes for t in titles),
        }
        for c, col in zip(classes, cols, strict=True):
            row[col] = int(c in titles)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No annotated images found in {dataset_dir}")
    frame["no_damage_label"] = (frame[cols].sum(axis=1) == 0).astype(int)
    return frame


# --------------------------------------------------------------------------- near-duplicates


def dhash(image: Image.Image, size: int = 8) -> int:
    """64-bit difference hash: robust to resizing and recompression, sensitive to content."""
    small = image.convert("L").resize((size + 1, size), Image.BILINEAR)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.parent[self.find(i)] = self.find(j)

    def labels(self) -> list[int]:
        roots: dict[int, int] = {}
        return [roots.setdefault(self.find(i), len(roots)) for i in range(len(self.parent))]


def duplicate_pairs(paths: list[Path], max_distance: int = 3) -> list[tuple[int, int]]:
    """Index pairs of near-identical files (difference-hash distance of at most ``max_distance`` bits).

    Kept strict on purpose: at 6-8 bits the hash starts pairing different cars with similar framing.
    """
    hashes = []
    for p in paths:
        with Image.open(p) as im:
            hashes.append(dhash(im))
    return [
        (i, j)
        for i in range(len(hashes))
        for j in range(i + 1, len(hashes))
        if hamming(hashes[i], hashes[j]) <= max_distance
    ]


def similar_pairs(features: np.ndarray, threshold: float = 0.90) -> list[tuple[int, int]]:
    """Index pairs whose image embeddings have cosine similarity of at least ``threshold``.

    Used to catch other photos of the same vehicle. The threshold is a judgement call: check the
    pairs it produces with ``make dup-check`` before trusting it.
    """
    feats = np.asarray(features, dtype=np.float32)
    feats = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)
    sims = feats @ feats.T
    i, j = np.where(np.triu(sims, k=1) >= threshold)
    return list(zip(i.tolist(), j.tolist(), strict=True))


def group_ids(n: int, pairs: Iterable[tuple[int, int]]) -> list[int]:
    """Connected components over the given pairs: one group id per image."""
    uf = _UnionFind(n)
    for i, j in pairs:
        uf.union(i, j)
    return uf.labels()


def duplicate_groups(paths: list[Path], max_distance: int = 3) -> list[int]:
    """Group near-identical files only (no embeddings)."""
    return group_ids(len(paths), duplicate_pairs(paths, max_distance))


# --------------------------------------------------------------------------- split


def stratified_group_split(
    labels: np.ndarray,
    groups: list[int],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 7,
) -> list[str]:
    """Assign each image to train/val/test, keeping groups together and balancing every label.

    Iterative stratification (Sechidis et al., 2011) at group level: groups carrying the rarest
    labels are placed first, each into the split that still "needs" that label the most. This keeps
    rare classes (e.g. 'Cracked', 67 images) present in validation and test.
    """
    rng = random.Random(seed)
    labels = np.asarray(labels, dtype=int)
    by_group: dict[int, list[int]] = defaultdict(list)
    for idx, g in enumerate(groups):
        by_group[g].append(idx)
    group_ids = list(by_group)
    rng.shuffle(group_ids)
    group_labels = {g: labels[by_group[g]].max(axis=0) for g in group_ids}
    group_size = {g: len(by_group[g]) for g in group_ids}

    total = labels.sum(axis=0).astype(float)
    target_label = np.outer(np.array(ratios), total)  # positives wanted per split x label
    target_size = np.array(ratios) * len(labels)
    need_label = target_label.copy()
    need_size = target_size.copy()
    assignment: dict[int, int] = {}
    remaining = set(group_ids)
    while remaining:
        counts = np.zeros(labels.shape[1])
        for g in remaining:
            counts += group_labels[g]
        positive = [k for k in range(labels.shape[1]) if counts[k] > 0]
        if positive:
            k = min(positive, key=lambda c: counts[c])  # rarest label still unassigned
            pool = [g for g in remaining if group_labels[g][k]]
        else:
            k, pool = None, list(remaining)
        for g in sorted(pool, key=lambda g: -group_size[g]):
            open_splits = [s for s in range(3) if need_size[s] > 0] or list(range(3))

            def score(s: int, k=k) -> tuple:
                # Relative demand, so the larger train split does not win every tie by sheer size.
                label_need = need_label[s, k] / max(target_label[s, k], 1e-9) if k is not None else 0.0
                return (label_need, need_size[s] / target_size[s], rng.random())

            s = max(open_splits, key=score)
            assignment[g] = s
            need_label[s] -= group_labels[g]
            need_size[s] -= group_size[g]
            remaining.discard(g)
    return [SPLITS[assignment[g]] for g in groups]


# --------------------------------------------------------------------------- prepare


@dataclass
class PrepareSummary:
    n_images: int
    n_groups: int
    n_duplicates: int
    n_same_vehicle: int
    split_sizes: dict[str, int]
    positives: pd.DataFrame
    pairs: list[tuple[int, int, str]]
    largest_group: int


def resize_image(src: Path, dst: Path, short_side: int = 288) -> None:
    """Save an RGB copy whose shorter side is ``short_side`` pixels (keeps training fast on a laptop)."""
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = short_side / min(w, h)
        if scale < 1:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="JPEG", quality=92)


def prepare(
    raw_dir: str | Path = "data/raw",
    out_dir: str | Path = "data",
    seed: int = 7,
    short_side: int = 288,
    max_distance: int = 3,
    features: Optional[np.ndarray] = None,
    similarity: float = 0.90,
) -> PrepareSummary:
    """Build ``data/splits.csv`` and resized copies under ``data/processed/``.

    Args:
        features: optional image embeddings (one row per image, in ``splits.csv`` order) used to
            group other photos of the same vehicle. ``cli prepare`` computes them with a pretrained
            model when PyTorch is available.

    ``splits.csv`` lists every image with its labels, group and split; it is small, contains no
    images, and is committed so that everyone trains and tests on exactly the same split.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    dataset_dir = find_damage_dir(raw_dir)
    frame = load_labels(dataset_dir)
    paths = [dataset_dir.parent / p for p in frame["image"]]
    frame["processed"] = [str(Path("processed") / (Path(p).stem + ".jpg")) for p in frame["image"]]
    for src, rel in zip(paths, frame["processed"], strict=True):
        dst = out_dir / rel
        if not dst.exists():
            resize_image(src, dst, short_side)

    pairs = [(i, j, "duplicate") for i, j in duplicate_pairs(paths, max_distance)]
    n_dup_pairs = len(pairs)
    if features is not None:
        seen = {(i, j) for i, j, _ in pairs}
        pairs += [(i, j, "same_vehicle") for i, j in similar_pairs(features, similarity) if (i, j) not in seen]
    frame["group"] = group_ids(len(frame), [(i, j) for i, j, _ in pairs])
    cols = label_columns()
    frame["split"] = stratified_group_split(frame[cols].to_numpy(), frame["group"].tolist(), seed=seed)
    frame.to_csv(out_dir / "splits.csv", index=False)
    positives = frame.groupby("split")[cols].sum().T.reindex(columns=list(SPLITS))
    positives.index = list(DAMAGE_CLASSES)
    return PrepareSummary(
        n_images=len(frame),
        n_groups=frame["group"].nunique(),
        n_duplicates=n_dup_pairs,
        n_same_vehicle=len(pairs) - n_dup_pairs,
        split_sizes=frame["split"].value_counts().reindex(list(SPLITS)).to_dict(),
        positives=positives,
        pairs=pairs,
        largest_group=int(frame["group"].value_counts().max()),
    )


def pairs_montage(
    frame: pd.DataFrame, pairs: list[tuple[int, int, str]], data_dir: str | Path, out: str | Path, limit: int = 24
) -> Path:
    """Save grouped image pairs side by side so the grouping can be checked by eye."""
    data_dir = Path(data_dir)
    tiles = []
    for i, j, _kind in pairs[:limit]:
        pair = Image.new("RGB", (410, 140), "white")
        for k, idx in enumerate((i, j)):
            with Image.open(data_dir / frame["processed"].iloc[idx]) as im:
                pair.paste(im.convert("RGB").resize((200, 140)), (k * 210, 0))
        tiles.append(pair)
    sheet = Image.new("RGB", (410, max(1, len(tiles)) * 145), "white")
    for k, tile in enumerate(tiles):
        sheet.paste(tile, (0, k * 145))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=80)
    return out
