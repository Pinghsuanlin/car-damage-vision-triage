"""PyTorch parts: models, training, prediction, embeddings for grouping, and Grad-CAM heatmaps.

Two ways to adapt a pretrained image model (trained on ImageNet's 1.2M everyday photos) to car damage:

* **linear probe**: keep the pretrained network frozen and train only a small final layer on top. Cheap
  and hard to overfit; shows how much the general-purpose features already know about damage.
* **fine-tune**: also update the pretrained weights, with a smaller learning rate than the new final
  layer so the useful pretrained features are adjusted rather than overwritten.

Runs on Apple Silicon (MPS), NVIDIA (CUDA) or CPU, picked automatically.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from damage_triage.data import DAMAGE_CLASSES, label_columns
from damage_triage.evaluate import average_precision, prob_columns

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- data


def transforms_for(train: bool, size: int = 224):
    """Training adds random crops, flips and colour changes so the model sees varied versions of each
    photo; evaluation uses one fixed centre crop so scores are repeatable."""
    from torchvision import transforms as T

    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size), T.ToTensor(), norm])


class DamageDataset(Dataset):
    """Images and 0/1 label vectors from ``splits.csv``; an optional degradation is applied first."""

    def __init__(self, frame: pd.DataFrame, data_dir: str | Path, transform,
                 degrade: Optional[Callable[[Image.Image], Image.Image]] = None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.degrade = degrade
        self.labels = self.frame[label_columns()].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        with Image.open(self.data_dir / self.frame["processed"].iloc[i]) as im:
            image = im.convert("RGB")
        if self.degrade is not None:
            image = self.degrade(image)
        return self.transform(image), torch.from_numpy(self.labels[i])


# --------------------------------------------------------------------------- model


def build_model(name: str = "efficientnet_b0", num_classes: int = len(DAMAGE_CLASSES), pretrained: bool = True) -> nn.Module:
    """A timm model with a fresh final layer for the damage classes."""
    import timm

    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def set_trainable(model: nn.Module, mode: str) -> list[dict]:
    """Freeze or unfreeze layers and return optimiser parameter groups for ``linear`` or ``finetune``."""
    head = model.get_classifier()
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        p.requires_grad = mode == "finetune" or id(p) in head_ids
    head_params = [p for p in model.parameters() if id(p) in head_ids]
    if mode == "linear":
        return [{"params": head_params, "lr_scale": 1.0}]
    body = [p for p in model.parameters() if id(p) not in head_ids]
    return [{"params": body, "lr_scale": 0.1}, {"params": head_params, "lr_scale": 1.0}]


@dataclass
class TrainConfig:
    model: str = "efficientnet_b0"
    mode: str = "finetune"  # or "linear"
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    image_size: int = 224
    patience: int = 4
    seed: int = 7
    pretrained: bool = True
    num_workers: int = 0


def pos_weights(labels: np.ndarray, cap: float = 10.0) -> torch.Tensor:
    """Up-weight rare damage types in the loss (negatives / positives, capped) so they are not ignored."""
    pos = labels.sum(axis=0)
    neg = len(labels) - pos
    return torch.tensor(np.clip(neg / np.clip(pos, 1, None), 1.0, cap), dtype=torch.float32)


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(DAMAGE_CLASSES)))


def _macro_ap(y: np.ndarray, p: np.ndarray) -> float:
    aps = [average_precision(y[:, k], p[:, k]) for k in range(y.shape[1])]
    aps = [a for a in aps if a is not None]
    return float(np.mean(aps)) if aps else 0.0


def train(splits: pd.DataFrame, data_dir: str | Path, out_dir: str | Path, cfg: TrainConfig,
          device: Optional[torch.device] = None, log: Callable[[str], None] = print) -> dict:
    """Train on the train split, keep the epoch with the best validation macro AP, save it.

    Stops early when validation macro AP has not improved for ``patience`` epochs. The test split is
    not touched here.
    """
    seed_everything(cfg.seed)
    device = device or pick_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = splits[splits["split"] == "train"]
    va = splits[splits["split"] == "val"]
    train_ds = DamageDataset(tr, data_dir, transforms_for(True, cfg.image_size))
    val_ds = DamageDataset(va, data_dir, transforms_for(False, cfg.image_size))
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                          drop_last=len(train_ds) > cfg.batch_size)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(cfg.model, pretrained=cfg.pretrained).to(device)
    groups = set_trainable(model, cfg.mode)
    optimiser = torch.optim.AdamW(
        [{"params": g["params"], "lr": cfg.lr * g["lr_scale"]} for g in groups], weight_decay=cfg.weight_decay
    )
    steps = max(1, cfg.epochs * math.ceil(len(train_ds) / cfg.batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=[g["lr"] for g in optimiser.param_groups], total_steps=steps, pct_start=0.15
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights(train_ds.labels).to(device))

    best, best_epoch, history, wait = -1.0, -1, [], 0
    start = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        if cfg.mode == "linear":  # keep frozen batch-norm statistics fixed too
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
            scheduler.step()
            total += float(loss) * len(x)
        val_ap = _macro_ap(val_ds.labels, predict_probs(model, val_dl, device))
        history.append({"epoch": epoch + 1, "train_loss": total / len(train_ds), "val_macro_ap": val_ap})
        log(f"epoch {epoch + 1:2d}  loss {total / len(train_ds):.4f}  val macro AP {val_ap:.3f}")
        if val_ap > best:
            best, best_epoch, wait = val_ap, epoch + 1, 0
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            wait += 1
            if wait >= cfg.patience:
                log(f"stopping early: no improvement for {cfg.patience} epochs")
                break
    summary = {"config": asdict(cfg), "device": str(device), "best_epoch": best_epoch,
               "best_val_macro_ap": best, "minutes": round((time.time() - start) / 60, 1), "history": history}
    (out_dir / "train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_trained(run_dir: str | Path, device: Optional[torch.device] = None) -> tuple[nn.Module, TrainConfig]:
    run_dir = Path(run_dir)
    cfg = TrainConfig(**json.loads((run_dir / "train.json").read_text(encoding="utf-8"))["config"])
    model = build_model(cfg.model, pretrained=False)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    return model.to(device or pick_device()).eval(), cfg


def write_predictions(model: nn.Module, frame: pd.DataFrame, data_dir: str | Path, out_csv: str | Path,
                      image_size: int = 224, degrade=None, device: Optional[torch.device] = None,
                      batch_size: int = 32) -> Path:
    """Probabilities per class for every image in ``frame`` (one CSV row per image)."""
    device = device or next(model.parameters()).device
    ds = DamageDataset(frame, data_dir, transforms_for(False, image_size), degrade=degrade)
    probs = predict_probs(model, DataLoader(ds, batch_size=batch_size), device)
    out = pd.DataFrame(probs, columns=prob_columns())
    out.insert(0, "image", frame["image"].to_numpy())
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv


# --------------------------------------------------------------------------- embeddings for grouping


@torch.no_grad()
def embed_images(paths: list[Path], model_name: str = "resnet50", device: Optional[torch.device] = None,
                 batch_size: int = 32, pretrained: bool = True) -> np.ndarray:
    """Pretrained image features (no final layer) used to find other photos of the same vehicle."""
    import timm

    device = device or pick_device()
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0).to(device).eval()
    tf = transforms_for(False)
    feats = []
    for start in range(0, len(paths), batch_size):
        batch = []
        for p in paths[start:start + batch_size]:
            with Image.open(p) as im:
                batch.append(tf(im.convert("RGB")))
        feats.append(model(torch.stack(batch).to(device)).float().cpu().numpy())
    return np.concatenate(feats)


# --------------------------------------------------------------------------- Grad-CAM


def gradcam(model: nn.Module, image: Image.Image, class_index: int, image_size: int = 224) -> np.ndarray:
    """Heatmap (values 0-1, image_size x image_size) of where the model looked for one damage type.

    Grad-CAM: weight the last convolutional feature maps by how much each one raises that class's
    score, then keep the positive part. Works for convolutional timm models (EfficientNet, ConvNeXt).
    """
    device = next(model.parameters()).device
    x = transforms_for(False, image_size)(image.convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    feats = model.forward_features(x)
    feats.retain_grad()
    logits = model.forward_head(feats)
    model.zero_grad()
    logits[0, class_index].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(image_size, image_size), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().float().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def overlay(image: Image.Image, cam: np.ndarray, image_size: int = 224, alpha: float = 0.45) -> Image.Image:
    """Red heatmap blended over the centre-cropped photo, as the model saw it."""
    from torchvision import transforms as T

    base = T.Compose([T.Resize(int(image_size * 256 / 224)), T.CenterCrop(image_size)])(image.convert("RGB"))
    heat = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    heat[..., 0] = (cam * 255).astype(np.uint8)
    return Image.blend(base, Image.fromarray(heat), alpha)
