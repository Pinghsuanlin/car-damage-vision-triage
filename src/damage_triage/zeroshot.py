"""Zero-shot baseline with CLIP: describe each damage type in words, no training at all.

CLIP was trained on hundreds of millions of image-caption pairs, so it can score how well a photo
matches a sentence. For each damage type the photo is compared with two sentences, "a photo of a car
with a dent" and "a photo of a car with no dent"; the share going to the first is the probability.
This shows what a general vision-language model gets for free, and therefore what training adds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from damage_triage.data import DAMAGE_CLASSES
from damage_triage.evaluate import prob_columns

DESCRIPTIONS = {
    "Dent": "a dent",
    "Scratch": "a scratch",
    "Broken part": "a broken part",
    "Missing part": "a missing part",
    "Paint chip": "chipped paint",
    "Flaking": "flaking paint",
    "Corrosion": "rust",
    "Cracked": "a crack",
}


def prompts() -> list[str]:
    """Positive then negative sentence for each class, in class order."""
    out = []
    for cls in DAMAGE_CLASSES:
        d = DESCRIPTIONS[cls]
        bare = d[2:] if d.startswith("a ") else d
        out += [f"a photo of a car with {d}", f"a photo of a car with no {bare}"]
    return out


def pair_probabilities(similarity: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """Turn image-to-sentence similarities (images x 2*classes) into one probability per class."""
    s = np.asarray(similarity, dtype=float).reshape(len(similarity), -1, 2) * scale
    s = s - s.max(axis=2, keepdims=True)
    e = np.exp(s)
    return e[..., 0] / e.sum(axis=2)


def zeroshot_predictions(frame: pd.DataFrame, data_dir: str | Path, out_csv: str | Path,
                         model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k",
                         device: Optional[str] = None, batch_size: int = 32) -> Path:
    import open_clip
    import torch

    from damage_triage.vision import pick_device

    dev = torch.device(device) if device else pick_device()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(dev).eval()
    with torch.no_grad():
        text = model.encode_text(tokenizer(prompts()).to(dev)).float()
        text = text / text.norm(dim=-1, keepdim=True)
        sims = []
        for start in range(0, len(frame), batch_size):
            batch = []
            for rel in frame["processed"].iloc[start:start + batch_size]:
                with Image.open(Path(data_dir) / rel) as im:
                    batch.append(preprocess(im.convert("RGB")))
            img = model.encode_image(torch.stack(batch).to(dev)).float()
            img = img / img.norm(dim=-1, keepdim=True)
            sims.append((img @ text.T).cpu().numpy())
    probs = pair_probabilities(np.concatenate(sims))
    out = pd.DataFrame(probs, columns=prob_columns())
    out.insert(0, "image", frame["image"].to_numpy())
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv
