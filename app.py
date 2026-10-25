"""Streamlit demo: upload a car photo, see damage probabilities, the automate/review decision and a heatmap.

Run with ``make app`` after training and ``make compare`` (which stores the thresholds).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from damage_triage.data import DAMAGE_CLASSES
from damage_triage.evaluate import uncertainty

RUN = Path("results/efficientnet_b0_finetune")

st.set_page_config(page_title="Car damage triage", layout="wide")
st.title("Car damage triage")
st.caption("Fine-tuned image model on the Humans in the Loop car damage dataset (CC0). Demo only, not for real claims.")

run = Path(st.sidebar.text_input("Run folder", str(RUN)))
if not (run / "model.pt").exists() or not (run / "per_class.csv").exists():
    st.warning(f"No trained model in {run}. Run `make train predict compare` first.")
    st.stop()


@st.cache_resource
def load(run_dir: str):
    from damage_triage import vision

    model, cfg = vision.load_trained(run_dir)
    per_class = pd.read_csv(Path(run_dir) / "per_class.csv")
    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    return model, cfg, per_class["threshold"].to_numpy(), metrics


model, cfg, thresholds, metrics = load(str(run))
cutoff = st.sidebar.slider("Review if uncertainty above", 0.0, 1.0, 0.5, 0.05)
upload = st.file_uploader("Car photo", type=["jpg", "jpeg", "png"])
if upload is None:
    st.stop()

from damage_triage import vision

image = Image.open(upload).convert("RGB")
import torch

x = vision.transforms_for(False, cfg.image_size)(image).unsqueeze(0).to(next(model.parameters()).device)
with torch.no_grad():
    probs = torch.sigmoid(model(x)).cpu().numpy()[0]
u = float(uncertainty(probs[None, :], thresholds)[0])
table = pd.DataFrame({"damage": DAMAGE_CLASSES, "probability": probs.round(3), "threshold": thresholds.round(3),
                      "detected": probs >= thresholds})
left, right = st.columns(2)
with left:
    st.image(image, use_container_width=True)
    decision = "Automate" if u <= cutoff else "Send to a person"
    st.metric("Decision", decision, f"uncertainty {u:.2f}")
    st.dataframe(table, hide_index=True, use_container_width=True)
with right:
    k = int(np.argmax(probs))
    st.subheader(f"Where the model looked for: {DAMAGE_CLASSES[k]}")
    cam = vision.gradcam(model, image, k, cfg.image_size)
    st.image(vision.overlay(image, cam, cfg.image_size), use_container_width=True)
