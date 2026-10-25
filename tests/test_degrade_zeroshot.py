from __future__ import annotations

import numpy as np
from PIL import Image

from damage_triage.degrade import DEGRADATIONS
from damage_triage.zeroshot import pair_probabilities, prompts


def test_degradations_keep_size_and_are_deterministic() -> None:
    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((64, 96, 3)) * 255).astype(np.uint8))
    for name, fn in DEGRADATIONS.items():
        a, b = fn(img), fn(img)
        assert a.size == img.size, name
        assert np.array_equal(np.asarray(a), np.asarray(b)), name


def test_prompts_and_pair_probabilities() -> None:
    p = prompts()
    assert len(p) == 16 and p[0] == "a photo of a car with a dent" and p[1] == "a photo of a car with no dent"
    assert "a photo of a car with no chipped paint" in p
    sims = np.array([[0.30, 0.20] * 8, [0.20, 0.30] * 8])
    probs = pair_probabilities(sims)
    assert probs.shape == (2, 8) and (probs[0] > 0.5).all() and (probs[1] < 0.5).all()
