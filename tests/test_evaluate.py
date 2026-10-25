from __future__ import annotations

import numpy as np
import pandas as pd

from damage_triage.data import label_columns
from damage_triage.evaluate import (
    average_precision,
    best_threshold,
    compare,
    coverage_at_cutoff,
    expected_calibration_error,
    prob_columns,
    triage_curve,
    uncertainty,
)


def test_average_precision_matches_hand_calculation() -> None:
    y = np.array([1, 0, 1, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.3, 0.1])
    assert abs(average_precision(y, p) - (1 + 2 / 3 + 3 / 4) / 3) < 1e-9
    assert average_precision(np.zeros(3), np.ones(3)) is None


def test_best_threshold_separates_clean_classes() -> None:
    y = np.array([0, 0, 1, 1])
    t = best_threshold(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 < t <= 0.8


def test_calibration_perfect_and_bad() -> None:
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) < 1e-9
    assert expected_calibration_error(y, y * 0.0 + 0.95) > 0.4


def test_uncertainty_and_triage_curve() -> None:
    t = np.array([0.5, 0.5])
    probs = np.array([[0.99, 0.01], [0.55, 0.02], [0.5, 0.5]])
    u = uncertainty(probs, t)
    assert u[0] < u[1] < u[2] and abs(u[2] - 1) < 1e-9
    y = np.array([[1, 0], [0, 0], [1, 1]])
    curve = triage_curve(y, probs, t)
    assert curve["coverage"].is_monotonic_increasing and curve["error"].iloc[0] == 0
    coverage, error = coverage_at_cutoff(y, probs, t, cutoff=float(u[0]))
    assert coverage == 1 / 3 and error == 0


def test_compare_writes_markdown(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n = 80
    labels = (rng.random((n, 8)) < 0.3).astype(int)
    images = [f"img/{i}.png" for i in range(n)]
    splits = pd.DataFrame(labels, columns=label_columns())
    splits.insert(0, "image", images)
    splits["split"] = ["val"] * 40 + ["test"] * 40
    splits.to_csv(tmp_path / "splits.csv", index=False)
    run = tmp_path / "results" / "toy_run"
    run.mkdir(parents=True)
    probs = np.clip(labels * 0.5 + rng.random((n, 8)) * 0.5, 0, 1)
    for name, sl in (("val", slice(0, 40)), ("test", slice(40, 80)), ("test_blur", slice(40, 80))):
        frame = pd.DataFrame(probs[sl], columns=prob_columns())
        frame.insert(0, "image", images[sl])
        frame.to_csv(run / f"preds_{name}.csv", index=False)
    table = compare(tmp_path / "results", tmp_path / "splits.csv", bootstrap_iters=50)
    assert table.loc[0, "run"] == "toy_run" and 0 <= table.loc[0, "macro_ap"] <= 1
    text = (tmp_path / "results" / "comparison.md").read_text()
    assert "Macro AP" in text and "blur" in text and (run / "per_class.csv").exists()


def test_paired_difference_and_chance_columns(tmp_path) -> None:
    from damage_triage.evaluate import class_triage, paired_difference

    rng = np.random.default_rng(1)
    y = (rng.random((100, 3)) < [0.5, 0.3, 0.05]).astype(int)
    good = np.clip(y * 0.8 + rng.random(y.shape) * 0.4, 0, 1)
    noise = rng.random(y.shape)
    diff, lo, hi = paired_difference(y, good, noise, iters=100)
    assert lo > 0 and lo <= diff <= hi
    # A class where "always absent" is already within the error limit is reported as n/a, not as 100% automated.
    out = class_triage(y[:50], good[:50], y[50:], good[50:], np.array([0.5, 0.5, 0.5]), max_error=0.10)
    assert np.isnan(out.loc[2, "auto_coverage"]) and out.loc[0, "auto_coverage"] >= 0
