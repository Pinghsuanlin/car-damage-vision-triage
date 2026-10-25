"""Scoring: per-class metrics, uncertainty ranges, calibration and the automate-vs-review trade-off.

Everything here works on prediction files (one row per image, a probability per class) and plain
NumPy, so runs from any model (zero-shot CLIP, linear probe, fine-tuned network) are scored the
same way. Thresholds are chosen on the validation split and only then applied to the test split, so
the test numbers are never tuned on.

Key terms, in plain words:

* **Average precision (AP)** per damage type: how well the model ranks photos with that damage
  above photos without it, across all thresholds. 1.0 is perfect; a random guess scores roughly the
  share of photos with that damage. **Macro AP** is the plain average over the 8 types.
* **F1** at the chosen threshold: balance of precision (flagged photos that really have it) and
  recall (photos with it that were flagged).
* **Calibration error (ECE)**: when the model says 70%, is it right about 70% of the time?
* **Triage**: a photo is "automated" when every damage type is confidently present or absent;
  otherwise it goes to a person. The curve shows how many photos can be automated at a given error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from damage_triage.data import DAMAGE_CLASSES, label_columns

PROB_PREFIX = "p_"
# A damage type only counts as having "skill" when its AP beats a no-skill guess by at least this much.
SKILL_MARGIN = 0.10


def prob_columns(classes=DAMAGE_CLASSES) -> list[str]:
    return [PROB_PREFIX + c.lower().replace(" ", "_") for c in classes]


# --------------------------------------------------------------------------- basic metrics


def average_precision(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    """Area under the precision-recall curve (step-wise, as in scikit-learn). None if no positives."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n_pos = y.sum()
    if n_pos == 0:
        return None
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y_sorted).sum() / n_pos)


def f1_at(y: np.ndarray, p: np.ndarray, t: float) -> tuple[float, float, float]:
    """Precision, recall and F1 when predicting positive for probability >= t."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(p) >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold that maximises F1 on the given (validation) data; 0.5 if the class never occurs."""
    if np.asarray(y).sum() == 0:
        return 0.5
    candidates = np.unique(np.concatenate([np.asarray(p, dtype=float), [0.5]]))
    scores = [(f1_at(y, p, t)[2], -abs(t - 0.5), t) for t in candidates]
    return float(max(scores)[2])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Average gap between stated probability and observed frequency, weighted by bin size."""
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


# --------------------------------------------------------------------------- triage


def uncertainty(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Per photo: how close its least certain damage type is to the threshold (0 = sure, 1 = coin flip)."""
    probs = np.asarray(probs, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    margin = np.where(probs >= t, (probs - t) / np.clip(1 - t, 1e-9, None), (t - probs) / np.clip(t, 1e-9, None))
    return 1 - margin.min(axis=1)


def triage_curve(y: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Coverage vs error: automate the most certain photos first, then add less certain ones.

    ``error`` is the share of automated photos with at least one wrong damage label.
    """
    y = np.asarray(y).astype(int)
    pred = (np.asarray(probs) >= np.asarray(thresholds)).astype(int)
    wrong = (pred != y).any(axis=1)
    u = uncertainty(probs, thresholds)
    order = np.argsort(u, kind="mergesort")
    n = np.arange(1, len(order) + 1)
    errors = np.cumsum(wrong[order])
    return pd.DataFrame({"uncertainty_cutoff": u[order], "coverage": n / len(order), "error": errors / n})


def cutoff_for_error(curve: pd.DataFrame, max_error: float) -> Optional[float]:
    """Largest uncertainty cutoff whose automated set stays within ``max_error`` (chosen on validation)."""
    ok = curve[curve["error"] <= max_error]
    return None if ok.empty else float(ok["uncertainty_cutoff"].max())


def coverage_at_cutoff(y: np.ndarray, probs: np.ndarray, thresholds: np.ndarray, cutoff: Optional[float]) -> tuple[float, float]:
    """Share of photos automated and their error rate when automating everything at or below ``cutoff``."""
    if cutoff is None:
        return 0.0, 0.0
    y = np.asarray(y).astype(int)
    pred = (np.asarray(probs) >= np.asarray(thresholds)).astype(int)
    auto = uncertainty(probs, thresholds) <= cutoff
    if not auto.any():
        return 0.0, 0.0
    return float(auto.mean()), float((pred[auto] != y[auto]).any(axis=1).mean())


def class_triage(val_y: np.ndarray, val_p: np.ndarray, test_y: np.ndarray, test_p: np.ndarray,
                 thresholds: np.ndarray, max_error: float = 0.10) -> pd.DataFrame:
    """Per damage type: how many photos could be decided without a person at ``max_error``.

    The stricter photo-level triage needs all 8 types right at once. This asks the question one damage
    type at a time (for example, only "broken part"), with the cutoff chosen on validation.
    """
    rows = []
    for c in range(val_y.shape[1]):
        trivial_error = min(val_y[:, c].mean(), 1 - val_y[:, c].mean())
        if trivial_error <= max_error:
            # Saying "absent" (or "present") for every photo is already within the limit, so automation means nothing.
            rows.append({"auto_coverage": float("nan"), "auto_error": float("nan")})
            continue
        t = np.full(1, thresholds[c])
        vcurve = triage_curve(val_y[:, [c]], val_p[:, [c]], t)
        cutoff = cutoff_for_error(vcurve, max_error)
        coverage, error = coverage_at_cutoff(test_y[:, [c]], test_p[:, [c]], t, cutoff)
        rows.append({"auto_coverage": coverage, "auto_error": error})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- run-level evaluation


def bootstrap_ci(stat, n: int, iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of ``stat(indices)`` over resampled photos."""
    rng = np.random.default_rng(seed)
    values = [stat(rng.integers(0, n, n)) for _ in range(iters)]
    values = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def paired_difference(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, iters: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Macro AP of run A minus run B on the same resampled photos: (difference, low, high).

    Comparing two separate 95% ranges is stricter than needed, because both runs see the same photos.
    Resampling the photos once and scoring both runs on that sample gives the interval of the difference.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(iters):
        i = rng.integers(0, n, n)
        a, b = _macro_ap(y[i], p_a[i]), _macro_ap(y[i], p_b[i])
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(a - b)
    return _macro_ap(y, p_a) - _macro_ap(y, p_b), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _macro_ap(y: np.ndarray, p: np.ndarray) -> float:
    aps = [average_precision(y[:, k], p[:, k]) for k in range(y.shape[1])]
    aps = [a for a in aps if a is not None]
    return float(np.mean(aps)) if aps else float("nan")


def _micro_f1(y: np.ndarray, p: np.ndarray, t: np.ndarray) -> float:
    pred = p >= t
    tp = (pred & (y == 1)).sum()
    fp = (pred & (y == 0)).sum()
    fn = (~pred & (y == 1)).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


@dataclass
class RunResult:
    name: str
    metrics: dict
    per_class: pd.DataFrame


def load_preds(path: str | Path, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Join a prediction file to the labels on ``image``; returns (y, probs) in label-file order."""
    preds = pd.read_csv(path)
    merged = labels.merge(preds, on="image", how="inner")
    if len(merged) != len(labels):
        raise ValueError(f"{path}: {len(labels) - len(merged)} labelled images have no prediction")
    return merged[label_columns()].to_numpy(), merged[prob_columns()].to_numpy()


def evaluate_run(
    name: str,
    val_y: np.ndarray,
    val_p: np.ndarray,
    test_y: np.ndarray,
    test_p: np.ndarray,
    max_error: float = 0.10,
    bootstrap_iters: int = 1000,
) -> RunResult:
    """Choose thresholds and the triage cutoff on validation, then score the test split."""
    k = val_y.shape[1]
    thresholds = np.array([best_threshold(val_y[:, c], val_p[:, c]) for c in range(k)])
    rows = []
    for c, cls in enumerate(DAMAGE_CLASSES):
        precision, recall, f1 = f1_at(test_y[:, c], test_p[:, c], thresholds[c])
        rows.append({"class": cls, "test_positives": int(test_y[:, c].sum()),
                     "ap": average_precision(test_y[:, c], test_p[:, c]),
                     "precision": precision, "recall": recall, "f1": f1, "threshold": thresholds[c]})
    per_class = pd.DataFrame(rows)
    n = len(test_y)
    # A model with no skill scores about the share of photos that have the damage.
    per_class["chance_ap"] = per_class["test_positives"] / n
    per_class = pd.concat([per_class, class_triage(val_y, val_p, test_y, test_p, thresholds, max_error)], axis=1)
    cutoff = cutoff_for_error(triage_curve(val_y, val_p, thresholds), max_error)
    coverage, auto_error = coverage_at_cutoff(test_y, test_p, thresholds, cutoff)
    metrics = {
        "run": name,
        "n_test": int(n),
        "macro_ap": _macro_ap(test_y, test_p),
        "chance_macro_ap": float(per_class.loc[per_class["test_positives"] > 0, "chance_ap"].mean()),
        "macro_ap_ci": bootstrap_ci(lambda i: _macro_ap(test_y[i], test_p[i]), n, bootstrap_iters),
        "micro_f1": _micro_f1(test_y, test_p, thresholds),
        "micro_f1_ci": bootstrap_ci(lambda i: _micro_f1(test_y[i], test_p[i], thresholds), n, bootstrap_iters),
        "exact_match": float(((test_p >= thresholds).astype(int) == test_y).all(axis=1).mean()),
        "ece": expected_calibration_error(test_y, test_p),
        "triage_max_error": max_error,
        "triage_coverage": coverage,
        "triage_error": auto_error,
    }
    return RunResult(name=name, metrics=metrics, per_class=per_class)


def compare(results_dir: str | Path = "results", splits_csv: str | Path = "data/splits.csv",
            max_error: float = 0.10, bootstrap_iters: int = 1000) -> pd.DataFrame:
    """Score every run folder in ``results/`` that has val and test predictions; write comparison.md.

    Degraded test predictions (``preds_test_<degradation>.csv``) are scored with the thresholds from
    the clean validation set, giving the robustness columns.
    """
    results_dir = Path(results_dir)
    splits = pd.read_csv(splits_csv)
    val_labels = splits[splits["split"] == "val"][["image", *label_columns()]]
    test_labels = splits[splits["split"] == "test"][["image", *label_columns()]]
    rows, details, test_preds = [], {}, {}
    for run_dir in sorted(p for p in results_dir.iterdir() if (p / "preds_val.csv").exists() and (p / "preds_test.csv").exists()):
        vy, vp = load_preds(run_dir / "preds_val.csv", val_labels)
        ty, tp = load_preds(run_dir / "preds_test.csv", test_labels)
        res = evaluate_run(run_dir.name, vy, vp, ty, tp, max_error, bootstrap_iters)
        thresholds = res.per_class["threshold"].to_numpy()
        for deg_file in sorted(run_dir.glob("preds_test_*.csv")):
            deg = deg_file.stem.replace("preds_test_", "")
            dy, dp = load_preds(deg_file, test_labels)
            res.metrics[f"macro_ap_{deg}"] = _macro_ap(dy, dp)
            res.metrics[f"micro_f1_{deg}"] = _micro_f1(dy, dp, thresholds)
        res.per_class.to_csv(run_dir / "per_class.csv", index=False)
        (run_dir / "metrics.json").write_text(json.dumps(res.metrics, indent=2), encoding="utf-8")
        rows.append(res.metrics)
        details[run_dir.name] = res.per_class
        test_preds[run_dir.name] = (ty, tp)
    table = pd.DataFrame(rows)
    diffs = _paired_table(table, test_preds, bootstrap_iters)
    (results_dir / "comparison.md").write_text(comparison_markdown(table, details, diffs), encoding="utf-8")
    return table


def _paired_table(table: pd.DataFrame, test_preds: dict, iters: int) -> pd.DataFrame:
    """Best run (by macro AP) against each other run, as a paired macro AP difference."""
    if len(table) < 2:
        return pd.DataFrame()
    best = table.sort_values("macro_ap", ascending=False)["run"].iloc[0]
    rows = []
    for run in table["run"]:
        if run == best:
            continue
        y, p_best = test_preds[best]
        _, p_other = test_preds[run]
        diff, lo, hi = paired_difference(y, p_best, p_other, iters)
        rows.append({"best": best, "other": run, "diff": diff, "low": lo, "high": hi, "clear": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def _pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


def comparison_markdown(table: pd.DataFrame, details: dict[str, pd.DataFrame], diffs: Optional[pd.DataFrame] = None) -> str:
    """Summary table (one column per run) plus per-class AP, ready to paste into the README."""
    if table.empty:
        return "# Comparison\n\nNo runs with val and test predictions found.\n"
    runs = table["run"].tolist()
    t = table.set_index("run")

    def row(label: str, fn) -> str:
        return f"| {label} | " + " | ".join(fn(t.loc[r]) for r in runs) + " |"

    def ci(m, key):
        lo, hi = m[key + "_ci"]
        return f"{m[key]:.2f} ({lo:.2f}-{hi:.2f})"

    lines = ["# Comparison", "", f"Held-out test photos: {int(t['n_test'].iloc[0])}. Ranges are 95% bootstrap intervals.", "",
             "| Metric | " + " | ".join(runs) + " |", "|:---|" + "---:|" * len(runs) + ""]
    lines += [
        row("Macro AP (ranking quality, 8 damage types)", lambda m: ci(m, "macro_ap")),
        row("Macro AP expected from guessing (average share of photos with each damage)", lambda m: _pct(m["chance_macro_ap"])),
        row("Micro F1 at validation thresholds", lambda m: ci(m, "micro_f1")),
        row("Photos with every label right", lambda m: _pct(m["exact_match"])),
        row("Calibration error (lower is better)", lambda m: _pct(m["ece"])),
        row(f"Photos automated at <= {int(table['triage_max_error'].iloc[0] * 100)}% error", lambda m: _pct(m["triage_coverage"])),
        row("Error among automated photos", lambda m: _pct(m["triage_error"])),
    ]
    for deg in ("blur", "low_light", "jpeg", "low_res"):
        key = f"macro_ap_{deg}"
        if key in t.columns:
            lines.append(row(f"Macro AP, {deg.replace('_', ' ')} test photos", lambda m, k=key: _pct(m.get(k))))
    if diffs is not None and not diffs.empty:
        lines += ["", "## Is the best run really better? (same photos, paired comparison)", "",
                  "| Best run | Compared with | Macro AP difference | 95% range | Clearly better? |", "|:---|:---|---:|---:|:---|"]
        for _, d in diffs.iterrows():
            lines.append(f"| {d['best']} | {d['other']} | {d['diff']:+.2f} | {d['low']:+.2f} to {d['high']:+.2f} | "
                         + ("yes" if d["clear"] else "not clearly") + " |")
    lines += ["", "## Average precision per damage type", "",
              "Guess = what a model with no skill would score (the share of test photos with that damage).", "",
              "| Damage | Test photos | Guess | " + " | ".join(runs) + " |", "|:---|---:|---:|" + "---:|" * len(runs)]
    first = details[runs[0]]
    for i, cls in enumerate(first["class"]):
        cells = [_pct(details[r]["ap"].iloc[i]) for r in runs]
        guess = f"{first['chance_ap'].iloc[i]:.2f}"
        lines.append(f"| {cls} | {int(first['test_positives'].iloc[i])} | {guess} | " + " | ".join(cells) + " |")
    limit = int(table["triage_max_error"].iloc[0] * 100)
    lines += ["", "## Photos that could skip review, one damage type at a time", "",
              f"Share of test photos where the decision for that damage type could be automated at <= {limit}% error "
              "(cutoff chosen on validation), with the error actually seen on the test photos in brackets. Far less strict than "
              "needing all 8 types right at once.", "",
              "- n/a (guess) = answering the same way for every photo is already within the error limit.",
              f"- n/a (no skill) = that run's AP for the type is within {SKILL_MARGIN:.2f} of guessing, so a high share would only "
              "reflect how common the type is, not what the model learned.", "",
              "| Damage | " + " | ".join(runs) + " |", "|:---|" + "---:|" * len(runs)]
    for i, cls in enumerate(first["class"]):
        cells = []
        for r in runs:
            d = details[r]
            if pd.isna(d["auto_coverage"].iloc[i]):
                cells.append("n/a (guess)")
            elif d["ap"].iloc[i] < d["chance_ap"].iloc[i] + SKILL_MARGIN:
                cells.append("n/a (no skill)")
            else:
                cells.append(f"{d['auto_coverage'].iloc[i]:.2f} (error {d['auto_error'].iloc[i]:.2f})")
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
