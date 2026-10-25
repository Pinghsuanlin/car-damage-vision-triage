# Running it on your Mac

About 45 minutes the first time, mostly downloads and training. Run every command from the project
folder:

```bash
cd ~/Documents/CoworkOS/career/portfolio/car-damage-triage
```

## 1. Install (5–10 min, once)

```bash
make install
```

This creates `.venv` and `uv.lock` with PyTorch (with Apple Silicon support), timm and OpenCLIP.
Commit `uv.lock` later.

## 2. Check the code (2 min)

```bash
make test
make lint
```

Expect all tests to pass and lint to report no errors. The model tests train a tiny untrained network
on fake images, so no downloads are needed. If anything fails, paste the output to Claude: this was
written in a sandbox without PyTorch, so a small fix on first run is possible.

## 3. Prepare the data (3–5 min)

```bash
make prepare
```

It finds the damage folder (the release names the folders the wrong way round), builds one 0/1 label
per damage type, resizes the photos into `data/processed/`, groups photos of the same car, and writes
the split to `data/splits.csv`. It downloads a ResNet-50 (about 100 MB) once, to compare photos.

Expected output: 814 images, split roughly 570 / 120 / 120, and a table with every damage type
present in all three splits.

**Then check the grouping by eye:**

```bash
make dup-check
open docs/images/grouped_pairs.jpg
```

Each row is a pair the code thinks shows the same car.

- **Most rows are the same car from another angle:** good, keep it.
- **Many rows are different cars** (often the same yard background): the threshold is too loose.
  Rerun with a stricter one and check again:
  ```bash
  uv run python -m damage_triage prepare --similarity 0.95
  make dup-check
  ```
- **If prepare prints a WARNING about a large group**, do the same.

Once you're happy, don't rerun `prepare`: the split must stay fixed for every model.

## 4. Run the three approaches (15–25 min)

```bash
make zeroshot     # CLIP, no training; downloads ~600 MB once
make linear       # frozen model, trains the final layer only
make finetune     # fine-tunes the whole model; also scores blurred / dark / compressed / low-res photos
```

Training prints one line per epoch with the validation score (macro AP). It stops early when it
stops improving. On an M5 each run should take a few minutes.

## 5. Score everything on the test set (1 min)

```bash
make compare
```

This writes `results/comparison.md`. Paste it, or send it to Claude, before putting numbers in the
README. Things worth checking:

1. **Fine-tuned vs linear probe vs CLIP.** Is fine-tuning clearly better, with ranges that don't
   overlap? If the linear probe is close, that's a real finding: the cheap option is good enough.
2. **Per damage type.** Expect Cracked, Flaking and Corrosion (10–17 test photos each) to be weakest
   and noisiest.
3. **Degraded photos.** How much does macro AP drop on blur, low light, JPEG and low resolution?
4. **Automated share.** How many test photos could skip review at ≤ 10% error?

## 6. Heatmaps and demo (5 min)

```bash
make cam
open docs/images
```

Check the heatmaps sit on the damage, not the background. Pick one or two good ones for the README.

```bash
make app
```

Opens a page where you can upload a car photo and see the damage probabilities, the
automate-or-review decision and the heatmap. A screenshot makes a good README image.

## 7. Publish

Same as before: GitHub Desktop → **File → Add local repository** → this folder → create the
repository → **Publish**, unticking "Keep this code private". Before publishing, check the changed
files: `data/raw`, `data/processed` and any `model.pt` must **not** be there (the `.gitignore`
excludes them). `data/splits.csv`, `results/comparison.md` and the per-run `metrics.json` and
`per_class.csv` should be included.

## Troubleshooting

| Problem | Fix |
|:---|:---|
| `MPS backend out of memory` | Add `--batch-size 16` to the train command in the Makefile |
| Training is slow / says `device cpu` | Check `uv run python -c "import torch; print(torch.backends.mps.is_available())"` prints True |
| `No meta.json with damage classes` | Both dataset folders must sit inside `data/raw/` |
| Download errors for weights | Rerun; the files are cached after the first success |
