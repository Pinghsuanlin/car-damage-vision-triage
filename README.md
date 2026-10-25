# Car Damage Triage

Look at a photo of a damaged car, say which kinds of damage it shows, and decide whether the answer
is certain enough to automate or should go to a person. Built by fine-tuning a pretrained image model
on 814 labelled car-damage photos, and compared with a zero-shot vision-language baseline (CLIP) and
a cheaper "frozen model" approach, all scored on a held-out test set.

## Why

Motor insurers receive photos with almost every claim. Sorting them by damage type is repetitive
work, but a model that is confidently wrong is worse than no model. The useful product is one that
handles the clear cases and **routes the uncertain ones to a person**, with a known error rate on the
cases it handles. That is the same question as in my
[pdf-data-extractor](https://github.com/Pinghsuanlin/pdf-data-extractor): which answers can you trust?

## Results

Held-out test set: 123 photos never used for training, threshold choice or model selection. Ranges
are 95% bootstrap intervals. Full tables: [results/comparison.md](results/comparison.md).

| Metric | CLIP zero-shot | Linear probe | Fine-tuned |
|:---|---:|---:|---:|
| Macro average precision (8 damage types) | 0.45 (0.42-0.50) | 0.42 (0.40-0.47) | **0.52 (0.48-0.58)** |
| Macro AP expected from guessing | 0.41 | 0.41 | 0.41 |
| Micro F1 at validation thresholds | 0.66 (0.64-0.69) | 0.65 (0.62-0.68) | 0.66 (0.63-0.69) |
| Photos with every label right | 0.02 | 0.02 | 0.06 |
| Calibration error (lower is better) | 0.14 | 0.22 | **0.12** |
| Photos automated at ≤ 10% error | 0.00 | 0.00 | 0.00 |
| Macro AP, blurred test photos | n/a | 0.43 | 0.44 |
| Macro AP, low-light test photos | n/a | 0.43 | 0.47 |
| Macro AP, heavy JPEG test photos | n/a | 0.43 | 0.44 |
| Macro AP, low-resolution test photos | n/a | 0.43 | 0.43 |

Average precision per damage type. "Guess" is what a model with no skill would score: the share of
test photos that have that damage.

| Damage | Test photos | Guess | CLIP zero-shot | Linear probe | Fine-tuned |
|:---|---:|---:|---:|---:|---:|
| Broken part | 88 | 0.72 | 0.77 | 0.76 | **0.84** |
| Missing part | 65 | 0.53 | 0.48 | 0.53 | **0.74** |
| Scratch | 78 | 0.63 | 0.75 | 0.66 | 0.74 |
| Dent | 87 | 0.71 | 0.71 | 0.69 | 0.72 |
| Flaking | 16 | 0.13 | 0.16 | 0.15 | **0.44** |
| Paint chip | 42 | 0.34 | 0.41 | 0.33 | 0.41 |
| Corrosion | 17 | 0.14 | 0.23 | 0.17 | 0.17 |
| Cracked | 10 | 0.08 | 0.10 | 0.09 | 0.13 |

### What the results say

- **Fine-tuning helps, but the model is not ready to automate anything.** On the same test photos,
  fine-tuning beats the frozen linear probe by 0.10 macro AP (95% range +0.05 to +0.15) and beats CLIP,
  which was never trained on this data, by 0.07 (+0.02 to +0.13). Both gaps are clear, though the CLIP
  one is small. No test photo could be automated at 10% error, and only 6% have all eight labels
  right even for the best model.
- **The linear probe is no better than guessing** (0.42 against 0.41), so frozen features from a
  general model do not carry this task. Micro F1 hides this: all three runs score about 0.65, while
  labelling every damage type as present on every photo would already score about 0.58.
- **The model is good at structural damage and weak elsewhere.** Broken part (0.84) and missing
  part (0.74) are clearly above chance, and fine-tuning is what makes missing part work (CLIP 0.48).
  Flaking is three times the guess. Dent, corrosion and cracked are at chance. Dent is the surprise,
  since it is the most common type; it may be hard to see in photos or inconsistently labelled, which
  I have not checked.
- **Degraded photos take it close to chance.** Blur, heavy compression and low resolution bring the
  fine-tuned model from 0.52 to 0.43-0.44; low light is less harmful (0.47).
- **Calibration:** fine-tuning is best calibrated (0.12), the linear probe worst (0.22).
- **Deciding one damage type at a time is a looser bar, and it still says little.** A type only counts
  when the model beats guessing by at least 0.10 AP, because a type that is rare (or very common)
  looks "automatable" simply from how often it occurs. The first version of this table showed corrosion
  as 54% automatable for the fine-tuned model, but its AP (0.17) is no better than guessing (0.14), so
  that figure reflected the 86% of photos without corrosion, not skill. With the check, only scratch
  and flaking remain for the fine-tuned model, and neither holds up: scratch decisions were automated
  for 14% of test photos but 24% of those were wrong (the cut-off chosen on validation did not carry
  over to the test photos), and flaking had no errors but covers only about 9 photos, too few to show
  the error is under 10%. No damage type can be reliably automated with this data and model.

### How to read it

- **Average precision (AP)** measures how well the model ranks photos that have a damage type above
  photos that do not (1.0 = perfect; a model with no skill scores about the share of photos with that
  damage). Macro AP averages the 8 types, so rare ones count as much as common ones.
- **Photos automated at ≤ 10% error**: the share of test photos the system would decide on its own
  when the cut-off is chosen on validation data to keep errors among those photos under 10%. An error
  here is a photo with *any* wrong damage label, which is a strict bar with eight types.
- With 123 test photos, the ranges matter. Two separate ranges that overlap can still hide a clear
  difference, because both runs are scored on the same photos; `make compare` therefore also writes a
  paired comparison (resample the photos once, score both runs, take the difference).

## Data

[Humans in the Loop: Car Parts and Car Damages](https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/),
released under **CC0 1.0** (public domain) for automated claims-processing research. The damage part
has 814 photos with polygon labels for 8 damage types:

| Dent | Scratch | Broken part | Missing part | Paint chip | Flaking | Corrosion | Cracked |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 629 | 517 | 664 | 434 | 279 | 108 | 109 | 67 |

(photos containing each type; most photos show several). The images are not redistributed here:
download them and put both folders in `data/raw/`. `data/splits.csv` (labels, groups and split, no
images) is committed so everyone uses the same split.

Two quirks of the release, handled in code: the two folders are **named the wrong way round**
(`Car_parts_dataset` holds the damage labels), so the damage folder is found from its class list;
and **many cars appear in two or three photos**, which is handled below.

## How it works

```
data/raw (Supervisely JSON) ─► prepare: 0/1 label per damage type, resize to 288 px
                                 │  group photos of the same car (perceptual hash + pretrained embeddings)
                                 │  split 70/15/15, groups kept together, every damage type balanced
                                 ▼
        ┌───────────────────────┬─────────────────────────┬──────────────────────────┐
        │ CLIP zero-shot        │ linear probe            │ fine-tune                │
        │ "a car with a dent"   │ frozen EfficientNet-B0, │ EfficientNet-B0, all     │
        │ vs "with no dent"     │ train final layer only  │ layers, smaller LR body  │
        └───────────┬───────────┴────────────┬────────────┴─────────────┬────────────┘
                    └──── probabilities per damage type for val and test ┘
                                 ▼
  compare: thresholds and review cut-off chosen on val ─► scored on test (+ degraded test photos)
                                 ▼
  results/comparison.md · per_class.csv · Grad-CAM heatmaps · Streamlit demo
```

| Module | Job |
|:---|:---|
| `data.py` | Labels from annotations, duplicate and same-vehicle grouping, stratified group split |
| `vision.py` | timm models, linear probe vs fine-tune, training on MPS / CUDA / CPU, predictions, Grad-CAM |
| `zeroshot.py` | CLIP baseline with paired "with / with no" prompts |
| `degrade.py` | Blur, low light, heavy JPEG, low resolution for the robustness test |
| `evaluate.py` | AP, F1, bootstrap ranges, calibration, automate-vs-review curve, comparison table |
| `cli.py`, `app.py` | Command line and Streamlit demo |

## Design decisions and trade-offs

- **Group photos of the same car before splitting.** If one shot of a car is in training and another
  in test, the model can recognise the car rather than the damage, and the test score is inflated.
  Near-identical files are caught by a perceptual hash; other shots of the same vehicle by cosine
  similarity of pretrained image embeddings. The threshold is checked by eye (`make dup-check`),
  because similar backgrounds at the same salvage yard can fool it.
- **Multi-label, not single-label.** Most photos show several damage types, so each type gets its own
  yes/no output and its own threshold.
- **Three approaches, on purpose.** Zero-shot CLIP shows what a general model knows for free; the
  linear probe shows what pretrained features already capture; fine-tuning shows what adapting the
  network adds. Choosing between them is the real decision in practice, and cost differs a lot.
- **Thresholds, the review cut-off and the best epoch are chosen on validation only.** The test set
  is scored once at the end.
- **Rare classes are up-weighted in the loss** (capped at 10x) so the model does not ignore Cracked
  and Flaking; their scores are still the least reliable.
- **Robustness on degraded photos** uses the same thresholds as the clean test, so the drop shows
  what happens when real claim photos are worse than the training ones.
- **Grad-CAM heatmaps** show where the model looked, a quick check that it uses the damage and not
  the background.

## How to run

Step-by-step guide: [RUNNING.md](RUNNING.md). In short:

```bash
make install    # uv sync (PyTorch with Apple Silicon support)
make test
make prepare    # then: make dup-check, and look at docs/images/grouped_pairs.jpg
make zeroshot linear finetune compare cam
make app
```

On an Apple Silicon laptop each training run takes a few minutes.

## Limitations and next steps

- 814 photos is small: wide ranges, and only 10 test photos for the rarest class. The three
  weakest damage types are at chance, so no photo could be automated at the 10% error target.
- Labels are image-level (is this damage anywhere in the photo?), not where it is. Next: use the
  polygons for detection or segmentation, and the 441 photos that also have car-part labels to say
  *which part* is damaged, which matters most for repair cost.
- Severity is not labelled; a claims system would need it.
- Calibration could be improved with temperature scaling on the validation set.
- A larger dataset (for example CarDD, about 4,000 photos, licence on request) would test whether
  the ranking of the three approaches holds.
- Next checks on the weak types: look at the dent photos and their labels, and see whether more
  training, stronger augmentation or a larger model moves the rare classes above chance.

## License

Code: MIT, see [LICENSE](LICENSE). Data: CC0 1.0 by Humans in the Loop; not included in this repo.
