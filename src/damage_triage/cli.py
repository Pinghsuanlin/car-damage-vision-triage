"""Command line: ``damage-triage prepare | train | zeroshot | predict | compare | cam | dup-check``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from damage_triage.data import DAMAGE_CLASSES, find_damage_dir, load_labels, pairs_montage, prepare


def _splits(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"{path} not found. Run `damage-triage prepare` first.")
    return pd.read_csv(path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="damage-triage", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="labels, same-vehicle groups, split, resized images")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-embeddings", action="store_true", help="group exact duplicates only")
    p.add_argument("--similarity", type=float, default=0.90)

    p = sub.add_parser("dup-check", help="save grouped photo pairs as an image to check by eye")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("docs/images/grouped_pairs.jpg"))

    p = sub.add_parser("train", help="linear probe or fine-tune a pretrained model")
    p.add_argument("--model", default="efficientnet_b0")
    p.add_argument("--mode", choices=["linear", "finetune"], default="finetune")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--name", default=None, help="results/<name>; default <model>_<mode>")
    p.add_argument("--data", type=Path, default=Path("data"))

    p = sub.add_parser("zeroshot", help="CLIP zero-shot baseline predictions for val and test")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--name", default="clip_zeroshot")

    p = sub.add_parser("predict", help="val + test (+ degraded test) predictions for a trained run")
    p.add_argument("run", type=Path, help="results/<run>")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--no-degrade", action="store_true")

    p = sub.add_parser("compare", help="score every run in results/ and write results/comparison.md")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--max-error", type=float, default=0.10)

    p = sub.add_parser("cam", help="Grad-CAM heatmaps for a few test photos")
    p.add_argument("run", type=Path)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--out", type=Path, default=Path("docs/images"))

    args = parser.parse_args(argv)

    if args.command == "prepare":
        features = None
        if not args.no_embeddings:
            try:
                from damage_triage.vision import embed_images

                ds = find_damage_dir(args.raw)
                frame = load_labels(ds)
                print("computing image embeddings to group photos of the same vehicle...")
                features = embed_images([ds.parent / p for p in frame["image"]])
            except ImportError:
                print("PyTorch not installed: grouping exact duplicates only.")
        s = prepare(args.raw, args.out, seed=args.seed, features=features, similarity=args.similarity)
        (args.out / "groups.json").write_text(json.dumps(s.pairs), encoding="utf-8")
        print(f"{s.n_images} images, {s.n_groups} groups ({s.n_duplicates} duplicate pairs, "
              f"{s.n_same_vehicle} same-vehicle pairs); split sizes {s.split_sizes}")
        print(s.positives.to_string())
        if s.largest_group > 6:
            print(f"WARNING: one group has {s.largest_group} photos. The similarity threshold is probably "
                  "pairing different cars (e.g. same yard background). Run `make dup-check`, then rerun "
                  "prepare with a higher --similarity (e.g. 0.95).")
        return 0

    if args.command == "dup-check":
        frame = _splits(args.data / "splits.csv")
        pairs = json.loads((args.data / "groups.json").read_text(encoding="utf-8"))
        if not pairs:
            print("No grouped pairs.")
            return 0
        print(f"wrote {pairs_montage(frame, [tuple(p) for p in pairs], args.data, args.out)} ({len(pairs)} pairs)")
        return 0

    if args.command == "compare":
        from damage_triage.evaluate import compare

        compare(args.results, args.data / "splits.csv", max_error=args.max_error)
        print((args.results / "comparison.md").read_text(encoding="utf-8"))
        return 0

    if args.command == "zeroshot":
        from damage_triage.zeroshot import zeroshot_predictions

        frame = _splits(args.data / "splits.csv")
        out = Path("results") / args.name
        for split in ("val", "test"):
            zeroshot_predictions(frame[frame["split"] == split], args.data, out / f"preds_{split}.csv")
        print(f"wrote {out}/preds_val.csv and preds_test.csv")
        return 0

    from damage_triage import vision

    if args.command == "train":
        frame = _splits(args.data / "splits.csv")
        defaults = {"linear": (20, 3e-3), "finetune": (15, 1e-3)}[args.mode]
        cfg = vision.TrainConfig(model=args.model, mode=args.mode, epochs=args.epochs or defaults[0],
                                 lr=args.lr or defaults[1], batch_size=args.batch_size, seed=args.seed)
        out = Path("results") / (args.name or f"{args.model}_{args.mode}")
        summary = vision.train(frame, args.data, out, cfg)
        print(f"best val macro AP {summary['best_val_macro_ap']:.3f} at epoch {summary['best_epoch']} "
              f"({summary['minutes']} min on {summary['device']}); saved {out}/model.pt")
        return 0

    if args.command == "predict":
        from damage_triage.degrade import DEGRADATIONS

        frame = _splits(args.data / "splits.csv")
        model, cfg = vision.load_trained(args.run)
        for split in ("val", "test"):
            vision.write_predictions(model, frame[frame["split"] == split], args.data,
                                     args.run / f"preds_{split}.csv", cfg.image_size)
        if not args.no_degrade:
            test = frame[frame["split"] == "test"]
            for name, fn in DEGRADATIONS.items():
                vision.write_predictions(model, test, args.data, args.run / f"preds_test_{name}.csv",
                                         cfg.image_size, degrade=fn)
        print(f"wrote predictions to {args.run}")
        return 0

    if args.command == "cam":
        from PIL import Image

        frame = _splits(args.data / "splits.csv")
        model, cfg = vision.load_trained(args.run)
        test = frame[frame["split"] == "test"].head(args.n)
        args.out.mkdir(parents=True, exist_ok=True)
        for _, row in test.iterrows():
            with Image.open(args.data / row["processed"]) as im:
                image = im.convert("RGB")
            labels = [i for i, c in enumerate(DAMAGE_CLASSES) if row["y_" + c.lower().replace(" ", "_")]]
            k = labels[0] if labels else 0
            cam = vision.gradcam(model, image, k, cfg.image_size)
            name = Path(row["image"]).stem.replace(" ", "_")
            vision.overlay(image, cam, cfg.image_size).save(args.out / f"cam_{name}_{DAMAGE_CLASSES[k].replace(' ', '_')}.jpg")
        print(f"wrote {len(test)} heatmaps to {args.out}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
