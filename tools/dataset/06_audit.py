"""Step 6: audit the built YOLO dataset before training.

Usage:
  python tools/dataset/06_audit.py --data data/household/data.yaml
  python tools/dataset/06_audit.py --data ... --no-model       # skip the YOLO26n missing-label check

Checks:
  * every image has a label file and vice versa; label lines valid (ids, 0-1 coordinates)
  * per-class images / instances in train and val; classes missing from val; imbalance
  * near-duplicates ACROSS train and val (perceptual hash) - must be zero
  * probable missing labels: the current YOLO26n (COCO) run on Open Images images; a
    confident detection (>= 0.6) of a V1 class with no matching label (IoU >= 0.3)
    is written to data/_cache/audit/missing_label_candidates.csv for manual review
  * one sample grid per class with boxes drawn, in data/_cache/audit/
Exit code 1 if any hard error (broken labels, orphan files, cross-split duplicates).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CACHE, REPO, iou  # noqa: E402

AUDIT = CACHE / "audit"
# COCO names the current yolo26n.pt predicts -> V1 class names
COCO_TO_V1 = {"bed": "bed", "couch": "sofa", "chair": "chair", "dining table": "table", "tv": "tv",
              "laptop": "laptop", "refrigerator": "refrigerator", "microwave": "microwave",
              "bottle": "bottle", "person": "person"}


def read_split(root: Path, split: str, nc: int):
    images = {p.stem: p for p in (root / "images" / split).glob("*.jpg")}
    labels = {p.stem: p for p in (root / "labels" / split).glob("*.txt")}
    errors, data = [], {}
    for stem in sorted(set(images) ^ set(labels)):
        errors.append(f"{split}/{stem}: image without label or label without image")
    for stem in sorted(set(images) & set(labels)):
        rows = []
        for n, line in enumerate(labels[stem].read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            try:
                c, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:5])
                assert len(parts) == 5 and 0 <= c < nc
                assert 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1
            except (ValueError, AssertionError, TypeError):
                errors.append(f"{split}/{stem}.txt:{n} invalid line {line!r}")
                continue
            rows.append((c, cx, cy, w, h))
        data[stem] = (images[stem], rows)
    return data, errors


def grid(samples, classes, cls_id, path, tile=256, cols=4):
    rows = (len(samples) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile, max(rows, 1) * tile), "white")
    for i, (img_path, labels) in enumerate(samples):
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im.thumbnail((tile, tile))
            d = ImageDraw.Draw(im)
            for c, cx, cy, w, h in labels:
                x0, y0 = (cx - w / 2) * im.width, (cy - h / 2) * im.height
                x1, y1 = (cx + w / 2) * im.width, (cy + h / 2) * im.height
                d.rectangle([x0, y0, x1, y1], outline="red" if c == cls_id else "yellow", width=2)
            canvas.paste(im, ((i % cols) * tile, (i // cols) * tile))
    canvas.save(path, quality=85)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO / "data" / "household" / "data.yaml")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--grid", type=int, default=12, help="sample images per class grid")
    args = ap.parse_args()

    import imagehash

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module
    near_duplicate_pairs = import_module("05_build_yolo").near_duplicate_pairs

    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    classes = [str(cfg["names"][i]) for i in sorted(cfg["names"])]
    root = args.data.parent
    AUDIT.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    splits = {}
    for s in ("train", "val"):
        splits[s], errs = read_split(root, s, len(classes))
        errors += errs

    print(f"{'class':17}{'train imgs':>11}{'val imgs':>10}{'train boxes':>12}{'val boxes':>10}{'median area':>13}")
    imgs_per = {s: defaultdict(int) for s in splits}
    inst_per = {s: defaultdict(int) for s in splits}
    areas = defaultdict(list)
    for s, data in splits.items():
        for _, rows in data.values():
            for c in {r[0] for r in rows}:
                imgs_per[s][c] += 1
            for c, _, _, w, h in rows:
                inst_per[s][c] += 1
                areas[c].append(w * h)
    for i, c in enumerate(classes):
        med = statistics.median(areas[i]) if areas[i] else 0
        print(f"{c:17}{imgs_per['train'][i]:11}{imgs_per['val'][i]:10}{inst_per['train'][i]:12}"
              f"{inst_per['val'][i]:10}{med:13.3f}")
    no_val = [c for i, c in enumerate(classes) if imgs_per["val"][i] == 0]
    if no_val:
        print("Warning: classes with no validation images:", no_val)
    boxes = [inst_per["train"][i] for i in range(len(classes)) if inst_per["train"][i]]
    if boxes:
        med = statistics.median(boxes)
        heavy = [classes[i] for i in range(len(classes)) if inst_per["train"][i] > 3 * med]
        if heavy:
            print(f"Note: classes with > 3x median train boxes ({med:.0f}): {heavy}")

    print("Checking near-duplicates across train/val...")
    hashes = {}
    for s, data in splits.items():
        for stem, (img, _) in data.items():
            with Image.open(img) as im:
                hashes[f"{s}/{stem}"] = int(str(imagehash.phash(im)), 16)
    cross = [(a, b) for a, b in near_duplicate_pairs(hashes) if a.split("/")[0] != b.split("/")[0]]
    for a, b in cross:
        errors.append(f"near-duplicate across splits: {a} ~ {b}")
    print(f"  cross-split near-duplicates: {len(cross)}")

    for i, c in enumerate(classes):
        samples = [(img, rows) for img, rows in splits["train"].values() if any(r[0] == i for r in rows)][: args.grid]
        if samples:
            grid(samples, classes, i, AUDIT / f"grid_{c.replace(' ', '_')}.jpg")
    print(f"  sample grids: {AUDIT}")

    if not args.no_model:
        from ultralytics import YOLO

        model = YOLO(str(REPO / "backend" / "models" / "yolo26n.pt"))
        cid = {c: i for i, c in enumerate(classes)}
        out = []
        oi = [(s, stem, img, rows) for s, data in splits.items() for stem, (img, rows) in data.items() if stem.startswith("oi_")]
        print(f"Running YOLO26n on {len(oi)} Open Images images for missing-label candidates...")
        for s, stem, img, rows in oi:
            res = model.predict(str(img), conf=0.6, verbose=False)[0]
            gt = [(r[0], r[1] - r[3] / 2, r[2] - r[4] / 2, r[1] + r[3] / 2, r[2] + r[4] / 2) for r in rows]
            for box, cls_idx, conf in zip(res.boxes.xyxyn.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                v1 = COCO_TO_V1.get(res.names[int(cls_idx)])
                if v1 is None:
                    continue
                if not any(g[0] == cid[v1] and iou(tuple(box), g[1:]) >= 0.3 for g in gt):
                    out.append([s, stem, v1, f"{conf:.2f}", *(f"{v:.3f}" for v in box)])
        with open(AUDIT / "missing_label_candidates.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["split", "image", "class", "conf", "xmin", "ymin", "xmax", "ymax"])
            w.writerows(out)
        print(f"  missing-label candidates: {len(out)} (review {AUDIT / 'missing_label_candidates.csv'})")

    print("\nERRORS:" if errors else "\nERRORS: none")
    for e in errors[:50]:
        print("  !", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
