"""Step 5: build the final YOLO dataset in data/household/{images,labels}/{train,val}.

Usage:
  python tools/dataset/05_build_yolo.py --version v1 --val-ratio 0.15 --seed 42
  python tools/dataset/05_build_yolo.py ... --overwrite     # replace an existing build

Inputs: Open Images selection + boxes (02), downloaded images (03), custom photos (04).
  * Open Images corner boxes -> YOLO "cls cx cy w h" (normalised)
  * boxes smaller than min_box_side_px or min_box_area_fraction are dropped
  * same-class boxes overlapping at IoU > 0.7 are merged (e.g. an OI "Person" + "Man" pair)
  * near-duplicate images (perceptual hash, Hamming <= 4) are put in the same group, and
    custom photos are grouped by <class>/<session>; whole groups go to one split, so
    train and val never share a near-duplicate
  * group-level stratified split: groups with the rarest classes are assigned first so
    every class gets ~val_ratio of its images in val
Writes data/household/ATTRIBUTION.csv (Open Images credits) and data/_cache/build_report.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CACHE, CUSTOM_CACHE, HOUSEHOLD, OI_CACHE, OI_IMAGES, iou, load_classes, load_source_mapping  # noqa: E402

DUP_IOU = 0.7
PHASH_MAX_DISTANCE = 4


# ------------------------------------------------------------------ pure helpers (unit-testable)

def to_yolo(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) normalised -> (cx, cy, w, h) normalised, clipped to [0, 1]."""
    x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in box)
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def merge_duplicates(boxes: list[tuple[int, float, float, float, float]]) -> list[tuple[int, float, float, float, float]]:
    """Drop same-class boxes overlapping an earlier kept box at IoU > DUP_IOU (larger boxes first)."""
    kept: list[tuple[int, float, float, float, float]] = []
    for b in sorted(boxes, key=lambda b: -(b[3] - b[1]) * (b[4] - b[2])):
        if all(k[0] != b[0] or iou(k[1:], b[1:]) <= DUP_IOU for k in kept):
            kept.append(b)
    return kept


def near_duplicate_pairs(hashes: dict[str, int], max_distance: int = PHASH_MAX_DISTANCE) -> list[tuple[str, str]]:
    """Pairs of 64-bit hashes within max_distance bits.

    Pigeonhole: split into max_distance + 1 chunks; two hashes within the distance share at
    least one identical chunk, so only same-chunk buckets need comparing.
    """
    chunks = max_distance + 1
    width = 64 // chunks
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for key, h in hashes.items():
        for i in range(chunks):
            shift = i * width
            bits = width if i < chunks - 1 else 64 - shift
            buckets[(i, (h >> shift) & ((1 << bits) - 1))].append(key)
    pairs: set[tuple[str, str]] = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if bin(hashes[a] ^ hashes[b]).count("1") <= max_distance:
                    pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def stratified_group_split(groups: dict[str, set[int]], val_ratio: float, seed: int) -> set[str]:
    """Return the set of group ids assigned to val.

    groups: group id -> set of class ids present in that group (empty = background).
    """
    rng = random.Random(seed)
    totals: dict[int, int] = defaultdict(int)
    for cls_set in groups.values():
        for c in cls_set:
            totals[c] += 1
    quota = {c: max(1, round(n * val_ratio)) for c, n in totals.items()}
    got: dict[int, int] = defaultdict(int)
    ids = sorted(groups)
    rng.shuffle(ids)
    ids.sort(key=lambda g: min((totals[c] for c in groups[g]), default=10**9))  # rarest first, stable
    val: set[str] = set()
    for g in ids:
        cls_set = groups[g]
        if cls_set:
            # Only if EVERY class in the group still needs val images. With `any`, a group was
            # sent to val for one under-target class even when its other classes were already
            # full, which pushed co-occurring rare classes to ~45% val.
            if all(got[c] < quota[c] for c in cls_set):
                val.add(g)
                for c in cls_set:
                    got[c] += 1
        elif rng.random() < val_ratio:  # background
            val.add(g)
    return val


# ------------------------------------------------------------------ build

def load_oi(classes: list[str], cfg: dict) -> tuple[list[dict], list[dict]]:
    sel_path, box_path = OI_CACHE / "v1_selection.csv", OI_CACHE / "v1_boxes.csv"
    if not sel_path.exists() or not box_path.exists():
        print("No Open Images selection found; building from custom photos only.")
        return [], []
    cid = {c: i for i, c in enumerate(classes)}
    boxes: dict[str, list] = defaultdict(list)
    with open(box_path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            boxes[r["image_id"]].append((cid[r["class"]], *(float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))))
    min_frac = float(cfg.get("min_box_area_fraction", 0.0))
    min_px = int(cfg.get("min_box_side_px", 0))
    items, attribution, missing = [], [], 0
    with open(sel_path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            img = OI_IMAGES / f"{r['image_id']}.jpg"
            if not img.exists():
                missing += 1
                continue
            with Image.open(img) as im:
                w, h = im.size
            kept = [b for b in merge_duplicates(boxes.get(r["image_id"], []))
                    if (b[3] - b[1]) * (b[4] - b[2]) >= min_frac
                    and (b[3] - b[1]) * w >= min_px and (b[4] - b[2]) * h >= min_px]
            if r["reason"] != "background" and not kept:
                continue  # every V1 box was too small
            items.append({"stem": f"oi_{r['image_id']}", "src": img, "group": f"oi_{r['image_id']}",
                          "labels": [(b[0], *to_yolo(b[1:])) for b in kept], "source": "open_images"})
            attribution.append({k: r[k] for k in ("image_id", "author", "author_profile", "license", "original_url", "title")})
    if missing:
        print(f"Warning: {missing} selected Open Images images are not downloaded (run 03 again)")
    return items, attribution


def load_custom() -> list[dict]:
    manifest = CUSTOM_CACHE / "manifest.csv"
    if not manifest.exists():
        print("No custom photos imported yet (04_import_custom.py).")
        return []
    items = []
    with open(manifest, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            labels = []
            for line in (CUSTOM_CACHE / "labels" / f"{r['stem']}.txt").read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    c, *v = line.split()
                    labels.append((int(c), *(float(x) for x in v)))
            items.append({"stem": r["stem"], "src": CUSTOM_CACHE / "images" / f"{r['stem']}.jpg",
                          "group": f"custom_{r['group']}", "labels": labels, "source": "custom"})
    return items


def clear_output(out: Path, overwrite: bool) -> None:
    dirs = [out / k / s for k in ("images", "labels") for s in ("train", "val")]
    existing = [p for d in dirs if d.exists() for p in d.iterdir() if p.name != ".gitkeep"]
    if existing and not overwrite:
        raise SystemExit(f"{out} already contains {len(existing)} files; pass --overwrite to rebuild")
    for p in existing:
        p.unlink()
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1", choices=["v1"])
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=HOUSEHOLD, help="dataset root (default data/household)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import imagehash  # data-prep dependency (tools/requirements-data.txt)

    classes = load_classes(args.version)
    cfg = load_source_mapping()
    oi_items, attribution = load_oi(classes, cfg)
    items = oi_items + load_custom()
    if not items:
        raise SystemExit("Nothing to build")

    print(f"Hashing {len(items)} images for near-duplicate detection...")
    hashes = {}
    for it in items:
        with Image.open(it["src"]) as im:
            hashes[it["stem"]] = int(str(imagehash.phash(im)), 16)
    uf = UnionFind()
    for it in items:
        uf.union(it["stem"], it["group"])  # custom session / OI image id
    pairs = near_duplicate_pairs(hashes)
    for a, b in pairs:
        uf.union(a, b)
    group_classes: dict[str, set[int]] = defaultdict(set)
    for it in items:
        g = uf.find(it["stem"])
        it["final_group"] = g
        group_classes[g].update(lbl[0] for lbl in it["labels"])
    val_groups = stratified_group_split(group_classes, args.val_ratio, args.seed)

    clear_output(args.out, args.overwrite)
    counts = {s: {"images": 0, "instances": defaultdict(int), "images_per_class": defaultdict(int)} for s in ("train", "val")}
    for it in items:
        split = "val" if it["final_group"] in val_groups else "train"
        shutil.copy2(it["src"], args.out / "images" / split / f"{it['stem']}.jpg")
        (args.out / "labels" / split / f"{it['stem']}.txt").write_text(
            "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in it["labels"]), encoding="utf-8")
        counts[split]["images"] += 1
        for c in {lbl[0] for lbl in it["labels"]}:
            counts[split]["images_per_class"][classes[c]] += 1
        for lbl in it["labels"]:
            counts[split]["instances"][classes[lbl[0]]] += 1

    if attribution:
        with open(args.out / "ATTRIBUTION.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(attribution[0]))
            w.writeheader()
            w.writerows(attribution)
    report = {"near_duplicate_pairs": len(pairs), "groups": len(group_classes), "val_groups": len(val_groups),
              "splits": {s: {"images": v["images"], "images_per_class": dict(v["images_per_class"]),
                             "instances": dict(v["instances"])} for s, v in counts.items()}}
    (CACHE / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{'class':17}{'train imgs':>11}{'val imgs':>10}{'train boxes':>12}{'val boxes':>10}")
    for c in classes:
        print(f"{c:17}{counts['train']['images_per_class'][c]:11}{counts['val']['images_per_class'][c]:10}"
              f"{counts['train']['instances'][c]:12}{counts['val']['instances'][c]:10}")
    print(f"\nTrain {counts['train']['images']} / val {counts['val']['images']} images; "
          f"{len(pairs)} near-duplicate pairs kept within one split. Output: {args.out}")


if __name__ == "__main__":
    main()
