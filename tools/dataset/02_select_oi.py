"""Step 2: choose which Open Images images to use for the household dataset. Downloads nothing.

Usage:
  python tools/dataset/02_select_oi.py --version v1 --seed 42 --dry-run   # report only
  python tools/dataset/02_select_oi.py --version v1 --seed 42             # also writes the selection

Rules (data/household/source_mapping.yaml, docs/household-model-plan.md):
  * only Open Images labels mapped to a V1 class; everything else is ignored
  * boxes flagged IsDepiction are dropped; images with a V1 IsGroupOf box are dropped
    (the group's members would otherwise be unlabelled)
  * conflict rule: an image with a generic/overlapping label (e.g. "Oven", "Home appliance")
    that does not overlap a V1 box at IoU >= 0.5 is dropped
  * only images with Rotation 0 and license CC BY 2.0
  * greedy per-class selection, rarest class first, up to v1_targets; an image counts
    toward every V1 class it contains
Outputs (not in dry-run), in data/_cache/openimages/:
  v1_selection.csv  one row per image with source URL and attribution
  v1_boxes.csv      normalised V1 boxes for those images
  v1_selection_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BOX_FILES, CC_BY_2, IMAGE_META_FILES, OI_CACHE, OI_IMAGE_URL, OI_SPLITS, free_disk_bytes,
    human_bytes, iou, load_classes, load_source_mapping, oi_name_to_class, read_oi_classes,
    require_metadata,
)

CONFLICT_IOU = 0.5
BUFFER = 1.4  # over-sample before the per-image filters, then trim to the targets


def scan_presence(v1_mids, conflict_mids, bg_mids, mid_cls, cls_bit):
    """Pass 1: per image, which V1 classes it has (clean boxes), plus conflict/background flags."""
    mask: dict[tuple[str, str], int] = {}
    conflict: set[tuple[str, str]] = set()
    bg: set[tuple[str, str]] = set()
    any_v1: set[tuple[str, str]] = set()
    watch = v1_mids | conflict_mids | bg_mids
    for split, f in BOX_FILES.items():
        print(f"  pass 1: {f.filename}", flush=True)
        with open(f.path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                mid = row[2]
                if mid not in watch:
                    continue
                key = (split, row[0])
                if mid in v1_mids:
                    any_v1.add(key)
                    if row[11] != "1":  # not a depiction
                        mask[key] = mask.get(key, 0) | cls_bit[mid_cls[mid]]
                elif mid in conflict_mids:
                    conflict.add(key)
                elif row[10] != "1" and row[11] != "1":
                    bg.add(key)
    bg -= any_v1
    bg -= conflict
    return mask, bg


def eligible_keys(wanted: set[tuple[str, str]]) -> set[tuple[str, str]]:
    ok: set[tuple[str, str]] = set()
    for split, f in IMAGE_META_FILES.items():
        print(f"  license/rotation: {f.filename}", flush=True)
        with open(f.path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                key = (split, row["ImageID"])
                if key in wanted and row["Rotation"].strip() in ("", "0", "0.0") and row["License"].strip() == CC_BY_2:
                    ok.add(key)
    return ok


def greedy_pick(pool_by_cls, targets, order, rng, already=None):
    chosen: dict[tuple[str, str], str] = dict(already or {})
    for c in order:
        want = targets.get(c, 0)
        have = sum(1 for k in chosen if k in pool_by_cls[c])
        need = want - have
        if need <= 0:
            continue
        pool = sorted(k for k in pool_by_cls[c] if k not in chosen)
        rng.shuffle(pool)
        for k in pool[:need]:
            chosen[k] = c
    return chosen


def collect_boxes(keys, v1_mids, conflict_mids, mid_cls):
    boxes: dict[tuple[str, str], list] = defaultdict(list)
    conflicts: dict[tuple[str, str], list] = defaultdict(list)
    group_of: set[tuple[str, str]] = set()
    for split, f in BOX_FILES.items():
        print(f"  pass 2: {f.filename}", flush=True)
        with open(f.path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                mid = row[2]
                if mid not in v1_mids and mid not in conflict_mids:
                    continue
                key = (split, row[0])
                if key not in keys:
                    continue
                xmin, xmax, ymin, ymax = (float(v) for v in row[4:8])
                box = (xmin, ymin, xmax, ymax)
                if mid in v1_mids:
                    if row[10] == "1":
                        group_of.add(key)
                    elif row[11] != "1":
                        boxes[key].append((mid_cls[mid], *box))
                else:
                    conflicts[key].append(box)
    return boxes, conflicts, group_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1", choices=["v1"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="report only; write no selection files")
    args = ap.parse_args()
    require_metadata()

    classes = load_classes(args.version)
    cfg = load_source_mapping()
    targets = {k: int(v) for k, v in cfg.get("v1_targets", {}).items()}
    name_to_cls = oi_name_to_class(args.version)
    _, name_to_mid = read_oi_classes()
    mid_cls = {name_to_mid[n]: c for n, c in name_to_cls.items()}
    v1_mids = set(mid_cls)
    conflict_mids = {name_to_mid[n] for n in cfg.get("v1_open_images_conflict_classes", [])}
    bg_mids = {name_to_mid[n] for n in cfg.get("v1_open_images_background_classes", [])}
    cls_bit = {c: 1 << i for i, c in enumerate(classes)}

    print("Scanning metadata (no downloads)...")
    mask, bg = scan_presence(v1_mids, conflict_mids, bg_mids, mid_cls, cls_bit)
    ok = eligible_keys(set(mask) | bg)
    pool = {c: {k for k, m in mask.items() if m & cls_bit[c] and k in ok} for c in classes}
    bg_pool = {k for k in bg if k in ok}
    order = sorted((c for c in classes if targets.get(c, 0) > 0), key=lambda c: len(pool[c]))

    rng = random.Random(args.seed)
    buffered = {c: int(t * BUFFER) for c, t in targets.items()}
    candidates = greedy_pick(pool, buffered, order, rng)
    bg_list = sorted(bg_pool)
    rng.shuffle(bg_list)
    bg_candidates = bg_list[: int(targets.get("background", 0) * BUFFER)]

    keys = set(candidates) | set(bg_candidates)
    boxes, conflicts, group_of = collect_boxes(keys, v1_mids, conflict_mids, mid_cls)

    dropped = {"group_of": 0, "conflict": 0, "no_boxes": 0}
    clean_pool = {c: set() for c in classes}
    for k in candidates:
        if k in group_of:
            dropped["group_of"] += 1
            continue
        v1_boxes = boxes.get(k, [])
        if not v1_boxes:
            dropped["no_boxes"] += 1
            continue
        if any(all(iou(cb, b[1:]) < CONFLICT_IOU for b in v1_boxes) for cb in conflicts.get(k, [])):
            dropped["conflict"] += 1
            continue
        for b in v1_boxes:
            clean_pool[b[0]].add(k)
    final = greedy_pick(clean_pool, targets, order, random.Random(args.seed))
    final_bg = [k for k in bg_candidates if not boxes.get(k) and k not in conflicts and k not in group_of]
    final_bg = final_bg[: targets.get("background", 0)]

    # Attribution + size for the final images
    meta: dict[tuple[str, str], dict] = {}
    wanted = set(final) | set(final_bg)
    for split, f in IMAGE_META_FILES.items():
        with open(f.path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                key = (split, row["ImageID"])
                if key in wanted:
                    meta[key] = row

    per_class = []
    for c in classes:
        imgs = [k for k in final if any(b[0] == c for b in boxes[k])]
        per_class.append({
            "class": c, "eligible_images": len(pool[c]), "target": targets.get(c, 0),
            "selected_images": len(imgs), "boxes": sum(1 for k in imgs for b in boxes[k] if b[0] == c),
        })
    total_bytes = sum(int(meta[k]["OriginalSize"]) for k in wanted if k in meta and meta[k]["OriginalSize"].isdigit())
    report = {
        "seed": args.seed, "per_class": per_class, "background_selected": len(final_bg),
        "images_total": len(wanted), "dropped_candidates": dropped,
        "download_bytes_estimate": total_bytes, "free_disk_bytes": free_disk_bytes(OI_CACHE),
    }

    print(f"\n{'class':17}{'eligible':>10}{'target':>8}{'images':>8}{'boxes':>8}")
    for r in per_class:
        flag = "  <- below target" if 0 < r["selected_images"] < r["target"] else ""
        print(f"{r['class']:17}{r['eligible_images']:10,}{r['target']:8}{r['selected_images']:8}{r['boxes']:8}{flag}")
    print(f"{'background':17}{len(bg_pool):10,}{targets.get('background', 0):8}{len(final_bg):8}")
    print(f"\nDropped candidates: {dropped}")
    print(f"Unique images: {len(wanted):,}; estimated download {human_bytes(total_bytes)} "
          f"(free disk {human_bytes(report['free_disk_bytes'])})")

    (OI_CACHE / "v1_selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.dry_run:
        print("\nDry run: no selection written, nothing downloaded.")
        return

    sel_path, box_path = OI_CACHE / "v1_selection.csv", OI_CACHE / "v1_boxes.csv"
    with open(sel_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["oi_split", "image_id", "reason", "url", "license", "author", "author_profile",
                    "original_url", "title", "original_size"])
        for k in sorted(wanted):
            m = meta.get(k, {})
            w.writerow([k[0], k[1], final.get(k, "background"), OI_IMAGE_URL.format(split=k[0], image_id=k[1]),
                        m.get("License", ""), m.get("Author", ""), m.get("AuthorProfileURL", ""),
                        m.get("OriginalURL", ""), m.get("Title", ""), m.get("OriginalSize", "")])
    with open(box_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["oi_split", "image_id", "class", "xmin", "ymin", "xmax", "ymax"])
        for k in sorted(final):
            for cls, x0, y0, x1, y1 in boxes[k]:
                w.writerow([k[0], k[1], cls, f"{x0:.6f}", f"{y0:.6f}", f"{x1:.6f}", f"{y1:.6f}"])
    print(f"\nWrote {sel_path} and {box_path}")


if __name__ == "__main__":
    main()
