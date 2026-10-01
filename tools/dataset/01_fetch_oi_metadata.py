"""Step 1: download Open Images V7 *metadata* (no images) and validate the V1 mapping.

Usage:
  python tools/dataset/01_fetch_oi_metadata.py              # download missing files, then validate
  python tools/dataset/01_fetch_oi_metadata.py --check-only # validate already-downloaded files

Validation (no image downloads):
  * every mapped Open Images label exists in the boxable class list
  * no Open Images label maps to two household classes
  * hierarchy: unmapped child labels of mapped classes, and mapped labels under conflict classes
  * per-class availability across train/validation/test after the box filters
    (IsDepiction / IsGroupOf), the rotation filter and the CC BY 2.0 license filter
Writes data/_cache/openimages/metadata_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BOX_FILES, CC_BY_2, IMAGE_META_FILES, META, OI_CACHE, OI_IMAGE_URL, OI_METADATA, OI_SPLITS,
    download, free_disk_bytes, human_bytes, load_classes, load_source_mapping, oi_name_to_class,
    read_oi_classes, remote_size,
)

BOX_HEADER = ["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax",
              "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]
RESIZED_IMAGE_BYTES = 160_000  # typical JPEG after resizing to max 1024 px (estimate)


def fetch(check_only: bool) -> dict[str, int]:
    sizes: dict[str, int] = {}
    todo = []
    for f in OI_METADATA:
        expected = remote_size(f.url)
        if f.path.exists() and expected is not None and f.path.stat().st_size == expected:
            sizes[f.filename] = expected
        elif check_only:
            raise SystemExit(f"--check-only but {f.filename} is missing or incomplete")
        else:
            todo.append((f, expected))
    need = sum(e or 0 for _, e in todo)
    if todo:
        free = free_disk_bytes(OI_CACHE)
        print(f"Downloading {len(todo)} metadata file(s), {human_bytes(need)} (free disk: {human_bytes(free)})")
        if need * 1.2 > free:
            raise SystemExit("Not enough free disk space")
    for f, expected in todo:
        print(f"  -> {f.filename} ({human_bytes(expected or 0)})")
        sizes[f.filename] = download(f.url, f.path, expected)
    return sizes


def hierarchy_index(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """(child mid -> parent mid, mid -> all descendant mids)."""
    tree = json.loads(path.read_text(encoding="utf-8"))
    parent: dict[str, str] = {}
    desc: dict[str, set[str]] = defaultdict(set)

    def walk(node: dict, ancestors: list[str]) -> None:
        mid = node["LabelName"]
        for a in ancestors:
            desc[a].add(mid)
        if ancestors:
            parent.setdefault(mid, ancestors[-1])
        for child in node.get("Subcategory", []):
            walk(child, ancestors + [mid])

    walk(tree, [])
    return parent, desc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true", help="do not download; validate existing files")
    ap.add_argument("--version", default="v1", choices=["v1"])
    args = ap.parse_args()

    sizes = fetch(args.check_only)
    total_meta = sum(sizes.values())
    print(f"\nMetadata present: {len(sizes)} files, {human_bytes(total_meta)} ({total_meta:,} bytes)")

    problems: list[str] = []
    classes = load_classes(args.version)
    mapping_cfg = load_source_mapping()
    name_to_cls = oi_name_to_class(args.version)          # raises on double mapping
    mid_to_name, name_to_mid = read_oi_classes()

    # --- 1. mapping names exist ------------------------------------------------
    unknown = [n for n in name_to_cls if n not in name_to_mid]
    if unknown:
        problems.append(f"Mapped Open Images labels not in the boxable list: {unknown}")
    conflict_names = mapping_cfg.get("v1_open_images_conflict_classes", [])
    background_names = mapping_cfg.get("v1_open_images_background_classes", [])
    for n in conflict_names + background_names:
        if n not in name_to_mid:
            problems.append(f"Conflict/background label not in the boxable list: {n!r}")
    overlap = set(conflict_names) & set(name_to_cls)
    if overlap:
        problems.append(f"Labels are both mapped and conflict: {sorted(overlap)}")

    cls_to_mids: dict[str, set[str]] = defaultdict(set)
    for n, c in name_to_cls.items():
        if n in name_to_mid:
            cls_to_mids[c].add(name_to_mid[n])
    no_source = [c for c in classes if not cls_to_mids.get(c)]
    custom_only = [c for c in no_source if mapping_cfg.get(c, {}).get("custom")]
    unmapped = [c for c in no_source if c not in custom_only]
    if unmapped:
        problems.append(f"Classes with no Open Images source and not marked custom: {unmapped}")

    # --- 2. hierarchy ------------------------------------------------------------
    parent, desc = hierarchy_index(META["hierarchy"].path)
    mapped_mids = {m for mids in cls_to_mids.values() for m in mids}
    conflict_mids = {name_to_mid[n] for n in conflict_names if n in name_to_mid}
    background_mids = {name_to_mid[n] for n in background_names if n in name_to_mid}
    hierarchy_notes = []
    for mid in sorted(mapped_mids, key=lambda m: mid_to_name[m]):
        children = [c for c in desc.get(mid, ()) if c in mid_to_name and c not in mapped_mids]
        if children:
            hierarchy_notes.append(f"{mid_to_name[mid]} ({name_to_cls[mid_to_name[mid]]}) has unmapped "
                                   f"sub-labels: {sorted(mid_to_name[c] for c in children)}")
        anc, p = [], parent.get(mid)
        while p:
            anc.append(p)
            p = parent.get(p)
        under_conflict = [mid_to_name[a] for a in anc if a in conflict_mids]
        if under_conflict:
            hierarchy_notes.append(f"{mid_to_name[mid]} is a sub-label of conflict class(es) {under_conflict} "
                                   f"(fine: the conflict rule only drops non-overlapping generic boxes)")

    # --- 3. availability -----------------------------------------------------------
    watch = mapped_mids | conflict_mids | background_mids
    stats = {s: {"boxes": defaultdict(int), "clean": defaultdict(int)} for s in OI_SPLITS}
    img_classes: dict[tuple[str, str], int] = {}     # (split, image) -> bitmask of V1 classes (clean boxes)
    bg_candidates: set[tuple[str, str]] = set()
    has_conflict: set[tuple[str, str]] = set()
    cls_bit = {c: 1 << i for i, c in enumerate(classes)}
    mid_cls = {m: c for c, mids in cls_to_mids.items() for m in mids}
    for split, f in BOX_FILES.items():
        print(f"Scanning {f.filename} ...", flush=True)
        with open(f.path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            # The train file adds XClick* columns after these; the positions used below are the same.
            if header[: len(BOX_HEADER)] != BOX_HEADER:
                problems.append(f"{f.filename}: unexpected header {header}")
                continue
            for row in reader:
                mid = row[2]
                if mid not in watch:
                    continue
                key = (split, row[0])
                clean = row[10] != "1" and row[11] != "1"
                if mid in mid_cls:
                    c = mid_cls[mid]
                    stats[split]["boxes"][c] += 1
                    if clean:
                        stats[split]["clean"][c] += 1
                        img_classes[key] = img_classes.get(key, 0) | cls_bit[c]
                elif mid in conflict_mids:
                    has_conflict.add(key)
                elif mid in background_mids and clean:
                    bg_candidates.add(key)
    bg_candidates -= set(img_classes)
    bg_candidates -= has_conflict

    eligible: dict[str, dict[str, int]] = {c: {s: 0 for s in OI_SPLITS} for c in classes}
    raw_images: dict[str, dict[str, int]] = {c: {s: 0 for s in OI_SPLITS} for c in classes}
    rotated = non_cc = bg_eligible = 0
    orig_bytes: list[int] = []
    wanted = set(img_classes) | bg_candidates
    for split, f in IMAGE_META_FILES.items():
        print(f"Scanning {f.filename} ...", flush=True)
        with open(f.path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for need in ("ImageID", "License", "Rotation", "OriginalSize"):
                if need not in (reader.fieldnames or []):
                    problems.append(f"{f.filename}: missing column {need}")
            for row in reader:
                key = (split, row["ImageID"])
                if key not in wanted:
                    continue
                ok_rot = row["Rotation"].strip() in ("", "0", "0.0")
                ok_lic = row["License"].strip() == CC_BY_2
                mask = img_classes.get(key, 0)
                for c, bit in cls_bit.items():
                    if mask & bit:
                        raw_images[c][split] += 1
                        if ok_rot and ok_lic:
                            eligible[c][split] += 1
                if key in bg_candidates and ok_rot and ok_lic:
                    bg_eligible += 1
                rotated += not ok_rot
                non_cc += not ok_lic
                if ok_rot and ok_lic and row["OriginalSize"].isdigit():
                    orig_bytes.append(int(row["OriginalSize"]))

    # --- 4. estimates ------------------------------------------------------------------
    targets = mapping_cfg.get("v1_targets", {})
    per_class = []
    selected_total = 0
    for c in classes:
        avail = sum(eligible[c].values())
        target = int(targets.get(c, 0))
        take = min(avail, target)
        selected_total += take
        per_class.append({
            "class": c,
            "open_images_labels": sorted(n for n, cc in name_to_cls.items() if cc == c),
            "raw_boxes": sum(stats[s]["boxes"][c] for s in OI_SPLITS),
            "clean_boxes": sum(stats[s]["clean"][c] for s in OI_SPLITS),
            "images_any_license": sum(raw_images[c].values()),
            "eligible_images": avail,
            "eligible_by_split": eligible[c],
            "target": target,
            "expected_selected": take,
            "custom_photos": bool(mapping_cfg.get(c, {}).get("custom")),
        })
    bg_target = int(targets.get("background", 0))
    bg_take = min(bg_eligible, bg_target)
    # Per-class picks overlap (one image can serve several classes); selection counts each image once.
    oi_images_upper = selected_total + bg_take
    image_host_ok = False
    try:
        probe = OI_IMAGE_URL.format(split="validation", image_id="0001eeaf4aed83f9")
        image_host_ok = remote_size(probe) is not None  # HEAD only, no image body
    except Exception as exc:
        problems.append(f"Image host HEAD check failed: {exc}")
    report = {
        "image_host_reachable": image_host_ok,
        "metadata_files": sizes,
        "metadata_total_bytes": total_meta,
        "classes": per_class,
        "background": {"eligible_images": bg_eligible, "target": bg_target, "expected_selected": bg_take},
        "excluded_images_rotation": rotated,
        "excluded_images_non_cc_by_2": non_cc,
        "custom_only_classes": custom_only,
        "hierarchy_notes": hierarchy_notes,
        "problems": problems,
        "estimate": {
            "open_images_images_upper_bound": oi_images_upper,
            "avg_original_bytes": int(sum(orig_bytes) / len(orig_bytes)) if orig_bytes else None,
            "download_bytes_upper_bound": int(oi_images_upper * (sum(orig_bytes) / len(orig_bytes)))
            if orig_bytes else None,
            "resized_bytes_estimate": oi_images_upper * RESIZED_IMAGE_BYTES,
        },
    }
    out = OI_CACHE / "metadata_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- 5. print ------------------------------------------------------------------------
    print(f"\n{'class':17}{'OI labels':52}{'clean boxes':>12}{'eligible imgs':>14}{'target':>8}{'expect':>8}")
    for r in per_class:
        labels = ", ".join(r["open_images_labels"]) or ("(custom photos only)" if r["custom_photos"] else "-")
        print(f"{r['class']:17}{labels[:50]:52}{r['clean_boxes']:12,}{r['eligible_images']:14,}"
              f"{r['target']:8}{r['expected_selected']:8}")
    print(f"{'background':17}{'(indoor, no V1 object, no conflict label)':52}{'':12}{bg_eligible:14,}"
          f"{bg_target:8}{bg_take:8}")
    print("\nHierarchy notes:" if hierarchy_notes else "\nHierarchy notes: none")
    for n in hierarchy_notes:
        print("  -", n)
    print(f"\nExcluded by rotation: {rotated:,} images; by license (not CC BY 2.0): {non_cc:,} images")
    est = report["estimate"]
    print(f"Estimated Open Images images to select (upper bound, before de-dup): {oi_images_upper:,}")
    if est["download_bytes_upper_bound"]:
        print(f"Estimated original download (upper bound): {human_bytes(est['download_bytes_upper_bound'])}"
              f"; resized dataset ~{human_bytes(est['resized_bytes_estimate'])}")
    print(f"Image host reachable (HEAD only): {image_host_ok}")
    print("\nPROBLEMS:" if problems else "\nPROBLEMS: none")
    for p in problems:
        print("  !", p)
    print(f"\nReport written to {out}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
