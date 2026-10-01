"""Step 4: import your own labelled household photos.

Expected layout (exported from Label Studio / CVAT / Roboflow in YOLO format):
  data/raw/custom/<class>/<home-or-session>/
      IMG_0001.jpg
      IMG_0001.txt        # YOLO labels:  <class_id> <cx> <cy> <w> <h>   (normalised 0-1)
      classes.txt         # optional: the exporter's class order (one name per line)

If classes.txt exists (in the session or class folder), its names are mapped onto data.yaml
by name, so the exporter's id order doesn't matter. Otherwise ids must already match data.yaml.
Every image needs a label file (use an empty .txt for a photo with no V1 object).

Each photo is rotated upright from EXIF, saved WITHOUT metadata (removes GPS etc.) and
resized to max --max-side. Labels are assumed to have been drawn on the upright image, as
Label Studio and CVAT display it. Output: data/_cache/custom/{images,labels}/ and manifest.csv
(group = <class>/<session>, so a whole session stays in one split).

Usage:
  python tools/dataset/04_import_custom.py --src data/raw/custom
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CUSTOM_CACHE, DATA, load_classes  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-") or "x"


def class_order(folder: Path, stop: Path) -> list[str] | None:
    for d in (folder, folder.parent):
        f = d / "classes.txt"
        if f.exists():
            return [line.strip() for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        if d == stop:
            break
    return None


def read_labels(path: Path, remap: dict[int, int] | None, nc: int) -> tuple[list[str], list[str]]:
    lines, errors = [], []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{n} expected 5 values, got {len(parts)}")
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{n} not numeric")
            continue
        if remap is not None:
            if cid not in remap:
                errors.append(f"{path.name}:{n} class id {cid} is not a V1 class")
                continue
            cid = remap[cid]
        if not 0 <= cid < nc:
            errors.append(f"{path.name}:{n} class id {cid} out of range 0..{nc - 1}")
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errors.append(f"{path.name}:{n} coordinates outside 0-1")
            continue
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines, errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DATA / "raw" / "custom")
    ap.add_argument("--version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args()

    classes = load_classes(args.version)
    by_name = {c: i for i, c in enumerate(classes)}
    if not args.src.exists():
        raise SystemExit(f"{args.src} does not exist")
    (CUSTOM_CACHE / "images").mkdir(parents=True, exist_ok=True)
    (CUSTOM_CACHE / "labels").mkdir(parents=True, exist_ok=True)

    manifest, errors = [], []
    for img_path in sorted(p for p in args.src.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        rel = img_path.relative_to(args.src)
        if len(rel.parts) < 3:
            errors.append(f"{rel}: expected <class>/<session>/<file>")
            continue
        folder_class, session = rel.parts[0], rel.parts[1]
        if folder_class not in by_name:
            errors.append(f"{rel}: folder {folder_class!r} is not a {args.version} class")
            continue
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            errors.append(f"{rel}: missing label file {label_path.name}")
            continue
        order = class_order(img_path.parent, args.src)
        remap = None
        if order is not None:
            unknown = [n for n in order if n not in by_name]
            if unknown:
                errors.append(f"{rel}: classes.txt has names not in data.yaml: {unknown}")
                continue
            remap = {i: by_name[n] for i, n in enumerate(order)}
        lines, errs = read_labels(label_path, remap, len(classes))
        if errs:
            errors.extend(f"{rel.parent}/{e}" for e in errs)
            continue
        stem = f"custom_{safe(folder_class)}_{safe(session)}_{safe(img_path.stem)}"
        try:
            with Image.open(img_path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((args.max_side, args.max_side))
                im.save(CUSTOM_CACHE / "images" / f"{stem}.jpg", "JPEG", quality=92)  # no exif= -> stripped
        except Exception as exc:
            errors.append(f"{rel}: cannot read image ({exc})")
            continue
        (CUSTOM_CACHE / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        manifest.append({"stem": stem, "group": f"{folder_class}/{session}", "source_file": str(rel)})

    with open(CUSTOM_CACHE / "manifest.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stem", "group", "source_file"])
        w.writeheader()
        w.writerows(manifest)
    print(f"Imported {len(manifest)} photos into {CUSTOM_CACHE}")
    if errors:
        print(f"{len(errors)} problem(s) - fix these and re-run:")
        for e in errors[:50]:
            print("  !", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
