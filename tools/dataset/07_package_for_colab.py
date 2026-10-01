"""Step 7: package the finished YOLO dataset into ONE zip for Google Colab / Kaggle.

Usage:
  python tools/dataset/07_package_for_colab.py

Creates data/_cache/colab/household_dataset.zip (git-ignored) containing only:
  household/data.yaml
  household/ATTRIBUTION.csv
  household/images/{train,val}/*.jpg
  household/labels/{train,val}/*.txt
Label caches (*.cache), .gitkeep files and the V2/source configs are left out.
JPEGs are stored uncompressed (they don't shrink), so packing is fast.

Also writes household_dataset.manifest.json with per-split counts and the zip's SHA-256,
which the Colab notebook uses to verify the upload.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CACHE, HOUSEHOLD, human_bytes, load_classes  # noqa: E402

OUT_DIR = CACHE / "colab"
ZIP_PATH = OUT_DIR / "household_dataset.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    classes = load_classes("v1")
    splits = {}
    files: list[tuple[Path, str]] = [(HOUSEHOLD / "data.yaml", "household/data.yaml")]
    if (HOUSEHOLD / "ATTRIBUTION.csv").exists():
        files.append((HOUSEHOLD / "ATTRIBUTION.csv", "household/ATTRIBUTION.csv"))
    for split in ("train", "val"):
        images = sorted((HOUSEHOLD / "images" / split).glob("*.jpg"))
        labels = sorted((HOUSEHOLD / "labels" / split).glob("*.txt"))
        if {p.stem for p in images} != {p.stem for p in labels}:
            raise SystemExit(f"{split}: images and labels don't match - run 06_audit.py")
        if not images:
            raise SystemExit(f"{split}: no images - build the dataset first (05_build_yolo.py)")
        splits[split] = {"images": len(images), "labels": len(labels)}
        files += [(p, f"household/images/{split}/{p.name}") for p in images]
        files += [(p, f"household/labels/{split}/{p.name}") for p in labels]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ZIP_PATH.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
        for src, arc in files:
            method = zipfile.ZIP_STORED if src.suffix == ".jpg" else zipfile.ZIP_DEFLATED
            zf.write(src, arc, compress_type=method)
    tmp.replace(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        bad = zf.testzip()
        if bad:
            raise SystemExit(f"Zip integrity check failed at {bad}")
    manifest = {
        "zip": ZIP_PATH.name,
        "sha256": sha256(ZIP_PATH),
        "bytes": ZIP_PATH.stat().st_size,
        "files": len(files),
        "nc": len(classes),
        "names": classes,
        "splits": splits,
    }
    (OUT_DIR / "household_dataset.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {ZIP_PATH} ({human_bytes(manifest['bytes'])}, {len(files):,} files)")
    print(f"  train: {splits['train']['images']:,} images | val: {splits['val']['images']:,} images")
    print(f"  sha256: {manifest['sha256']}")
    print(f"Upload this zip to Google Drive: My Drive/household-object-ai/{ZIP_PATH.name}")


if __name__ == "__main__":
    main()
