"""Shared helpers for the household dataset pipeline (tools/dataset/0*_*.py).

Everything downloaded or generated lives under data/_cache/ (git-ignored).
Only data/household/{images,labels} receive the final YOLO dataset (also git-ignored).
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
HOUSEHOLD = DATA / "household"
CACHE = DATA / "_cache"
OI_CACHE = CACHE / "openimages"
OI_META = OI_CACHE / "metadata"
OI_IMAGES = OI_CACHE / "images"
CUSTOM_CACHE = CACHE / "custom"

# Large CSVs have very long rows only in rare cases; lift the default limit anyway.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

OI_SPLITS = ("train", "validation", "test")
OI_IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"


@dataclass(frozen=True)
class RemoteFile:
    key: str
    url: str
    filename: str

    @property
    def path(self) -> Path:
        return OI_META / self.filename


# Metadata needed for V1. No image files.
OI_METADATA = [
    RemoteFile("classes", "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
               "oidv7-class-descriptions-boxable.csv"),
    RemoteFile("hierarchy", "https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json",
               "bbox_labels_600_hierarchy.json"),
    RemoteFile("boxes_train", "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
               "oidv6-train-annotations-bbox.csv"),
    RemoteFile("boxes_validation", "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
               "validation-annotations-bbox.csv"),
    RemoteFile("boxes_test", "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
               "test-annotations-bbox.csv"),
    RemoteFile("images_train",
               "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
               "train-images-boxable-with-rotation.csv"),
    RemoteFile("images_validation",
               "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
               "validation-images-with-rotation.csv"),
    RemoteFile("images_test",
               "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
               "test-images-with-rotation.csv"),
]
META = {f.key: f for f in OI_METADATA}
BOX_FILES = {"train": META["boxes_train"], "validation": META["boxes_validation"], "test": META["boxes_test"]}
IMAGE_META_FILES = {"train": META["images_train"], "validation": META["images_validation"], "test": META["images_test"]}

CC_BY_2 = "https://creativecommons.org/licenses/by/2.0/"


# ---------------------------------------------------------------- config

def load_classes(version: str) -> list[str]:
    """Class names in id order from data.yaml (v1) or data_v2.yaml (v2)."""
    name = {"v1": "data.yaml", "v2": "data_v2.yaml"}[version]
    cfg = yaml.safe_load((HOUSEHOLD / name).read_text(encoding="utf-8"))
    names = [str(cfg["names"][i]) for i in sorted(cfg["names"])]
    if cfg.get("nc") != len(names) or len(set(names)) != len(names):
        raise SystemExit(f"{name}: nc/names mismatch or duplicate class names")
    return names


def load_source_mapping() -> dict:
    return yaml.safe_load((HOUSEHOLD / "source_mapping.yaml").read_text(encoding="utf-8"))


def oi_name_to_class(version: str) -> dict[str, str]:
    """Open Images display name -> household class, for the classes of `version`.

    Fails loudly if a class is missing from the mapping or an OI name maps to two classes.
    """
    classes = load_classes(version)
    mapping = load_source_mapping()
    out: dict[str, str] = {}
    for cls in classes:
        entry = mapping.get(cls)
        if not isinstance(entry, dict):
            raise SystemExit(f"source_mapping.yaml has no entry for class {cls!r}")
        for oi_name in entry.get("open_images", []):
            if oi_name in out and out[oi_name] != cls:
                raise SystemExit(f"Open Images label {oi_name!r} maps to both {out[oi_name]!r} and {cls!r}")
            out[oi_name] = cls
    return out


def read_oi_classes() -> tuple[dict[str, str], dict[str, str]]:
    """(mid -> display name, display name -> mid) from the boxable class list."""
    mid_to_name: dict[str, str] = {}
    with open(META["classes"].path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mid_to_name[row["LabelName"]] = row["DisplayName"]
    name_to_mid = {v: k for k, v in mid_to_name.items()}
    if len(name_to_mid) != len(mid_to_name):
        raise SystemExit("Open Images class list has duplicate display names")
    return mid_to_name, name_to_mid


# ---------------------------------------------------------------- geometry

def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two (xmin, ymin, xmax, ymax) boxes in any common unit."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- io

def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} TB"


def remote_size(url: str, timeout: float = 30) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        length = resp.headers.get("Content-Length")
        return int(length) if length else None


def download(url: str, dest: Path, expected_size: int | None = None, attempts: int = 5) -> int:
    """Resumable download to dest (via dest.part), retrying dropped connections. Returns the size."""
    for attempt in range(1, attempts + 1):
        try:
            return _download_once(url, dest, expected_size)
        except (OSError, IncompleteDownload) as exc:
            if attempt == attempts:
                raise SystemExit(f"{dest.name}: giving up after {attempts} attempts ({exc}); re-run to resume")
            print(f"    {dest.name}: {exc} - resuming (attempt {attempt + 1}/{attempts})", flush=True)
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


class IncompleteDownload(Exception):
    pass


def _download_once(url: str, dest: Path, expected_size: int | None, chunk: int = 1 << 20) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest.stat().st_size
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(url, headers=headers)
    started, last_print = time.time(), 0.0
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:  # server ignored Range: start over
            have = 0
        with open(part, "ab" if have else "wb") as out:
            done = have
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                now = time.time()
                if now - last_print > 5:
                    rate = (done - have) / max(now - started, 1e-6)
                    total = f" / {human_bytes(expected_size)}" if expected_size else ""
                    print(f"    {dest.name}: {human_bytes(done)}{total} ({human_bytes(rate)}/s)", flush=True)
                    last_print = now
    size = part.stat().st_size
    if expected_size is not None and size != expected_size:
        raise IncompleteDownload(f"connection ended at {human_bytes(size)} of {human_bytes(expected_size)}")
    os.replace(part, dest)
    return size


def require_metadata() -> None:
    missing = [f.filename for f in OI_METADATA if not f.path.exists()]
    if missing:
        raise SystemExit("Missing Open Images metadata; run 01_fetch_oi_metadata.py first:\n  " + "\n  ".join(missing))


def free_disk_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free
