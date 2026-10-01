"""Step 3: download ONLY the images listed in the selection from 02_select_oi.py.

Usage:
  python tools/dataset/03_download_oi.py --selection data/_cache/openimages/v1_selection.csv
  python tools/dataset/03_download_oi.py --selection ... --limit 20      # small trial first

Images come from the public Open Images bucket over HTTPS (no account). Each image is
checked to decode, converted to RGB and resized so the longer side is <= --max-side,
then saved to data/_cache/openimages/images/<image_id>.jpg. Re-running skips finished
images; failures are listed in data/_cache/openimages/download_failures.csv.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OI_CACHE, OI_IMAGES, free_disk_bytes, human_bytes  # noqa: E402


def fetch_one(row: dict, max_side: int, retries: int = 3) -> tuple[str, str | None]:
    dest = OI_IMAGES / f"{row['image_id']}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return row["image_id"], None
    last_err = "unknown"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(row["url"], timeout=60) as resp:
                data = resp.read()
            with Image.open(io.BytesIO(data)) as img:
                img = img.convert("RGB")
                img.thumbnail((max_side, max_side))
                tmp = dest.with_suffix(".tmp")
                img.save(tmp, "JPEG", quality=90)
            tmp.replace(dest)
            return row["image_id"], None
        except Exception as exc:  # network or decode error; retry with backoff
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * (attempt + 1))
    return row["image_id"], last_err


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", type=Path, default=OI_CACHE / "v1_selection.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="download only the first N (trial run)")
    args = ap.parse_args()

    if not args.selection.exists():
        raise SystemExit(f"{args.selection} not found; run 02_select_oi.py without --dry-run first")
    with open(args.selection, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    OI_IMAGES.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (OI_IMAGES / f"{r['image_id']}.jpg").exists()]
    est = sum(int(r["original_size"]) for r in todo if r.get("original_size", "").isdigit())
    free = free_disk_bytes(OI_IMAGES)
    print(f"{len(rows):,} selected, {len(todo):,} to download (~{human_bytes(est)} before resizing), "
          f"free disk {human_bytes(free)}")
    if est > free * 0.8:
        raise SystemExit("Not enough free disk space")

    failures: list[tuple[str, str]] = []
    done = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, r, args.max_side) for r in todo]
        for fut in as_completed(futures):
            image_id, err = fut.result()
            done += 1
            if err:
                failures.append((image_id, err))
            if done % 200 == 0 or done == len(todo):
                rate = done / max(time.time() - started, 1e-6)
                print(f"  {done:,}/{len(todo):,} ({rate:.1f} img/s), failures {len(failures)}", flush=True)

    fail_path = OI_CACHE / "download_failures.csv"
    with open(fail_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "error"])
        w.writerows(failures)
    ok = sum(1 for r in rows if (OI_IMAGES / f"{r['image_id']}.jpg").exists())
    print(f"Done: {ok:,}/{len(rows):,} images present; {len(failures)} failed (see {fail_path})")


if __name__ == "__main__":
    main()
