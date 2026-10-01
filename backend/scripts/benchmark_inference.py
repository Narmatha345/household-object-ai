"""Benchmark local inference: PyTorch (.pt via Ultralytics) vs ONNX Runtime (.onnx).

Uses the app's own detector classes, so it measures the real request code path.
Each configuration runs in a fresh subprocess, so memory and thread settings
don't leak between runs.

    python scripts/benchmark_inference.py --onnx models/household-yolo26n.onnx \
        --images ../data/household/images/val/oi_1beafa4064d0ae71.jpg --iterations 10
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
THRESHOLD = 0.70


def _run_one(backend: str, model: str, threads: int, images: list[str], iterations: int, image_size: int) -> dict:
    """Runs inside the child process."""
    import psutil
    from PIL import Image

    sys.path.insert(0, str(BACKEND_DIR))
    proc = psutil.Process()
    rss = lambda: round(proc.memory_info().rss / 2**20)  # noqa: E731

    if backend == "pytorch":
        from app.ml.detector import LocalObjectDetector

        detector = LocalObjectDetector(Path(model), image_size=image_size)
    else:
        from app.ml.onnx_detector import OnnxObjectDetector

        detector = OnnxObjectDetector(Path(model), image_size=image_size, threads=threads or None)

    t0 = time.perf_counter()
    detector.load()
    load_ms = (time.perf_counter() - t0) * 1000
    rss_loaded = rss()

    report: dict = {"backend": backend, "threads": threads, "load_ms": round(load_ms), "rss_after_load_mb": rss_loaded}
    per_image = []
    for path in images:
        image = Image.open(path).convert("RGB")
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            detections = detector.detect(image)
            times.append((time.perf_counter() - start) * 1000)
        steady = times[1:] or times
        per_image.append({
            "image": Path(path).name,
            "first_ms": round(times[0], 1),
            "avg_ms": round(statistics.mean(steady), 1),
            "min_ms": round(min(steady), 1),
            "max_ms": round(max(steady), 1),
            "detections": [
                {"label": d.label, "confidence": round(d.confidence, 4), "box": d.box}
                for d in sorted(detections, key=lambda d: d.confidence, reverse=True)
                if d.confidence >= THRESHOLD
            ],
        })
    report["rss_after_inference_mb"] = rss()
    report["torch_imported"] = "torch" in sys.modules
    report["images"] = per_image
    return report


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _compare(ref: list[dict], other: list[dict]) -> dict:
    """Match detections by label + best IoU; report confidence and box differences."""
    unmatched = list(other)
    pairs = []
    for r in ref:
        candidates = [o for o in unmatched if o["label"] == r["label"]]
        if not candidates:
            continue
        best = max(candidates, key=lambda o: _iou(r["box"], o["box"]))
        unmatched.remove(best)
        pairs.append((r, best))
    return {
        "same_labels": sorted(d["label"] for d in ref) == sorted(d["label"] for d in other),
        "matched": len(pairs),
        "ref_count": len(ref),
        "other_count": len(other),
        "max_conf_diff": round(max((abs(r["confidence"] - o["confidence"]) for r, o in pairs), default=0.0), 4),
        "min_box_iou": round(min((_iou(r["box"], o["box"]) for r, o in pairs), default=1.0), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pt", default="models/household-yolo26n.pt")
    parser.add_argument("--onnx", default="models/household-yolo26n.onnx")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 0], help="0 = library default")
    parser.add_argument("--child", nargs=2, metavar=("BACKEND", "THREADS"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        backend, threads = args.child[0], int(args.child[1])
        model = args.pt if backend == "pytorch" else args.onnx
        print(json.dumps(_run_one(backend, model, threads, args.images, args.iterations, args.imgsz)))
        return

    results = []
    for backend in ("pytorch", "onnx"):
        if backend == "onnx" and not Path(args.onnx).exists():
            print(f"skip onnx: {args.onnx} not found")
            continue
        for threads in args.threads:
            env = dict(os.environ)
            env.pop("OMP_NUM_THREADS", None)
            if threads:
                env["OMP_NUM_THREADS"] = str(threads)  # torch reads this at import
            cmd = [sys.executable, __file__, "--child", backend, str(threads), "--pt", args.pt, "--onnx", args.onnx,
                   "--iterations", str(args.iterations), "--imgsz", str(args.imgsz), "--images", *args.images]
            out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=BACKEND_DIR, check=True)
            results.append(json.loads(out.stdout.strip().splitlines()[-1]))

    print(f"\n{'backend':8} {'thr':>4} {'load':>6} {'RSS load':>9} {'RSS run':>8} {'torch':>6} | image: first / avg / min / max ms")
    for r in results:
        thr = r["threads"] or "def"
        imgs = "  ".join(f"{i['image'][:14]}: {i['first_ms']:.0f}/{i['avg_ms']:.0f}/{i['min_ms']:.0f}/{i['max_ms']:.0f}"
                         for i in r["images"])
        print(f"{r['backend']:8} {thr:>4} {r['load_ms']:>5}ms {r['rss_after_load_mb']:>7}MB {r['rss_after_inference_mb']:>6}MB "
              f"{str(r['torch_imported']):>6} | {imgs}")

    ref = next((r for r in results if r["backend"] == "pytorch"), None)
    print("\nDetections >= 70% (PyTorch reference vs each run):")
    for r in results:
        for i, img in enumerate(r["images"]):
            dets = ", ".join(f"{d['label']} {d['confidence']:.2f}" for d in img["detections"])
            line = f"  {r['backend']:7} thr={r['threads'] or 'def':<3} {img['image'][:22]:22} {dets}"
            if ref and r is not ref:
                line += f"\n      vs PyTorch: {_compare(ref['images'][i]['detections'], img['detections'])}"
            print(line)


if __name__ == "__main__":
    main()
