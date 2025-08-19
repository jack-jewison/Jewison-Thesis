#!/usr/bin/env python3
"""
Benchmark multiple Roboflow models locally on a set of images.

Example:
  python rf_benchmark.py \
    --images ./img \
    --models flower-counter/10 flower-counter/11 flower-counter/12 flower-counter/13 \
    --api-key "YOUR_API_KEY" \
    --warmup 2 --repeats 1 --confidence 0.4 --csv bench.csv

Install:
  pip install inference
"""

import argparse
import csv
import os
import sys
import time
import statistics as stats
from pathlib import Path
from typing import List, Dict, Any

# Roboflow Inference (local, no server)
try:
    from inference import get_model
except Exception as e:
    print("Missing dependency. Install with:  pip install inference")
    raise

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def find_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Images folder not found: {folder}")
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not imgs:
        raise FileNotFoundError(f"No images found in: {folder}")
    return sorted(imgs)

def load_models(model_ids: List[str], api_key: str):
    models = {}
    for mid in model_ids:
        print(f"Loading model: {mid} ...", flush=True)
        # get_model downloads & initializes the model locally for inference
        models[mid] = get_model(model_id=mid, api_key=api_key)
    return models

def time_infer(model, image_path: Path, confidence: float) -> float:
    t0 = time.perf_counter()
    _ = model.infer(str(image_path), confidence=confidence)
    return (time.perf_counter() - t0) * 1000.0  # milliseconds

def summarize(latencies_ms: List[float]) -> Dict[str, Any]:
    latencies_ms = [x for x in latencies_ms if x is not None]
    if not latencies_ms:
        return {"count": 0, "avg_ms": None, "med_ms": None, "p95_ms": None, "fps": None}
    avg = sum(latencies_ms) / len(latencies_ms)
    med = stats.median(latencies_ms)
    # If <20 samples, p95 ~ max as a conservative proxy
    p95 = stats.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 20 else max(latencies_ms)
    fps = 1000.0 / avg if avg > 0 else None
    return {"count": len(latencies_ms), "avg_ms": avg, "med_ms": med, "p95_ms": p95, "fps": fps}

def human(ms: float) -> str:
    return "n/a" if ms is None else f"{ms:.2f} ms"

def main():
    ap = argparse.ArgumentParser(description="Benchmark Roboflow models locally on images.")
    ap.add_argument("--images", required=True, type=Path, help="Folder with images")
    ap.add_argument("--models", nargs="+", required=True, help="Roboflow model IDs (e.g., project/10)")
    ap.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"), help="Roboflow API key")
    ap.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    ap.add_argument("--warmup", type=int, default=1, help="Warm-up runs per model (on the first image)")
    ap.add_argument("--repeats", type=int, default=1, help="Extra repeats per image for tighter timing")
    ap.add_argument("--csv", type=Path, default=None, help="Optional path to save per-inference results CSV")
    args = ap.parse_args()

    if not args.api_key:
        print("Error: Provide --api-key or set ROBOFLOW_API_KEY.", file=sys.stderr)
        sys.exit(1)

    images = find_images(args.images)
    print(f"Found {len(images)} images under {args.images}")

    models = load_models(args.models, args.api_key)

    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = args.csv.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["model_id", "image", "latency_ms"])

    for mid, model in models.items():
        print(f"\n=== Benchmarking {mid} ===")
        # Warm-up on first image
        first_img = images[0]
        for i in range(max(0, args.warmup)):
            try:
                _ = model.infer(str(first_img), confidence=args.confidence)
            except Exception as e:
                print(f"Warmup error on {first_img.name}: {e}")

        latencies: List[float] = []
        for img in images:
            for _ in range(max(1, args.repeats)):
                try:
                    ms = time_infer(model, img, args.confidence)
                    latencies.append(ms)
                    if csv_writer:
                        csv_writer.writerow([mid, str(img), f"{ms:.3f}"])
                except Exception as e:
                    print(f"Inference error on {img.name}: {e}")

        summary = summarize(latencies)
        if summary["count"] == 0:
            print("No successful inferences.")
            continue

        print(f"Images x repeats: {summary['count']}")
        print(f"Avg: {human(summary['avg_ms'])}  "
              f"Median: {human(summary['med_ms'])}  "
              f"P95: {human(summary['p95_ms'])}  "
              f"Throughput: {summary['fps']:.2f} FPS")

    if csv_file:
        csv_file.close()
        print(f"\nSaved per-inference results to: {args.csv}")

if __name__ == "__main__":
    main()
