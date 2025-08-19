#!/usr/bin/env python3
"""
Analyze a locally downloaded Roboflow COCO dataset to compute the number of
annotations per image, save a histogram, and emit a LaTeX table with mean/std.

Examples:

# PowerShell (one line)
python .\rf_local_annotation_stats.py --dataset-dir "Flower Counter.v13i.coco" --save-hist hist.png --save-tex annotation_summary.tex --xmax 150 --bin01 --bins 30 --figwidth 6 --figheight 4 --dpi 300 --xtickstep 10 --pdf

# Git Bash/WSL (multi-line)
python rf_local_annotation_stats.py \
  --dataset-dir "Flower Counter.v13i.coco" \
  --save-hist hist.png \
  --save-tex annotation_summary.tex \
  --xmax 150 --bin01 --bins 30 --figwidth 6 --figheight 4 --dpi 300 --xtickstep 10 --pdf

Dependencies:
  pip install numpy matplotlib pandas
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

SPLIT_DIRS = ["train", "valid", "val", "test"]

def find_coco_jsons(dataset_dir: Path) -> List[Path]:
    """Find COCO annotation jsons in common Roboflow export locations."""
    jsons: List[Path] = []

    # Case 1: annotations/ folder at root
    anno_dir = dataset_dir / "annotations"
    if anno_dir.exists():
        for js in sorted(anno_dir.glob("*.json")):
            try:
                with open(js, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if isinstance(d, dict) and "images" in d and "annotations" in d:
                    jsons.append(js)
            except Exception:
                pass

    # Case 2: split-level files like train/_annotations.coco.json
    for split in SPLIT_DIRS:
        sd = dataset_dir / split
        if not sd.exists():
            continue
        for js in sorted(sd.glob("*.json")):
            try:
                with open(js, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if isinstance(d, dict) and "images" in d and "annotations" in d:
                    jsons.append(js)
            except Exception:
                pass

    # Deduplicate
    uniq = []
    seen = set()
    for p in jsons:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

def counts_from_coco(json_path: Path) -> Dict[str, int]:
    """Count annotations per image for one COCO json."""
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    img_id_to_name = {img["id"]: img["file_name"] for img in d.get("images", []) if "id" in img and "file_name" in img}
    counts = {img_id_to_name[k]: 0 for k in img_id_to_name}
    for ann in d.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id in img_id_to_name:
            counts[img_id_to_name[img_id]] += 1
    return counts

def main():
    ap = argparse.ArgumentParser(description="Local COCO dataset annotation histogram + LaTeX summary")
    ap.add_argument("--dataset-dir", type=Path, required=True, help="Path to local COCO dataset folder (e.g., 'Flower Counter.v13i.coco')")
    ap.add_argument("--save-hist", type=Path, default=Path("annotation_hist.png"), help="Path to save histogram image (PNG)")
    ap.add_argument("--save-tex", type=Path, default=Path("annotation_summary.tex"), help="Path to save LaTeX table")
    ap.add_argument("--bins", type=int, default=30, help="Number of bins for histogram (used when --bin01 is off, or for >1 range when on)")
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV of per-image counts")
    ap.add_argument("--xmax", type=int, default=None, help="Max x-axis range for histogram display (e.g., 150)")
    ap.add_argument("--bin01", action="store_true", help="Use a dedicated first bin that groups counts 0 and 1 together")

    # Publication-quality figure controls
    ap.add_argument("--dpi", type=int, default=300, help="PNG export resolution (dpi)")
    ap.add_argument("--figwidth", type=float, default=6.0, help="Figure width in inches")
    ap.add_argument("--figheight", type=float, default=4.0, help="Figure height in inches")
    ap.add_argument("--xtickstep", type=int, default=10, help="Step between major x ticks (0 = auto)")
    ap.add_argument("--pdf", action="store_true", help="Also export a vector PDF next to the PNG")
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    coco_jsons = find_coco_jsons(dataset_dir)
    if not coco_jsons:
        raise FileNotFoundError(f"No COCO annotation JSONs found under: {dataset_dir}")

    # Aggregate counts across all found jsons (e.g., train/valid/test)
    total_counts: Dict[str, int] = {}
    for js in coco_jsons:
        sub_counts = counts_from_coco(js)
        for k, v in sub_counts.items():
            total_counts[k] = total_counts.get(k, 0) + v

    if not total_counts:
        raise SystemExit("No images or annotations found.")

    df = pd.DataFrame([{"image": k, "annotations": v} for k, v in total_counts.items()])
    df = df.sort_values("annotations").reset_index(drop=True)

    n = len(df)
    mean = float(np.mean(df["annotations"]))
    std = float(np.std(df["annotations"], ddof=1)) if n > 1 else 0.0

    print(f"Images: {n} | Mean annotations/image: {mean:.3f} | Std dev: {std:.3f}")

    # Optional CSV
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Saved per-image counts to: {args.csv}")

    # Build custom bins if requested
    custom_bins = None
    data_max = int(df["annotations"].max()) if len(df) else 0
    right_edge = max(data_max, args.xmax if args.xmax is not None else data_max)
    if args.bin01:
        # First bin groups [0, 1] via [-0.5, 1.5]; remaining bins cover (1.5, right_edge]
        rest_bins = max(args.bins - 1, 1)
        rest_edges = np.linspace(1.5, right_edge, num=rest_bins + 1).tolist()
        custom_bins = [-0.5, 1.5] + rest_edges
    else:
        custom_bins = args.bins

    # --- Plot: publication-friendly styling ---
    plt.rcParams.update({
        "font.size": 11,       # base text size
        "axes.titlesize": 12,  # title
        "axes.labelsize": 11,  # axis labels
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })
    fig = plt.figure(figsize=(args.figwidth, args.figheight), dpi=args.dpi)
    ax = plt.gca()

    ax.hist(df["annotations"], bins=custom_bins)
    ax.axvline(mean, linestyle="--", label=f"Mean = {mean:.2f}")
    ax.axvline(mean - std, linestyle=":", label=f"±1 SD = {std:.2f}")
    ax.axvline(mean + std, linestyle=":")

    ax.set_xlabel("Annotations per Image")
    ax.set_ylabel("Number of Images")
    ax.set_title("Annotations per Image")
    if args.xmax is not None:
        ax.set_xlim(0, args.xmax)

    # Integer-aware ticks
    if args.xtickstep and args.xtickstep > 0:
        xmax = ax.get_xlim()[1]
        ax.set_xticks(np.arange(0, int(xmax) + 1, args.xtickstep))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Clean aesthetics
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    ax.legend()
    fig.tight_layout()

    # Save PNG
    fig.savefig(args.save_hist, dpi=args.dpi)
    print(f"Saved histogram to: {args.save_hist}")

    # Optional vector PDF
    if args.pdf:
        pdf_path = args.save_hist.with_suffix(".pdf")
        fig.savefig(pdf_path)
        print(f"Saved vector PDF to: {pdf_path}")

    # LaTeX table (booktabs-style). Add \usepackage{booktabs} in your preamble.
    latex = f"""
\\begin{{table}}[h!]
\\centering
\\caption{{Summary of Annotation Counts Per Image for {dataset_dir.name}}}
\\label{{tab:annotation-summary}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{6pt}}
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Number of Images & {n} \\\\
Mean Annotations/Image & {mean:.2f} \\\\
Standard Deviation & {std:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""".strip()

    args.save_tex.write_text(latex, encoding="utf-8")
    print(f"LaTeX table saved to: {args.save_tex}")

if __name__ == "__main__":
    main()
