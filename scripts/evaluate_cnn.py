"""Evaluate the trained CNN matcher and compare against the NCC baseline.

Usage
-----
From the repository root with the active virtual environment::

    python scripts/evaluate_cnn.py [--checkpoint PATH] [--no-plots]

Outputs
-------
* ``outputs/results/evaluation_summary.json``
* ``outputs/results/cnn_results.csv``
* ``outputs/results/ncc_results.csv``
* ``outputs/figures/cnn_vs_ncc_comparison.png``
* ``outputs/figures/training_curve.png``  (if checkpoint available)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from terrainmatch.evaluation import (
    DEFAULT_CHECKPOINT,
    FIGURES_DIR,
    RESULTS_DIR,
    evaluate_cnn,
    evaluate_ncc,
    plot_comparison,
    plot_training_curve,
    save_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CNN terrain matcher vs. NCC baseline."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix appended to all output filenames before the extension "
             "(e.g. '_exp1' writes evaluation_summary_exp1.json). "
             "Default: '' (no suffix, baseline behaviour).",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(
            f"ERROR: Checkpoint not found at {args.checkpoint}\n"
            "Train the model first:\n"
            "  python -m terrainmatch.train --epochs 30\n"
            "Or run a smoke test:\n"
            "  python -m terrainmatch.train --smoke-test"
        )
        sys.exit(1)

    # CNN evaluation
    cnn_metrics = evaluate_cnn(checkpoint_path=args.checkpoint)

    # NCC evaluation
    ncc_metrics = evaluate_ncc()

    # Save results
    save_results(cnn_metrics, ncc_metrics, RESULTS_DIR, suffix=args.output_suffix)

    # Plots
    if not args.no_plots:
        fig_path = plot_comparison(
            cnn_metrics, ncc_metrics, FIGURES_DIR, suffix=args.output_suffix
        )
        print(f"Comparison plot: {fig_path}")
        curve_path = plot_training_curve(
            args.checkpoint, FIGURES_DIR, suffix=args.output_suffix
        )
        if curve_path:
            print(f"Training curve : {curve_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"CNN Top-1 accuracy  : {cnn_metrics['top1_accuracy']*100:.1f}%")
    print(f"CNN Top-3 accuracy  : {cnn_metrics['top3_accuracy']*100:.1f}%")
    print(f"NCC exact-match     : {ncc_metrics['exact_match_rate']*100:.1f}%")
    print(
        f"\nScope: {cnn_metrics['n_gallery_patches']} DEM patches across "
        f"{cnn_metrics['n_regions']} regions; synthetic benchmark only. "
        "See outputs/results/evaluation_summary.json for details."
    )


if __name__ == "__main__":
    main()
