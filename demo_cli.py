"""Command-line demonstration of the TerrainMatch-TAN pipeline.

Runs end-to-end without Streamlit: loads a DEM patch, simulates an
observation, runs CNN matching, runs NCC matching, prints results, and
saves a comparison figure.

Usage
-----
    python demo_cli.py [--patch NAME] [--noise SIGMA] [--row-disp N] [--col-disp N]

Example
-------
    python demo_cli.py --patch darjeeling_patch_01.npy --noise 0.05
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from terrainmatch.dataset import DATASET_DIR, MANIFEST_PATH, load_manifest, load_patch, normalize_patch
from terrainmatch.inference import DEFAULT_CHECKPOINT, run_inference
from terrainmatch.matching import normalized_cross_correlation
from terrainmatch.simulation import shift_terrain, simulate_observation


def main() -> None:
    parser = argparse.ArgumentParser(description="TerrainMatch-TAN CLI demo")
    parser.add_argument("--patch", default="darjeeling_patch_01.npy",
                        help="Query patch filename from outputs/terrain_dataset/")
    parser.add_argument("--noise", type=float, default=0.02,
                        help="Gaussian noise std (normalised units, default 0.02)")
    parser.add_argument("--row-disp", type=int, default=30,
                        help="Row displacement for observation crop (pixels, default 30)")
    parser.add_argument("--col-disp", type=int, default=30,
                        help="Column displacement for observation crop (pixels, default 30)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top CNN matches to display (default 5)")
    parser.add_argument("--no-figure", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    # Load manifest and query patch
    manifest = load_manifest(MANIFEST_PATH)
    items_by_file = {m["file"]: m for m in manifest}

    if args.patch not in items_by_file:
        print("ERROR: '{}' not found in manifest.".format(args.patch))
        print("Available patches:")
        for f in items_by_file:
            print("  " + f)
        sys.exit(1)

    query_item = items_by_file[args.patch]
    query_patch = load_patch(query_item, DATASET_DIR)
    print("\nQuery patch  : " + args.patch)
    print("Region       : " + query_item["region"])
    print("Patch size   : {}".format(query_patch.shape))
    elev_range = "{:.0f}-{:.0f} m".format(
        query_item["elevation_min_m"], query_item["elevation_max_m"]
    )
    print("Elevation    : " + elev_range)

    # Simulate observation
    print("\nSimulating observation (row_disp={}, col_disp={}, noise={:.3f})...".format(
        args.row_disp, args.col_disp, args.noise
    ))
    noise_abs = args.noise * (query_patch.max() - query_patch.min())
    obs, actual_row, actual_col = simulate_observation(
        query_patch,
        noise_std=noise_abs,
        row_displacement=args.row_disp,
        col_displacement=args.col_disp,
        crop_size=64,
        seed=42,
    )
    print("Observation shape: {}, actual offset: row={}, col={}".format(
        obs.shape, actual_row, actual_col
    ))

    # CNN matching
    if not DEFAULT_CHECKPOINT.exists():
        print("\nWARNING: Checkpoint not found at " + str(DEFAULT_CHECKPOINT))
        print("Train first: python -m terrainmatch.train --epochs 30")
        cnn_results = []
        top1_correct = False
    else:
        print("\nRunning CNN matching (checkpoint: {})...".format(DEFAULT_CHECKPOINT.name))
        t0 = time.perf_counter()
        cnn_results = run_inference(obs, checkpoint_path=DEFAULT_CHECKPOINT, top_k=args.top_k)
        cnn_ms = (time.perf_counter() - t0) * 1000

        print("\nCNN Top-{} matches (query time: {:.1f} ms):".format(args.top_k, cnn_ms))
        print("{:<6} {:<32} {:<14} {:>10} {:>8} {:>8}  Match?".format("Rank", "File", "Region", "Similarity", "Row", "Col"))
        print("-" * 95)
        for r in cnn_results:
            correct = "<<" if r.patch_file == args.patch else "  "
            print("{:<6} {:<32} {:<14} {:>10.4f} {:>8} {:>8}  {}".format(
                r.rank, r.patch_file, r.region, r.similarity,
                r.window_row, r.window_col, correct
            ))

        top1_correct = bool(cnn_results and cnn_results[0].patch_file == args.patch)
        print("\nTop-1 {} at local window row={}, col={}".format(
            "CORRECT" if top1_correct else "INCORRECT",
            cnn_results[0].window_row, cnn_results[0].window_col
        ))
        if cnn_results[0].score_margin is not None:
            print("Top-window score margin: {:.4f}".format(cnn_results[0].score_margin))

    # NCC matching: shift the query patch itself so both grids have the same size
    print("\nRunning NCC baseline (shift test on query patch)...")
    patch_ds = query_patch[::6, ::6]
    if patch_ds.shape[0] >= 10 and patch_ds.shape[1] >= 10:
        ref_list = patch_ds.tolist()
        obs_list = shift_terrain(ref_list, row_offset=3, col_offset=2, fill_value=0.0)
        t_ncc = time.perf_counter()
        ncc_res = normalized_cross_correlation(ref_list, obs_list, max_offset=5, min_overlap=0.6)
        ncc_ms = (time.perf_counter() - t_ncc) * 1000
        print("NCC score     : {:.4f}".format(ncc_res.score))
        print("NCC recovered : row={}, col={} (expected row=3, col=2)".format(
            ncc_res.row_offset, ncc_res.col_offset
        ))
        print("NCC time      : {:.1f} ms".format(ncc_ms))
    else:
        print("NCC skipped: downsampled patch too small.")

    # Figure
    if not args.no_figure:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        axes[0].imshow(normalize_patch(query_patch), cmap="terrain")
        axes[0].set_title("Reference patch\n" + args.patch)
        axes[0].axis("off")

        axes[1].imshow(normalize_patch(obs), cmap="terrain")
        axes[1].set_title(
            "Simulated observation\n(noise={:.3f}, row={}, col={})".format(
                args.noise, actual_row, actual_col
            )
        )
        axes[1].axis("off")

        if cnn_results:
            best_item = items_by_file.get(cnn_results[0].patch_file)
            if best_item:
                best_patch = load_patch(best_item, DATASET_DIR)
                axes[2].imshow(normalize_patch(best_patch), cmap="terrain")
            axes[2].set_title(
                "Best CNN match\n{}\nsim={:.4f}".format(
                    cnn_results[0].patch_file, cnn_results[0].similarity
                )
            )
        else:
            axes[2].text(0.5, 0.5, "No CNN results\n(train model first)",
                        ha="center", va="center", transform=axes[2].transAxes)
            axes[2].set_title("Best CNN match")
        axes[2].axis("off")

        fig.suptitle(
            "TerrainMatch-TAN CLI Demo  |  Experimental prototype",
            fontsize=11
        )
        plt.tight_layout()

        fig_dir = PROJECT_ROOT / "outputs" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = fig_dir / "demo_cli_result.png"
        fig.savefig(out_path, dpi=150)
        print("\nFigure saved: " + str(out_path))
        plt.close(fig)


if __name__ == "__main__":
    main()
