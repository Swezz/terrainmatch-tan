"""Evaluation module for CNN-based and NCC-based terrain matching.

Produces quantitative metrics and comparison plots for both matchers.

Metrics reported
----------------
* Top-1 and Top-3 patch-retrieval accuracy (CNN).
* Mean and median cosine similarity for correct vs. incorrect matches (CNN).
* NCC exact-match rate and mean offset error (classical baseline).
* Per-region breakdowns.
* Inference time per query.

Scope
-----
* Results use the DEM-derived patches in the active manifest, split 80/20
  by parent patch index.
* Patches from the same DEM may be spatially adjacent; they are NOT from
  independent geographic regions.  Results should be interpreted as a
  preliminary prototype benchmark only.
* CNN matching tests whether the model assigns higher similarity to a
  query patch than to other gallery patches.  "Correct" means the
  query's own parent patch ranks first.
* NCC evaluation uses controlled synthetic translations of the same patches.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import (
    CROP_SIZE,
    DATASET_DIR,
    MANIFEST_PATH,
    RANDOM_SEED,
    load_manifest,
    load_patch,
    normalize_patch,
    train_test_split,
)
from .inference import build_gallery, load_model, preprocess_patch
from .matching import normalized_cross_correlation
from .simulation import shift_terrain

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "terrain_cnn.pt"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

NCC_TEST_OFFSETS = [(2, -3), (-3, 1), (4, 2), (0, -4), (-2, 3)]
NCC_DOWNSAMPLE = 6
NCC_MAX_OFFSET = 4
CNN_SEARCH_STRIDE = 16
CNN_NOISE_LEVELS = (0.0, 0.01, 0.02, 0.05)


def _rank_patch_candidates(
    similarities: np.ndarray,
    gallery_items: list[dict],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Rank parent patches by their best window without building per-window results."""
    if len(similarities) != len(gallery_items):
        raise ValueError("Similarity scores and gallery items must have equal lengths")
    if not gallery_items:
        raise ValueError("The gallery must contain at least one candidate window")

    starts = [0]
    patch_files = [gallery_items[0]["file"]]
    for index in range(1, len(gallery_items)):
        if gallery_items[index]["file"] != gallery_items[index - 1]["file"]:
            starts.append(index)
            patch_files.append(gallery_items[index]["file"])

    ends = starts[1:] + [len(gallery_items)]
    patch_scores = np.maximum.reduceat(similarities, starts)
    best_window_indices = np.asarray([
        start + int(np.argmax(similarities[start:end]))
        for start, end in zip(starts, ends)
    ])
    ranked_patches = np.argsort(-patch_scores, kind="stable")
    return patch_files, ranked_patches, best_window_indices


# ---------------------------------------------------------------------------
# CNN evaluation
# ---------------------------------------------------------------------------

def evaluate_cnn(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    manifest_path: Path = MANIFEST_PATH,
    dataset_dir: Path = DATASET_DIR,
    seed: int = RANDOM_SEED,
    search_stride: int = CNN_SEARCH_STRIDE,
    noise_levels: tuple[float, ...] = CNN_NOISE_LEVELS,
) -> dict:
    """Evaluate displaced crop localization against sliding DEM windows.

    Each held-out patch supplies a known crop at a non-central offset. That
    crop is matched against sliding candidate windows across the full DEM
    gallery. Synthetic noise is added at several fractions of the patch's
    elevation range. The patch gallery is reference-map data; no query crop or
    its augmentation is used to train the encoder.

    Returns
    -------
    dict
        Keys: ``top1_accuracy``, ``top3_accuracy``, ``results``,
              ``mean_query_time_ms``.
    """
    model = load_model(checkpoint_path)
    manifest = load_manifest(manifest_path)
    _, test_manifest = train_test_split(manifest, test_fraction=0.2, seed=seed)

    # The stored DEMs are the reference map. Search windows, not patch centers.
    gallery_embs, gallery_items = build_gallery(
        manifest, model, dataset_dir, stride=search_stride
    )

    results = []
    query_times = []
    rng = np.random.default_rng(seed)
    for test_index, item in enumerate(test_manifest):
        query_patch = load_patch(item, dataset_dir)
        max_offset = query_patch.shape[0] - CROP_SIZE
        # Select reproducible, varied positions away from the center.
        row = int((17 + 31 * test_index) % (max_offset + 1))
        col = int((43 + 47 * test_index) % (max_offset + 1))
        clean_query = query_patch[row:row + CROP_SIZE, col:col + CROP_SIZE]
        for noise_fraction in noise_levels:
            noise_std = noise_fraction * float(np.ptp(query_patch))
            query = clean_query + rng.normal(0.0, noise_std, clean_query.shape).astype(np.float32)
            t0 = time.perf_counter()
            with torch.no_grad():
                query_tensor = preprocess_patch(query, crop_size=CROP_SIZE)
                query_embedding = model.encode(query_tensor).squeeze(0)
                similarities = torch.mv(gallery_embs, query_embedding).cpu().numpy()
            patch_files, ranked_patches, best_window_indices = _rank_patch_candidates(
                similarities, gallery_items
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            query_times.append(elapsed_ms)
            top1_patch_index = int(ranked_patches[0])
            top1_window = gallery_items[int(best_window_indices[top1_patch_index])]
            top1_file = patch_files[top1_patch_index]
            top1_similarity = float(similarities[int(best_window_indices[top1_patch_index])])
            top3_parent_files = {
                patch_files[int(index)] for index in ranked_patches[:3]
            }
            top1_correct = top1_file == item["file"]
            top3_correct = item["file"] in top3_parent_files
            position_error = float(np.hypot(
                top1_window.get("window_row", 0) - row,
                top1_window.get("window_col", 0) - col,
            )) if top1_correct else None
            top1_margin = (
                top1_similarity
                - float(similarities[int(best_window_indices[int(ranked_patches[1])])])
                if len(ranked_patches) > 1
                else None
            )
            results.append({
                "query_file": item["file"],
                "query_region": item["region"],
                "noise_fraction_of_range": noise_fraction,
                "true_window_row": row,
                "true_window_col": col,
                "top1_match": top1_file,
                "top1_window_row": top1_window.get("window_row", 0),
                "top1_window_col": top1_window.get("window_col", 0),
                "top1_correct": top1_correct,
                "top3_correct": top3_correct,
                "position_error_px": position_error,
                "top1_similarity": top1_similarity,
                "top1_margin": top1_margin,
                "query_time_ms": elapsed_ms,
            })

    top1_acc = np.mean([r["top1_correct"] for r in results])
    top3_acc = np.mean([r["top3_correct"] for r in results])
    per_noise = {
        str(level): {
            "top1_accuracy": float(np.mean([r["top1_correct"] for r in results if r["noise_fraction_of_range"] == level])),
            "top3_accuracy": float(np.mean([r["top3_correct"] for r in results if r["noise_fraction_of_range"] == level])),
            "mean_position_error_px": float(np.mean([r["position_error_px"] for r in results if r["noise_fraction_of_range"] == level and r["position_error_px"] is not None])) if any(r["noise_fraction_of_range"] == level and r["position_error_px"] is not None for r in results) else None,
        }
        for level in noise_levels
    }

    print("\n=== CNN Evaluation ===")
    print(f"Displaced noisy queries: {len(results)}")
    print(f"Top-1 accuracy         : {100*top1_acc:.1f}%")
    print(f"Top-3 accuracy         : {100*top3_acc:.1f}%")
    print(f"Mean query time        : {np.mean(query_times):.1f} ms")
    for level, metrics in per_noise.items():
        print(f"Noise {float(level):.0%} of range: top-1={metrics['top1_accuracy']:.1%}, top-3={metrics['top3_accuracy']:.1%}, mean position error={metrics['mean_position_error_px']}")
    print("\nReference gallery includes the held-out DEM windows; geography remains highly limited.")

    return {
        "top1_accuracy": float(top1_acc),
        "top3_accuracy": float(top3_acc),
        "results": results,
        "mean_query_time_ms": float(np.mean(query_times)),
        "n_test_patches": len(test_manifest),
        "n_gallery_patches": len(manifest),
        "n_regions": len({item["region"] for item in manifest}),
        "n_queries": len(results),
        "search_stride_px": search_stride,
        "noise_results": per_noise,
        "mean_position_error_px": float(np.mean([r["position_error_px"] for r in results if r["position_error_px"] is not None])) if any(r["position_error_px"] is not None for r in results) else None,
    }


# ---------------------------------------------------------------------------
# NCC evaluation
# ---------------------------------------------------------------------------

def evaluate_ncc(
    manifest_path: Path = MANIFEST_PATH,
    dataset_dir: Path = DATASET_DIR,
    downsample: int = NCC_DOWNSAMPLE,
    test_offsets: list = NCC_TEST_OFFSETS,
    max_offset: int = NCC_MAX_OFFSET,
) -> dict:
    """Evaluate the NCC baseline on synthetic translations.

    Returns
    -------
    dict
        Keys: ``exact_match_rate``, ``mean_error_pixels``,
              ``median_error_pixels``, ``results``.
    """
    manifest = load_manifest(manifest_path)
    results = []

    for item in manifest:
        patch = load_patch(item, dataset_dir).astype(np.float64)
        if not np.isfinite(patch).all():
            continue
        arr = patch[::downsample, ::downsample]
        reference = arr.tolist()

        for dr, dc in test_offsets:
            observation = shift_terrain(reference, dr, dc, fill_value=0.0)
            t0 = time.perf_counter()
            match = normalized_cross_correlation(
                reference, observation, max_offset=max_offset, min_overlap=0.7
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            row_err = match.row_offset - dr
            col_err = match.col_offset - dc
            error = float(np.hypot(row_err, col_err))

            results.append({
                "region": item["region"],
                "patch": item["file"],
                "true_row_offset": dr,
                "true_col_offset": dc,
                "pred_row_offset": match.row_offset,
                "pred_col_offset": match.col_offset,
                "ncc_score": match.score,
                "error_pixels": error,
                "exact_match": error == 0.0,
                "query_time_ms": elapsed_ms,
            })

    errors = np.array([r["error_pixels"] for r in results])
    exact = np.array([r["exact_match"] for r in results])

    print("\n=== NCC Baseline Evaluation ===")
    print(f"Total trials           : {len(results)}")
    print(f"Exact match rate       : {100*exact.mean():.1f}%")
    print(f"Mean offset error      : {errors.mean():.3f} pixels")
    print(f"Median offset error    : {np.median(errors):.3f} pixels")

    return {
        "exact_match_rate": float(exact.mean()),
        "mean_error_pixels": float(errors.mean()),
        "median_error_pixels": float(np.median(errors)),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    cnn_metrics: dict,
    ncc_metrics: dict,
    figures_dir: Path,
    suffix: str = "",
) -> Path:
    """Generate a comparison bar chart and return the saved figure path."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel 1: CNN retrieval accuracy
    ax = axes[0]
    bars = ax.bar(
        ["Top-1 Accuracy", "Top-3 Accuracy"],
        [cnn_metrics["top1_accuracy"] * 100, cnn_metrics["top3_accuracy"] * 100],
        color=["#4C72B0", "#55A868"],
        width=0.5,
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}%",
            ha="center", fontsize=11,
        )
    ax.set_ylim(0, 115)
    ax.set_title(
        "CNN Displaced-Window Retrieval\n"
        f"({cnn_metrics.get('n_gallery_patches', 'unknown')}-patch gallery)"
    )
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: NCC exact-match rate
    ax2 = axes[1]
    ax2.bar(
        ["Exact Match\nRate"],
        [ncc_metrics["exact_match_rate"] * 100],
        color=["#C44E52"],
        width=0.4,
    )
    ax2.set_ylim(0, 115)
    ax2.set_title("NCC Baseline Exact-Match Rate\n(synthetic translations)")
    ax2.set_ylabel("Rate (%)")
    ax2.text(
        0,
        ncc_metrics["exact_match_rate"] * 100 + 1,
        f"{ncc_metrics['exact_match_rate']*100:.1f}%",
        ha="center", fontsize=11,
    )
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "TerrainMatch-TAN: CNN vs. NCC Comparison\n"
        "(Synthetic benchmark; not real-world positioning validation)",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = figures_dir / f"cnn_vs_ncc_comparison{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_training_curve(
    checkpoint_path: Path,
    figures_dir: Path,
    suffix: str = "",
) -> Path | None:
    """Plot training and validation loss curves from a checkpoint."""
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict):
            return None
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        if isinstance(train_losses, (list, tuple)) and isinstance(val_losses, (list, tuple)):
            train_losses = list(train_losses)
            val_losses = list(val_losses)
        if (
            not isinstance(train_losses, (list, tuple))
            or not isinstance(val_losses, (list, tuple))
            or not train_losses
            or len(train_losses) != len(val_losses)
            or not all(isinstance(value, (int, float)) for value in train_losses + val_losses)
            or not np.isfinite(train_losses + val_losses).all()
        ):
            return None
    except Exception:
        return None

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, label="Train loss", marker="o", markersize=3)
    ax.plot(epochs, val_losses, label="Val loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive loss")
    ax.set_title("CNN Training History (contrastive loss)")
    ax.legend()
    ax.grid(alpha=0.3)
    out_path = figures_dir / f"training_curve{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    cnn_metrics: dict,
    ncc_metrics: dict,
    results_dir: Path,
    suffix: str = "",
) -> None:
    """Persist evaluation results to JSON and CSV."""
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "cnn": {
            "top1_accuracy": cnn_metrics["top1_accuracy"],
            "top3_accuracy": cnn_metrics["top3_accuracy"],
            "mean_query_time_ms": cnn_metrics["mean_query_time_ms"],
            "n_test_patches": cnn_metrics["n_test_patches"],
            "n_gallery_patches": cnn_metrics["n_gallery_patches"],
            "n_regions": cnn_metrics["n_regions"],
            "n_queries": cnn_metrics["n_queries"],
            "search_stride_px": cnn_metrics["search_stride_px"],
            "mean_position_error_px_when_patch_correct": cnn_metrics["mean_position_error_px"],
            "noise_results": cnn_metrics["noise_results"],
        },
        "ncc": {
            "exact_match_rate": ncc_metrics["exact_match_rate"],
            "mean_error_pixels": ncc_metrics["mean_error_pixels"],
            "median_error_pixels": ncc_metrics["median_error_pixels"],
        },
        "scope": (
            f"Synthetic displaced-window benchmark on "
            f"{cnn_metrics['n_gallery_patches']} DEM-derived patches across "
            f"{cnn_metrics['n_regions']} regions. Queries use an 80/20 random "
            "patch-level split; the gallery includes the full reference map. "
            "Nearby patches may be geographically correlated. NCC uses "
            "controlled synthetic translations. These results are not "
            "validation on real sensor data or unseen geographic regions."
        ),
    }

    summary_path = results_dir / f"evaluation_summary{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cnn_csv = results_dir / f"cnn_results{suffix}.csv"
    if cnn_metrics["results"]:
        with cnn_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cnn_metrics["results"][0].keys()))
            w.writeheader()
            w.writerows(cnn_metrics["results"])

    ncc_csv = results_dir / f"ncc_results{suffix}.csv"
    if ncc_metrics["results"]:
        with ncc_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ncc_metrics["results"][0].keys()))
            w.writeheader()
            w.writerows(ncc_metrics["results"])

    print(f"\nResults saved to: {results_dir}")
