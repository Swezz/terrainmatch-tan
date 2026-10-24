"""Extract a reproducible, spatially separated patch dataset from all DEMs.

Any GeoTIFF placed in ``data/`` is discovered automatically. Patches are
selected from a shuffled non-overlapping raster grid, then recorded in one
manifest for training and evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEM_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "terrain_dataset"
DEFAULT_PATCH_SIZE = 200
DEFAULT_PATCHES_PER_DEM = 100
RANDOM_SEED = 42


def extract_dataset(
    dem_dir: Path = DEFAULT_DEM_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    patch_size: int = DEFAULT_PATCH_SIZE,
    patches_per_dem: int = DEFAULT_PATCHES_PER_DEM,
    stride: int | None = None,
    seed: int = RANDOM_SEED,
    minimum_std_m: float = 20.0,
) -> list[dict]:
    """Create patches and a manifest from every GeoTIFF in ``dem_dir``."""
    if patch_size < 1:
        raise ValueError("patch_size must be positive.")
    if patches_per_dem < 1:
        raise ValueError("patches_per_dem must be positive.")
    if minimum_std_m < 0:
        raise ValueError("minimum_std_m must be non-negative.")
    if stride is None:
        stride = patch_size
    if stride < 1:
        raise ValueError("stride must be positive.")
    if stride < patch_size:
        print(
            "WARNING: stride is smaller than patch size; extracted patches "
            "will overlap and are not independent samples."
        )

    dem_paths = sorted(
        [*dem_dir.glob("*.tif"), *dem_dir.glob("*.tiff")],
        key=lambda path: path.name.lower(),
    )
    if not dem_paths:
        raise FileNotFoundError(f"No .tif or .tiff DEM files found in {dem_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest: list[dict] = []

    for dem_path in dem_paths:
        region = dem_path.stem.lower().replace("-", "_").replace(" ", "_")
        print(f"\nProcessing {region}: {dem_path.name}")
        with rasterio.open(dem_path) as source:
            height, width = source.height, source.width
            if height < patch_size or width < patch_size:
                print("Skipped: DEM is smaller than the patch size.")
                continue

            candidates = [
                (row, col)
                for row in range(0, height - patch_size + 1, stride)
                for col in range(0, width - patch_size + 1, stride)
            ]
            rng.shuffle(candidates)
            accepted = 0

            for row, col in candidates:
                if accepted >= patches_per_dem:
                    break

                window = source.read(
                    1,
                    window=Window(col, row, patch_size, patch_size),
                    masked=True,
                    out_dtype="float32",
                )
                valid = window.compressed()
                if valid.size != patch_size * patch_size:
                    continue
                if not np.isfinite(valid).all() or float(valid.std()) < minimum_std_m:
                    continue

                file_name = f"{region}_patch_{accepted + 1:04d}.npy"
                np.save(output_dir / file_name, window.filled(np.nan).astype(np.float32))
                transform = source.window_transform(
                    Window(col, row, patch_size, patch_size)
                )
                manifest.append({
                    "region": region,
                    "file": file_name,
                    "source_dem": dem_path.name,
                    "row_start": row,
                    "col_start": col,
                    "patch_size": patch_size,
                    "elevation_min_m": float(valid.min()),
                    "elevation_max_m": float(valid.max()),
                    "elevation_mean_m": float(valid.mean()),
                    "elevation_std_m": float(valid.std()),
                    "crs": str(source.crs),
                    "transform": list(transform),
                })
                accepted += 1

            print(
                f"Accepted patches: {accepted}/{min(patches_per_dem, len(candidates))} "
                f"(non-overlapping grid stride: {stride}px)"
            )

    if not manifest:
        raise RuntimeError(
            "No valid patches were created. Check raster coverage, patch size, "
            "and the minimum elevation-variation filter."
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nTotal patches created: {len(manifest)} from {len(dem_paths)} DEM(s)")
    print(f"Manifest: {manifest_path}")
    print("Existing unused .npy files are left untouched; only manifest entries are active.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract non-overlapping terrain patches from all GeoTIFFs in a directory."
    )
    parser.add_argument("--dem-dir", type=Path, default=DEFAULT_DEM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--patches-per-dem", type=int, default=DEFAULT_PATCHES_PER_DEM)
    parser.add_argument(
        "--stride",
        type=int,
        help="Grid stride in pixels (default: patch size, so patches do not overlap).",
    )
    parser.add_argument("--minimum-std-m", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    extract_dataset(
        dem_dir=args.dem_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        patches_per_dem=args.patches_per_dem,
        stride=args.stride,
        seed=args.seed,
        minimum_std_m=args.minimum_std_m,
    )


if __name__ == "__main__":
    main()
