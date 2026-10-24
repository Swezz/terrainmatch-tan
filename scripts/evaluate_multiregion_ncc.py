
from pathlib import Path
import json
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from terrainmatch.matching import normalized_cross_correlation

DATA_DIR = PROJECT_ROOT / "outputs" / "terrain_dataset"
MANIFEST_PATH = DATA_DIR / "manifest.json"

PATCH_SIZE = 200
OBSERVATION_SIZE = 80
SEARCH_MARGIN = 15
RANDOM_SEED = 42


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    results = []
    offsets = [
        (-10, -7),
        (-6, 8),
        (5, -9),
        (9, 6),
        (0, 11),
    ]

    for item in manifest:
        terrain = np.load(DATA_DIR / item["file"]).astype(np.float64)

        if terrain.shape != (PATCH_SIZE, PATCH_SIZE):
            continue

        if not np.isfinite(terrain).all():
            continue

        # Pick a fixed-size reference grid.
        reference_top = 50
        reference_left = 50
        reference = terrain[
            reference_top:reference_top + OBSERVATION_SIZE,
            reference_left:reference_left + OBSERVATION_SIZE,
        ]

        for dy, dx in offsets:
            observation_top = reference_top + dy
            observation_left = reference_left + dx

            observation = terrain[
                observation_top:observation_top + OBSERVATION_SIZE,
                observation_left:observation_left + OBSERVATION_SIZE,
            ]

            if observation.shape != reference.shape:
                continue

            # The grids have equal dimensions, as required by the matcher.
            # Test a controlled translation of the observation relative
            # to the reference grid.
            shifted = np.roll(observation, shift=(-dy, -dx), axis=(0, 1))

            result = normalized_cross_correlation(
                reference.tolist(),
                shifted.tolist(),
                max_offset=SEARCH_MARGIN,
                min_overlap=0.5,
            )

            results.append({
                "region": item["region"],
                "patch": item["file"],
                "intended_dy": dy,
                "intended_dx": dx,
                "predicted_dy": result.row_offset,
                "predicted_dx": result.col_offset,
                "score": result.score,
            })

    if not results:
        raise RuntimeError("No matching trials were completed.")

    errors = [
        np.hypot(
            r["predicted_dy"] - r["intended_dy"],
            r["predicted_dx"] - r["intended_dx"],
        )
        for r in results
    ]

    correct = [
        error == 0 for error in errors
    ]

    print(f"Total trials: {len(results)}")
    print(f"Exact matches: {sum(correct)}/{len(results)}")
    print(f"Exact match rate: {100 * np.mean(correct):.1f}%")
    print(f"Mean offset error: {np.mean(errors):.3f} pixels")
    print(f"Maximum offset error: {np.max(errors):.3f} pixels")

    for region in sorted({r["region"] for r in results}):
        indices = [
            i for i, result in enumerate(results)
            if result["region"] == region
        ]
        region_correct = sum(correct[i] for i in indices)
        print(
            f"{region}: {region_correct}/{len(indices)} "
            "exact matches"
        )


if __name__ == "__main__":
    main()
