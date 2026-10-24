
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from terrainmatch.matching import normalized_cross_correlation


def shift_without_wrapping(array, row_shift, col_shift):
    height, width = array.shape
    shifted = np.zeros_like(array)

    src_r0 = max(0, -row_shift)
    src_r1 = min(height, height - row_shift)
    src_c0 = max(0, -col_shift)
    src_c1 = min(width, width - col_shift)

    dst_r0 = max(0, row_shift)
    dst_r1 = min(height, height + row_shift)
    dst_c0 = max(0, col_shift)
    dst_c1 = min(width, width + col_shift)

    shifted[dst_r0:dst_r1, dst_c0:dst_c1] = array[
        src_r0:src_r1, src_c0:src_c1
    ]

    return shifted


patch_path = Path("outputs/darjeeling_patch.npy")
terrain = np.load(patch_path).astype(np.float64)

if not np.isfinite(terrain).all():
    raise ValueError("Terrain patch contains invalid values.")

true_row, true_col = 5, 7
base_observation = shift_without_wrapping(terrain, true_row, true_col)

# Only add noise to the valid overlapping terrain region.
valid = np.zeros(terrain.shape, dtype=bool)
valid[true_row:, true_col:] = True

# Noise levels are standard deviations in meters.
noise_levels = [0, 10, 25, 50, 100]
trials_per_level = 5

rng = np.random.default_rng(42)

print("\n--- NCC Noise Robustness Experiment ---")
print("True offset:", (true_row, true_col))
print("Trials per noise level:", trials_per_level)
print("Noise is Gaussian with the stated standard deviation.")
print(f"{'Noise (m)':>12} {'Correct':>10} {'Mean error (pixels)':>22}")

for noise_std in noise_levels:
    correct_count = 0
    errors = []

    for _ in range(trials_per_level):
        observation = base_observation.copy()

        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=observation.shape,
        )
        observation[valid] += noise[valid]

        result = normalized_cross_correlation(
            terrain.tolist(),
            observation.tolist(),
            max_offset=10,
            min_overlap=0.5,
        )

        row_error = result.row_offset - true_row
        col_error = result.col_offset - true_col

        errors.append(np.hypot(row_error, col_error))

        if row_error == 0 and col_error == 0:
            correct_count += 1

    print(
        f"{noise_std:>12} "
        f"{correct_count}/{trials_per_level:>7} "
        f"{np.mean(errors):>22.3f}"
    )
