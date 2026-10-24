
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from terrainmatch.matching import normalized_cross_correlation


def shift_without_wrapping(array, row_shift, col_shift):
    """Shift a grid without wrapping values around its edges."""
    height, width = array.shape
    shifted = np.zeros_like(array)

    # Source region in the original grid.
    src_r0 = max(0, -row_shift)
    src_r1 = min(height, height - row_shift)
    src_c0 = max(0, -col_shift)
    src_c1 = min(width, width - col_shift)

    # Corresponding destination region.
    dst_r0 = max(0, row_shift)
    dst_r1 = min(height, height + row_shift)
    dst_c0 = max(0, col_shift)
    dst_c1 = min(width, width + col_shift)

    shifted[dst_r0:dst_r1, dst_c0:dst_c1] = array[
        src_r0:src_r1, src_c0:src_c1
    ]

    return shifted


patch_path = Path("outputs/darjeeling_patch.npy")
if not patch_path.exists():
    raise FileNotFoundError(f"Patch not found: {patch_path.resolve()}")

terrain = np.load(patch_path).astype(np.float64)

if not np.isfinite(terrain).all():
    raise ValueError("Terrain patch contains invalid values.")

test_offsets = [
    (0, 0),
    (5, 7),
    (-4, 6),
    (3, -5),
    (-6, -2),
]

print("\n--- NCC Multiple-Offset Evaluation ---")
print(f"{'Actual':>14} {'Estimated':>14} {'Score':>10} {'Result':>10}")

passed = 0

for true_row, true_col in test_offsets:
    observation = shift_without_wrapping(
        terrain, true_row, true_col
    )

    result = normalized_cross_correlation(
        terrain.tolist(),
        observation.tolist(),
        max_offset=10,
        min_overlap=0.5,
    )

    correct = (
        result.row_offset == true_row
        and result.col_offset == true_col
    )

    passed += int(correct)

    print(
        f"{str((true_row, true_col)):>14} "
        f"{str((result.row_offset, result.col_offset)):>14} "
        f"{result.score:>10.6f} "
        f"{'PASS' if correct else 'CHECK':>10}"
    )

print(f"\nSummary: {passed}/{len(test_offsets)} offsets recovered.")
