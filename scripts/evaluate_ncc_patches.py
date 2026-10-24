
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from terrainmatch.matching import normalized_cross_correlation


PATCH_PATH = PROJECT_ROOT / "outputs" / "darjeeling_patch.npy"

PATCH_SIZE = 100
MAX_OFFSET = 10
TRIALS_PER_PATCH = 10
NOISE_STD_METERS = 25
SEED = 42


def shift_without_wrapping(array, row_shift, col_shift):
    """Shift a 2D array without wrapping values around its edges."""
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


def main():
    if not PATCH_PATH.exists():
        raise FileNotFoundError(
            f"Patch not found: {PATCH_PATH}\n"
            "Run scripts/extract_terrain_patch.py first."
        )

    terrain = np.load(PATCH_PATH).astype(np.float64)

    if terrain.ndim != 2 or not np.isfinite(terrain).all():
        raise ValueError("The terrain patch must be a finite 2D array.")

    if min(terrain.shape) < PATCH_SIZE:
        raise ValueError(f"Patch is too small: {terrain.shape}")

    rng = np.random.default_rng(SEED)

    # Use several spatially different windows from the existing DEM patch.
    starts = [
        (0, 0),
        (0, 50),
        (50, 0),
        (50, 50),
        (25, 25),
        (0, 25),
        (25, 0),
        (25, 50),
        (50, 25),
    ]

    starts = [
        (r, c)
        for r, c in starts
        if r + PATCH_SIZE <= terrain.shape[0]
        and c + PATCH_SIZE <= terrain.shape[1]
    ]

    offsets = [
        (3, 5),
        (-4, 6),
        (6, -3),
        (-5, -4),
        (7, 2),
        (2, -7),
        (-6, 3),
        (4, 4),
        (-2, -5),
        (5, -6),
    ]

    total = 0
    correct = 0
    errors_pixels = []

    print("\n--- NCC Multi-Patch Evaluation ---")
    print("Source patch:", PATCH_PATH)
    print("Source dimensions:", terrain.shape)
    print("Evaluation window:", f"{PATCH_SIZE} x {PATCH_SIZE}")
    print("Noise standard deviation:", NOISE_STD_METERS, "meters")
    print("Trials per window:", TRIALS_PER_PATCH)
    print("\nWindow  True offset  Estimated offset  Score      Result")

    for index, (start_r, start_c) in enumerate(starts):
        r0 = int(rng.integers(0, terrain.shape[0] - PATCH_SIZE + 1))
        c0 = int(rng.integers(0, terrain.shape[1] - PATCH_SIZE + 1))

        # The fixed windows ensure repeatable coverage; the random window
        # adds variation when the source patch is larger than 100 x 100.
        if index < len(starts):
            r0, c0 = starts[index]

        reference = terrain[
            r0:r0 + PATCH_SIZE,
            c0:c0 + PATCH_SIZE,
        ].copy()

        for trial in range(TRIALS_PER_PATCH):
            true_row, true_col = offsets[trial]
            observation = shift_without_wrapping(
                reference, true_row, true_col
            )

            valid = np.zeros(reference.shape, dtype=bool)
            row_start = max(0, true_row)
            col_start = max(0, true_col)
            valid[
                row_start:,
                col_start:,
            ] = True

            noise = rng.normal(
                0.0, NOISE_STD_METERS, size=observation.shape
            )
            observation[valid] += noise[valid]

            result = normalized_cross_correlation(
                reference.tolist(),
                observation.tolist(),
                max_offset=MAX_OFFSET,
                min_overlap=0.5,
            )

            row_error = result.row_offset - true_row
            col_error = result.col_offset - true_col
            error = float(np.hypot(row_error, col_error))

            is_correct = row_error == 0 and col_error == 0
            correct += int(is_correct)
            total += 1
            errors_pixels.append(error)

            print(
                f"{index + 1:>6}  "
                f"({true_row:>2}, {true_col:>2})       "
                f"({result.row_offset:>2}, {result.col_offset:>2})       "
                f"{result.score:>6.3f}  "
                f"{'PASS' if is_correct else 'CHECK'}"
            )

    print("\n--- Summary ---")
    print(f"Correct matches: {correct}/{total}")
    print(f"Exact recovery rate: {100 * correct / total:.1f}%")
    print(f"Mean position error: {np.mean(errors_pixels):.3f} pixels")
    print(f"Maximum position error: {np.max(errors_pixels):.3f} pixels")
    print(
        "\nImportant: these windows all come from the same Darjeeling "
        "source patch. They are not independent geographic regions."
    )


if __name__ == "__main__":
    main()
