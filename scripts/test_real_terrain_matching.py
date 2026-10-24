
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Make the project root available for imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from terrainmatch.matching import normalized_cross_correlation

# Load the real Darjeeling elevation patch.
patch_path = Path("outputs/darjeeling_patch.npy")
output_dir = Path("outputs")

if not patch_path.exists():
    raise FileNotFoundError(f"Patch not found: {patch_path.resolve()}")

reference = np.load(patch_path).astype(np.float64)

if not np.isfinite(reference).all():
    raise ValueError("The terrain patch contains invalid values.")

# Define a known displacement.
true_row_offset = 5
true_col_offset = 7

height, width = reference.shape

# Create a shifted observation without wrapping around the edges.
# Zero is only used to fill the uncovered edges.
observation = np.zeros_like(reference)

observation[
    true_row_offset:, true_col_offset:
] = reference[
    :height - true_row_offset,
    :width - true_col_offset
]

# Run the existing NCC implementation.
result = normalized_cross_correlation(
    reference.tolist(),
    observation.tolist(),
    max_offset=10,
    min_overlap=0.5,
)

print("\n--- Real Terrain NCC Test ---")
print("Patch dimensions:", reference.shape)
print("Known row offset:", true_row_offset)
print("Known column offset:", true_col_offset)
print("Estimated row offset:", result.row_offset)
print("Estimated column offset:", result.col_offset)
print("NCC score:", round(result.score, 6))

if (
    result.row_offset == true_row_offset
    and result.col_offset == true_col_offset
):
    print("RESULT: PASS — known displacement recovered.")
else:
    print("RESULT: CHECK — estimated displacement differs.")

# Save a comparison visualization.
low, high = np.percentile(reference, [2, 98])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(reference, cmap="terrain", vmin=low, vmax=high)
axes[0].set_title("Reference terrain")

axes[1].imshow(observation, cmap="terrain", vmin=low, vmax=high)
axes[1].set_title("Shifted observation")

for ax in axes:
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

plt.tight_layout()

preview_path = output_dir / "ncc_test_patches.png"
plt.savefig(preview_path, dpi=180)
print("Saved preview:", preview_path.resolve())

plt.show()
