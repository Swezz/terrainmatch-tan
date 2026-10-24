
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

dem_path = Path("data/srtm_darjeeling.tif")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Size of the terrain patch in raster pixels
PATCH_SIZE = 200

with rasterio.open(dem_path) as src:
    # Read the elevation data while preserving NoData pixels
    window = Window(
        col_off=200,
        row_off=200,
        width=PATCH_SIZE,
        height=PATCH_SIZE,
    )

    patch = src.read(1, window=window, masked=True).astype("float32")

    if patch.count() == 0:
        raise ValueError("The extracted patch contains no valid elevations.")

    valid = patch.compressed()

    print("Patch dimensions:", patch.shape)
    print("Valid pixels:", patch.count(), "/", patch.size)
    print("Elevation range:", float(valid.min()), "to", float(valid.max()), "meters")
    print("Patch mean elevation:", float(valid.mean()), "meters")

    # Save the actual elevation values for reuse
    patch_path = output_dir / "darjeeling_patch.npy"
    np.save(patch_path, patch.filled(np.nan))

    # Create a readable visualization
    low, high = np.percentile(valid, [2, 98])

    plt.figure(figsize=(8, 7))
    image = plt.imshow(
        patch,
        cmap="terrain",
        vmin=low,
        vmax=high,
    )
    plt.colorbar(image, label="Elevation (meters)")
    plt.title("Darjeeling Terrain Patch — 200 × 200 pixels")
    plt.xlabel("Patch column")
    plt.ylabel("Patch row")
    plt.tight_layout()

    preview_path = output_dir / "darjeeling_patch.png"
    plt.savefig(preview_path, dpi=180)
    print("Saved elevation array:", patch_path.resolve())
    print("Saved patch preview:", preview_path.resolve())

    plt.show()
