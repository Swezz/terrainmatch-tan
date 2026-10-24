
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

dem_path = Path("data/srtm_darjeeling.tif")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

if not dem_path.exists():
    raise FileNotFoundError(
        f"DEM not found: {dem_path.resolve()}"
    )

with rasterio.open(dem_path) as src:
    elevation = src.read(1, masked=True).astype("float32")

    print("Dimensions:", src.width, "x", src.height)
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)
    print("Valid pixels:", elevation.count())

    if elevation.count() == 0:
        raise ValueError("The DEM has no valid elevation pixels.")

    valid = elevation.compressed()
    low, high = np.percentile(valid, [2, 98])

    print("Elevation range:", float(valid.min()), "to", float(valid.max()))
    print("Display range (2nd to 98th percentile):", low, high)

    if low == high:
        raise ValueError("The elevation values have no variation.")

    plt.figure(figsize=(10, 8))

    image = plt.imshow(
        elevation,
        cmap="terrain",
        vmin=low,
        vmax=high,
    )

    plt.colorbar(image, label="Elevation (meters)")
    plt.title("SRTM GL1 DEM — Darjeeling Region")
    plt.xlabel("Raster column")
    plt.ylabel("Raster row")
    plt.tight_layout()

    output_path = output_dir / "dem_preview.png"
    plt.savefig(output_path, dpi=180)

    print("Saved visualization to:", output_path.resolve())

    plt.show()
