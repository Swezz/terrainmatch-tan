
from pathlib import Path

import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEM_FILES = {
    "Darjeeling": PROJECT_ROOT / "data" / "srtm_darjeeling.tif",
    "Nepal": PROJECT_ROOT / "data" / "srtm_nepal.tif",
}


def inspect_dem(name, path):
    print(f"\n--- {name} ---")

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    with rasterio.open(path) as src:
        elevation = src.read(1, masked=True)
        valid = elevation.compressed()

        print(f"File: {path.name}")
        print(f"CRS: {src.crs}")
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"Resolution: {src.res}")
        print(f"Valid pixels: {valid.size:,}")
        print(f"NoData pixels: {elevation.size - valid.size:,}")

        if valid.size:
            print(f"Minimum elevation: {valid.min():.2f} m")
            print(f"Maximum elevation: {valid.max():.2f} m")
            print(f"Mean elevation: {valid.mean():.2f} m")
            print(
                "Elevation standard deviation: "
                f"{valid.std():.2f} m"
            )


def main():
    for name, path in DEM_FILES.items():
        inspect_dem(name, path)


if __name__ == "__main__":
    main()
