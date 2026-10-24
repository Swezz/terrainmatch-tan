"""Download a bounded global DEM from OpenTopography.

Example (PowerShell):
    $env:OPENTOPOGRAPHY_API_KEY = "your-private-key"
    .\\.venv\\Scripts\\python.exe scripts\\fetch_opentopography_dem.py `
        --name western-india --south 18 --north 20 --west 72 --east 75

The API key is read from the environment and is never written to disk.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://portal.opentopography.org/API/globaldem"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_DEM_TYPE = "SRTMGL3"
SUPPORTED_DEM_TYPES = (
    "SRTMGL3",
    "SRTMGL1",
    "NASADEM",
    "AW3D30",
    "COP30",
    "COP90",
)
MAX_AREA_KM2 = 450_000


def validate_bbox(south: float, north: float, west: float, east: float) -> None:
    """Validate a WGS84 bounding box and conservative API area limit."""
    values = (south, north, west, east)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Bounding-box coordinates must be finite numbers.")
    if not (-90 <= south < north <= 90):
        raise ValueError("Latitude bounds must satisfy -90 <= south < north <= 90.")
    if not (-180 <= west < east <= 180):
        raise ValueError("Longitude bounds must satisfy -180 <= west < east <= 180.")

    mean_latitude = math.radians((south + north) / 2)
    area_km2 = (
        (north - south)
        * 111.32
        * (east - west)
        * 111.32
        * math.cos(mean_latitude)
    )
    if area_km2 > MAX_AREA_KM2:
        raise ValueError(
            f"Requested area is approximately {area_km2:,.0f} km²; "
            f"limit this request to at most {MAX_AREA_KM2:,} km²."
        )


def safe_region_name(name: str) -> str:
    """Return a normalized name safe for use in a local file name."""
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_-")
    if not normalized:
        raise ValueError("Region name must contain at least one letter or number.")
    return normalized.lower()


def build_request_url(
    *,
    dem_type: str,
    south: float,
    north: float,
    west: float,
    east: float,
    api_key: str,
) -> str:
    """Build the documented OpenTopography Global Datasets API request."""
    if dem_type not in SUPPORTED_DEM_TYPES:
        raise ValueError(
            f"Unsupported DEM type {dem_type!r}. Choose from: "
            + ", ".join(SUPPORTED_DEM_TYPES)
        )
    validate_bbox(south, north, west, east)
    if not api_key.strip():
        raise ValueError("OpenTopography API key must not be empty.")

    query = urlencode({
        "demtype": dem_type,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    })
    return f"{API_URL}?{query}"


def load_region_config(config_path: Path) -> list[dict]:
    """Read and validate a JSON list of named DEM bounding boxes."""
    try:
        regions = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read region config {config_path}: {exc}") from exc
    if not isinstance(regions, list) or not regions:
        raise ValueError("Region config must be a non-empty JSON list.")

    normalized_regions = []
    seen_names = set()
    required_fields = {"name", "south", "north", "west", "east"}
    allowed_fields = required_fields | {"dem_type"}
    for index, region in enumerate(regions, start=1):
        if not isinstance(region, dict):
            raise ValueError(f"Region entry {index} must be a JSON object.")
        missing_fields = required_fields - region.keys()
        unknown_fields = region.keys() - allowed_fields
        if missing_fields:
            raise ValueError(
                f"Region entry {index} is missing: {', '.join(sorted(missing_fields))}."
            )
        if unknown_fields:
            raise ValueError(
                f"Region entry {index} has unknown fields: "
                f"{', '.join(sorted(unknown_fields))}."
            )

        try:
            normalized = {
                "name": safe_region_name(str(region["name"])),
                "south": float(region["south"]),
                "north": float(region["north"]),
                "west": float(region["west"]),
                "east": float(region["east"]),
                "dem_type": region.get("dem_type", DEFAULT_DEM_TYPE),
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid values in region entry {index}: {exc}") from exc
        if normalized["name"] in seen_names:
            raise ValueError(
                f"Duplicate output name {normalized['name']!r} in region config."
            )
        if normalized["dem_type"] not in SUPPORTED_DEM_TYPES:
            raise ValueError(
                f"Unsupported DEM type in region {normalized['name']!r}: "
                f"{normalized['dem_type']!r}."
            )
        validate_bbox(
            normalized["south"],
            normalized["north"],
            normalized["west"],
            normalized["east"],
        )
        seen_names.add(normalized["name"])
        normalized_regions.append(normalized)
    return normalized_regions


def download_dem(url: str, output_path: Path) -> Path:
    """Download and validate a GeoTIFF before atomically replacing output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        url,
        headers={"User-Agent": "TerrainMatch-TAN research prototype"},
    )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.stem}_",
            suffix=".download",
            dir=output_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            try:
                with urlopen(request, timeout=180) as response:
                    while chunk := response.read(1024 * 1024):
                        temp_file.write(chunk)
            except HTTPError as exc:
                raise RuntimeError(
                    f"OpenTopography returned HTTP {exc.code}: {exc.reason}. "
                    "Check the API key, selected dataset, and bounding box."
                ) from None
            except URLError as exc:
                raise RuntimeError(
                    f"Could not reach OpenTopography: {exc.reason}"
                ) from None

        try:
            with rasterio.open(temp_path) as raster:
                if raster.count < 1 or raster.width < 1 or raster.height < 1:
                    raise ValueError("The downloaded raster contains no elevation band.")
                if raster.crs is None:
                    raise ValueError("The downloaded raster has no coordinate reference system.")
        except (rasterio.errors.RasterioError, ValueError) as exc:
            raise RuntimeError(
                "OpenTopography did not return a readable GeoTIFF. "
                "The response may contain an API error message."
            ) from exc

        os.replace(temp_path, output_path)
        temp_path = None
        return output_path
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download an SRTM or other global DEM bounding box from "
            "OpenTopography for terrain matching."
        )
    )
    parser.add_argument("--name", help="Short region label for the output file.")
    parser.add_argument("--south", type=float, help="Southern latitude in WGS84.")
    parser.add_argument("--north", type=float, help="Northern latitude in WGS84.")
    parser.add_argument("--west", type=float, help="Western longitude in WGS84.")
    parser.add_argument("--east", type=float, help="Eastern longitude in WGS84.")
    parser.add_argument(
        "--regions-file",
        type=Path,
        help="JSON list of region names and WGS84 bounding boxes to download in one run.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based first region to process from --regions-file (default: 1).",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Maximum number of regions to process from --regions-file.",
    )
    parser.add_argument(
        "--dem-type",
        choices=SUPPORTED_DEM_TYPES,
        default=DEFAULT_DEM_TYPE,
        help="Global DEM dataset (default: SRTMGL3, approx. 90 m).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store the downloaded GeoTIFF.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENTOPOGRAPHY_API_KEY", "").strip()
    if not api_key:
        parser.error(
            "Set OPENTOPOGRAPHY_API_KEY in your local environment first; "
            "never commit or share the key."
        )

    individual_values = (args.name, args.south, args.north, args.west, args.east)
    if args.regions_file:
        if any(value is not None for value in individual_values):
            parser.error("--regions-file cannot be combined with --name or bounds.")
        if args.start_index < 1:
            parser.error("--start-index must be at least 1.")
        if args.count is not None and args.count < 1:
            parser.error("--count must be at least 1.")
        try:
            regions = load_region_config(args.regions_file)
        except ValueError as exc:
            parser.error(str(exc))
        configured_count = len(regions)
        regions = regions[args.start_index - 1:]
        if args.count is not None:
            regions = regions[:args.count]
        if not regions:
            parser.error(
                f"--start-index {args.start_index} is beyond the "
                f"{configured_count} configured regions."
            )
        print(
            f"Processing regions {args.start_index}–"
            f"{args.start_index + len(regions) - 1} of {configured_count}."
        )
    else:
        if args.start_index != 1 or args.count is not None:
            parser.error("--start-index and --count require --regions-file.")
        if any(value is None for value in individual_values):
            parser.error(
                "Provide --name and all four bounds, or use --regions-file."
            )
        try:
            region = {
                "name": safe_region_name(args.name),
                "south": args.south,
                "north": args.north,
                "west": args.west,
                "east": args.east,
                "dem_type": args.dem_type,
            }
            validate_bbox(
                region["south"], region["north"], region["west"], region["east"]
            )
            regions = [region]
        except ValueError as exc:
            parser.error(str(exc))

    failures = []
    for region in regions:
        dem_type = region["dem_type"]
        output_path = args.output_dir / f"{region['name']}_{dem_type.lower()}.tif"
        if output_path.is_file():
            try:
                with rasterio.open(output_path) as raster:
                    if raster.count > 0 and raster.crs is not None:
                        print(f"\nAlready downloaded; keeping {output_path}")
                        continue
            except rasterio.errors.RasterioError:
                pass

        print(f"\nRequesting {dem_type} DEM for {region['name']}...")
        try:
            url = build_request_url(
                dem_type=dem_type,
                south=region["south"],
                north=region["north"],
                west=region["west"],
                east=region["east"],
                api_key=api_key,
            )
            path = download_dem(url, output_path)
            with rasterio.open(path) as raster:
                print(
                    f"Saved {path} ({raster.width} x {raster.height} pixels, "
                    f"{raster.crs}, "
                    f"{raster.res[0]:.6f} x {raster.res[1]:.6f} map units/pixel)"
                )
        except (RuntimeError, ValueError, OSError) as exc:
            failures.append((region["name"], str(exc)))
            print(f"Failed {region['name']}: {exc}")

    if failures:
        failed_names = ", ".join(name for name, _ in failures)
        parser.exit(
            1,
            f"\nDownloaded {len(regions) - len(failures)} of {len(regions)} "
            f"regions. Failed regions: {failed_names}\n"
            "Fix the reported issue and rerun the command; successful files "
            "remain in the output directory.\n",
        )
    print("\nNext: run scripts/build_terrain_dataset.py to extract patches from all DEMs.")


if __name__ == "__main__":
    main()
