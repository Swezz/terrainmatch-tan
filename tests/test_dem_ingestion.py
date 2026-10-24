"""Tests for OpenTopography download request construction and DEM extraction."""

import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts.build_terrain_dataset import extract_dataset
from scripts.fetch_opentopography_dem import (
    build_request_url,
    load_region_config,
    safe_region_name,
    validate_bbox,
)


class TestOpenTopographyRequests(unittest.TestCase):
    def test_example_config_has_twenty_distinct_regions(self):
        config_path = Path(__file__).resolve().parents[1] / "data" / "regions.example.json"
        regions = load_region_config(config_path)

        self.assertEqual(len(regions), 20)
        self.assertEqual(len({region["name"] for region in regions}), 20)
        names = {region["name"] for region in regions}
        self.assertIn("sikkim-gangtok", names)
        self.assertIn("western-ghats-kerala", names)
        self.assertIn("eastern-ghats-araku", names)
        self.assertIn("kolkata-ganges-delta", names)
        self.assertIn("sahara-morocco", names)

    def test_global_example_config_has_fifty_supported_unique_regions(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "global_regions.example.json"
        )
        regions = load_region_config(config_path)

        self.assertEqual(len(regions), 50)
        self.assertEqual(len({region["name"] for region in regions}), 50)
        self.assertTrue(all(region["dem_type"] == "SRTMGL3" for region in regions))
        self.assertTrue(all(region["north"] <= 60 for region in regions))
        self.assertTrue(all(region["south"] >= -56 for region in regions))
        names = {region["name"] for region in regions}
        self.assertIn("australia-blue-mountains", names)
        self.assertIn("spain-pyrenees", names)
        self.assertIn("portugal-serra-da-estrela", names)
        self.assertIn("brazil-amazon", names)
        self.assertIn("germany-bavarian-alps", names)
        self.assertIn("argentina-mendoza-andes", names)
        self.assertIn("canada-banff", names)
        self.assertIn("south-africa-drakensberg", names)
        self.assertIn("chile-atacama-andes", names)

    def test_request_uses_wgs84_bounds_and_keeps_key_in_query(self):
        url = build_request_url(
            dem_type="SRTMGL3",
            south=18,
            north=19,
            west=72,
            east=73,
            api_key="test-secret",
        )

        query = parse_qs(urlparse(url).query)
        self.assertEqual(query["demtype"], ["SRTMGL3"])
        self.assertEqual(query["outputFormat"], ["GTiff"])
        self.assertEqual(query["API_Key"], ["test-secret"])
        self.assertEqual(query["north"], ["19"])

    def test_rejects_invalid_bounding_boxes_and_dem_types(self):
        with self.assertRaises(ValueError):
            validate_bbox(south=20, north=19, west=72, east=73)
        with self.assertRaises(ValueError):
            build_request_url(
                dem_type="NOT_A_DEM",
                south=18,
                north=19,
                west=72,
                east=73,
                api_key="test-secret",
            )

    def test_region_name_is_safe_for_a_filename(self):
        self.assertEqual(safe_region_name("Western India / 01"), "western_india_01")
        with self.assertRaises(ValueError):
            safe_region_name(" .. / ")

    def test_region_config_loads_and_rejects_duplicate_output_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "regions.json"
            config_path.write_text(
                json.dumps([
                    {
                        "name": "western-ghats",
                        "south": 10,
                        "north": 11,
                        "west": 76,
                        "east": 77,
                    },
                ]),
                encoding="utf-8",
            )
            regions = load_region_config(config_path)
            self.assertEqual(regions[0]["name"], "western-ghats")
            self.assertEqual(regions[0]["dem_type"], "SRTMGL3")

            config_path.write_text(
                json.dumps([
                    {
                        "name": "same",
                        "south": 10,
                        "north": 11,
                        "west": 76,
                        "east": 77,
                    },
                    {
                        "name": "same",
                        "south": 12,
                        "north": 13,
                        "west": 76,
                        "east": 77,
                    },
                ]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate output name"):
                load_region_config(config_path)


class TestGeoTiffPatchExtraction(unittest.TestCase):
    def test_extracts_non_overlapping_patches_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dem_dir = root / "data"
            output_dir = root / "patches"
            dem_dir.mkdir()
            values = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
            dem_path = dem_dir / "western_india_srtm.tif"
            with rasterio.open(
                dem_path,
                "w",
                driver="GTiff",
                height=64,
                width=64,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(72.0, 20.0, 0.001, 0.001),
            ) as dataset:
                dataset.write(values, 1)

            manifest = extract_dataset(
                dem_dir=dem_dir,
                output_dir=output_dir,
                patch_size=16,
                patches_per_dem=8,
                seed=3,
                minimum_std_m=0,
            )

            self.assertEqual(len(manifest), 8)
            starts = {(item["row_start"], item["col_start"]) for item in manifest}
            self.assertEqual(len(starts), 8)
            for item in manifest:
                patch = np.load(output_dir / item["file"])
                self.assertEqual(patch.shape, (16, 16))
                self.assertEqual(item["region"], "western_india_srtm")
                self.assertEqual(item["crs"], "EPSG:4326")
            saved_manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest, manifest)


if __name__ == "__main__":
    unittest.main()
