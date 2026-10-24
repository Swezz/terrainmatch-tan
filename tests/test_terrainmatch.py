"""Unit tests for the TerrainMatch-TAN package.

Tests cover:
* NCC matching (existing)
* Simulation module (restored)
* CNN model forward pass and contrastive loss
* Dataset pair generation
* Inference pipeline
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from terrainmatch import (
    generate_synthetic_terrain,
    normalized_cross_correlation,
    normalize_patch,
    shift_terrain,
    simulate_observation,
)
from terrainmatch.model import SiameseTerrainNet, contrastive_loss
from terrainmatch.dataset import (
    load_manifest,
    train_test_split,
    generate_pairs,
    TerrainPairDataset,
    DATASET_DIR,
)
from terrainmatch.inference import build_gallery, match_patch
from terrainmatch.evaluation import _rank_patch_candidates


class TestEvaluationRanking(unittest.TestCase):
    def test_ranks_parent_patches_by_best_window(self):
        gallery_items = [
            {"file": "a.npy"},
            {"file": "a.npy"},
            {"file": "b.npy"},
            {"file": "b.npy"},
            {"file": "c.npy"},
        ]

        patch_files, ranking, best_windows = _rank_patch_candidates(
            np.array([0.1, 0.7, 0.4, 0.5, 0.6], dtype=np.float32),
            gallery_items,
        )

        self.assertEqual(patch_files, ["a.npy", "b.npy", "c.npy"])
        self.assertEqual(ranking.tolist(), [0, 2, 1])
        self.assertEqual(best_windows.tolist(), [1, 3, 4])


class TestSimulation(unittest.TestCase):
    def test_generate_synthetic_terrain_shape(self):
        t = generate_synthetic_terrain(32, seed=1)
        self.assertEqual(len(t), 32)
        self.assertEqual(len(t[0]), 32)

    def test_generate_synthetic_terrain_reproducible(self):
        a = generate_synthetic_terrain(16, seed=5)
        b = generate_synthetic_terrain(16, seed=5)
        self.assertEqual(a, b)

    def test_shift_terrain_positive(self):
        source = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        shifted = shift_terrain(source, row_offset=1, col_offset=1)
        self.assertEqual(shifted, [[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 4.0, 5.0]])

    def test_normalize_patch_range(self):
        arr = np.array([[0.0, 50.0], [100.0, 200.0]], dtype=np.float32)
        out = normalize_patch(arr)
        self.assertAlmostEqual(float(out.min()), 0.0, places=5)
        self.assertAlmostEqual(float(out.max()), 1.0, places=2)

    def test_simulate_observation_shape(self):
        patch = np.random.rand(200, 200).astype(np.float32)
        obs, r, c = simulate_observation(patch, crop_size=64)
        self.assertEqual(obs.shape, (64, 64))
        self.assertGreaterEqual(r, 0)
        self.assertGreaterEqual(c, 0)

    def test_simulate_observation_with_noise(self):
        patch = np.ones((100, 100), dtype=np.float32) * 500.0
        obs, _, _ = simulate_observation(patch, noise_std=5.0, seed=42, crop_size=40)
        self.assertFalse(np.all(obs == 500.0))


class TestMatching(unittest.TestCase):
    def test_positive_offsets_move_down_and_right(self):
        source = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        shifted = shift_terrain(source, row_offset=1, col_offset=1)
        self.assertEqual(shifted, [[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 4.0, 5.0]])

    def test_match_recovers_known_positive_offset(self):
        reference = generate_synthetic_terrain(size=48, seed=17)
        observation = shift_terrain(reference, row_offset=4, col_offset=7)
        result = normalized_cross_correlation(reference, observation, max_offset=10)
        self.assertEqual((result.row_offset, result.col_offset), (4, 7))
        self.assertAlmostEqual(result.score, 1.0, places=10)

    def test_match_recovers_known_negative_offset(self):
        reference = generate_synthetic_terrain(size=48, seed=23)
        observation = shift_terrain(reference, row_offset=-3, col_offset=-5)
        result = normalized_cross_correlation(reference, observation, max_offset=8)
        self.assertEqual((result.row_offset, result.col_offset), (-3, -5))
        self.assertAlmostEqual(result.score, 1.0, places=10)

    def test_no_shift_is_recovered(self):
        reference = generate_synthetic_terrain(size=32, seed=1)
        result = normalized_cross_correlation(reference, reference, max_offset=4)
        self.assertEqual((result.row_offset, result.col_offset), (0, 0))
        self.assertAlmostEqual(result.score, 1.0, places=10)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = SiameseTerrainNet(embedding_dim=64)
        self.model.eval()

    def test_forward_output_shape(self):
        a = torch.randn(4, 1, 64, 64)
        b = torch.randn(4, 1, 64, 64)
        ea, eb = self.model(a, b)
        self.assertEqual(ea.shape, (4, 64))
        self.assertEqual(eb.shape, (4, 64))

    def test_embeddings_are_unit_normalised(self):
        a = torch.randn(3, 1, 64, 64)
        b = torch.randn(3, 1, 64, 64)
        ea, eb = self.model(a, b)
        norms_a = ea.norm(dim=1)
        norms_b = eb.norm(dim=1)
        for n in norms_a:
            self.assertAlmostEqual(float(n.detach()), 1.0, places=5)
        for n in norms_b:
            self.assertAlmostEqual(float(n.detach()), 1.0, places=5)

    def test_contrastive_loss_positive_is_zero_for_identical(self):
        ea = torch.tensor([[1.0, 0.0]])
        eb = torch.tensor([[1.0, 0.0]])
        labels = torch.tensor([1.0])
        loss = contrastive_loss(ea, eb, labels, margin=1.0)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_contrastive_loss_positive_pair_penalises_distance(self):
        ea = torch.tensor([[1.0, 0.0]])
        eb = torch.tensor([[0.0, 1.0]])
        labels = torch.tensor([1.0])
        loss_pos = contrastive_loss(ea, eb, labels, margin=1.0)
        self.assertGreater(float(loss_pos), 0.0)

    def test_encode_same_as_forward(self):
        x = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            ea, _ = self.model(x, x)
            enc = self.model.encode(x)
        self.assertTrue(torch.allclose(ea, enc, atol=1e-6))


class TestDataset(unittest.TestCase):
    def test_manifest_loads(self):
        manifest = load_manifest()
        self.assertGreater(len(manifest), 0)

    def test_train_test_split_sizes(self):
        manifest = load_manifest()
        train, test = train_test_split(manifest, test_fraction=0.2)
        self.assertEqual(len(train) + len(test), len(manifest))
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)

    def test_pair_generation(self):
        manifest = load_manifest()
        train, _ = train_test_split(manifest, test_fraction=0.2)
        anchors, samples, labels = generate_pairs(train, pairs_per_patch=2)
        # Each patch generates 2 positive + 2 negative = 4 pairs
        self.assertEqual(len(anchors), len(train) * 4)
        self.assertIn(0, labels)
        self.assertIn(1, labels)

    def test_dataset_getitem(self):
        manifest = load_manifest()
        train, _ = train_test_split(manifest)
        anchors, samples, labels = generate_pairs(train, pairs_per_patch=2)
        ds = TerrainPairDataset(anchors, samples, labels)
        a, b, y = ds[0]
        self.assertEqual(a.shape, (1, 64, 64))
        self.assertEqual(b.shape, (1, 64, 64))
        self.assertIn(float(y), [0.0, 1.0])


class TestSlidingWindowInference(unittest.TestCase):
    def test_gallery_scans_windows_and_returns_location(self):
        manifest = load_manifest()[:1]
        model = SiameseTerrainNet(embedding_dim=16).eval()
        embeddings, items = build_gallery(
            manifest, model, DATASET_DIR, crop_size=64, stride=64
        )
        self.assertEqual(len(items), 16)  # 0, 64, 128, and final edge at 136
        self.assertEqual(embeddings.shape, (16, 16))
        self.assertEqual(items[-1]["window_row"], 136)
        self.assertEqual(items[-1]["window_col"], 136)

        patch = np.load(DATASET_DIR / manifest[0]["file"])
        query = patch[64:128, 64:128]
        ranked = match_patch(query, embeddings, items, model, top_k=2)
        self.assertEqual(ranked[0].patch_file, manifest[0]["file"])
        self.assertEqual((ranked[0].window_row, ranked[0].window_col), (64, 64))
        self.assertIsNotNone(ranked[0].score_margin)

    def test_match_reports_map_coordinate_from_dem_transform(self):
        model = SiameseTerrainNet(embedding_dim=16).eval()
        item = {
            "file": "sample_patch.npy",
            "region": "test-region",
            "row_start": 100,
            "col_start": 200,
            "transform": [
                0.001, 0.0, 72.0,
                0.0, -0.001, 20.0,
                0.0, 0.0, 1.0,
            ],
            "window_row": 32,
            "window_col": 16,
        }

        ranked = match_patch(
            np.ones((64, 64), dtype=np.float32),
            torch.zeros((1, 16)),
            [item],
            model,
            top_k=1,
        )

        self.assertAlmostEqual(ranked[0].latitude, 19.936)
        self.assertAlmostEqual(ranked[0].longitude, 72.048)


if __name__ == "__main__":
    unittest.main()
