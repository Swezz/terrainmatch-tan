import unittest

from terrainmatch import (
    generate_synthetic_terrain,
    normalized_cross_correlation,
    sliding_window_ncc,
    shift_terrain,
)


class TerrainMatchingTests(unittest.TestCase):
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

    def test_sliding_window_ncc_recovers_observation_location(self):
        reference = generate_synthetic_terrain(size=40, seed=31)
        observation = [
            [reference[r][c] for c in range(9, 19)]
            for r in range(13, 23)
        ]

        result = sliding_window_ncc(reference, observation)

        self.assertEqual((result.row, result.col), (13, 9))
        self.assertAlmostEqual(result.score, 1.0, places=10)

    def test_sliding_window_ncc_rejects_observation_larger_than_reference(self):
        with self.assertRaises(ValueError):
            sliding_window_ncc([[1.0]], [[1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
