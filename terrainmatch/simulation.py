"""Dependency-free synthetic terrain and observation simulation."""

from __future__ import annotations

import math
import random

Terrain = list[list[float]]


def generate_synthetic_terrain(size: int = 128, seed: int = 0) -> Terrain:
    """Generate a repeatable square height map from smooth hills and noise.

    Values are arbitrary elevation-like units. The seeded generator makes
    examples and tests reproducible; this is synthetic data, not real terrain.
    """
    if size < 8:
        raise ValueError("size must be at least 8")

    rng = random.Random(seed)
    hills = [
        (rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8),
         rng.uniform(0.12, 0.42), rng.uniform(0.3, 1.0))
        for _ in range(max(5, size // 12))
    ]
    terrain: Terrain = []
    for row in range(size):
        y = 2.0 * row / (size - 1) - 1.0
        line: list[float] = []
        for col in range(size):
            x = 2.0 * col / (size - 1) - 1.0
            value = 0.25 * math.sin(5.1 * x + 1.7 * y)
            value += 0.20 * math.cos(3.3 * y - 1.2 * x)
            for hx, hy, width, amplitude in hills:
                distance_sq = (x - hx) ** 2 + (y - hy) ** 2
                value += amplitude * math.exp(-distance_sq / (2 * width**2))
            value += rng.gauss(0.0, 0.015)
            line.append(value)
        terrain.append(line)
    return terrain


def shift_terrain(terrain: list[list[float]], row_offset: int,
                  col_offset: int, fill_value: float = 0.0) -> Terrain:
    """Translate terrain without wrapping, filling newly exposed cells.

    A positive ``row_offset`` moves cells down (toward larger row indices);
    a positive ``col_offset`` moves cells right (toward larger column indices).
    Thus a source cell at ``(r, c)`` appears at
    ``(r + row_offset, c + col_offset)`` in the returned observation.
    """
    height, width = _shape(terrain)
    shifted = [[fill_value for _ in range(width)] for _ in range(height)]
    for row in range(height):
        target_row = row + row_offset
        if not 0 <= target_row < height:
            continue
        for col in range(width):
            target_col = col + col_offset
            if 0 <= target_col < width:
                shifted[target_row][target_col] = float(terrain[row][col])
    return shifted


def _shape(terrain: list[list[float]]) -> tuple[int, int]:
    if not terrain or not terrain[0]:
        raise ValueError("terrain must be a non-empty rectangular grid")
    width = len(terrain[0])
    if any(len(row) != width for row in terrain):
        raise ValueError("terrain must be a non-empty rectangular grid")
    return len(terrain), width
