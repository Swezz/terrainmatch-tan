"""Dependency-free synthetic terrain and observation simulation."""

from __future__ import annotations

import math
import random
import numpy as np

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


def normalize_patch(patch: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert an elevation array to finite float32 values in [0, 1]."""
    values = np.asarray(patch, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("patch must contain at least one finite elevation")
    fill = float(values[finite].mean())
    values = np.where(finite, values, fill)
    low, high = float(values.min()), float(values.max())
    if high - low <= eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def simulate_observation(
    reference_patch: np.ndarray,
    *,
    noise_std: float = 0.0,
    row_displacement: int = 0,
    col_displacement: int = 0,
    crop_size: int | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Extract a displaced DEM window and add reproducible Gaussian noise.

    Noise is in the same elevation units as ``reference_patch``. This is a
    synthetic disturbance model and does not model a specific sensor.
    Displacements specify the crop's upper-left pixel in the parent array.
    """
    values = np.asarray(reference_patch, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("reference_patch must be a non-empty 2-D array")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if crop_size is None:
        crop_size = min(values.shape) // 2
    if crop_size < 1 or crop_size > min(values.shape):
        raise ValueError("crop_size must fit within the reference patch")
    row = int(np.clip(row_displacement, 0, values.shape[0] - crop_size))
    col = int(np.clip(col_displacement, 0, values.shape[1] - crop_size))
    observation = values[row:row + crop_size, col:col + crop_size].copy()
    if not np.isfinite(observation).all():
        valid = observation[np.isfinite(observation)]
        if not valid.size:
            raise ValueError("observation window has no finite elevation cells")
        observation = np.where(np.isfinite(observation), observation, float(valid.mean()))
    if noise_std:
        generator = rng if rng is not None else np.random.default_rng(seed)
        observation += generator.normal(0.0, noise_std, observation.shape).astype(np.float32)
    return observation, row, col
