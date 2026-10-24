"""Core terrain generation and matching primitives for TerrainMatch-TAN."""

from .matching import (
    MatchResult,
    WindowMatchResult,
    normalized_cross_correlation,
    sliding_window_ncc,
)
from .simulation import generate_synthetic_terrain, shift_terrain, normalize_patch, simulate_observation

__all__ = [
    "MatchResult",
    "WindowMatchResult",
    "generate_synthetic_terrain",
    "normalized_cross_correlation",
    "normalize_patch",
    "shift_terrain",
    "simulate_observation",
    "sliding_window_ncc",
]
