"""Core terrain generation and matching primitives for TerrainMatch."""

from .matching import MatchResult, normalized_cross_correlation
from .simulation import generate_synthetic_terrain, shift_terrain

__all__ = [
    "MatchResult",
    "generate_synthetic_terrain",
    "normalized_cross_correlation",
    "shift_terrain",
]
