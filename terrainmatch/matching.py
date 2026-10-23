"""Exhaustive normalized cross-correlation for small terrain grids."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .simulation import _shape


@dataclass(frozen=True)
class MatchResult:
    """Best translation of the reference terrain into the observation grid.

    ``row_offset`` and ``col_offset`` follow :func:`shift_terrain`: positive
    row moves down and positive column moves right. ``score`` is Pearson's
    normalized cross-correlation over the overlapping cells, in [-1, 1].
    """

    row_offset: int
    col_offset: int
    score: float


def normalized_cross_correlation(
    reference: list[list[float]],
    observation: list[list[float]],
    max_offset: int | tuple[int, int],
    min_overlap: float = 0.5,
) -> MatchResult:
    """Find the integer translation maximizing overlap-wise Pearson NCC.

    Both grids must have identical dimensions. For each candidate ``(dr, dc)``,
    the matcher compares ``reference[r][c]`` with
    ``observation[r + dr][c + dc]`` wherever both coordinates overlap.
    Candidates are visited from most negative to positive offsets; exact score
    ties therefore resolve to the first candidate in row-major offset order.

    ``max_offset`` is a symmetric search bound, in pixels, as either one int
    for both axes or ``(max_row, max_col)``. ``min_overlap`` is the minimum
    fraction of the full grid required for a candidate to be considered.
    """
    height, width = _shape(reference)
    if _shape(observation) != (height, width):
        raise ValueError("reference and observation must have the same shape")
    if isinstance(max_offset, int):
        max_row = max_col = max_offset
    else:
        max_row, max_col = max_offset
    if max_row < 0 or max_col < 0:
        raise ValueError("max_offset values must be non-negative")
    if not 0.0 < min_overlap <= 1.0:
        raise ValueError("min_overlap must be in (0, 1]")

    required = math.ceil(height * width * min_overlap)
    best: MatchResult | None = None
    for dr in range(-max_row, max_row + 1):
        for dc in range(-max_col, max_col + 1):
            ref_r0, ref_r1 = max(0, -dr), min(height, height - dr)
            ref_c0, ref_c1 = max(0, -dc), min(width, width - dc)
            count = (ref_r1 - ref_r0) * (ref_c1 - ref_c0)
            if count < required:
                continue
            score = _overlap_ncc(reference, observation, dr, dc,
                                 ref_r0, ref_r1, ref_c0, ref_c1, count)
            if best is None or score > best.score:
                best = MatchResult(dr, dc, score)
    if best is None:
        raise ValueError("no candidate offset meets min_overlap")
    return best


def _overlap_ncc(reference: list[list[float]], observation: list[list[float]],
                 dr: int, dc: int, r0: int, r1: int, c0: int, c1: int,
                 count: int) -> float:
    ref_sum = obs_sum = 0.0
    for r in range(r0, r1):
        for c in range(c0, c1):
            ref_sum += float(reference[r][c])
            obs_sum += float(observation[r + dr][c + dc])
    ref_mean, obs_mean = ref_sum / count, obs_sum / count
    covariance = ref_var = obs_var = 0.0
    for r in range(r0, r1):
        for c in range(c0, c1):
            a = float(reference[r][c]) - ref_mean
            b = float(observation[r + dr][c + dc]) - obs_mean
            covariance += a * b
            ref_var += a * a
            obs_var += b * b
    denominator = math.sqrt(ref_var * obs_var)
    return covariance / denominator if denominator else 0.0
