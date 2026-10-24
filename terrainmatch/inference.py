"""CNN inference pipeline for terrain-patch matching.

This module loads a trained :class:`~terrainmatch.model.SiameseTerrainNet`
checkpoint and uses it to rank a gallery of candidate reference patches
against a query observation.

Pipeline
--------
1. Load model weights from a checkpoint file.
2. Pre-compute embeddings for all candidate reference patches.
3. Given a query patch, compute its embedding.
4. Rank candidates by cosine similarity (= dot product for unit vectors).
5. Return a ranked list of candidates with similarity scores.

Note
----
This module performs patch-level retrieval only.  It does not produce
an operational navigation fix. If spatial metadata is available in the
manifest, the module reports the matching DEM-window center in map coordinates
along with its patch-relative pixel location.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

from .dataset import (
    CROP_SIZE,
    DATASET_DIR,
    MANIFEST_PATH,
    load_manifest,
    load_patch,
    normalize_patch,
)
from .model import SiameseTerrainNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "terrain_cnn.pt"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """A single candidate match result."""

    rank: int
    patch_file: str
    region: str
    similarity: float
    row_start: int | None
    col_start: int | None
    window_row: int = 0
    window_col: int = 0
    score_margin: float | None = None
    latitude: float | None = None
    longitude: float | None = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    embedding_dim: int = 128,
    device: str | torch.device = "cpu",
) -> SiameseTerrainNet:
    """Load a trained :class:`SiameseTerrainNet` from a checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pt`` file written by :func:`terrainmatch.train.train`.
    embedding_dim:
        Must match the value used during training (read from the checkpoint
        config when available).
    device:
        Inference device.

    Returns
    -------
    SiameseTerrainNet
        Model in evaluation mode.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the checkpoint using PyTorch's restricted "
            "weights-only loader. Use a checkpoint produced by "
            "terrainmatch.train.train and verify that the file is intact."
        ) from exc

    if not isinstance(ckpt, Mapping):
        raise ValueError("Checkpoint must contain a mapping of model data.")
    state_dict = ckpt.get("model_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Checkpoint is missing a non-empty model_state_dict.")
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError("Checkpoint model_state_dict must contain tensors only.")

    cfg = ckpt.get("config", {})
    if not isinstance(cfg, Mapping):
        raise ValueError("Checkpoint config must be a mapping when provided.")
    dim = cfg.get("embedding_dim", embedding_dim)
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValueError("Checkpoint embedding_dim must be a positive integer.")
    model = SiameseTerrainNet(embedding_dim=dim)
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise ValueError(
            "Checkpoint weights do not match the SiameseTerrainNet model."
        ) from exc
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Patch preprocessing
# ---------------------------------------------------------------------------

def preprocess_patch(
    patch: np.ndarray,
    crop_size: int = CROP_SIZE,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Normalise, centre-crop, and convert a patch to a model-ready tensor.

    Parameters
    ----------
    patch:
        2-D numpy array (H × W) of elevation values.
    crop_size:
        Side length of the square centre crop.
    device:
        Target device.

    Returns
    -------
    torch.Tensor
        Shape ``(1, 1, crop_size, crop_size)``.
    """
    h, w = patch.shape
    if h >= crop_size and w >= crop_size:
        r = (h - crop_size) // 2
        c = (w - crop_size) // 2
        patch = patch[r : r + crop_size, c : c + crop_size]
    patch = normalize_patch(patch)
    t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return t.to(device)


# ---------------------------------------------------------------------------
# Gallery embedding
# ---------------------------------------------------------------------------

def build_gallery(
    manifest_items: List[dict],
    model: SiameseTerrainNet,
    dataset_dir: Path = DATASET_DIR,
    crop_size: int = CROP_SIZE,
    device: str | torch.device = "cpu",
    stride: int | None = None,
    batch_size: int = 128,
) -> tuple[torch.Tensor, List[dict]]:
    """Compute embeddings for centered patches or sliding DEM windows.

    Parameters
    ----------
    manifest_items:
        List of manifest dicts.
    model:
        Loaded model in eval mode.
    dataset_dir:
        Directory containing the ``.npy`` patch files.
    crop_size:
        Side of each square candidate window.
    device:
        Inference device.
    stride:
        If set, scan each DEM patch with this pixel stride, also including
        the final valid row and column. ``None`` preserves center-crop mode.
    batch_size:
        Number of windows encoded at once.

    Returns
    -------
    embeddings:
        Float32 tensor of shape ``(N, embedding_dim)`` — L2-normalised.
    items:
        The same list passed in (kept for correspondence with embeddings).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    windows: list[np.ndarray] = []
    gallery_items: list[dict] = []
    for item in manifest_items:
        patch = load_patch(item, dataset_dir)
        h, w = patch.shape
        if stride is None or h < crop_size or w < crop_size:
            windows.append(patch)
            gallery_items.append({**item, "window_row": 0, "window_col": 0})
            continue
        if stride < 1:
            raise ValueError("stride must be positive")
        row_positions = list(range(0, h - crop_size + 1, stride))
        col_positions = list(range(0, w - crop_size + 1, stride))
        if row_positions[-1] != h - crop_size:
            row_positions.append(h - crop_size)
        if col_positions[-1] != w - crop_size:
            col_positions.append(w - crop_size)
        for row in row_positions:
            for col in col_positions:
                windows.append(patch[row:row + crop_size, col:col + crop_size])
                gallery_items.append({**item, "window_row": row, "window_col": col})

    with torch.no_grad():
        encoded_batches = []
        for start in range(0, len(windows), batch_size):
            batch = torch.cat(
                [preprocess_patch(p, crop_size, device) for p in windows[start:start + batch_size]],
                dim=0,
            )
            encoded_batches.append(model.encode(batch))

    return torch.cat(encoded_batches, dim=0), gallery_items


# ---------------------------------------------------------------------------
# Inference: ranked matching
# ---------------------------------------------------------------------------

def match_patch(
    query_patch: np.ndarray,
    gallery_embeddings: torch.Tensor,
    gallery_items: List[dict],
    model: SiameseTerrainNet,
    crop_size: int = CROP_SIZE,
    device: str | torch.device = "cpu",
    top_k: int = 5,
) -> List[MatchResult]:
    """Rank gallery patches against a query and return the top-k matches.

    Parameters
    ----------
    query_patch:
        2-D numpy array of the observation (elevation values).
    gallery_embeddings:
        Pre-computed gallery embeddings, shape ``(N, D)``.
    gallery_items:
        Manifest dicts corresponding to each row of ``gallery_embeddings``.
    model:
        Loaded model in eval mode.
    crop_size:
        Preprocessing crop size (must match the gallery).
    device:
        Inference device.
    top_k:
        Number of top results to return.

    Returns
    -------
    List[MatchResult]
        Ranked list, best match first.
    """
    with torch.no_grad():
        query_t = preprocess_patch(query_patch, crop_size, device)
        query_emb = model.encode(query_t).squeeze(0)  # (D,)

    # Cosine similarity — equivalent to dot product for unit vectors
    sims = torch.mv(gallery_embeddings, query_emb)  # (N,)
    ranked_idx = sims.argsort(descending=True).tolist()

    results: List[MatchResult] = []
    for rank, idx in enumerate(ranked_idx[:top_k], start=1):
        item = gallery_items[idx]
        next_score = float(sims[ranked_idx[rank]]) if rank < len(ranked_idx) else None
        window_row = item.get("window_row", 0)
        window_col = item.get("window_col", 0)
        latitude, longitude = _window_center_coordinates(
            item, window_row, window_col, crop_size
        )
        results.append(
            MatchResult(
                rank=rank,
                patch_file=item["file"],
                region=item["region"],
                similarity=float(sims[idx]),
                row_start=item.get("row_start"),
                col_start=item.get("col_start"),
                window_row=window_row,
                window_col=window_col,
                score_margin=float(sims[idx]) - next_score if next_score is not None else None,
                latitude=latitude,
                longitude=longitude,
            )
        )
    return results


def _window_center_coordinates(
    item: dict,
    window_row: int,
    window_col: int,
    crop_size: int,
) -> tuple[float | None, float | None]:
    """Convert a candidate window center using its source-patch affine transform."""
    transform = item.get("transform")
    if transform is None or len(transform) < 6:
        return None, None

    a, b, c, d, e, f = (float(value) for value in transform[:6])
    pixel_col = window_col + crop_size / 2
    pixel_row = window_row + crop_size / 2
    longitude = a * pixel_col + b * pixel_row + c
    latitude = d * pixel_col + e * pixel_row + f
    return latitude, longitude


# ---------------------------------------------------------------------------
# Convenience: end-to-end single query
# ---------------------------------------------------------------------------

def run_inference(
    query_patch: np.ndarray,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    manifest_path: Path = MANIFEST_PATH,
    dataset_dir: Path = DATASET_DIR,
    top_k: int = 5,
    search_stride: int | None = 16,
) -> List[MatchResult]:
    """One-shot inference: load model, build gallery, return ranked matches.

    Parameters
    ----------
    query_patch:
        2-D numpy array for the observation patch.
    checkpoint_path:
        Path to the trained model checkpoint.
    manifest_path:
        Path to the manifest JSON.
    dataset_dir:
        Directory containing the ``.npy`` patch files.
    top_k:
        Number of ranked results to return.

    Returns
    -------
    List[MatchResult]
        Best matches, ranked by cosine similarity.
    """
    model = load_model(checkpoint_path)
    manifest = load_manifest(manifest_path)
    gallery_embs, gallery_items = build_gallery(
        manifest, model, dataset_dir, stride=search_stride
    )
    return match_patch(query_patch, gallery_embs, gallery_items, model, top_k=top_k)
