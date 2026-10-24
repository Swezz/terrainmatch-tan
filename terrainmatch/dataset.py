"""Dataset utilities for CNN-based terrain matching.

Loads DEM-derived patches from the manifest, generates augmented
training pairs (positive / negative), and exposes PyTorch Dataset
classes ready for a DataLoader.

Design notes
------------
* Positive pairs are formed from the **same parent patch** by applying
  independent random crops of size ``CROP_SIZE``.  The two crops share the
  same underlying terrain signal but may differ slightly due to offset.
* Negative pairs are formed from patches drawn from **different parent
  indices**.
* The manifest is split by *parent patch index*, not by augmented copies,
  so no augmented copy of a training patch appears in the test set.
* The current local manifest contains 887 patches across 52 regions. The
  random patch-level split is not geographically independent, so results
  should be interpreted as preliminary only.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .simulation import normalize_patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "terrain_dataset" / "manifest.json"
DATASET_DIR = PROJECT_ROOT / "outputs" / "terrain_dataset"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CROP_SIZE: int = 64          # CNN input resolution (pixels)
PAIRS_PER_PATCH: int = 8     # augmented pairs generated per parent patch
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path = MANIFEST_PATH) -> List[dict]:
    """Return the parsed manifest list."""
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_patch(item: dict, dataset_dir: Path = DATASET_DIR) -> np.ndarray:
    """Load and return a float32 numpy array for a manifest entry."""
    arr = np.load(dataset_dir / item["file"]).astype(np.float32)
    # Replace NaN with the patch mean (should not occur after builder filter)
    if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, np.nanmean(arr))
    return arr


def random_crop(patch: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    """Extract a random square crop of side ``size`` from a 2-D array."""
    h, w = patch.shape
    if h < size or w < size:
        raise ValueError(
            f"Patch ({h}×{w}) is smaller than crop size ({size}×{size})."
        )
    r = rng.randint(0, h - size)
    c = rng.randint(0, w - size)
    return patch[r : r + size, c : c + size]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def train_test_split(
    manifest: List[dict],
    test_fraction: float = 0.2,
    seed: int = RANDOM_SEED,
) -> Tuple[List[dict], List[dict]]:
    """Split the manifest by **parent patch index** into train / test subsets.

    Spatial proximity between adjacent patches in the same DEM means that
    the two subsets are **not** independently sampled from truly disjoint
    geographic regions.  The split is still useful for a held-out evaluation
    of the CNN's ability to distinguish its own training patches from
    held-out ones, but results do not generalise to unseen terrain areas.

    Parameters
    ----------
    manifest:
        Full list of patch metadata dicts.
    test_fraction:
        Fraction of parent patches to place in the test set.
    seed:
        Random seed for reproducibility.
    """
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    if not manifest:
        raise ValueError("manifest must not be empty")

    rng = random.Random(seed)
    indices = list(range(len(manifest)))
    rng.shuffle(indices)

    if len(indices) == 1:
        return [manifest[indices[0]]], []

    n_test = int(round(len(indices) * test_fraction))
    if n_test == 0:
        n_test = 1
    if n_test >= len(indices):
        n_test = len(indices) - 1

    test_idx = set(indices[:n_test])
    train = [m for i, m in enumerate(manifest) if i not in test_idx]
    test = [m for i, m in enumerate(manifest) if i in test_idx]
    return train, test


# ---------------------------------------------------------------------------
# Pair generator
# ---------------------------------------------------------------------------

def generate_pairs(
    manifest_items: List[dict],
    dataset_dir: Path = DATASET_DIR,
    crop_size: int = CROP_SIZE,
    pairs_per_patch: int = PAIRS_PER_PATCH,
    seed: int = RANDOM_SEED,
    noise_std: float = 0.02,
    shifted_positives: bool = False,
    max_shift: int = 8,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Generate (anchor, sample, label) triples for contrastive training.

    For each parent patch, ``pairs_per_patch`` positive pairs are created.
    An equal number of negative pairs is created by pairing each anchor with
    a crop from a randomly selected **different** parent patch.

    Parameters
    ----------
    manifest_items:
        Manifest entries for this split (train or test).
    dataset_dir:
        Directory containing the ``.npy`` files.
    crop_size:
        Side length of each square crop.
    pairs_per_patch:
        Number of positive pairs to generate per parent patch.
    seed:
        Random seed.
    noise_std:
        Standard deviation of Gaussian noise added after per-crop
        normalisation.  Units are the normalised [0, 1] range.
    shifted_positives:
        If ``True``, the positive sample is drawn from the **same parent
        patch** at a position shifted from the anchor by a random integer
        displacement in ``[-max_shift, max_shift]`` pixels in each
        direction, clamped to valid patch bounds.  This trains the encoder
        for the sub-stride translation invariance required at evaluation
        time (Experiment 1).  When ``False`` (default), two independent
        random crops are used — the original behaviour.
    max_shift:
        Maximum pixel displacement (row and column independently) for the
        positive crop when ``shifted_positives=True``.  Ignored when
        ``shifted_positives=False``.  Default 8 matches the worst-case
        query-to-gallery misalignment at stride=16 (average ≈ 3.8 px).

    Returns
    -------
    anchors, samples, labels:
        Parallel lists.  ``labels[i] == 1`` for a positive pair,
        ``labels[i] == 0`` for a negative pair.
        Each entry in ``anchors`` and ``samples`` is a float32 numpy array
        of shape ``(crop_size, crop_size)`` normalised to ``[0, 1]``.
    """
    if shifted_positives and not (0 < max_shift < crop_size):
        raise ValueError(
            f"max_shift must be positive and less than crop_size ({crop_size}); "
            f"got max_shift={max_shift}"
        )
    rng = random.Random(seed)
    noise_rng = np.random.default_rng(seed)
    patches = [load_patch(item, dataset_dir) for item in manifest_items]
    n = len(patches)

    anchors: List[np.ndarray] = []
    samples: List[np.ndarray] = []
    labels: List[int] = []

    for i, patch in enumerate(patches):
        h, w = patch.shape
        for _ in range(pairs_per_patch):
            # Anchor — random crop position recorded so it can be reused for
            # the shifted positive.
            r_a = rng.randint(0, h - crop_size)
            c_a = rng.randint(0, w - crop_size)
            anchor_crop = patch[r_a : r_a + crop_size, c_a : c_a + crop_size]
            a = _normalise_with_noise(anchor_crop, noise_rng, noise_std)

            # Positive pair — same parent patch.
            if shifted_positives:
                # Experiment 1: positive is the same parent patch cropped at a
                # position shifted from the anchor by ≤max_shift pixels in
                # each direction.  This guarantees ≥76 % pixel overlap at the
                # maximum shift, directly matching the evaluation scenario.
                delta_r = rng.randint(-max_shift, max_shift)
                delta_c = rng.randint(-max_shift, max_shift)
                r_p = max(0, min(h - crop_size, r_a + delta_r))
                c_p = max(0, min(w - crop_size, c_a + delta_c))
                pos_crop = patch[r_p : r_p + crop_size, c_p : c_p + crop_size]
            else:
                # Original behaviour: independent random crop from same parent.
                pos_crop = random_crop(patch, crop_size, rng)
            b = _normalise_with_noise(pos_crop, noise_rng, noise_std)
            anchors.append(a)
            samples.append(b)
            labels.append(1)

            # Negative pair — anchor from patch i, sample from a different patch.
            j = rng.choice([k for k in range(n) if k != i])
            neg = _normalise_with_noise(random_crop(patches[j], crop_size, rng), noise_rng, noise_std)
            anchors.append(a)
            samples.append(neg)
            labels.append(0)

    return anchors, samples, labels


def _normalise_with_noise(
    patch: np.ndarray, rng: np.random.Generator, noise_std: float
) -> np.ndarray:
    """Normalize a crop and add reproducible unit-range sensor noise."""
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    normalised = normalize_patch(patch)
    if noise_std:
        normalised = normalised + rng.normal(0.0, noise_std, normalised.shape).astype(np.float32)
    return normalised.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TerrainPairDataset(Dataset):
    """PyTorch Dataset of (anchor, sample, label) terrain-patch pairs.

    Parameters
    ----------
    anchors, samples, labels:
        Outputs of :func:`generate_pairs`.
    """

    def __init__(
        self,
        anchors: List[np.ndarray],
        samples: List[np.ndarray],
        labels: List[int],
    ) -> None:
        assert len(anchors) == len(samples) == len(labels)
        self.anchors = anchors
        self.samples = samples
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        a = torch.from_numpy(self.anchors[idx]).unsqueeze(0)   # (1, H, W)
        b = torch.from_numpy(self.samples[idx]).unsqueeze(0)   # (1, H, W)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return a, b, y


class TerrainPatchDataset(Dataset):
    """Dataset of individual normalised patches for embedding / retrieval.

    Parameters
    ----------
    manifest_items:
        Manifest entries.
    dataset_dir:
        Directory containing the ``.npy`` files.
    crop_size:
        If not ``None``, a centre crop of this size is extracted.
    """

    def __init__(
        self,
        manifest_items: List[dict],
        dataset_dir: Path = DATASET_DIR,
        crop_size: int | None = CROP_SIZE,
    ) -> None:
        self.items = manifest_items
        self.dataset_dir = dataset_dir
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        patch = load_patch(self.items[idx], self.dataset_dir)
        if self.crop_size is not None:
            h, w = patch.shape
            r = (h - self.crop_size) // 2
            c = (w - self.crop_size) // 2
            patch = patch[r : r + self.crop_size, c : c + self.crop_size]
        patch = normalize_patch(patch)
        return torch.from_numpy(patch).unsqueeze(0)   # (1, H, W)

    @property
    def labels(self) -> List[str]:
        return [item["file"] for item in self.items]

    @property
    def regions(self) -> List[str]:
        return [item["region"] for item in self.items]
