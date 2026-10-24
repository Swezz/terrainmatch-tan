"""Training loop for the Siamese CNN terrain-matching model.

Usage (command-line)
--------------------
Run from the repository root with the active virtual environment::

    python -m terrainmatch.train [OPTIONS]

Options
-------
--epochs N          Number of training epochs (default: 30).
--batch-size N      Mini-batch size (default: 32).
--lr F              Learning rate for Adam (default: 1e-3).
--embedding-dim N   Embedding vector length (default: 128).
--margin F          Contrastive loss margin (default: 1.0).
--smoke-test        Run a single epoch of 4 steps to verify the pipeline.
--checkpoint PATH   Path to save the model checkpoint (default: models/terrain_cnn.pt).

Notes
-----
* The active manifest determines the training set size. The current saved
  checkpoint was trained on 887 patches using a random patch-level split.
  Results are preliminary and are not geographically independent validation.
* If ``--smoke-test`` is set, training runs for exactly 1 epoch with
  up to 4 gradient steps regardless of the dataset size.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import (
    CROP_SIZE,
    DATASET_DIR,
    MANIFEST_PATH,
    PAIRS_PER_PATCH,
    RANDOM_SEED,
    TerrainPairDataset,
    generate_pairs,
    load_manifest,
    train_test_split,
)
from .model import SiameseTerrainNet, contrastive_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "terrain_cnn.pt"


def train(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    embedding_dim: int = 128,
    margin: float = 1.0,
    pair_noise_std: float = 0.02,
    smoke_test: bool = False,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    seed: int = RANDOM_SEED,
    shifted_positives: bool = False,
    max_shift: int = 8,
) -> dict:
    """Train the Siamese CNN and return a metrics dictionary.

    Parameters
    ----------
    epochs:
        Number of complete passes over the training pairs.
    batch_size:
        Mini-batch size.
    lr:
        Learning rate for the Adam optimiser.
    embedding_dim:
        Length of the embedding vector produced by the encoder.
    margin:
        Contrastive loss margin (in L2-distance units for unit-sphere
        embeddings, so values around 0.5–1.0 are reasonable).
    smoke_test:
        If ``True``, run only 1 epoch with at most 4 gradient steps and
        do not save a full checkpoint.  Use this to verify the pipeline
        quickly without a long wait.
    checkpoint_path:
        Where to save the final model checkpoint.
    seed:
        Global random seed for reproducibility.
    shifted_positives:
        If ``True``, positive training pairs are constructed using a
        shifted-crop strategy (Experiment 1): the positive sample is drawn
        from the same parent patch but displaced from the anchor by up to
        ``max_shift`` pixels.  When ``False`` (default) the original
        independent-random-crop behaviour is used.
    max_shift:
        Maximum pixel displacement for the positive crop when
        ``shifted_positives=True``.  Ignored when ``shifted_positives=False``.

    Returns
    -------
    dict
        Keys: ``train_losses`` (list of per-epoch mean losses),
              ``val_losses`` (list of per-epoch mean losses on validation pairs),
              ``config`` (training configuration dict),
              ``checkpoint_path`` (str or None).
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")   # CPU-only prototype

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    manifest = load_manifest(MANIFEST_PATH)
    train_manifest, val_manifest = train_test_split(manifest, test_fraction=0.2, seed=seed)

    print(f"Training patches: {len(train_manifest)}, Validation patches: {len(val_manifest)}")

    train_anchors, train_samples, train_labels = generate_pairs(
        train_manifest, DATASET_DIR, CROP_SIZE, PAIRS_PER_PATCH,
        seed=seed, noise_std=pair_noise_std,
        shifted_positives=shifted_positives, max_shift=max_shift,
    )
    val_anchors, val_samples, val_labels = generate_pairs(
        val_manifest, DATASET_DIR, CROP_SIZE, PAIRS_PER_PATCH // 2,
        seed=seed + 1, noise_std=pair_noise_std,
        shifted_positives=shifted_positives, max_shift=max_shift,
    )

    train_ds = TerrainPairDataset(train_anchors, train_samples, train_labels)
    val_ds = TerrainPairDataset(val_anchors, val_samples, val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    print(f"Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = SiameseTerrainNet(embedding_dim=embedding_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.5)

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "embedding_dim": embedding_dim,
        "margin": margin,
        "pair_noise_std": pair_noise_std,
        "shifted_positives": shifted_positives,
        "max_shift": max_shift if shifted_positives else None,
        "smoke_test": smoke_test,
        "seed": seed,
        "train_patches": len(train_manifest),
        "val_patches": len(val_manifest),
        "train_pairs": len(train_ds),
        "val_pairs": len(val_ds),
    }

    train_losses: list[float] = []
    val_losses: list[float] = []

    max_steps = 4 if smoke_test else None
    actual_epochs = 1 if smoke_test else epochs

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    t0 = time.time()
    for epoch in range(actual_epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = 0
        for a, b, y in train_loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            emb_a, emb_b = model(a, b)
            loss = contrastive_loss(emb_a, emb_b, y, margin=margin)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_steps += 1
            if max_steps and n_steps >= max_steps:
                break

        train_losses.append(epoch_loss / max(n_steps, 1))

        # Validation
        model.eval()
        v_loss = 0.0
        v_steps = 0
        with torch.no_grad():
            for a, b, y in val_loader:
                a, b, y = a.to(device), b.to(device), y.to(device)
                emb_a, emb_b = model(a, b)
                loss = contrastive_loss(emb_a, emb_b, y, margin=margin)
                v_loss += loss.item()
                v_steps += 1
        val_losses.append(v_loss / max(v_steps, 1))

        scheduler.step()

        if not smoke_test or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{actual_epochs} | "
                f"Train loss: {train_losses[-1]:.4f} | "
                f"Val loss:   {val_losses[-1]:.4f}"
            )

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Save checkpoint
    # -----------------------------------------------------------------------
    saved_path: str | None = None
    if not smoke_test:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "epoch": actual_epochs,
            },
            checkpoint_path,
        )
        config_path = checkpoint_path.with_suffix(".json")
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Checkpoint saved: {checkpoint_path}")
        saved_path = str(checkpoint_path)
    else:
        print("[smoke-test] Model not saved (smoke_test=True).")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "config": config,
        "checkpoint_path": saved_path,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Siamese terrain CNN.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--pair-noise-std", type=float, default=0.02,
                        help="Gaussian augmentation std after unit-range normalization")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--shifted-positives", action="store_true",
        help="Experiment 1: use shifted-crop positive pairs instead of "
             "independent random crops.  Default: False (original behaviour).",
    )
    parser.add_argument(
        "--max-shift", type=int, default=8,
        help="Maximum pixel displacement for shifted-crop positives "
             "(used only with --shifted-positives).  Default: 8.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    metrics = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        margin=args.margin,
        pair_noise_std=args.pair_noise_std,
        smoke_test=args.smoke_test,
        checkpoint_path=args.checkpoint,
        shifted_positives=args.shifted_positives,
        max_shift=args.max_shift,
    )

    if metrics["train_losses"]:
        print(f"\nFinal train loss : {metrics['train_losses'][-1]:.4f}")
        print(f"Final val loss   : {metrics['val_losses'][-1]:.4f}")


if __name__ == "__main__":
    main()
