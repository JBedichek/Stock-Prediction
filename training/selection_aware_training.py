#!/usr/bin/env python3
"""
Selection-Aware Training: Focus on Top Predictions

This training strategy focuses on the sequences that the model predicts will perform best,
which are the ones that would actually be traded. This helps the model learn to discriminate
better among high-value predictions.

Algorithm:
1. Sample 10k sequences randomly
2. Run inference to get model predictions
3. Calculate expected return from predictions
4. Keep top 100 (1%) with highest expected return
5. Train on those 100 for 3 epochs
6. Repeat

Usage:
    python training/selection_aware_training.py --data all_complete_dataset.h5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from lion_pytorch import Lion
import wandb
import argparse
from tqdm import tqdm
import os
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ===== Performance Optimizations =====
# These optimizations target the main bottleneck: CPU-bound data loading during inference
#
# 1. CPU Thread Optimization:
torch.set_num_threads(torch.get_num_threads())  # Use all available CPU threads
torch.set_num_interop_threads(4)  # Limit inter-op parallelism to reduce overhead
#
# 2. CPU Kernel Optimization:
if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = True  # Enable Intel MKL-DNN optimizations
#
# 3. GPU Optimizations:
torch.set_float32_matmul_precision('medium')  # TF32 on Ampere GPUs (~8x faster matmul)
torch.backends.cudnn.benchmark = True  # Autotune cuDNN algorithms for this input size
#
# 4. DataLoader Optimizations (applied in sample_and_select_top_sequences):
#    - num_workers=4: Parallel data loading (keeps GPU fed)
#    - prefetch_factor=2: Preload batches ahead
#    - persistent_workers=False: Disabled to avoid CUDA illegal memory access with HDF5
#    - pin_memory=True: Faster GPU transfers
#    - non_blocking=True: Async GPU transfers (overlap with computation)
#
# 5. Batch Size Optimization:
#    - inference_batch_size=256 (default): 4x larger than training (no gradients)
#
# 6. torch.compile (optional --compile flag):
#    - mode='max-autotune': Triton autotuning for 2-3x inference speedup

from training.hdf5_data_loader import StockDataModule
from training.train_new_format import (
    SimpleTransformerPredictor,
    convert_price_ratios_to_bins_vectorized,
    validate
)
from utils.utils import set_nan_inf


def get_expected_return(pred_output: tuple, bin_edges: torch.Tensor, horizon_idx: int = 1) -> tuple:
    """
    Calculate expected return and confidence from model predictions.

    Args:
        pred_output: (pred_logits, confidence) tuple from model
        bin_edges: (num_bins+1,) bin edges for price ratios
        horizon_idx: which horizon to use (0=1day, 1=5day, 2=10day, 3=20day)

    Returns:
        expected_returns: (batch,) expected price ratio for each sample
        confidences: (batch,) confidence values [0, 1]
    """
    pred_logits, confidence = pred_output  # Unpack

    # Extract logits and confidence for specific horizon
    logits = pred_logits[:, :, horizon_idx]  # (batch, num_bins)
    horizon_confidence = confidence[:, horizon_idx]  # (batch,)

    # Convert to probabilities
    probs = F.softmax(logits, dim=1)  # (batch, num_bins)

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # (num_bins,)
    bin_centers = bin_centers.to(probs.device)

    # Expected value = sum(prob * bin_center)
    expected = torch.sum(probs * bin_centers.unsqueeze(0), dim=1)  # (batch,)

    return expected, horizon_confidence


def sample_and_select_top_sequences(model: nn.Module,
                                     full_dataset,
                                     bin_edges: torch.Tensor,
                                     sample_size: int = 10000,
                                     top_k: int = 100,
                                     horizon_idx: int = 1,
                                     device: str = 'cuda',
                                     batch_size: int = 64) -> list:
    """
    Sample sequences, run inference, and return indices of top performers.

    Args:
        model: The prediction model
        full_dataset: Full training dataset
        bin_edges: Bin edges for expected return calculation
        sample_size: Number of sequences to sample
        top_k: Number of top sequences to keep
        horizon_idx: Which prediction horizon to use
        device: cuda or cpu
        batch_size: Batch size for inference

    Returns:
        List of indices for top_k sequences
    """
    model.eval()

    # Sample random indices
    total_size = len(full_dataset)
    sample_indices = np.random.choice(total_size, size=min(sample_size, total_size), replace=False)

    # Create subset and dataloader
    subset = Subset(full_dataset, sample_indices)
    # Optimize for high throughput inference:
    # - num_workers=4: Parallel data loading (8 caused CUDA memory issues with HDF5)
    # - prefetch_factor=2: Preload batches ahead
    # - persistent_workers=False: Avoid CUDA illegal memory access with multiprocessing
    # - pin_memory=True: Still faster despite no persistent workers
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False  # Disabled to avoid CUDA memory errors
    )

    # Run inference
    all_expected_returns = []
    all_confidences = []

    with torch.no_grad():
        for features, prices, _, _ in tqdm(loader, desc="  Inference", leave=False):
            # Async GPU transfer (overlaps with previous batch computation)
            features = features.to(device, non_blocking=True)

            # Forward pass
            with autocast():
                pred = model(features)

            # Calculate expected returns and confidences
            expected_returns, confidences = get_expected_return(pred, bin_edges, horizon_idx)
            all_expected_returns.append(expected_returns.cpu())
            all_confidences.append(confidences.cpu())

    # Concatenate all returns and confidences
    all_expected_returns = torch.cat(all_expected_returns, dim=0)  # (sample_size,)
    all_confidences = torch.cat(all_confidences, dim=0)  # (sample_size,)

    # Filter by confidence percentile (not absolute threshold)
    confidence_percentile = 0.6  # Keep top 40% by confidence
    confidence_threshold = torch.quantile(all_confidences.float(), confidence_percentile)
    high_confidence_mask = all_confidences >= confidence_threshold
    filtered_returns = all_expected_returns[high_confidence_mask]
    filtered_indices = sample_indices[high_confidence_mask.numpy()]
    filtered_confidences = all_confidences[high_confidence_mask]

    # Weight by confidence
    confidence_weighted_returns = filtered_returns * filtered_confidences

    # Get top-k by weighted returns
    if len(confidence_weighted_returns) < top_k:
        print(f"    ⚠️  Warning: Only {len(confidence_weighted_returns)} samples in top {100*(1-confidence_percentile):.0f}% confidence, using all")
        top_indices = filtered_indices.tolist()
    else:
        _, top_indices_in_filtered = torch.topk(confidence_weighted_returns, k=top_k, largest=True)
        top_indices = [filtered_indices[i] for i in top_indices_in_filtered.numpy()]

    # Print statistics
    print(f"    📊 Expected return stats:")
    print(f"      Mean: {all_expected_returns.mean():.4f}")
    print(f"      Median: {all_expected_returns.median():.4f}")
    print(f"    📊 Confidence filtering (percentile-based):")
    print(f"      Confidence threshold (p{confidence_percentile:.0%}): {confidence_threshold:.4f}")
    print(f"      Samples in top {100*(1-confidence_percentile):.0f}%: {high_confidence_mask.sum().item()}/{len(all_confidences)}")
    print(f"      Mean confidence (all): {all_confidences.mean():.4f}")
    print(f"      Confidence range: [{all_confidences.min():.4f}, {all_confidences.max():.4f}]")
    if len(filtered_confidences) > 0:
        print(f"      Mean confidence (filtered): {filtered_confidences.mean():.4f}")
    if len(top_indices) > 0:
        top_mask = torch.tensor([i in top_indices for i in filtered_indices])
        print(f"      Mean confidence (selected): {filtered_confidences[top_mask].mean():.4f}")

    return top_indices


def train_on_subset(model: nn.Module,
                    subset_indices: list,
                    full_dataset,
                    optimizer,
                    bin_edges: torch.Tensor,
                    num_epochs: int = 3,
                    batch_size: int = 32,
                    device: str = 'cuda',
                    use_amp: bool = True,
                    scaler=None) -> float:
    """
    Train on a subset of sequences for a few epochs.

    Args:
        model: The prediction model
        subset_indices: List of indices to train on
        full_dataset: Full dataset
        optimizer: Optimizer
        bin_edges: Bin edges for classification
        num_epochs: Number of epochs to train
        batch_size: Batch size
        device: cuda or cpu
        use_amp: Use mixed precision
        scaler: Gradient scaler

    Returns:
        Average loss
    """
    model.train()

    # Create subset and dataloader
    subset = Subset(full_dataset, subset_indices)
    # Optimize for training throughput:
    # - num_workers=2: Lower for training (smaller subset, gradients enabled)
    # - prefetch_factor=2: Prefetch ahead
    # - persistent_workers=False: Avoid CUDA memory errors
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False
    )

    total_loss = 0
    num_batches = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_batches = 0

        for features, prices, _, _ in loader:
            # Async GPU transfer
            features = features.to(device, non_blocking=True)
            prices = prices.to(device, non_blocking=True)

            # Forward pass
            with autocast(enabled=use_amp):
                pred, confidence = model(features)

                # Classification loss (all horizons)
                bin_indices = convert_price_ratios_to_bins_vectorized(prices, bin_edges)
                main_loss = 0
                for day_idx in range(prices.shape[1]):
                    targets = bin_indices[:, day_idx]
                    main_loss += F.cross_entropy(pred[:, :, day_idx], targets)

                # Confidence loss
                from training.train_new_format import compute_confidence_targets
                with torch.no_grad():
                    confidence_targets = compute_confidence_targets(pred, bin_indices)
                confidence_loss = F.mse_loss(confidence, confidence_targets)

                # Combined loss
                confidence_weight = 0.5
                loss = main_loss + confidence_weight * confidence_loss

            # Backward
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_batches += 1

        total_loss += epoch_loss / epoch_batches
        num_batches += 1

    avg_loss = total_loss / num_epochs
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Selection-aware training')

    # Data args
    parser.add_argument('--data', type=str, default='data/all_complete_dataset.h5',
                       help='Path to dataset HDF5 file')
    parser.add_argument('--seq-len', type=int, default=3000,
                       help='Sequence length')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Number of sequences to sample each iteration')
    parser.add_argument('--top-k', type=int, default=100,
                       help='Number of top sequences to keep (1%% of sample)')
    parser.add_argument('--train-epochs-per-iteration', type=int, default=3,
                       help='Number of epochs to train on top-k per iteration')

    # Model args
    parser.add_argument('--hidden-dim', type=int, default=1024,
                       help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=16,
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=16,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.15,
                       help='Dropout rate')

    # Training args
    parser.add_argument('--num-iterations', type=int, default=100,
                       help='Number of sample-select-train iterations')
    parser.add_argument('--lr', type=float, default=4e-5,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--inference-batch-size', type=int, default=128,
                       help='Inference batch size for selection')
    parser.add_argument('--horizon-idx', type=int, default=0,
                       help='Prediction horizon for selection (0=1d, 1=5d, 2=10d, 3=20d)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use mixed precision')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile with max-autotune (~2-3x inference speedup)')
    parser.add_argument('--val-every', type=int, default=5,
                       help='Validate every N iterations')

    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume-from', type=str, default="./checkpoints/best_model.pt",
                       help='Resume from checkpoint')

    # Logging
    parser.add_argument('--use-wandb', action='store_true', default=True,
                       help='Use Weights & Biases logging')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize wandb
    if args.use_wandb:
        wandb.init(project='stock-prediction-selection-aware', config=vars(args))

    print("\n" + "="*80)
    print("SELECTION-AWARE TRAINING")
    print("="*80)
    print(f"\nStrategy:")
    print(f"  Sample {args.sample_size} sequences")
    print(f"  Select top {args.top_k} ({args.top_k/args.sample_size*100:.1f}%) by expected return")
    print(f"  Train on them for {args.train_epochs_per_iteration} epochs")
    print(f"  Repeat {args.num_iterations} times\n")

    # Load data
    print("\n📦 Loading data...")
    dm = StockDataModule(
        dataset_path=args.data,
        batch_size=args.batch_size,
        num_workers=0,
        seq_len=args.seq_len,
        pred_days=[1, 5, 10, 20],
        val_max_size=1000,
        test_max_size=1000
    )

    train_dataset = dm.train_dataset
    val_loader = dm.val_dataloader()

    print(f"\n  📊 Dataset:")
    print(f"    Train: {len(train_dataset):,} sequences")
    print(f"    Val:   {len(dm.val_dataset):,} sequences")

    # Load bin edges
    print(f"\n🔢 Loading bin edges...")
    from training.hdf5_data_loader import compute_adaptive_bin_edges
    bin_edges = compute_adaptive_bin_edges(
        dataset_path=args.data,
        num_bins=100,
        pred_days=dm.pred_days,
        max_samples=50000
    )
    print(f"  ✅ Bin edges loaded")

    # Create model
    print(f"\n🏗️  Building model...")
    model = SimpleTransformerPredictor(
        input_dim=dm.total_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_pred_days=len(dm.pred_days),
        pred_mode='classification'
    )
    model = model.to(args.device)

    # Compile model for faster inference if requested
    if args.compile:
        print(f"  🔧 Compiling model with torch.compile (max-autotune mode)...")
        print(f"     This may take a few minutes on first run...")
        print(f"     Triton will benchmark multiple kernel configurations...")
        # max-autotune: Best performance, autotuning of Triton kernels
        # First iteration will be slower due to compilation + autotuning
        # Subsequent iterations will be ~2-3x faster
        model = torch.compile(model, mode='max-autotune')
        print(f"  ✅ Model compiled successfully")

    # Optimizer
    optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Mixed precision scaler
    scaler = None
    if args.use_amp and args.device == 'cuda':
        scaler = GradScaler()
        print(f"\n⚡ Mixed Precision (FP16) ENABLED")

    # Resume from checkpoint if specified
    start_iteration = 0
    best_val_loss = float('inf')

    if args.resume_from is not None and os.path.exists(args.resume_from):
        print(f"\n📂 Loading checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=args.device)

        # Load with strict=False to allow new confidence head (not in old checkpoints)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if missing_keys:
            print(f"  ⚠️  Missing keys (will be randomly initialized): {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
        if unexpected_keys:
            print(f"  ⚠️  Unexpected keys in checkpoint: {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")

        # Don't load optimizer state if architecture changed (confidence head added)
        if not missing_keys:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
        else:
            print(f"  ℹ️  Skipping optimizer state (architecture changed)")

        start_iteration = checkpoint.get('iteration', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  ✅ Resumed from iteration {start_iteration}")

    # Training loop
    print(f"\n🚀 Starting selection-aware training...")
    print(f"  Iterations: {args.num_iterations} (starting from {start_iteration})")
    print(f"  Horizon: {['1d', '5d', '10d', '20d'][args.horizon_idx]}")

    for iteration in range(start_iteration, args.num_iterations):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{args.num_iterations}")
        print(f"{'='*80}")

        # Step 1: Sample and select top sequences
        print(f"\n  🎲 Sampling {args.sample_size} sequences and selecting top {args.top_k}...")
        top_indices = sample_and_select_top_sequences(
            model=model,
            full_dataset=train_dataset,
            bin_edges=bin_edges,
            sample_size=args.sample_size,
            top_k=args.top_k,
            horizon_idx=args.horizon_idx,
            device=args.device,
            batch_size=args.inference_batch_size
        )

        # Step 2: Train on top sequences
        print(f"\n  🏋️  Training on top {args.top_k} for {args.train_epochs_per_iteration} epochs...")
        train_loss = train_on_subset(
            model=model,
            subset_indices=top_indices,
            full_dataset=train_dataset,
            optimizer=optimizer,
            bin_edges=bin_edges,
            num_epochs=args.train_epochs_per_iteration,
            batch_size=args.batch_size,
            device=args.device,
            use_amp=args.use_amp,
            scaler=scaler
        )

        print(f"  ✅ Training loss: {train_loss:.6f}")

        # Step 3: Validate
        if (iteration + 1) % args.val_every == 0:
            print(f"\n  📊 Validating...")
            val_loss = validate(
                model=model,
                dataloader=val_loader,
                device=args.device,
                pred_mode='classification',
                convert_fn=None,
                use_amp=args.use_amp,
                bin_edges=bin_edges,
                show_progress=False
            )
            print(f"  ✅ Validation loss: {val_loss:.6f}")

            # Log to wandb
            if args.use_wandb:
                wandb.log({
                    'iteration': iteration + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.save_dir, 'best_model_selection_aware.pt')
                checkpoint = {
                    'iteration': iteration + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'config': vars(args)
                }
                if scaler is not None:
                    checkpoint['scaler_state_dict'] = scaler.state_dict()
                torch.save(checkpoint, save_path)
                print(f"  ✅ Saved best model (val_loss: {val_loss:.6f})")
        else:
            # Log training loss only
            if args.use_wandb:
                wandb.log({
                    'iteration': iteration + 1,
                    'train_loss': train_loss
                })

        # Save periodic checkpoint
        if (iteration + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_iter_{iteration+1}.pt')
            checkpoint = {
                'iteration': iteration + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': vars(args)
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint, save_path)
            print(f"  💾 Saved checkpoint: {save_path}")

    print(f"\n{'='*80}")
    print("✅ SELECTION-AWARE TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
