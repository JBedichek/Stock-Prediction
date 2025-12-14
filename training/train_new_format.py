#!/usr/bin/env python3
"""
Training Script for New Dataset Format

Simple training loop for the new enhanced dataset format.
Supports both regression and classification approaches.

Usage:
    python -m training.train_new_format --data <path_to_dataset.pkl>

    OR from the project root:
    python training/train_new_format.py --data <path_to_dataset.pkl>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from lion_pytorch import Lion
import wandb
import argparse
from tqdm import tqdm
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import set_nan_inf

# Vectorized bin conversion for fast classification (no GPU-CPU sync)
def convert_price_ratios_to_bins_vectorized(ratios: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """
    Vectorized conversion of price ratios to bin indices (GPU-only, no CPU sync).

    Args:
        ratios: (batch_size, num_days) tensor of price ratios
        bin_edges: (num_bins+1,) tensor with bin edges

    Returns:
        bin_indices: (batch_size, num_days) long tensor with bin indices
    """
    # Move bin_edges to same device as ratios
    bin_edges = bin_edges.to(ratios.device)

    # Use searchsorted to find bins (vectorized, GPU-only)
    # ratios: (batch, days) -> flatten -> (batch*days,)
    flat_ratios = ratios.reshape(-1)
    flat_bins = torch.searchsorted(bin_edges, flat_ratios, right=False) - 1

    # Clip to valid range [0, num_bins-1]
    num_bins = len(bin_edges) - 1
    flat_bins = torch.clamp(flat_bins, 0, num_bins - 1)

    # Reshape back to (batch, days)
    bin_indices = flat_bins.reshape(ratios.shape)

    return bin_indices


def compute_confidence_targets(pred_logits: torch.Tensor,
                               bin_indices: torch.Tensor) -> torch.Tensor:
    """
    Compute confidence targets from per-sample cross-entropy loss.

    Uses exponential mapping: confidence = exp(-CE_loss)
    This provides dense signal (not sparse like accuracy) and naturally maps to [0, 1].

    Low CE (good prediction) → high confidence (near 1.0)
    High CE (bad prediction) → low confidence (near 0.0)

    Args:
        pred_logits: (batch, num_bins, num_pred_days) - model output logits
        bin_indices: (batch, num_pred_days) - ground truth bin indices

    Returns:
        confidence_targets: (batch, num_pred_days) - values in [0, 1]
    """
    batch_size = pred_logits.shape[0]
    num_pred_days = pred_logits.shape[2]

    confidence_targets = torch.zeros(batch_size, num_pred_days, device=pred_logits.device)

    for day_idx in range(num_pred_days):
        targets = bin_indices[:, day_idx]  # (batch,)

        # Compute per-sample cross-entropy loss (no reduction)
        per_sample_ce = F.cross_entropy(
            pred_logits[:, :, day_idx],
            targets,
            reduction='none'
        )  # (batch,)

        # Map to confidence: exp(-CE)
        # When CE=0 (perfect), confidence=1.0
        # When CE=large, confidence→0
        confidence_targets[:, day_idx] = torch.exp(-per_sample_ce)

    return confidence_targets  # (batch, num_pred_days) in [0, 1]


def compute_expected_value(pred_logits: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """
    Compute expected value (mean) of the predicted distribution.

    Expected value = sum(probability * bin_center) for each bin.

    This provides a point estimate from the distribution that can be compared
    to the actual price ratio for regularization.

    Args:
        pred_logits: (batch, num_bins, num_pred_days) - model output logits
        bin_edges: (num_bins+1,) - bin edges defining the bins

    Returns:
        expected_values: (batch, num_pred_days) - expected price ratio for each sample
    """
    # Move bin_edges to same device
    bin_edges = bin_edges.to(pred_logits.device)

    # Compute bin centers from edges
    # bin_edges: [e0, e1, e2, ..., en] -> centers: [(e0+e1)/2, (e1+e2)/2, ..., (e(n-1)+en)/2]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # (num_bins,)

    # Convert logits to probabilities
    probs = F.softmax(pred_logits, dim=1)  # (batch, num_bins, num_pred_days)

    # Compute expected value for each day
    # probs: (batch, num_bins, num_pred_days)
    # bin_centers: (num_bins,) -> reshape to (1, num_bins, 1)
    bin_centers = bin_centers.reshape(1, -1, 1)

    # Expected value = sum over bins (prob * center)
    expected_values = (probs * bin_centers).sum(dim=1)  # (batch, num_pred_days)

    return expected_values


# Auto-detect data loader based on file extension
def get_data_module(dataset_path: str):
    """Auto-detect and import the correct data loader."""
    if dataset_path.endswith('.h5') or dataset_path.endswith('.hdf5'):
        from .hdf5_data_loader import StockDataModule, convert_price_ratio_to_one_hot, compute_adaptive_bin_edges
        print("  ✅ Using HDF5 data loader (fast)")
    else:
        from .new_data_loader import StockDataModule, convert_price_ratio_to_one_hot, compute_adaptive_bin_edges
        print("  ⚠️  Using pickle data loader (slow)")
    return StockDataModule, convert_price_ratio_to_one_hot, compute_adaptive_bin_edges


class SimpleTransformerPredictor(nn.Module):
    """
    Simple transformer-based predictor for stock prices.

    Architecture:
    1. Input projection
    2. Transformer encoder layers
    3. Prediction head
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_pred_days: int = 4,
                 pred_mode: str = 'regression'):
        """
        Initialize model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            num_pred_days: Number of future days to predict
            pred_mode: 'regression' or 'classification'
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_pred_days = num_pred_days
        self.pred_mode = pred_mode

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 3000, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction head
        if pred_mode == 'regression':
            # Directly predict price ratios
            self.pred_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_pred_days)
            )
        else:
            # Classification: predict distribution over bins
            num_bins = 100
            self.num_bins = num_bins
            self.pred_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, num_bins * num_pred_days)
            )

        # Confidence head (predicts uncertainty for each horizon)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_pred_days),
            nn.Sigmoid()  # Output in [0, 1]
        )

        print(f"\n📊 Model Architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Transformer layers: {num_layers}")
        print(f"  Attention heads: {num_heads}")
        print(f"  Pooling: mean")
        print(f"  Prediction mode: {pred_mode}")
        print(f"  Predicting {num_pred_days} future days")
        print(f"  Confidence output: {num_pred_days} values [0, 1]")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim) tensor

        Returns:
            predictions: (batch, num_pred_days) for regression
                        or (batch, num_bins, num_pred_days) for classification
            confidence: (batch, num_pred_days) confidence values in [0, 1]
        """
        batch_size, seq_len, _ = x.shape

        # Handle NaN/Inf
        x = set_nan_inf(x)

        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer
        x = self.transformer(x)  # (batch, seq_len, hidden_dim)

        # Mean pooling across sequence dimension
        x = x.mean(dim=1)  # (batch, hidden_dim)

        # Prediction head
        pred = self.pred_head(x)

        if self.pred_mode == 'classification':
            # Reshape to (batch, num_bins, num_pred_days)
            pred = pred.reshape(batch_size, self.num_bins, self.num_pred_days)

        # Confidence head
        confidence = self.confidence_head(x)  # (batch, num_pred_days)

        return pred, confidence


def train_epoch(model, dataloader, optimizer, device, pred_mode='regression', convert_fn=None, scaler=None, use_amp=False,
                val_loader=None, eval_every=1000, global_step=0, ema_loss=None, ema_alpha=0.98, bin_edges=None,
                grad_accum_steps=1, best_val_loss=float('inf'), save_dir='./checkpoints', epoch=0, args=None,
                confidence_weight=0.5, expected_value_weight=0.3):
    """Train for one epoch with optional mixed precision, gradient accumulation, and step-based validation."""
    model.train()
    total_loss = 0
    num_batches = 0

    # Initialize EMA if not provided
    if ema_loss is None:
        ema_loss = 0.0

    # Track accumulated batches
    accum_iter = 0

    pbar = tqdm(dataloader, desc="Training")
    for features, prices, _, _ in pbar:
        features = features.to(device)
        prices = prices.to(device)

        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            # Forward
            pred, confidence = model(features)

            # Loss
            if pred_mode == 'regression':
                loss = F.mse_loss(pred, prices)
                main_loss = loss
                confidence_loss = torch.tensor(0.0)
                expected_value_loss = torch.tensor(0.0)
            else:
                # Classification: vectorized bin conversion (fast, GPU-only)
                # prices: (batch, num_days), bin_edges: (num_bins+1,)
                bin_indices = convert_price_ratios_to_bins_vectorized(prices, bin_edges)  # (batch, num_days)

                # Main classification loss
                main_loss = 0
                for day_idx in range(prices.shape[1]):
                    targets = bin_indices[:, day_idx]  # (batch,) long tensor
                    main_loss += F.cross_entropy(pred[:, :, day_idx], targets)

                # Confidence loss: train confidence to match exp(-CE)
                with torch.no_grad():
                    confidence_targets = compute_confidence_targets(pred, bin_indices)

                confidence_loss = F.mse_loss(confidence, confidence_targets)

                # Expected Value Regularization: penalize difference between
                # expected value from distribution and actual price movement
                expected_values = compute_expected_value(pred, bin_edges)  # (batch, num_pred_days)
                expected_value_loss = F.mse_loss(expected_values, prices)

                # Combined loss (using configurable weights)
                loss = main_loss + confidence_weight * confidence_loss + expected_value_weight * expected_value_loss

        # Store unscaled losses for logging
        unscaled_loss = loss.item()
        unscaled_main_loss = main_loss.item() if isinstance(main_loss, torch.Tensor) else main_loss
        unscaled_confidence_loss = confidence_loss.item() if isinstance(confidence_loss, torch.Tensor) else confidence_loss
        unscaled_expected_value_loss = expected_value_loss.item() if isinstance(expected_value_loss, torch.Tensor) else expected_value_loss

        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps

        # Backward with gradient scaling for mixed precision
        if use_amp and scaler is not None:
            # Mixed precision backward
            scaler.scale(loss).backward()
        else:
            # Standard backward
            loss.backward()

        # Only step optimizer every grad_accum_steps
        accum_iter += 1
        if accum_iter % grad_accum_steps == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad()

        # Update EMA loss (use unscaled loss)
        batch_loss = unscaled_loss
        if num_batches == 0:
            ema_loss = batch_loss
        else:
            ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * batch_loss

        total_loss += batch_loss
        num_batches += 1
        global_step += 1

        # Update progress bar
        pbar.set_postfix({'loss': batch_loss, 'ema_loss': ema_loss})

        # Log to wandb every step
        if wandb.run is not None:
            log_dict = {
                'train/loss': batch_loss,
                'train/ema_loss': ema_loss,
                'train/main_loss': unscaled_main_loss,
                'train/confidence_loss': unscaled_confidence_loss,
                'train/expected_value_loss': unscaled_expected_value_loss,
                'global_step': global_step
            }

            # Add confidence statistics if in classification mode
            if pred_mode == 'classification':
                log_dict['train/mean_confidence'] = confidence.mean().item()
                if 'confidence_targets' in locals():
                    log_dict['train/mean_confidence_target'] = confidence_targets.mean().item()
                if 'expected_values' in locals():
                    log_dict['train/mean_expected_value'] = expected_values.mean().item()
                    log_dict['train/mean_actual_value'] = prices.mean().item()

            wandb.log(log_dict, step=global_step)

        # Evaluate on validation set every eval_every steps
        if val_loader is not None and global_step % eval_every == 0:
            val_loss = validate(model, val_loader, device, pred_mode, convert_fn, use_amp, bin_edges, show_progress=False)

            # Log validation metrics to wandb
            if wandb.run is not None:
                wandb.log({
                    'val/loss_step': val_loss,
                    'global_step': global_step
                }, step=global_step)

            print(f"\n  📊 Step {global_step}: Val loss = {val_loss:.6f}")

            # Save checkpoint if validation improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(save_dir, 'best_model.pt')
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_ema_loss': ema_loss,
                    'config': vars(args) if args is not None else {}
                }
                if scaler is not None:
                    checkpoint['scaler_state_dict'] = scaler.state_dict()
                torch.save(checkpoint, save_path)
                print(f"  ✅ Saved best model (val_loss: {val_loss:.6f})")

            # Back to training mode
            model.train()

    return total_loss / num_batches, global_step, ema_loss, best_val_loss


@torch.no_grad()
def validate(model, dataloader, device, pred_mode='regression', convert_fn=None, use_amp=False, bin_edges=None, show_progress=True):
    """Validate the model with optional mixed precision."""
    model.eval()
    total_loss = 0
    num_batches = 0

    iterator = tqdm(dataloader, desc="Validating") if show_progress else dataloader
    for features, prices, _, _ in iterator:
        features = features.to(device)
        prices = prices.to(device)

        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            # Forward
            pred, confidence = model(features)

            # Loss
            if pred_mode == 'regression':
                loss = F.mse_loss(pred, prices)
            else:
                # Classification: vectorized bin conversion (fast, GPU-only)
                bin_indices = convert_price_ratios_to_bins_vectorized(prices, bin_edges)  # (batch, num_days)

                # Main classification loss
                main_loss = 0
                for day_idx in range(prices.shape[1]):
                    targets = bin_indices[:, day_idx]  # (batch,) long tensor
                    main_loss += F.cross_entropy(pred[:, :, day_idx], targets)

                # Confidence loss
                confidence_targets = compute_confidence_targets(pred, bin_indices)
                confidence_loss = F.mse_loss(confidence, confidence_targets)

                # Combined loss
                confidence_weight = 0.5  # Hyperparameter (should match training)
                loss = main_loss + confidence_weight * confidence_loss

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train stock predictor with new data format')

    # Data args
    parser.add_argument('--data', type=str, default='data/outliers_full_dataset.h5',
                       help='Path to dataset pickle file')
    parser.add_argument('--seq-len', type=int, default=3000,
                       help='Sequence length (default: 60)')
    parser.add_argument('--batch-size', type=int, default=12,
                       help='Batch size (default: 32)')
    parser.add_argument('--num-workers', type=int, default=6,
                       help='Number of data workers (default: 4)')

    # Model args
    parser.add_argument('--hidden-dim', type=int, default=1024,
                       help='Hidden dimension (default: 512)')
    parser.add_argument('--num-layers', type=int, default=10,
                       help='Number of transformer layers (default: 6)')
    parser.add_argument('--num-heads', type=int, default=16,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--dropout', type=float, default=0.15,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--pred-mode', type=str, default='classification',
                       choices=['regression', 'classification'],
                       help='Prediction mode (default: regression)')

    # Loss weights
    parser.add_argument('--confidence-weight', type=float, default=0.5,
                       help='Weight for confidence loss (default: 0.5)')
    parser.add_argument('--expected-value-weight', type=float, default=0.3,
                       help='Weight for expected value regularization (default: 0.3)')

    # Training args
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=4e-5,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.8,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'lion'],
                       help='Optimizer (default: adamw)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use automatic mixed precision (fp16) training for 2x speedup and memory savings')
    parser.add_argument('--compile', action='store_true', default=True,
                       help='Use torch.compile with max-autotune for additional speedup (requires PyTorch 2.0+)')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                       help='Gradient accumulation steps (default: 1, use >1 for larger effective batch size)')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                       help='Path to checkpoint file to resume training from (default: None)')

    # Logging
    parser.add_argument('--use-wandb', action='store_true', default=True,
                       help='Use Weights & Biases logging')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--eval-every', type=int, default=1000,
                       help='Evaluate on test set every N steps (default: 1000)')
    parser.add_argument('--ema-alpha', type=float, default=0.98,
                       help='EMA smoothing factor for training loss (default: 0.98)')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize wandb
    if args.use_wandb:
        config = vars(args).copy()
        config['effective_batch_size'] = args.batch_size * args.grad_accum_steps
        wandb.init(project='stock-prediction', config=config)

    print("\n" + "="*80)
    print("STOCK PREDICTION TRAINING")
    print("="*80)

    # Create data module (auto-detect format)
    print("\n📦 Loading data...")
    StockDataModule, convert_price_ratio_to_one_hot, compute_adaptive_bin_edges = get_data_module(args.data)
    dm = StockDataModule(
        dataset_path=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
        pred_days=[1, 5, 10, 20],
        val_max_size=1000,  # Limit validation to 5k sequences for faster eval
        test_max_size=1000  # Limit test to 5k sequences
    )

    # Create dataloaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    print(f"\n  📊 Dataset splits:")
    print(f"    Train: {len(dm.train_dataset):,} sequences")
    print(f"    Val:   {len(dm.val_dataset):,} sequences")
    print(f"    Test:  {len(dm.test_dataset):,} sequences")

    # Compute adaptive bin edges for classification mode
    bin_edges = None
    if args.pred_mode == 'classification':
        print(f"\n🔢 Computing adaptive bin edges for classification...")
        bin_edges = compute_adaptive_bin_edges(
            dataset_path=args.data,
            num_bins=100,  # Using 100 bins like in model
            pred_days=dm.pred_days,
            max_samples=50000
        )
        print(f"  ✅ Bin edges computed and cached")

    # Create model
    print("\n🏗️  Building model...")
    model = SimpleTransformerPredictor(
        input_dim=dm.total_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_pred_days=len(dm.pred_days),
        pred_mode=args.pred_mode
    )
    model = model.to(args.device)

    # Count and print parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 Model Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1e6:.1f} MB (fp32)")

    # Print loss configuration
    if args.pred_mode == 'classification':
        print(f"\n⚖️  Loss Configuration:")
        print(f"  Main loss: Cross-Entropy (classification)")
        print(f"  Confidence loss weight: {args.confidence_weight}")
        print(f"  Expected value regularization weight: {args.expected_value_weight}")
        print(f"  Total loss = CE + {args.confidence_weight}*conf_loss + {args.expected_value_weight}*ev_loss")

    # Compile model with torch.compile for speedup
    if args.compile:
        try:
            print(f"\n⚙️  Compiling model with torch.compile (max-autotune)...")
            print(f"  This may take a few minutes on first run...")
            model = torch.compile(model, mode='default')
            print(f"  ✅ Model compiled successfully")
            print(f"  Expected: Additional 20-40% speedup after warmup")
        except Exception as e:
            print(f"\n⚠️  torch.compile failed: {e}")
            print(f"  Continuing without compilation (requires PyTorch 2.0+)")
            args.compile = False

    # Create optimizer
    if args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initialize gradient scaler for mixed precision
    scaler = None
    if args.use_amp and args.device == 'cuda':
        scaler = GradScaler()
        print(f"\n⚡ Mixed Precision (FP16) Training ENABLED")
        print(f"  Expected: ~2x speedup + 50% memory reduction")
    elif args.use_amp:
        print(f"\n⚠️  Mixed precision requested but device is {args.device}, using FP32")
        args.use_amp = False

    # Initialize training state
    best_val_loss = float('inf')
    global_step = 0
    ema_loss = None
    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint is not None:
        if os.path.exists(args.resume_from_checkpoint):
            print(f"\n📂 Loading checkpoint from: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=args.device)

            # Load model state (strict=False allows new confidence head)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            if missing_keys:
                print(f"  ⚠️  Missing keys (will be randomly initialized): {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
            if unexpected_keys:
                print(f"  ⚠️  Unexpected keys in checkpoint: {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")

            print(f"  ✅ Loaded model weights")

            # Only load optimizer if architecture matches (no missing keys)
            if not missing_keys:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"  ✅ Loaded optimizer state")

                # Load scaler state if available
                if scaler is not None and 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    print(f"  ✅ Loaded gradient scaler state")
            else:
                print(f"  ℹ️  Skipping optimizer/scaler state (architecture changed)")

            # Load training state
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"  ✅ Resuming from epoch {start_epoch}")

            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
                print(f"  ✅ Resuming from global step {global_step}")

            if 'val_loss' in checkpoint:
                best_val_loss = checkpoint['val_loss']
                print(f"  ✅ Best val loss: {best_val_loss:.6f}")

            if 'train_ema_loss' in checkpoint:
                ema_loss = checkpoint['train_ema_loss']
                print(f"  ✅ Train EMA loss: {ema_loss:.6f}")

            print(f"\n  🎯 Checkpoint loaded successfully!")
        else:
            print(f"\n⚠️  WARNING: Checkpoint file not found: {args.resume_from_checkpoint}")
            print(f"  Starting training from scratch...")

    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.grad_accum_steps

    print(f"\n🚀 Starting training...")
    print(f"  Epochs: {args.epochs} (starting from epoch {start_epoch + 1})")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.grad_accum_steps}")
    print(f"  Effective batch size: {effective_batch_size} ({args.batch_size} × {args.grad_accum_steps})")
    print(f"  Mixed precision (FP16): {'✅ Enabled' if args.use_amp else '❌ Disabled'}")
    print(f"  torch.compile: {'✅ Enabled (max-autotune)' if args.compile else '❌ Disabled'}")
    print(f"  Eval every: {args.eval_every} steps")
    print(f"  EMA alpha: {args.ema_alpha}")
    if args.resume_from_checkpoint:
        print(f"  Resumed from: {args.resume_from_checkpoint}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")

        # Train
        train_loss, global_step, ema_loss, best_val_loss = train_epoch(
            model, train_loader, optimizer, args.device,
            args.pred_mode, convert_price_ratio_to_one_hot,
            scaler=scaler, use_amp=args.use_amp,
            val_loader=val_loader, eval_every=args.eval_every,
            global_step=global_step, ema_loss=ema_loss, ema_alpha=args.ema_alpha,
            bin_edges=bin_edges, grad_accum_steps=args.grad_accum_steps,
            best_val_loss=best_val_loss, save_dir=args.save_dir, epoch=epoch + 1, args=args,
            confidence_weight=args.confidence_weight, expected_value_weight=args.expected_value_weight
        )

        # Validate
        val_loss = validate(
            model, val_loader, args.device,
            args.pred_mode, convert_price_ratio_to_one_hot,
            use_amp=args.use_amp,
            bin_edges=bin_edges
        )

        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"  Train loss: {train_loss:.6f}")
        print(f"  Train EMA loss: {ema_loss:.6f}")
        print(f"  Val loss: {val_loss:.6f}")
        print(f"  Global step: {global_step}")

        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'epoch/train_loss': train_loss,
                'epoch/train_ema_loss': ema_loss,
                'epoch/val_loss': val_loss,
                'global_step': global_step
            }, step=global_step)

        # Save best model (overwrites previous best)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'best_model.pt')
            checkpoint = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_ema_loss': ema_loss,
                'config': vars(args)
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint, save_path)
            print(f"  ✅ Saved best model (val_loss: {val_loss:.6f})")

    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
