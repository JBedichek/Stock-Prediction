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
from datetime import datetime

from utils.utils import set_nan_inf

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
        self.pos_encoding = nn.Parameter(torch.randn(1, 2000, hidden_dim) * 0.02)

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

        print(f"\n📊 Model Architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Transformer layers: {num_layers}")
        print(f"  Attention heads: {num_heads}")
        print(f"  Pooling: mean")
        print(f"  Prediction mode: {pred_mode}")
        print(f"  Predicting {num_pred_days} future days")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim) tensor

        Returns:
            predictions: (batch, num_pred_days) for regression
                        or (batch, num_bins, num_pred_days) for classification
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

        return pred


def train_epoch(model, dataloader, optimizer, device, pred_mode='regression', convert_fn=None, scaler=None, use_amp=False,
                test_loader=None, eval_every=1000, global_step=0, ema_loss=None, ema_alpha=0.98, bin_edges=None):
    """Train for one epoch with optional mixed precision and step-based evaluation."""
    model.train()
    total_loss = 0
    num_batches = 0

    # Initialize EMA if not provided
    if ema_loss is None:
        ema_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for features, prices, _, _ in pbar:
        features = features.to(device)
        prices = prices.to(device)

        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            # Forward
            pred = model(features)

            # Loss
            if pred_mode == 'regression':
                loss = F.mse_loss(pred, prices)
            else:
                # Classification: convert prices to bins using adaptive binning
                batch_size = prices.shape[0]
                loss = 0
                for day_idx in range(prices.shape[1]):
                    targets = []
                    for batch_idx in range(batch_size):
                        ratio = prices[batch_idx, day_idx].item()
                        # Pass bin_edges for adaptive binning
                        target = convert_fn(ratio, num_bins=model.num_bins, bin_edges=bin_edges)
                        targets.append(target)
                    targets = torch.stack(targets).to(device)

                    # Cross-entropy loss for this day
                    loss += F.cross_entropy(pred[:, :, day_idx], targets)

        # Backward with gradient scaling for mixed precision
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            # Mixed precision backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Update EMA loss
        batch_loss = loss.item()
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
            wandb.log({
                'train/loss': batch_loss,
                'train/ema_loss': ema_loss,
                'global_step': global_step
            }, step=global_step)

        # Evaluate on test set every eval_every steps
        if test_loader is not None and global_step % eval_every == 0:
            test_loss = validate(model, test_loader, device, pred_mode, convert_fn, use_amp, bin_edges)

            # Log test metrics to wandb
            if wandb.run is not None:
                wandb.log({
                    'test/loss': test_loss,
                    'global_step': global_step
                }, step=global_step)

            print(f"\n  📊 Step {global_step}: Test loss = {test_loss:.6f}")

            # Back to training mode
            model.train()

    return total_loss / num_batches, global_step, ema_loss


@torch.no_grad()
def validate(model, dataloader, device, pred_mode='regression', convert_fn=None, use_amp=False, bin_edges=None):
    """Validate the model with optional mixed precision."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for features, prices, _, _ in tqdm(dataloader, desc="Validating"):
        features = features.to(device)
        prices = prices.to(device)

        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            # Forward
            pred = model(features)

            # Loss
            if pred_mode == 'regression':
                loss = F.mse_loss(pred, prices)
            else:
                batch_size = prices.shape[0]
                loss = 0
                for day_idx in range(prices.shape[1]):
                    targets = []
                    for batch_idx in range(batch_size):
                        ratio = prices[batch_idx, day_idx].item()
                        # Pass bin_edges for adaptive binning
                        target = convert_fn(ratio, num_bins=model.num_bins, bin_edges=bin_edges)
                        targets.append(target)
                    targets = torch.stack(targets).to(device)
                    loss += F.cross_entropy(pred[:, :, day_idx], targets)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train stock predictor with new data format')

    # Data args
    parser.add_argument('--data', type=str, default='all_complete_dataset.pkl',
                       help='Path to dataset pickle file')
    parser.add_argument('--seq-len', type=int, default=2000,
                       help='Sequence length (default: 60)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data workers (default: 4)')

    # Model args
    parser.add_argument('--hidden-dim', type=int, default=2048,
                       help='Hidden dimension (default: 512)')
    parser.add_argument('--num-layers', type=int, default=24,
                       help='Number of transformer layers (default: 6)')
    parser.add_argument('--num-heads', type=int, default=16,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--dropout', type=float, default=0.15,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--pred-mode', type=str, default='classification',
                       choices=['regression', 'classification'],
                       help='Prediction mode (default: regression)')

    # Training args
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--optimizer', type=str, default='lion',
                       choices=['adamw', 'lion'],
                       help='Optimizer (default: adamw)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use automatic mixed precision (fp16) training for 2x speedup and memory savings')
    parser.add_argument('--compile', action='store_true', default=False,
                       help='Use torch.compile with max-autotune for additional speedup (requires PyTorch 2.0+)')

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
        wandb.init(project='stock-prediction', config=vars(args))

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
        val_max_size=5000,  # Limit validation to 5k sequences for faster eval
        test_max_size=5000  # Limit test to 5k sequences
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

    print(f"\n🚀 Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Device: {args.device}")
    print(f"  Mixed precision (FP16): {'✅ Enabled' if args.use_amp else '❌ Disabled'}")
    print(f"  torch.compile: {'✅ Enabled (max-autotune)' if args.compile else '❌ Disabled'}")
    print(f"  Eval every: {args.eval_every} steps")
    print(f"  EMA alpha: {args.ema_alpha}")

    best_val_loss = float('inf')
    global_step = 0
    ema_loss = None

    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")

        # Train
        train_loss, global_step, ema_loss = train_epoch(
            model, train_loader, optimizer, args.device,
            args.pred_mode, convert_price_ratio_to_one_hot,
            scaler=scaler, use_amp=args.use_amp,
            test_loader=test_loader, eval_every=args.eval_every,
            global_step=global_step, ema_loss=ema_loss, ema_alpha=args.ema_alpha,
            bin_edges=bin_edges
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

        # Save best model
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

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pt')
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
            print(f"  💾 Saved checkpoint")

    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
