#!/usr/bin/env python3
"""
Cross-Sectional Stock Selection Model

Instead of predicting returns for each stock independently, this model:
1. Takes a batch of N stocks from the same trading date
2. Encodes each stock to get embeddings
3. Applies cross-stock attention so stocks can "compare" to each other
4. Outputs a softmax distribution over stocks (portfolio weights)
5. Trains with soft targets derived from actual returns

Key insight: The model's loss DIRECTLY corresponds to portfolio performance.
If target = softmax(returns/τ) and prediction = softmax(logits), then:
- Low CE loss → predicted weights align with return-based weights
- The model learns to put more weight on better-performing stocks

This is highly interpretable:
- Portfolio return = Σ(weight_i × return_i)
- The model is directly optimizing for stock selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math
import numpy as np
from tqdm import tqdm

from training.multimodal_model import (
    FeatureConfig,
    FundamentalsEncoder,
    TechnicalEncoder,
    NewsEncoder,
)
from tests.evaluate_encoder import (
    evaluate_encoder,
    run_linear_probe_eval,
    compute_ic,
)


@dataclass
class CrossSectionalConfig:
    """Configuration for cross-sectional model."""
    # Number of stocks per cross-section (batch)
    num_stocks: int = 32

    # Temperature for soft targets: target = softmax(returns / temperature)
    # Lower = more peaky (winner-take-all), Higher = more uniform
    target_temperature: float = 1.0

    # Whether to use top-k only (ignore bottom stocks in loss)
    top_k_only: bool = False
    top_k: int = 10

    # Cross-stock attention config
    num_cross_attention_layers: int = 2
    num_cross_attention_heads: int = 4

    # Prediction horizon index (0=1day, 1=5day, etc.)
    horizon_idx: int = 0


class StockEncoder(nn.Module):
    """
    Encodes a single stock's features into a fixed-size embedding.

    Reuses the modality-specific encoders from the multimodal model.
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.config = feature_config
        self.hidden_dim = hidden_dim

        # Scale output dimensions with hidden_dim
        fundamental_output_dim = max(64, hidden_dim // 4)
        technical_output_dim = max(128, hidden_dim // 2)
        news_output_dim = max(64, hidden_dim // 4)

        # Modality-specific encoders
        self.fundamental_encoder = FundamentalsEncoder(
            input_dim=feature_config.fundamental_dim,
            hidden_dim=hidden_dim,
            output_dim=fundamental_output_dim,
            dropout=dropout,
        )

        self.technical_encoder = TechnicalEncoder(
            input_dim=feature_config.technical_dim,
            hidden_dim=hidden_dim,
            output_dim=technical_output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self.news_encoder = NewsEncoder(
            input_dim=feature_config.news_dim,
            hidden_dim=hidden_dim,
            output_dim=news_output_dim,
            dropout=dropout,
            use_temporal_attention=True,
        )

        # Combined output dimension
        self.output_dim = fundamental_output_dim + technical_output_dim + news_output_dim

        # Final projection to hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _split_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split input tensor into modality-specific features."""
        # Fundamentals: simple slice
        fundamentals = x[:, :, self.config.fundamental_indices[0]:self.config.fundamental_indices[1]]

        # Technicals: exclude leaky features
        tech_start, tech_end = self.config.technical_indices
        tech_indices = [
            i for i in range(tech_start, tech_end)
            if i not in self.config.excluded_technical_indices
        ]
        technicals = x[:, :, tech_indices]

        # News: simple slice
        news = x[:, :, self.config.news_indices[0]:self.config.news_indices[1]]

        return {
            'fundamentals': fundamentals,
            'technicals': technicals,
            'news': news,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a stock's features into an embedding.

        Args:
            x: (batch, seq_len, features) input tensor

        Returns:
            (batch, hidden_dim) stock embedding
        """
        # Handle NaN/Inf
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Split by modality
        features = self._split_features(x)

        # Encode each modality
        fund_emb = self.fundamental_encoder(features['fundamentals'])  # (batch, fund_dim)
        _, tech_pooled = self.technical_encoder(features['technicals'])  # (batch, tech_dim)
        _, news_pooled = self.news_encoder(features['news'])  # (batch, news_dim)

        # Concatenate modality embeddings
        combined = torch.cat([fund_emb, tech_pooled, news_pooled], dim=-1)

        # Project to hidden_dim
        embedding = self.output_proj(combined)

        return embedding


class ContrastiveEncoderWrapper(nn.Module):
    """
    Wraps StockEncoder with contrastive learning capability.

    Used to pretrain the encoder on unlabeled data, then save
    just the encoder weights for downstream tasks.
    """

    def __init__(
        self,
        encoder: StockEncoder,
        projection_dim: int = 128,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

        # Projection head for contrastive learning
        # Use LayerNorm instead of BatchNorm to avoid issues with batch_size=1
        hidden_dim = encoder.hidden_dim
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

        # Augmentation parameters
        self.temporal_jitter = 5
        self.feature_mask_prob = 0.15
        self.noise_std = 0.05

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to input."""
        if not self.training:
            return x

        batch_size, seq_len, num_features = x.shape
        device = x.device

        # Temporal jittering
        if self.temporal_jitter > 0:
            shifts = torch.randint(-self.temporal_jitter, self.temporal_jitter + 1,
                                   (batch_size,), device=device)
            x_aug = torch.zeros_like(x)
            for i in range(batch_size):
                x_aug[i] = torch.roll(x[i], shifts=shifts[i].item(), dims=0)
            x = x_aug

        # Feature masking
        if self.feature_mask_prob > 0:
            mask = torch.rand(batch_size, 1, num_features, device=device) > self.feature_mask_prob
            x = x * mask.float()

        # Gaussian noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        return x

    def forward_contrastive(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive learning.

        Args:
            x: (batch, seq_len, features)

        Returns:
            z1, z2: (batch, projection_dim) normalized projections
        """
        # Create two augmented views
        x1 = self.augment(x)
        x2 = self.augment(x)

        # Encode
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        # Project and normalize
        z1 = F.normalize(self.projection_head(h1), dim=-1)
        z2 = F.normalize(self.projection_head(h2), dim=-1)

        return z1, z2

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE contrastive loss."""
        z1, z2 = self.forward_contrastive(x)
        batch_size = z1.shape[0]

        # Concatenate both views
        z = torch.cat([z1, z2], dim=0)

        # Similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask self-similarity
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))

        # Labels: positive pairs
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss


class DirectPricePredictor(nn.Module):
    """
    Direct price prediction model using pretrained encoder.

    Simple architecture:
    - StockEncoder (frozen or trainable)
    - MLP regression head
    - Predicts returns directly

    Trained with MSE loss, evaluated with IC.
    """

    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        hidden_dim: int = 256,
        num_encoder_layers: int = 4,
        num_encoder_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        num_pred_days: int = 1,  # Number of prediction horizons (1 or 4)
    ):
        super().__init__()

        self.config = feature_config or FeatureConfig()
        self.hidden_dim = hidden_dim
        self.num_pred_days = num_pred_days

        # Encoder
        self.encoder = StockEncoder(
            feature_config=self.config,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_encoder_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Regression head - outputs num_pred_days predictions
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_pred_days),
        )

        self._print_architecture()

    def _print_architecture(self):
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = sum(p.numel() for p in self.regression_head.parameters())

        print("\n" + "=" * 60)
        print("Direct Price Prediction Model")
        print("=" * 60)
        print(f"  Prediction horizons: {self.num_pred_days}")
        print(f"  Encoder params: {encoder_params:,}")
        print(f"  Head params: {head_params:,}")
        print(f"  Total params: {total_params:,}")
        print("=" * 60 + "\n")

    def load_encoder(self, encoder_path: str, freeze: bool = True) -> None:
        """Load pretrained encoder weights."""
        checkpoint = torch.load(encoder_path, map_location='cpu')

        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = {
                k.replace('stock_encoder.', ''): v
                for k, v in checkpoint['model_state_dict'].items()
                if k.startswith('stock_encoder.')
            }
        else:
            state_dict = checkpoint

        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: Missing keys: {missing[:3]}...")
        if unexpected:
            print(f"  Warning: Unexpected keys: {unexpected[:3]}...")

        print(f"Loaded encoder from: {encoder_path}")

        if freeze:
            self.freeze_encoder()
            print("  Encoder frozen for finetuning")

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor, return_tuple: bool = True):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, features)
            return_tuple: If True, returns (pred, confidence) for compatibility with
                          walk-forward training. If False, returns just pred.

        Returns:
            If return_tuple: (pred, confidence) where confidence is None
            If not return_tuple: (batch,) if num_pred_days=1, else (batch, num_pred_days)
        """
        embedding = self.encoder(x)  # (batch, hidden_dim)
        pred = self.regression_head(embedding)  # (batch, num_pred_days)
        if self.num_pred_days == 1:
            pred = pred.squeeze(-1)  # (batch,) for backwards compatibility

        if return_tuple:
            # Return (pred, confidence) for walk-forward training compatibility
            return pred, None
        return pred


def train_direct_predictor(
    model: DirectPricePredictor,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    device: torch.device = torch.device('cuda'),
) -> DirectPricePredictor:
    """
    Train direct price predictor.

    Uses MSE loss, reports IC metrics.
    """
    from scipy.stats import pearsonr, spearmanr

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)\n")

    best_val_ic = -float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_preds, train_targets = [], []
        total_loss = 0
        num_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for features, returns, mask in train_pbar:
            # features: (batch, 1, seq, feat) -> (batch, seq, feat)
            features = features.squeeze(1).to(device)
            returns = returns.squeeze(1).to(device)

            optimizer.zero_grad()
            preds = model(features, return_tuple=False)
            loss = F.mse_loss(preds, returns)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(returns.cpu().numpy())

            train_pbar.set_postfix({'loss': f'{total_loss/num_batches:.6f}'})

        scheduler.step()

        # Compute train IC
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        train_ic = pearsonr(train_preds, train_targets)[0]
        train_rank_ic = spearmanr(train_preds, train_targets)[0]

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0
        val_batches = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ")
        with torch.no_grad():
            for features, returns, mask in val_pbar:
                # Handle both single-stock and multi-stock formats
                # features: (batch, num_stocks, seq_len, feat_dim)
                # returns: (batch, num_stocks)
                # mask: (batch, num_stocks)
                batch_size, num_stocks = returns.shape[:2]

                # Flatten batch and stocks dimensions
                features = features.view(-1, features.shape[-2], features.shape[-1]).to(device)
                returns_flat = returns.view(-1).to(device)
                mask_flat = mask.view(-1)

                preds = model(features, return_tuple=False)  # (batch * num_stocks,)

                # Only compute loss on valid stocks
                valid_mask = mask_flat.to(device)
                if valid_mask.sum() > 0:
                    loss = F.mse_loss(preds[valid_mask], returns_flat[valid_mask])
                    val_loss += loss.item()
                    val_batches += 1

                    # Collect valid predictions
                    val_preds.extend(preds[valid_mask].cpu().numpy())
                    val_targets.extend(returns_flat[valid_mask].cpu().numpy())

                val_pbar.set_postfix({'loss': f'{val_loss/max(1,val_batches):.6f}'})

        # Compute val IC
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_ic = pearsonr(val_preds, val_targets)[0]
        val_rank_ic = spearmanr(val_preds, val_targets)[0]

        # Print summary
        improved = val_ic > best_val_ic
        marker = " *" if improved else ""
        print(f"  Train IC: {train_ic:+.4f} | Val IC: {val_ic:+.4f} | "
              f"Val Rank IC: {val_rank_ic:+.4f}{marker}")

        if improved:
            best_val_ic = val_ic
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nTraining complete! Best val IC: {best_val_ic:+.4f}")
    return model


def pretrain_encoder(
    data_path: str,
    prices_path: str,
    save_path: str,
    start_date: str = '2010-01-01',
    end_date: str = '2022-12-31',
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    seq_len: int = 64,
    num_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-4,
    temperature: float = 0.1,
    device: str = 'cuda',
    # Evaluation options
    eval_every: int = 0,  # Evaluate every N epochs (0 = only at end)
    eval_train_start: str = '2010-01-01',
    eval_train_end: str = '2018-12-31',
    eval_test_start: str = '2019-01-01',
    eval_test_end: str = '2020-12-31',
    eval_num_samples: int = 5000,
):
    """
    Pretrain stock encoder with contrastive learning and save weights.

    Args:
        data_path: Path to features HDF5
        prices_path: Path to prices HDF5
        save_path: Where to save encoder weights
        start_date: Training data start
        end_date: Training data end
        hidden_dim: Encoder hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        seq_len: Sequence length
        num_epochs: Pretraining epochs
        batch_size: Batch size
        lr: Learning rate
        temperature: InfoNCE temperature
        device: Device to train on
        eval_every: Run linear probe evaluation every N epochs (0 = only at end)
        eval_train_start/end: Date range for fitting linear probe
        eval_test_start/end: Date range for evaluation
        eval_num_samples: Number of samples for evaluation

    Returns:
        Path to saved encoder
    """
    import os

    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 60)
    print("ENCODER CONTRASTIVE PRETRAINING")
    print("=" * 60)
    print(f"Data: {start_date} to {end_date}")
    print(f"Model: hidden_dim={hidden_dim}, layers={num_layers}, heads={num_heads}")
    print(f"Training: epochs={num_epochs}, batch_size={batch_size}, lr={lr}")
    print(f"Temperature: {temperature}")
    if eval_every > 0:
        print(f"Evaluation: every {eval_every} epochs")
    else:
        print(f"Evaluation: at end only")
    print("=" * 60 + "\n")

    # Create dataset (reuse CrossSectionalDataset but we only need features)
    dataset = CrossSectionalDataset(
        dataset_path=data_path,
        prices_path=prices_path,
        start_date=start_date,
        end_date=end_date,
        seq_len=seq_len,
        num_stocks=1,  # We just need individual stocks
        pred_day=1,
        min_stocks_per_date=1,
    )

    # Create encoder
    encoder = StockEncoder(
        feature_config=FeatureConfig(),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=seq_len,
    )

    # Wrap with contrastive learning
    model = ContrastiveEncoderWrapper(
        encoder=encoder,
        projection_dim=128,
        temperature=temperature,
    ).to(device_obj)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # We need a custom collate that gives us individual stocks
    def collate_fn(batch):
        # batch is list of (features, returns, mask) where features is (num_stocks, seq, feat)
        # We want individual stocks
        features = torch.cat([b[0] for b in batch], dim=0)  # (total_stocks, seq, feat)
        return features

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"Dataset: {len(dataset)} dates")
    print(f"Training encoder with {sum(p.numel() for p in encoder.parameters()):,} parameters\n")

    def do_eval(encoder_module, epoch_num):
        """Wrapper to call the factored-out linear probe evaluation."""
        return run_linear_probe_eval(
            encoder=encoder_module,
            data_path=data_path,
            prices_path=prices_path,
            train_start=eval_train_start,
            train_end=eval_train_end,
            test_start=eval_test_start,
            test_end=eval_test_end,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device_obj,
            num_samples=eval_num_samples,
            num_stocks_per_date=50,  # Use 50 stocks for proper cross-sectional IC
            epoch_num=epoch_num,
            verbose=False,
        )

    # Track evaluation history
    eval_history = []

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for features in pbar:
            features = features.to(device_obj)

            optimizer.zero_grad()
            loss = model.compute_loss(features)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})

        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Run evaluation if requested
        if eval_every > 0 and (epoch + 1) % eval_every == 0:
            print(f"  Running linear probe evaluation...")
            metrics = do_eval(encoder, epoch + 1)
            eval_history.append(metrics)
            print(f"    Encoder IC: {metrics['encoder_ic']:+.4f} | "
                  f"Raw IC: {metrics['raw_ic']:+.4f} | "
                  f"Improvement: {metrics['improvement_ic']:+.4f}")

    # Save just the encoder weights
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'config': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'seq_len': seq_len,
        },
    }, save_path)

    print(f"\nEncoder saved to: {save_path}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL LINEAR PROBE EVALUATION")
    print("=" * 60)
    final_metrics = do_eval(encoder, num_epochs)
    eval_history.append(final_metrics)

    print(f"\n{'Method':<25} {'IC':>10} {'Rank IC':>10}")
    print("-" * 47)
    print(f"{'Raw Features (baseline)':<25} {final_metrics['raw_ic']:>10.4f} {final_metrics['raw_rank_ic']:>10.4f}")
    print(f"{'Pretrained Encoder':<25} {final_metrics['encoder_ic']:>10.4f} {final_metrics['encoder_rank_ic']:>10.4f}")
    print("-" * 47)
    print(f"{'Improvement':<25} {final_metrics['improvement_ic']:>+10.4f} {final_metrics['improvement_ic']:>+10.4f}")

    if final_metrics['improvement_ic'] > 0:
        print(f"\nEncoder learned useful representations (+{final_metrics['improvement_ic']:.4f} IC)")
    else:
        print(f"\nEncoder did not improve over raw features ({final_metrics['improvement_ic']:.4f} IC)")

    # Print history if we have intermediate evaluations
    if len(eval_history) > 1:
        print("\n" + "-" * 47)
        print("Evaluation History:")
        print(f"{'Epoch':<10} {'Encoder IC':>12} {'Improvement':>12}")
        for m in eval_history:
            print(f"{m['epoch']:<10} {m['encoder_ic']:>+12.4f} {m['improvement_ic']:>+12.4f}")

    print("=" * 60)

    dataset.close()

    return save_path


class CrossStockAttention(nn.Module):
    """
    Self-attention across stocks in a cross-section.

    Allows stocks to "exchange information" and compare to each other.
    For example, a stock might learn to downweight itself if there's
    a similar but better-performing stock in the cross-section.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        stock_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply self-attention across stocks.

        Args:
            stock_embeddings: (batch, num_stocks, hidden_dim)
            attention_mask: (batch, num_stocks) bool tensor, True = valid stock

        Returns:
            (batch, num_stocks, hidden_dim) contextualized embeddings
        """
        x = stock_embeddings

        # Convert mask to attention format if provided
        # TransformerEncoderLayer expects (batch, seq_len) for src_key_padding_mask
        # True = IGNORE this position
        if attention_mask is not None:
            # Invert: our mask is True=valid, transformer wants True=ignore
            padding_mask = ~attention_mask
        else:
            padding_mask = None

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        x = self.final_norm(x)

        return x


class CrossSectionalSelector(nn.Module):
    """
    Cross-sectional stock selection model.

    Takes N stocks from the same date, outputs portfolio weights.

    Architecture:
        Input: (batch, num_stocks, seq_len, features)
               or list of N tensors each (batch, seq_len, features)

        1. Per-stock encoding: shared StockEncoder
        2. Cross-stock attention: stocks exchange information
        3. Score head: each stock gets a scalar logit
        4. Softmax: convert to portfolio weights

    Training:
        - Target: softmax(returns / temperature) - soft labels based on actual returns
        - Loss: Cross-entropy between predicted and target distributions

    Inference:
        - Output portfolio weights directly
        - Theoretical return = sum(weight_i * return_i)
    """

    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        hidden_dim: int = 256,
        num_encoder_layers: int = 4,
        num_encoder_heads: int = 4,
        num_cross_attention_layers: int = 2,
        num_cross_attention_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        target_temperature: float = 1.0,
    ):
        super().__init__()

        self.config = feature_config or FeatureConfig()
        self.hidden_dim = hidden_dim
        self.target_temperature = target_temperature

        # Per-stock encoder (shared across all stocks)
        self.stock_encoder = StockEncoder(
            feature_config=self.config,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_encoder_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Cross-stock attention
        self.cross_attention = CrossStockAttention(
            hidden_dim=hidden_dim,
            num_heads=num_cross_attention_heads,
            num_layers=num_cross_attention_layers,
            dropout=dropout,
        )

        # Score head: projects each stock embedding to a scalar logit
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._print_architecture()

    def _print_architecture(self):
        """Print model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())

        print("\n" + "=" * 60)
        print("Cross-Sectional Stock Selection Model")
        print("=" * 60)
        print(f"\nHidden dimension: {self.hidden_dim}")
        print(f"Target temperature: {self.target_temperature}")
        print(f"\nTotal Parameters: {total_params:,}")
        print("=" * 60 + "\n")

    def load_encoder(self, encoder_path: str, freeze: bool = True) -> None:
        """
        Load pretrained encoder weights.

        Args:
            encoder_path: Path to saved encoder checkpoint
            freeze: Whether to freeze encoder weights (for finetuning)
        """
        checkpoint = torch.load(encoder_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        elif 'model_state_dict' in checkpoint:
            # Full model checkpoint - extract encoder keys
            state_dict = {
                k.replace('stock_encoder.', ''): v
                for k, v in checkpoint['model_state_dict'].items()
                if k.startswith('stock_encoder.')
            }
        else:
            state_dict = checkpoint

        # Load weights
        missing, unexpected = self.stock_encoder.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: Missing keys in encoder: {missing[:5]}...")
        if unexpected:
            print(f"  Warning: Unexpected keys in encoder: {unexpected[:5]}...")

        print(f"Loaded encoder from: {encoder_path}")

        # Optionally freeze encoder
        if freeze:
            self.freeze_encoder()
            print("  Encoder weights frozen for finetuning")

    def freeze_encoder(self) -> None:
        """Freeze encoder weights for finetuning."""
        for param in self.stock_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights."""
        for param in self.stock_encoder.parameters():
            param.requires_grad = True

    def encode_stocks(
        self,
        stock_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode all stocks and apply cross-attention.

        Args:
            stock_features: (batch, num_stocks, seq_len, features)
            attention_mask: (batch, num_stocks) bool, True = valid stock

        Returns:
            (batch, num_stocks, hidden_dim) contextualized stock embeddings
        """
        batch_size, num_stocks, seq_len, num_features = stock_features.shape

        # Reshape to encode all stocks at once
        # (batch * num_stocks, seq_len, features)
        flat_features = stock_features.view(batch_size * num_stocks, seq_len, num_features)

        # Encode each stock
        # (batch * num_stocks, hidden_dim)
        flat_embeddings = self.stock_encoder(flat_features)

        # Reshape back
        # (batch, num_stocks, hidden_dim)
        stock_embeddings = flat_embeddings.view(batch_size, num_stocks, -1)

        # Apply cross-stock attention
        contextualized = self.cross_attention(stock_embeddings, attention_mask)

        return contextualized

    def forward(
        self,
        stock_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            stock_features: (batch, num_stocks, seq_len, features)
            attention_mask: (batch, num_stocks) bool, True = valid stock
            return_embeddings: if True, also return stock embeddings

        Returns:
            logits: (batch, num_stocks) raw scores
            weights: (batch, num_stocks) softmax portfolio weights
            embeddings: (batch, num_stocks, hidden_dim) if return_embeddings=True
        """
        # Get contextualized embeddings
        embeddings = self.encode_stocks(stock_features, attention_mask)

        # Compute logits
        logits = self.score_head(embeddings).squeeze(-1)  # (batch, num_stocks)

        # Apply mask to logits if provided
        if attention_mask is not None:
            # Set invalid stocks to -inf so softmax gives them 0 weight
            logits = logits.masked_fill(~attention_mask, float('-inf'))

        # Softmax to get portfolio weights
        weights = F.softmax(logits, dim=-1)

        if return_embeddings:
            return logits, weights, embeddings

        return logits, weights

    def compute_soft_targets(
        self,
        returns: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute soft targets from returns.

        target = softmax(returns / temperature)

        This creates a probability distribution where higher-return stocks
        get more weight. Temperature controls peakiness:
        - Low temp (0.1): almost one-hot on best stock
        - High temp (10.0): nearly uniform
        - temp=1.0: balanced

        Args:
            returns: (batch, num_stocks) actual returns
            attention_mask: (batch, num_stocks) bool, True = valid stock

        Returns:
            (batch, num_stocks) soft target distribution
        """
        # Scale returns by temperature
        scaled_returns = returns / self.target_temperature

        # Mask invalid stocks
        if attention_mask is not None:
            scaled_returns = scaled_returns.masked_fill(~attention_mask, float('-inf'))

        # Softmax to get target distribution
        targets = F.softmax(scaled_returns, dim=-1)

        return targets

    def compute_loss(
        self,
        logits: torch.Tensor,
        returns: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute cross-entropy loss between predicted and target distributions.

        Args:
            logits: (batch, num_stocks) model output logits
            returns: (batch, num_stocks) actual returns
            attention_mask: (batch, num_stocks) bool, True = valid stock

        Returns:
            loss: scalar loss value
            metrics: dict with additional metrics for logging
        """
        # Compute soft targets
        targets = self.compute_soft_targets(returns, attention_mask)

        # Compute predicted distribution
        if attention_mask is not None:
            logits = logits.masked_fill(~attention_mask, float('-inf'))
        log_probs = F.log_softmax(logits, dim=-1)

        # Cross-entropy: -sum(target * log_prob)
        # This is KL divergence up to a constant (target entropy)
        loss = -torch.sum(targets * log_probs, dim=-1).mean()

        # Compute metrics
        with torch.no_grad():
            weights = F.softmax(logits, dim=-1)

            # Theoretical portfolio return
            portfolio_return = (weights * returns).sum(dim=-1).mean()

            # Best possible return (if we picked the best stock)
            if attention_mask is not None:
                masked_returns = returns.masked_fill(~attention_mask, float('-inf'))
            else:
                masked_returns = returns
            best_return = masked_returns.max(dim=-1)[0].mean()

            # Average return (equal weight baseline)
            if attention_mask is not None:
                num_valid = attention_mask.float().sum(dim=-1, keepdim=True)
                equal_weights = attention_mask.float() / num_valid.clamp(min=1)
            else:
                equal_weights = torch.ones_like(returns) / returns.shape[-1]
            avg_return = (equal_weights * returns).sum(dim=-1).mean()

            # Weight entropy (how concentrated is the portfolio?)
            weight_entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1).mean()
            max_entropy = math.log(returns.shape[-1])  # uniform distribution

            # Top-k accuracy: what fraction of weight goes to actual top-k stocks?
            k = min(5, returns.shape[-1])
            top_k_indices = returns.topk(k, dim=-1).indices
            top_k_weight = torch.gather(weights, -1, top_k_indices).sum(dim=-1).mean()

        metrics = {
            'portfolio_return': portfolio_return.item(),
            'best_return': best_return.item(),
            'avg_return': avg_return.item(),
            'excess_return': (portfolio_return - avg_return).item(),
            'weight_entropy': weight_entropy.item(),
            'entropy_ratio': (weight_entropy / max_entropy).item(),
            'top_5_weight': top_k_weight.item(),
        }

        return loss, metrics


class CrossSectionalDataset(torch.utils.data.Dataset):
    """
    Dataset that returns batches of stocks from the same trading date.

    Each sample is a cross-section: (num_stocks, seq_len, features) + returns

    This enables training the CrossSectionalSelector model.
    """

    def __init__(
        self,
        dataset_path: str,
        prices_path: str,
        start_date: str,
        end_date: str,
        seq_len: int = 256,
        num_stocks: int = 32,
        pred_day: int = 1,  # Which prediction horizon to use (trading days)
        min_stocks_per_date: int = 20,
        seed: int = 42,
    ):
        import h5py

        self.dataset_path = dataset_path
        self.prices_path = prices_path
        self.seq_len = seq_len
        self.num_stocks = num_stocks
        self.pred_day = pred_day
        self.seed = seed

        # Open HDF5 files
        self.h5f = h5py.File(dataset_path, 'r')
        self.prices_h5f = h5py.File(prices_path, 'r')

        # Get common tickers
        feature_tickers = set(self.h5f.keys())
        price_tickers = set(self.prices_h5f.keys())
        self.tickers = sorted(feature_tickers & price_tickers)

        # Build date-to-tickers mapping
        # For each valid date, store which tickers have data
        self.date_to_tickers = {}
        self._build_date_index(start_date, end_date, min_stocks_per_date)

        # List of valid dates
        self.dates = sorted(self.date_to_tickers.keys())

        print(f"CrossSectionalDataset: {len(self.dates)} dates, "
              f"{num_stocks} stocks per sample")

    def _get_trading_dates(self, ticker: str) -> List[str]:
        """Get sorted trading dates for a ticker."""
        return sorted([d.decode('utf-8') for d in self.prices_h5f[ticker]['dates'][:]])

    def _build_date_index(self, start_date: str, end_date: str, min_stocks: int):
        """Build mapping from dates to available tickers."""
        from collections import defaultdict

        date_ticker_map = defaultdict(list)

        for ticker in self.tickers:
            # Get trading dates
            trading_dates = self._get_trading_dates(ticker)
            trading_set = set(trading_dates)

            # Get feature dates
            feature_dates = [d.decode('utf-8') for d in self.h5f[ticker]['dates'][:]]
            num_features = len(feature_dates)

            # For each valid position
            for i in range(self.seq_len, num_features):
                current_date = feature_dates[i - 1]

                # Check date range
                if current_date < start_date or current_date > end_date:
                    continue

                # Check if trading day
                if current_date not in trading_set:
                    continue

                # Check if future date exists
                try:
                    trading_idx = trading_dates.index(current_date)
                    future_idx = trading_idx + self.pred_day
                    if future_idx >= len(trading_dates):
                        continue
                except ValueError:
                    continue

                # This date-ticker pair is valid
                date_ticker_map[current_date].append((ticker, i))

        # Filter dates with enough tickers
        for date, ticker_list in date_ticker_map.items():
            if len(ticker_list) >= min_stocks:
                self.date_to_tickers[date] = ticker_list

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a cross-sectional sample.

        Returns:
            features: (num_stocks, seq_len, num_features)
            returns: (num_stocks,) returns for each stock
            mask: (num_stocks,) bool, True = valid stock
        """
        date = self.dates[idx]
        available_tickers = self.date_to_tickers[date]

        # Sample num_stocks tickers (with replacement if needed)
        rng = np.random.RandomState(self.seed + idx)
        if len(available_tickers) >= self.num_stocks:
            selected_indices = rng.choice(len(available_tickers), self.num_stocks, replace=False)
        else:
            selected_indices = rng.choice(len(available_tickers), self.num_stocks, replace=True)

        selected = [available_tickers[i] for i in selected_indices]

        # Load features and compute returns
        features_list = []
        returns_list = []

        # Expected feature dimension
        expected_features = 1123  # FeatureConfig.total_features

        for ticker, end_idx in selected:
            # Load features
            start_idx = end_idx - self.seq_len
            feats = self.h5f[ticker]['features'][start_idx:end_idx]

            # Pad to expected dimensions if needed
            if feats.shape[0] < self.seq_len:
                pad_len = self.seq_len - feats.shape[0]
                feats = np.pad(feats, ((pad_len, 0), (0, 0)), mode='constant')
            if feats.shape[1] < expected_features:
                pad_width = expected_features - feats.shape[1]
                feats = np.pad(feats, ((0, 0), (0, pad_width)), mode='constant')
            elif feats.shape[1] > expected_features:
                feats = feats[:, :expected_features]

            features_list.append(feats)

            # Compute return
            feature_dates = [d.decode('utf-8') for d in self.h5f[ticker]['dates'][:]]
            current_date = feature_dates[end_idx - 1]

            trading_dates = self._get_trading_dates(ticker)
            trading_idx = trading_dates.index(current_date)
            future_idx = trading_idx + self.pred_day
            future_date = trading_dates[future_idx]

            # Get prices
            prices = self.prices_h5f[ticker]['prices'][:]
            price_dates = [d.decode('utf-8') for d in self.prices_h5f[ticker]['dates'][:]]
            current_price = prices[price_dates.index(current_date)]
            future_price = prices[price_dates.index(future_date)]

            ret = (future_price / current_price) - 1.0
            ret = np.clip(ret, -0.5, 0.5)  # Clip extreme returns
            returns_list.append(ret)

        # Stack into tensors
        features = np.stack(features_list, axis=0).astype(np.float32)
        returns = np.array(returns_list, dtype=np.float32)
        mask = np.ones(self.num_stocks, dtype=bool)

        return (
            torch.from_numpy(features),
            torch.from_numpy(returns),
            torch.from_numpy(mask),
        )

    def close(self):
        """Close HDF5 files."""
        if self.h5f is not None:
            self.h5f.close()
        if self.prices_h5f is not None:
            self.prices_h5f.close()


def train_cross_sectional(
    model: CrossSectionalSelector,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    device: torch.device = torch.device('cuda'),
    log_interval: int = 50,
) -> CrossSectionalSelector:
    """
    Train cross-sectional selector model.

    Args:
        model: CrossSectionalSelector instance
        train_loader: DataLoader yielding (features, returns, mask) tuples
        val_loader: Validation DataLoader
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: AdamW weight decay
        device: Device to train on
        log_interval: Log every N batches

    Returns:
        Trained model
    """
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\n{'='*60}")
    print("Cross-Sectional Training")
    print(f"{'='*60}")
    print(f"Epochs: {num_epochs}, LR: {lr}, Device: {device}")
    print(f"{'='*60}\n")

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            leave=True,
            dynamic_ncols=True,
        )

        for features, returns, mask in train_pbar:
            features = features.to(device)
            returns = returns.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            logits, weights = model(features, mask)
            loss, metrics = model.compute_loss(logits, returns, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            num_batches += 1

            # Update progress bar with running averages
            avg_loss = total_loss / num_batches
            avg_ret = total_metrics['portfolio_return'] / num_batches
            excess_ret = total_metrics['excess_return'] / num_batches
            top5_wt = total_metrics['top_5_weight'] / num_batches

            train_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'ret': f'{avg_ret*100:+.4f}%',
                'excess': f'{excess_ret*10000:+.4f}bp',
                'top5': f'{top5_wt:.2%}',
            })

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {}
        val_batches = 0

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ",
            leave=True,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for features, returns, mask in val_pbar:
                features = features.to(device)
                returns = returns.to(device)
                mask = mask.to(device)

                logits, weights = model(features, mask)
                loss, metrics = model.compute_loss(logits, returns, mask)

                val_loss += loss.item()
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v
                val_batches += 1

                # Update progress bar
                avg_val_loss = val_loss / val_batches
                avg_val_ret = val_metrics['portfolio_return'] / val_batches
                val_excess = val_metrics['excess_return'] / val_batches
                val_top5 = val_metrics['top_5_weight'] / val_batches

                val_pbar.set_postfix({
                    'loss': f'{avg_val_loss:.4f}',
                    'ret': f'{avg_val_ret*100:+.4f}%',
                    'excess': f'{val_excess*10000:+.4f}bp',
                    'top5': f'{val_top5:.2%}',
                })

        avg_val_loss = val_loss / max(val_batches, 1)
        avg_val_ret = val_metrics.get('portfolio_return', 0) / max(val_batches, 1)
        avg_val_excess = val_metrics.get('excess_return', 0) / max(val_batches, 1)
        avg_val_top5 = val_metrics.get('top_5_weight', 0) / max(val_batches, 1)

        # Print epoch summary
        improved = avg_val_loss < best_val_loss
        marker = " *" if improved else ""
        print(f"  Summary: Train {total_loss/num_batches:.4f} | "
              f"Val {avg_val_loss:.4f} | "
              f"Excess {avg_val_excess*10000:+.4f}bp | "
              f"Top5 {avg_val_top5:.2%}{marker}")

        # Save best model
        if improved:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    return model


def main():
    """Main training script for cross-sectional stock selection."""
    import argparse
    import os
    import json
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description='Train cross-sectional stock selection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument('--data', type=str, default='all_complete_dataset.h5',
                       help='Path to features HDF5 file')
    parser.add_argument('--prices', type=str, default='actual_prices_clean.h5',
                       help='Path to prices HDF5 file')
    parser.add_argument('--train-start', type=str, default='2010-01-01',
                       help='Training start date')
    parser.add_argument('--train-end', type=str, default='2020-12-31',
                       help='Training end date')
    parser.add_argument('--val-start', type=str, default='2021-01-01',
                       help='Validation start date')
    parser.add_argument('--val-end', type=str, default='2022-12-31',
                       help='Validation end date')

    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num-encoder-layers', type=int, default=4,
                       help='Number of per-stock encoder layers')
    parser.add_argument('--num-encoder-heads', type=int, default=4,
                       help='Number of attention heads in encoder')
    parser.add_argument('--num-cross-layers', type=int, default=2,
                       help='Number of cross-stock attention layers')
    parser.add_argument('--num-cross-heads', type=int, default=4,
                       help='Number of cross-stock attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')

    # Cross-sectional config
    parser.add_argument('--num-stocks', type=int, default=32,
                       help='Number of stocks per cross-section')
    parser.add_argument('--num-eval-stocks', type=int, default=100,
                       help='Number of stocks per date for evaluation IC (regression mode)')
    parser.add_argument('--seq-len', type=int, default=64,
                       help='Sequence length (trading days of history)')
    parser.add_argument('--pred-day', type=int, default=1,
                       help='Prediction horizon in trading days')
    parser.add_argument('--target-temperature', type=float, default=1.0,
                       help='Temperature for soft targets (lower = more peaky)')
    parser.add_argument('--min-stocks', type=int, default=20,
                       help='Minimum stocks per date to include')

    # Prediction mode
    parser.add_argument('--pred-mode', type=str, default='selection',
                       choices=['selection', 'regression'],
                       help='Prediction mode: selection (softmax over stocks) or regression (direct price prediction)')

    # Training
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (number of cross-sections per batch)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/cross_sectional',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-interval', type=int, default=20,
                       help='Log every N batches')

    # Encoder pretraining
    parser.add_argument('--pretrain-encoder', action='store_true',
                       help='Run encoder contrastive pretraining instead of full training')
    parser.add_argument('--save-encoder', type=str, default=None,
                       help='Path to save pretrained encoder (only with --pretrain-encoder)')
    parser.add_argument('--contrastive-epochs', type=int, default=10,
                       help='Number of contrastive pretraining epochs')
    parser.add_argument('--contrastive-lr', type=float, default=1e-4,
                       help='Learning rate for contrastive pretraining')
    parser.add_argument('--contrastive-temperature', type=float, default=0.1,
                       help='Temperature for InfoNCE loss')
    parser.add_argument('--eval-every', type=int, default=0,
                       help='Run linear probe evaluation every N epochs during pretraining (0 = only at end)')
    parser.add_argument('--eval-train-start', type=str, default='2010-01-01',
                       help='Start date for linear probe training data')
    parser.add_argument('--eval-train-end', type=str, default='2018-12-31',
                       help='End date for linear probe training data')
    parser.add_argument('--eval-test-start', type=str, default='2019-01-01',
                       help='Start date for linear probe test data')
    parser.add_argument('--eval-test-end', type=str, default='2020-12-31',
                       help='End date for linear probe test data')

    # Loading pretrained encoder
    parser.add_argument('--load-encoder', type=str, default=None,
                       help='Path to pretrained encoder weights')
    parser.add_argument('--no-freeze-encoder', action='store_true',
                       help='Do not freeze encoder when loading pretrained weights')

    # Encoder evaluation
    parser.add_argument('--evaluate-encoder', type=str, default=None,
                       help='Path to encoder checkpoint to evaluate (compares IC vs raw features)')
    parser.add_argument('--eval-num-samples', type=int, default=10000,
                       help='Number of samples for encoder evaluation')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Handle encoder pretraining mode
    if args.pretrain_encoder:
        save_path = args.save_encoder or os.path.join(args.checkpoint_dir, 'pretrained_encoder.pt')
        pretrain_encoder(
            data_path=args.data,
            prices_path=args.prices,
            save_path=save_path,
            start_date=args.train_start,
            end_date=args.train_end,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_encoder_layers,
            num_heads=args.num_encoder_heads,
            seq_len=args.seq_len,
            num_epochs=args.contrastive_epochs,
            batch_size=args.batch_size,
            lr=args.contrastive_lr,
            temperature=args.contrastive_temperature,
            device=args.device,
            eval_every=args.eval_every,
            eval_train_start=args.eval_train_start,
            eval_train_end=args.eval_train_end,
            eval_test_start=args.eval_test_start,
            eval_test_end=args.eval_test_end,
            eval_num_samples=args.eval_num_samples,
        )
        print("\nEncoder pretraining complete!")
        print(f"To train with this encoder, use: --load-encoder {save_path}")
        print(f"To evaluate encoder quality, use: --evaluate-encoder {save_path}")
        return

    # Handle encoder evaluation mode
    if args.evaluate_encoder:
        evaluate_encoder(
            encoder_path=args.evaluate_encoder,
            data_path=args.data,
            prices_path=args.prices,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.val_start,
            test_end=args.val_end,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=args.device,
            num_samples=args.eval_num_samples,
        )
        return

    # Print configuration
    print("\n" + "=" * 70)
    if args.pred_mode == 'regression':
        print("DIRECT PRICE PREDICTION TRAINING")
    else:
        print("CROSS-SECTIONAL STOCK SELECTION TRAINING")
    print("=" * 70)
    print(f"\nData:")
    print(f"  Features: {args.data}")
    print(f"  Prices: {args.prices}")
    print(f"  Train: {args.train_start} to {args.train_end}")
    print(f"  Val: {args.val_start} to {args.val_end}")
    print(f"\nModel:")
    print(f"  Mode: {args.pred_mode}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Encoder: {args.num_encoder_layers} layers, {args.num_encoder_heads} heads")
    if args.pred_mode == 'selection':
        print(f"  Cross-attention: {args.num_cross_layers} layers, {args.num_cross_heads} heads")
        print(f"\nCross-sectional:")
        print(f"  Stocks per sample: {args.num_stocks}")
        print(f"  Target temperature: {args.target_temperature}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Prediction horizon: {args.pred_day} day(s)")
    print(f"\nTraining:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    if args.load_encoder:
        print(f"\nPretrained encoder:")
        print(f"  Path: {args.load_encoder}")
        print(f"  Freeze: {not args.no_freeze_encoder}")
    print("=" * 70 + "\n")

    # For regression mode, use num_stocks=1 for training
    num_stocks = 1 if args.pred_mode == 'regression' else args.num_stocks
    # For evaluation, use more stocks per date to compute meaningful IC
    num_eval_stocks = args.num_eval_stocks if args.pred_mode == 'regression' else args.num_stocks

    # Create datasets
    print("Creating training dataset...")
    train_dataset = CrossSectionalDataset(
        dataset_path=args.data,
        prices_path=args.prices,
        start_date=args.train_start,
        end_date=args.train_end,
        seq_len=args.seq_len,
        num_stocks=num_stocks,
        pred_day=args.pred_day,
        min_stocks_per_date=1 if args.pred_mode == 'regression' else args.min_stocks,
        seed=args.seed,
    )

    print(f"Creating validation dataset (num_eval_stocks={num_eval_stocks})...")
    val_dataset = CrossSectionalDataset(
        dataset_path=args.data,
        prices_path=args.prices,
        start_date=args.val_start,
        end_date=args.val_end,
        seq_len=args.seq_len,
        num_stocks=num_eval_stocks,
        pred_day=args.pred_day,
        min_stocks_per_date=min(10, num_eval_stocks) if args.pred_mode == 'regression' else args.min_stocks,
        seed=args.seed + 1000,  # Different seed for val
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # HDF5 doesn't support multiprocessing
        pin_memory=True,
    )

    # For regression mode with many eval stocks, use smaller batch size
    val_batch_size = max(1, args.batch_size // num_eval_stocks) if args.pred_mode == 'regression' else args.batch_size
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.pred_mode == 'regression':
        # Direct price prediction mode
        model = DirectPricePredictor(
            feature_config=FeatureConfig(),
            hidden_dim=args.hidden_dim,
            num_encoder_layers=args.num_encoder_layers,
            num_encoder_heads=args.num_encoder_heads,
            dropout=args.dropout,
            max_seq_len=args.seq_len,
        )

        # Load pretrained encoder if specified
        if args.load_encoder:
            print(f"\nLoading pretrained encoder from: {args.load_encoder}")
            model.load_encoder(
                encoder_path=args.load_encoder,
                freeze=not args.no_freeze_encoder,
            )

        # Train
        model = train_direct_predictor(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )

        # Save final model
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'pred_mode': 'regression',
                'hidden_dim': args.hidden_dim,
                'num_encoder_layers': args.num_encoder_layers,
                'num_encoder_heads': args.num_encoder_heads,
                'seq_len': args.seq_len,
                'pred_day': args.pred_day,
            },
            'args': vars(args),
        }, checkpoint_path)
        print(f"\nSaved model to: {checkpoint_path}")

    else:
        # Cross-sectional selection mode
        model = CrossSectionalSelector(
            feature_config=FeatureConfig(),
            hidden_dim=args.hidden_dim,
            num_encoder_layers=args.num_encoder_layers,
            num_encoder_heads=args.num_encoder_heads,
            num_cross_attention_layers=args.num_cross_layers,
            num_cross_attention_heads=args.num_cross_heads,
            dropout=args.dropout,
            max_seq_len=args.seq_len,
            target_temperature=args.target_temperature,
        )

        # Load pretrained encoder if specified
        if args.load_encoder:
            print(f"\nLoading pretrained encoder from: {args.load_encoder}")
            model.load_encoder(
                encoder_path=args.load_encoder,
                freeze=not args.no_freeze_encoder,
            )

            # Print trainable parameter count
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                  f"({100*trainable_params/total_params:.1f}%)\n")

        # Train
        model = train_cross_sectional(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            log_interval=args.log_interval,
        )

        # Save final model
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'pred_mode': 'selection',
                'hidden_dim': args.hidden_dim,
                'num_encoder_layers': args.num_encoder_layers,
                'num_encoder_heads': args.num_encoder_heads,
                'num_cross_attention_layers': args.num_cross_layers,
                'num_cross_attention_heads': args.num_cross_heads,
                'num_stocks': args.num_stocks,
                'seq_len': args.seq_len,
                'pred_day': args.pred_day,
                'target_temperature': args.target_temperature,
            },
            'args': vars(args),
        }, checkpoint_path)
        print(f"\nSaved model to: {checkpoint_path}")

    # Save training config
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved config to: {config_path}")

    # Cleanup
    train_dataset.close()
    val_dataset.close()

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
