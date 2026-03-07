#!/usr/bin/env python3
"""
Multi-Modal Stock Prediction Model

Processes different feature types through separate encoders:
1. Fundamentals (quarterly, static) → MLP encoder
2. Technicals (daily, sequential) → Temporal Transformer
3. News Embeddings (sparse, event-driven) → MLP with learned "no news" token
4. Cross-Modal Fusion → Combines modality representations

This architecture addresses the issue of concatenating all features into tokens,
which loses the inherent structure of different data types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import math


@dataclass
class FeatureConfig:
    """Configuration for feature indices in the input tensor.

    Based on actual dataset analysis (1123 features):
    - Features 0-223: Fundamentals (quarterly, forward-filled, low daily variance)
    - Features 224-352: Technical indicators (change 50%+ of days)
    - Features 353-1122: News embeddings (mostly sparse/zeros)

    IMPORTANT: Multiple technical features are excluded because they correlate
    with same-day return (|r| > 0.3). Including these would be data leakage.
    The most critical are 225 and 254 (both r=0.9468 with same-day return).
    """
    # Total features in dataset
    total_features: int = 1123

    # Feature ranges (inclusive start, exclusive end)
    fundamental_indices: Tuple[int, int] = (0, 224)       # Quarterly metrics, forward-filled
    technical_indices: Tuple[int, int] = (224, 353)       # Daily technicals, momentum, volatility
    news_indices: Tuple[int, int] = (353, 1123)           # News embeddings (770-dim, mostly sparse)

    # Features to EXCLUDE from technicals (same-day return leakage)
    # Only exclude features that ARE the same-day return (r > 0.8)
    # 225: r=+0.9998 (z-scored same-day return - THIS IS THE ACTUAL RETURN)
    # 254: r=+0.9998 (identical to 225)
    # 229: r=+0.8332 (highly correlated, likely derived from same-day return)
    excluded_technical_indices: Tuple[int, ...] = (225, 254, 229)

    @property
    def fundamental_dim(self) -> int:
        return self.fundamental_indices[1] - self.fundamental_indices[0]

    @property
    def technical_dim(self) -> int:
        """Technical dimension AFTER excluding leaky features."""
        base_dim = self.technical_indices[1] - self.technical_indices[0]
        # Count how many excluded indices fall within technical range
        excluded_count = sum(
            1 for idx in self.excluded_technical_indices
            if self.technical_indices[0] <= idx < self.technical_indices[1]
        )
        return base_dim - excluded_count

    @property
    def news_dim(self) -> int:
        return self.news_indices[1] - self.news_indices[0]


class FundamentalsEncoder(nn.Module):
    """
    Encodes quarterly fundamental data.

    Since fundamentals are forward-filled and don't change daily,
    we only need to process them once (not per-timestep).
    Uses the last timestep's values.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, fundamental_dim) - fundamental features over time

        Returns:
            (batch, output_dim) - single embedding per sample
        """
        # Use last timestep (most recent fundamentals)
        x_last = x[:, -1, :]  # (batch, fundamental_dim)
        return self.encoder(x_last)


class TechnicalEncoder(nn.Module):
    """
    Encodes daily technical indicators using a Transformer.

    Technical features change daily and have sequential patterns
    (momentum, mean-reversion, trends).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, technical_dim) - technical features over time

        Returns:
            sequence: (batch, seq_len, output_dim) - per-timestep representations
            pooled: (batch, output_dim) - aggregated representation
        """
        batch_size, seq_len, _ = x.shape

        # Project to hidden dim
        h = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Add positional encoding
        h = h + self.pos_encoding[:, :seq_len, :]

        # Transform
        h = self.transformer(h)  # (batch, seq_len, hidden_dim)

        # Project to output
        sequence = self.output_proj(h)  # (batch, seq_len, output_dim)

        # Mean pooling for aggregated representation
        pooled = sequence.mean(dim=1)  # (batch, output_dim)

        return sequence, pooled


class NewsEncoder(nn.Module):
    """
    Encodes news embeddings with special handling for missing news.

    News is sparse (99%+ zeros). We use:
    - A learned "no news" embedding when news is missing
    - MLP to project non-zero news to a smaller dimension
    - Optionally: attention over news history
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 64,
        dropout: float = 0.1,
        use_temporal_attention: bool = True,
        num_attention_heads: int = 4,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.use_temporal_attention = use_temporal_attention

        # Learned embedding for "no news" case
        self.no_news_embedding = nn.Parameter(torch.randn(output_dim) * 0.02)

        # News projection
        self.news_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Optional: temporal attention over news history
        if use_temporal_attention:
            self.news_attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, news_dim) - news embeddings over time

        Returns:
            sequence: (batch, seq_len, output_dim) - per-timestep news representations
            pooled: (batch, output_dim) - aggregated news representation
        """
        batch_size, seq_len, news_dim = x.shape

        # Detect missing news (all zeros or very small values)
        news_magnitude = x.abs().sum(dim=-1)  # (batch, seq_len)
        has_news = news_magnitude > 0.01  # Boolean mask

        # Project all news through MLP
        projected = self.news_proj(x)  # (batch, seq_len, output_dim)

        # Replace zero-news timesteps with learned "no news" embedding
        no_news = self.no_news_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, output_dim)
        no_news = no_news.expand(batch_size, seq_len, -1)

        # Mask: where no news, use no_news_embedding
        has_news_expanded = has_news.unsqueeze(-1)  # (batch, seq_len, 1)
        sequence = torch.where(has_news_expanded, projected, no_news)

        if self.use_temporal_attention:
            # Self-attention over news sequence
            attn_out, _ = self.news_attention(sequence, sequence, sequence)
            sequence = self.attention_norm(sequence + attn_out)

        # Weighted pooling: weight by whether news exists
        # This gives more weight to timesteps with actual news
        weights = has_news.float() + 0.1  # Small weight even for no-news
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (sequence * weights.unsqueeze(-1)).sum(dim=1)

        return sequence, pooled


class CrossModalFusion(nn.Module):
    """
    Fuses representations from different modalities.

    Uses cross-attention to let the technical sequence attend to
    fundamental and news context.
    """

    def __init__(
        self,
        technical_dim: int,
        fundamental_dim: int,
        news_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project all modalities to same dimension for attention
        self.tech_proj = nn.Linear(technical_dim, hidden_dim)
        self.fund_proj = nn.Linear(fundamental_dim, hidden_dim)
        self.news_proj = nn.Linear(news_dim, hidden_dim)

        # Cross-attention: technical attends to fundamentals + news
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        technical_seq: torch.Tensor,  # (batch, seq_len, tech_dim)
        fundamental_emb: torch.Tensor,  # (batch, fund_dim)
        news_emb: torch.Tensor,  # (batch, news_dim)
    ) -> torch.Tensor:
        """
        Returns:
            fused: (batch, hidden_dim) - fused representation
        """
        batch_size, seq_len, _ = technical_seq.shape

        # Project to common dimension
        tech = self.tech_proj(technical_seq)  # (batch, seq_len, hidden)
        fund = self.fund_proj(fundamental_emb).unsqueeze(1)  # (batch, 1, hidden)
        news = self.news_proj(news_emb).unsqueeze(1)  # (batch, 1, hidden)

        # Concatenate fund and news as context
        context = torch.cat([fund, news], dim=1)  # (batch, 2, hidden)

        # Technical sequence attends to context
        # Use mean of technical sequence as query
        tech_pooled = tech.mean(dim=1, keepdim=True)  # (batch, 1, hidden)

        attn_out, _ = self.cross_attention(
            query=tech_pooled,
            key=context,
            value=context,
        )  # (batch, 1, hidden)

        # Residual + norm
        fused = self.cross_norm(tech_pooled + attn_out)  # (batch, 1, hidden)

        # FFN
        fused = fused + self.ffn(fused)
        fused = self.ffn_norm(fused)

        return fused.squeeze(1)  # (batch, hidden)


class MultiModalStockPredictor(nn.Module):
    """
    Multi-modal stock prediction model with separate encoders for each data type.

    Architecture:
        Fundamentals (static) → MLP → 64d
        Technicals (sequential) → Transformer → 128d
        News (sparse) → MLP + Attention → 64d
                    ↓
            Cross-Modal Fusion
                    ↓
            Prediction Head
    """

    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        hidden_dim: int = 256,
        fundamental_output_dim: Optional[int] = None,
        technical_output_dim: Optional[int] = None,
        news_output_dim: Optional[int] = None,
        num_technical_layers: int = 4,
        num_technical_heads: int = 4,
        num_pred_days: int = 4,
        pred_mode: str = 'regression',
        dropout: float = 0.1,
        max_seq_len: int = 4096,  # Support up to 4096 seq_len by default
    ):
        # Scale output dimensions proportionally with hidden_dim
        # Default ratios: fundamental=0.25, technical=0.5, news=0.25
        if fundamental_output_dim is None:
            fundamental_output_dim = max(64, hidden_dim // 4)
        if technical_output_dim is None:
            technical_output_dim = max(128, hidden_dim // 2)
        if news_output_dim is None:
            news_output_dim = max(64, hidden_dim // 4)
        super().__init__()

        self.config = feature_config or FeatureConfig()
        self.pred_mode = pred_mode
        self.num_pred_days = num_pred_days

        # Modality-specific encoders
        self.fundamental_encoder = FundamentalsEncoder(
            input_dim=self.config.fundamental_dim,
            hidden_dim=hidden_dim,
            output_dim=fundamental_output_dim,
            dropout=dropout,
        )

        # Technical encoder (after excluding leaky features)
        self.technical_encoder = TechnicalEncoder(
            input_dim=self.config.technical_dim,
            hidden_dim=hidden_dim,
            output_dim=technical_output_dim,
            num_layers=num_technical_layers,
            num_heads=num_technical_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self.news_encoder = NewsEncoder(
            input_dim=self.config.news_dim,
            hidden_dim=hidden_dim,
            output_dim=news_output_dim,
            dropout=dropout,
            use_temporal_attention=True,
        )

        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            technical_dim=technical_output_dim,
            fundamental_dim=fundamental_output_dim,
            news_dim=news_output_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
        )

        # Also concatenate pooled representations for prediction
        combined_dim = hidden_dim + fundamental_output_dim + technical_output_dim + news_output_dim

        # Prediction head
        if pred_mode == 'regression':
            self.pred_head = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_pred_days),
            )
        else:
            num_bins = 100
            self.num_bins = num_bins
            self.pred_head = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, num_bins * num_pred_days),
            )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_pred_days),
            nn.Sigmoid(),
        )

        self._print_architecture()

    def _print_architecture(self):
        """Print model architecture summary."""
        print("\n" + "="*60)
        print("Multi-Modal Stock Prediction Model")
        print("="*60)
        print(f"\nFeature Configuration:")
        print(f"  Fundamentals: indices {self.config.fundamental_indices}, dim={self.config.fundamental_dim}")
        print(f"  Technicals:   indices {self.config.technical_indices}, dim={self.config.technical_dim} (after exclusions)")
        print(f"  News:         indices {self.config.news_indices}, dim={self.config.news_dim}")
        print(f"  Excluded:     {self.config.excluded_technical_indices} (same-day return leakage)")
        print(f"\nEncoder Output Dimensions:")
        print(f"  Fundamentals: {self.fundamental_encoder.encoder[-1].normalized_shape[0]}")
        print(f"  Technicals:   {self.technical_encoder.output_proj[-1].normalized_shape[0]}")
        print(f"  News:         {self.news_encoder.output_dim}")
        print(f"\nPrediction:")
        print(f"  Mode: {self.pred_mode}")
        print(f"  Horizons: {self.num_pred_days}")
        print(f"\nTotal Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print("="*60 + "\n")

    def _split_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split input tensor into modality-specific features.

        Filters out excluded indices from technicals to prevent data leakage.
        """
        # Fundamentals: simple slice
        fundamentals = x[:, :, self.config.fundamental_indices[0]:self.config.fundamental_indices[1]]

        # Technicals: need to exclude leaky features (same-day returns)
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, total_features) input tensor

        Returns:
            predictions: (batch, num_pred_days) for regression
                        or (batch, num_bins, num_pred_days) for classification
            confidence: (batch, num_pred_days) confidence values in [0, 1]
        """
        # Handle NaN/Inf
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Split features by modality
        features = self._split_features(x)

        # Encode each modality
        fund_emb = self.fundamental_encoder(features['fundamentals'])  # (batch, fund_dim)

        # Technical features (already filtered to exclude leaky same-day returns)
        tech_seq, tech_pooled = self.technical_encoder(features['technicals'])  # (batch, seq, tech_dim), (batch, tech_dim)

        news_seq, news_pooled = self.news_encoder(features['news'])  # (batch, seq, news_dim), (batch, news_dim)

        # Cross-modal fusion
        fused = self.fusion(tech_seq, fund_emb, news_pooled)  # (batch, hidden)

        # Concatenate all representations
        combined = torch.cat([fused, fund_emb, tech_pooled, news_pooled], dim=-1)

        # Predictions
        if self.pred_mode == 'regression':
            predictions = self.pred_head(combined)  # (batch, num_pred_days)
        else:
            logits = self.pred_head(combined)  # (batch, num_bins * num_pred_days)
            predictions = logits.view(-1, self.num_bins, self.num_pred_days)

        # Confidence
        confidence = self.confidence_head(combined)  # (batch, num_pred_days)

        return predictions, confidence


def create_multimodal_model(
    input_dim: int = 998,
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.1,
    num_pred_days: int = 4,
    pred_mode: str = 'regression',
    **kwargs,
) -> MultiModalStockPredictor:
    """
    Factory function to create a multi-modal model.

    Args match the SimpleTransformerPredictor interface for compatibility.
    """
    return MultiModalStockPredictor(
        feature_config=FeatureConfig(),
        hidden_dim=hidden_dim,
        num_technical_layers=num_layers,
        num_technical_heads=num_heads,
        num_pred_days=num_pred_days,
        pred_mode=pred_mode,
        dropout=dropout,
    )


# =============================================================================
# Contrastive Learning Components
# =============================================================================

class TimeSeriesAugmenter(nn.Module):
    """
    Data augmentations for financial time series contrastive learning.

    Augmentation strategies:
    1. Temporal jittering - shift sequence by a few timesteps
    2. Feature masking - randomly mask some features
    3. Gaussian noise - add small noise to features
    4. Subsequence crop - use random subsequence
    """

    def __init__(
        self,
        temporal_jitter: int = 5,
        feature_mask_prob: float = 0.15,
        noise_std: float = 0.05,
        crop_ratio: float = 0.8,
    ):
        super().__init__()
        self.temporal_jitter = temporal_jitter
        self.feature_mask_prob = feature_mask_prob
        self.noise_std = noise_std
        self.crop_ratio = crop_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to input sequence.

        Args:
            x: (batch, seq_len, features) input tensor

        Returns:
            Augmented tensor of same shape
        """
        batch_size, seq_len, num_features = x.shape
        device = x.device

        # 1. Temporal jittering - roll sequence by random amount
        if self.temporal_jitter > 0 and self.training:
            shifts = torch.randint(-self.temporal_jitter, self.temporal_jitter + 1, (batch_size,), device=device)
            x_aug = torch.zeros_like(x)
            for i in range(batch_size):
                x_aug[i] = torch.roll(x[i], shifts=shifts[i].item(), dims=0)
            x = x_aug

        # 2. Feature masking - randomly zero out some features
        if self.feature_mask_prob > 0 and self.training:
            mask = torch.rand(batch_size, 1, num_features, device=device) > self.feature_mask_prob
            x = x * mask.float()

        # 3. Gaussian noise
        if self.noise_std > 0 and self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return x


class ContrastiveProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.

    Projects encoder representations to a lower-dimensional space
    where contrastive loss is computed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to contrastive space and L2 normalize."""
        z = self.projector(x)
        return F.normalize(z, dim=-1)


def info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Compute InfoNCE contrastive loss.

    For each sample, the positive pair is (z1[i], z2[i]) from two augmented views.
    All other samples in the batch are negatives.

    Args:
        z1: (batch, dim) L2-normalized embeddings from view 1
        z2: (batch, dim) L2-normalized embeddings from view 2
        temperature: Softmax temperature

    Returns:
        Scalar loss value
    """
    batch_size = z1.shape[0]
    device = z1.device

    # Concatenate both views
    z = torch.cat([z1, z2], dim=0)  # (2*batch, dim)

    # Compute similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2*batch, 2*batch)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs: (i, i+batch) and (i+batch, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=device),
        torch.arange(0, batch_size, device=device),
    ])

    loss = F.cross_entropy(sim, labels)
    return loss


class ContrastiveMultiModalModel(nn.Module):
    """
    Contrastive learning wrapper for MultiModalStockPredictor.

    Two-phase training:
    1. Pretrain: Learn representations via contrastive loss on augmented views
    2. Finetune: Freeze encoder, train only prediction head

    Architecture:
        Input -> Augmenter -> Encoder -> ProjectionHead -> ContrastiveLoss
                              |
                              v (after pretraining)
                         PredictionHead -> Predictions
    """

    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        hidden_dim: int = 256,
        projection_dim: int = 128,
        num_technical_layers: int = 4,
        num_technical_heads: int = 4,
        num_pred_days: int = 4,
        pred_mode: str = 'regression',
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        temperature: float = 0.1,
        # Augmentation params
        temporal_jitter: int = 5,
        feature_mask_prob: float = 0.15,
        noise_std: float = 0.05,
    ):
        super().__init__()

        self.temperature = temperature
        self.pred_mode = pred_mode

        # Data augmentation
        self.augmenter = TimeSeriesAugmenter(
            temporal_jitter=temporal_jitter,
            feature_mask_prob=feature_mask_prob,
            noise_std=noise_std,
        )

        # Main encoder (MultiModalStockPredictor without prediction heads)
        self.encoder = MultiModalStockPredictor(
            feature_config=feature_config,
            hidden_dim=hidden_dim,
            num_technical_layers=num_technical_layers,
            num_technical_heads=num_technical_heads,
            num_pred_days=num_pred_days,
            pred_mode=pred_mode,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Get the combined dimension from encoder
        config = feature_config or FeatureConfig()
        fundamental_output_dim = max(64, hidden_dim // 4)
        technical_output_dim = max(128, hidden_dim // 2)
        news_output_dim = max(64, hidden_dim // 4)
        combined_dim = hidden_dim + fundamental_output_dim + technical_output_dim + news_output_dim

        # Contrastive projection head
        self.projection_head = ContrastiveProjectionHead(
            input_dim=combined_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )

        self.combined_dim = combined_dim
        self._pretrain_mode = True

    def get_encoder_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get encoder representation (combined modality embeddings).

        Args:
            x: (batch, seq_len, features) input tensor

        Returns:
            (batch, combined_dim) representation
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Split features
        features = self.encoder._split_features(x)

        # Encode each modality
        fund_emb = self.encoder.fundamental_encoder(features['fundamentals'])
        tech_seq, tech_pooled = self.encoder.technical_encoder(features['technicals'])
        news_seq, news_pooled = self.encoder.news_encoder(features['news'])

        # Cross-modal fusion
        fused = self.encoder.fusion(tech_seq, fund_emb, news_pooled)

        # Concatenate
        combined = torch.cat([fused, fund_emb, tech_pooled, news_pooled], dim=-1)

        return combined

    def forward_contrastive(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive pretraining.

        Creates two augmented views of each sample and returns their projections.

        Args:
            x: (batch, seq_len, features) input tensor

        Returns:
            z1, z2: (batch, projection_dim) normalized projections for each view
        """
        # Create two augmented views
        x1 = self.augmenter(x)
        x2 = self.augmenter(x)

        # Get representations
        h1 = self.get_encoder_representation(x1)
        h2 = self.get_encoder_representation(x2)

        # Project to contrastive space
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        return z1, z2

    def compute_contrastive_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for a batch."""
        z1, z2 = self.forward_contrastive(x)
        return info_nce_loss(z1, z2, self.temperature)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for prediction (after pretraining).

        Uses the encoder's prediction heads.
        """
        return self.encoder(x)

    def freeze_encoder(self):
        """Freeze encoder weights for finetuning."""
        # Freeze all encoder components except prediction heads
        for name, param in self.encoder.named_parameters():
            if 'pred_head' not in name and 'confidence_head' not in name:
                param.requires_grad = False

        # Also freeze projection head (not needed for finetuning)
        for param in self.projection_head.parameters():
            param.requires_grad = False

        self._pretrain_mode = False

        # Count frozen vs trainable
        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        trainable = sum(1 for p in self.parameters() if p.requires_grad)
        print(f"Encoder frozen: {frozen} params frozen, {trainable} params trainable")

    def unfreeze_encoder(self):
        """Unfreeze all encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self._pretrain_mode = True


def pretrain_contrastive(
    model: ContrastiveMultiModalModel,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-4,
    device: torch.device = torch.device('cuda'),
    log_interval: int = 100,
) -> ContrastiveMultiModalModel:
    """
    Pretrain model with contrastive learning.

    Args:
        model: ContrastiveMultiModalModel instance
        dataloader: DataLoader yielding (features, targets) tuples (targets ignored)
        num_epochs: Number of pretraining epochs
        lr: Learning rate
        device: Device to train on
        log_interval: Log every N batches

    Returns:
        Pretrained model
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\n{'='*60}")
    print("Contrastive Pretraining")
    print(f"{'='*60}")
    print(f"Epochs: {num_epochs}, LR: {lr}, Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (features, _targets) in enumerate(dataloader):
            features = features.to(device)

            optimizer.zero_grad()
            loss = model.compute_contrastive_loss(features)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1} | Loss: {avg_loss:.4f}")

        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} complete | Avg Loss: {avg_loss:.4f}")

    print(f"\nContrastive pretraining complete!")
    return model


if __name__ == '__main__':
    # Test the model
    print("Testing MultiModalStockPredictor...")

    config = FeatureConfig()
    model = MultiModalStockPredictor(feature_config=config)

    # Simulate input: (batch=4, seq_len=256, features=1123)
    x = torch.randn(4, 256, config.total_features)

    # Zero out most news (simulate sparse news) - news is at indices 353-1123
    x[:, :, config.news_indices[0]:config.news_indices[1]] = 0
    # Add some news for first sample
    x[0, -10:, config.news_indices[0]:config.news_indices[0]+768] = torch.randn(10, 768) * 0.1

    pred, conf = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Confidence shape: {conf.shape}")
    print(f"Prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"Confidence range: [{conf.min():.4f}, {conf.max():.4f}]")

    # Verify excluded features are not used
    print(f"\nExcluded features: {config.excluded_technical_indices}")
    print(f"Technical dim (after exclusions): {config.technical_dim}")

    # Test backward pass
    loss = pred.sum()
    loss.backward()
    print("\nBackward pass successful!")
