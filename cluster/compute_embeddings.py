"""
Compute Embeddings from Dataset

Encodes the entire dataset using transformer mean pooling and saves the embeddings.

Process:
1. Load trained price predictor
2. Encode all stocks using transformer activations (mean pool over sequence)
3. Save embeddings to disk for later clustering
"""

import torch
import h5py
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pickle


class ClusterEncoder:
    """
    Encode dataset using transformer mean pooling.
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize encoder.

        Args:
            model_path: Path to trained price predictor checkpoint
            device: Device to run on
        """
        self.device = device
        print(f"\n{'='*80}")
        print("INITIALIZING CLUSTER ENCODER")
        print(f"{'='*80}")

        print(f"\n📦 Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # Handle both direct model saves and checkpoint dicts
        if isinstance(checkpoint, dict):
            # Checkpoint dict format - need to reconstruct model
            if 'model' in checkpoint:
                self.model = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                # SimpleTransformerPredictor from train_new_format.py
                from training.train_new_format import SimpleTransformerPredictor

                config = checkpoint.get('config', {})

                # Get input_dim from model state dict
                state_dict = checkpoint['model_state_dict']
                # Find input_dim from first layer (input_proj.0.weight)
                for key in state_dict.keys():
                    if 'input_proj.0.weight' in key:
                        input_dim = state_dict[key].shape[1]
                        break
                else:
                    input_dim = 245  # Default from dataset (218 price + 27 summary)

                self.model = SimpleTransformerPredictor(
                    input_dim=input_dim,
                    hidden_dim=config.get('hidden_dim', 1024),
                    num_layers=config.get('num_layers', 10),
                    num_heads=config.get('num_heads', 16),
                    dropout=config.get('dropout', 0.15),
                    num_pred_days=4,  # [1, 5, 10, 20]
                    pred_mode=config.get('pred_mode', 'classification')
                )
                # Use strict=False to allow missing keys (e.g., confidence_head if added later)
                missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

                if missing_keys:
                    print(f"   ⚠️  Missing keys (will use random init): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
                if unexpected_keys:
                    print(f"   ⚠️  Unexpected keys in checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            else:
                raise ValueError("Checkpoint does not contain 'model' or 'model_state_dict'")
        else:
            # Direct model save
            self.model = checkpoint

        self.model.eval()
        self.model.to(device)

        print(f"✅ Model loaded successfully")

    def encode_dataset(self, dataset_path: str,
                      max_stocks: Optional[int] = None,
                      samples_per_stock: int = 10,
                      batch_size: int = 64) -> Dict[str, np.ndarray]:
        """
        Encode dataset using transformer mean pooling with temporal sampling.

        For robust clustering, we sample multiple timesteps per stock to capture
        different market regimes and conditions.

        Args:
            dataset_path: Path to HDF5 dataset
            max_stocks: Optional limit on number of stocks to encode
            samples_per_stock: Number of random timesteps to sample per stock
            batch_size: Batch size for encoding

        Returns:
            Dictionary mapping (ticker, timestep_idx) -> embedding
        """
        print(f"\n{'='*80}")
        print("ENCODING DATASET WITH TEMPORAL SAMPLING")
        print(f"{'='*80}")

        print(f"\n📂 Opening dataset: {dataset_path}")
        h5_file = h5py.File(dataset_path, 'r')
        tickers = sorted(list(h5_file.keys()))

        if max_stocks:
            tickers = tickers[:max_stocks]

        # Check first ticker to understand data structure
        if len(tickers) > 0:
            first_ticker = tickers[0]
            sample_features = h5_file[first_ticker]['features'][:]
            print(f"\n📋 Data structure (sample from {first_ticker}):")
            print(f"   Features shape: {sample_features.shape}")
            print(f"   Features dtype: {sample_features.dtype}")

        print(f"\n📊 Encoding {len(tickers)} stocks with {samples_per_stock} samples each...")
        print(f"   Total samples: {len(tickers) * samples_per_stock:,}")

        # Create all (ticker, timestep) pairs to encode
        samples_to_encode = []
        for ticker in tickers:
            if ticker not in h5_file:
                continue

            features = h5_file[ticker]['features'][:]
            if len(features.shape) != 2:
                continue

            num_timesteps = features.shape[0]
            if num_timesteps < samples_per_stock:
                # Use all available timesteps if fewer than requested
                timestep_indices = list(range(num_timesteps))
            else:
                # Randomly sample timesteps
                timestep_indices = np.random.choice(num_timesteps, samples_per_stock, replace=False)

            for timestep_idx in timestep_indices:
                samples_to_encode.append((ticker, int(timestep_idx)))

        print(f"   Generated {len(samples_to_encode):,} samples to encode")

        embeddings = {}

        # Process in batches
        for i in tqdm(range(0, len(samples_to_encode), batch_size), desc="Encoding"):
            batch_samples = samples_to_encode[i:i+batch_size]
            batch_embeddings = self._encode_batch_with_timesteps(h5_file, batch_samples)

            for (ticker, timestep_idx), emb in zip(batch_samples, batch_embeddings):
                # Key format: "TICKER_timestep_123"
                key = f"{ticker}_t{timestep_idx}"
                embeddings[key] = emb

        h5_file.close()

        print(f"\n✅ Encoded {len(embeddings):,} (stock, time) samples")
        print(f"   Embedding dimension: {list(embeddings.values())[0].shape[0]}")

        return embeddings

    def _encode_batch_with_timesteps(self, h5_file: h5py.File, samples: List[Tuple[str, int]]) -> np.ndarray:
        """
        Encode a batch of (ticker, timestep) pairs.

        Args:
            h5_file: Open HDF5 file
            samples: List of (ticker, timestep_idx) tuples

        Returns:
            Batch of embeddings (batch_size, embedding_dim)
        """
        batch_features = []
        expected_feature_dim = None

        for ticker, timestep_idx in samples:
            if ticker not in h5_file:
                continue

            try:
                # Get features for this specific timestep
                features = h5_file[ticker]['features'][:]  # (num_dates, num_features)

                if len(features.shape) != 2:
                    continue

                if timestep_idx >= features.shape[0]:
                    continue

                # Extract features for this specific timestep
                features = features[timestep_idx]

                # Check feature dimension consistency
                if expected_feature_dim is None:
                    expected_feature_dim = len(features)
                elif len(features) != expected_feature_dim:
                    # Skip stocks with different feature dimensions
                    continue

                batch_features.append(features)

            except Exception as e:
                # Skip stocks with errors
                continue

        if len(batch_features) == 0:
            return np.array([])

        # Convert to tensor
        batch_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32).to(self.device)

        # Check if we need to pad/adjust features to match model input_dim
        current_feature_dim = batch_tensor.shape[1]
        model_input_dim = self.model.input_dim

        if current_feature_dim != model_input_dim:
            if not hasattr(self, '_dimension_warning_shown'):
                print(f"\n⚠️  Feature dimension mismatch:")
                print(f"   Dataset features: {current_feature_dim}")
                print(f"   Model expects: {model_input_dim}")
                if current_feature_dim < model_input_dim:
                    print(f"   → Padding with {model_input_dim - current_feature_dim} zeros")
                else:
                    print(f"   → Truncating to {model_input_dim} features")
                self._dimension_warning_shown = True

            if current_feature_dim < model_input_dim:
                # Pad with zeros to match model input dimension
                padding = torch.zeros(batch_tensor.shape[0], model_input_dim - current_feature_dim, device=self.device)
                batch_tensor = torch.cat([batch_tensor, padding], dim=1)
            else:
                # Truncate to match model input dimension
                batch_tensor = batch_tensor[:, :model_input_dim]

        # Reshape to (batch, seq_len, feature_dim)
        if len(batch_tensor.shape) == 2:
            # Expand to create sequence (use same features for all timesteps)
            seq_len = 2000  # Match model's seq_len
            batch_tensor = batch_tensor.unsqueeze(1).expand(-1, seq_len, -1)

        # Get transformer activations (SimpleTransformerPredictor)
        # Extract mean-pooled transformer output using a forward hook
        with torch.no_grad():
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output
                return hook

            # Register hook on transformer output
            handle = self.model.transformer.register_forward_hook(get_activation('transformer'))

            # Forward pass
            _ = self.model(batch_tensor)

            # Remove hook
            handle.remove()

            # Get transformer output and mean pool
            transformer_out = activation['transformer']  # (batch, seq_len, hidden_dim)
            t_act = transformer_out.mean(dim=1)  # (batch, hidden_dim)

        # t_act shape: (batch, transformer_dim) - already mean pooled
        embeddings = t_act.cpu().numpy()

        return embeddings


def save_embeddings(embeddings: Dict[str, np.ndarray], output_path: str):
    """
    Save embeddings to disk.

    Args:
        embeddings: Dictionary mapping key -> embedding
        output_path: Path to save embeddings (pickle format)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("SAVING EMBEDDINGS")
    print(f"{'='*80}")
    print(f"Saving {len(embeddings):,} embeddings to: {output_path}")

    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    # Print file size
    file_size_mb = output_path.stat().st_size / 1e6
    print(f"   ✅ Embeddings saved successfully")
    print(f"   📁 File: {output_path}")
    print(f"   💾 Size: {file_size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Compute embeddings from dataset')

    # Required
    parser.add_argument('--model-path', type=str, default="./checkpoints/best_model_100m_1.18.pt",
                       help='Path to trained predictor')
    parser.add_argument('--dataset-path', type=str, default="./data/all_complete_dataset.h5",
                       help='Path to HDF5 dataset')
    parser.add_argument('--output-path', type=str, default='./data/embeddings.pkl',
                       help='Path to save embeddings (pickle format)')

    # Encoding parameters
    parser.add_argument('--max-stocks', type=int, default=100000,
                       help='Max stocks to encode')
    parser.add_argument('--samples-per-stock', type=int, default=100,
                       help='Number of random timesteps to sample per stock')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for encoding')

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create encoder
    encoder = ClusterEncoder(args.model_path, args.device)

    # Encode dataset
    embeddings = encoder.encode_dataset(
        args.dataset_path,
        max_stocks=args.max_stocks,
        samples_per_stock=args.samples_per_stock,
        batch_size=args.batch_size
    )

    # Save embeddings
    save_embeddings(embeddings, args.output_path)

    print(f"\n{'='*80}")
    print("ENCODING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nNext step: Create clusters from embeddings")
    print(f"  python -m cluster.cluster_embeddings \\")
    print(f"    --embeddings-path {args.output_path} \\")
    print(f"    --n-clusters 50 \\")
    print(f"    --output-dir ./cluster_results")


if __name__ == '__main__':
    main()
