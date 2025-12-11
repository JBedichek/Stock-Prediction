"""
News Embedder using Nomic Embed Text v1.5

Embeds news articles using nomic-ai/nomic-embed-text-v1.5 from HuggingFace.
This model produces high-quality 768-dimensional embeddings for text.

Features:
- Batched embedding for efficiency
- GPU support
- Aggregation of multiple news articles per day
- Temporal weighting (recent news weighted more)
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date as dt_date
from typing import Dict, List, Optional, Tuple
import sys
import os
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import save_pickle, pic_load


class NomicNewsEmbedder:
    """
    Embedder for news articles using Nomic Embed Text v1.5.
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5",
                 device: str = None,
                 batch_size: int = 32):
        """
        Initialize embedder.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda' or 'cpu')
            batch_size: Batch size for embedding
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading Nomic Embed model on {self.device}...")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded: {model_name}")
        print(f"   Embedding dimension: 768")

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling to get sentence embeddings.

        Args:
            model_output: Model output
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_texts(self, texts: List[str], prefix: str = "search_document: ") -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings
            prefix: Prefix for embeddings (Nomic requires this)

        Returns:
            Array of embeddings (N, 768)
        """
        if not texts:
            return np.array([])

        # Add prefix (required by Nomic)
        texts = [prefix + text for text in texts]

        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]

                # Tokenize
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=8192,  # Nomic supports long context
                    return_tensors='pt'
                )

                # Move to device
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

                # Get embeddings
                model_output = self.model(**encoded_input)

                # Pool
                batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

                # Normalize
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def embed_article(self, article: Dict) -> np.ndarray:
        """
        Embed a single news article.

        Args:
            article: Article dict with 'title', 'description', 'full_text'

        Returns:
            Embedding vector (768,)
        """
        # Combine title, description, and full text
        text_parts = []

        if article.get('title'):
            text_parts.append(f"Title: {article['title']}")

        if article.get('description'):
            text_parts.append(f"Summary: {article['description']}")

        if article.get('full_text'):
            # Limit full text to avoid context length issues
            full_text = article['full_text'][:5000]
            text_parts.append(f"Content: {full_text}")

        combined_text = " ".join(text_parts)

        if not combined_text.strip():
            # Return zero vector if no text
            return np.zeros(768, dtype=np.float32)

        # Embed
        embedding = self.embed_texts([combined_text])
        return embedding[0]

    def embed_articles_batch(self, articles: List[Dict]) -> np.ndarray:
        """
        Embed multiple articles efficiently.

        Args:
            articles: List of article dicts

        Returns:
            Array of embeddings (N, 768)
        """
        # Prepare texts
        texts = []
        for article in articles:
            text_parts = []

            if article.get('title'):
                text_parts.append(f"Title: {article['title']}")

            if article.get('description'):
                text_parts.append(f"Summary: {article['description']}")

            if article.get('full_text'):
                full_text = article['full_text'][:5000]
                text_parts.append(f"Content: {full_text}")

            combined_text = " ".join(text_parts)
            texts.append(combined_text if combined_text.strip() else "No content")

        # Embed all at once
        return self.embed_texts(texts)


class NewsAggregator:
    """
    Aggregates news embeddings by date and ticker.
    """

    def __init__(self, embedder: NomicNewsEmbedder):
        """
        Initialize aggregator.

        Args:
            embedder: NomicNewsEmbedder instance
        """
        self.embedder = embedder

    def parse_article_date(self, article: Dict) -> Optional[dt_date]:
        """
        Parse article date from various formats.

        Args:
            article: Article dict

        Returns:
            Date object or None
        """
        date_str = article.get('published_date', '')

        if not date_str:
            return None

        try:
            # Try various date formats
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%a, %d %b %Y %H:%M:%S %Z',
                       '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.date()
                except:
                    continue

            # Try pandas parsing as fallback
            dt = pd.to_datetime(date_str)
            return dt.date()

        except:
            return None

    def aggregate_by_date(self, articles: List[Dict], method: str = 'mean',
                         temporal_weight: bool = True) -> Dict[dt_date, np.ndarray]:
        """
        Aggregate article embeddings by date.

        Args:
            articles: List of articles for a ticker
            method: Aggregation method ('mean', 'max', 'weighted_mean')
            temporal_weight: Whether to weight recent articles more

        Returns:
            Dict of {date: aggregated_embedding}
        """
        # Embed all articles
        print(f"    Embedding {len(articles)} articles...")
        embeddings = self.embedder.embed_articles_batch(articles)

        # Group by date
        date_to_embeddings = {}

        for article, embedding in zip(articles, embeddings):
            article_date = self.parse_article_date(article)

            if article_date is None:
                # Use today if no date
                article_date = datetime.now().date()

            if article_date not in date_to_embeddings:
                date_to_embeddings[article_date] = []

            date_to_embeddings[article_date].append(embedding)

        # Aggregate embeddings for each date
        aggregated = {}

        for date, date_embeddings in date_to_embeddings.items():
            date_embeddings = np.array(date_embeddings)

            if method == 'mean':
                aggregated[date] = np.mean(date_embeddings, axis=0)

            elif method == 'max':
                # Max pooling across articles
                aggregated[date] = np.max(date_embeddings, axis=0)

            elif method == 'weighted_mean' and temporal_weight:
                # Weight more recent articles more heavily
                # (This assumes all articles on same date, so just mean)
                aggregated[date] = np.mean(date_embeddings, axis=0)

        return aggregated

    def create_daily_embeddings_dataset(self, all_news: Dict[str, List[Dict]],
                                       start_date: dt_date,
                                       end_date: dt_date,
                                       fill_missing_with_zeros: bool = True,
                                       output_file: str = None,
                                       save_every: int = 500) -> Dict[str, Dict[dt_date, torch.Tensor]]:
        """
        Create daily news embeddings for all stocks.

        Args:
            all_news: Dict of {ticker: [articles]}
            start_date: Start date
            end_date: End date
            fill_missing_with_zeros: If True, creates zero embeddings for stocks with no news
                                    If False, skips stocks with no news entirely
            output_file: If provided, saves incrementally to this file
            save_every: Save progress every N stocks (to avoid memory buildup)

        Returns:
            Dict of {ticker: {date: embedding_tensor}}
        """
        print(f"\n{'='*80}")
        print(f"CREATING DAILY NEWS EMBEDDINGS")
        print(f"{'='*80}")
        print(f"Stocks: {len(all_news)}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Fill missing stocks with zeros: {fill_missing_with_zeros}")
        if output_file:
            print(f"Incremental saving: Every {save_every} stocks to {output_file}")
        print(f"{'='*80}\n")

        # Check for existing progress
        daily_embeddings = {}
        if output_file and save_pickle.__module__:  # Check if file exists
            import os
            if os.path.exists(output_file):
                try:
                    daily_embeddings = pic_load(output_file)
                    print(f"📂 Found existing embeddings: {len(daily_embeddings)} stocks completed")
                    print(f"   Resuming from next stock...\n")
                except:
                    pass

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        stock_count = 0
        for ticker, articles in tqdm(all_news.items(), desc="Processing stocks"):
            # Skip already embedded stocks
            if ticker in daily_embeddings:
                print(f"\n{ticker}: ⏭️  Already embedded, skipping...")
                continue

            print(f"\n{ticker}: {len(articles)} articles")
            stock_count += 1

            # Handle stocks with no articles
            if not articles or len(articles) == 0:
                if fill_missing_with_zeros:
                    print(f"  ⚠️  No articles, will use zeros on-the-fly (sparse storage)")
                    # Store empty dict - zeros will be computed on-the-fly when needed
                    daily_embeddings[ticker] = {}
                else:
                    print(f"  ⚠️  No articles, skipping ticker")
                continue

            try:
                # Aggregate by date
                date_embeddings = self.aggregate_by_date(articles, method='mean')

                if not date_embeddings:
                    print(f"  ⚠️  Failed to create embeddings, storing empty (sparse)")
                    # Store empty dict - zeros will be computed on-the-fly
                    daily_embeddings[ticker] = {}
                    continue

                # Store only days with actual news (sparse storage)
                ticker_daily = {}
                for date_obj, embedding in date_embeddings.items():
                    # Convert to tensor and store
                    ticker_daily[date_obj] = torch.tensor(embedding, dtype=torch.float32)

                daily_embeddings[ticker] = ticker_daily
                print(f"  ✅ {len(date_embeddings)} unique dates with news (sparse storage)")

            except Exception as e:
                print(f"  ❌ Error embedding {ticker}: {e}")
                import traceback
                traceback.print_exc()

                # Even on error, store empty dict if requested
                if fill_missing_with_zeros:
                    print(f"  Creating empty embeddings as fallback (sparse storage)")
                    daily_embeddings[ticker] = {}

                continue

            # Save incrementally and clear memory
            if output_file and stock_count % save_every == 0:
                print(f"\n💾 Saving progress ({len(daily_embeddings)} stocks)...")
                save_pickle(daily_embeddings, output_file)

                # Clear GPU cache to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Force garbage collection
                import gc
                gc.collect()

                print(f"   ✅ Saved, cleared GPU cache, and ran garbage collection")

                # Show memory usage
                try:
                    import psutil
                    import os as os_module
                    process = psutil.Process(os_module.getpid())
                    mem_gb = process.memory_info().rss / 1024**3
                    print(f"   📊 Current RAM usage: {mem_gb:.2f} GB")
                except:
                    pass

        # Final save
        if output_file:
            print(f"\n💾 Final save ({len(daily_embeddings)} stocks)...")
            save_pickle(daily_embeddings, output_file)

        print(f"\n✅ Daily embeddings created for {len(daily_embeddings)} stocks")

        # Count stocks with actual embeddings
        stocks_with_news = sum(1 for dates in daily_embeddings.values() if len(dates) > 0)
        total_stored_days = sum(len(dates) for dates in daily_embeddings.values())

        print(f"   Stocks with actual news: {stocks_with_news}/{len(daily_embeddings)}")
        print(f"   Total days stored (sparse): {total_stored_days:,}")
        print(f"   Avg days per stock: {total_stored_days / len(daily_embeddings):.1f}")
        print(f"   💾 Sparse storage: Only days with news are stored (zeros computed on-the-fly)")

        return daily_embeddings


def get_embedding_for_date(embeddings_dict: Dict[str, Dict[dt_date, torch.Tensor]],
                          ticker: str,
                          date: dt_date,
                          default_dim: int = 768) -> torch.Tensor:
    """
    Get embedding for a specific ticker and date, returning zeros if not found.

    Args:
        embeddings_dict: Dict of {ticker: {date: embedding}}
        ticker: Stock ticker
        date: Date to lookup
        default_dim: Dimension for zero vector (default: 768)

    Returns:
        Embedding tensor (768,) - either real embedding or zeros
    """
    if ticker not in embeddings_dict:
        return torch.zeros(default_dim, dtype=torch.float32)

    ticker_embeddings = embeddings_dict[ticker]

    if date in ticker_embeddings:
        return ticker_embeddings[date]
    else:
        return torch.zeros(default_dim, dtype=torch.float32)


def create_news_embeddings(news_data_file: str,
                          output_file: str,
                          start_date: str,
                          end_date: str = None,
                          device: str = None,
                          chunk_size: int = None,
                          reload_model_between_chunks: bool = True) -> Dict[str, Dict[dt_date, torch.Tensor]]:
    """
    Create news embeddings from scraped news data.

    Args:
        news_data_file: Path to news data pickle
        output_file: Output file for embeddings
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (defaults to today)
        device: Device for embedding ('cuda' or 'cpu')
        chunk_size: Process stocks in chunks of this size (default: all at once)
                   Useful for large datasets to avoid GPU OOM. Recommended: 50-200 for 32GB GPU
        reload_model_between_chunks: If True, unload and reload model between chunks to clear GPU memory

    Returns:
        Dict of {ticker: {date: embedding_tensor}}
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Load news data
    print("📂 Loading news data...")
    all_news = pic_load(news_data_file)
    total_stocks = len(all_news)
    print(f"  ✅ Loaded news for {total_stocks} stocks\n")

    # Check for existing progress
    import os
    import gc
    daily_embeddings = {}
    if os.path.exists(output_file):
        try:
            daily_embeddings = pic_load(output_file)
            print(f"📂 Found existing embeddings: {len(daily_embeddings)} stocks completed")
            print(f"   Resuming from next stock...\n")
        except:
            pass

    # Filter out already completed stocks
    all_news_items = list(all_news.items())
    pending_stocks = [(ticker, articles) for ticker, articles in all_news_items
                     if ticker not in daily_embeddings]

    if not pending_stocks:
        print("✅ All stocks already embedded!")
        return daily_embeddings

    print(f"📊 Stocks to process: {len(pending_stocks)}/{total_stocks}")
    print(f"   Already completed: {len(daily_embeddings)}")
    print(f"   Remaining: {len(pending_stocks)}\n")

    # Determine chunking strategy
    if chunk_size is None:
        # Process all at once (may cause OOM on large datasets)
        chunk_size = len(pending_stocks)
        print(f"⚠️  Processing all {chunk_size} stocks at once (may cause OOM)")
        print(f"   Consider using --chunk-size 100 for large datasets\n")
    else:
        num_chunks = (len(pending_stocks) + chunk_size - 1) // chunk_size
        print(f"📦 Chunked processing: {num_chunks} chunks of ~{chunk_size} stocks")
        print(f"   Model reload between chunks: {reload_model_between_chunks}\n")

    # Process in chunks
    for chunk_idx in range(0, len(pending_stocks), chunk_size):
        chunk_stocks = dict(pending_stocks[chunk_idx:chunk_idx + chunk_size])
        chunk_num = chunk_idx // chunk_size + 1
        total_chunks = (len(pending_stocks) + chunk_size - 1) // chunk_size

        print(f"\n{'='*80}")
        print(f"CHUNK {chunk_num}/{total_chunks}: Processing {len(chunk_stocks)} stocks")
        print(f"{'='*80}\n")

        # Initialize embedder for this chunk
        embedder = NomicNewsEmbedder(device=device)
        aggregator = NewsAggregator(embedder)

        # Process this chunk
        chunk_embeddings = aggregator.create_daily_embeddings_dataset(
            chunk_stocks, start_dt, end_dt,
            output_file=output_file,
            save_every=500  # Save every 500 stocks within chunk
        )

        # Merge chunk results
        daily_embeddings.update(chunk_embeddings)

        # Save after each chunk
        print(f"\n💾 Saving after chunk {chunk_num}/{total_chunks}...")
        save_pickle(daily_embeddings, output_file)
        print(f"   ✅ Saved {len(daily_embeddings)}/{total_stocks} stocks")

        # Clean up between chunks
        if reload_model_between_chunks and chunk_num < total_chunks:
            print(f"\n🧹 Cleaning up GPU memory...")

            # Delete embedder and aggregator
            del embedder
            del aggregator
            del chunk_embeddings

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

                # Show GPU memory usage
                allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
                reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
                print(f"   GPU memory: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")

            # Force garbage collection
            gc.collect()

            # Show RAM usage
            try:
                import psutil
                process = psutil.Process(os.getpid())
                mem_gb = process.memory_info().rss / 1024**3
                print(f"   RAM usage: {mem_gb:.2f}GB")
            except:
                pass

            print(f"   ✅ Cleaned up, ready for next chunk\n")

    # Final save
    print(f"\n💾 Final save...")
    save_pickle(daily_embeddings, output_file)

    print(f"\n{'='*80}")
    print(f"✅ EMBEDDINGS COMPLETE!")
    print(f"{'='*80}")
    print(f"Output: {output_file}")
    print(f"Stocks processed: {len(daily_embeddings)}")
    print(f"Embedding dimension: 768")
    print(f"{'='*80}\n")

    return daily_embeddings


def fetch_date_range_news(date_range: List[dt_date]) -> Dict[dt_date, torch.Tensor]:
    """
    Fetch and embed news for a range of dates.

    NOTE: This is a stub implementation that returns zero embeddings.
    Full implementation would require:
    1. Fetching news articles from API (AlphaVantage, GNews, etc.)
    2. Embedding them with Nomic model
    3. Aggregating per day

    For now, we return zero-filled embeddings to avoid blocking dataset updates.

    Args:
        date_range: List of dates to fetch news for

    Returns:
        Dict mapping dates to news embedding tensors (768-dim)
    """
    print(f"  ⚠️  News fetching not yet implemented")
    print(f"     Using zero embeddings (news features will be neutral)")

    # Return zero embeddings for each date (768 dimensions)
    news_by_date = {}
    for date in date_range:
        news_by_date[date] = torch.zeros(768, dtype=torch.float32)

    return news_by_date


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Embed news articles')
    parser.add_argument('--news_data', type=str, required=True,
                       help='Input news data pickle')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (default: input with _embeddings suffix)')
    parser.add_argument('--start_date', type=str,
                       default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                       help='Start date')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (default: today)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device (default: auto-detect)')

    args = parser.parse_args()

    if args.output is None:
        args.output = args.news_data.replace('.pkl', '_embeddings.pkl')

    embeddings = create_news_embeddings(
        news_data_file=args.news_data,
        output_file=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        device=args.device
    )

    print(f"\n✅ Done! Embeddings saved to {args.output}")
