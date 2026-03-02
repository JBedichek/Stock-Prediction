# Stock Prediction Docker Image
#
# Build:
#   docker build -t stock-prediction .
#
# Run (interactive):
#   docker run -it --gpus all -v $(pwd)/data:/app/data stock-prediction
#
# Run training:
#   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints \
#       stock-prediction python -m training.walk_forward_training \
#       --data data/all_complete_dataset.h5 --prices data/actual_prices_clean.h5

FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt \
    && pip install --no-cache-dir --break-system-packages huggingface_hub

# Copy source code
COPY . .

# Create directories for data and checkpoints
RUN mkdir -p /app/data /app/checkpoints

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command: download data and start shell
CMD ["bash", "-c", "echo 'Stock Prediction Environment Ready' && \
    echo '' && \
    echo 'To download data:' && \
    echo '  python scripts/download_data.py --repo-id YOUR_REPO_ID' && \
    echo '' && \
    echo 'To run evaluation:' && \
    echo '  python -m inference.principled_evaluation --checkpoint-dir checkpoints/walk_forward --data data/all_complete_dataset.h5 --prices data/actual_prices_clean.h5 --sweep' && \
    echo '' && \
    exec bash"]
