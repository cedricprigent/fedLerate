# Use an official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Automatically expose all GPUs and compute capabilities
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Upgrade build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy configuration files first for caching layers
COPY pyproject.toml ./

# Option A: Pre-install dependencies defined in pyproject.toml
# (Installing editable mode with an empty directory structure registers the package)
COPY . /app
RUN pip install --no-cache-dir .

# Default command
CMD ["python3"]