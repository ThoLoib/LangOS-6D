# Base image with GPU support (CUDA 12.2)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV OLLAMA_HOST=0.0.0.0:11434

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.11 python3.11-dev curl \
    wget unzip git ca-certificates \
    zstd libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 \
    libgl1 libglib2.0-0 \
 && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install pip explicitly
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    opencv-python \
    accelerate && \
    pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Download and extract Blender dataset
RUN mkdir -p /blender && \
    cd /blender && \
    wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip && \
    unzip blender.zip && \
    rm blender.zip

# Expose needed ports   
EXPOSE 11434

# Copy and prepare startup script
COPY start.sh .
RUN chmod +x start.sh

# Default command
CMD ["/app/start.sh"]


