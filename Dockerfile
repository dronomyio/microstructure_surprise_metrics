FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set non-interactive frontend for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libcurl4-openssl-dev \
    libomp-dev \
    libeigen3-dev \
    libboost-all-dev \
    rapidjson-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/
RUN pip3 install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

# Install pybind11
RUN pip3 install pybind11[global]

# Set working directory
WORKDIR /app

# Copy source code
COPY . /app

# Make setup script executable and run it
RUN chmod +x setup.sh && ./setup.sh

# Build the project
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86" && \
    make -j$(nproc)

# Set Python path
ENV PYTHONPATH=/app/build:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/app/build:${LD_LIBRARY_PATH}

# Default command
CMD ["./build/surprise_metrics_runner"]
