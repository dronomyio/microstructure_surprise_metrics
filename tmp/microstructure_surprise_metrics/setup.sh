#!/bin/bash

# Setup script to install dependencies and prepare the build environment

echo "Setting up SurpriseMetrics build environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install system dependencies based on distribution
install_dependencies() {
    if command_exists apt-get; then
        echo -e "${GREEN}Installing dependencies via apt-get...${NC}"
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            git \
            wget \
            curl \
            libcurl4-openssl-dev \
            libomp-dev \
            python3-dev \
            python3-pip \
            libeigen3-dev \
            libboost-all-dev \
            rapidjson-dev
            
    elif command_exists yum; then
        echo -e "${GREEN}Installing dependencies via yum...${NC}"
        sudo yum install -y \
            gcc-c++ \
            cmake3 \
            git \
            wget \
            curl \
            libcurl-devel \
            eigen3-devel \
            python3-devel \
            boost-devel
    else
        echo -e "${RED}Unsupported package manager. Please install dependencies manually.${NC}"
        exit 1
    fi
}

# Install Python dependencies
install_python_deps() {
    echo -e "${GREEN}Installing Python dependencies...${NC}"
    pip3 install --user numpy pandas cython pybind11 setuptools wheel
}

# Download and install pybind11 if not found
install_pybind11() {
    if [ ! -d "third_party/pybind11" ]; then
        echo -e "${GREEN}Installing pybind11...${NC}"
        mkdir -p third_party
        cd third_party
        git clone https://github.com/pybind/pybind11.git
        cd ..
    fi
}

# Create necessary directories
create_directories() {
    echo -e "${GREEN}Creating project directories...${NC}"
    mkdir -p build
    mkdir -p data
    mkdir -p output
    mkdir -p logs
    mkdir -p cmake
}

# Main setup
echo -e "${YELLOW}This script will install dependencies for SurpriseMetrics${NC}"
echo -e "${YELLOW}You may be prompted for sudo password${NC}"

install_dependencies
install_python_deps
install_pybind11
create_directories

echo -e "${GREEN}Setup complete! You can now build the project:${NC}"
echo "  cd build"
echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "  make -j\$(nproc)"
