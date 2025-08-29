#!/bin/bash

# Script to setup pybind11 for the microstructure surprise metrics project

echo "Setting up pybind11..."

# Check if pybind11 is already available
if [ -f "third_party/pybind11/include/pybind11/pybind11.h" ]; then
    echo "pybind11 already exists in third_party/"
    exit 0
fi

# Try to install via pip first
echo "Trying to install pybind11 via pip..."
pip install pybind11 2>/dev/null
if [ $? -eq 0 ]; then
    echo "pybind11 installed via pip successfully"
    exit 0
fi

# If pip fails, download pybind11 to third_party
echo "Downloading pybind11 to third_party/..."
cd third_party/

# Remove existing empty directory if it exists
rm -rf pybind11

# Clone pybind11
git clone https://github.com/pybind/pybind11.git
if [ $? -eq 0 ]; then
    echo "pybind11 downloaded successfully to third_party/pybind11"
    echo "You can now run cmake and make to build the project with Python bindings"
else
    echo "Failed to download pybind11"
    echo "Please install pybind11 manually:"
    echo "  pip install pybind11"
    echo "  or"
    echo "  git clone https://github.com/pybind/pybind11.git third_party/pybind11"
    exit 1
fi

