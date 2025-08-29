# Build Instructions for Microstructure Surprise Metrics

## Prerequisites

### Required Dependencies
- **CUDA Toolkit** (version 11.0 or later)
- **CMake** (version 3.16 or later)
- **C++ Compiler** with C++17 support (GCC 9+ or Clang 10+)
- **OpenMP** support
- **Python** (3.7 or later) - for Python bindings

### Optional Dependencies
- **pybind11** - for Python bindings
- **HDF5** - for HDF5 file support
- **Apache Arrow/Parquet** - for Parquet file support
- **Eigen3** - for additional linear algebra operations

## Build Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd microstructure_surprise_metrics
```

### 2. Setup Python Bindings (Optional)
If you want Python bindings, you need pybind11:

**Option A: Install via pip**
```bash
pip install pybind11
```

**Option B: Use the provided setup script**
```bash
./setup_pybind11.sh
```

**Option C: Manual setup**
```bash
git clone https://github.com/pybind/pybind11.git third_party/pybind11
```

### 3. Build the Project
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Build Targets

The build system creates several targets:

### Core Libraries
- **libcuda_metrics.so** - CUDA kernels library
- **libsurprise_metrics.so** - Main C++ library

### Executables
- **surprise_metrics_runner** - Command-line tool for running metrics calculations

### Python Bindings (if pybind11 is available)
- **pysurprise_metrics** - Python module for the library

## Troubleshooting

### Common Issues

1. **CUDA not found**
   - Ensure CUDA toolkit is installed and `nvcc` is in your PATH
   - Set `CUDA_ROOT` environment variable if needed

2. **pybind11 not found**
   - Run `./setup_pybind11.sh` to download pybind11
   - Or install via pip: `pip install pybind11`

3. **Missing OpenMP**
   - Install OpenMP development libraries:
     - Ubuntu/Debian: `sudo apt install libomp-dev`
     - CentOS/RHEL: `sudo yum install libgomp-devel`

4. **C++ Compiler too old**
   - Ensure you have GCC 9+ or Clang 10+
   - Update your compiler or use a newer toolchain

### Build Configuration Options

You can customize the build with CMake options:

```bash
# Disable Python bindings
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=OFF

# Specify CUDA architectures
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="70;75;80"

# Enable debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

## Testing the Build

After successful compilation, you can test the build:

```bash
# Test the command-line tool
./surprise_metrics_runner

# Test Python bindings (if built)
python -c "import pysurprise_metrics; print('Python bindings work!')"
```

## Performance Notes

- The library is optimized for NVIDIA GPUs with compute capability 7.0+
- AVX2/AVX512 SIMD instructions are used when available
- Multi-GPU support is available for large datasets

## Support

If you encounter build issues:
1. Check that all prerequisites are installed
2. Verify CUDA installation with `nvcc --version`
3. Check CMake configuration output for missing dependencies
4. Refer to the error messages in the build log

