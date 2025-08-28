cmake_minimum_required(VERSION 3.16)
project(SurpriseMetrics LANGUAGES C CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add custom module path for Find*.cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# SIMD flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
CHECK_CXX_COMPILER_FLAG("-mavx512f" COMPILER_SUPPORTS_AVX512)

if(COMPILER_SUPPORTS_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl")
elseif(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -fopenmp")

# CUDA architecture selection (support multiple architectures)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# Optional external libraries - make them optional with fallbacks
find_package(Arrow QUIET)
find_package(Parquet QUIET)
find_package(HDF5 QUIET)
find_package(Eigen3 QUIET)

# If Eigen3 not found via package, try to include as header-only
if(NOT Eigen3_FOUND)
    if(EXISTS "/usr/include/eigen3")
        include_directories(/usr/include/eigen3)
        set(EIGEN3_FOUND TRUE)
    elseif(EXISTS "/usr/local/include/eigen3")
        include_directories(/usr/local/include/eigen3)
        set(EIGEN3_FOUND TRUE)
    else()
        message(WARNING "Eigen3 not found. Some features will be disabled.")
    endif()
endif()

# Set preprocessor definitions based on available libraries
if(Arrow_FOUND)
    add_definitions(-DHAS_ARROW=1)
endif()
if(Parquet_FOUND)
    add_definitions(-DHAS_PARQUET=1)
endif()
if(HDF5_FOUND)
    add_definitions(-DHAS_HDF5=1)
endif()
if(EIGEN3_FOUND)
    add_definitions(-DHAS_EIGEN=1)
endif()

# CUDA sources
set(CUDA_SOURCES
    src/cuda_kernels.cu
)

# C++ sources - core sources only
set(CPP_SOURCES
    src/surprise_metrics.cpp
    src/simd_ops.cpp
)

# Add optional sources based on available libraries
if(Arrow_FOUND OR Parquet_FOUND)
    list(APPEND CPP_SOURCES src/polygon_parser.cpp)
endif()

# Create CUDA library
cuda_add_library(cuda_metrics SHARED ${CUDA_SOURCES})
target_link_libraries(cuda_metrics ${CUDA_LIBRARIES})

# Create main library
add_library(surprise_metrics SHARED ${CPP_SOURCES})

# Build target link libraries list
set(SURPRISE_METRICS_LIBS
    cuda_metrics
    ${CUDA_LIBRARIES}
    OpenMP::OpenMP_CXX
    Threads::Threads
)

# Add optional libraries if found
if(Arrow_FOUND)
    list(APPEND SURPRISE_METRICS_LIBS Arrow::arrow_shared)
endif()
if(Parquet_FOUND)
    list(APPEND SURPRISE_METRICS_LIBS Parquet::parquet_shared)
endif()
if(HDF5_FOUND)
    list(APPEND SURPRISE_METRICS_LIBS ${HDF5_LIBRARIES})
endif()
if(TARGET Eigen3::Eigen)
    list(APPEND SURPRISE_METRICS_LIBS Eigen3::Eigen)
endif()

target_link_libraries(surprise_metrics ${SURPRISE_METRICS_LIBS})

# Python bindings - make optional
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(pysurprise_metrics python/bindings.cpp)
    target_link_libraries(pysurprise_metrics PRIVATE surprise_metrics)
else()
    message(WARNING "pybind11 not found. Python bindings will not be built.")
endif()

# Executable
add_executable(surprise_metrics_runner src/main.cpp)
target_link_libraries(surprise_metrics_runner surprise_metrics)

# Print configuration summary
message(STATUS "")
message(STATUS "========================================")
message(STATUS "SurpriseMetrics Configuration Summary:")
message(STATUS "========================================")
message(STATUS "CUDA Found: YES")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenMP Found: YES")
message(STATUS "Python Found: YES (${Python_VERSION})")
message(STATUS "AVX2 Support: ${COMPILER_SUPPORTS_AVX2}")
message(STATUS "AVX512 Support: ${COMPILER_SUPPORTS_AVX512}")
message(STATUS "Arrow Found: ${Arrow_FOUND}")
message(STATUS "Parquet Found: ${Parquet_FOUND}")
message(STATUS "HDF5 Found: ${HDF5_FOUND}")
message(STATUS "Eigen3 Found: ${EIGEN3_FOUND}")
message(STATUS "pybind11 Found: ${pybind11_FOUND}")
message(STATUS "========================================")
message(STATUS "")
