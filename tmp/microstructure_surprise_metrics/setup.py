from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "pysurprise_metrics",
        ["python/bindings.cpp"],
        include_dirs=[
            "include",
            "/usr/local/cuda/include",
            pybind11.get_include(),
        ],
        libraries=["surprise_metrics", "cuda_metrics"],
        library_dirs=["build"],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="surprise_metrics",
    version="0.1.0",
    author="Your Name",
    description="High-performance surprise metrics for trading",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.8",
)
