from setuptools import setup, Extension
import pybind11

extension = Extension(
    "orderbook_cpp",
    sources=["cpp/bindings.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["/O2"] if __import__("sys").platform == "win32" else ["-O3"],
)

setup(
    name="orderbook_cpp",
    version="0.1.0",
    ext_modules=[extension],
)