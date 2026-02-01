from setuptools import setup, find_packages

setup(
    name="model-analyzer",
    version="1.0.0",
    description="Analyze TensorFlow Lite and ONNX models to extract operator statistics",
    author="Claude Code",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.8.0",
        "onnx>=1.12.0",
        "onnxruntime>=1.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "openpyxl>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "model-analyzer=model_analyzer.cli:main",
        ],
    },
)
