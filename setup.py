"""
Setup script for GSPO-based Unit Test Generation project.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="gspo-utg",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="GSPO-based Unit Test Generation and Refinement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gspo_utg_tesis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "peft>=0.6.0",
        "accelerate>=0.24.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "javalang>=0.13.0",
        "scikit-learn>=1.3.0",
        "tensorboard>=2.14.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "wandb": ["wandb>=0.16.0"],
        "quantization": ["bitsandbytes>=0.41.0"],
    },
    entry_points={
        "console_scripts": [
            "gspo-utg=experiments.run_experiment:main",
        ],
    },
)
