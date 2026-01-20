from setuptools import setup, find_packages

setup(
    name="dual-embedding-protein-localization",
    version="0.1.0",
    description="Protein subcellular localization using dual embeddings",
    author="Your Team",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
)
