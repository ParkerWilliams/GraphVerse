from setuptools import find_packages, setup
import os

def readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()

setup(
    name="graphverse",
    version="1.0.0",
    packages=find_packages(
        include=[
            "graphverse",
            "graphverse.*",
        ]
    ),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "pre-commit>=2.15.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    author="Parker Williams",
    author_email="parker.williams@gmail.com", 
    description="How Small LLMs Learn Graph Traversal Rules They've Never Seen",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/graphverse",
    project_urls={
        "Documentation": "https://github.com/yourusername/graphverse/docs",
        "Bug Reports": "https://github.com/yourusername/graphverse/issues",
        "Source": "https://github.com/yourusername/graphverse",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine learning, transformers, graph neural networks, rule learning, interpretability, AI safety",
    entry_points={
        "console_scripts": [
            "graphverse-run=run_analysis:main",
        ],
    },
)
