from setuptools import find_packages, setup
import os

def readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()

setup(
    name="graphverse",
    version="0.0.1",
    packages=find_packages(
        include=[
            "graphverse",
            "graphverse.*",
        ]
    ),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "pandas",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    author="Parker Williams",
    author_email="parker.williams@gmail.com",
    description="A package for graph generation, random walks, and LLM training",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/graphverse",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        # If you want to add CLI scripts in the future, do it here
        # "console_scripts": [
        #     "graphverse-cli=graphverse.cli:main",
        # ],
    },
)
