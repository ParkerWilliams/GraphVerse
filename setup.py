from setuptools import find_packages, setup

setup(
    name="graphverse",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "networkx",
        "numpy",
        "torch",
        "matplotlib",
        "pandas",
    ],
    author="Parker Williams",
    author_email="parker.williams@gmail.com",
    description="A package for graph generation, random walks, and LLM training",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/graphverse",
)
