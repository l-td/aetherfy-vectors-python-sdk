"""
Setup configuration for aetherfy-vectors Python SDK.

Package configuration and metadata for PyPI distribution.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Read version from package
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "aetherfy_vectors", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split('"')[1]
    return "1.0.0"

setup(
    name="aetherfy-vectors",
    version=get_version(),
    description="Global vector database client with automatic replication and sub-50ms latency worldwide",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Aetherfy",
    author_email="developers@aetherfy.com",
    url="https://github.com/aetherfy/aetherfy-vectors-python",
    project_urls={
        "Homepage": "https://aetherfy.com",
        "Documentation": "https://docs.aetherfy.com/vectors",
        "Source": "https://github.com/aetherfy/aetherfy-vectors-python",
        "Bug Reports": "https://github.com/aetherfy/aetherfy-vectors-python/issues",
        "API Reference": "https://vectors.aetherfy.com/docs",
    },
    license="MIT",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.8.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.8.0",
            "responses>=0.21.0",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Database",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        "Typing :: Typed",
    ],
    keywords=[
        "vector database",
        "similarity search", 
        "machine learning",
        "artificial intelligence",
        "AI",
        "ML",
        "embeddings",
        "qdrant",
        "global database",
        "edge computing",
        "vector search",
        "semantic search",
        "neural search",
        "recommendation systems",
        "retrieval augmented generation",
        "RAG",
    ],
    entry_points={
        "console_scripts": [
            # Future: Add CLI tool if needed
            # "aetherfy-vectors=aetherfy_vectors.cli:main",
        ],
    },
    zip_safe=False,
    test_suite="tests",
    tests_require=[
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0", 
        "pytest-mock>=3.8.0",
    ],
    # Package data
    package_data={
        "aetherfy_vectors": [
            "py.typed",  # Indicates this package has type hints
        ],
    },
    # Additional metadata for better discoverability
    platforms=["any"],
    maintainer="Aetherfy Team",
    maintainer_email="developers@aetherfy.com",
    download_url="https://pypi.org/project/aetherfy-vectors/",
)