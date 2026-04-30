from setuptools import setup, find_packages

setup(
    name="neural-platform",
    version="0.1.0",
    description="NeuralForge — a multi-framework platform for building, training, and deploying neural networks",
    author="NeuralForge",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "httpx>=0.24.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "datasets>=2.14.0",
        "pillow>=10.0.0",
        "pandas>=2.0.0",
        "python-multipart>=0.0.6",
        "websockets>=11.0",
        "aiofiles>=23.0.0",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.13.0"],
        "jax": ["jax>=0.4.0", "flax>=0.7.0"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "isort>=5.12.0"],
    },
    entry_points={
        "console_scripts": [
            "neural=neural_platform.cli.commands:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "neural_platform": ["web/static/*.html", "web/static/*.css", "web/static/*.js"],
    },
)
