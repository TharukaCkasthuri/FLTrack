from setuptools import setup, find_packages

setup(
    name="FLTrack",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.0.1",
        "numpy==1.25.2",
        "pandas==1.5.3",
        "tqdm==4.66.0",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "tensorboard==2.14.0",
    ],
    entry_points={
        "console_scripts": [
            "fltrack-cli = FLTrack.cli:main",
        ],
    },
    author="Tharuka Kasthuriarachchige",
    author_email="tak@bth.se",
    description="Client behavior tracking for federated learning",
    url="https://github.com/tharuka.ckasthuri/FLTrack",
)
