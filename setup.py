#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="slowfast",
    version="1.0",
    author="FAIR",
    url="unknown",
    description="SlowFast Video Understanding",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "av>=12,<18",
        "matplotlib",
        "termcolor>=1.1",
        "simplejson",
        "tqdm",
        "psutil",
        "opencv-python",
        "pandas",
        "torch>=2.6",
        "torchvision>=0.21",
        "scikit-learn",
        "tensorboard",
        "fvcore",
        "iopath",
        "numpy<2",
        "timm>=1.0",
        "transformers>=4.53,<5",
    ],
    extras_require={
        "tensorboard_video_visualization": ["moviepy"],
        # Only needed for the demo person detector and detection visualization.
        "demo": ["detectron2"],
        # Only needed for MViT activation checkpointing / reversible blocks.
        "fairscale": ["fairscale"],
    },
    packages=find_packages(exclude=("configs", "tests")),
)
