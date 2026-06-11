#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .contrastive import ContrastiveModel  # noqa
from .custom_video_model_builder import *  # noqa
from .masked import MaskMViT  # noqa
from .video_model_builder import MViT, ResNet, SlowFast  # noqa

try:
    from .ptv_model_builder import (
        PTVCSN,
        PTVX3D,
        PTVR2plus1D,
        PTVResNet,
        PTVSlowFast,
    )  # noqa
except Exception:
    # pytorchvideo is no longer a dependency; the PTV* wrapper models are
    # unavailable unless it is installed separately.
    pass

try:
    from .timm_model_builder import TimmVideoModel  # noqa
except ImportError:
    # timm not installed (e.g. legacy environment).
    pass

try:
    from .hf_model_builder import HFVideoModel  # noqa
except ImportError:
    # transformers not installed (e.g. legacy environment).
    pass
