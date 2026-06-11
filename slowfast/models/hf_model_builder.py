#!/usr/bin/env python3

"""Video classification models from the HuggingFace Hub.

Wraps AutoModelForVideoClassification (VideoMAE, TimeSformer, ViViT,
V-JEPA 2, ...) behind the slowfast model contract: a single-pathway input
list of [B, C, T, H, W] tensors in, raw logits out, with the head activation
applied only at eval time (matching ptv_model_builder.py).

Pretrained weights come from `from_pretrained`; leave
TRAIN.CHECKPOINT_FILE_PATH empty so the slowfast checkpoint loader does not
overwrite them. Saving/resuming slowfast checkpoints works as usual.
"""

import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForVideoClassification,
)

import slowfast.utils.logging as logging

from .build import MODEL_REGISTRY

logger = logging.get_logger(__name__)

# Repo-wide DATA.MEAN/STD defaults; HF models are pretrained with their own
# stats, so matching these exactly almost certainly means a config oversight.
_SLOWFAST_DEFAULT_MEAN = [0.45, 0.45, 0.45]
_SLOWFAST_DEFAULT_STD = [0.225, 0.225, 0.225]


def get_head_act(act_func):
    """
    Return the eval-time head activation given its config name.
    """
    if act_func == "softmax":
        return nn.Softmax(dim=1)
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func == "none":
        return nn.Identity()
    else:
        raise NotImplementedError(
            "{} is not supported as a head activation "
            "function.".format(act_func)
        )


@MODEL_REGISTRY.register()
class HFVideoModel(nn.Module):
    """
    HuggingFace video classification model behind the slowfast contract.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(HFVideoModel, self).__init__()
        assert (
            len(cfg.DATA.INPUT_CHANNEL_NUM) == 1
        ), "HFVideoModel supports a single input pathway."
        assert cfg.SOLVER.LAYER_DECAY == 1.0, (
            "Layer-wise lr decay is not supported for HF models (their "
            "param naming differs); set SOLVER.LAYER_DECAY to 1.0."
        )

        if cfg.HF.PRETRAINED:
            # ignore_mismatched_sizes re-initializes the classifier head at
            # NUM_CLASSES while keeping the pretrained backbone weights.
            self.model = AutoModelForVideoClassification.from_pretrained(
                cfg.HF.MODEL_NAME,
                num_labels=cfg.MODEL.NUM_CLASSES,
                ignore_mismatched_sizes=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                cfg.HF.MODEL_NAME, num_labels=cfg.MODEL.NUM_CLASSES
            )
            self.model = AutoModelForVideoClassification.from_config(config)

        if cfg.HF.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()

        self._check_data_cfg(cfg)
        self.post_act = get_head_act(cfg.MODEL.HEAD_ACT)

    def _check_data_cfg(self, cfg):
        """
        Fail fast on frame-count/crop-size mismatches; warn when DATA.MEAN/STD
        were left at the repo defaults, which no HF model is pretrained with.
        """
        hf_config = self.model.config
        expected_frames = getattr(
            hf_config, "num_frames", getattr(hf_config, "frames_per_clip", None)
        )
        if (
            expected_frames is not None
            and expected_frames != cfg.DATA.NUM_FRAMES
        ):
            raise ValueError(
                "{} expects {} frames but DATA.NUM_FRAMES is {}.".format(
                    cfg.HF.MODEL_NAME, expected_frames, cfg.DATA.NUM_FRAMES
                )
            )
        expected_size = getattr(hf_config, "image_size", None)
        if expected_size is not None and cfg.HF.PRETRAINED:
            for key, crop in (
                ("TRAIN_CROP_SIZE", cfg.DATA.TRAIN_CROP_SIZE),
                ("TEST_CROP_SIZE", cfg.DATA.TEST_CROP_SIZE),
            ):
                if crop != expected_size:
                    raise ValueError(
                        "{} expects {}x{} inputs but DATA.{} is {}.".format(
                            cfg.HF.MODEL_NAME,
                            expected_size,
                            expected_size,
                            key,
                            crop,
                        )
                    )
        if (
            list(cfg.DATA.MEAN) == _SLOWFAST_DEFAULT_MEAN
            and list(cfg.DATA.STD) == _SLOWFAST_DEFAULT_STD
        ):
            logger.warning(
                "DATA.MEAN/STD are the slowfast defaults; set them to the "
                "values of {}'s video processor (see configs/hf/ examples) "
                "or pretrained accuracy will silently degrade.".format(
                    cfg.HF.MODEL_NAME
                )
            )

    def no_weight_decay(self):
        # Substring-matched by the optimizer; covers pos-embed/cls-token
        # naming across VideoMAE/TimeSformer/ViViT/V-JEPA 2.
        return {"position_embeddings", "pos_embed", "cls_token"}

    def forward(self, x, bboxes=None):
        x = x[0]  # [B, C, T, H, W]
        # HF video models take pixel_values as [B, T, C, H, W].
        x = x.permute(0, 2, 1, 3, 4)
        logits = self.model(pixel_values=x).logits
        if not self.training:
            logits = self.post_act(logits)
        return logits
