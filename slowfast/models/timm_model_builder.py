#!/usr/bin/env python3

"""Video classification models built on timm image backbones.

Frames are encoded independently by a 2D timm backbone and aggregated over
time (mean or attention pooling), following the wrapper pattern of
ptv_model_builder.py: models take a single-pathway input list and return raw
logits, with the head activation applied only at eval time.
"""

import timm
import torch
import torch.nn as nn

import slowfast.utils.logging as logging

from .build import MODEL_REGISTRY

logger = logging.get_logger(__name__)


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
class TimmVideoModel(nn.Module):
    """
    Per-frame timm backbone + temporal aggregation + linear classifier.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(TimmVideoModel, self).__init__()
        assert (
            len(cfg.DATA.INPUT_CHANNEL_NUM) == 1
        ), "TimmVideoModel supports a single input pathway."

        self.backbone = timm.create_model(
            cfg.TIMM.MODEL_NAME,
            pretrained=cfg.TIMM.PRETRAINED,
            num_classes=0,  # pooled features, no classifier
            drop_path_rate=cfg.TIMM.DROP_PATH_RATE,
        )
        feat_dim = self.backbone.num_features
        self._check_normalization(cfg)

        if cfg.TIMM.FREEZE_BACKBONE:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.temporal_pool = cfg.TIMM.TEMPORAL_POOL
        if self.temporal_pool == "attention":
            self.temporal_query = nn.Parameter(torch.zeros(1, 1, feat_dim))
            nn.init.trunc_normal_(self.temporal_query, std=0.02)
            self.temporal_attn = nn.MultiheadAttention(
                feat_dim, cfg.TIMM.ATTN_POOL_HEADS, batch_first=True
            )
        elif self.temporal_pool != "mean":
            raise NotImplementedError(
                "Unsupported TIMM.TEMPORAL_POOL: {}".format(self.temporal_pool)
            )

        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_RATE)
        self.projection = nn.Linear(feat_dim, cfg.MODEL.NUM_CLASSES)
        self.post_act = get_head_act(cfg.MODEL.HEAD_ACT)

    def _check_normalization(self, cfg):
        """
        Warn when DATA.MEAN/STD don't match the backbone's pretraining stats
        (the repo default 0.45/0.225 silently degrades pretrained accuracy).
        """
        pretrained_cfg = getattr(self.backbone, "pretrained_cfg", None) or {}
        expected_mean = pretrained_cfg.get("mean")
        expected_std = pretrained_cfg.get("std")
        if expected_mean is None:
            return
        if any(
            abs(a - b) > 1e-3
            for a, b in zip(cfg.DATA.MEAN, expected_mean)
        ) or any(
            abs(a - b) > 1e-3 for a, b in zip(cfg.DATA.STD, expected_std)
        ):
            logger.warning(
                "DATA.MEAN/STD ({}/{}) do not match the {} pretraining "
                "stats ({}/{}); set them in the config to use the "
                "pretrained weights properly.".format(
                    cfg.DATA.MEAN,
                    cfg.DATA.STD,
                    cfg.TIMM.MODEL_NAME,
                    list(expected_mean),
                    list(expected_std),
                )
            )

    def no_weight_decay(self):
        skip = set()
        if hasattr(self.backbone, "no_weight_decay"):
            skip = {"backbone." + name for name in self.backbone.no_weight_decay()}
        if self.temporal_pool == "attention":
            skip.add("temporal_query")
        return skip

    def get_num_layers(self):
        """
        Block count for layer-wise lr decay (SOLVER.LAYER_DECAY < 1.0); only
        meaningful for block-structured backbones such as ViTs.
        """
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise NotImplementedError(
                "SOLVER.LAYER_DECAY < 1.0 requires a block-structured "
                "backbone (e.g. a ViT); {} has no `blocks` attribute, set "
                "SOLVER.LAYER_DECAY to 1.0.".format(type(self.backbone).__name__)
            )
        return len(blocks)

    def forward(self, x, bboxes=None):
        x = x[0]  # [B, C, T, H, W]
        batch, channels, time, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width
        )
        feats = self.backbone(x)  # [B*T, D]
        feats = feats.reshape(batch, time, -1)

        if self.temporal_pool == "attention":
            query = self.temporal_query.expand(batch, -1, -1)
            feats, _ = self.temporal_attn(query, feats, feats)
            feats = feats.squeeze(1)
        else:
            feats = feats.mean(dim=1)

        x = self.projection(self.dropout(feats))
        if not self.training:
            x = self.post_act(x)
        return x
