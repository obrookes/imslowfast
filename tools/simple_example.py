"""
This file presents a minimal example of background compensation on dummy data on the CPU
"""

import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

import slowfast.utils.weight_init_helper as init_helper

# import default config
from slowfast.config.defaults import get_cfg
from slowfast.models import (
    batchnorm_helper,
    head_helper,
    resnet_helper,
    stem_helper,
)  # noqa

torch.random.manual_seed(0)

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3)}


# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ]
}

_POOL1 = {
    "slow": [[1, 1, 1]],
}


class ResNetFGBGMixup(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNetFGBGMixup, self).__init__()
        self.norm_module = batchnorm_helper.get_norm(cfg)
        self.num_pathways = 1
        self.fg_bg_mixup_enable = cfg.FG_BG_MIXUP.ENABLE
        self.mix_on_eval = cfg.FG_BG_MIXUP.MIX_ON_EVAL
        self.sub_bg = cfg.FG_BG_MIXUP.SUBTRACT_BG.ENABLE

        self.sub_bg_alpha_max = cfg.FG_BG_MIXUP.SUBTRACT_BG.ALPHA_MAX
        self.concat_bg_frames = cfg.FG_BG_MIXUP.CONCAT_BG_FRAMES.ENABLE
        self.concat_bg_frames_ratio = cfg.FG_BG_MIXUP.CONCAT_BG_FRAMES.RATIO
        self.dataset = cfg.TRAIN.DATASET

        self._construct_network(cfg)
        init_helper.init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.RESNET.ZERO_INIT_FINAL_BN,
            cfg.RESNET.ZERO_INIT_FINAL_CONV,
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        self.cfg = cfg

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        # Based on profiling data of activation size, s1 and s2 have the activation sizes
        # that are 4X larger than the second largest. Therefore, checkpointing them gives
        # best memory savings. Further tuning is possible for better memory saving and tradeoffs
        # with recomputing FLOPs.
        self.s1 = s1
        self.s2 = s2

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        self.feat_agg = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.head = head_helper.ResNetBasicHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=(
                [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ]
            ),
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            detach_head=cfg.MODEL.DETACH_HEAD,
            detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
            cfg=cfg,
        )

        self.projection = self.head.projection

    def forward(self, x, alpha=0.0, beta=None, labels=None):

        emb_dict = {}  # fg_frames, bg_frames

        if self.dataset == "bkinetics":
            # Assert keys are 'concat_frames', and 'bg_frames'
            assert (
                "concat_frames" in x.keys() and "bg_frames" in x.keys()
            ), f"Keys: {x.keys()}"

            # Rename 'concat_frames' to 'fg_frames'
            x["fg_frames"] = x.pop("concat_frames")

        mask = x["mask"]

        for k, v in x.items():
            if (k != "mask") and (k != "utm"):
                if k == "bg_frames":
                    x = v[:]
                    x = self.s1(x)
                    x = self.s2(x)
                    y = []
                    for pathway in range(self.num_pathways):
                        pool = getattr(self, "pathway{}_pool".format(pathway))
                        y.append(pool(x[pathway]))
                    x = self.s3(y)
                    x = self.s4(x)
                    x = self.s5(x)
                    x = torch.cat(x, 1)
                    x = self.feat_agg(x)
                    x = torch.flatten(x, 1)
                    emb_dict[k] = x
                else:
                    x = v[:]  # avoid pass by reference
                    x = self.s1(x)
                    x = self.s2(x)
                    y = []  # Don't modify x list in place due to activation checkpoint.
                    for pathway in range(self.num_pathways):
                        pool = getattr(self, "pathway{}_pool".format(pathway))
                        y.append(pool(x[pathway]))
                    x = self.s3(y)
                    x = self.s4(x)
                    x = self.s5(x)
                    x = torch.cat(x, 1)
                    x = self.feat_agg(x)
                    x = torch.flatten(x, 1)
                    emb_dict[k] = x

        mask = mask.clone().detach().bool()

        if (self.training and self.fg_bg_mixup_enable) or (
            (not self.training) and (self.mix_on_eval)
        ):
            # Mix embeddings based on the batch
            embs = self.mix_fg_bg(
                emb_dict["fg_frames"], emb_dict["bg_frames"], mask, alpha
            )
        else:
            embs = emb_dict["fg_frames"]
        x = self.projection(embs)

        return x

    def mix_fg_bg(self, fg_embs, bg_embs, mask, alpha=None):
        """
        Process video embeddings based on the given criteria and UTM locations using PyTorch.

        Args:
        foreground_embeddings: torch.Tensor of shape (batch_size, embedding_dim)
        background_embeddings: torch.Tensor of shape (batch_size, embedding_dim)
        mask: torch.Tensor of shape (batch_size,), True for negative foregrounds
        alpha: float, alpha value for mixup

        Returns:
        processed_embeddings: torch.Tensor of shape (batch_size, embedding_dim)
        """

        # Create copies to avoid modifying the original tensors
        processed_embeddings = fg_embs.clone()

        # Create a boolean mask for positive foregrounds
        positive_mask = ~mask

        bg_embs_list = []
        bg_sub_embs_list = []

        positive_indices = torch.where(positive_mask)[0]
        for i in positive_indices:
            if self.sub_bg:
                if alpha > 0.0:
                    print("Subtracting background embeddings with alpha parameter")
                    # Subtract background embeddings with alpha
                    bg_emb = bg_embs[i] * (1 - alpha)

                else:
                    # Subtract background embeddings
                    bg_emb = bg_embs[i]

                fg_emb = fg_embs[i]
                bg_sub_emb = fg_emb - bg_emb

                processed_embeddings[i] = bg_sub_emb

                # Append the background embeddings for orthogonalisation
                bg_embs_list.append(bg_emb)
                bg_sub_embs_list.append(bg_sub_emb)

        return processed_embeddings


def load_config():
    # load slowfast model's default config
    default_cfg = get_cfg()
    # Create a dummy ResNet-50 configuration
    model_cfg = {
        "TRAIN": {
            "ENABLE": True,
            "DATASET": "bkinetics",
            "BATCH_SIZE": 2,
            "EVAL_PERIOD": 1,
            # "MIXED_PRECISION": False,
        },
        "TEST": {
            "ENABLE": False,
            "DATASET": "nkinetics",
        },
        "DATA": {
            "NUM_FRAMES": 4,
            "INPUT_CHANNEL_NUM": [3],
            "MULTI_LABEL": True,
            "ENSEMBLE_METHOD": "max",
        },
        "FG_BG_MIXUP": {
            "ENABLE": True,
            "SUBTRACT_BG": {
                "ENABLE": True,
                "ALPHA_MIN": 0.0,
                "ALPHA_MAX": 1.0,
                "SCHEDULER": "linear",
            },
        },
        "RESNET": {
            "ZERO_INIT_FINAL_BN": True,
            "DEPTH": 50,
            "NUM_BLOCK_TEMP_KERNEL": [[3], [4], [6], [3]],
        },
        "SOLVER": {
            "BASE_LR": 2.5e-1,
            "MAX_EPOCH": 10,
            "WARMUP_EPOCHS": 5.0,
            "WARMUP_START_LR": 2.5e-2,
        },
        "MODEL": {
            "NUM_CLASSES": 14,
            "ARCH": "slow",
            "MODEL_NAME": "ResNetFGBGMixup",
            "HEAD_ACT": "none",
            "LOSS_FUNC": "bce_logit",
            "DROPOUT_RATE": 0.5,
        },
        "NUM_GPUS": 1,
    }

    # Merge the example configuration with the default configuration
    default_cfg.merge_from_other_cfg(CN(model_cfg))
    return default_cfg


if __name__ == "__main__":

    # Load the configuration
    default_cfg = load_config()

    fg_frames = torch.randn(
        default_cfg.TRAIN.BATCH_SIZE,
        default_cfg.DATA.INPUT_CHANNEL_NUM[0],
        default_cfg.DATA.NUM_FRAMES,
        default_cfg.DATA.TRAIN_CROP_SIZE,
        default_cfg.DATA.TRAIN_CROP_SIZE,
    )
    bg_frames = torch.randn(
        default_cfg.TRAIN.BATCH_SIZE,
        default_cfg.DATA.INPUT_CHANNEL_NUM[0],
        default_cfg.DATA.NUM_FRAMES,
        default_cfg.DATA.TRAIN_CROP_SIZE,
        default_cfg.DATA.TRAIN_CROP_SIZE,
    )

    # Create dummy data with shape (batch_size, channels, num_frames, height, width)
    data = {
        "concat_frames": [fg_frames],
        "bg_frames": [bg_frames],
        "mask": torch.tensor(False),
    }

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    for k, v in data.items():
        if isinstance(v, list):
            data[k] = [item.to(device) for item in v]
        else:
            data[k] = v.to(device)

    print("Initialise model with following configuration:", default_cfg)
    model = ResNetFGBGMixup(cfg=default_cfg)
    model.to(device)
    output = model(data, alpha=0.5)

    print("inputs:")
    print("fg_frames:", fg_frames[0].shape)
    print("bg_frames:", {bg_frames[0].shape})

    print("output:", output[0].shape)
