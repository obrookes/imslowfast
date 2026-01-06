#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from functools import partial

import torch
import torch.nn as nn
from pytorchvideo.losses.soft_target_cross_entropy import (
    SoftTargetCrossEntropyLoss,
)


class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, dummy_labels=None):
        targets = torch.zeros(inputs.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss(reduction=self.reduction).cuda()(inputs, targets)
        return loss


class MultipleMSELoss(nn.Module):
    """
    Compute multiple mse losses and return their average.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MultipleMSELoss, self).__init__()
        self.mse_func = nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        loss_sum = 0.0
        multi_loss = []
        for xt, yt in zip(x, y):
            if isinstance(yt, (tuple,)):
                if len(yt) == 2:
                    yt, wt = yt
                    lt = "mse"
                elif len(yt) == 3:
                    yt, wt, lt = yt
                else:
                    raise NotImplementedError
            else:
                wt, lt = 1.0, "mse"
            if lt == "mse":
                loss = self.mse_func(xt, yt)
            else:
                raise NotImplementedError
            loss_sum += loss * wt
            multi_loss.append(loss)
        return loss_sum, multi_loss


class AssumeNegativeLabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon=0.1, reduction="mean"):
        super(AssumeNegativeLabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, probabilities, targets):
        """
        Compute the Assume Negative Label Smoothing Loss.

        Args:
        probabilities (torch.Tensor): Model predictions after sigmoid (fn), shape (batch_size, num_classes)
        targets (torch.Tensor): Target labels (zn), shape (batch_size, num_classes)

        Returns:
        torch.Tensor: Computed AN-LS loss
        """
        # Ensure inputs are float tensors
        probabilities = probabilities.float()
        targets = targets.float()

        # Get the number of classes (L)
        num_classes = probabilities.size(1)

        # Apply label smoothing
        smoothed_targets = targets * (1 - self.epsilon) + (1 - targets) * (
            self.epsilon / 2
        )

        # Clip probabilities to avoid log(0) and log(1)
        probabilities = torch.clamp(probabilities, 1e-7, 1 - 1e-7)

        # Compute loss
        loss = -(
            smoothed_targets * torch.log(probabilities)
            + (1 - smoothed_targets) * torch.log(1 - probabilities)
        )

        # Average over classes (this implements the 1/L normalization)
        loss = loss.sum(dim=1) / num_classes

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class WeakAssumeNegativeLoss(nn.Module):
    def __init__(self, gamma=1.0, epsilon=1e-7, reduction="mean"):
        super(WeakAssumeNegativeLoss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, probabilities, targets):
        """
        Compute the  Weak Assume Negative (WAN) loss.

        Args:
        probabilities (torch.Tensor): Model predictions after sigmoid (fn), shape (batch_size, num_classes)
        targets (torch.Tensor): Target labels (zn), shape (batch_size, num_classes)

        Returns:
        torch.Tensor: Computed WAN loss
        """
        # Ensure inputs are float tensors
        probabilities = probabilities.float()
        targets = targets.float()

        # Clip probabilities to avoid log(0) and log(1)
        probabilities = torch.clamp(probabilities, self.epsilon, 1 - self.epsilon)

        # Compute positive and negative parts of the loss
        pos_loss = targets * torch.log(probabilities)
        neg_loss = (1 - targets) * torch.log(1 - probabilities)

        # Apply weighting to negative loss
        neg_loss = self.gamma * neg_loss

        # Combine positive and negative losses
        loss_per_sample = -(pos_loss + neg_loss)

        # Apply reduction
        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:  # 'none'
            return loss_per_sample


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": partial(SoftTargetCrossEntropyLoss, normalize_targets=False),
    "contrastive_loss": ContrastiveLoss,
    "mse": nn.MSELoss,
    "multi_mse": MultipleMSELoss,
    "anls": AssumeNegativeLabelSmoothingLoss,
    "wan": WeakAssumeNegativeLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
