#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class LabelProbabilityFetcher:
    def __init__(self, class_probs, combination_probs):
        self.class_probs = class_probs
        self.combination_probs = combination_probs

    def multi_hot_to_one_hot(self, multi_hot_tensor):
        # Check if the multi-hot tensor is a zero vector
        if torch.all(multi_hot_tensor == 0):
            return multi_hot_tensor.unsqueeze(
                0
            )  # Return the zero vector as a 2D tensor
        indices = torch.nonzero(multi_hot_tensor)
        num_classes = multi_hot_tensor.size(0)
        num_active = indices.size(0)
        one_hot_tensor = torch.zeros((num_active, num_classes), dtype=torch.float32)
        one_hot_tensor[torch.arange(num_active), indices] = 1
        return one_hot_tensor

    def get_multi_hot_probability(self, multi_hot_tensor):
        if torch.all(multi_hot_tensor == 0):
            return 1.0
        multi_hot_str = str([int(x) for x in multi_hot_tensor.tolist()])
        multi_hot_str = multi_hot_str.replace(" ", "")
        return self.combination_probs[multi_hot_str]

    def get_one_hot_probability(self, one_hot_tensor):
        index = torch.argmax(one_hot_tensor).item()
        return self.class_probs[index]

    def process_label(self, multi_hot_tensor):
        multi_hot_prob = self.get_multi_hot_probability(multi_hot_tensor)
        one_hot_tensors = self.multi_hot_to_one_hot(multi_hot_tensor)
        one_hot_probs = [
            self.get_one_hot_probability(tensor) for tensor in one_hot_tensors
        ]

        return {
            "multi_hot_label": multi_hot_tensor,
            "multi_hot_weight": multi_hot_prob,
            "one_hot_labels": one_hot_tensors,
            "one_hot_weight": one_hot_probs,
        }


class MultiHotCrossEntropyLoss(nn.Module):
    def __init__(self, class_probs, combination_probs):
        super().__init__()
        self.fetcher = LabelProbabilityFetcher(class_probs, combination_probs)

    def forward(self, logits, multi_hot_labels):
        batch_size, _ = logits.shape
        device = logits.device

        total_loss = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            sample_logits = logits[i]
            multi_hot_label = multi_hot_labels[i]

            label_info = self.fetcher.process_label(multi_hot_label)
            one_hot_labels = label_info["one_hot_labels"]
            one_hot_probs = label_info["one_hot_weight"]
            multi_hot_prob = label_info["multi_hot_weight"]

            if torch.all(multi_hot_label == 0):
                one_hot_probs = [1.0]

            sample_loss = torch.zeros(1, device=device)
            for label, prob in zip(one_hot_labels, one_hot_probs):

                # Move label to the same device as the logits
                label = label.to(device)
                ce_loss = F.cross_entropy(
                    sample_logits.unsqueeze(0), torch.argmax(label).unsqueeze(0)
                )
                weighted_loss = ce_loss * prob
                sample_loss += weighted_loss
            sample_loss *= multi_hot_prob

            total_loss[i] = sample_loss
        return total_loss.mean()


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": partial(SoftTargetCrossEntropyLoss, normalize_targets=False),
    "contrastive_loss": ContrastiveLoss,
    "mse": nn.MSELoss,
    "multi_mse": MultipleMSELoss,
    "unique_combination_loss": MultiHotCrossEntropyLoss,
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
