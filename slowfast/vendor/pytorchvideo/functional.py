# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Vendored from pytorchvideo/transforms/functional.py (only convert_to_one_hot,
# which slowfast.vendor.pytorchvideo.soft_target_cross_entropy depends on).

import torch


def convert_to_one_hot(
    targets: torch.Tensor,
    num_class: int,
    label_smooth: float = 0.0,
) -> torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.

    Args:
        targets (torch.Tensor): Index labels to be converted.
        num_class (int): Total number of classes.
        label_smooth (float): Label smooth value for non-target classes. Label smooth
            is disabled by default (0).
    """
    assert torch.max(targets).item() < num_class, (
        "Class Index must be less than number of classes"
    )
    assert 0 <= label_smooth < 1.0, "Label smooth value needs to be between 0 and 1."

    non_target_value = label_smooth / num_class
    target_value = 1.0 - label_smooth + non_target_value
    one_hot_targets = torch.full(
        (targets.shape[0], num_class),
        non_target_value,
        dtype=torch.long if label_smooth == 0.0 else None,
        device=targets.device,
    )
    one_hot_targets.scatter_(1, targets.long().view(-1, 1), target_value)
    return one_hot_targets
