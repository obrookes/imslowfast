#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import math
import pickle as pkl
import pprint

import numpy as np
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.models.contrastive import (
    contrastive_forward,
    contrastive_parameter_surgery,
)
from slowfast.models.head_helper import ResNetBasicHead
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule

logger = logging.get_logger(__name__)


def inverse_mapping(x, low=1, high=5):
    # Ensure x is within the valid range
    x = torch.clamp(x, low, high)

    # Calculate the inverse mapping
    return low + high - x


def calculate_loss_with_pseudo_labels(preds, labels, loss_func, store_dict):
    """
    Calculate the loss between the predictions and the original labels using pseudo labels.

    Args:
    preds (torch.Tensor): Tensor of model predictions.
    labels (torch.Tensor): Tensor of original labels.
    store_dict (dict): Dictionary containing pseudo labels for each original label.

    Returns:
    torch.Tensor: Loss value.
    """
    total_loss = 0
    for i in range(len(labels)):
        original_label = labels[i].cpu().numpy()
        num_labels_present = torch.tensor(int(sum(original_label)), dtype=int)
        if num_labels_present > 0:
            # Select k random pseudo labels
            pseudo_labels = store_dict[str(original_label)]["pseudo_labels"]
            # pseudo_probs = store_dict[str(original_label)]["probs"]
            idxs = np.random.choice(
                len(pseudo_labels),
                int(inverse_mapping(num_labels_present)),
                replace=False,
            )
            # TODO: weight losses by chained sequence probability
            for idx in idxs:
                loss = loss_func(
                    preds[i],
                    torch.tensor(
                        pseudo_labels[idx], dtype=torch.float32, device=preds[i].device
                    ),
                )
                total_loss += loss
        else:
            continue
    return total_loss


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
    pseudo_labels=None,
    alpha=0.0,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
        alpha (float): alpha value for FG-BG mixup.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    # with torch.autograd.set_detect_anomaly(True):
    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    if cfg.MODEL.FROZEN_BN:
        misc.frozen_bn_stats(model)

    # Explicitly declare reduction to mean.
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
    for cur_iter, (inputs, labels, index, time, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            elif isinstance(inputs, (dict,)):
                for key, val in inputs.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            if isinstance(val[i], (list,)):
                                for j in range(len(val[i])):
                                    val[i][j] = val[i][j].cuda(non_blocking=True)
                            else:
                                try:
                                    val[i] = val[i].cuda(non_blocking=True)
                                except:
                                    continue
                    else:
                        inputs[key] = val.cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if isinstance(labels, dict):
                for key, val in labels.items():
                    labels[key] = val.cuda(non_blocking=True)
            elif not isinstance(labels, list):
                labels = labels.cuda(non_blocking=True)
                index = index.cuda(non_blocking=True)
                time = time.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if not isinstance(val[i], str):
                            val[i] = val[i].cuda(non_blocking=True)
                        else:
                            continue
                else:
                    meta[key] = val.cuda(non_blocking=True)

        try:
            batch_size = (
                inputs[0][0].size(0)
                if isinstance(inputs[0], list)
                else inputs[0].size(0)
            )
        except:
            try:
                try:
                    batch_size = (
                        inputs["fg_frames"][0].size(0)
                        if isinstance(inputs, dict)
                        else inputs["fg_frames"].size(0)
                    )
                except:
                    batch_size = (
                        inputs["concat_frames"][0].size(0)
                        if isinstance(inputs, dict)
                        else inputs["fg_frames"].size(0)
                    )
            except:
                batch_size = (
                    inputs["f1"][0].size(0)
                    if isinstance(inputs, dict)
                    else inputs["f1"].size(0)
                )
        # Update the learning rate.
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            # Explicitly declare reduction to mean.
            perform_backward = True
            optimizer.zero_grad()

            # Forward pass model
            if cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                (
                    model,
                    preds,
                    partial_loss,
                    perform_backward,
                ) = contrastive_forward(
                    model, cfg, inputs, index, time, epoch_exact, scaler
                )
            elif cfg.DETECTION.ENABLE:
                # Compute the predictions.
                preds = model(inputs, meta["boxes"])
            elif cfg.MASK.ENABLE:
                preds, labels = model(inputs)
            elif cfg.AUG.MANIFOLD_MIXUP:
                if cfg.AUG.MANIFOLD_MIXUP_PAIRS:
                    preds, y_a, y_b, lam = model(inputs, labels)
                elif cfg.AUG.MANIFOLD_MIXUP_TRIPLETS:
                    preds, y_a, y_b, y_c, lam1, lam2 = model(inputs, labels)
                else:
                    raise NotImplementedError(
                        "Manifold Mixup requires pairs or triplets"
                    )
            elif cfg.FGFG_MIXUP.ENABLE:
                preds, y_a, y_b, lam = model(inputs, labels)
            elif cfg.FG_BG_MIXUP.ENABLE:
                if cfg.FG_BG_MIXUP.SUBTRACT_BG.APPLY_CLASSWISE.ENABLE:
                    if cfg.FG_BG_MIXUP.SUBTRACT_BG.ENABLE:
                        if (
                            cfg.FG_BG_MIXUP.ADD_BG2.ENABLE
                            and cur_epoch >= cfg.FG_BG_MIXUP.ADD_BG2.START_FROM_EPOCH
                        ):
                            beta = 1 - alpha
                            if cfg.FG_BG_MIXUP.SUBTRACT_BG.ORTHO_EMBS:
                                preds, loss_ortho = model(
                                    inputs, alpha, beta, labels=labels
                                )
                            else:
                                preds = model(inputs, alpha, beta, labels=labels)
                        else:
                            if cfg.FG_BG_MIXUP.SUBTRACT_BG.ORTHO_EMBS:
                                preds, loss_ortho = model(inputs, alpha, labels=labels)
                            else:
                                preds = model(inputs, alpha, labels=labels)
                    elif (
                        cfg.FG_BG_MIXUP.ADD_BG.ENABLE
                        and cfg.FG_BG_MIXUP.SUBTRACT_BG.ENABLE is False
                    ):
                        preds = model(inputs, alpha, labels=labels)
                else:
                    if cfg.FG_BG_MIXUP.SUBTRACT_BG.ENABLE:
                        if (
                            cfg.FG_BG_MIXUP.ADD_BG2.ENABLE
                            and cur_epoch >= cfg.FG_BG_MIXUP.ADD_BG2.START_FROM_EPOCH
                        ):
                            beta = 1 - alpha
                            if cfg.FG_BG_MIXUP.SUBTRACT_BG.ORTHO_EMBS:
                                preds, loss_ortho = model(inputs, alpha, beta)
                            else:
                                preds = model(inputs, alpha, beta)
                        else:
                            if cfg.FG_BG_MIXUP.SUBTRACT_BG.ORTHO_EMBS:
                                preds, loss_ortho = model(inputs, alpha)
                            else:
                                preds = model(inputs, alpha)
                    elif (
                        cfg.FG_BG_MIXUP.ADD_BG.ENABLE
                        and cfg.FG_BG_MIXUP.SUBTRACT_BG.ENABLE is False
                    ):
                        preds = model(inputs, alpha)

                    else:
                        preds = model(inputs, alpha)

            elif cfg.FRAMEWISE_MIXUP.ENABLE:
                preds, lam, index = model(inputs)
            else:
                preds = model(inputs)

            # Get labels and compute the loss.
            if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                labels = torch.zeros(
                    preds.size(0), dtype=labels.dtype, device=labels.device
                )
            if cfg.MODEL.MODEL_NAME == "ContrastiveModel" and partial_loss:
                loss = partial_loss
            elif cfg.AUG.MANIFOLD_MIXUP:
                if cfg.AUG.MANIFOLD_MIXUP_PAIRS:
                    l = lam * loss_fun(preds, y_a) + (1 - lam) * loss_fun(preds, y_b)
                    loss = l.mean()
                elif cfg.AUG.MANIFOLD_MIXUP_TRIPLETS:
                    l = (
                        lam1 * loss_fun(preds, y_a)
                        + lam2 * loss_fun(preds, y_b)
                        + (1 - lam1 - lam2) * loss_fun(preds, y_c)
                    )
                    loss = l.mean()
            elif cfg.DATA.PSEUDO_LABELS:
                loss = (
                    calculate_loss_with_pseudo_labels(
                        preds, labels, loss_fun, pseudo_labels
                    )
                    * cfg.DATA.PSEUDO_LABELS_WEIGHT
                ) + loss_fun(preds, labels)
            elif cfg.FGFG_MIXUP.ENABLE:
                loss = lam * loss_fun(preds, y_a) + (1 - lam) * loss_fun(preds, y_b)
                loss = loss.mean()
            elif cfg.FRAMEWISE_MIXUP.ENABLE:
                y_a, y_b = labels, labels[index]
                if lam.size(1) > 1:
                    # For independent frame mixing
                    lam = lam.squeeze(-1)  # Shape: (B, T)
                    loss = 0
                    for t in range(preds.size(1)):  # Iterate over time steps
                        loss_t = lam[:, t] * loss_fun(preds[:, t], y_a) + (
                            1 - lam[:, t]
                        ) * loss_fun(preds[:, t], y_b)
                        loss += loss_t.mean()
                    loss /= preds.size(1)  # Average over time steps
                else:
                    # For single lambda per sample
                    # lam = lam.squeeze()  # Shape: (B,)
                    loss = lam * loss_fun(preds, y_a) + (1 - lam) * loss_fun(preds, y_b)
                    loss = loss.mean()
            else:
                # Compute the loss.
                if cfg.FG_BG_MIXUP.SUBTRACT_BG.ORTHO_EMBS:
                    assert len(preds) == len(labels)
                    loss = loss_fun(preds, labels) + loss_ortho
                else:
                    loss = loss_fun(preds, labels)

        loss_extra = None
        if isinstance(loss, (list, tuple)):
            loss, loss_extra = loss

        # check Nan Loss.
        misc.check_nan_losses(loss)

        if perform_backward:
            scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            grad_norm = torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        else:
            grad_norm = optim.get_grad_norm_(model.parameters())
        # Update the parameters. (defaults to True)
        model, update_param = contrastive_parameter_surgery(
            model, cfg, epoch_exact, cur_iter
        )
        if update_param:
            scaler.step(optimizer)
        scaler.update()

        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds = preds.detach()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm = du.all_reduce([loss, grad_norm])
                    if cfg.FGFG_MIXUP.ENABLE:
                        preds, labels = du.all_gather([preds, labels["y1"]])
                    else:
                        preds, labels = du.all_gather([preds, labels])
                # Copy the stats from GPU to CPU (sync point).
                loss, grad_norm = (
                    loss.item(),
                    grad_norm.item(),
                )
            elif cfg.MASK.ENABLE:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm = du.all_reduce([loss, grad_norm])
                    if loss_extra:
                        loss_extra = du.all_reduce(loss_extra)
                loss, grad_norm, top1_err, top5_err = (
                    loss.item(),
                    grad_norm.item(),
                    0.0,
                    0.0,
                )
                if loss_extra:
                    loss_extra = [one_loss.item() for one_loss in loss_extra]
            else:
                # Compute the errors.
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm, top1_err, top5_err = du.all_reduce(
                        [loss.detach(), grad_norm, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, grad_norm, top1_err, top5_err = (
                    loss.item(),
                    grad_norm.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            if cfg.FGFG_MIXUP.ENABLE:
                # Update and log stats.
                train_meter.update_predictions(preds.detach(), labels.detach())
            else:
                # Update and log stats.
                train_meter.update_predictions(preds.detach(), labels.detach())
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                grad_norm,
                batch_size
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                loss_extra,
            )
            # write to tensorboard format if available.
            if writer is not None:
                if cfg.DATA.MULTI_LABEL:
                    writer.add_scalars(
                        {
                            "Train/loss": loss,
                            "Train/lr": lr,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )
                else:
                    writer.add_scalars(
                        {
                            "Train/loss": loss,
                            "Train/lr": lr,
                            "Train/Top1_err": top1_err,
                            "Train/Top5_err": top5_err,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )
        train_meter.iter_toc()  # do measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        torch.cuda.synchronize()
        train_meter.iter_tic()

        # for the last iteration, we need to update the model parameters
        # and log the stats.

        # if cur_iter == data_size:
        #    bg_model = copy.deepcopy(model)

    del inputs

    # in case of fragmented memory
    torch.cuda.empty_cache()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)

    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DATA.MULTI_LABEL:
            writer.add_scalars(
                {
                    "Train/micro_mAP": train_meter.micro_map,
                    "Train/macro_mAP": train_meter.macro_map,
                },
                global_step=cur_epoch,
            )
            writer.add_scalars({"Train/APs": train_meter.aps}, global_step=cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(
    val_loader, model, val_meter, cur_epoch, cfg, train_loader, writer, alpha=0.0
):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, index, time, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            elif isinstance(inputs, (dict,)):
                for key, val in inputs.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        inputs[key] = val.cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if isinstance(labels, dict):
                for key, val in labels.items():
                    labels[key] = val.cuda(non_blocking=True)
            else:
                labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if not isinstance(val[i], str):
                            val[i] = val[i].cuda(non_blocking=True)
                        else:
                            continue
                else:
                    meta[key] = val.cuda(non_blocking=True)
            index = index.cuda()
            time = time.cuda()
        try:
            batch_size = (
                inputs[0][0].size(0)
                if isinstance(inputs[0], list)
                else inputs[0].size(0)
            )
        except:
            try:
                try:
                    batch_size = (
                        inputs["fg_frames"][0].size(0)
                        if isinstance(inputs, dict)
                        else inputs["fg_frames"].size(0)
                    )
                except:
                    batch_size = (
                        inputs["concat_frames"][0].size(0)
                        if isinstance(inputs, dict)
                        else inputs["fg_frames"].size(0)
                    )
            except:
                batch_size = (
                    inputs["f1"][0].size(0)
                    if isinstance(inputs, dict)
                    else inputs["f1"].size(0)
                )
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                if not cfg.CONTRASTIVE.KNN_ON:
                    return
                train_labels = (
                    model.module.train_labels
                    if hasattr(model, "module")
                    else model.train_labels
                )
                yd, yi = model(inputs, index, time)
                K = yi.shape[1]
                C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)
                retrieval_one_hot = torch.zeros((batch_size * K, C)).cuda()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
                probs = torch.mul(
                    retrieval_one_hot.view(batch_size, -1, C),
                    yd_transform.view(batch_size, -1, 1),
                )
                preds = torch.sum(probs, 1)
            elif cfg.AUG.MANIFOLD_MIXUP:
                preds = model(inputs, labels)
            elif cfg.FGFG_MIXUP.ENABLE:
                preds = model(inputs, labels)
            elif cfg.FG_BG_MIXUP.ENABLE:
                if cfg.FG_BG_MIXUP.ADD_BG2.ENABLE and cur_epoch >= (
                    cfg.FG_BG_MIXUP.ADD_BG2.START_FROM_EPOCH
                ):
                    beta = 1 - alpha
                    preds = model(inputs, alpha, beta)
                else:
                    preds = model(inputs, alpha)
            elif cfg.FRAMEWISE_MIXUP.ENABLE:
                preds = model(inputs)
            else:
                preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    if cfg.FGFG_MIXUP.ENABLE:
                        preds, labels = du.all_gather([preds, labels["y1"]])
                    else:
                        preds, labels = du.all_gather([preds, labels])
            else:
                if cfg.DATA.IN22k_VAL_IN1K != "":
                    preds = preds[:, :1000]
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    batch_size
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DATA.MULTI_LABEL:
            writer.add_scalars(
                {
                    "Val/micro_mAP": val_meter.micro_map,
                    "Val/macro_mAP": val_meter.macro_map,
                },
                global_step=cur_epoch,
            )
            writer.add_scalars({"Val/APs": val_meter.aps}, global_step=cur_epoch)
        if cfg.DETECTION.ENABLE:
            writer.add_scalars({"Val/mAP": val_meter.full_map}, global_step=cur_epoch)
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [label.clone().detach() for label in val_meter.all_labels]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(preds=all_preds, labels=all_labels, global_step=cur_epoch)

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        flops, params = misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(cfg, "train", is_precise_bn=True)
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if cfg.FG_BG_MIXUP.SUBTRACT_BG.ENABLE is True:
        if cfg.FG_BG_MIXUP.SUBTRACT_BG.SCHEDULER == "exp":
            alpha_scheduler = torch.logspace(-10, 0, cfg.SOLVER.MAX_EPOCH, base=torch.e)

        elif cfg.FG_BG_MIXUP.SUBTRACT_BG.SCHEDULER == "linear":
            alpha_scheduler = torch.linspace(
                cfg.FG_BG_MIXUP.SUBTRACT_BG.ALPHA_MIN,
                cfg.FG_BG_MIXUP.SUBTRACT_BG.ALPHA_MAX,
                cfg.SOLVER.MAX_EPOCH,
            )
    elif cfg.FG_BG_MIXUP.ADD_BG.ENABLE is True:
        if cfg.FG_BG_MIXUP.ADD_BG.SCHEDULER == "linear":
            alpha_scheduler = torch.linspace(
                cfg.FG_BG_MIXUP.ADD_BG.ALPHA_MIN,
                cfg.FG_BG_MIXUP.ADD_BG.ALPHA_MAX,
                cfg.SOLVER.MAX_EPOCH,
            )
    else:
        alpha_scheduler = None

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    flops, params = 0.0, 0.0
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        flops, params = misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task=cfg.TASK)
        if last_checkpoint is not None:
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
            start_epoch = checkpoint_epoch + 1
        elif "ssl_eval" in cfg.TASK:
            last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task="ssl")
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
                epoch_reset=True,
                clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            )
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = 0
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    if (
        cfg.TASK == "ssl"
        and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        and cfg.CONTRASTIVE.KNN_ON
    ):
        if hasattr(model, "module"):
            model.module.init_knn_labels(train_loader)
        else:
            model.init_knn_labels(train_loader)

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Reinitialise classifier head if required
    if cfg.MODEL.REINIT_HEAD:
        print(f"Reinitialising head of model {cfg.MODEL.ARCH}...")
        assert isinstance(model.head, ResNetBasicHead), "Head must be a ResNetBasicHead"
        model.head.reset_weights()

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cur_epoch > 0 and cfg.DATA.LOADER_CHUNK_SIZE > 0:
            num_chunks = math.ceil(
                cfg.DATA.LOADER_CHUNK_OVERALL_SIZE / cfg.DATA.LOADER_CHUNK_SIZE
            )
            skip_rows = (cur_epoch) % num_chunks * cfg.DATA.LOADER_CHUNK_SIZE
            logger.info(
                f"=================+++ num_chunks {num_chunks} skip_rows {skip_rows}"
            )
            cfg.DATA.SKIP_ROWS = skip_rows
            logger.info(f"|===========| skip_rows {skip_rows}")
            train_loader = loader.construct_loader(cfg, "train")
            loader.shuffle_dataset(train_loader, cur_epoch)

        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(
                        cfg.OUTPUT_DIR, task=cfg.TASK
                    )
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer)

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        if hasattr(train_loader.dataset, "_set_epoch_num"):
            train_loader.dataset._set_epoch_num(cur_epoch)

        # Pseudo labels
        if cfg.DATA.PSEUDO_LABELS:
            with open(cfg.DATA.PSEUDO_LABELS, "rb") as f:
                pseudo_labels = pkl.load(f)
        else:
            pseudo_labels = None

        # Train for one epoch.
        epoch_timer.epoch_tic()
        if alpha_scheduler is not None:
            alpha = alpha_scheduler[cur_epoch]
        else:
            alpha = 0.0

        if (
            cfg.MODEL.MODEL_NAME == "DualResNetFGBG"
            or cfg.MODEL.MODEL_NAME == "DualMViTFGBG"
        ):
            # print("DualMViTFGBG")
            if cfg.NUM_GPUS > 1:
                print("Loading FG model")
                cu.load_checkpoint(
                    cfg.TRAIN.FG_MODEL_CHECKPOINT_FILE_PATH,
                    model.module.fg_model,
                    False,
                    None,
                    inflation=False,
                    epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
                    convert_from_caffe2=cfg.TRAIN.FG_MODEL_CHECKPOINT_TYPE == "caffe2",
                    image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
                )

                print("Loading BG model")
                cu.load_checkpoint(
                    cfg.TRAIN.BG_MODEL_CHECKPOINT_FILE_PATH,
                    model.module.bg_model,
                    False,
                    None,
                    inflation=False,
                    epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
                    convert_from_caffe2=cfg.TRAIN.BG_MODEL_CHECKPOINT_TYPE == "caffe2",
                    image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
                )

            else:
                print("Loading FG model")
                cu.load_checkpoint(
                    cfg.TRAIN.FG_MODEL_CHECKPOINT_FILE_PATH,
                    model.fg_model,
                    False,
                    None,
                    inflation=False,
                    epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
                    convert_from_caffe2=cfg.TRAIN.FG_MODEL_CHECKPOINT_TYPE == "caffe2",
                    image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
                )

                print("Loading BG model")
                cu.load_checkpoint(
                    cfg.TRAIN.BG_MODEL_CHECKPOINT_FILE_PATH,
                    model.bg_model,
                    False,
                    None,
                    inflation=False,
                    epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
                    convert_from_caffe2=cfg.TRAIN.BG_MODEL_CHECKPOINT_TYPE == "caffe2",
                    image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
                )

        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
            writer,
            pseudo_labels,
            alpha,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = (
            cu.is_checkpoint_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
            or cur_epoch == cfg.SOLVER.MAX_EPOCH - 1
        )
        is_eval_epoch = (
            misc.is_eval_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
            and not cfg.MASK.ENABLE
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            alpha_scheduler_value = (
                alpha_scheduler[cur_epoch] if alpha_scheduler is not None else 0.0
            )

            eval_epoch(
                val_loader,
                model,
                val_meter,
                cur_epoch,
                cfg,
                train_loader,
                writer,
                alpha_scheduler_value,
            )
    if (
        start_epoch == cfg.SOLVER.MAX_EPOCH and not cfg.MASK.ENABLE
    ):  # final checkpoint load
        eval_epoch(val_loader, model, val_meter, start_epoch, cfg, train_loader, writer)
    if writer is not None:
        writer.close()
    result_string = (
        "_p{:.2f}_f{:.2f} _t{:.2f}_m{:.2f} _a{:.2f} Top5 Acc: {:.2f} MEM: {:.2f} f: {:.4f}"
        "".format(
            params / 1e6,
            flops,
            (
                epoch_timer.median_epoch_time() / 60.0
                if len(epoch_timer.epoch_times)
                else 0.0
            ),
            misc.gpu_mem_usage(),
            100 - val_meter.min_top1_err,
            100 - val_meter.min_top5_err,
            misc.gpu_mem_usage(),
            flops,
        )
    )
    logger.info("training done: {}".format(result_string))

    return result_string
