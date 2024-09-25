import ast
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchmetrics.functional.classification import (
    multilabel_f1_score,
)


def get_feature_map(model, sample, layer="s5"):
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move model and data to device
    sample = sample.to(device)
    model = model.to(device)
    if layer == "s1":
        with torch.no_grad():
            feature_map = model.s1([sample.unsqueeze(0)])[0]
    elif layer == "s2":
        with torch.no_grad():
            feature_map = model.s2(model.s1([sample.unsqueeze(0)]))[0]
    elif layer == "s3":
        with torch.no_grad():
            feature_map = model.s3(model.s2(model.s1([sample.unsqueeze(0)])))[0]
    elif layer == "s4":
        with torch.no_grad():
            feature_map = model.s4(model.s3(model.s2(model.s1([sample.unsqueeze(0)]))))[
                0
            ]
    elif layer == "s5":
        with torch.no_grad():
            feature_map = model.s5(
                model.s4(model.s3(model.s2(model.s1([sample.unsqueeze(0)]))))
            )[
                0
            ]  # TODO: Investigate features at earlier layers
    return feature_map


def extract_framewise_features(feature_map, t):
    spatially_pooled = F.adaptive_avg_pool3d(feature_map, (t, 1, 1))
    frame_wise_features = torch.flatten(spatially_pooled, start_dim=2)
    return frame_wise_features


def get_video_level_features(feature_map):
    video_level_features = F.adaptive_avg_pool3d(feature_map, (1, 1, 1))
    video_level_features = torch.flatten(video_level_features, start_dim=1)
    return video_level_features


def extract_feats_logits(fg_sample, bg_sample, classifier, feats=False, logits=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fg_framewise_features = extract_framewise_features(fg_sample, t=16)
    bg_framewise_features = extract_framewise_features(bg_sample, t=16)

    # Squeeze and permute the dimensions
    fg_framewise_features = fg_framewise_features.squeeze(0)
    fg_framewise_features = fg_framewise_features.T

    bg_framewise_features = bg_framewise_features.squeeze(0)
    bg_framewise_features = bg_framewise_features.T

    # Compute framewise logits
    fg_frame_wise_logits = []
    for feat in fg_framewise_features.squeeze(0):
        logits = classifier(feat)
        if device == "cuda":
            logits = torch.sigmoid(logits).cpu().detach()
        fg_frame_wise_logits.append(torch.sigmoid(logits).numpy())
    fg_frame_wise_logits = np.array(fg_frame_wise_logits)

    bg_frame_wise_logits = []
    for feat in bg_framewise_features.squeeze(0):
        logits = classifier(feat)
        if device == "cuda":
            logits = torch.sigmoid(logits).cpu().detach()
        bg_frame_wise_logits.append(torch.sigmoid(logits).numpy())
    bg_frame_wise_logits = np.array(bg_frame_wise_logits)

    # Video feats
    # Foreground features and logits
    fg_video_level_features = F.adaptive_avg_pool3d(fg_sample, (1, 1, 1))
    fg_video_level_features = torch.flatten(fg_video_level_features, start_dim=1)
    fg_video_logits = classifier(fg_video_level_features)
    if device == "cuda":
        fg_video_logits = torch.sigmoid(fg_video_logits).cpu().detach()
    fg_video_level_logits = np.array(torch.sigmoid(fg_video_logits).numpy())

    # Background features and logits
    bg_video_level_features = F.adaptive_avg_pool3d(bg_sample, (1, 1, 1))
    bg_video_level_features = torch.flatten(bg_video_level_features, start_dim=1)
    bg_video_level_logits = classifier(bg_video_level_features)
    if device == "cuda":
        bg_video_level_logits = torch.sigmoid(bg_video_level_logits).cpu().detach()
    bg_video_level_logits = np.array(torch.sigmoid(bg_video_level_logits).numpy())

    if feats:
        return fg_framewise_features, bg_framewise_features
    elif logits:
        return (
            fg_frame_wise_logits,
            bg_frame_wise_logits,
            fg_video_level_logits,
            bg_video_level_logits,
        )
    else:
        return (
            fg_framewise_features,
            bg_framewise_features,
            fg_frame_wise_logits,
            bg_frame_wise_logits,
            fg_video_level_logits,
            bg_video_level_logits,
        )


def gen_temporal_mask(
    fg_framewise_features,
    bg_framewise_features,
    weight_thresh=0.5,
    mask_thresh=0.5,
    t=16,
):
    sims = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for frame_idx in range(t):
        # Randomly sample a frame/feature from the bg
        rand_bg_frame = bg_framewise_features[frame_idx]

        # Compute the cosine similarity between the random frame and all the frames in the fg
        cosine_sim = torch.cosine_similarity(
            rand_bg_frame, fg_framewise_features, dim=1
        ).unsqueeze(0)

        if device == "cuda":
            cosine_sim = cosine_sim.cpu().detach().numpy()
        else:
            cosine_sim = cosine_sim.numpy()

        # Append the similarity matrix to the list
        sims.append(cosine_sim)

    # Average the similarity matrices
    sims = np.array(sims)
    cosine_sim = np.mean(sims, axis=0)

    # Apply thresholding to cosine similarity
    cosine_sim[cosine_sim <= weight_thresh] = 0

    # Invert the cosine similarity
    cosine_sim = 1 - cosine_sim

    # Create mask
    mask = np.zeros_like(cosine_sim)
    mask[cosine_sim > mask_thresh] = 1
    mask[mask <= mask_thresh] = 0
    mask = torch.tensor(mask).bool()  # [1, 16]

    # If mask is all zero, invert it to ones
    if (mask.sum() == 0) or (mask.sum() == 1):
        cosine_sim = np.ones(cosine_sim.shape)

    return cosine_sim


# Weight features with the mask
def weight_features(framewise_features, mask):
    # [B, T, D]
    # [B, T] i.e., [1,.1, 0.75, 0.5]...
    weighted_features = []
    for frame_idx in range(len(mask)):
        # Weight the features
        weighted_feature = framewise_features[frame_idx] * mask[frame_idx]
        weighted_features.append(weighted_feature)
    weighted_features = torch.stack(weighted_features)
    return weighted_features


def weight_and_renormalize_features(framewise_features, mask):
    weighted_features = []
    original_norms = []

    for frame_idx in range(len(mask)):
        # Store the original norm
        original_norm = torch.norm(framewise_features[frame_idx])
        original_norms.append(original_norm)

        # Weight the features
        weighted_feature = framewise_features[frame_idx] * mask[frame_idx]
        weighted_features.append(weighted_feature)

    weighted_features = torch.stack(weighted_features)

    # TODO: enhance noise when x is small

    # Renormalize the weighted features
    for frame_idx in range(len(mask)):
        current_norm = torch.norm(weighted_features[frame_idx])
        if current_norm > 0:  # Avoid division by zero
            scale_factor = original_norms[frame_idx] / current_norm
            weighted_features[frame_idx] *= scale_factor

    return weighted_features


def get_weighted_features(
    fg_sample,
    bg_sample,
    classifier,
    weight_thresh=None,
    mask_thresh=None,
    weight_features_by_mask=True,
    return_mask=False,
    renormalize=False,
):

    weighted_feats = []

    # Extract frame-wise features
    fg_framewise_features, bg_framewise_features = extract_feats_logits(
        fg_sample, bg_sample, classifier, feats=True
    )  # [16, 2048], [16, 2048]

    # Generate temporal mask
    mask = gen_temporal_mask(
        fg_framewise_features,
        bg_framewise_features,
        weight_thresh=weight_thresh,
        mask_thresh=mask_thresh,
    )  # [1, 16]

    if weight_features_by_mask:
        if renormalize:
            weighted_framewise_features = weight_and_renormalize_features(
                fg_framewise_features, mask[0]
            )
        else:
            # Weight the frame-wise features with the mask
            weighted_framewise_features = weight_features(
                fg_framewise_features, mask[0]
            )  # [16, 2048]
        weighted_feats.append(weighted_framewise_features)
    else:
        mask[mask > mask_thresh] = 1
        mask[mask <= mask_thresh] = 0
        mask = torch.tensor(mask).bool()  # [1, 16]

        # If mask is all zero, invert it to ones
        if mask.sum() == 0:
            mask = ~mask

        # Sample only positive frames
        weighted_framewise_features = fg_framewise_features[
            mask.squeeze(0), :
        ]  # [N, 2048]

        # 2D pool over the weighted features
        weighted_framewise_features = F.adaptive_avg_pool2d(
            weighted_framewise_features.unsqueeze(0), (1, 2048)
        ).squeeze(
            0
        )  # [1, 2048]

        weighted_feats.append(weighted_framewise_features)

    # Stack first and then pool
    if weight_features_by_mask:
        weighted_feats = torch.stack(weighted_feats)
        weighted_feats = (
            F.adaptive_avg_pool2d(weighted_feats, (1, 2048)).permute(1, 0, 2).squeeze(0)
        )
    else:
        weighted_feats = torch.stack(weighted_feats).permute(1, 0, 2).squeeze(0)

    non_weighted_feats = (
        F.adaptive_avg_pool2d(fg_framewise_features.unsqueeze(0), (1, 2048))
        .permute(1, 0, 2)
        .squeeze(0)
    )

    return (
        (weighted_feats, non_weighted_feats, mask)
        if return_mask
        else (weighted_feats, non_weighted_feats)
    )


def calculate_masking_results(
    fg_fewshot,
    bg_fewshot,
    labels,
    classifier,
    behaviours,
    segments,
    weight_features_by_mask=True,
    return_mask=True,
    weight_thresh=None,
    mask_thresh=None,
):

    results_df = None

    # Create threshold values from 0.1 to 0.9 with 0.1 increments
    thresholds = np.arange(0.1, 1, 0.1)
    for thresh in thresholds:
        weighted_feats, non_weighted_feats, masks = [], [], []
        for i, (fg_sample, bg_sample, fg_name, bg_name) in enumerate(
            zip(
                fg_fewshot.feat_map,
                bg_fewshot.feat_map,
                fg_fewshot.name,
                bg_fewshot.name,
            )
        ):
            weighted_feat, non_weighted_feat, mask = get_weighted_features(
                fg_sample,
                bg_sample,
                classifier,
                weight_thresh=weight_thresh,
                mask_thresh=mask_thresh,
                weight_features_by_mask=weight_features_by_mask,
                return_mask=return_mask,
            )
            weighted_feats.append(weighted_feat)
            non_weighted_feats.append(non_weighted_feat)
            masks.append(mask)

        weighted_feats = torch.stack(weighted_feats)  # [N, 1, 2048]
        non_weighted_feats = torch.stack(non_weighted_feats)  # [N, 1, 2048]

        # Permute and squeeze the dimensions
        if weight_features_by_mask:
            weighted_feats = weighted_feats.permute(1, 0, 2)
        else:
            weighted_feats = weighted_feats.squeeze(1)

        # Apply the classifier
        weighted_preds = torch.sigmoid(classifier(weighted_feats))

        # Weighted performance
        weighted_f1 = multilabel_f1_score(
            weighted_preds.squeeze(0), labels, average="none", num_labels=14
        )

        result_series = pd.DataFrame(
            {
                f"weighted_f1@{round(thresh, 2)}": weighted_f1.numpy(),
            }
        )

        # Append results to the dataframe
        if results_df is None:
            results_df = result_series
        else:
            results_df = pd.concat([results_df, result_series], axis=1)

    # 3D pool over the non-weighted features
    non_weighted_feats = F.adaptive_avg_pool2d(non_weighted_feats, (1, 2048)).permute(
        1, 0, 2
    )

    original_preds = torch.sigmoid(classifier(non_weighted_feats))
    original_f1 = multilabel_f1_score(
        original_preds.squeeze(0), labels, average="none", num_labels=14
    )

    results_df["original_f1"] = original_f1.numpy()
    results_df = results_df[
        ["original_f1"] + [col for col in results_df.columns if col != "original_f1"]
    ]

    results_df["behaviour"] = behaviours
    results_df["segment"] = segments

    results_df = results_df[["behaviour", "segment"] + results_df.columns[:-2].tolist()]

    avg_df = pd.DataFrame(
        {
            "overall": results_df[results_df.columns[2:]].mean(),
            "head": results_df[results_df.segment == "head"][
                results_df.columns[2:]
            ].mean(),
            "tail": results_df[results_df.segment == "tail"][
                results_df.columns[2:]
            ].mean(),
            "fewshot": results_df[results_df.segment == "few_shot"][
                results_df.columns[2:]
            ].mean(),
        }
    ).T

    return results_df, avg_df
