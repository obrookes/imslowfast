import ast
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchmetrics.functional.classification import (
    multilabel_average_precision,
)

plt.style.use("science")


def read_files(model_results, epoch):
    with open(model_results[epoch]["train"]["file_path"], "rb") as f:
        train_data = pkl.load(f)

    with open(model_results[epoch]["val"]["file_path"], "rb") as f:
        val_data = pkl.load(f)

    return train_data, val_data


def results2df(train_data, val_data, metadata_df):
    # Process subclips
    subclips = []
    for i, split in enumerate([train_data, val_data]):
        for name, pred, feat, label in zip(
            split["names"], split["preds"], split["feats"], split["labels"]
        ):
            subclips.append(
                {
                    "name": name,
                    "split": i,
                    "pred": pred,
                    "feat": feat,
                    "negative": True if sum(label) == 0 else False,
                }
            )

    df = pd.DataFrame(subclips, columns=["name", "split", "pred", "feat", "negative"])

    df["split"] = df.split.map({0: "train", 1: "val"})
    df = df.merge(metadata_df, how="left", left_on="name", right_on="subject_id")

    # Apply sigmoid to predictions
    df["pred"] = df.pred.apply(lambda x: torch.sigmoid(torch.tensor(x)))

    # Convert label from str to int
    df.label = df.label.apply(lambda x: np.array(ast.literal_eval(x)))

    # Add negative
    df["negative"] = df.label.apply(lambda x: sum(x) == 0)

    # Add global location count to dataframe
    df["location_count"] = df.utm.map(df.utm.value_counts())

    # filter out negative samples from the dataframe
    # df = df[~df.negative]

    # Return train and val dataframes
    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]

    return train_df, val_df


def return_ct_location_segments(df, head=50, tail=10):
    """
    Returns the location segments based on the given dataframe and thresholds.

    Args:
        df (pandas.DataFrame): The input dataframe containing the location data.
        head (int, optional): The threshold percentage for selecting locations that make up the top percentage of data. Defaults to 50.
        tail (int, optional): The threshold count for selecting locations outside the top percentage with more than this count. Defaults to 10.

    Returns:
        tuple: A tuple containing three dataframes:
            - head_locations: Dataframe containing the top locations and their video counts.
            - tail_locations: Dataframe containing the locations outside the top percentage with more than the tail count.
            - few_shot_locations: Dataframe containing the locations with fewer than the tail count.
    """
    # Group by 'utm' and count the number of videos for each location
    location_counts = df["utm"].value_counts().reset_index()
    location_counts.columns = ["utm", "video_count"]

    # Calculate the total number of videos
    total_videos = location_counts["video_count"].sum()

    # Sort locations by video count in descending order and calculate cumulative percentage
    location_counts = location_counts.sort_values("video_count", ascending=False)
    location_counts["cumulative_count"] = location_counts["video_count"].cumsum()
    location_counts["cumulative_percentage"] = (
        location_counts["cumulative_count"] / total_videos * 100
    )

    # Select locations that make up 50% of the data
    head_locations = location_counts[location_counts["cumulative_percentage"] <= head]

    # Calculate locations outside the top 50% with more than 10 samples
    tail_locations = location_counts[location_counts["cumulative_percentage"] > head]
    tail_locations = tail_locations[tail_locations["video_count"] > tail]

    # Calculate locations with fewer than 10 samples
    few_shot_locations = location_counts[location_counts["video_count"] < tail]

    return (
        head_locations[["utm", "video_count"]],
        tail_locations[["utm", "video_count"]],
        few_shot_locations[["utm", "video_count"]],
    )


def print_per_segment_performance_old(
    map, behaviour_list, segment_list, segment, show_per_class=True
):
    res = []
    for i, (b, s) in enumerate(zip(map, segment_list)):
        if s == segment:
            res.append({behaviour_list[i]: b})
    agg_values = []
    for r in res:
        for _, value in r.items():
            agg_values.append(value)

    if show_per_class:
        return {
            segment: {
                "mean": np.round(np.mean(agg_values), 3),
                "values": res,
            }
        }
    else:
        return {
            segment: {
                "mean": np.round(np.mean(agg_values), 3),
            }
        }


def print_per_segment_performance(
    map, behaviour_list, camera_list, segment, show_per_class=True
):
    res = []
    for i, (b, s) in enumerate(zip(map, camera_list)):
        if s == segment:
            res.append({behaviour_list[i]: b})
    agg_values = []
    for r in res:
        for _, value in r.items():
            agg_values.append(value)

    if show_per_class:
        return {
            segment: {
                "mean": np.round(np.mean(agg_values), 3),
                "values": res,
            }
        }
    else:
        return {
            segment: {
                "mean": np.round(np.mean(agg_values), 3),
            }
        }


def calculate_metrics(
    df, behaviour_list, segment_list, round_to=3, show_per_class=False
):
    map = multilabel_average_precision(
        torch.tensor(np.stack(df["pred"])),
        torch.tensor(np.stack(df["label"])),
        num_labels=14,
        average="none",
    )

    # replace -0.0 with 0.0
    # map[map == -0.0] = 0.0

    map_head = print_per_segment_performance(
        map, behaviour_list, segment_list, "head", show_per_class
    )
    map_tail = print_per_segment_performance(
        map, behaviour_list, segment_list, "tail", show_per_class
    )
    map_fs = print_per_segment_performance(
        map, behaviour_list, segment_list, "few_shot", show_per_class
    )

    if show_per_class:
        map_head_values = map_head["head"]["values"]
        map_tail_values = map_tail["tail"]["values"]
        map_fs_values = map_fs["few_shot"]["values"]

    map_head = round(float(map_head["head"]["mean"]), round_to)
    map_tail = round(float(map_tail["tail"]["mean"]), round_to)
    map_fs = round(float(map_fs["few_shot"]["mean"]), round_to)

    avg_map = round(map.mean().item(), round_to)

    if show_per_class:
        for i in range(len(map_head_values)):
            for key, value in map_head_values[i].items():
                map_head_values[i][key] = round(value.item(), round_to)
        for i in range(len(map_tail_values)):
            for key, value in map_tail_values[i].items():
                map_tail_values[i][key] = round(value.item(), round_to)

        for i in range(len(map_fs_values)):
            for key, value in map_fs_values[i].items():
                map_fs_values[i][key] = round(value.item(), round_to)

        return (
            avg_map,
            map_head,
            map_head_values,
            map_tail,
            map_tail_values,
            map_fs,
            map_fs_values,
        )

    return avg_map, map_head, map_tail, map_fs


def calculate_all_metrics(segments, behaviour_list, segment_list, show_per_class=False):
    metrics = {}
    for key in ["h", "t", "f"]:
        (
            metrics[f"avg_map_{key}"],
            metrics[f"map_{key}_head"],
            metrics[f"map_{key}_head_values"],
            metrics[f"map_{key}_tail"],
            metrics[f"map_{key}_tail_values"],
            metrics[f"map_{key}_fs"],
            metrics[f"map_{key}_fs_values"],
        ) = calculate_metrics(
            df=segments[f"{key}"],
            behaviour_list=behaviour_list,
            segment_list=segment_list,
            show_per_class=show_per_class,
        )
    return metrics
