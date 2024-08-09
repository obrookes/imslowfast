import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("science")
plt.rcParams.update({"font.family": "Times New Roman"})


def plot_camera_locations_distribution(df, head=50, tail=10, use_proportion=False):
    # Group by 'utm' and count the number of videos for each location
    location_counts = df["utm"].value_counts().reset_index()
    location_counts.columns = ["utm", "video_count"]

    # Calculate the total number of videos
    total_videos = location_counts["video_count"].sum()

    # Sort locations by video count in descending order and calculate cumulative sum and percentage
    location_counts = location_counts.sort_values("video_count", ascending=False)
    location_counts["cumulative_count"] = location_counts["video_count"].cumsum()
    location_counts["cumulative_percentage"] = (
        location_counts["cumulative_count"] / total_videos * 100
    )

    # Identify the indices for head (50%), tail (>10), and few-shot (<10) segments
    head_index = location_counts[
        location_counts["cumulative_percentage"] <= head
    ].index[-1]
    tail_index = location_counts[location_counts["video_count"] > tail].index[-1]

    # Determine y-axis values based on use_proportion
    y_values = (
        location_counts["cumulative_percentage"]
        if use_proportion
        else location_counts["cumulative_count"]
    )
    y_label = (
        "Cumulative Proportion of Videos"
        if use_proportion
        else "Cumulative Number of Videos"
    )
    y_max = 100 if use_proportion else total_videos

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(location_counts)), y_values, "b-")

    # Add vertical lines and annotations for segments
    plt.axvline(x=head_index, color="r", linestyle="--", label="Head (50%)")
    plt.axvline(x=tail_index, color="g", linestyle="--", label="Tail (>10 samples)")

    # Fill areas for each segment
    plt.fill_between(
        range(head_index + 1),
        y_values[: head_index + 1],
        alpha=0.3,
        color="r",
        label="Head",
    )
    plt.fill_between(
        range(head_index + 1, tail_index + 1),
        y_values[head_index + 1 : tail_index + 1],
        alpha=0.3,
        color="g",
        label="Tail",
    )
    plt.fill_between(
        range(tail_index + 1, len(location_counts)),
        y_values[tail_index + 1 :],
        alpha=0.3,
        color="y",
        label="Few-shot",
    )

    # Customize the plot
    plt.title("Cumulative Distribution of Videos Across Camera Locations")
    plt.xlabel("Camera Locations (sorted by video count)")
    plt.ylabel(y_label)
    plt.legend()

    # Add text annotations
    plt.text(
        head_index,
        y_max / 2,
        f"Head: {head_index+1} locations",
        rotation=90,
        verticalalignment="center",
    )
    plt.text(
        tail_index,
        y_max / 2,
        f"Tail: {tail_index-head_index} locations",
        rotation=90,
        verticalalignment="center",
    )
    plt.text(
        len(location_counts) - 1,
        y_max / 2,
        f"Few-shot: {len(location_counts)-tail_index-1} locations",
        rotation=90,
        verticalalignment="center",
    )

    plt.ylim(0, y_max)
    plt.show()


def plot_heatmap(data, behavioural_dict, title, ax, labels, behaviors=None):
    sns.heatmap(
        data,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=False,
        cbar=False,
        square=True,
        fmt=".2f",
    )

    ax.set_xticklabels(labels, fontsize=8, rotation=45)

    if behaviors is None:
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text_color = "black"
                if data[i, j] > 0.5:
                    text_color = "white"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{data[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                )
    else:
        for i in range(len(behaviors)):
            for j in range(len(labels)):
                text_color = "black"
                if behaviors[i] in behavioural_dict["head"]:
                    text_color = "lime"
                elif behaviors[i] in behavioural_dict["tail"]:
                    text_color = "magenta"
                elif behaviors[i] in behavioural_dict["few_shot"]:
                    text_color = "olive"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                )
        ax.set_yticklabels(
            behaviors,
            fontsize=8,
            rotation=0,
            # va="center",
            # ha="right",
            # rotation_mode="anchor",
        )
        ax.legend(
            [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lime"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="magenta"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="olive"),
            ],
            [
                "Head",
                "Tail",
                "Few-shot",
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
            fontsize=8,
        )

    ax.set_xlabel("Camera locations", fontsize=8)
    ax.set_ylabel("Behaviors", fontsize=8)
    ax.set_title(title, fontsize=10)


def plot_training_progression(datas, split="train_data", dpi=300):
    # Generate doubling epochs
    epochs = [int(x) for x in datas.keys()]

    behaviors = ["head", "tail", "few shot"]
    locations = ["head_loc", "tail_loc", "few shot_loc"]

    # Generate sample data
    np.random.seed(42)
    data = datas

    # Prepare data for plotting
    plot_data = {
        f"{behavior}_{location}": [data[str(epoch)][split][i][j] for epoch in epochs]
        for i, behavior in enumerate(behaviors)
        for j, location in enumerate(locations)
    }

    # Set up the plot
    plt.figure(figsize=(12, 8), dpi=dpi)
    colors = ["red", "green", "blue"]
    line_styles = ["-", "--", ":"]

    # Plot the data
    for i, behavior in enumerate(behaviors):
        for j, location in enumerate(locations):
            key = f"{behavior}_{location}"
            plt.plot(
                epochs,
                plot_data[key],
                label=f"{behavior} ({location})",
                color=colors[i],
                linestyle=line_styles[j],
            )

    # Customize the plot
    plt.xlabel("Epoch")
    plt.ylabel("Performance")
    plt.title(f"{split.capitalize()} Behavior Performance over Epochs")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xscale("log")  # Use logarithmic scale for x-axis
    plt.xticks(
        epochs, [str(epoch) for epoch in epochs]
    )  # Set x-ticks to our specific epochs

    # Add color legend
    color_legend = plt.text(
        1.05,
        0.5,
        "Color Legend:\nRed: Head Behaviours\nGreen: Tail Behaviours\nBlue: Few Shot Behaviours",
        transform=plt.gca().transAxes,
        verticalalignment="center",
    )

    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_training_progression_with_variation(datas, split="train_data", dpi=300):
    epochs = [int(x) for x in datas.keys()]
    behaviors = ["head", "tail", "few shot"]
    locations = ["head", "tail", "few shot"]

    data = datas

    # Prepare data for plotting
    plot_data = {
        behavior: {
            location: [data[str(epoch)][split][i][j] for epoch in epochs]
            for j, location in enumerate(locations)
        }
        for i, behavior in enumerate(behaviors)
    }

    plt.figure(figsize=(12, 8), dpi=dpi)
    colors = ["red", "green", "blue"]

    for i, behavior in enumerate(behaviors):
        behavior_data = plot_data[behavior]

        # Calculate min and max for each epoch
        min_vals = [
            min(behavior_data[loc][k] for loc in locations) for k in range(len(epochs))
        ]
        max_vals = [
            max(behavior_data[loc][k] for loc in locations) for k in range(len(epochs))
        ]

        # Plot shaded area for variation
        plt.fill_between(epochs, min_vals, max_vals, alpha=0.3, color=colors[i])

        # Plot average line
        avg_vals = [(min_vals[k] + max_vals[k]) / 2 for k in range(len(epochs))]
        plt.plot(epochs, avg_vals, label=behavior, color=colors[i], linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Performance")
    plt.title(f"{split.capitalize()} Behavior Performance over Epochs (with Variation)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xscale("log")
    plt.xticks(epochs, [str(epoch) for epoch in epochs])

    # Add color legend
    color_legend = plt.text(
        1.05,
        0.5,
        "Color Legend:\nRed: Head\nGreen: Tail\nBlue: Few Shot",
        transform=plt.gca().transAxes,
        verticalalignment="center",
    )

    plt.tight_layout()
    plt.show()


def plot_map_values(datas, dpi=300):
    epochs = sorted([int(x) for x in datas.keys()])
    locations = ["train_head_loc", "train_tail_loc", "train_few_shot_loc"]
    colors = ["red", "green", "blue"]

    plt.figure(figsize=(12, 8), dpi=dpi)

    for i, location in enumerate(locations):
        values = [datas[str(epoch)]["map"][location] for epoch in epochs]
        plt.plot(epochs, values, label=location, color=colors[i], marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("MAP Value")
    plt.title("MAP Values across Epochs for Different Locations")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xscale("log")  # Use logarithmic scale for x-axis
    plt.xticks(
        epochs, [str(epoch) for epoch in epochs]
    )  # Set x-ticks to our specific epochs

    # Add color legend
    color_legend = plt.text(
        1.05,
        0.5,
        "Color Legend:\nRed: Head Location\nGreen: Tail Location\nBlue: Few Shot Location",
        transform=plt.gca().transAxes,
        verticalalignment="center",
    )

    plt.tight_layout()
    plt.show()


def plot_behavior_distribution(df, behavior_list, annot="percentage", dpi=300):
    # Ensure the number of behaviors matches the length of multihot encodings
    num_behaviors = len(df["label"].iloc[0])
    if len(behavior_list) != num_behaviors:
        raise ValueError(
            f"The length of behavior_list ({len(behavior_list)}) does not match the number of behaviors in the data ({num_behaviors})"
        )

    # Sum up the occurrences of each behavior
    behavior_counts = np.sum(df["label"].tolist(), axis=0)

    # Calculate the percentage of videos featuring each behavior
    if annot == "percentage":
        behavior_percentages = (behavior_counts / len(df)) * 100
    elif annot == "count":
        behavior_percentages = behavior_counts
    else:
        raise ValueError("annot must be either 'percentage' or 'count'")

    # Create a DataFrame for seaborn
    plot_df = pd.DataFrame(
        {"Behavior": behavior_list, "Percentage": behavior_percentages}
    )

    # Sort the DataFrame by percentage in descending order
    plot_df = plot_df.sort_values("Percentage", ascending=False)

    # Create the plot
    plt.figure(figsize=(8, 8), dpi=dpi)
    sns.set(style="whitegrid")

    # Create the bar plot
    ax = sns.barplot(x="Behavior", y="Percentage", data=plot_df)

    # Customize the plot
    plt.title("Distribution of Behaviors Across Videos", fontsize=16)
    plt.xlabel("Behaviors", fontsize=12)
    plt.ylabel("Percentage of Videos", fontsize=12)
    plt.ylim(0, max(plot_df["Percentage"]) * 1.1)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Add percentage labels on top of each bar
    for i, v in enumerate(plot_df["Percentage"]):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom")

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    plt.show()


def plot_aggregated_behavior_distribution(
    df, behavior_list, segment_list, plot_type="bar", segment="all", dpi=300
):
    # Ensure the lengths match
    if len(behavior_list) != len(segment_list) or len(behavior_list) != len(
        df["label"].iloc[0]
    ):
        raise ValueError(
            "Lengths of behavior_list, segment_list, and label encodings must match"
        )

    if plot_type not in ["bar", "pie"]:
        raise ValueError("plot_type must be either 'bar' or 'pie'")

    # Sum up the occurrences of each behavior
    behavior_counts = np.sum(df["label"].tolist(), axis=0)

    # Calculate the percentage of videos featuring each behavior
    behavior_percentages = (behavior_counts / len(df)) * 100

    # Create a DataFrame for aggregation
    plot_df = pd.DataFrame(
        {
            "Behavior": behavior_list,
            "Percentage": behavior_percentages,
            "Segment": segment_list,
        }
    )

    # Aggregate percentages by segment
    aggregated_df = plot_df.groupby("Segment")["Percentage"].sum().reset_index()

    # Calculate average percentage per behavior in each segment
    avg_df = (
        plot_df.groupby("Segment")
        .agg(Avg_Percentage=("Percentage", "mean"), Count=("Behavior", "count"))
        .reset_index()
    )

    # Merge aggregated_df with avg_df
    merged_df = pd.merge(aggregated_df, avg_df, on="Segment")

    # Ensure all segments are present and in the correct order
    all_segments = ["head", "tail", "few_shot"]
    merged_df = merged_df.set_index("Segment").reindex(all_segments).reset_index()
    merged_df = merged_df.fillna(0)  # Fill NaN values with 0 for any missing segments

    # Set seaborn style
    sns.set(style="whitegrid")

    if plot_type == "bar":
        # Create the bar plot
        plt.figure(figsize=(12, 6), dpi=dpi)
        ax = sns.barplot(
            x="Segment",
            y="Percentage",
            data=merged_df,
            palette="viridis",
            order=all_segments,
        )

        plt.title("Aggregated Distribution of Behaviors by Segment", fontsize=16)
        plt.xlabel("Segment", fontsize=12)
        plt.ylabel("Total Percentage of Videos", fontsize=12)
        plt.ylim(0, min(100, merged_df["Percentage"].max() * 1.1))

        # Add percentage labels on top of each bar
        for i, row in merged_df.iterrows():
            ax.text(
                i,
                row["Percentage"] + 0.5,
                f'{row["Percentage"]:.1f}%',
                ha="center",
                va="bottom",
            )

            # Add average percentage and count as text
            ax.text(
                i,
                row["Percentage"] / 2,
                f"Avg: {row['Avg_Percentage']:.1f}%\nCount: {row['Count']}",
                ha="center",
                va="center",
                fontweight="bold",
            )

    elif plot_type == "pie":
        # Create the pie chart
        plt.figure(figsize=(10, 8), dpi=dpi)
        colors = sns.color_palette("viridis", n_colors=len(all_segments))
        plt.pie(
            merged_df["Percentage"],
            labels=merged_df["Segment"],
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
        )
        plt.title(
            f"Proportion of Behaviors in {segment.capitalize()} Segment Camera Locations",
            fontsize=16,
        )

        legend_labels = [
            f"{segment} (Avg: {row['Avg_Percentage']:.1f}%, Count: {row['Count']})"
            for segment, row in merged_df.iterrows()
        ]
        plt.legend(
            legend_labels,
            title="Segment Details",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )

        plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_multiple_behavior_distributions(
    dfs, behavior_list, annot="proportion", titles=None, dpi=300
):
    if not isinstance(dfs, list):
        raise ValueError("dfs must be a list of dataframes")

    num_dfs = len(dfs)
    num_behaviors = len(dfs[0]["label"].iloc[0])

    if len(behavior_list) != num_behaviors:
        raise ValueError(
            f"The length of behavior_list ({len(behavior_list)}) does not match the number of behaviors in the data ({num_behaviors})"
        )

    fig, axes = plt.subplots(
        1, num_dfs, figsize=(8 * num_dfs, 8), squeeze=False, dpi=dpi
    )
    fig.suptitle("Distribution of Behaviors Across Videos", fontsize=16)

    for i, df in enumerate(dfs):
        # Sum up the occurrences of each behavior
        behavior_counts = np.sum(df["label"].tolist(), axis=0)

        # Calculate the proportion or count of videos featuring each behavior
        if annot == "proportion":
            behavior_values = behavior_counts / len(df) * 100
            ylabel = "Proportion of Videos"
            value_format = "{:.1f}%"
        elif annot == "count":
            behavior_values = behavior_counts
            ylabel = "Count of Videos"
            value_format = "{:.0f}"
        else:
            raise ValueError("annot must be either 'proportion' or 'count'")

        # Create a DataFrame for seaborn, maintaining the original order
        plot_df = pd.DataFrame({"Behavior": behavior_list, "Value": behavior_values})

        # Create the bar plot
        ax = sns.barplot(
            x="Behavior", y="Value", data=plot_df, ax=axes[0, i], order=behavior_list
        )

        # Customize the plot
        if titles and len(titles) == num_dfs:
            ax.set_title(titles[i], fontsize=14)
        ax.set_xlabel("Behaviors", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, max(plot_df["Value"]) * 1.1)

        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        # Add value labels on top of each bar
        for j, v in enumerate(plot_df["Value"]):
            ax.text(j, v, value_format.format(v), ha="center", va="bottom")

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    plt.show()
