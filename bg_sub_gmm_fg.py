import argparse
import logging
import os
import time

import cv2
import pandas as pd

"""
Extract foreground (behaviour) from videos using Gaussian Mixture Model for background subtraction

"""


def process_video_gmm(video_path, num_frames=100, history=500, var_threshold=16):
    """
    Process video using Gaussian Mixture Model for background subtraction

    Parameters:
    video_path (str): Path to the video file
    num_frames (int): Number of frames to process
    history (int): Length of history for GMM
    var_threshold (float): Threshold for variance in GMM
    """
    # Open Video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("Error: Could not open video file")
        return None

    # Create GMM background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=True
    )

    # Initialize arrays to store frames
    frames = []
    masks = []
    fg_frames = []

    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(num_frames, total_frames)

    # Process frames
    for _ in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break

        # Apply GMM
        fgMask = backSub.apply(frame)

        # foreground frame
        fg_frame = cv2.bitwise_and(frame, frame, mask=fgMask)

        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # fg_frame_rgb = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2RGB)

        frames.append(frame_rgb)
        fg_frames.append(fg_frame)
        masks.append(fgMask)

    cap.release()

    if not frames:
        logging.error("Error: No frames to process")
        return None

    # Get background model
    background = backSub.getBackgroundImage()
    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    return background_rgb, fg_frames, fps, frame_height, frame_width, total_frames


# save background image as video file (mp4)  by duplicating the background frame with the same resolution as the input video
def write_video_file(target_path, foreground_frames, fps, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(target_path, fourcc, fps, (frame_width, frame_height))

    for f in foreground_frames:
        f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        out.write(f)

    out.release()


def main():
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir",
        type=str,
        default="dataset/videos",
        help="Path to the video files",
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="dataset/annotations/standard/fg_only",
        help="Path to the annotations directory",
    )

    parser.add_argument(
        "--target_dir",
        type=str,
        default="data/fg_videos",
        help="Path to the target directory",
    )

    args = parser.parse_args()

    start_time = time.time()
    # set up logging
    logging.basicConfig(level=logging.INFO)

    # concatenate all csv files
    video_ids = []

    for t in ["train", "val", "test"]:
        csv_file = os.path.join(args.annotations_dir, f"{t}.csv")
        df = pd.read_csv(csv_file, delimiter=",", header=None)
        video_ids.extend(df[0].tolist())

    logging.info(f"Total number of videos: {len(video_ids)}")

    # create target directory if it does not exist
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    for video_id in video_ids:
        video_path = os.path.join(args.video_dir, video_id)
        target_dir = os.path.join(args.target_dir, f"{video_id}")
        # print(f"Processing video: {target_path}")

        try:
            (
                background,
                foreground_frames,
                fps,
                frame_height,
                frame_width,
                _,
            ) = process_video_gmm(video_path)
        except:
            logging.error(f"Error processing video: {video_path}")
            continue

        if background is None:
            continue

        try:
            write_video_file(
                target_dir,
                foreground_frames,
                fps,
                frame_width,
                frame_height,
            )
        except:
            logging.error(f"Error writing video file: {target_dir}")
            continue

    end_time = time.time()

    logging.info(
        f"Processing complete. Total time taken: {end_time - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
