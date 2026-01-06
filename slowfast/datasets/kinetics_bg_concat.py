#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import ast
import os
import random

import numpy as np
import pandas
import torch
import torch.utils.data
from torchvision import transforms
import torch.nn.functional as F
import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import decoder as decoder
from . import transform as transform
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import (
    MaskingGenerator,
    MaskingGenerator3D,
    create_random_augment,
)

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Bkinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1, path_to_bg_video_1, label_1
        path_to_video_2, path_to_bg_video_2, label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB
        self._video_meta = {}
        self._num_retries = num_retries
        self._num_epoch = 0.0
        self._num_yielded = 0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS
        self.use_chunk_loading = (
            True
            if self.mode in ["train"] and self.cfg.DATA.LOADER_CHUNK_SIZE > 0
            else False
        )
        self.dummy_output = None
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0
        self.cur_epoch = 0

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert pathmgr.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_fg_videos = []
        self._path_to_bg_videos = []
        self._is_video_negative = []
        self._utms = []
        self._labels = []
        self._spatial_temporal_idx = []
        self.cur_iter = 0
        self.chunk_epoch = 0
        self.epoch = 0.0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS

        with pathmgr.open(path_to_file, "r") as f:
            if self.use_chunk_loading:
                rows = self._get_chunk(f, self.cfg.DATA.LOADER_CHUNK_SIZE)
            else:
                rows = f.read().splitlines()
            for clip_idx, path_label in enumerate(rows):
                fetch_info = path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)
                if len(fetch_info) > 3:
                    fg_path, bg_path, label, is_negative, utm = (
                        self._separate_list_components(fetch_info)
                    )
                else:
                    raise RuntimeError(
                        "Failed to parse video fetch {} info {} retries.".format(
                            path_to_file, fetch_info
                        )
                    )
                for idx in range(self._num_clips):
                    self._path_to_fg_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, fg_path)
                    )
                    self._path_to_bg_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, bg_path)
                    )
                    self._is_video_negative.append(ast.literal_eval(is_negative))
                    self._utms.append(ast.literal_eval(utm))
                    try:
                        self._labels.append(int(label))
                    except:
                        self._labels.append(ast.literal_eval(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_fg_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        assert (len(self._path_to_bg_videos)) > 0, "Failed to load bg videos"
        # Assert len of bg videos is same as fg videos
        assert len(self._path_to_fg_videos) == len(self._path_to_bg_videos)
        logger.info(
            "Constructing kinetics dataloader (size: {} skip_rows {}) from {} ".format(
                len(self._path_to_fg_videos), self.skip_rows, path_to_file
            )
        )

    def _set_epoch_num(self, epoch):
        self.epoch = epoch

    def _get_chunk(self, path_to_file, chunksize):
        try:
            for chunk in pandas.read_csv(
                path_to_file,
                chunksize=self.cfg.DATA.LOADER_CHUNK_SIZE,
                skiprows=self.skip_rows,
            ):
                break
        except Exception:
            self.skip_rows = 0
            return self._get_chunk(path_to_file, chunksize)
        else:
            return pandas.array(chunk.values.flatten(), dtype="string")

    def sample_consecutive_frames(self, video_tensor, num_frames, random_start=True):
        """
        Sample N consecutive frames from a video tensor.

        Args:
            video_tensor (torch.Tensor): Video tensor of shape [C, T, H, W]
            num_frames (int): Number of consecutive frames to sample
            random_start (bool): If True, randomly select starting frame. If False, start from frame 0

        Returns:
            torch.Tensor: Sampled frames of shape [C, N, H, W]
        """
        channels, total_frames, height, width = video_tensor.shape

        # Ensure we don't sample more frames than available
        if num_frames > total_frames:
            raise ValueError(
                f"Requested {num_frames} frames but video only has {total_frames} frames"
            )

        # Calculate the maximum possible starting index
        max_start_idx = total_frames - num_frames

        if max_start_idx < 0:
            raise ValueError(
                "Not enough frames in video to sample the requested number of consecutive frames"
            )

        # Get starting index
        if random_start:
            start_idx = torch.randint(0, max_start_idx + 1, (1,)).item()
        else:
            start_idx = 0

        # Create sequential frame indices
        frame_indices = torch.arange(start_idx, start_idx + num_frames)

        # Sample the frames
        sampled_frames = video_tensor[:, frame_indices]

        return sampled_frames

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, self._num_yielded = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index
        if self.dummy_output is not None:
            return self.dummy_output
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S)
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS)
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))
        num_decode = (
            self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL if self.mode in ["train"] else 1
        )
        min_scale, max_scale, crop_size = [min_scale], [max_scale], [crop_size]
        if len(min_scale) < num_decode:
            min_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * (
                num_decode - len(min_scale)
            )
            max_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[1]] * (
                num_decode - len(max_scale)
            )
            crop_size += (
                [self.cfg.MULTIGRID.DEFAULT_S] * (num_decode - len(crop_size))
                if self.cfg.MULTIGRID.LONG_CYCLE or self.cfg.MULTIGRID.SHORT_CYCLE
                else [self.cfg.DATA.TRAIN_CROP_SIZE] * (num_decode - len(crop_size))
            )
            assert self.mode in ["train", "val"]
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_fg_container = None
            video_bg_container = None
            try:
                video_fg_container = container.get_video_container(
                    self._path_to_fg_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video foreground from {} with error {}".format(
                        self._path_to_fg_videos[index], e
                    )
                )

            try:
                video_bg_container = container.get_video_container(
                    self._path_to_bg_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load background video from {} with error {}".format(
                        self._path_to_bg_videos[index], e
                    )
                )

                if self.mode not in ["test"]:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_fg_videos) - 1)
                continue  # Select a random video if the current video was not able to access.

            if (video_fg_container is None) or (video_bg_container is None):
                logger.warning(
                    "Failed to meta load video idx {} from {} or {} or {}; trial {}".format(
                        index,
                        self._path_to_fg_videos[index],
                        self._path_to_bg_videos[index],
                        i_try,
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 8:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_fg_videos) - 1)
                continue

            frames_decoded, time_idx_decoded = (
                [None] * num_decode,
                [None] * num_decode,
            )

            # for i in range(num_decode):
            num_frames = [self.cfg.DATA.NUM_FRAMES]
            sampling_rate = utils.get_random_sampling_rate(
                self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
                self.cfg.DATA.SAMPLING_RATE,
            )
            sampling_rate = [sampling_rate]
            if len(num_frames) < num_decode:
                num_frames.extend(
                    [num_frames[-1] for i in range(num_decode - len(num_frames))]
                )
                # base case where keys have same frame-rate as query
                sampling_rate.extend(
                    [sampling_rate[-1] for i in range(num_decode - len(sampling_rate))]
                )
            elif len(num_frames) > num_decode:
                num_frames = num_frames[:num_decode]
                sampling_rate = sampling_rate[:num_decode]

            if self.mode in ["train"]:
                assert len(min_scale) == len(max_scale) == len(crop_size) == num_decode

            target_fps = self.cfg.DATA.TARGET_FPS
            if self.cfg.DATA.TRAIN_JITTER_FPS > 0.0 and self.mode in ["train"]:
                target_fps += random.uniform(0.0, self.cfg.DATA.TRAIN_JITTER_FPS)

            # Decode video. Meta info is used to perform selective decoding.
            fg_frames, time_idx, tdiff = decoder.decode(
                video_fg_container,
                sampling_rate,
                num_frames,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=(
                    self._video_meta[index] if len(self._video_meta) < 5e6 else {}
                ),  # do not cache on huge datasets
                target_fps=target_fps,
                backend=self.cfg.DATA.DECODING_BACKEND,
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                max_spatial_scale=(
                    min_scale[0] if all(x == min_scale[0] for x in min_scale) else 0
                ),  # if self.mode in ["test"] else 0,
                time_diff_prob=self.p_convert_dt if self.mode in ["train"] else 0.0,
                temporally_rnd_clips=True,
                min_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,
                max_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,
            )

            bg_frames, _, _ = decoder.decode(
                video_bg_container,
                sampling_rate,
                num_frames,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=(
                    self._video_meta[index] if len(self._video_meta) < 5e6 else {}
                ),  # do not cache on huge datasets
                target_fps=target_fps,
                backend=self.cfg.DATA.DECODING_BACKEND,
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                max_spatial_scale=(
                    min_scale[0] if all(x == min_scale[0] for x in min_scale) else 0
                ),  # if self.mode in ["test"] else 0,
                time_diff_prob=self.p_convert_dt if self.mode in ["train"] else 0.0,
                temporally_rnd_clips=True,
                min_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,
                max_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,
            )

            fg_frames_decoded = fg_frames
            bg_frames_decoded = bg_frames
            time_idx_decoded = time_idx

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if (
                (fg_frames_decoded is None)
                or (bg_frames_decoded is None)
                or (None in fg_frames_decoded)
                or (None in bg_frames_decoded)
            ):
                logger.warning(
                    "Failed to decode video idx {} from {} or {} or {}; trial {}".format(
                        index,
                        self._path_to_fg_videos[index],
                        self._path_to_bg_videos[index],
                        i_try,
                    )
                )
                if (
                    self.mode not in ["test"]
                    and (i_try % (self._num_retries // 8)) == 0
                ):
                    # let's try another one
                    index = random.randint(0, len(self._path_to_fg_videos) - 1)
                continue

            num_aug = (
                self.cfg.DATA.TRAIN_CROP_NUM_SPATIAL * self.cfg.AUG.NUM_SAMPLE
                if self.mode in ["train"]
                else 1
            )
            num_out = num_aug * num_decode
            fg_out, time_idx_out = [None] * num_out, [None] * num_out
            bg_out, bg2_out = [None] * num_out, [None] * num_out

            idx = -1
            # Handle label
            label = self._labels[index]
            # Handle negative indicator
            negative = self._is_video_negative[index]
            # Handle UTM
            utm = self._utms[index]
            label = ast.literal_eval(label)
            if isinstance(label, list):
                label = torch.tensor(label, dtype=torch.float64)
            for i in range(num_decode):
                for _ in range(num_aug):
                    idx += 1
                    fg_out[idx] = fg_frames_decoded[i].clone()
                    bg_out[idx] = bg_frames_decoded[i].clone()

                    time_idx_out[idx] = time_idx_decoded[i, :]

                    # [0,1] normalization
                    fg_out[idx] = fg_out[idx].float()
                    fg_out[idx] = fg_out[idx] / 255.0
                    bg_out[idx] = bg_out[idx].float()
                    bg_out[idx] = bg_out[idx] / 255.0

                    if self.mode in ["train"] and self.cfg.DATA.SSL_COLOR_JITTER:
                        fg_out[idx] = transform.color_jitter_video_ssl(
                            fg_out[idx],
                            bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,
                            hue=self.cfg.DATA.SSL_COLOR_HUE,
                            p_convert_gray=self.p_convert_gray,
                            moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,
                            gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                            gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                        )
                        bg_out[idx] = transform.color_jitter_video_ssl(
                            bg_out[idx],
                            bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,
                            hue=self.cfg.DATA.SSL_COLOR_HUE,
                            p_convert_gray=self.p_convert_gray,
                            moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,
                            gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                            gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                        )

                    if self.aug and self.cfg.AUG.AA_TYPE:
                        aug_transform = create_random_augment(
                            input_size=(fg_out[idx].size(1), fg_out[idx].size(2)),
                            auto_augment=self.cfg.AUG.AA_TYPE,
                            interpolation=self.cfg.AUG.INTERPOLATION,
                        )
                        # T H W C -> T C H W.
                        fg_out[idx] = fg_out[idx].permute(0, 3, 1, 2)
                        bg_out[idx] = bg_out[idx].permute(0, 3, 1, 2)
                        bg2_out[idx] = bg2_out[idx].permute(0, 3, 1, 2)

                        fg_list_img = self._frame_to_list_img(fg_out[idx])
                        fg_list_img = aug_transform(fg_list_img)

                        bg_list_img = self._frame_to_list_img(bg_out[idx])
                        bg_list_img = aug_transform(bg_list_img)

                        fg_out[idx] = self._list_img_to_frames(fg_list_img)
                        bg_out[idx] = self._list_img_to_frames(bg_list_img)

                        fg_out[idx] = fg_out[idx].permute(0, 2, 3, 1)
                        bg_out[idx] = bg_out[idx].permute(0, 2, 3, 1)

                    # Perform color normalization.
                    if self.cfg.DATA.USE_MEAN:
                        fg_out[idx] = utils.tensor_normalize(
                            fg_out[idx],
                            self.cfg.DATA.MEAN,
                            self.cfg.DATA.STD,
                        )

                        bg_out[idx] = utils.tensor_normalize(
                            bg_out[idx],
                            self.cfg.DATA.MEAN,
                            self.cfg.DATA.STD,
                        )

                    # T H W C -> C T H W.
                    fg_out[idx] = fg_out[idx].permute(3, 0, 1, 2)
                    bg_out[idx] = bg_out[idx].permute(3, 0, 1, 2)

                    scl, asp = (
                        self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
                        self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
                    )
                    relative_scales = (
                        None if (self.mode not in ["train"] or len(scl) == 0) else scl
                    )
                    relative_aspect = (
                        None if (self.mode not in ["train"] or len(asp) == 0) else asp
                    )

                    if self.cfg.DATA.SPATIAL_SAMPLING:
                        fg_out[idx] = utils.spatial_sampling(
                            fg_out[idx],
                            spatial_idx=spatial_sample_index,
                            min_scale=min_scale[i],
                            max_scale=max_scale[i],
                            crop_size=crop_size[i],
                            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                            aspect_ratio=relative_aspect,
                            scale=relative_scales,
                            motion_shift=(
                                self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
                                if self.mode in ["train"]
                                else False
                            ),
                        )

                        bg_out[idx] = utils.spatial_sampling(
                            bg_out[idx],
                            spatial_idx=spatial_sample_index,
                            min_scale=min_scale[i],
                            max_scale=max_scale[i],
                            crop_size=crop_size[i],
                            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                            aspect_ratio=relative_aspect,
                            scale=relative_scales,
                            motion_shift=(
                                self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
                                if self.mode in ["train"]
                                else False
                            ),
                        )
                    else:
                        fg_out[idx] = F.interpolate(
                            fg_out[idx],
                            size=(
                                self.cfg.DATA.DECODING_SHORT_SIZE,
                                self.cfg.DATA.DECODING_SHORT_SIZE,
                            ),
                            mode="nearest",
                        )

                        bg_out[idx] = F.interpolate(
                            bg_out[idx],
                            size=(
                                self.cfg.DATA.DECODING_SHORT_SIZE,
                                self.cfg.DATA.DECODING_SHORT_SIZE,
                            ),
                            mode="nearest",
                        )

                    if self.rand_erase:
                        erase_transform = RandomErasing(
                            self.cfg.AUG.RE_PROB,
                            mode=self.cfg.AUG.RE_MODE,
                            max_count=self.cfg.AUG.RE_COUNT,
                            num_splits=self.cfg.AUG.RE_COUNT,
                            device="cpu",
                        )
                        fg_out[idx] = erase_transform(
                            fg_out[idx].permute(1, 0, 2, 3)
                        ).permute(1, 0, 2, 3)

                        # Not neccessary to apply random erasing on backgrounds...

                    fg_out[idx] = utils.pack_pathway_output(self.cfg, fg_out[idx])
                    bg_out[idx] = utils.pack_pathway_output(self.cfg, bg_out[idx])

                    if self.cfg.AUG.GEN_MASK_LOADER:
                        mask = self._gen_mask()
                        fg_out[idx] = fg_out[idx] + [torch.Tensor(), mask]
                        # Not neccessary to apply mask on backgrounds...

            fg_frames = fg_out[0] if num_out == 1 else fg_out
            bg_frames = bg_out[0] if num_out == 1 else bg_out
            time_idx = np.array(time_idx_out)

            if (
                num_aug * num_decode > 1
                and not self.cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            ):
                label = [label] * num_aug * num_decode
                index = [index] * num_aug * num_decode

            # select random bg frames
            bg_frames = bg_frames[0]
            fg_frames = fg_frames[0]

            if self.mode in ["train"]:
                if (
                    self.cfg.FG_BG_MIXUP.CONCAT_BG_FRAMES
                    and self.cfg.FG_BG_MIXUP.CONCAT_BG_FRAMES.RATIO > 0
                ):
                    # take a random subset of bg_frames using concat_bg_frames_ratio
                    num_bg_frames = int(
                        bg_frames.shape[1] * self.cfg.FG_BG_MIXUP.CONCAT_BG_FRAMES.RATIO
                    )

                    frames_quotient = num_bg_frames // bg_frames.shape[1]
                    frames_remainder = num_bg_frames % bg_frames.shape[1]

                    if frames_quotient > 0:
                        indices = torch.cat(
                            [
                                torch.arange(bg_frames.shape[1])
                                for _ in range(frames_quotient)
                            ]
                        )
                    else:
                        indices = torch.tensor([], dtype=torch.int32)

                    if frames_remainder > 0:
                        remainder_indices = torch.randperm(bg_frames.shape[1])[
                            :frames_remainder
                        ]

                        if self.cfg.FG_BG_MIXUP.CONCAT_BG_FRAMES.SORT_BG_FRAMES:
                            indices = torch.cat(
                                [
                                    indices,
                                    remainder_indices,
                                ]
                            )
                            # sort all indices
                            indices = torch.sort(indices).values

                        else:
                            # sort only the remainder_indices
                            remainder_indices = torch.sort(remainder_indices).values

                            indices = torch.cat(
                                [
                                    indices,
                                    remainder_indices,
                                ]
                            )

                    selected_bg_frames = bg_frames[:, indices, :, :]

                    # concatenate bg_frames to fg_frames
                    concat_frames = torch.cat(
                        [
                            fg_frames,
                            selected_bg_frames,
                        ],
                        dim=1,
                    )

                    assert concat_frames.shape[1] == fg_frames.shape[1] + num_bg_frames

                    # If subsample concatenated frames
                    if self.cfg.FG_BG_MIXUP.CONCAT_BG_FRAMES.SUBSAMPLE_CONCAT_FRAMES:
                        concat_frames = self.sample_consecutive_frames(
                            concat_frames, self.cfg.DATA.NUM_FRAMES
                        )

                    fg_frames = concat_frames

            if self.cfg.DATA.DUMMY_LOAD:
                if self.dummy_output is None:
                    self.dummy_output = (
                        fg_frames,
                        bg_frames,
                        label,
                        negative,
                        utm,
                        index,
                        time_idx,
                        {},
                        {},
                    )

            inputs = {
                "concat_frames": [fg_frames],
                "bg_frames": [bg_frames],
                "mask": negative,
                "utm": utm,
            }
            meta = {
                "fg_video_name": self._path_to_fg_videos[index].split("/")[-1],
                "bg_video_name": self._path_to_bg_videos[index].split("/")[-1],
            }
            return (
                inputs,
                label,
                index,
                time_idx,
                meta,
            )
        else:
            logger.warning(
                "Failed to fetch video after {} retries.".format(self._num_retries)
            )

    def _separate_list_components(self, input_list):
        # Join all elements of the list into a single string
        joined_string = "".join(input_list)

        # Split the string by comma, but not within square brackets
        parts = []
        bracket_count = 0
        current_part = ""

        for char in joined_string:
            if char == "," and bracket_count == 0:
                parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1

        # Append the last part
        if current_part:
            parts.append(current_part.strip())

        # Separate the components
        video1 = parts[0]
        video2 = parts[1]
        array = parts[2]
        boolean = parts[3]
        utm = parts[4]

        return video1, video2, array, boolean, utm

    def _gen_mask(self):
        if self.cfg.AUG.MASK_TUBE:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            min_mask = num_masking_patches // 5
            masked_position_generator = MaskingGenerator(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=None,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
            mask = np.tile(mask, (8, 1, 1))
        elif self.cfg.AUG.MASK_FRAMES:
            mask = np.zeros(shape=self.cfg.AUG.MASK_WINDOW_SIZE, dtype=int)
            n_mask = round(self.cfg.AUG.MASK_WINDOW_SIZE[0] * self.cfg.AUG.MASK_RATIO)
            mask_t_ind = random.sample(
                range(0, self.cfg.AUG.MASK_WINDOW_SIZE[0]), n_mask
            )
            mask[mask_t_ind, :, :] += 1
        else:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            max_mask = np.prod(self.cfg.AUG.MASK_WINDOW_SIZE[1:])
            min_mask = max_mask // 5
            masked_position_generator = MaskingGenerator3D(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=max_mask,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
        return mask

    def _frame_to_list_img(self, frames):
        img_list = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        # Assert that the number of fg videos is the same as the number of bg videos
        assert len(self._path_to_fg_videos) == len(self._path_to_bg_videos)
        return len(self._path_to_fg_videos)
