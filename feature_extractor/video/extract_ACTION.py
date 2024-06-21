#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
`extract_MVIT.py`: Module to extract MVIT features from video.
"""

import numpy as np
import torch
import torch.fx
import typing

from moviepy.editor import VideoFileClip
from utils.general_utils import frame2time
from utils.logger import return_logger
from typing import Optional, Type
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import ShortSideScale

logger = return_logger(__name__)

class ACTION:
    def __init__(self: Type["ACTION"],
                 device: torch.device = torch.device('cpu'),
                 stride:Optional[int]=16,
                ) -> None:

        self.chunk_size = 32
        self.stride = stride or self.chunk_size//2
        self.device = device
        self.model_names = {"mvit": "mvit_base_32x3",
                            "x3d": "x3d_l",
                            "i3d": "i3d_r50"}
        
        # define transform function
        side_size = 256
        crop_size = 224
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        self.transform = Compose(
                [
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size)
                ]
            )

        # define model
        self.model = torch.hub.load("facebookresearch/pytorchvideo", model=self.model_names["mvit"], pretrained=True)
        self.model.head = self.model.head.sequence_pool

        # self.model = nn.DataParallel(, device_ids=[0,1,2,3])
        # bring it on device (cuda) and freeze it
        self.model.to(self.device)
        self.model.eval()

    def vid2ACTION(self,
                   start_end: typing.Optional[typing.Tuple[int, int]],
                   shot_no:Optional[int],
                   vid_path: typing.Union[str, VideoFileClip]
                  )->np.ndarray:
        """
        Convert the video read from given path to its corresponding I3D
        features. Note that here one embedding vector of size `1024` will be
        for `chunk_size` number of frames. So if `self.frequency < self.chunk_size`
        the adjacent chunks will overlap.
        ----------------------------------------------------------------------------
        Args:
            - `start_end`: Tuple consisting of starting and ending `frame index`.
            - `shot_no`: Shot number of the video.
            - `vid_path`: Path to video-file or `VideoFileClip` object itself.

        Return:
            - `np.ndarray` feature of size `(n, 1024)`.
        """
        vid = VideoFileClip(vid_path) if type(vid_path) == str else vid_path

        if start_end is None:
            clip = vid
            frame_cnt = round(clip.fps * clip.duration)
        else:
            clip = vid.subclip(frame2time(start_end[0], vid.fps), frame2time(start_end[1], vid.fps))
            frame_cnt = start_end[1] - start_end[0] + 1 # as inclusive
        
        # Cut frames
        assert(frame_cnt >= self.chunk_size)
        logger.info(f"Started processing for video at {vid_path}")
        
        # for window size of self.stride, we need to pad zeros
        # frames at end to account for total coverage
        big_frame_arr = []
        for frame_cnt, frame in enumerate(clip.iter_frames()):
            big_frame_arr.append(frame)
        num_of_zero_arr_toadd = self.stride-(frame_cnt+1)%self.stride
        img_shape = big_frame_arr[0].shape
        big_frame_arr = np.concatenate((np.stack(big_frame_arr), 
                        np.zeros((num_of_zero_arr_toadd, img_shape[0], img_shape[1], 3))), axis=0)
        
        num_of_chunks = (big_frame_arr.shape[0]//16) - 1

        encoding_dict = {}
        with torch.no_grad():
            for idx in range(num_of_chunks):
                arr = big_frame_arr[idx*self.stride : idx*self.stride + self.chunk_size]
                video_data = torch.from_numpy(arr.transpose([3, 0, 1, 2])).to(torch.float32)
                # Apply a transform to normalize the video input
                inputs = self.transform(video_data)
                # Move the inputs to the desired device
                inputs = inputs.to(self.device)[None, ...]
                preds = self.model(inputs).reshape((-1,))
                encoding_dict["shot_{0:04n}_fn_{1:06n}".format(shot_no, start_end[0]+idx*self.stride)] = preds.detach().cpu().numpy()
        
        return encoding_dict
