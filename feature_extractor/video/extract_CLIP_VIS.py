#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
`extract_CLIP_VIS.py` is a module to extract features from video frames using CLIP model.
"""

import torch
import typing

from moviepy.editor import VideoFileClip
from utils.general_utils import frame2time
from utils.logger import return_logger
from transformers import CLIPProcessor, CLIPModel

logger = return_logger(__name__)

class CLIP(object):
    def __init__(self: typing.Type["CLIP"],
                 device: torch.device = torch.device('cpu'),
                 skip: int = 11,
                ) -> None:
        """
        Args:
            - device: to which you want to map.
            - skip: No. of frames to skip, to sample (uniform sampling) frames
        """
        # max_memory_mapping = {0: "1GB", 1: "2GB", 2:"2GB", 3:"2GB"}
        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="../cache/", max_memory=max_memory_mapping)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="../cache/")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="../cache/")
        self.clip_model.eval()
        self.clip_model.to(device)
        self.skip = skip
        self.device = device

    def get_CLIP_feats(self,
                       start: int,
                       end: int,
                       vid: typing.Union[str, VideoFileClip],
                       shot: int = -1,
                      ) -> typing.Union[typing.Tuple, dict]:
        """
        Extract features from video frame with `DenseNet169` for a given video clip.
        Also it returns 'selected' frames to be written in local storage.
        -----------------------------------------------------------------------------
        Args:
            - `start`: Starting Frame Index.
            - `end`: Ending Frame Index.
            - `vid`: The path to video from which we want to extract frames
            - `encoder`: The CNN encoder object.
            - `shot`: The shot number.
            
        Return:
            - Dictionary of Encoding of Frames with key as frame's string
        """

        # Loading video file
        if type(vid) == str:
            vid = VideoFileClip(vid)
        encode_arr = {}
        if start == end:
            image = vid.get_frame(frame2time(start, vid.fps))
            key = "shot_{0:04n}_fn_{1:06n}".format(shot, start) if\
                shot is not None else "shot_NONE_fn_{0:06n}".format(start)
            logger.info(f"Encoding Frame: {start} of shot {shot}.")
            inputs = self.clip_processor(images=image, return_tensors='pt').to(self.device)
            # extract feature
            with torch.no_grad():
                encode_arr[key] = self.clip_model.get_image_features(**inputs).detach().cpu().numpy()
        else:
            clip = vid.subclip(frame2time(start, vid.fps), frame2time(end, vid.fps))
            old_k = -(self.skip + 1)
            tmp_img_lst = []
            tmp_img_name_lst = []
            for k, image in enumerate(clip.iter_frames()):
                if k - old_k - 1 == self.skip:
                    key = "shot_{0:04n}_fn_{1:06n}".format(shot, start + k) if\
                        shot is not None else "shot_NONE_fn_{0:06n}".format(start + k)
                    logger.info(f"Encoding Frame: {start + k} of shot {shot}.")
                    tmp_img_lst.append(image)
                    tmp_img_name_lst.append(key)
                    old_k = k
            inputs = self.clip_processor(images=tmp_img_lst, return_tensors='pt').to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs).detach().cpu().numpy()
            encode_arr = {key: image_features[i] for i, key in enumerate(tmp_img_name_lst)}
        return encode_arr

# - black_thresh: Decision value for blackiness.
# - white_thresh: Decision value for whiteness.
#   Threshold to reject uniformly colored black/white frames.
