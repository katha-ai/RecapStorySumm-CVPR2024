#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
vid_utils.py: consists of helper functions.
"""

import os
import cv2

from pathlib import Path
from datetime import time
from utils.logger import return_logger
from moviepy.editor import VideoFileClip
from feature_extractor.video.extract_IMAGENET import IMAGE
from feature_extractor.video.extract_ACTION import ACTION
from feature_extractor.video.extract_CLIP_VIS import CLIP

from typing import Dict, List, Tuple
from utils.general_utils import (readVidEvents, save_pickle, load_pickle, sec2datetime)

logger = return_logger(__name__)

class shot:
    def __init__(self,
                 shot_no: int,
                 episode_name: str = "",
                 frame_counts: int = 0,
                 frame_names: List[str] = list(),
                 time: Tuple[time, time] = (time(), time()),
                 matching_shots: List = []
                ) -> None:
        self.shot_no = shot_no
        self.episode_name = episode_name
        self.frame_counts = frame_counts
        self.frame_names = frame_names
        self.time = time
        self.matching_shots = matching_shots

# ========================================= Saving Encoding =========================================

def frame_saver(shot_no:int,
                frame_lst:List,
                fn_lst:List,
                vid_type:str,
                directory:Path
               )->None:
    """
    save the frames extracted using "fromVideo" mode
    """
    dir_path = directory/f"{vid_type}_html_images/"
    os.makedirs(dir_path, exist_ok=True)
    for idx, img in zip(fn_lst, frame_lst):
        if idx >= 0:
            img_name = "shot_{0:04n}".format(shot_no) + "_fn_{0:06n}.jpg".format(abs(idx))
        else:
            img_name = "shot_{0:04n}".format(shot_no) + "_fn_{0:06n}_bw.jpg".format(abs(idx))
        cv2.imwrite((dir_path/img_name).as_posix(), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def extractEncodedRecapOrEpisode(eps_name: str,
                                 config: Dict,
                                 imagenet_extractor: IMAGE,
                                 mvit_extractor: ACTION,
                                 clip_extractor: CLIP,
                                 vid_type:str="recap",
                                 save_frames:bool=False
                                ) -> Tuple[list, dict]:
    """
    Encode the RECAP or EPISODE part of a given episode (as asked).
    Here we intend to extract three varities of features: `Motion (Action)`
    using `MVIT`, `CLIP` image features, and `Dense` features using `DenseNet169`.
    ----------------------------------------------------------------
    Args:
        - `eps_name`: Episode Name (in format: Exx, e.g., E03).
        - `config`: Dictionary that consists required paths.
        - `imagenet_extractor`: Imagenet architecture namely, `DenseNet169`
          that extracts video frames features.
        - `mvit_extractor`: Action embeddings from MVIT model.
        - `clip_extractor`: CLIP Image model to extract embeddings that semantically
          similar to its textual description.
        - `vid_type`: What you want to retrieve --> "recap" or "episode".
        - `save_frames`: Whether to save the frames for each `video_shot`. Note
          that `imagenet` needs to be `True`.

    Return:
        - A dictionary containing shot's `time_stamp` or `fn` with their
          corresponding frame embeddings.
        - A wrapper dictionary that consists of above-mentioned dictionaries
          with their types.
    """

    def vidSavingModule(tmp_lst:List,
                        vid:VideoFileClip,
                        shot0:int,
                        vid_type:str,
                        frame_path:Path
                       ) -> Tuple[dict, dict, dict]:
        """
        Save the video shots in form of frame encodings
        from different pre-trained backbones.

        Args:
            - tmp_lst: List of frame numbers.
            - vid: VideoClip object.
            - shot0: Shot number.
            - vid_type: Type of video (recap or episode).
            - path: Path to save the frames.
        
        Return:
            - Frame encodings dictionary from `DenseNet169/ResNet-50` (ImageNet).
            - Frame encodings dictionary from `CLIP` (Semantics).
            - Frame encodings dictionary from `MVIT` (Action).
        """
        logger.info(f"=================== Processing Frames ({vid_type}; Shot: {shot0}) ===================")
        # ==================== Extract ImageNEt features ====================
        tmp_dict, img_lst, img_fn = imagenet_extractor.get_DENSE_feats(tmp_lst[0], tmp_lst[-1], vid, shot=shot0)
        if save_frames:
            logger.info(f"=================== Saving Frames ({vid_type}; Shot: {shot0}) ===================\n")
            frame_saver(shot0, img_lst, img_fn, vid_type, frame_path)
        logger.info(f"ImageNET - Done!")
        # ==================== Extract CLIP features ====================
        clip_dict = clip_extractor.get_CLIP_feats(tmp_lst[0], tmp_lst[-1], vid, shot=shot0)
        logger.info(f"CLIP - Done!")
        # ==================== Extract MVIT features ====================
        # if shot length is >= 32, do it. Else None
        if tmp_lst[-1] - tmp_lst[0] > 30:
            action_feats = mvit_extractor.vid2ACTION((tmp_lst[0], tmp_lst[-1]), shot0, vid)
        else:
            action_feats = {"shot_{0:04n}_fn_{1:06n}".format(shot0, tmp_lst[0]): None}
        logger.info(f"MVIT - Done!")
        return tmp_dict, clip_dict, action_feats

    parent_pth = Path(config["parent_folder"])
    path = parent_pth/f'{parent_pth.stem}{eps_name}/'
    videvents = readVidEvents(path/f"videvents/{vid_type}.videvents", split_val=3)
    vid = VideoFileClip((path/f"{path.stem}.mp4").as_posix())
    k=0; tmp_lst = []; time_lst = []
    vid_dense_dict = dict(); vid_clip_dict = dict(); vid_action_dict = dict()
    while k < len(videvents)-1:
        shot0, fn, ft = videvents[k]
        shot1, _, _ = videvents[k+1]
        if shot0 == shot1:
            tmp_lst.append(fn)
            time_lst.append(ft)
        else:
            tmp_lst.append(fn)
            time_lst.append(ft)
            # we want to avoid rapid changes in the shot... hence 0.32 sec (=8 frames if 25fps)
            if time_lst[-1] - time_lst[0] > 0.32:
                imageNet_dict, clip_dict, action_dict = vidSavingModule(tmp_lst, vid, shot0, vid_type, path)
                vid_dense_dict.update(imageNet_dict)
                vid_clip_dict.update(clip_dict)
                vid_action_dict.update(action_dict)
            else:
                logger.info(f"Skipping {vid_type} Shot: {shot0} as it's of {time_lst[-1] - time_lst[0]} sec.")
            tmp_lst = []
            time_lst = []
        k+=1
    # ==================== Save the last shot ====================
    if time_lst and (time_lst[-1] - time_lst[0] > 0.32):
        imageNet_dict, clip_dict, action_dict = vidSavingModule(tmp_lst, vid, shot0, vid_type, path)
        vid_dense_dict.update(imageNet_dict)
        vid_clip_dict.update(clip_dict)
        vid_action_dict.update(action_dict)
    out = {'imagenet': vid_dense_dict, 'mvit': vid_action_dict, 'clip': vid_clip_dict}
    return videvents, out

def encodeEpisode(eps_name:str,
                  config: Dict,
                  imagenet_extractor: IMAGE,
                  mvit_extractor: ACTION,
                  clip_extractor: CLIP,
                  generate_encodings:bool = False,
                  recap_exists:bool = True,
                  save_frames: bool = False
                 )->None:
    """
    Encode the whole Episode that consists of both RECAP and EPISODE.
    ---------------------------------------------------------------------
    Args:
        - eps_name: Episode Name (in format: Exx, e.g., E03).
        - config: Dictionary that consists required paths.
        - imagenet_extractor: Imagenet architecture namely, `DenseNet169`
          that extracts video frames features.
        - mvit_extractor: Action embeddings from MVIT model.
        - clip_extractor: CLIP Image model to extract embeddings that semantically
          similar to its textual description.
        - encoder: To encode the frames (i.e., extracting feature maps).
        - generate_encodings: A boolean variable that confirms whether to
          generate frame encodings from scratch or to use it from cache.
        - recap_exists: A boolean variable that confirms whether the recap
          exists or not for that episode.
        - save_frames: Whether to save the frames for each `video_shot`

    Return:
    `None`
    """
    def shotConstructor(sh: int, vidEncodings: Dict, eps_name: str, frame_time: List[float])->shot:
        shot_obj = shot(shot_no=sh, episode_name=eps_name)
        tmp_lst = []
        for frame_name in vidEncodings:
            if int(frame_name[5:9]) == sh:
                shot_obj.frame_counts += 1
                tmp_lst.append(frame_name)
        # can have shot OBJ that isn't present in vidEncodings
        shot_obj.frame_names = tmp_lst
        shot_obj.time = (sec2datetime(frame_time[0]),
                        sec2datetime(frame_time[-1]))
        return shot_obj


    def constructShotObj(videvents: List, vidEncodings: Dict)->List:
        k = 0
        shot_lst, frame_time = [], []
        while k < len(videvents)-1:
            sh, _, ft = videvents[k]
            sh1, _, _ = videvents[k+1]
            if sh == sh1:
                frame_time.append(ft)
            else:
                frame_time.append(ft)
                if frame_time[-1] - frame_time[0] > 0.32:
                    shot_lst.append(shotConstructor(sh, vidEncodings, eps_name, frame_time))
                frame_time = []
            k += 1
        # ==================== Save the last shot ====================
        if frame_time and (frame_time[-1] - frame_time[0] > 0.32):
            shot_lst.append(shotConstructor(sh, vidEncodings, eps_name, frame_time))
        return shot_lst

    # ==================== Extract Encodings ====================
    parent_pth = Path(config["parent_folder"])
    path = parent_pth/f'{parent_pth.stem}{eps_name}/'
    if generate_encodings:
        if recap_exists:
            rec_vidEvents, rec_vidEncodings = extractEncodedRecapOrEpisode(eps_name=eps_name,
                                                                        imagenet_extractor=imagenet_extractor,
                                                                        mvit_extractor=mvit_extractor,
                                                                        clip_extractor=clip_extractor,
                                                                        vid_type="recap",
                                                                        save_frames=save_frames,
                                                                        config=config)

            logger.info(f"Frame encodings saved for RECAP part of {eps_name}.\n")
        eps_vidEvents, eps_vidEncodings = extractEncodedRecapOrEpisode(eps_name=eps_name,
                                                                       imagenet_extractor=imagenet_extractor,
                                                                       mvit_extractor=mvit_extractor,
                                                                       clip_extractor=clip_extractor,
                                                                       vid_type="episode",
                                                                       save_frames=save_frames,
                                                                       config=config)
        logger.info(f"Frame encodings saved for EPISODE part of {eps_name}.\n")
    else:
        if recap_exists:
            rec_vidEncodings = load_pickle(path/f"encodings/vid_encodings/recap_encodings.pkl")
            rec_vidEvents = readVidEvents(path/"videvents/recap.videvents", split_val=3)
        eps_vidEncodings = load_pickle(path/f"encodings/vid_encodings/episode_encodings.pkl")
        eps_vidEvents = readVidEvents(path/"videvents/episode.videvents", split_val=3)
    
    # ==================== Construct Shot Objects ====================
    if recap_exists:
        rec_shot_lst = constructShotObj(rec_vidEvents, rec_vidEncodings['imagenet'])
    eps_shot_lst = constructShotObj(eps_vidEvents, eps_vidEncodings['imagenet'])

    logger.info(f"============ Saving Frame Encodings and their collective Shot Objects for {eps_name} ============")
    vid_en_path = path/"encodings/vid_encodings/"
    os.makedirs(vid_en_path, exist_ok=True)
    if generate_encodings:
        if recap_exists:
            save_pickle((vid_en_path/"recap_encodings.pkl"), rec_vidEncodings)
        save_pickle((vid_en_path/"episode_encodings.pkl"), eps_vidEncodings)
    if recap_exists:
        save_pickle((vid_en_path/"recap_OBJ.pkl"), rec_shot_lst)
    save_pickle((vid_en_path/"episode_OBJ.pkl"), eps_shot_lst)
    return None

