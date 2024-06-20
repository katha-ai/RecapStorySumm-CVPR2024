#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
make_html.py: consists of helper functions
that parse pickle files and make HTML files.
"""

import os
import jinja2
import pickle
import argparse
import numpy as np

from pathlib import Path
from utils.logger import return_logger
from typing import DefaultDict, List, Dict
from utils.general_utils import get_config

logger = return_logger(__name__)

# construct data for dialog HTML visualization... 
def dia2HTML(dialogue_lst: List)->List:
    """
    Constructs data to be used by Jinja2 template for visualizing dialogue matching.
    -----------------------------------------------------------------------
    Args:
        - dialogue_lst: `List`, list of dialogue objects.

    Returns:
        - data: `List`, list of lists, where each list is a row in the table.
    """
    def unpackDiaObj(dia_obj):
        """
        Unpacking object attributes into list.
        """
        time_lst = dia_obj.to_time()
        lst = [f"{dia_obj.dialogue_number}",
               dia_obj.episode_name,
               f"{time_lst[0]} to {time_lst[1]}",
               str([shot for shot in dia_obj.vid_shots])]
        try:
            return [dia_obj.sim_score] + lst
        except AttributeError:
            return [dia_obj.dialog_txt] + lst

    data = []
    for rec_obj in dialogue_lst:
        rec_lst = unpackDiaObj(rec_obj)
        rec_lst.append([ele_obj.dialog_txt for ele_obj in rec_obj.matching_dialogues])
        rec_lst.append([unpackDiaObj(ele_obj) for
                    ele_obj in rec_obj.matching_dialogues])
        data.append(rec_lst)
    return data

def vis2HTML(recap_dict: DefaultDict[str, dict],
             path2: Path,
             default_img: str,
             season:str,
             vid_part_type: str = "episode",
            )->List:
    """
    Constructs data to be used by Jinja2 template for visualizing visual-shot matching.
    -----------------------------------------------------------------------
    Args:
        - recap_dict: `DefaultDict[str, dict]`, dictionary of recap objects.
        - path2: `PurePath`, path to directory containing images.
        - default_img: `str`, path to default image.
        - season: `str`, season name.
        - vid_part_type: `str`, type of video part, either `episode` or `recap`.

    Returns:
        - data: `List`, list of lists, where each list is a row in the table.
    """
    data = []
    for frame, val in recap_dict.items():
        tmp_lst = [(path2/(frame+".jpg")).as_posix(), int(frame[5:9]), int(frame[13:19])] # ".png"
        if val == "NA":
            tmp_lst.append([default_img])
            tmp_lst.append([["NA", "NA", "NA"]])
        else:
            path_lst, other_info = [], []
            for ep_name in val:
                path1 = Path("../../../../")/(season+ep_name)
                for matched_frame in val[ep_name]:
                    if matched_frame[0] == "NA":
                        continue
                    else:
                        shot_info = int(matched_frame[0][5:9])
                        path_lst.append((path1/f"{vid_part_type}_html_images/"/
                                            (matched_frame[0]+".jpg")).as_posix())
                        other_info.append([ep_name, shot_info, f"{matched_frame[1]:.3f}"])
            tmp_lst.append(path_lst)
            tmp_lst.append(other_info)
        data.append(tmp_lst)
    return data

def scene2HTML(config: Dict,
               threshold: float = 0.5,
               data_path: Path = Path("../data/24/bassl/")
              ) -> None:
    """
    Based on scene bounaries obtained via manual thresholding, construct
    HTML file for visualization.
    (Note that this illustration is not used in the paper and just for experimentation.
    Further we did this for Series `24` only.)
    -----------------------------------------------------------------------
    Args:
        - threshold: `float`, threshold for scene boundary.
    """

    # Load data
    with open(data_path/"pred_metadata.pkl", "rb") as f:
        pred_meta = pickle.load(f)

    pred = np.load(data_path/"pred.npy")

    # Bring up all seasons and their episodes
    seasons = [f"S%02d" % i for i in range(2, 9)]
    episodes = [f"E%02d" % i for i in range(1, 25)]
    
    # Make directory for storing HTML files
    os.makedirs(data_path/f"scene_html_files_{threshold:.3f}", exist_ok=True)
    cnt=0

    # Making data for Jinja2 template
    for i in seasons:
        for j in episodes:
            ep_name = i + j
            len_ep = len([k for k in pred_meta['vid'] if k == ep_name])
            per_ep_pred = pred[cnt:cnt+len_ep]>threshold
            cnt+=len_ep
            global_path_and_row_lst = []
            path_row, shot_row = [], []
            for m, n in enumerate(per_ep_pred):
                if n:
                    path_row.append([f"../240P_frames/{ep_name}/shot_{m:04d}_img_{l}.jpg"
                                     for l in range(3)])
                    shot_row.extend([m])
                    global_path_and_row_lst.append([path_row, shot_row])
                    path_row, shot_row = [], []
                else:
                    path_row.append([f"../240P_frames/{ep_name}/shot_{m:04d}_img_{l}.jpg"
                                     for l in range(3)])
                    shot_row.extend([m])
            template = jinja2.Template(open(config["scene_html_template"]).read())
            logger.info("Generating HTML file...")
            with open(data_path/f"scene_html_files_{threshold:.3f}/{ep_name}.html", 'w') as fid:
                fid.write(template.render(data=global_path_and_row_lst))
            logger.info(f"HTML file Generated Successfully for {ep_name} :)\n\n")

# ************************************* MAIN *************************************
if __name__ == "__main__":
    
    config = get_config("configs/feats_config.yaml")
    if config['generate_scene_html']:
        # for scene2HTML
        scene2HTML(config, threshold=0.9, data_path=Path("../data/24/bassl/"))
    else:
        # how to run (python make_html.py (for False) | python make_html.py --w3frames (for True))
        # for vid2HTML and dia2HTML
        logger = return_logger(__name__)
        parser = argparse.ArgumentParser()
        parser.add_argument("-w3F", "--w3frames", action="store_true", default=False,
                            help="Whether to use 3 frames or frames extracted from video for shot labeling")
        args = parser.parse_args()
        fV = False if args.w3frames else True

        default_img = config["default_img_path"]
        template = jinja2.Template(open(config["html_template"]).read())
        parent_path = config["parent_folder"]
        recap_episode = config["recap_episode"]  # "S03E04"
        if not fV:
            path2 = os.path.join("../../", "shot_frames/")
        else:
            path2 = os.path.join("../../", "recap_fromVid_DenseNet169/")
        # Write to HTML file...
        logger.info("Generating HTML file...")
        with open(os.path.join(parent_path, recap_episode, "scores/vid_scores/revised_frame_matching.pkl"), "rb") as f:
            revised_recap_dict = pickle.load(f)
        revised_data = vis2HTML(recap_dict=revised_recap_dict, path2=path2, fV=fV)
        with open(os.path.join(parent_path, recap_episode, "scores/vid_scores/revised_frame_matching.html"), 'w') as fid:
            fid.write(template.render(data=revised_data))
        logger.info("HTML file Generated Successfully :)")
