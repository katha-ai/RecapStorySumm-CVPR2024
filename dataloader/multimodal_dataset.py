#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
`multimodal_dataset.py`: Load video and dialogue dataset and combine based
on arguments to create a multimodal dataset.
"""

import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from utils.logger import return_logger
from typing import Dict, List, Tuple, Optional, Type
from dataloader.dia_dataset import DialogueDataset
from dataloader.vid_dataset import VisualDataset3MC, VisualDataset3MNC
from utils.general_utils import load_pickle, get_file_name, time2secs, pad

logger = return_logger(__name__)


class MultiModalDataset(Dataset):
    """
    Given list of Episodes, this module will try to extract encodings
    for each `dialogue` as well as `shots`. Also it will try to extract
    `GT` for next episode for both dialogue and shots.
    
    And finally
    it will return a tuple of (input, mask, target) for
    each episode.
    ------------------------------------------------------
    Note: The episodes naming format would be EXX which
    should be in increasing order only and by default the Ground
    Truth labels would be imported according to the episode one
    ahead of the latest episode in the given list. 
    """

    def __init__(self: Type["MultiModalDataset"],
                 ep_names: List[Path],
                 vary_window_size: bool,
                 scene_boundary_threshold: float,
                 window_size: int,
                 bin_size: int,
                 withGROUP: bool,
                 normalize_group_labels: bool,
                 which_features: List[str],
                 modality: str,
                 vid_label_type: str,
                 dia_label_type: str,
                 which_dia_model: str,
                 get_word_level: bool,
                 max_cap: int,
                 sampling_type: str,
                 concatenation: bool,
                 **kwargs
                 ) -> None:
        """
        Args:
            - `ep_names`: List of names of EPISODES that are to be used for model training or inference.
            - `vary_window_size`: If `True`, it will vary the window for selecting shots.
            - `scene_boundary_threshold`: Threshold for obtaining the scene boundaries. 
               Should be used only if `vary_window_size` is `True`.
            - `window_size`: Size of the window to be considered for 
              for grouping the dialogue and shots. Should be used only if
              `vary_window_size` is `False`.
            - `bin_size`: Size of the bin to be considered for bucketing
              the dialogue and shots.
            - `withGROUP`: If `True`, it will add [GROUP] token placeholders to the input.
            - `normalize_group_labels`: If `True`, it will normalize the labels (the GROUP labels).
              Should be used only if `withGROUP=True`.
            - `which_features`: List of which extracted (from model) features to be used for training.
            - `modality`: Which modality to use for training. Available options are `dia`, `vid` and `both`.
            - `vid_label_type`: Which type of labels to be used for video shots. Available options are `GT` and `SL`.
            - `dia_label_type`: Which type of labels to be used for dialogue. Available options are `GT`, `SL` and `SLV`.
            - `which_dia_model`: Which model to use for dialogue feature extraction.
                Available options are `roberta-large`, `all-mpnet-base-v2`, `all-MiniLM-L6-v2`, `pegasus-large`. 
            - `get_word_level`: If `True`, it will return the word level features extracted
              from `pegasus-large` model. For this set `which_dia_model = pegasus-large`.
            - `max_cap`: Maximum number of frames to be considered for each shot.
            - `sampling_type`: Type of sampling to be used for each shot. Available options are
              `middle`, `random` and `uniform`.
            - `concatenation`: If `True`, it will concatenate the features extracted from different models.
        """
        super(MultiModalDataset, self).__init__(**kwargs)
        if modality != "vid":
            self.diaDataset = DialogueDataset(ep_names,
                                              get_word_level=get_word_level,
                                              model_name=which_dia_model,
                                              label_type=dia_label_type)
        if modality != "dia":
            if concatenation:
                self.vidDataset = VisualDataset3MC(ep_names, max_cap=max_cap,
                                                   which_features=which_features,
                                                   sampling_type=sampling_type,
                                                   label_type=vid_label_type)
            else:
                self.vidDataset = VisualDataset3MNC(ep_names, imagenet_max_cap=max_cap,
                                                    which_features=which_features,
                                                    sampling_type=sampling_type,
                                                    label_type=vid_label_type)
        # declare other attributes
        self.ep_names = ep_names
        self.vary_window_size = vary_window_size
        self.scene_boundary_threshold = scene_boundary_threshold
        self.window_size = window_size
        self.bin_size = bin_size
        self.withGROUP = withGROUP
        self.normalize_group_labels = normalize_group_labels
        self.modality = modality
    
    def get_time_stamps(self,
                        vid_pth: Path,
                        dia_pth: Path
                       ) -> Tuple[List[Tuple[float, float]],
                                  List[Tuple[float, float]]]:
        """
        Given a path to a pickle file, it will return a list of time stamps
        for each dialogue or video shots for that episode.
        """
        OBJ1 = load_pickle(vid_pth)
        vid_t = [tuple(map(time2secs, shot_obj.time)) for shot_obj in OBJ1]
        eps_file = get_file_name(dia_pth)
        start = int(eps_file.stem.split("_")[-1])
        OBJ2 = load_pickle(eps_file)[start+1:]
        dia_t = [tuple(map(time2secs, [dia_obj.start.to_time(), dia_obj.end.to_time()])) for dia_obj in OBJ2]
        return vid_t, dia_t
    
    def groupby_mid(self,
                     vid_time: List[Tuple[float, float]],
                     dia_time: List[Tuple[float, float]],
                     scene_length: Optional[np.ndarray]
                    ) -> Tuple[List[List[Tuple[float, float]]],
                               List[List[Tuple[float, float]]]]:
        """
        Given a list of time stamps for video shots and dialogue, it will
        group them as per window size mentioned and will return chunks of
        video shots and dialogue.
        ------------------------------------------------------
        Args:

            - vid_time: List of time stamps for video shots.
            - dia_time: List of time stamps for dialogue.
            - scene_length: Array of scene lengths for each episode.
        
        Returns:
            - Two list of grouped video as well as dialogue tuples.
        """
        if self.vary_window_size: m = 0
        k, j = 0, 0
        vid_t, dia_t = [], []
        while k < len(vid_time):
            if self.vary_window_size:
                self.window_size = scene_length[m]
                m += 1
            vtimes = vid_time[k: k+self.window_size]
            k += self.window_size
            end = vtimes[-1][1]
            l = 0
            dtimes = []
            for ele in dia_time[j:]:
                dia_mid_time = (ele[0] + ele[1]) / 2
                if dia_mid_time < end:
                    # removing cond: dia_mid_time >= start as dia-bound
                    # can come withn a shot-interval
                    dtimes.append(ele)
                    l += 1
                else:
                    break
            j += l
            vid_t.append(vtimes)
            dia_t.append(dtimes)
        # append the rest in dia if any
        if k >= len(vid_time) and j < len(dia_time):
            dia_t[-1].extend(dia_time[j:])
        return vid_t, dia_t

    def time2bin(self,
                 v_time: List[List[Tuple[float, float]]],
                 d_time: List[List[Tuple[float, float]]]
                ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Given whole big time interval of complete episode bucketize
        the video shots and dialogue into bins of size 1 sec according
        to their mid time.
        """
        start = min(v_time[0][0][0], d_time[0][0][0]) if len(d_time[0]) > 0 else v_time[0][0][0]
        end = max(v_time[-1][-1][1], d_time[-1][-1][1]) if len(d_time[-1]) > 0 else v_time[-1][-1][1]
        bins = np.arange(start, end, self.bin_size)
        bins = np.append(bins, end)
        vtime = [[(shot[0]+shot[1])/2 for shot in lst] for lst in v_time]
        dtime = [[(dia[0]+dia[1])/2 for dia in lst] for lst in d_time]
        v = [np.digitize(ele, bins) for ele in vtime]
        d = [np.digitize(ele, bins) for ele in dtime]
        return v, d
    
    def get_scene_length(self,
                         vid_pth: Path,
                         scene_boundary_path: Path
                        ) -> np.ndarray:
        """
        Scene length is the number of shots in a scene.
        ------------------------------------------------------
        Args:
            - ep_videvents_shots: List of shot boundaries in an episode.
            - scene_boundary_locations: List of scene boundaries (in terms
              of shot boundaries) in an episode.

        Returns:
            - List of scene lengths.
        """
        # load the episode's shot boundaries (without any irrelevant shots)
        OBJ1 = load_pickle(vid_pth)
        ep_videvents_shots = np.unique([shot_obj.shot_no for shot_obj in OBJ1])

        # Which shots account for the scene boundaries
        scene_boundary_locations = np.argwhere(np.load(scene_boundary_path)
                                               > self.scene_boundary_threshold).squeeze()

        # isin: Which scene_boundary_locations are in ep_videvents_shots
        scene_boundary_locations = scene_boundary_locations[np.isin(scene_boundary_locations,
                                                                    ep_videvents_shots,
                                                                    assume_unique=True)]
        # check if first scene boundary is the starting shot of episode
        if scene_boundary_locations[0] > ep_videvents_shots[0]:
            # first scene's length calculated including the first shot
            start_boundary = ep_videvents_shots[0] - 1
            scene_boundary_locations = np.insert(scene_boundary_locations, 0, start_boundary)
        else:
            # first scene's length calculated including the first shot
            scene_boundary_locations[0] -= 1

        # similarly for last scene
        if scene_boundary_locations[-1] < ep_videvents_shots[-1]:
            scene_boundary_locations = np.append(scene_boundary_locations, ep_videvents_shots[-1])

        # check if shot boundaries are inceremented by 1 then at abrupt
        # increment extra scene boundary locations will be inserted.
        # assert np.any(np.ediff1d(ep_videvents_shots) - 1)
        if not np.all(np.ediff1d(ep_videvents_shots) == 1):
            scene_length = []
            skip_locs = np.argwhere(np.ediff1d(ep_videvents_shots) != 1).reshape(-1,)
            for ele in skip_locs:
                a, b = ep_videvents_shots[ele:ele+2]
                scene_bounds_chunk = scene_boundary_locations[scene_boundary_locations <= a]
                if scene_bounds_chunk[-1] < a:
                    scene_bounds_chunk = np.append(scene_bounds_chunk, a)
                scene_length.append(np.ediff1d(scene_bounds_chunk))
                scene_boundary_locations = scene_boundary_locations[scene_boundary_locations >= b]
                if b < scene_boundary_locations[0] or len(scene_boundary_locations) == 1:
                    scene_boundary_locations = np.insert(scene_boundary_locations, 0, b)
                scene_boundary_locations[0] -= 1
            scene_length.append(np.ediff1d(scene_boundary_locations))
            scene_length = np.concatenate(scene_length)
        else:
            scene_length = np.ediff1d(scene_boundary_locations)
        assert np.sum(scene_length) == len(ep_videvents_shots), \
            "Total scene length not equal to number of shots in Episode"
        return scene_length

    def __len__(self)->int:
        return len(self.ep_names)

    def __getitem__(self,
                    idx: int
                   ) -> Tuple[Optional[Dict], Optional[Dict],
                              np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray, np.ndarray]:
        vid_time_path = self.ep_names[idx] / "encodings/vid_encodings/episode_OBJ.pkl"
        dia_time_path = self.ep_names[idx] / fr"encodings/dia_encodings/newSrtObj_*.pkl"

        if self.vary_window_size:
            scene_boundary_path = self.ep_names[idx] / "encodings/vid_encodings/scene_boundary.npy"
            # get the array for scene length of the episode
            scene_length = self.get_scene_length(vid_time_path, scene_boundary_path)
        else:
            scene_length = None

        vid_time_stamps, dia_time_stamps = self.get_time_stamps(vid_time_path, dia_time_path)
        vid_grp, dia_grp = self.groupby_mid(vid_time_stamps, dia_time_stamps, scene_length)
        vbin, dbin = self.time2bin(vid_grp, dia_grp)
        bin_indices = np.array([]); subgroup_len = np.zeros((0, 2))
        token_type = np.array([]); group_idx = np.array([])
        if self.withGROUP:
            if self.modality != "dia":
                vid_labels = self.vidDataset[idx]["labels"]
            if self.modality != "vid":
                dia_labels = self.diaDataset[idx]["labels"]
            labels = []; v_cnt, d_cnt = 0, 0
        for k, (v, d) in enumerate(zip(vbin, dbin)):
            bin_indices = np.concatenate([bin_indices, v, d, [0]])
            subgroup_len = np.concatenate([subgroup_len, np.array([[len(v), len(d)]])], axis=0)
            token_type = np.concatenate([token_type, np.zeros(len(v)), np.ones(len(d)), [2]])
            group_idx = np.concatenate([group_idx, np.ones(len(v)+len(d)+1)*(k+1)])
            if self.withGROUP:
                # TODO: shan't we do min-max normalization after taking mean?
                # FIXME: check if this is correct
                if self.modality == "both":
                    labels.append(np.concatenate([vid_labels[v_cnt:v_cnt+len(v)],
                                                dia_labels[d_cnt:d_cnt+len(d)]]).mean())
                elif self.modality == "vid":
                    labels.append(vid_labels[v_cnt:v_cnt+len(v)].mean())
                elif self.modality == "dia":
                    labels.append(dia_labels[d_cnt:d_cnt+len(d)].mean())
                v_cnt += len(v)
                d_cnt += len(d)

        # construct mask
        mask = np.ones_like(token_type)
        if not self.withGROUP:
            mask[token_type == 2] = 0
        if self.modality == "vid":
            mask[token_type == 1] = 0
        elif self.modality == "dia":
            mask[token_type == 0] = 0

        tup = (bin_indices, token_type, mask, group_idx, subgroup_len)

        if self.withGROUP:
            labels = np.nan_to_num(np.array(labels))
            if self.normalize_group_labels:
                labels -= labels.min()
                labels /= labels.max()
            tup = tup + (labels,)
        # else:
        #     tup = (bin_indices, token_type, mask, group_idx, subgroup_len, np.zeros(len(subgroup_len)))

        if self.modality == "both":
            return self.vidDataset[idx], self.diaDataset[idx], *tup
        elif self.modality == "vid":
            return self.vidDataset[idx], None, *tup
        elif self.modality == "dia":
            return None, self.diaDataset[idx], *tup
        else:
            raise ValueError(f"Invalid modality, modality obtained {self.modality}")

    def collate_fn(self,
                   batch: List[Tuple]
                  ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor],\
                       np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Given a batch of data, it will collate them and return a batch of data.
        """
        # bin_indices, token_type, mask, group_idx, subgroup_len, labels
        if self.withGROUP:
            vid_batch, dia_batch, bin_indices, token_type, \
                mask, group_idx, subgroup_len, labels = zip(*batch)
        else:
            vid_batch, dia_batch, bin_indices, token_type, \
                mask, group_idx, subgroup_len = zip(*batch)

        if self.modality != "vid":
            dia_batch = self.diaDataset.collate(dia_batch)
        if self.modality != "dia":
            vid_batch = self.vidDataset.collate(vid_batch)
        
        # pad the bin_indices, token_type, mask and subgroup_len
        bin_indices = pad(bin_indices)
        token_type = pad(token_type, pad_value=3)
        mask = pad(mask)
        group_idx = pad(group_idx)
        subgroup_len = pad(subgroup_len, padding_dim=0, pad_value=-1)

        # convert to tensor
        bin_indices = torch.from_numpy(bin_indices).to(dtype=torch.int32)
        token_type = torch.from_numpy(token_type).to(dtype=torch.int32)
        mask = torch.from_numpy(mask).to(dtype=torch.int32)
        group_idx = torch.from_numpy(group_idx).to(dtype=torch.int32)
        subgroup_len = torch.from_numpy(subgroup_len).to(dtype=torch.int32)
        
        tup = (bin_indices, token_type, mask, group_idx, subgroup_len)
        
        if self.withGROUP:
            labels = pad(labels, padding_dim=0, pad_value=-1)
            labels = torch.from_numpy(labels).to(dtype=torch.float32)
            tup = tup + (labels,)

        if self.modality == "both":
            return (vid_batch, dia_batch), *tup
        elif self.modality == "vid":
            return vid_batch, *tup
        elif self.modality == "dia":
            return dia_batch, *tup
        
if __name__ == '__main__':
    import ipdb
    withGROUP = False
    dataset = MultiModalDataset(ep_names=[Path("data/24/S03/S03E03"),
                                          Path("data/24/S04/S04E18"),
                                          Path("data/24/S02/S02E09")],
                                vary_window_size=False,
                                scene_boundary_threshold=0.7,
                                window_size=20,
                                bin_size=1,
                                withGROUP=withGROUP,
                                normalize_group_labels=True,
                                which_features=['imagenet', 'mvit', 'clip'],
                                modality='both',
                                which_dia_model='pegasus-large',
                                get_word_level=True,
                                max_cap=25,
                                sampling_type='random',
                                vid_label_type='SL',
                                dia_label_type='SLV',
                                concatenation=True,
                                condition_on_current=False)
    t1 = dataset[0]
    t2 = dataset[1]
    batch = [t1, t2]
    if withGROUP:
        (vid_batch, dia_batch), bin_indices, token_type, mask, \
            group_idx, subgroup_len, labels = dataset.collate_fn(batch)
    else:
        (vid_batch, dia_batch), bin_indices, token_type, mask, \
            group_idx, subgroup_len = dataset.collate_fn(batch)
    # import ipdb; ipdb.set_trace()

# https://numpy.org/doc/stable/user/basics.indexing.html
