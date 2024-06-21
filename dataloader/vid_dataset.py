#!/usr/bin/env python

"""
vid_dataset.py: Define video dataset for training and inference.
"""

import torch
import random
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from utils.logger import return_logger
from utils.general_utils import load_pickle
from typing import Dict, List, Tuple, Optional, Union, Type

logger = return_logger(__name__)
   
def sampler(sampling_type: str,
            shot_arr: np.ndarray,
            max_cap: int,
            indices: Optional[np.ndarray] = None
           ) -> np.ndarray:
    """
    Given a sampling type, this function will return
    the indices of the samples that are to be used for
    training.
    """
    if sampling_type == 'uniform':
        idx = np.round(np.linspace(0, len(shot_arr)-1, max_cap)).astype(np.int32)
        if indices is not None:
            indices = indices[idx]
        mask = np.zeros(len(shot_arr))
        mask[idx] = 1
        mask = mask.astype('bool')
        # mask = np.array(sampling(len(shot_arr), max_cap)).astype('bool')
        capped_shot_arr = shot_arr[mask]
    elif sampling_type == 'middle':
        diff = len(shot_arr) - max_cap
        if diff > 1:
            capped_shot_arr = shot_arr[(diff//2):(diff//2)+max_cap]
            if indices is not None:
                indices = indices[(diff//2):(diff//2)+max_cap]
        else:
            capped_shot_arr = shot_arr[:max_cap]
            if indices is not None:
                indices = indices[:max_cap]
    elif sampling_type == 'random':
        idx = np.asarray(sorted(random.sample(range(len(shot_arr)), max_cap)))
        if indices is not None:
            indices = indices[idx]
        mask = np.zeros(len(shot_arr))
        mask[idx] = 1
        mask = mask.astype('bool')
        capped_shot_arr = shot_arr[mask]
    else:
        raise ValueError(f"Expected sampling_type to be 'uniform', 'middle', or 'random'. Got {sampling_type}")
    if indices is not None:
        return capped_shot_arr, indices
    return capped_shot_arr, idx if sampling_type in ['uniform', 'random'] else np.arange(len(capped_shot_arr))

    

class VisualDataset3MC(Dataset):
    """
    Given list of Episodes, this module will try to 
    extract encodings for each shot and will merge them
    in form of a list of shot-encodings as well as their respective
    labels. CONCATENATION of encodings is done in OR fashion for all
    3 types of embeddings (DenseNet169, MViT, CLIP). Hence, total 7 
    possible data shapes are expected according to user input for `which_features`.
    Hence the name: `3M` = 3 Models, `C` = Concatenation.
    ------------------------------------------------------
    Note: The episodes naming format would be EXX which
    should be in increasing order only and by default the Ground
    Truth labels would be imported according to the episode one
    ahead of the latest episode in the given list. 
    """

    def __init__(self: Type['VisualDataset3MC'],
                 ep_names: List[Path],
                 max_cap: int = 35,
                 which_features: List[str] = ["imagenet", "mvit", "clip"],
                 sampling_type: str = 'middle',
                 label_type: str = "SL",
                 ) -> None:
        """
        Args:
            - ep_names: List of names of EPISODES that are to be used
              for model training or inference. e.g., EXX
            - max_cap: The maximum number of frames we want per shot.
              It depends which feature type we're using.
            - which_features: List of pretrained extracted features to be used for
                training. e.g., `["imagenet", "mvit", "clip"]`
            - sampling_type: Whether to uniformly sample frames from 
              a given video shot (`uniform`) or just Randomly sample
              (`random`) or just take the middle deck (`middle`) of frames.
              `default=middle`.
            - label_type: Type of labels to be used. It can be either
                `GT` or `SLV` or `H` or `F`.
        """
        super(VisualDataset3MC, self).__init__()
        feature_dim = {"imagenet": 1664, "mvit": 768, "clip": 512}
        self.feature_dim = sum([feature_dim[feat] for feat in which_features])
        self.which_features = which_features
        self.ep_names = ep_names
        self.max_cap = max_cap
        self.sampling_type = sampling_type
        self.label_type = label_type

    def fillANDconcat(self,
                      model_arr: Dict[str, np.ndarray],
                      indices_arr: Dict[str, List[int]]
                     ) -> np.ndarray:
        """
        If required, fill `MVIT` encodings to match the
        length of `IMAGENET` encodings and then concatenate
        them in OR fashion.
        """
        model_type = "imagenet" if "imagenet" in self.which_features else "clip"
        if not len(model_arr[model_type]):
            return np.array([])
        mvit_idx = np.array([(i+16) for i in indices_arr["mvit"]])
        modified_mvit_idx = np.abs(np.tile(np.array(indices_arr[model_type]), (len(mvit_idx), 1)) -
                                   mvit_idx.reshape(-1, 1)).argmin(axis=0)
        model_arr["mvit"] = model_arr["mvit"][modified_mvit_idx]
        indices_arr["mvit"] = (mvit_idx[modified_mvit_idx]).tolist()
        assert model_arr["mvit"].shape[0] == model_arr[model_type].shape[0], \
            f"MVIT and Imagenet arrays are of different length. Got MVIT Shape = \
                {model_arr['mvit'].shape}; Imagenet Shape = {model_arr[model_type].shape}"
        return np.hstack(list(model_arr.values()))

    def fname2fidx(self, fname_lst: List[str]) -> List[int]:
        """
        Given a imagename, will return the shot index.
        """
        return [int(fname[13:19]) for fname in fname_lst]

    def shotWiseEncoding(self,
                         ep_path: Path,
                         vid_part_type: str = "episode"
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compile frame level encoding into one array for a given shot.
        Plus, generate masks for these array indicating which are real frames
        and which are padding.
        --------------------------------------------------------------
        Args:
            - ep_path: Path to episode.
            - vid_part_type: Whether `episodic` part or `recap` part. `default="episode"`
        """
        eps_OBJ = load_pickle(ep_path/f"encodings/vid_encodings/{vid_part_type}_OBJ.pkl")
        eps_enc = load_pickle(ep_path/f"encodings/vid_encodings/{vid_part_type}_encodings.pkl")
        final_enc, final_mask, final_idx = [], [], []
        for eps_shot_obj in eps_OBJ:
            model_arr, arr_indices = {}, {}
            for feat in self.which_features:
                if feat == "mvit":
                    mvit_enc_arr = [frame_enc for fname, frame_enc in eps_enc['mvit'].items()
                                       if int(fname[5:9]) == eps_shot_obj.shot_no and frame_enc is not None]

                    # when shot does not have any mvit encoding, we'll just give zeros.
                    model_arr[feat] = np.vstack(mvit_enc_arr) if len(mvit_enc_arr) else np.zeros((2, 768))

                    # extract the shot-wise mvit frame indices.
                    arr_indices[feat] = self.fname2fidx([fname for fname, frame_enc in eps_enc['mvit'].items()
                                        if int(fname[5:9]) == eps_shot_obj.shot_no and frame_enc is not None])\
                                            if len(mvit_enc_arr) else [0, 0]
                else:
                    arr = [eps_enc[feat][fname[:19]] if feat == "clip" else eps_enc[feat][fname]\
                            for fname in eps_shot_obj.frame_names]
                    model_arr[feat] = np.vstack(arr) if len(arr) else np.array([])

                    # extract the shot-wise clip and imagenet frame indices.
                    arr_indices[feat] = self.fname2fidx(eps_shot_obj.frame_names) if len(arr) else []
            # duplicate MVIT encodings, so as to match the length of imagenet/clip encodings.
            if "mvit" in self.which_features and len(self.which_features) > 1:
                final_arr = self.fillANDconcat(model_arr, arr_indices)
            elif "mvit" in self.which_features and len(self.which_features) == 1:
                final_arr = model_arr["mvit"]
            else:
                final_arr = np.hstack(list(model_arr.values()))

            # For sure there would be atleast one frame in a shot.
            # if Not then we'll just give zeros.
            shot_len = final_arr.shape[0]
            if shot_len:
                if shot_len < self.max_cap:
                    zero_arr = np.zeros((self.max_cap - shot_len, final_arr.shape[1]))
                    capped_shot_arr = np.vstack([final_arr, zero_arr])
                    mask_arr = np.hstack([np.ones(shot_len), np.zeros(zero_arr.shape[0])])
                    idx = np.arange(capped_shot_arr.shape[0])
                else:
                    capped_shot_arr, idx = sampler(self.sampling_type, final_arr, self.max_cap)
                    mask_arr = np.ones(self.max_cap)
            else:
                # Handling the exception, where no frame is present
                # (like all dark frames).
                capped_shot_arr = np.zeros((self.max_cap, self.feature_dim))
                mask_arr = np.zeros(self.max_cap)
                idx = np.arange(capped_shot_arr.shape[0])
            final_enc.append(capped_shot_arr)
            final_mask.append(mask_arr)
            final_idx.append(idx)
        return np.stack(final_enc), np.stack(final_mask), np.stack(final_idx)

    def collate(self, data_batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        collate function to be used with DataLoader. It essentially pads the
        data to the maximum length in the batch.
        """
        # pad according the longest episode in the given batch.
        longest_ep_len = max([ep['vid_enc'].shape[0] for ep in data_batch])
        for episode in data_batch:

            paddingLen = longest_ep_len - episode['labels'].shape[0]
            episode['vid_enc'] = np.pad(episode['vid_enc'], ((0, paddingLen), (0, 0), (0, 0)))
            episode['vid_mask'] = np.pad(episode['vid_mask'], ((0, paddingLen), (0, 0)))
            episode['vid_idx'] = np.pad(episode['vid_idx'], ((0, paddingLen), (0, 0)))
            episode['labels'] = np.pad(episode['labels'], ((0, paddingLen)))

        batch_dict = {k: torch.from_numpy(np.stack([ep[k] for ep in data_batch])) for k in data_batch[0]}
        return batch_dict

    def __len__(self)-> int:
        return len(self.ep_names)

    def __getitem__(self, idx:int)->Dict[str, np.ndarray]:
        eps_ENC, eps_MASK, eps_IDX = self.shotWiseEncoding(self.ep_names[idx])
        next_ep = "E{0:02n}".format(int(self.ep_names[idx].stem[-2:]) + 1)
        parent_dir = self.ep_names[idx].parent
        gt_file_name = self.label_type.upper() + ".pkl"
        labels = load_pickle(parent_dir/f"{parent_dir.stem}{next_ep}/scores/"/\
                            f"vid_scores/recapVepisode/{gt_file_name}")[self.ep_names[idx].stem[-3:]]
        assert eps_ENC.shape[:2] == eps_MASK.shape, \
            (f"Mask and Encoding mismatch in shape. Enc shape = {eps_ENC.shape};"
             f" Mask Shape = {eps_MASK.shape}")
        assert eps_ENC.shape[0] == labels.shape[0], \
            (f"Labels and Encoding are of different length. Enc Shape = {eps_ENC.shape};"
             f" Labels shape = {labels.shape}")
        assert eps_ENC.shape[0] == eps_IDX.shape[0], \
            (f"IDX and Encoding are of different length. Enc Shape = {eps_ENC.shape};"
                f" IDX shape = {eps_IDX.shape}")
        return {"vid_enc": eps_ENC, "vid_mask": eps_MASK, "vid_idx": eps_IDX, "labels": labels}

class VisualDataset3MNC(Dataset):
    """
    Given list of Episodes, this module will try to
    extract encodings for each shot and will merge them
    in List of shot-encodings as well as their respective
    labels. `NO CONCATENATION OF ENCODINGS IS DONE HERE`.
    Hence NC is appended to the class name. 
    `3M` and `3MNC`. `3M` means that we will use 3 models of visual.
    `3MNC` means that we will use 3 models of visual but without
    concatenation of the encodings from all models.
    ------------------------------------------------------
    Note: The episodes naming format would be EXX which
    should be in increasing order only and by default the Ground
    Truth labels would be imported according to the episode one
    ahead of the latest episode in the given list. 
    """

    def __init__(self: Type['VisualDataset3MNC'],
                 ep_names: List[Path],
                 which_features: List[str]=['imagenet', 'mvit', 'clip'],
                 imagenet_max_cap: int = 35,
                 mvit_max_cap: int = 10,
                 sampling_type: str = 'middle',
                 label_type: str = "SL",
                ) -> None:
        """
        Args:
            - ep_names: List of Episodes' path that are to be used for model training or inference.
            - which_features: List of features to be used for training. `default=['imagenet', 'mvit', 'clip']`
            - <backbone>_max_cap: The maximum number of frames we want per shot for corresponding backbone.
            - encoded_feature_dim: Embedding dimension of each visual frame.
            - sampling_type: Whether to uniformly sample frames from 
              a given video shot (`uniform`) or just Randomly sample
              (`random`) or just take the middle deck (`middle`) of frames.
              `default=middle`
            - label_type: Type of labels to be used. It can be either
                `GT` or `SLV` or `H` or `F`.
        """
        super(VisualDataset3MNC, self).__init__()
        self.ep_names = ep_names
        self.sampling_type = sampling_type
        self.label_type = label_type
        max_cap = {'imagenet': imagenet_max_cap, 'mvit': mvit_max_cap, 'clip': imagenet_max_cap}
        feature_dim = {'imagenet': 1664, 'mvit': 768, 'clip': 512}
        self.max_cap = {k: max_cap[k] for k in which_features}
        self.feature_dim = {k: feature_dim[k] for k in which_features}
        self.which_features = which_features

    def fname2fidx(self, fname_lst: List[str]) -> List[int]:
        r"""
        Given a imagename, will return the shot index.
        """
        return sorted([int(fname[13:19]) for fname in fname_lst])
    
    def reindex_mvit_feats(self,
                           imagenet_idx: Union[List[int], np.ndarray],
                           mvit_idx: Union[List[int], np.ndarray]
                          ) -> np.ndarray:
        r"""
        Given the imagenet and mvit indices, will return the
        mvit indices according to nearness to imagenet indices.
        """
        if isinstance(imagenet_idx, list):
            imagenet_idx = np.array(imagenet_idx)
        if isinstance(mvit_idx, list):
            mvit_idx = np.array(mvit_idx)
        if len(imagenet_idx):
            which_arr = np.abs(np.tile(imagenet_idx, (len(mvit_idx), 1)).T - mvit_idx.reshape(1, -1)).argmin(axis=0)
            return imagenet_idx[which_arr] - imagenet_idx.min()
        else:
            return mvit_idx - mvit_idx.min()
            

    def selectMaxCapFrames(self,
                           max_cap: int,
                           arr: np.ndarray,
                           indices: np.ndarray,
                           feature_dim: int
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        If no. of frames are less than `max_cap`, then pad the remaining frames with zeros. If no. of frames are more
        than `max_cap`, then sample `max_cap` frames from the given array with given sampling type.
        """
        tot_frames = arr.shape[0]
        if tot_frames:
            if tot_frames < max_cap:
                zero_arr = np.zeros((max_cap - tot_frames, arr.shape[1]))
                capped_shot_arr = np.vstack([arr, zero_arr])
                mask_arr = np.hstack([np.ones(tot_frames), np.zeros(zero_arr.shape[0])])
                indices = np.hstack((indices, np.tile(indices[-1], max_cap - tot_frames)))
            else:
                capped_shot_arr, indices = sampler(self.sampling_type, arr, max_cap, indices)
                mask_arr = np.ones(max_cap)
        else:
            # handle if arr is empty
            capped_shot_arr = np.zeros((max_cap, feature_dim))
            mask_arr = np.zeros(max_cap)
            indices = np.zeros(max_cap)
        return capped_shot_arr, mask_arr, indices

    def shotWiseEncoding(self,
                         ep_path: str,
                         apply_prefix: str = '',
                         vid_part_type: str = "episode"
                        ) -> Dict[str, np.ndarray]:
        """
        Compile frame level encoding into one array for a given shot.
        Plus, generate masks for these array indicating which are real frames
        and which are padding.
        --------------------------------------------------------------
        Args:
            - ep_path: The episode on which we'll do this operation.
            - vid_part_type: Whether `episodic` part or `recap` part. `default="episode"`
        """
        eps_OBJ = load_pickle(ep_path/f"encodings/vid_encodings/{vid_part_type}_OBJ.pkl")
        eps_enc = load_pickle(ep_path/f"encodings/vid_encodings/{vid_part_type}_encodings.pkl")
        final_dict = {}
        for k in self.which_features:
            final_dict.update({f'{apply_prefix}{k}_enc': [],
                              f'{apply_prefix}{k}_mask': [],
                              f'{apply_prefix}{k}_idx': []})

        for eps_shot_obj in eps_OBJ:
            # extract the frame encodings for the given shot.
            # Hence every shot have atleast one frame.
            for key in self.which_features:
                fnames = self.fname2fidx(eps_shot_obj.frame_names)
                if key == 'mvit':
                    mvit_enc_arr = [frame_enc for fname, frame_enc in eps_enc['mvit'].items()
                                    if int(fname[5:9]) == eps_shot_obj.shot_no and frame_enc is not None]
                    shot_enc_arr = np.vstack(mvit_enc_arr) if mvit_enc_arr else np.array([])
                    # modfiy mvit frame-indices according to imagenet/clip frame-indices.
                    mvit_fnames = [fname for fname, frame_enc in eps_enc['mvit'].items()
                                    if int(fname[5:9]) == eps_shot_obj.shot_no and frame_enc is not None]
                    shot_indices = self.reindex_mvit_feats(fnames, self.fname2fidx(mvit_fnames))\
                                        if mvit_enc_arr else np.array([])
                else:
                    # Note: all frame names in eps_shot_obj have passed black/white test. Hence no condition.
                    arr = [eps_enc[key][fname[:19]] if key == 'clip' else eps_enc[key][fname]\
                                                        for fname in eps_shot_obj.frame_names]
                    shot_enc_arr = np.vstack(arr) if arr else np.array([])
                    # extract the shot-wise clip or imagenet frame indices.
                    shot_indices = np.array(fnames)-fnames[0]  if arr else np.array([])

                # sample frames with given sampling type.
                shot_arr, shot_mask, shot_idx = self.selectMaxCapFrames(self.max_cap[key],
                                                                        shot_enc_arr,
                                                                        shot_indices,
                                                                        self.feature_dim[key])
                final_dict[apply_prefix+key+'_enc'].append(shot_arr)
                final_dict[apply_prefix+key+'_mask'].append(shot_mask)
                final_dict[apply_prefix+key+'_idx'].append(shot_idx)
        final_dict = {k: np.stack(final_dict[k]) for k in final_dict}
        return final_dict

    def collate(self, data_batch: List[dict]) -> Dict[str, torch.Tensor]:
        """
        collate function to be used with DataLoader. It essentially pads the
        data to the maximum length in the batch.
        """
        # get a key
        key = self.which_features[0]
        # define the longest episode length in the batch.
        longest_ep_len = max([ep[key+'_enc'].shape[0] for ep in data_batch])
        for episode in data_batch:
            padLen = longest_ep_len - episode[key+'_mask'].shape[-2]
            for k in self.which_features:
                episode[k+'_enc'] = np.pad(episode[k+'_enc'], ((0, padLen), (0, 0), (0, 0)))
                episode[k+'_mask'] = np.pad(episode[k+'_mask'], ((0, padLen), (0, 0)))
                episode[k+'_idx'] = np.pad(episode[k+'_idx'], ((0, padLen), (0, 0)))

            episode['labels'] = np.pad(episode['labels'], (0, padLen))
        
        batch_dict = {k: torch.from_numpy(np.stack([ep[k] for ep in data_batch])) for k in data_batch[0]}
        return batch_dict
        
    def __len__(self)->int:
        return len(self.ep_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_dict = self.shotWiseEncoding(self.ep_names[idx])
        next_ep = "E{0:02n}".format(int(self.ep_names[idx].stem[-2:]) + 1)
        parent_dir = self.ep_names[idx].parent
        gt_file_name = self.label_type.upper() + ".pkl"
        labels = load_pickle(parent_dir/f"{parent_dir.stem}{next_ep}/scores/"/\
                            f"vid_scores/recapVepisode/{gt_file_name}")[self.ep_names[idx].stem[-3:]]
        return {**ep_dict, 'labels': labels}

if __name__ == '__main__':
    # Define the Non-Concatenated Dataset.
    vid_3mc = VisualDataset3MC(ep_names=[Path("../../data/24/S02/S02E09"), Path("../../data/24/S03/S03E03")],
                               max_cap=30,
                               which_features=['clip', 'mvit'],
                               sampling_type='random',
                               label_type='SLV')
    t1 = vid_3mc[0]
    t2 = vid_3mc[1]
    batch = vid_3mc.collate([t1, t2])
    vid_3mnc = VisualDataset3MNC(ep_names=[Path("../../data/24/S02/S02E09"), Path("../../data/24/S03/S03E03")],
                                 which_features=['clip', 'mvit'],
                                 imagenet_max_cap=30,
                                 mvit_max_cap=10,
                                 sampling_type='random',
                                 label_type='SLV')
    t3 = vid_3mnc[0]
    t4 = vid_3mnc[1]
    batch1 = vid_3mnc.collate([t3, t4])
    # import ipdb; ipdb.set_trace(context=10)
